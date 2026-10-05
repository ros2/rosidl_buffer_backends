// Copyright 2026 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rclcpp/serialized_message.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include "intel_buffer/intel_buffer_api.hpp"
#include "intel_memory_core/intel_memory_ipc_manager.hpp"
#include "intel_memory_core/gpu_cl_dmabuf_import.hpp"
#include "intel_memory_core/l0_external_memmap.hpp"
#include "intel_buffer_backend_msgs/msg/intel_buffer_descriptor.hpp"
#include <unistd.h>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>

#include "latency_stats.hpp"

using IntelBufferDescriptor = intel_buffer_backend_msgs::msg::IntelBufferDescriptor;

namespace
{
inline int64_t mono_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

class UsmOpenvinoPipeline : public rclcpp::Node
{
public:
  explicit UsmOpenvinoPipeline(const rclcpp::NodeOptions & options)
  : Node("openvino_pipeline", options)
  {
    this->declare_parameter<std::string>("model", "");
    this->declare_parameter<std::string>("device", "GPU");
    this->declare_parameter<std::string>("frame_topic", "/usm/camera0/frame");
    this->declare_parameter<std::string>(
      "detections_topic", "/ros2_openvino_toolkit/detected_objects");
    this->declare_parameter<double>("score_thresh", 0.4);
    this->declare_parameter<std::string>("input_format", "bgr8");

    model_path_ = this->get_parameter("model").as_string();
    device_ = this->get_parameter("device").as_string();
    input_format_ = this->get_parameter("input_format").as_string();
    std::string frame_topic = this->get_parameter("frame_topic").as_string();
    std::string det_topic = this->get_parameter("detections_topic").as_string();
    score_thresh_ = this->get_parameter("score_thresh").as_double();
    nv12_input_ = (input_format_ == "nv12");

    det_pub_ = this->create_publisher<
      vision_msgs::msg::Detection2DArray>(det_topic, 10);

    RCLCPP_INFO(this->get_logger(),
      "USM OpenVINO pipeline: device=%s topic=%s (compiling model...)",
      device_.c_str(), frame_topic.c_str());

    if (!model_path_.empty() && !(device_ == "NPU" && nv12_input_)) {
      compile_model();
    } else if (!model_path_.empty()) {
      RCLCPP_INFO(this->get_logger(),
        "NPU + NV12: deferring compile to first frame "
        "(source resolution needed for the static NV12 preprocessor)");
    }

    frame_sub_ = this->create_subscription<IntelBufferDescriptor>(
      frame_topic, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      std::bind(&UsmOpenvinoPipeline::on_frame, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
      "USM OpenVINO pipeline: ready, subscribed to %s", frame_topic.c_str());
  }

  ~UsmOpenvinoPipeline() override
  {
    intel_buffer_backend::release_all_cl_mem();
  }

private:
  bool compile_model(int src_h = 0, int src_w = 0)
  {
    bool device_nv12 = nv12_input_ &&
      (device_ != "NPU" || (src_h > 0 && src_w > 0));
    try {
      ov::Core core;
      auto model = core.read_model(model_path_);

      auto input = model->input(0);
      auto pshape = input.get_partial_shape();

      std::string model_layout = "NHWC";
      int net_h = 300, net_w = 300;
      if (pshape.rank().get_length() == 4) {
        if (pshape[1].is_static() && pshape[1].get_length() == 3) {
          model_layout = "NCHW";
	  std::cout << "Model Layout is: "
		  << model_layout
		  << std::endl;
          if (pshape[2].is_static()) net_h = static_cast<int>(pshape[2].get_length());
          if (pshape[3].is_static()) net_w = static_cast<int>(pshape[3].get_length());
        } else if (pshape[3].is_static() && pshape[3].get_length() == 3) {
          model_layout = "NHWC";
	  std::cout << "Model Layout is: "
                  << model_layout
                  << std::endl;
          if (pshape[1].is_static()) net_h = static_cast<int>(pshape[1].get_length());
          if (pshape[2].is_static()) net_w = static_cast<int>(pshape[2].get_length());
        }
      }
      else
      {
	      std::cout << "No static height or width of the model"
		      << std::endl;
	      return false;

      }


      // Reshape to static dimensions (ovc-converted models may be dynamic)
      if (model_layout == "NCHW") {
        model->reshape({1, 3, net_h, net_w});
      } else {
        model->reshape({1, net_h, net_w, 3});
      }

      ov::preprocess::PrePostProcessor ppp(model);

      auto & input_info = ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC");

      if (device_nv12) {
        input_info.set_color_format(
          ov::preprocess::ColorFormat::NV12_SINGLE_PLANE);
      } else {
        input_info.set_color_format(ov::preprocess::ColorFormat::BGR);
      }

      // NPU requires fully static spatial shapes; GPU accepts dynamic input.
      if (device_ == "NPU") {
        if (device_nv12) {
          input_info.set_spatial_static_shape(src_h, src_w);
        } else {
          input_info.set_spatial_static_shape(net_h, net_w);
        }
      } else {
        input_info.set_spatial_dynamic_shape();
      }

      ppp.input().model().set_layout(ov::Layout(model_layout));
      auto & preprocess = ppp.input().preprocess();
      if (device_nv12) {
        preprocess.convert_color(ov::preprocess::ColorFormat::BGR);
      }
      preprocess
        .convert_element_type(ov::element::f32)
        .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
      model = ppp.build();

      ov::AnyMap config;
      if (device_ != "NPU") {
        config["PERFORMANCE_HINT"] = "LATENCY";
      }
      auto compiled = core.compile_model(model, device_, config);

      std::lock_guard<std::mutex> lock(model_mutex_);
      compiled_model_ = std::make_shared<ov::CompiledModel>(compiled);
      infer_request_ = compiled_model_->create_infer_request();
      net_h_ = net_h;
      net_w_ = net_w;
      nv12_gpu_preproc_ = device_nv12;
      model_ready_ = true;

      auto out_shape = compiled_model_->output(0).get_partial_shape();
      size_t num_outputs = compiled_model_->outputs().size();
      auto rank = out_shape.rank().get_length();

      // Single-tensor SSD: last dim == 7
      if (num_outputs == 1 && rank >= 2) {
        auto last_dim = out_shape[rank - 1];
        if (last_dim.is_static() && last_dim.get_length() == 7) {
          output_kind_ = "ssd";
        }
      }
      // TF Object Detection API: multiple outputs (boxes, classes, scores, num)
      else if (num_outputs >= 3) {
        output_kind_ = "ssd_tf";
      }

      RCLCPP_INFO(this->get_logger(),
        "output: shape=%s, rank=%zu, num_outputs=%zu, kind=%s",
        out_shape.to_string().c_str(), rank, num_outputs, output_kind_.c_str());

      RCLCPP_INFO(this->get_logger(),
        "model compiled: %s on %s (%dx%d, output=%s, preproc=%s)",
        model_path_.c_str(), device_.c_str(), net_w_, net_h_,
        output_kind_.c_str(), device_nv12 ? "device-NV12" : "host");
      return true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "model compile failed: %s", e.what());
      return false;
    }
  }

  bool try_compile_on_first_frame(int src_h, int src_w)
  {
    if (compile_attempted_) {
      return model_ready_;
    }
    compile_attempted_ = true;

    RCLCPP_INFO(this->get_logger(),
      "NPU + NV12: attempting on-device NV12 preprocessing (%dx%d)...",
      src_w, src_h);
    if (compile_model(src_h, src_w)) {
      RCLCPP_INFO(this->get_logger(),
        "NPU on-device NV12 preprocessing enabled (driver supports it)");
      return true;
    }

    RCLCPP_WARN(this->get_logger(),
      "NPU rejected on-device NV12 preprocessing — installed NPU driver does "
      "not support it; falling back to CPU NV12→BGR conversion");
    return compile_model();
  }

  void on_frame(const IntelBufferDescriptor::SharedPtr msg)
  {
    int img_h = msg->height;
    int img_w = msg->width;
    if (img_h <= 0 || img_w <= 0) return;

    if (!model_ready_ && device_ == "NPU" && nv12_input_) {
      if (!try_compile_on_first_frame(img_h, img_w)) return;
    }
    if (!model_ready_) return;

    auto compute_start = std::chrono::steady_clock::now();

    auto import = import_buffer(*msg);
    const uint8_t * frame_ptr = static_cast<const uint8_t *>(import.ptr);
    if (!frame_ptr) return;

    if (msg->buffer_ready_stamp_ns != 0) {
      double read_delay_ms = static_cast<double>(
        this->now().nanoseconds() - static_cast<int64_t>(msg->buffer_ready_stamp_ns)) / 1e6;
      read_delay_ms_accum_ += read_delay_ms;
      read_delay_count_++;
    }

    if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
      int64_t capture_ns = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000ll +
        static_cast<int64_t>(msg->header.stamp.nanosec);
      double capture_read_ms = static_cast<double>(mono_now_ns() - capture_ns) / 1e6;
      capture_read_ms_accum_ += capture_read_ms;
      capture_read_count_++;
    }

    if (infer_count_ % kSerdesSampleStride == 0) {
      static rclcpp::Serialization<IntelBufferDescriptor> serde;
      rclcpp::SerializedMessage serialized;
      auto ser_start = std::chrono::steady_clock::now();
      serde.serialize_message(msg.get(), &serialized);
      serialize_stats_.add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ser_start).count());

      IntelBufferDescriptor roundtrip;
      auto deser_start = std::chrono::steady_clock::now();
      serde.deserialize_message(&serialized, &roundtrip);
      deserialize_stats_.add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - deser_start).count());
    }

    ov::Tensor input_tensor;
    cv::Mat staged;

    if (nv12_gpu_preproc_) {
      ov::Shape input_shape = {1,
        static_cast<size_t>(img_h) * 3 / 2,
        static_cast<size_t>(img_w), 1};
      input_tensor = try_make_zero_copy_tensor(import, msg->transport_mode, input_shape);
      if (!input_tensor) {
        input_tensor = ov::Tensor(ov::element::u8, input_shape,
          const_cast<uint8_t *>(frame_ptr));
        staged_frames_++;   // host->GPU staging copy this frame
      } else {
        zero_copy_frames_++;  // GPU read shared pages directly, no staging copy
      }
    } else if (device_ == "NPU") {
      cv::Mat bgr;
      if (nv12_input_) {
        cv::Mat nv12(img_h * 3 / 2, img_w, CV_8UC1,
          const_cast<uint8_t *>(frame_ptr));
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
      } else {
        bgr = cv::Mat(img_h, img_w, CV_8UC3, const_cast<uint8_t *>(frame_ptr));
      }
      if (bgr.cols != net_w_ || bgr.rows != net_h_) {
        cv::resize(bgr, staged, cv::Size(net_w_, net_h_));
      } else {
        staged = bgr;
      }
      ov::Shape input_shape = {1,
        static_cast<size_t>(net_h_),
        static_cast<size_t>(net_w_), 3};
      input_tensor = ov::Tensor(ov::element::u8, input_shape, staged.data);
    } else {
      // GPU: PPP handles dynamic resize on device — pass full frame directly
      ov::Shape input_shape = {1,
        static_cast<size_t>(img_h),
        static_cast<size_t>(img_w), 3};
      // True zero-copy via GPU RemoteTensor (see nv12 branch); host fallback.
      input_tensor = try_make_zero_copy_tensor(import, msg->transport_mode, input_shape);
      if (!input_tensor) {
        input_tensor = ov::Tensor(ov::element::u8, input_shape,
          const_cast<uint8_t *>(frame_ptr));
        staged_frames_++;     // host->GPU staging copy this frame
      } else {
        zero_copy_frames_++;  // GPU read shared pages directly, no staging copy
      }
    }

    {
      std::lock_guard<std::mutex> lock(model_mutex_);
      infer_request_.set_input_tensor(input_tensor);
      infer_request_.infer();

      if (output_kind_ == "ssd") {
        publish_ssd_detections(msg->header, img_h, img_w);
      } else if (output_kind_ == "ssd_tf") {
        publish_ssd_tf_detections(msg->header, img_h, img_w);
      }
    }

    double compute_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - compute_start).count();
    compute_stats_.add(compute_ms);

    if (msg->transport_mode == IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP &&
      !intel_buffer_backend::ExternalMemMapImporter::generation_unchanged(
        import.ptr, import.region_size, msg->extmap_desc.slot_generation))
    {
      torn_frames_++;
    }

    intel_buffer_backend::IntelMemoryIPCManager::release_block(import.ipc_meta);

    infer_count_++;
    if (infer_count_ % 1000 == 0) {
      compute_stats_.report(this->get_logger(), "buffer-read+preproc+infer latency");
      serialize_stats_.report(
        this->get_logger(), "subscriber CDR serialize cost (sampled, descriptor)");
      deserialize_stats_.report(
        this->get_logger(), "subscriber CDR deserialize cost (sampled, descriptor)");
    }
    if (infer_count_ % 100 == 0) {
      if (read_delay_count_ > 0) {
        RCLCPP_INFO(this->get_logger(),
          "buffer-ready->read delay: avg %.3f ms over last %zu frames",
          read_delay_ms_accum_ / static_cast<double>(read_delay_count_), read_delay_count_);
        read_delay_ms_accum_ = 0.0;
        read_delay_count_ = 0;
      }
      if (capture_read_count_ > 0) {
        RCLCPP_INFO(this->get_logger(),
          "capture->read delay: avg %.3f ms over last %zu frames",
          capture_read_ms_accum_ / static_cast<double>(capture_read_count_), capture_read_count_);
        capture_read_ms_accum_ = 0.0;
        capture_read_count_ = 0;
      }
      RCLCPP_INFO(this->get_logger(),
        "inferences=%zu (zero-copy)", infer_count_);
      if (torn_frames_ > 0) {
        RCLCPP_WARN(this->get_logger(),
          "external-memmap torn reads: %zu of last 100 frames — ring depth may "
          "not cover this node's read+infer latency", torn_frames_);
      }
      torn_frames_ = 0;
      if (device_ == "GPU") {
        RCLCPP_INFO(this->get_logger(),
          "host->GPU staging: %zu zero-copy, %zu staged (of last 100 frames)",
          zero_copy_frames_, staged_frames_);
        zero_copy_frames_ = 0;
        staged_frames_ = 0;
      }
    }
  }

  bool ensure_remote_context()
  {
    if (remote_ctx_ready_) {
      return cl_context_handle_ != nullptr;
    }
    remote_ctx_ready_ = true;
    try {
      remote_ctx_ = compiled_model_->get_context();
      cl_context_handle_ = remote_ctx_.get_params()
        .at(ov::intel_gpu::ocl_context.name()).as<void *>();
    } catch (const std::exception & e) {
      RCLCPP_WARN_ONCE(this->get_logger(),
        "ZERO-COPY FALLBACK: could not obtain GPU cl_context (%s) — using "
        "host-staged ov::Tensor", e.what());
      cl_context_handle_ = nullptr;
    }
    return cl_context_handle_ != nullptr;
  }

  ov::Tensor try_make_zero_copy_tensor(
    const intel_buffer_backend::IntelMemoryIPCManager::ImportResult & import,
    uint8_t transport_mode,
    const ov::Shape & shape)
  {
    if (transport_mode == IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP) {
      return try_make_host_ptr_tensor(import.ptr, import.region_size, shape);
    }
    return try_make_remote_tensor(import.dmabuf_fd, shape);
  }

  ov::Tensor try_make_host_ptr_tensor(void * host_ptr, uint64_t region_size, const ov::Shape & shape)
  {
    if (device_ != "GPU" || host_ptr == nullptr) {
      return ov::Tensor();
    }
    if (!intel_buffer_backend::cl_dmabuf_import_available()) {
      RCLCPP_WARN_ONCE(this->get_logger(),
        "ZERO-COPY FALLBACK: intel_memory_core built without OpenCL — using "
        "host-staged ov::Tensor (host->GPU copy per frame)");
      return ov::Tensor();
    }
    if (!ensure_remote_context()) {
      return ov::Tensor();
    }

    const uint64_t tensor_size = cl_import_size(shape);
    const bool region_size_known = region_size != 0;
    const uint64_t wrap_size = region_size_known ? region_size : tensor_size;
    void * mem = intel_buffer_backend::get_or_wrap_host_ptr_cl_mem(
      cl_context_handle_, host_ptr, wrap_size);
    if (!mem) {
      RCLCPP_WARN_ONCE(this->get_logger(),
        "ZERO-COPY FALLBACK: OpenCL USE_HOST_PTR wrap rejected (ptr=%p) — using "
        "host-staged ov::Tensor", host_ptr);
      return ov::Tensor();
    }
    if (!intel_buffer_backend::sync_host_ptr_cl_mem(
        cl_context_handle_, mem, tensor_size))
    {
      RCLCPP_WARN_ONCE(this->get_logger(),
        "ZERO-COPY FALLBACK: OpenCL USE_HOST_PTR resync failed (ptr=%p) — using "
        "host-staged ov::Tensor", host_ptr);
      return ov::Tensor();
    }

    std::string outcome = region_size_known ?
      "GPU zero-copy enabled: producer-owned pages wrapped as an OCL_BUFFER "
      "RemoteTensor via CL_MEM_USE_HOST_PTR using the region's page-aligned "
      "size (" + std::to_string(wrap_size) + " bytes) — pinning alignment "
      "requirement met, no host->GPU staging copy expected" :
      "GPU zero-copy uncertain: producer-owned pages wrapped as an OCL_BUFFER "
      "RemoteTensor via CL_MEM_USE_HOST_PTR, but only the tensor's byte count "
      "(" + std::to_string(wrap_size) + " bytes, not known to be page-aligned) "
      "was available as the wrap size — the runtime may have silently copied "
      "instead of pinning";
    return make_ocl_buffer_tensor(mem, shape, outcome);
  }

  ov::Tensor try_make_remote_tensor(int fd, const ov::Shape & shape)
  {
    if (device_ != "GPU" || fd < 0) {
      return ov::Tensor();  // not the GPU dma_buf path
    }
    if (!intel_buffer_backend::cl_dmabuf_import_available()) {
      RCLCPP_WARN_ONCE(this->get_logger(),
        "ZERO-COPY FALLBACK: intel_memory_core built without OpenCL — using "
        "host-staged ov::Tensor (host->GPU copy per frame)");
      return ov::Tensor();
    }

    if (!ensure_remote_context()) {
      return ov::Tensor();
    }

    void * mem = intel_buffer_backend::get_or_import_cl_mem(
      cl_context_handle_, fd, cl_import_size(shape));
    if (!mem) {
      RCLCPP_WARN_ONCE(this->get_logger(),
        "ZERO-COPY FALLBACK: OpenCL dma_buf import rejected (fd=%d) — using "
        "host-staged ov::Tensor", fd);
      return ov::Tensor();
    }

    return make_ocl_buffer_tensor(
      mem, shape,
      "GPU zero-copy enabled: DMA-BUF pages imported as OCL_BUFFER "
      "RemoteTensor (no host->GPU staging copy)");
  }

  ov::Tensor make_ocl_buffer_tensor(
    void * mem, const ov::Shape & shape, const std::string & success_log)
  {
    try {
      ov::AnyMap params = {
        {ov::intel_gpu::shared_mem_type.name(),
          ov::intel_gpu::SharedMemType::OCL_BUFFER},
        {ov::intel_gpu::mem_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(mem)}};
      ov::Tensor t = remote_ctx_.create_tensor(ov::element::u8, shape, params);
      if (!zero_copy_logged_) {
        RCLCPP_INFO(this->get_logger(), "%s", success_log.c_str());
        zero_copy_logged_ = true;
      }
      return t;
    } catch (const std::exception & e) {
      RCLCPP_WARN_ONCE(this->get_logger(),
        "ZERO-COPY FALLBACK: create_tensor(OCL_BUFFER) failed (%s) — using "
        "host-staged ov::Tensor", e.what());
      return ov::Tensor();
    }
  }

  static uint64_t cl_import_size(const ov::Shape & shape)
  {
    uint64_t n = 1;
    for (auto d : shape) n *= static_cast<uint64_t>(d);
    return n;
  }

  intel_buffer_backend::IntelMemoryIPCManager::ImportResult import_buffer(
    const IntelBufferDescriptor& frame)
  {
    if (frame.transport_mode == IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP) {
      const auto & d = frame.extmap_desc;
      if (d.local_ptr != 0 && d.publisher_pid == static_cast<int32_t>(getpid())) {
        return {reinterpret_cast<void *>(d.local_ptr), nullptr};
      }
      if (d.size == 0) return {nullptr, nullptr};
      size_t region_size = 0;
      void * ptr = intel_buffer_backend::ExternalMemMapImporter::import(
        d.shm_name, d.size, d.publisher_pid, &region_size);
      if (!ptr) {
        RCLCPP_ERROR(this->get_logger(),
          "external-memmap import failed for %s", d.shm_name.c_str());
        return {nullptr, nullptr};
      }
      return {ptr, nullptr, -1, region_size};
    }

    // Intra-process fast path: same process, direct pointer access
    if (frame.transport_mode == IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM) {
      const auto & d = frame.l0_desc;
      if (d.local_ptr != 0 && d.publisher_pid == static_cast<int32_t>(getpid())) {
        return {reinterpret_cast<void *>(d.local_ptr), nullptr};
      }
      if (d.size == 0) return {nullptr, nullptr};
      try {
        return intel_buffer_backend::IntelMemoryIPCManager::import_level_zero_block(
          d.ze_ipc_handle.data(), d.ze_ipc_handle_size,
          d.ze_device_ordinal, d.ze_device_uuid.data(), d.size,
          d.publisher_pid, d.block_id, d.ipc_uid, d.l0_socket_path, d.pool_id);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "L0 IPC import failed: %s", e.what());
        return {nullptr, nullptr};
      }
    } else {
      const auto & d = frame.dmabuf_desc;
      if (d.local_ptr != 0 && d.publisher_pid == static_cast<int32_t>(getpid())) {
        return {reinterpret_cast<void *>(d.local_ptr), nullptr};
      }
      if (d.size == 0) return {nullptr, nullptr};
      try {
        return intel_buffer_backend::IntelMemoryIPCManager::import_dmabuf_block(
          d.dmabuf_socket_path, d.dmabuf_pid,
          d.dmabuf_pool_block_id, d.dmabuf_block_size, d.ipc_uid, d.pool_id);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "DMA-BUF import failed: %s", e.what());
        return {nullptr, nullptr};
      }
    }
  }

  void publish_ssd_detections(
    const std_msgs::msg::Header & header, int img_h, int img_w)
  {
    auto output_tensor = infer_request_.get_output_tensor();
    const float * data = output_tensor.data<float>();
    auto shape = output_tensor.get_shape();
    size_t n_detections = shape[2];

    vision_msgs::msg::Detection2DArray det_array;
    det_array.header = header;

    for (size_t i = 0; i < n_detections; ++i) {
      const float * row = data + i * 7;
      if (row[0] < 0) break;
      float score = row[2];
      if (score < score_thresh_) continue;

      float x1 = row[3] * static_cast<float>(img_w);
      float y1 = row[4] * static_cast<float>(img_h);
      float x2 = row[5] * static_cast<float>(img_w);
      float y2 = row[6] * static_cast<float>(img_h);
      if (x2 <= x1 || y2 <= y1) continue;

      vision_msgs::msg::Detection2D det;
      det.header = header;
      det.bbox.center.position.x = 0.5 * (x1 + x2);
      det.bbox.center.position.y = 0.5 * (y1 + y2);
      det.bbox.size_x = x2 - x1;
      det.bbox.size_y = y2 - y1;

      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = std::to_string(static_cast<int>(row[1]));
      hyp.hypothesis.score = score;
      det.results.push_back(hyp);
      det_array.detections.push_back(det);
    }

    det_pub_->publish(det_array);
  }

  void publish_ssd_tf_detections(
    const std_msgs::msg::Header & header, int img_h, int img_w)
  {
    const float * boxes = nullptr;
    const float * classes = nullptr;
    const float * scores = nullptr;
    size_t max_dets = 0;

    size_t num_outputs = compiled_model_->outputs().size();
    for (size_t idx = 0; idx < num_outputs; ++idx) {
      auto tensor = infer_request_.get_output_tensor(idx);
      auto shape = tensor.get_shape();
      const float * ptr = tensor.data<float>();

      if (shape.size() == 3 && shape[2] == 4) {
        boxes = ptr;
        max_dets = shape[1];
      } else if (shape.size() == 2) {
        if (!classes) classes = ptr;
        else if (!scores) scores = ptr;
      }
    }

    if (!boxes || !classes || !scores) {
      if (num_outputs >= 3) {
        auto t0 = infer_request_.get_output_tensor(0);
        auto t1 = infer_request_.get_output_tensor(1);
        auto t2 = infer_request_.get_output_tensor(2);
        boxes = t0.data<float>();
        max_dets = t0.get_size() / 4;
        classes = t1.data<float>();
        scores = t2.data<float>();
      }
    }

    if (!boxes || !classes || !scores || max_dets == 0) return;

    vision_msgs::msg::Detection2DArray det_array;
    det_array.header = header;

    for (size_t i = 0; i < max_dets; ++i) {
      float score = scores[i];
      if (score < score_thresh_) continue;

      float y1 = boxes[i * 4 + 0] * static_cast<float>(img_h);
      float x1 = boxes[i * 4 + 1] * static_cast<float>(img_w);
      float y2 = boxes[i * 4 + 2] * static_cast<float>(img_h);
      float x2 = boxes[i * 4 + 3] * static_cast<float>(img_w);
      if (x2 <= x1 || y2 <= y1) continue;

      vision_msgs::msg::Detection2D det;
      det.header = header;
      det.bbox.center.position.x = 0.5 * (x1 + x2);
      det.bbox.center.position.y = 0.5 * (y1 + y2);
      det.bbox.size_x = x2 - x1;
      det.bbox.size_y = y2 - y1;

      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = std::to_string(static_cast<int>(classes[i]));
      hyp.hypothesis.score = score;
      det.results.push_back(hyp);
      det_array.detections.push_back(det);
    }

    det_pub_->publish(det_array);
  }

  rclcpp::Subscription<IntelBufferDescriptor>::SharedPtr frame_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;

  std::string model_path_;
  std::string device_;
  std::string input_format_;
  bool nv12_input_{false};
  bool nv12_gpu_preproc_{false};
  bool compile_attempted_{false};
  double score_thresh_;

  std::mutex model_mutex_;
  std::shared_ptr<ov::CompiledModel> compiled_model_;
  ov::InferRequest infer_request_;
  bool model_ready_{false};
  // GPU zero-copy (DMA-BUF -> OpenCL cl_mem -> OCL_BUFFER RemoteTensor) state.
  ov::RemoteContext remote_ctx_;      // the GPU plugin's context
  void * cl_context_handle_{nullptr}; // its cl_context (opaque), for the core helper
  bool remote_ctx_ready_{false};      // context fetch attempted (success or fail)
  bool zero_copy_logged_{false};      // "zero-copy enabled" logged once
  size_t zero_copy_frames_{0};        // GPU frames read direct (per-100 window)
  size_t staged_frames_{0};           // GPU frames that host-staged (per-100 window)

  size_t torn_frames_{0};             // (per-100 window)
  int net_h_{300};
  int net_w_{300};
  std::string output_kind_{"unknown"};
  size_t infer_count_{0};

  LatencyStats compute_stats_{60};    // buffer-read+preproc+infer, reported per 1000
  double read_delay_ms_accum_{0.0};   // sum of buffer-ready->read ms in the window
  size_t read_delay_count_{0};        // stamped frames in the window
  double capture_read_ms_accum_{0.0}; // sum of capture->read ms in the window
  size_t capture_read_count_{0};      // capture-stamped frames in the window

  static constexpr size_t kSerdesSampleStride = 10;
  LatencyStats serialize_stats_{6};    // warmup scaled to the sample rate
  LatencyStats deserialize_stats_{6};
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(UsmOpenvinoPipeline)
