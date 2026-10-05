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
#include <vector>

#include <opencv2/imgproc.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "intel_buffer/intel_buffer_api.hpp"
#include "intel_memory_core/intel_memory_ipc_manager.hpp"
#include "intel_memory_core/l0_external_memmap.hpp"
#include "intel_buffer_backend_msgs/msg/intel_buffer_descriptor.hpp"
#include <unistd.h>

#include "latency_stats.hpp"

using IntelBufferDescriptor = intel_buffer_backend_msgs::msg::IntelBufferDescriptor;
using Detection2DArray = vision_msgs::msg::Detection2DArray;

namespace
{
inline int64_t mono_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

class UsmDetectionVisualizer : public rclcpp::Node
{
public:
  explicit UsmDetectionVisualizer(const rclcpp::NodeOptions & options)
  : Node("usm_detection_visualizer", options)
  {
    this->declare_parameter<std::string>("frame_topic", "/usm/camera0/frame");
    this->declare_parameter<std::string>(
      "detections_topic", "/ros2_openvino_toolkit/detected_objects");
    this->declare_parameter<std::string>(
      "annotated_topic", "/usm/camera0/annotated");
    this->declare_parameter<double>("throttle_hz", 30.0);

    std::string frame_topic = this->get_parameter("frame_topic").as_string();
    std::string det_topic = this->get_parameter("detections_topic").as_string();
    std::string ann_topic = this->get_parameter("annotated_topic").as_string();
    double throttle_hz = this->get_parameter("throttle_hz").as_double();

    min_period_ns_ = (throttle_hz > 0.0)
      ? static_cast<int64_t>(1e9 / throttle_hz) : 0;

    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(ann_topic, 10);

    frame_sub_ = this->create_subscription<IntelBufferDescriptor>(
      frame_topic, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      std::bind(&UsmDetectionVisualizer::on_frame, this, std::placeholders::_1));

    det_sub_ = this->create_subscription<Detection2DArray>(
      det_topic, 10,
      std::bind(&UsmDetectionVisualizer::on_detections, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
      "USM detection visualizer: frame=%s det=%s -> %s @ %.0f Hz",
      frame_topic.c_str(), det_topic.c_str(), ann_topic.c_str(), throttle_hz);
  }

private:
  void on_detections(const Detection2DArray::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(det_mutex_);
    latest_detections_ = msg;
  }

  void on_frame(const IntelBufferDescriptor::SharedPtr msg)
  {
    // Throttle output rate
    auto now_ns = this->now().nanoseconds();
    if (min_period_ns_ > 0 && (now_ns - last_publish_ns_) < min_period_ns_) {
      return;
    }

    int img_h = msg->height;
    int img_w = msg->width;
    if (img_h <= 0 || img_w <= 0) return;

    int64_t origin_ns = 0;
    if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
      origin_ns = static_cast<int64_t>(msg->header.stamp.sec) * 1000000000ll +
        static_cast<int64_t>(msg->header.stamp.nanosec);
    }

    auto import = import_buffer(*msg);
    const uint8_t * frame_ptr = static_cast<const uint8_t *>(import.ptr);
    if (!frame_ptr) {
      RCLCPP_WARN_ONCE(this->get_logger(), "buffer import returned null");
      return;
    }

    if (msg->buffer_ready_stamp_ns != 0) {
      double read_delay_ms = static_cast<double>(
        this->now().nanoseconds() - static_cast<int64_t>(msg->buffer_ready_stamp_ns)) / 1e6;
      read_delay_ms_accum_ += read_delay_ms;
      read_delay_count_++;
      if (read_delay_count_ == 100) {
        RCLCPP_INFO(this->get_logger(),
          "buffer-ready->read delay: avg %.3f ms over last 100 frames",
          read_delay_ms_accum_ / 100.0);
        read_delay_ms_accum_ = 0.0;
        read_delay_count_ = 0;
      }
    }

    cv::Mat annotated;
    if (msg->encoding == "nv12") {
      size_t y_step = (msg->stride >= static_cast<size_t>(img_w))
        ? msg->stride : static_cast<size_t>(img_w);
      cv::Mat nv12(img_h * 3 / 2, img_w, CV_8UC1,
        const_cast<uint8_t *>(frame_ptr), y_step);
      cv::cvtColor(nv12, annotated, cv::COLOR_YUV2BGR_NV12);
    } else {
      size_t min_step = static_cast<size_t>(img_w) * 3;
      size_t step = (msg->stride >= min_step) ? msg->stride : min_step;
      cv::Mat canvas(img_h, img_w, CV_8UC3,
        const_cast<uint8_t *>(frame_ptr), step);
      canvas.copyTo(annotated);
    }

    intel_buffer_backend::IntelMemoryIPCManager::release_block(import.ipc_meta);

    // Draw bounding boxes from latest detections
    {
      std::lock_guard<std::mutex> lock(det_mutex_);
      if (latest_detections_) {
        draw_detections(annotated, *latest_detections_);
      }
    }

    // Publish annotated image for RViz2
    sensor_msgs::msg::Image out;
    out.header = msg->header;
    out.height = img_h;
    out.width = img_w;
    out.encoding = "bgr8";
    out.step = img_w * 3;
    out.is_bigendian = 0;
    size_t data_size = static_cast<size_t>(img_h) * out.step;
    out.data.assign(annotated.data, annotated.data + data_size);

    image_pub_->publish(out);
    last_publish_ns_ = now_ns;

    if (origin_ns != 0) {
      double e2e_ms = static_cast<double>(mono_now_ns() - origin_ns) / 1e6;
      e2e_stats_.add(e2e_ms);
      if (++e2e_count_ % 100 == 0) {
        e2e_stats_.report(
          this->get_logger(), "end-to-end pipeline latency (V4L2 capture->visualized)");
      }
    }
  }

  static const char * coco_label(int class_id)
  {
    static const char * labels[] = {
      "background", "person", "bicycle", "car", "motorcycle", "airplane",
      "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
      "street sign", "stop sign", "parking meter", "bench", "bird", "cat",
      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
      "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
      "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
      "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup",
      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
      "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
      "chair", "couch", "potted plant", "bed", "mirror", "dining table",
      "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
      "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
      "sink", "refrigerator", "blender", "book", "clock", "vase",
      "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    static constexpr int n = sizeof(labels) / sizeof(labels[0]);
    if (class_id >= 0 && class_id < n) return labels[class_id];
    return "unknown";
  }

  void draw_detections(cv::Mat & img, const Detection2DArray & dets)
  {
    static const cv::Scalar colors[] = {
      {0, 255, 0}, {255, 0, 0}, {0, 0, 255},
      {255, 255, 0}, {0, 255, 255}, {255, 0, 255}
    };

    for (size_t i = 0; i < dets.detections.size(); ++i) {
      const auto & det = dets.detections[i];
      double cx = det.bbox.center.position.x;
      double cy = det.bbox.center.position.y;
      double sx = det.bbox.size_x;
      double sy = det.bbox.size_y;

      int x1 = static_cast<int>(cx - sx * 0.5);
      int y1 = static_cast<int>(cy - sy * 0.5);
      int x2 = static_cast<int>(cx + sx * 0.5);
      int y2 = static_cast<int>(cy + sy * 0.5);

      const cv::Scalar & color = colors[i % 6];
      cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

      if (!det.results.empty()) {
        const auto & hyp = det.results[0].hypothesis;
        int class_id = 0;
        try { class_id = std::stoi(hyp.class_id); } catch (...) {}
        std::string label = std::string(coco_label(class_id)) + " " +
          std::to_string(static_cast<int>(hyp.score * 100)) + "%";
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
          label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::rectangle(img,
          cv::Point(x1, y1 - text_size.height - 6),
          cv::Point(x1 + text_size.width + 4, y1),
          color, cv::FILLED);
        cv::putText(img, label, cv::Point(x1 + 2, y1 - 3),
          cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
      }
    }
  }

  intel_buffer_backend::IntelMemoryIPCManager::ImportResult import_buffer(
    const IntelBufferDescriptor & frame)
  {
    if (frame.transport_mode == IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP) {
      const auto & d = frame.extmap_desc;
      if (d.local_ptr != 0 && d.publisher_pid == static_cast<int32_t>(getpid())) {
        return {reinterpret_cast<void *>(d.local_ptr), nullptr};
      }
      if (d.size == 0) return {nullptr, nullptr};
      return {intel_buffer_backend::ExternalMemMapImporter::import(
          d.shm_name, d.size, d.publisher_pid), nullptr};
    }

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
      } catch (...) { return {nullptr, nullptr}; }
    } else if (frame.transport_mode == IntelBufferDescriptor::TRANSPORT_DMA_BUF) {
      const auto & d = frame.dmabuf_desc;
      if (d.local_ptr != 0 && d.publisher_pid == static_cast<int32_t>(getpid())) {
        return {reinterpret_cast<void *>(d.local_ptr), nullptr};
      }
      if (d.size == 0) return {nullptr, nullptr};
      try {
        return intel_buffer_backend::IntelMemoryIPCManager::import_dmabuf_block(
          d.dmabuf_socket_path, d.dmabuf_pid,
          d.dmabuf_pool_block_id, d.dmabuf_block_size, d.ipc_uid, d.pool_id);
      } catch (...) { return {nullptr, nullptr}; }
    }
    return {nullptr, nullptr};
  }

  rclcpp::Subscription<IntelBufferDescriptor>::SharedPtr frame_sub_;
  rclcpp::Subscription<Detection2DArray>::SharedPtr det_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  std::mutex det_mutex_;
  Detection2DArray::SharedPtr latest_detections_;
  int64_t min_period_ns_{0};
  int64_t last_publish_ns_{0};
  double read_delay_ms_accum_{0.0};   // sum of buffer-ready->read ms in the window
  size_t read_delay_count_{0};        // stamped frames in the window
  LatencyStats e2e_stats_{60};        // reported per 100 frames
  size_t e2e_count_{0};               // stamped frames seen (drives the per-100 report)
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(UsmDetectionVisualizer)
