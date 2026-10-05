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

/// @file usm_video_publisher_node.cpp
/// @brief Camera publisher with two source modes. Publishes IntelBufferDescriptor (metadata
/// only — ~200 bytes on the wire). Pixel data stays in shared physical memory.
///
///   "simulated" — video file → allocate_buffer() + write_to_buffer()
///                 (backing store selected via transport_mode: L0_USM/DMA_BUF),
///                 or, for transport_mode=EXTERNAL_MEMMAP, a ring of
///                 producer-owned SHM regions mapped to the device once and
///                 wrapped per frame with wrap_external_memmap() (no pool).
///   "camera"    — V4L2 camera → wrap_dmabuf() (zero memcpy end-to-end)

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "rclcpp/rclcpp.hpp"
#include "intel_buffer/intel_buffer_api.hpp"
#include "intel_memory_core/intel_memory_buffer_pool.hpp"
#include "intel_memory_core/l0_usm_ipc.hpp"
#include "intel_memory_core/l0_external_memmap.hpp"
#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_buffer_backend_msgs/msg/intel_buffer_descriptor.hpp"

#include "latency_stats.hpp"
using IntelBufferDescriptor = intel_buffer_backend_msgs::msg::IntelBufferDescriptor;

namespace
{

inline int64_t mono_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Pack a monotonic-ns value into a builtin_interfaces/Time header stamp.
inline void set_stamp_ns(builtin_interfaces::msg::Time & stamp, int64_t ns)
{
  stamp.sec = static_cast<int32_t>(ns / 1000000000ll);
  stamp.nanosec = static_cast<uint32_t>(ns % 1000000000ll);
}
}  // namespace

class UsmVideoPublisher : public rclcpp::Node
{
public:
  explicit UsmVideoPublisher(const rclcpp::NodeOptions & options)
  : Node("video_dmabuf_sim", options), frame_seq_(0)
  {
    this->declare_parameter<std::string>("source_mode", "simulated");
    this->declare_parameter<std::string>("video_path", "");
    this->declare_parameter<std::string>("camera_device", "/dev/video0");
    this->declare_parameter<double>("framerate", 0.0);
    this->declare_parameter<bool>("loop", true);
    this->declare_parameter<std::string>("raw_format", "YUYV");
    this->declare_parameter<std::string>("frame_topic", "/usm/camera0/frame");
    this->declare_parameter<std::string>("intel_memory_buffer_type", "L0_USM");

    source_mode_ = this->get_parameter("source_mode").as_string();
    video_path_ = this->get_parameter("video_path").as_string();
    camera_device_ = this->get_parameter("camera_device").as_string();
    loop_ = this->get_parameter("loop").as_bool();
    raw_format_ = this->get_parameter("raw_format").as_string();
    std::string transport_mode = this->get_parameter("intel_memory_buffer_type").as_string();
    std::string frame_topic = this->get_parameter("frame_topic").as_string();
    double req_fps = this->get_parameter("framerate").as_double();

    select_transport(transport_mode);

    publisher_ = this->create_publisher<IntelBufferDescriptor>(
      frame_topic, rclcpp::SensorDataQoS());

    if (source_mode_ == "camera") {
      init_dmabuf_source(req_fps);
    } else {
      init_simulated_source(req_fps);
    }

    RCLCPP_INFO(this->get_logger(),
      "USM publisher [%s]: %dx%d @ %.1f fps -> %s (waiting for subscribers...)",
      source_mode_.c_str(), width_, height_, fps_, frame_topic.c_str());

    wait_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(200),
      [this]() {
        auto count = publisher_->get_subscription_count();
        if (count >= 2) {
          wait_timer_->cancel();
          if (source_mode_ == "camera") {
            capture_running_ = true;
            capture_thread_ = std::thread(&UsmVideoPublisher::camera_capture_loop, this);
          } else {
            timer_ = this->create_wall_timer(
              std::chrono::microseconds(static_cast<int64_t>(1e6 / fps_)),
              std::bind(&UsmVideoPublisher::tick, this));
          }
          RCLCPP_INFO(this->get_logger(),
            "All subscribers ready (%zu) — starting frame production", count);
        } else {
          RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Waiting for subscribers... (current=%zu, need>=2)", count);
        }
      });
  }

  ~UsmVideoPublisher() override
  {
    capture_running_ = false;
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    cleanup_dmabuf_source();
  }

private:
  // ==========================================================================
  // Transport selection (simulated mode only)
  // ==========================================================================

  void select_transport(const std::string & transport_mode)
  {
    if (source_mode_ == "camera") {
      return;
    }

    if (transport_mode == "EXTERNAL_MEMMAP") {
      extmap_mode_ = true;
      RCLCPP_INFO(this->get_logger(),
        "simulated source: transport_mode=EXTERNAL_MEMMAP "
        "(producer-owned memory, buffer pool not used)");
      return;
    }

    // Map the node parameter onto the pool's INTEL_MEMORY_BUFFER_TYPE env var.
    if (transport_mode == "DMA_BUF") {
      setenv("INTEL_MEMORY_BUFFER_TYPE", "dmabuf", 1);
    } else if (transport_mode == "L0_USM") {
      setenv("INTEL_MEMORY_BUFFER_TYPE", "l0usm", 1);
    } else {
      throw std::runtime_error(
        "invalid transport_mode '" + transport_mode +
        "' (expected 'L0_USM', 'DMA_BUF' or 'EXTERNAL_MEMMAP')");
    }
    RCLCPP_INFO(this->get_logger(),
      "simulated source: transport_mode=%s", transport_mode.c_str());
  }

  // ==========================================================================
  // Simulated source: video file → allocate_buffer() + write_to_buffer()
  // ==========================================================================

  void init_simulated_source(double req_fps)
  {
    if (video_path_.empty()) {
      throw std::runtime_error("parameter 'video_path' is required for simulated mode");
    }
    cap_.open(video_path_);
    if (!cap_.isOpened()) {
      throw std::runtime_error("cannot open video: " + video_path_);
    }

    double native_fps = cap_.get(cv::CAP_PROP_FPS);
    fps_ = (req_fps > 0.0) ? req_fps : ((native_fps > 0.0) ? native_fps : 30.0);

    cv::Mat probe;
    if (!cap_.read(probe)) {
      throw std::runtime_error("cannot read first frame from video");
    }
    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);

    height_ = probe.rows;
    width_ = probe.cols;
    stride_ = width_ * 3;
    frame_bytes_ = static_cast<size_t>(height_) * static_cast<size_t>(stride_);

    if (extmap_mode_ && !extmap_ring_.create("/imb_extmap", frame_bytes_)) {
      throw std::runtime_error(
        "cannot set up the EXTERNAL_MEMMAP ring (extension unavailable or "
        "shared regions could not be created/mapped)");
    }
  }

  // ==========================================================================
  // DMA-BUF source: V4L2 camera → wrap_dmabuf() (zero copy)
  // ==========================================================================

  void init_dmabuf_source(double req_fps)
  {
    v4l2_fd_ = open(camera_device_.c_str(), O_RDWR | O_NONBLOCK);
    if (v4l2_fd_ < 0) {
      throw std::runtime_error(
        "cannot open camera device: " + camera_device_ +
        " (errno=" + std::to_string(errno) + ")");
    }

    struct v4l2_capability cap{};
    if (ioctl(v4l2_fd_, VIDIOC_QUERYCAP, &cap) < 0) {
      close(v4l2_fd_);
      v4l2_fd_ = -1;
      throw std::runtime_error("VIDIOC_QUERYCAP failed on " + camera_device_);
    }

    struct v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    ioctl(v4l2_fd_, VIDIOC_S_FMT, &fmt);
    ioctl(v4l2_fd_, VIDIOC_G_FMT, &fmt);

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_NV12) {
      char fourcc[5] = {
        static_cast<char>(fmt.fmt.pix.pixelformat & 0xff),
        static_cast<char>((fmt.fmt.pix.pixelformat >> 8) & 0xff),
        static_cast<char>((fmt.fmt.pix.pixelformat >> 16) & 0xff),
        static_cast<char>((fmt.fmt.pix.pixelformat >> 24) & 0xff), 0};
      cleanup_dmabuf_source();
      throw std::runtime_error(
        std::string("camera does not support NV12 capture (got '") + fourcc +
        "'); the GPU-preprocess pipeline requires NV12");
    }

    width_ = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    stride_ = fmt.fmt.pix.bytesperline;  // Y-plane stride (bytes per row)
    frame_bytes_ = fmt.fmt.pix.sizeimage;  // full NV12 buffer (~width*height*3/2)
    fps_ = (req_fps > 0.0) ? req_fps : 30.0;

    struct v4l2_requestbuffers reqbufs{};
    reqbufs.count = kNumV4L2Buffers;
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbufs.memory = V4L2_MEMORY_MMAP;
    if (ioctl(v4l2_fd_, VIDIOC_REQBUFS, &reqbufs) < 0) {
      close(v4l2_fd_);
      v4l2_fd_ = -1;
      throw std::runtime_error("VIDIOC_REQBUFS failed");
    }
    v4l2_buf_count_ = reqbufs.count;

    v4l2_dmabuf_fds_.resize(v4l2_buf_count_, -1);
    for (uint32_t i = 0; i < v4l2_buf_count_; ++i) {
      struct v4l2_exportbuffer expbuf{};
      expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      expbuf.index = i;
      expbuf.flags = O_RDWR;
      if (ioctl(v4l2_fd_, VIDIOC_EXPBUF, &expbuf) < 0) {
        cleanup_dmabuf_source();
        throw std::runtime_error("VIDIOC_EXPBUF failed for buffer " + std::to_string(i));
      }
      v4l2_dmabuf_fds_[i] = expbuf.fd;

      struct v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      ioctl(v4l2_fd_, VIDIOC_QBUF, &buf);
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(v4l2_fd_, VIDIOC_STREAMON, &type);
  }

  void cleanup_dmabuf_source()
  {
    if (v4l2_fd_ >= 0) {
      int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      ioctl(v4l2_fd_, VIDIOC_STREAMOFF, &type);
      for (int fd : v4l2_dmabuf_fds_) {
        if (fd >= 0) close(fd);
      }
      v4l2_dmabuf_fds_.clear();
      close(v4l2_fd_);
      v4l2_fd_ = -1;
    }
  }

  void camera_capture_loop()
  {
    while (capture_running_) {
      struct pollfd pfd{v4l2_fd_, POLLIN, 0};
      int r = poll(&pfd, 1, 200);  // ms; timeout just to re-check the flag
      if (r <= 0) continue;        // timeout or EINTR → loop and re-check
      if (pfd.revents & POLLIN) {
        tick_dmabuf();             // drains to newest + publishes
      }
    }
  }

  void tick()
  {
    tick_simulated();
  }

  void tick_simulated()
  {
    cv::Mat frame;
    if (!cap_.read(frame)) {
      if (loop_) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!cap_.read(frame)) {
          timer_->cancel();
          return;
        }
      } else {
        timer_->cancel();
        return;
      }
    }

    if (frame.rows != height_ || frame.cols != width_) {
      cv::resize(frame, frame, cv::Size(width_, height_));
    }

    cv::Mat bgr;
    if (raw_format_ == "YUYV") {
      // OpenCV has no BGR→YUYV encode; simulate lossy roundtrip via YUV I420
      cv::Mat yuv;
      cv::cvtColor(frame, yuv, cv::COLOR_BGR2YUV_I420);
      cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
    } else {
      cv::Mat nv12;
      cv::cvtColor(frame, nv12, cv::COLOR_BGR2YUV_I420);
      cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_I420);
    }

    if (extmap_mode_) {
      const size_t slot = extmap_ring_.advance();
      std::memcpy(extmap_ring_.data(slot), bgr.data, frame_bytes_);
      auto buffer = extmap_ring_.wrap(slot, frame_bytes_);
      publish_frame(buffer);
      return;
    }

    // Allocate from the pool (recycles released blocks) and write BGR
    auto buffer = intel_buffer_backend::allocate_buffer(frame_bytes_);
    intel_buffer_backend::write_to_buffer(buffer, bgr.data, frame_bytes_);

    publish_frame(buffer);
  }

  void tick_dmabuf()
  {
    if (v4l2_fd_ < 0) return;

    struct v4l2_buffer newest{};
    bool have_frame = false;
    while (true) {
      struct v4l2_buffer b{};
      b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      b.memory = V4L2_MEMORY_MMAP;
      if (ioctl(v4l2_fd_, VIDIOC_DQBUF, &b) < 0) break;  // EAGAIN: ring empty
      if (have_frame) ioctl(v4l2_fd_, VIDIOC_QBUF, &newest);  // requeue stale
      newest = b;
      have_frame = true;
    }
    if (!have_frame) return;
    struct v4l2_buffer & buf = newest;

    uint64_t capture_ns = 0;
    if ((buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
      capture_ns = static_cast<uint64_t>(buf.timestamp.tv_sec) * 1000000000ull +
                   static_cast<uint64_t>(buf.timestamp.tv_usec) * 1000ull;
    } else {
      RCLCPP_WARN_ONCE(this->get_logger(),
        "camera V4L2 timestamp is not CLOCK_MONOTONIC (flags=0x%x); "
        "capture->ready latency disabled", buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK);
    }

    int dmabuf_fd = v4l2_dmabuf_fds_[buf.index];
    auto buffer = intel_buffer_backend::wrap_dmabuf(
      dmabuf_fd, frame_bytes_, false);

    publish_frame_dmabuf(buffer, buf.index, capture_ns);

    ioctl(v4l2_fd_, VIDIOC_QBUF, &buf);
  }

  void publish_frame_dmabuf(rosidl::Buffer<uint8_t> & buffer, uint32_t buf_index,
                            uint64_t capture_ns)
  {
    IntelBufferDescriptor msg;
    set_stamp_ns(msg.header.stamp,
      capture_ns != 0 ? static_cast<int64_t>(capture_ns) : mono_now_ns());
    msg.header.frame_id = "camera";
    msg.width = width_;
    msg.height = height_;
    msg.stride = stride_;
    msg.encoding = "nv12";  // camera DMA-BUF is NV12; GPU/visualizer convert
    msg.frame_seq = frame_seq_;
    msg.transport_mode = IntelBufferDescriptor::TRANSPORT_DMA_BUF;

    auto & desc = msg.dmabuf_desc;
    desc.size = frame_bytes_;
    desc.local_ptr = 0;  // cross-process only for real camera
    desc.publisher_pid = static_cast<int32_t>(getpid());
    desc.dmabuf_pid = static_cast<int32_t>(getpid());
    desc.dmabuf_pool_block_id = buf_index;
    desc.dmabuf_block_size = frame_bytes_;

    auto * impl = dynamic_cast<intel_buffer_backend::IntelBufferImpl<uint8_t> *>(
      buffer.get_impl());
    if (impl && impl->get_block()) {
      auto * block = impl->get_block();
      block->block_id = buf_index;
      desc.dmabuf_socket_path =
        intel_buffer_backend::DmaBufIpc::register_block(block);
    }

    msg.buffer_ready_stamp_ns = this->now().nanoseconds();

    if (capture_ns != 0) {
      uint64_t ready_mono_ns = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch() / std::chrono::nanoseconds(1));
      if (ready_mono_ns > capture_ns) {
        capture_to_ready_stats_.add(
          static_cast<double>(ready_mono_ns - capture_ns) / 1e6);
      }
    }

    auto publish_start = std::chrono::steady_clock::now();
    publisher_->publish(msg);
    publish_stats_.add(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - publish_start).count());
    frame_seq_++;

    if (frame_seq_ % 1000 == 0) {
      capture_to_ready_stats_.report(
        this->get_logger(), "camera-capture->buffer-ready latency");
      publish_stats_.report(
        this->get_logger(), "publisher publish() serialize cost (descriptor)");
    }
    if (frame_seq_ % 100 == 0) {
      RCLCPP_INFO(this->get_logger(),
        "published %u frames [dmabuf] transport=DMA_BUF (zero-copy from camera)",
        frame_seq_);
    }
  }

  void publish_frame(rosidl::Buffer<uint8_t> & buffer)
  {
    IntelBufferDescriptor msg;
    set_stamp_ns(msg.header.stamp, mono_now_ns());
    msg.header.frame_id = "camera";
    msg.width = width_;
    msg.height = height_;
    msg.stride = stride_;
    msg.encoding = "bgr8";
    msg.frame_seq = frame_seq_;

    // Fill transport descriptor — zero copy IPC token
    fill_descriptor(buffer, msg);

    msg.buffer_ready_stamp_ns = this->now().nanoseconds();
    publisher_->publish(msg);
    frame_seq_++;

    if (frame_seq_ % 100 == 0) {
      const char * transport = "DMA_BUF";
      if (msg.transport_mode == IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM) {
        transport = "L0_USM";
      } else if (msg.transport_mode ==
        IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP)
      {
        transport = "EXTERNAL_MEMMAP";
      }
      RCLCPP_INFO(this->get_logger(),
        "published %u frames [%s] transport=%s",
        frame_seq_, source_mode_.c_str(), transport);
    }
  }

  void fill_descriptor(rosidl::Buffer<uint8_t> & buffer, IntelBufferDescriptor & msg)
  {
    auto * impl = dynamic_cast<intel_buffer_backend::IntelBufferImpl<uint8_t> *>(
      buffer.get_impl());
    if (!impl) return;

    auto * block = impl->get_block();
    if (!block) return;

    if (block->transport == intel_buffer_backend::TransportMode::EXTERNAL_MEMMAP) {
      msg.transport_mode = IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP;
      auto & desc = msg.extmap_desc;
      desc.size = impl->size();
      desc.shm_name = block->extmap_shm_name;
      desc.local_ptr = reinterpret_cast<uint64_t>(block->ptr);
      desc.publisher_pid = static_cast<int32_t>(getpid());
      return;
    }

    auto pool = intel_buffer_backend::IntelBufferImpl<uint8_t>::get_or_create_global_pool();

    if (block->transport == intel_buffer_backend::TransportMode::LEVEL_ZERO_USM) {
      msg.transport_mode = IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM;
      auto & desc = msg.l0_desc;
      desc.size = impl->size();
      desc.local_ptr = reinterpret_cast<uint64_t>(block->ptr);
      desc.publisher_pid = static_cast<int32_t>(getpid());
      desc.ze_device_ordinal = block->ze_device_ordinal;
      std::memcpy(desc.ze_device_uuid.data(), block->ze_device_uuid,
        sizeof(block->ze_device_uuid));
      desc.block_id = block->block_id;

      if (pool) {
        desc.pool_id = pool->pool_id();
        desc.pool_generation = pool->generation();
        desc.ipc_uid = pool->assign_uid(block);
      }

      // Delegate L0 IPC handle export to the library (no ze* calls in node code)
      if (block->ze_context && block->ptr) {
        uint64_t handle_size = 0;
        uint64_t context_id = 0;
        if (!intel_buffer_backend::L0UsmIpc::export_handle(
            block,
            desc.ze_ipc_handle.data(),
            handle_size,
            context_id))
        {
          RCLCPP_WARN_ONCE(this->get_logger(),
            "L0UsmIpc::export_handle failed");
        } else {
          desc.ze_ipc_handle_size = handle_size;
          desc.ze_context_id = context_id;
          if (pool) {
            desc.l0_socket_path = pool->register_block_for_ipc(block);
          }
        }
      }
    } else {
      msg.transport_mode = IntelBufferDescriptor::TRANSPORT_DMA_BUF;
      auto & desc = msg.dmabuf_desc;
      desc.size = impl->size();
      desc.local_ptr = reinterpret_cast<uint64_t>(block->ptr);
      desc.publisher_pid = static_cast<int32_t>(getpid());
      desc.dmabuf_pid = static_cast<int32_t>(getpid());
      desc.dmabuf_pool_block_id = block->block_id;
      desc.dmabuf_block_size = block->size;
      if (pool) {
        desc.pool_id = pool->pool_id();
        desc.pool_generation = pool->generation();
        desc.ipc_uid = pool->assign_uid(block);
        desc.dmabuf_socket_path = pool->register_block_for_ipc(block);
      }
    }
  }

  // Simulated source
  cv::VideoCapture cap_;

  bool extmap_mode_{false};
  intel_buffer_backend::ExternalMemMapRing extmap_ring_;

  // DMA-BUF source
  static constexpr uint32_t kNumV4L2Buffers = 4;
  int v4l2_fd_{-1};
  uint32_t v4l2_buf_count_{0};
  std::vector<int> v4l2_dmabuf_fds_;
  std::thread capture_thread_;              // camera: frame-arrival-driven publish
  std::atomic<bool> capture_running_{false};

  // Common
  rclcpp::Publisher<IntelBufferDescriptor>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr wait_timer_;

  std::string source_mode_;
  std::string video_path_;
  std::string camera_device_;
  std::string raw_format_;
  bool loop_;
  double fps_;
  int height_{0};
  int width_{0};
  int stride_{0};
  size_t frame_bytes_{0};
  uint32_t frame_seq_{0};
  LatencyStats capture_to_ready_stats_{60};  // camera-capture->buffer-ready, per 1000
  LatencyStats publish_stats_{60};
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(UsmVideoPublisher)
