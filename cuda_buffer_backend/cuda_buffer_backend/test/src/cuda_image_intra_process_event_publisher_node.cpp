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

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/bool.hpp"

class CudaImageIntraProcessEventPublisher : public rclcpp::Node
{
public:
  explicit CudaImageIntraProcessEventPublisher(const rclcpp::NodeOptions & options)
  : Node("cuda_image_intra_process_event_publisher", options)
  {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(
      &completion_event_, cudaEventDisableTiming));
    publisher_ = create_publisher<sensor_msgs::msg::Image>("test_cuda_image_intra_event", 10);
    nonblocking_publisher_ =
      create_publisher<std_msgs::msg::Bool>("intra_event_publish_nonblocking", 10);
    timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&CudaImageIntraProcessEventPublisher::publish_next, this));
  }

  ~CudaImageIntraProcessEventPublisher() override
  {
    if (stream_ != nullptr) {
      (void)cudaStreamSynchronize(stream_);
      (void)cudaEventDestroy(completion_event_);
      completion_event_ = nullptr;
      (void)cudaStreamDestroy(stream_);
    }
  }

private:
  static constexpr std::size_t kMessages = 5;
  static constexpr std::size_t kPayloadSize = 4096;
  static constexpr auto kStreamDelay = std::chrono::milliseconds(250);

  static std::uint8_t pattern_for(std::size_t sequence)
  {
    return static_cast<std::uint8_t>((sequence * 37u + 11u) & 0xffu);
  }

  static void stream_delay(void * user_data)
  {
    static_cast<void>(user_data);
    std::this_thread::sleep_for(kStreamDelay);
  }

  void publish_next()
  {
    if (sequence_ >= kMessages) {
      timer_->cancel();
      return;
    }
    if (publisher_->get_subscription_count() == 0u ||
      nonblocking_publisher_->get_subscription_count() == 0u)
    {
      return;
    }

    const std::size_t sequence = sequence_ + 1u;
    const std::uint8_t pattern = pattern_for(sequence);

    sensor_msgs::msg::Image msg;
    msg.header.stamp = now();
    msg.header.frame_id = "cuda_pool_dso_" + std::to_string(sequence);
    msg.height = 16;
    msg.width = 256;
    msg.encoding = "mono8";
    msg.is_bigendian = 0;
    msg.step = 256;
    msg.data = cuda_buffer_backend::allocate_buffer(kPayloadSize);

    {
      CUDA_CHECK(cudaLaunchHostFunc(stream_, stream_delay, nullptr));
      auto write_handle = cuda_buffer_backend::from_output_buffer(msg.data, stream_);
      CUDA_CHECK(cudaMemsetAsync(write_handle.get_ptr(), pattern, kPayloadSize, stream_));
      CUDA_CHECK(cudaEventRecord(completion_event_, stream_));
    }

    publisher_->publish(msg);
    cudaError_t completion_status = cudaEventQuery(completion_event_);
    const bool publish_was_nonblocking = completion_status == cudaErrorNotReady;
    if (publish_was_nonblocking) {
      (void)cudaGetLastError();
    } else {
      CUDA_CHECK(completion_status);
    }

    std_msgs::msg::Bool nonblocking_msg;
    nonblocking_msg.data = publish_was_nonblocking;
    nonblocking_publisher_->publish(nonblocking_msg);
    sequence_ = sequence;
    RCLCPP_INFO(
      get_logger(), "INTRA_EVENT_PUBLISH_RESULT message=%zu backend=%s pattern=%u nonblocking=%s",
      sequence_, msg.data.get_backend_type().c_str(), static_cast<unsigned>(pattern),
      publish_was_nonblocking ? "true" : "false");
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr nonblocking_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  cudaStream_t stream_{nullptr};
  cudaEvent_t completion_event_{nullptr};
  std::size_t sequence_{0};
};

RCLCPP_COMPONENTS_REGISTER_NODE(CudaImageIntraProcessEventPublisher)
