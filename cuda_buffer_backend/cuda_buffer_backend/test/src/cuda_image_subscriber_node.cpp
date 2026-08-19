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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64.hpp"
#include "cuda_buffer/cuda_buffer_api.hpp"

class CudaImageSubscriber : public rclcpp::Node
{
public:
  explicit CudaImageSubscriber(const rclcpp::NodeOptions & options)
  : Node("cuda_image_subscriber", options),
    received_count_(0),
    validation_passed_(true)
  {
    this->declare_parameter<std::string>("expected_backend", "cuda");
    this->declare_parameter<std::string>("acceptable_buffer_backends", "any");
    this->declare_parameter<bool>("validate_sequence_pattern", false);
    expected_backend_ = this->get_parameter("expected_backend").as_string();
    acceptable_buffer_backends_ =
      this->get_parameter("acceptable_buffer_backends").as_string();
    validate_sequence_pattern_ =
      this->get_parameter("validate_sequence_pattern").as_bool();

    cudaStreamCreate(&stream_);

    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.acceptable_buffer_backends = acceptable_buffer_backends_;
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "test_cuda_image", 10,
      std::bind(&CudaImageSubscriber::image_callback, this, std::placeholders::_1),
      sub_opts);

    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>("subscriber_count", 10);
    validation_publisher_ = this->create_publisher<std_msgs::msg::Bool>("validation_result", 10);
    backend_validation_publisher_ =
      this->create_publisher<std_msgs::msg::Bool>("backend_validation", 10);
    content_validation_publisher_ =
      this->create_publisher<std_msgs::msg::Bool>("content_validation", 10);
    metadata_validation_publisher_ =
      this->create_publisher<std_msgs::msg::Bool>("metadata_validation", 10);
    cpu_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("test_cuda_image_cpu",
      10);
    latency_publisher_ = this->create_publisher<std_msgs::msg::Float64>("latency_ms", 10);

    RCLCPP_INFO(this->get_logger(), "CUDA image subscriber started (expecting backend: %s)",
                expected_backend_.c_str());
  }

  ~CudaImageSubscriber() override
  {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    double latency_ms = (this->now() - msg->header.stamp).seconds() * 1000.0;
    received_count_++;

    bool metadata_valid = true;
    bool backend_valid = true;
    bool content_valid = true;
    const size_t channels = validate_sequence_pattern_ ? 1u : 3u;
    const size_t expected_size = msg->width * msg->height * channels;
    const char * expected_encoding = validate_sequence_pattern_ ? "mono8" : "rgb8";

    if (msg->encoding != expected_encoding) {
      RCLCPP_ERROR(this->get_logger(), "Wrong encoding: %s",
                   msg->encoding.c_str());
      metadata_valid = false;
    }

    if (msg->data.size() != expected_size) {
      RCLCPP_ERROR(this->get_logger(), "Wrong data size: %zu (expected %zu)",
                   msg->data.size(), expected_size);
      metadata_valid = false;
    }

    uint8_t expected_value = 0;
    if (validate_sequence_pattern_) {
      const std::string expected_frame_id =
        "cuda_pool_dso_" + std::to_string(received_count_);
      expected_value = static_cast<uint8_t>((received_count_ * 37u + 11u) & 0xffu);
      if (msg->header.frame_id != expected_frame_id || msg->encoding != "mono8" ||
        msg->height != 16u || msg->width != 256u || msg->step != 256u ||
        msg->data.size() != 4096u)
      {
        RCLCPP_ERROR(
          this->get_logger(), "Wrong DSO regression metadata for image #%u", received_count_);
        metadata_valid = false;
      }
    }

    const std::string backend_type = msg->data.get_backend_type();
    if (backend_type != expected_backend_) {
      RCLCPP_ERROR(this->get_logger(),
                   "Wrong backend type: %s (expected: %s)",
                   backend_type.c_str(), expected_backend_.c_str());
      backend_valid = false;
    }

    std::vector<uint8_t> cpu_data;
    try {
      if (backend_type == "cuda") {
        const rosidl::Buffer<uint8_t> & data = msg->data;
        cuda_buffer_backend::ReadHandle read_handle =
          cuda_buffer_backend::from_input_buffer(data, stream_);
        cpu_data.resize(msg->data.size());
        cudaMemcpyAsync(
          cpu_data.data(), read_handle.get_ptr(),
          msg->data.size(), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
      } else {
        cpu_data = msg->data;
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Exception during data read: %s", e.what());
      content_valid = false;
    }

    if (cpu_data.empty()) {
      content_valid = false;
    } else if (metadata_valid && backend_valid) {
      const uint8_t expected_val = validate_sequence_pattern_ ? expected_value : cpu_data[0];
      const size_t first_index = validate_sequence_pattern_ ? 0u : 1u;
      for (size_t i = first_index; i < cpu_data.size(); ++i) {
        if (cpu_data[i] != expected_val) {
          RCLCPP_ERROR(this->get_logger(),
            "Content corruption at byte %zu: expected 0x%02x, got 0x%02x",
            i, expected_val, cpu_data[i]);
          content_valid = false;
          break;
        }
      }
    }

    bool msg_valid = metadata_valid && backend_valid && content_valid;

    if (!cpu_data.empty()) {
      sensor_msgs::msg::Image cpu_msg;
      cpu_msg.header = msg->header;
      cpu_msg.height = msg->height;
      cpu_msg.width = msg->width;
      cpu_msg.encoding = msg->encoding;
      cpu_msg.is_bigendian = msg->is_bigendian;
      cpu_msg.step = msg->step;
      cpu_msg.data.resize(cpu_data.size());
      std::memcpy(cpu_msg.data.data(), cpu_data.data(), cpu_data.size());
      cpu_image_publisher_->publish(cpu_msg);
    }

    std_msgs::msg::Float64 latency_msg;
    latency_msg.data = latency_ms;
    latency_publisher_->publish(latency_msg);

    validation_passed_ = validation_passed_ && msg_valid;

    std_msgs::msg::UInt32 count_msg;
    count_msg.data = received_count_;
    count_publisher_->publish(count_msg);

    std_msgs::msg::Bool validation_msg;
    validation_msg.data = validation_passed_;
    validation_publisher_->publish(validation_msg);

    std_msgs::msg::Bool backend_msg;
    backend_msg.data = backend_valid;
    backend_validation_publisher_->publish(backend_msg);

    std_msgs::msg::Bool content_msg;
    content_msg.data = content_valid;
    content_validation_publisher_->publish(content_msg);

    std_msgs::msg::Bool metadata_msg;
    metadata_msg.data = metadata_valid;
    metadata_validation_publisher_->publish(metadata_msg);

    if (!msg_valid) {
      RCLCPP_ERROR(this->get_logger(),
                   "Received INVALID image #%u", received_count_);
    } else {
      RCLCPP_INFO(this->get_logger(),
                  "Received image #%u (%ux%u, %zu bytes, backend: %s, latency: %.3f ms)",
                  received_count_, msg->width, msg->height,
                  msg->data.size(), backend_type.c_str(), latency_ms);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr validation_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr backend_validation_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr content_validation_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr metadata_validation_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cpu_image_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr latency_publisher_;
  cudaStream_t stream_{nullptr};
  uint32_t received_count_;
  bool validation_passed_;
  std::string expected_backend_;
  std::string acceptable_buffer_backends_;
  bool validate_sequence_pattern_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(CudaImageSubscriber)
