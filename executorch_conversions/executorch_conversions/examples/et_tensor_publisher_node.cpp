// Copyright 2026 Open Source Robotics Foundation, Inc.
// Copyright 2026 Meta Platforms, Inc. and affiliates.
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
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "executorch_conversions/executorch_conversions.hpp"

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;
using executorch::extension::make_tensor_ptr;

class ExecuTorchTensorPublisher : public rclcpp::Node
{
public:
  explicit ExecuTorchTensorPublisher(const rclcpp::NodeOptions & options)
  : Node("et_tensor_publisher", options), count_(0)
  {
    const auto device = this->declare_parameter<std::string>("device", "");
    if (device == "cpu") {
      device_ = Device(DeviceType::CPU);
    } else if (device == "cuda") {
      device_ = Device(DeviceType::CUDA);
    } else if (!device.empty()) {
      throw std::invalid_argument("device must be cpu, cuda or empty");
    }
    this->declare_parameter<int>("max_publish_count", 0);
    this->declare_parameter<int>("publish_rate_ms", 200);
    this->declare_parameter<int>("tensor_width", 8);
    this->declare_parameter<int>("tensor_height", 8);

    max_publish_count_ = this->get_parameter("max_publish_count").as_int();
    int publish_rate_ms = this->get_parameter("publish_rate_ms").as_int();
    tensor_width_ = this->get_parameter("tensor_width").as_int();
    tensor_height_ = this->get_parameter("tensor_height").as_int();
    // The frame vector below is sized with int arithmetic.
    const auto frame_bytes = static_cast<int64_t>(tensor_width_) * tensor_height_ * 3;
    if (tensor_width_ <= 0 || tensor_height_ <= 0 ||
      frame_bytes > std::numeric_limits<int32_t>::max())
    {
      throw std::invalid_argument(
              "tensor_width and tensor_height must be positive, and a frame under 2 GiB");
    }

    publisher_ = this->create_publisher<tensor_msgs::msg::ExperimentalTensor>(
      "test_et_tensor", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(publish_rate_ms),
      std::bind(&ExecuTorchTensorPublisher::timer_callback, this));

    RCLCPP_INFO(
      this->get_logger(),
      "ExecuTorch tensor publisher started (max=%d, rate=%dms)",
      max_publish_count_, publish_rate_ms);
  }

private:
  void timer_callback()
  {
    if (max_publish_count_ > 0 && count_ >= static_cast<size_t>(max_publish_count_)) {
      timer_->cancel();
      return;
    }

    auto msg = executorch_conversions::allocate_tensor_msg(
      {tensor_height_, tensor_width_, 3}, ScalarType::Byte, device_);

    // ExecuTorch has no eager operators, so the frame is filled on the host and
    // copied into the message storage, which is device memory on CUDA.
    std::vector<uint8_t> frame(
      static_cast<size_t>(tensor_height_ * tensor_width_ * 3),
      static_cast<uint8_t>(count_ % 256));
    auto host = make_tensor_ptr(
      {tensor_height_, tensor_width_, 3}, std::move(frame), {}, {}, ScalarType::Byte);
    executorch_conversions::to_tensor_msg(*msg, *host);

    const std::string backend_type = msg->data.get_backend_type();
    publisher_->publish(std::move(msg));

    ++count_;

    if (count_ % 10 == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "Published %zu ExecuTorch tensors (data backend: %s)",
        count_, backend_type.c_str());
    }
  }

  rclcpp::Publisher<tensor_msgs::msg::ExperimentalTensor>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::optional<Device> device_;
  size_t count_;
  int max_publish_count_;
  int tensor_width_;
  int tensor_height_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(ExecuTorchTensorPublisher)
