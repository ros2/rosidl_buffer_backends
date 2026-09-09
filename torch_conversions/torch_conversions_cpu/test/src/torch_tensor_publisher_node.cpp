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

#include <torch/torch.h>
#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

class TorchTensorPublisher : public rclcpp::Node
{
public:
  explicit TorchTensorPublisher(const rclcpp::NodeOptions & options)
  : Node("torch_tensor_publisher", options), count_(0)
  {
    this->declare_parameter<int>("max_publish_count", 0);
    this->declare_parameter<int>("publish_rate_ms", 200);
    this->declare_parameter<int>("tensor_width", 8);
    this->declare_parameter<int>("tensor_height", 8);

    max_publish_count_ = this->get_parameter("max_publish_count").as_int();
    int publish_rate_ms = this->get_parameter("publish_rate_ms").as_int();
    tensor_width_ = this->get_parameter("tensor_width").as_int();
    tensor_height_ = this->get_parameter("tensor_height").as_int();

    publisher_ = this->create_publisher<tensor_msgs::msg::ExperimentalTensor>(
      "test_torch_tensor", 10);
    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>(
      "publisher_count", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(publish_rate_ms),
      std::bind(&TorchTensorPublisher::timer_callback, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Torch tensor publisher started (max=%d, rate=%dms)",
      max_publish_count_, publish_rate_ms);
  }

private:
  void timer_callback()
  {
    if (max_publish_count_ > 0 && count_ >= static_cast<size_t>(max_publish_count_)) {
      timer_->cancel();
      return;
    }

    torch_conversions::StreamGuard guard = torch_conversions::set_stream();

    auto msg = torch_conversions::allocate_tensor_msg(
      {tensor_height_, tensor_width_, 3}, torch::kByte);

    {
      at::Tensor output = torch_conversions::from_output_tensor_msg(*msg);
      output.fill_(static_cast<int>(count_ % 256));
    }

    const std::string backend_type = msg->data.get_backend_type();
    publisher_->publish(std::move(msg));

    std_msgs::msg::UInt32 count_msg;
    count_msg.data = ++count_;
    count_publisher_->publish(count_msg);

    if (count_ % 10 == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "Published %zu torch tensors (data backend: %s)",
        count_, backend_type.c_str());
    }
  }

  rclcpp::Publisher<tensor_msgs::msg::ExperimentalTensor>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  size_t count_;
  int max_publish_count_;
  int tensor_width_;
  int tensor_height_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(TorchTensorPublisher)
