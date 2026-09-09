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

#include <ATen/dlpack.h>
#include <torch/torch.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

class TorchTensorSubscriber : public rclcpp::Node
{
public:
  explicit TorchTensorSubscriber(const rclcpp::NodeOptions & options)
  : Node("torch_tensor_subscriber", options),
    received_count_(0),
    validation_passed_(true)
  {
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.acceptable_buffer_backends = "any";
    subscription_ = this->create_subscription<tensor_msgs::msg::ExperimentalTensor>(
      "test_torch_tensor", 10,
      std::bind(&TorchTensorSubscriber::tensor_callback, this, std::placeholders::_1),
      sub_opts);

    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>(
      "subscriber_count", 10);
    validation_publisher_ = this->create_publisher<std_msgs::msg::Bool>(
      "validation_result", 10);

    RCLCPP_INFO(this->get_logger(), "Torch tensor subscriber started");
  }

private:
  void tensor_callback(const tensor_msgs::msg::ExperimentalTensor::SharedPtr msg)
  {
    received_count_++;
    bool msg_valid = true;

    if (msg->shape.size() != 3u) {
      RCLCPP_ERROR(
        this->get_logger(), "Expected rank-3 tensor, got rank=%zu",
        msg->shape.size());
      msg_valid = false;
    }

    const bool is_uint8 =
      msg->dtype_code == static_cast<uint8_t>(kDLUInt) &&
      msg->dtype_bits == 8u &&
      msg->dtype_lanes == 1u;
    if (!is_uint8) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Expected dtype uint8 (code=1,bits=8,lanes=1), got code=%u bits=%u lanes=%u",
        static_cast<unsigned>(msg->dtype_code),
        static_cast<unsigned>(msg->dtype_bits),
        static_cast<unsigned>(msg->dtype_lanes));
      msg_valid = false;
    }

    const std::string backend_type = msg->data.get_backend_type();

    if (msg_valid && !msg->data.empty()) {
      torch_conversions::StreamGuard guard = torch_conversions::set_stream();
      at::Tensor tensor = torch_conversions::from_input_tensor_msg(
        *msg,
        /*clone=*/true);

      at::Tensor cpu_tensor = tensor.contiguous().cpu();
      at::Tensor flat = cpu_tensor.view({-1});
      if (flat.numel() > 0) {
        const uint8_t * ptr = flat.data_ptr<uint8_t>();
        uint8_t expected_val = ptr[0];
        for (int64_t i = 1; i < flat.numel(); ++i) {
          if (ptr[i] != expected_val) {
            RCLCPP_ERROR(
              this->get_logger(),
              "Content corruption at byte %ld: expected 0x%02x got 0x%02x",
              i, expected_val, ptr[i]);
            msg_valid = false;
            break;
          }
        }
      }
    }

    validation_passed_ = validation_passed_ && msg_valid;

    std_msgs::msg::UInt32 count_msg;
    count_msg.data = received_count_;
    count_publisher_->publish(count_msg);

    std_msgs::msg::Bool validation_msg;
    validation_msg.data = validation_passed_;
    validation_publisher_->publish(validation_msg);

    if (!msg_valid) {
      RCLCPP_ERROR(
        this->get_logger(), "Received INVALID tensor #%u", received_count_);
    } else {
      RCLCPP_INFO(
        this->get_logger(),
        "Received tensor #%u (shape=[%ld,%ld,%ld], %zu bytes, backend=%s)",
        received_count_,
        msg->shape.size() > 0 ? msg->shape[0] : 0,
        msg->shape.size() > 1 ? msg->shape[1] : 0,
        msg->shape.size() > 2 ? msg->shape[2] : 0,
        msg->data.size(), backend_type.c_str());
    }
  }

  rclcpp::Subscription<tensor_msgs::msg::ExperimentalTensor>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr validation_publisher_;
  uint32_t received_count_;
  bool validation_passed_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(TorchTensorSubscriber)
