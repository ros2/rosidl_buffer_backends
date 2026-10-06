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

#include <cstddef>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "torch_conversions/torch_conversions.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

class TorchTensorSubscriber : public rclcpp::Node
{
public:
  explicit TorchTensorSubscriber(const rclcpp::NodeOptions & options)
  : Node("torch_tensor_subscriber", options)
  {
    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.acceptable_buffer_backends = "any";
    subscription_ = create_subscription<tensor_msgs::msg::ExperimentalTensor>(
      "test_torch_tensor", 10,
      [this](tensor_msgs::msg::ExperimentalTensor::ConstSharedPtr received) {
        receive_tensor(*received);
      }, subscription_options);
  }

private:
  void receive_tensor(const tensor_msgs::msg::ExperimentalTensor & received)
  {
    // Select a stream for this callback (a no-op on CPU).
    auto guard = torch_conversions::set_stream();

    // Borrow the message storage; the message stays alive throughout this callback.
    auto input_tensor = torch_conversions::from_input_tensor_msg(received, /*clone=*/false);

    // Replace these reductions with your own processing of the read-only tensor.
    const auto minimum = input_tensor.min().item<double>();
    const auto maximum = input_tensor.max().item<double>();
    RCLCPP_INFO(
      get_logger(), "Received tensor (backend=%s, min=%g, max=%g, count=%zu)",
      received.data.get_backend_type().c_str(), minimum, maximum, ++received_count_);
  }

  rclcpp::Subscription<tensor_msgs::msg::ExperimentalTensor>::SharedPtr subscription_;
  std::size_t received_count_{0};
};

RCLCPP_COMPONENTS_REGISTER_NODE(TorchTensorSubscriber)
