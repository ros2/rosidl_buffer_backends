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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "test_models.hpp"

class OrtTensorSubscriber : public rclcpp::Node
{
public:
  explicit OrtTensorSubscriber(const rclcpp::NodeOptions & options)
  : Node("onnxruntime_tensor_subscriber", options),
    env_(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_tensor_subscriber"),
    stream_(onnxruntime_conversions::create_stream(env_)),
    session_(nullptr)
  {
    Ort::SessionOptions session_options;
    onnxruntime_conversions::configure_session_options(
      session_options, stream_);
    session_ = Ort::Session(
      env_, test_models::identity_model({2, 3}), session_options);

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.acceptable_buffer_backends = "any";
    subscription_ = create_subscription<onnxruntime_conversions::TensorMsg>(
      "test_onnxruntime_tensor", 10,
      std::bind(&OrtTensorSubscriber::receive, this, std::placeholders::_1),
      subscription_options);
    result_publisher_ =
      create_publisher<std_msgs::msg::UInt32>("validation_count", 10);
  }

private:
  void receive(const onnxruntime_conversions::TensorMsg::SharedPtr message)
  {
    bool valid = true;
    try {
      if (message->data.get_backend_type() != stream_.backend()) {
        throw std::runtime_error("Received tensor uses an unexpected backend");
      }
      auto output = onnxruntime_conversions::allocate_tensor_msg(
        {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, stream_);
      {
        Ort::IoBinding binding(session_);
        auto input_view =
          onnxruntime_conversions::from_input_tensor_msg(*message, stream_);
        auto output_view =
          onnxruntime_conversions::from_output_tensor_msg(*output, stream_);
        binding.BindInput("input", input_view.value());
        binding.BindOutput("output", output_view.value());
        Ort::RunOptions run_options;
        session_.Run(run_options, binding);
      }

      auto input_host = onnxruntime_conversions::allocate_tensor_msg(
        {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
      auto output_host = onnxruntime_conversions::allocate_tensor_msg(
        {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
      {
        auto input_view =
          onnxruntime_conversions::from_input_tensor_msg(*message, stream_);
        auto output_view =
          onnxruntime_conversions::from_input_tensor_msg(*output, stream_);
        onnxruntime_conversions::to_tensor_msg(*input_host, input_view.value(), stream_);
        onnxruntime_conversions::to_tensor_msg(*output_host, output_view.value(), stream_);
      }
      valid = std::equal(
        input_host->data.data(), input_host->data.data() + input_host->data.size(),
        output_host->data.data());
    } catch (const std::exception & error) {
      RCLCPP_ERROR(get_logger(), "%s", error.what());
      valid = false;
    }

    validation_passed_ = validation_passed_ && valid;
    std_msgs::msg::UInt32 result;
    ++received_count_;
    result.data = validation_passed_ ? received_count_ : 0;
    result_publisher_->publish(result);
  }

  Ort::Env env_;
  onnxruntime_conversions::Stream stream_;
  Ort::Session session_;
  rclcpp::Subscription<onnxruntime_conversions::TensorMsg>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr result_publisher_;
  uint32_t received_count_{0};
  bool validation_passed_{true};
};

RCLCPP_COMPONENTS_REGISTER_NODE(OrtTensorSubscriber)
