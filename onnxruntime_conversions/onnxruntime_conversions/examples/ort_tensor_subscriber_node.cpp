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
#include <cstddef>
#include <memory>
#include <string>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "example_models.hpp"

class OrtTensorSubscriber : public rclcpp::Node
{
public:
  explicit OrtTensorSubscriber(const rclcpp::NodeOptions & options)
  : Node("onnxruntime_tensor_subscriber", options),
    env_(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_tensor_subscriber"),
    stream_(onnxruntime_conversions::create_stream(
        env_, declare_parameter<std::string>("backend", "")))
  {
    Ort::SessionOptions session_options;
    // Configure inference to use the selected backend and stream.
    onnxruntime_conversions::configure_session_options(session_options, stream_);
    session_ = Ort::Session(
      env_, example_models::identity_model({2, 3}), session_options);

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.acceptable_buffer_backends = "any";
    subscription_ = create_subscription<onnxruntime_conversions::TensorMsg>(
      "test_onnxruntime_tensor", 10,
      [this](onnxruntime_conversions::TensorMsg::ConstSharedPtr received) {
        receive_tensor(*received);
      }, subscription_options);
  }

private:
  void receive_tensor(const onnxruntime_conversions::TensorMsg & received)
  {
    // Borrow the message storage; the message stays alive throughout this callback.
    auto input_tensor = onnxruntime_conversions::from_input_tensor_msg(received, stream_);
    Ort::IoBinding binding(session_);
    binding.BindInput("input", input_tensor.value());
    // Return the inference output on CPU so it can be read below.
    binding.BindOutput(
      "output", Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    session_.Run(Ort::RunOptions{}, binding);
    binding.SynchronizeOutputs();

    // Replace the identity model and this summary with your own inference and processing.
    const auto outputs = binding.GetOutputValues();
    const auto * values = outputs.front().GetTensorData<float>();
    const auto size = outputs.front().GetTensorTypeAndShapeInfo().GetElementCount();
    const auto [minimum, maximum] = std::minmax_element(values, values + size);
    RCLCPP_INFO(
      get_logger(), "Received tensor (backend=%s, min=%g, max=%g, count=%zu)",
      received.data.get_backend_type().c_str(),
      static_cast<double>(*minimum), static_cast<double>(*maximum), ++received_count_);
  }

  Ort::Env env_;
  onnxruntime_conversions::Stream stream_;
  Ort::Session session_{nullptr};
  rclcpp::Subscription<onnxruntime_conversions::TensorMsg>::SharedPtr subscription_;
  std::size_t received_count_{0};
};

RCLCPP_COMPONENTS_REGISTER_NODE(OrtTensorSubscriber)
