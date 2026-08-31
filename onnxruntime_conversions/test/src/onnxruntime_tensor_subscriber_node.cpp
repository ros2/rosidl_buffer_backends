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
#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace
{

const uint8_t identity_model[] = {
  8, 10, 58, 88, 10, 25, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111,
  117, 116, 112, 117, 116, 34, 8, 73, 100, 101, 110, 116, 105, 116, 121,
  18, 8, 105, 100, 101, 110, 116, 105, 116, 121, 90, 23, 10, 5, 105,
  110, 112, 117, 116, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 2, 10,
  2, 8, 3, 98, 24, 10, 6, 111, 117, 116, 112, 117, 116, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 3, 66, 4, 10, 0, 16, 18};

}  // namespace

class OnnxRuntimeTensorSubscriber : public rclcpp::Node
{
public:
  explicit OnnxRuntimeTensorSubscriber(const rclcpp::NodeOptions & options)
  : Node("onnxruntime_tensor_subscriber", options),
    env_(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_tensor_subscriber"),
    session_(nullptr),
    memory_info_("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)
  {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("Failed to create subscriber CUDA stream");
    }
    Ort::CUDAProviderOptions cuda_options;
    cuda_options.UpdateWithValue("user_compute_stream", stream_);
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    session_ = Ort::Session(env_, identity_model, sizeof(identity_model), session_options);

    rclcpp::SubscriptionOptions subscription_options;
    subscription_options.acceptable_buffer_backends = "any";
    subscription_ = this->create_subscription<tensor_msgs::msg::ExperimentalTensor>(
      "test_onnxruntime_tensor", 10,
      std::bind(&OnnxRuntimeTensorSubscriber::receive, this, std::placeholders::_1),
      subscription_options);
    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>(
      "subscriber_count", 10);
    validation_publisher_ = this->create_publisher<std_msgs::msg::Bool>(
      "validation_result", 10);
  }

  ~OnnxRuntimeTensorSubscriber() override
  {
    session_ = Ort::Session(nullptr);
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

private:
  void receive(const tensor_msgs::msg::ExperimentalTensor::SharedPtr msg)
  {
    bool valid = true;
    try {
      if (msg->data.get_backend_type() != "cuda") {
        throw std::runtime_error("Received tensor is not CUDA-backed");
      }
      auto output = std::shared_ptr<onnxruntime_conversions::TensorMsg>(
        onnxruntime_conversions::allocate_tensor_msg(
          {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"));
      {
        Ort::IoBinding binding(session_);
        std::shared_ptr<const onnxruntime_conversions::TensorMsg> input = msg;
        auto input_view = onnxruntime_conversions::from_input_tensor_msg(
          input, memory_info_, stream_);
        auto output_view = onnxruntime_conversions::from_output_tensor_msg(
          output, memory_info_, stream_);
        binding.BindInput("input", input_view.value());
        binding.BindOutput("output", output_view.value());
        Ort::RunOptions run_options;
        session_.Run(run_options, binding);
      }

      std::vector<float> input_values(6);
      std::vector<float> output_values(6);
      {
        std::shared_ptr<const onnxruntime_conversions::TensorMsg> input = msg;
        std::shared_ptr<const onnxruntime_conversions::TensorMsg> result = output;
        auto input_view = onnxruntime_conversions::from_input_tensor_msg(
          input, memory_info_, stream_);
        auto output_view = onnxruntime_conversions::from_input_tensor_msg(
          result, memory_info_, stream_);
        if (cudaMemcpyAsync(
            input_values.data(), input_view.value().GetTensorRawData(),
            input_values.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
          cudaMemcpyAsync(
            output_values.data(), output_view.value().GetTensorRawData(),
            output_values.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess)
        {
          throw std::runtime_error("Failed to copy inference result");
        }
      }
      if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize inference stream");
      }
      valid = input_values == output_values;
    } catch (const std::exception & error) {
      RCLCPP_ERROR(this->get_logger(), "%s", error.what());
      valid = false;
    }

    validation_passed_ = validation_passed_ && valid;
    std_msgs::msg::UInt32 count_msg;
    count_msg.data = ++received_count_;
    count_publisher_->publish(count_msg);
    std_msgs::msg::Bool validation_msg;
    validation_msg.data = validation_passed_;
    validation_publisher_->publish(validation_msg);
  }

  Ort::Env env_;
  Ort::Session session_;
  Ort::MemoryInfo memory_info_;
  cudaStream_t stream_{nullptr};
  rclcpp::Subscription<tensor_msgs::msg::ExperimentalTensor>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr validation_publisher_;
  uint32_t received_count_{0};
  bool validation_passed_{true};
};

RCLCPP_COMPONENTS_REGISTER_NODE(OnnxRuntimeTensorSubscriber)
