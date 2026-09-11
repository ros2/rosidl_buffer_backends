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
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

class OrtTensorPublisher : public rclcpp::Node
{
public:
  explicit OrtTensorPublisher(const rclcpp::NodeOptions & options)
  : Node("onnxruntime_tensor_publisher", options)
  {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("Failed to create publisher CUDA stream");
    }
    publisher_ = create_publisher<onnxruntime_conversions::TensorMsg>(
      "test_onnxruntime_cuda_tensor", 10);
    timer_ = create_wall_timer(
      std::chrono::milliseconds(100), std::bind(&OrtTensorPublisher::publish, this));
  }

  ~OrtTensorPublisher() override
  {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

private:
  void publish()
  {
    auto owner = onnxruntime_conversions::allocate_tensor_msg(
      {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
    const std::vector<float> values(6, static_cast<float>(count_ + 1));
    {
      auto view = onnxruntime_conversions::from_output_tensor_msg(*owner, stream_);
      if (cudaMemcpyAsync(
          view.value().GetTensorMutableRawData(), values.data(),
          values.size() * sizeof(float), cudaMemcpyHostToDevice, stream_) != cudaSuccess)
      {
        throw std::runtime_error("Failed to populate CUDA tensor");
      }
    }
    auto message =
      std::make_unique<onnxruntime_conversions::TensorMsg>(std::move(*owner));
    publisher_->publish(std::move(message));
    ++count_;
  }

  cudaStream_t stream_{nullptr};
  rclcpp::Publisher<onnxruntime_conversions::TensorMsg>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  uint32_t count_{0};
};

RCLCPP_COMPONENTS_REGISTER_NODE(OrtTensorPublisher)
