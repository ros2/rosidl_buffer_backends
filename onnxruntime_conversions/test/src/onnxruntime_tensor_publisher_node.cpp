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

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "std_msgs/msg/u_int32.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

class OnnxRuntimeTensorPublisher : public rclcpp::Node
{
public:
  explicit OnnxRuntimeTensorPublisher(const rclcpp::NodeOptions & options)
  : Node("onnxruntime_tensor_publisher", options),
    memory_info_("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)
  {
    this->declare_parameter<int>("publish_rate_ms", 100);
    const auto publish_rate = this->get_parameter("publish_rate_ms").as_int();
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("Failed to create publisher CUDA stream");
    }

    publisher_ = this->create_publisher<tensor_msgs::msg::ExperimentalTensor>(
      "test_onnxruntime_tensor", 10);
    count_publisher_ = this->create_publisher<std_msgs::msg::UInt32>(
      "publisher_count", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(publish_rate),
      std::bind(&OnnxRuntimeTensorPublisher::publish, this));
  }

  ~OnnxRuntimeTensorPublisher() override
  {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

private:
  void publish()
  {
    auto msg = onnxruntime_conversions::allocate_tensor_msg(
      {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
    const std::vector<float> values(6, static_cast<float>(count_ + 1));
    {
      auto owner = std::shared_ptr<onnxruntime_conversions::TensorMsg>(std::move(msg));
      auto view = onnxruntime_conversions::from_output_tensor_msg(
        owner, memory_info_, stream_);
      const auto result = cudaMemcpyAsync(
        view.value().GetTensorMutableRawData(), values.data(),
        values.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
      if (result != cudaSuccess) {
        throw std::runtime_error("Failed to populate CUDA tensor");
      }
      msg = std::make_unique<onnxruntime_conversions::TensorMsg>(std::move(*owner));
    }

    publisher_->publish(std::move(msg));
    std_msgs::msg::UInt32 count_msg;
    count_msg.data = ++count_;
    count_publisher_->publish(count_msg);
  }

  Ort::MemoryInfo memory_info_;
  cudaStream_t stream_{nullptr};
  rclcpp::Publisher<tensor_msgs::msg::ExperimentalTensor>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr count_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  uint32_t count_{0};
};

RCLCPP_COMPONENTS_REGISTER_NODE(OnnxRuntimeTensorPublisher)
