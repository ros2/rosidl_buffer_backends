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

#ifndef ONNXRUNTIME_CONVERSIONS__ONNXRUNTIME_CONVERSIONS_HPP_
#define ONNXRUNTIME_CONVERSIONS__ONNXRUNTIME_CONVERSIONS_HPP_

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace onnxruntime_conversions
{

using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

class OrtTensorView
{
public:
  OrtTensorView(OrtTensorView &&) noexcept;
  OrtTensorView & operator=(OrtTensorView &&) noexcept;
  ~OrtTensorView();

  OrtTensorView(const OrtTensorView &) = delete;
  OrtTensorView & operator=(const OrtTensorView &) = delete;

  Ort::Value & value();
  const Ort::Value & value() const;

private:
  struct Impl;

  explicit OrtTensorView(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend OrtTensorView from_input_tensor_msg(
    std::shared_ptr<const TensorMsg>,
    const Ort::MemoryInfo &,
    void *);
  friend OrtTensorView from_output_tensor_msg(
    std::shared_ptr<TensorMsg>,
    const Ort::MemoryInfo &,
    void *);
};

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & backend = "cpu");

OrtTensorView from_input_tensor_msg(
  std::shared_ptr<const TensorMsg> msg,
  const Ort::MemoryInfo & memory_info,
  void * execution_stream = nullptr);

OrtTensorView from_output_tensor_msg(
  std::shared_ptr<TensorMsg> msg,
  const Ort::MemoryInfo & memory_info,
  void * execution_stream = nullptr);

void to_tensor_msg(TensorMsg & msg, const Ort::Value & value);

std::unique_ptr<TensorMsg> to_tensor_msg(const Ort::Value & value);

}  // namespace onnxruntime_conversions

#include "onnxruntime_conversions/detail/onnxruntime_conversions_impl.hpp"

#endif  // ONNXRUNTIME_CONVERSIONS__ONNXRUNTIME_CONVERSIONS_HPP_
