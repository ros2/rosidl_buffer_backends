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

#include "onnxruntime_conversions/conversion_backend.hpp"
#include "onnxruntime_conversions/visibility_control.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace onnxruntime_conversions
{

using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

class OrtTensorView
{
public:
  ONNXRUNTIME_CONVERSIONS_PUBLIC
  OrtTensorView(OrtTensorView &&) noexcept;
  ONNXRUNTIME_CONVERSIONS_PUBLIC
  OrtTensorView & operator=(OrtTensorView &&) noexcept;
  ONNXRUNTIME_CONVERSIONS_PUBLIC
  ~OrtTensorView();

  OrtTensorView(const OrtTensorView &) = delete;
  OrtTensorView & operator=(const OrtTensorView &) = delete;

  ONNXRUNTIME_CONVERSIONS_PUBLIC
  Ort::Value & value();
  ONNXRUNTIME_CONVERSIONS_PUBLIC
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

ONNXRUNTIME_CONVERSIONS_PUBLIC
std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & backend = "auto");

ONNXRUNTIME_CONVERSIONS_PUBLIC
std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const BackendConfiguration & configuration);

ONNXRUNTIME_CONVERSIONS_PUBLIC
std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & backend,
  const BackendConfiguration & configuration);

ONNXRUNTIME_CONVERSIONS_PUBLIC
OrtTensorView from_input_tensor_msg(
  std::shared_ptr<const TensorMsg> msg,
  const Ort::MemoryInfo & memory_info,
  void * execution_stream = nullptr);

ONNXRUNTIME_CONVERSIONS_PUBLIC
OrtTensorView from_output_tensor_msg(
  std::shared_ptr<TensorMsg> msg,
  const Ort::MemoryInfo & memory_info,
  void * execution_stream = nullptr);

ONNXRUNTIME_CONVERSIONS_PUBLIC
void to_tensor_msg(
  TensorMsg & msg,
  const Ort::Value & value,
  void * execution_stream = nullptr);

ONNXRUNTIME_CONVERSIONS_PUBLIC
std::unique_ptr<TensorMsg> to_tensor_msg(
  const Ort::Value & value,
  const std::string & backend = "auto",
  void * execution_stream = nullptr);

ONNXRUNTIME_CONVERSIONS_PUBLIC
std::vector<std::string> available_backends();

ONNXRUNTIME_CONVERSIONS_PUBLIC
void configure_session_options(
  Ort::SessionOptions & session_options,
  const std::string & backend = "auto",
  const BackendConfiguration & configuration = {});

}  // namespace onnxruntime_conversions

#endif  // ONNXRUNTIME_CONVERSIONS__ONNXRUNTIME_CONVERSIONS_HPP_
