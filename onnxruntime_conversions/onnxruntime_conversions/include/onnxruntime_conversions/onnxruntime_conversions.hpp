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

#include <memory>
#include <string>
#include <vector>

#include "tensor_msgs/msg/experimental_tensor.hpp"
#include "onnxruntime_conversions/visibility_control.hpp"

namespace onnxruntime_conversions
{

using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

/// An Ort::Value over message storage. CPU storage must outlive the view.
class ONNXRUNTIME_CONVERSIONS_PUBLIC OrtTensorView
{
public:
  OrtTensorView();
  OrtTensorView(std::shared_ptr<void> lease, Ort::Value value);
  ~OrtTensorView() = default;

  OrtTensorView(OrtTensorView &&) noexcept = default;
  OrtTensorView & operator=(OrtTensorView &&) noexcept = default;
  OrtTensorView(const OrtTensorView &) = delete;
  OrtTensorView & operator=(const OrtTensorView &) = delete;

  Ort::Value & value();
  const Ort::Value & value() const;
  explicit operator bool() const noexcept;

private:
  // value_ must be destroyed before the storage lease is released.
  std::shared_ptr<void> lease_;
  Ort::Value value_{nullptr};
};

ONNXRUNTIME_CONVERSIONS_PUBLIC std::vector<std::string> available_backends();
ONNXRUNTIME_CONVERSIONS_PUBLIC bool backend_available(
  const std::string & backend);
ONNXRUNTIME_CONVERSIONS_PUBLIC std::string default_backend();

ONNXRUNTIME_CONVERSIONS_PUBLIC ONNXTensorElementDataType element_type(
  const TensorMsg & msg);
ONNXRUNTIME_CONVERSIONS_PUBLIC std::vector<int64_t> normalized_strides(
  const TensorMsg & msg);
ONNXRUNTIME_CONVERSIONS_PUBLIC size_t tensor_byte_count(const TensorMsg & msg);

ONNXRUNTIME_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & backend = {});

ONNXRUNTIME_CONVERSIONS_PUBLIC OrtTensorView from_input_tensor_msg(
  const TensorMsg & msg,
  void * execution_stream = nullptr);
ONNXRUNTIME_CONVERSIONS_PUBLIC OrtTensorView from_output_tensor_msg(
  TensorMsg & msg,
  void * execution_stream = nullptr);

ONNXRUNTIME_CONVERSIONS_PUBLIC void to_tensor_msg(
  TensorMsg & msg,
  const Ort::Value & value,
  void * execution_stream = nullptr);
ONNXRUNTIME_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> to_tensor_msg(
  const Ort::Value & value,
  void * execution_stream = nullptr);

ONNXRUNTIME_CONVERSIONS_PUBLIC void configure_session_options(
  Ort::SessionOptions & session_options,
  const std::string & backend = {},
  int device_id = 0,
  void * execution_stream = nullptr);

}  // namespace onnxruntime_conversions

#endif  // ONNXRUNTIME_CONVERSIONS__ONNXRUNTIME_CONVERSIONS_HPP_
