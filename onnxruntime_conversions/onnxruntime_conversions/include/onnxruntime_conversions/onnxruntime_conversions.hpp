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

/// Keep this handle alive until its sessions, tensor views, and queued work finish.
class ONNXRUNTIME_CONVERSIONS_PUBLIC Stream
{
public:
  const std::string & backend() const {return backend_;}
  int device_id() const {return device_id_;}
  void * handle() const;
  bool owns_stream() const {return owner_ != nullptr;}

private:
  friend Stream create_stream(Ort::Env &, const std::string &, int);
  friend Stream borrow_stream(void *, const std::string &, int);
  Stream(std::string backend, int device_id, void * handle, Ort::SyncStream owner);

  std::string backend_;
  int device_id_;
  void * borrowed_handle_;
  std::shared_ptr<Ort::SyncStream> owner_;
};

/// Create an ORT-owned accelerator stream, or an empty stream for CPU execution.
ONNXRUNTIME_CONVERSIONS_PUBLIC Stream create_stream(
  Ort::Env & env, const std::string & backend = {}, int device_id = 0);

/// Wrap a native stream without taking ownership. The caller must keep it alive.
ONNXRUNTIME_CONVERSIONS_PUBLIC Stream borrow_stream(
  void * handle, const std::string & backend, int device_id = 0);

/// An Ort::Value over borrowed storage. Keep msg alive and do not resize its buffer.
class ONNXRUNTIME_CONVERSIONS_PUBLIC OrtTensorView
{
public:
  OrtTensorView();
  OrtTensorView(std::shared_ptr<void> lease, Ort::Value value);
  ~OrtTensorView() = default;

  OrtTensorView(OrtTensorView &&) noexcept = default;
  OrtTensorView & operator=(OrtTensorView && other) noexcept;
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
  const Stream & stream);
ONNXRUNTIME_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & backend = {},
  int device_id = -1);

ONNXRUNTIME_CONVERSIONS_PUBLIC OrtTensorView from_input_tensor_msg(
  const TensorMsg & msg, const Stream & stream);
ONNXRUNTIME_CONVERSIONS_PUBLIC OrtTensorView from_input_tensor_msg(
  const TensorMsg & msg,
  void * execution_stream = nullptr);
ONNXRUNTIME_CONVERSIONS_PUBLIC OrtTensorView from_output_tensor_msg(
  TensorMsg & msg, const Stream & stream);
ONNXRUNTIME_CONVERSIONS_PUBLIC OrtTensorView from_output_tensor_msg(
  TensorMsg & msg,
  void * execution_stream = nullptr);

ONNXRUNTIME_CONVERSIONS_PUBLIC void to_tensor_msg(
  TensorMsg & msg, const Ort::Value & value, const Stream & stream);
ONNXRUNTIME_CONVERSIONS_PUBLIC void to_tensor_msg(
  TensorMsg & msg,
  const Ort::Value & value,
  void * execution_stream = nullptr);
ONNXRUNTIME_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> to_tensor_msg(
  const Ort::Value & value, const Stream & stream);
ONNXRUNTIME_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> to_tensor_msg(
  const Ort::Value & value,
  void * execution_stream = nullptr);

ONNXRUNTIME_CONVERSIONS_PUBLIC void configure_session_options(
  Ort::SessionOptions & session_options, const Stream & stream);
ONNXRUNTIME_CONVERSIONS_PUBLIC void configure_session_options(
  Ort::SessionOptions & session_options,
  const std::string & backend = {},
  int device_id = 0,
  void * execution_stream = nullptr);

}  // namespace onnxruntime_conversions

#endif  // ONNXRUNTIME_CONVERSIONS__ONNXRUNTIME_CONVERSIONS_HPP_
