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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// dlpack_conversions.hpp static_asserts the DLManagedTensor layout, so a
// DLPack header pulled in ahead of this one cannot silently disagree with the
// core this translation unit links against.
#include "dlpack_conversions/dlpack_conversions.hpp"

namespace onnxruntime_conversions
{

using TensorMsg = dlpack_conversions::TensorMsg;

namespace detail
{

inline size_t element_size(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return 16;
    default:
      throw std::invalid_argument(
              "onnxruntime_conversions: unsupported ONNX tensor element type");
  }
}

inline DLDataType dl_dtype(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return {kDLInt, 8, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return {kDLInt, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return {kDLInt, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return {kDLInt, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return {kDLUInt, 8, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return {kDLUInt, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return {kDLUInt, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return {kDLUInt, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return {kDLFloat, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return {kDLFloat, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return {kDLFloat, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return {kDLBfloat, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return {kDLComplex, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return {kDLComplex, 128, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return {kDLBool, 8, 1};
    default:
      throw std::invalid_argument(
              "onnxruntime_conversions: unsupported ONNX tensor element type");
  }
}

inline ONNXTensorElementDataType onnx_dtype(const DLDataType & dtype)
{
  if (dtype.lanes != 1) {
    throw std::invalid_argument(
            "onnxruntime_conversions: ONNX Runtime tensors require lanes == 1");
  }
  switch (dtype.code) {
    case kDLInt:
      switch (dtype.bits) {
        case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      }
      break;
    case kDLUInt:
      switch (dtype.bits) {
        case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
      }
      break;
    case kDLFloat:
      switch (dtype.bits) {
        case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
      }
      break;
    case kDLBfloat:
      if (dtype.bits == 16) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
      }
      break;
    case kDLComplex:
      switch (dtype.bits) {
        case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
        case 128: return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
      }
      break;
    case kDLBool:
      if (dtype.bits == 8) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      }
      break;
    default:
      break;
  }
  throw std::invalid_argument(
          "onnxruntime_conversions: tensor dtype is unsupported by ONNX Runtime");
}

inline std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (size_t index = shape.size(); index > 0; --index) {
    strides[index - 1] = stride;
    stride *= shape[index - 1];
  }
  return strides;
}

inline size_t byte_count_of(const DLTensor & tensor)
{
  size_t count = 1;
  for (int32_t index = 0; index < tensor.ndim; ++index) {
    const int64_t dimension = tensor.shape[index];
    if (dimension < 0) {
      throw std::invalid_argument(
              "onnxruntime_conversions: tensor shape dimensions must be nonnegative");
    }
    count *= static_cast<size_t>(dimension);
  }
  return count * ((static_cast<size_t>(tensor.dtype.bits) * tensor.dtype.lanes + 7) / 8);
}

/// ONNX Runtime only reads row-major storage, so a strided view cannot be
/// handed over without a copy.
inline void require_contiguous(const DLTensor & tensor)
{
  if (tensor.strides == nullptr) {
    return;
  }
  int64_t stride = 1;
  for (int32_t index = tensor.ndim; index > 0; --index) {
    if (tensor.strides[index - 1] != stride) {
      throw std::invalid_argument(
              "onnxruntime_conversions: ONNX Runtime requires contiguous tensor strides");
    }
    stride *= tensor.shape[index - 1];
  }
}

/// The storage plugin already decided where the tensor lives, so the memory
/// info follows from the DLPack device rather than from the caller.
inline Ort::MemoryInfo memory_info_for(const DLDevice & device)
{
  switch (device.device_type) {
    case kDLCPU:
      return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    case kDLCUDA:
      return Ort::MemoryInfo(
        "Cuda", OrtDeviceAllocator, device.device_id, OrtMemTypeDefault);
    case kDLROCM:
      return Ort::MemoryInfo(
        "Hip", OrtDeviceAllocator, device.device_id, OrtMemTypeDefault);
    default:
      throw std::invalid_argument(
              "onnxruntime_conversions: no ONNX Runtime allocator for DLPack device " +
              std::to_string(static_cast<int32_t>(device.device_type)));
  }
}

inline DLDevice dl_device_for(const Ort::ConstMemoryInfo & info)
{
  DLDevice device{};
  switch (info.GetDeviceType()) {
    case OrtMemoryInfoDeviceType_CPU:
      device.device_type = kDLCPU;
      device.device_id = 0;
      return device;
    case OrtMemoryInfoDeviceType_GPU:
      device.device_type = info.GetAllocatorName() == "Hip" ? kDLROCM : kDLCUDA;
      device.device_id = info.GetDeviceId();
      return device;
    default:
      throw std::invalid_argument(
              "onnxruntime_conversions: unsupported ONNX Runtime device type");
  }
}

/// A DLTensor over an Ort::Value, owning the shape and stride arrays it points
/// at so it stays valid for the duration of a core call.
struct OrtSource
{
  DLTensor tensor{};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

inline OrtSource as_dl_source(const Ort::Value & value)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("onnxruntime_conversions: Ort::Value is not a tensor");
  }
  const auto info = value.GetTensorTypeAndShapeInfo();
  OrtSource source;
  source.shape = info.GetShape();
  source.strides = contiguous_strides(source.shape);
  source.tensor.data = const_cast<void *>(value.GetTensorRawData());
  source.tensor.device = dl_device_for(value.GetTensorMemoryInfo());
  source.tensor.ndim = static_cast<int32_t>(source.shape.size());
  source.tensor.dtype = dl_dtype(info.GetElementType());
  source.tensor.shape = source.shape.data();
  source.tensor.strides = source.strides.data();
  source.tensor.byte_offset = 0;
  return source;
}

}  // namespace detail

/// An Ort::Value over tensor message storage.
///
/// The view holds the storage lease, so keep it alive for as long as ONNX
/// Runtime reads from or writes to the value.
class OrtTensorView
{
public:
  OrtTensorView() = default;
  OrtTensorView(dlpack_conversions::ManagedTensor managed, Ort::Value value)
  : managed_(std::move(managed)), value_(std::move(value)) {}

  OrtTensorView(OrtTensorView &&) noexcept = default;
  OrtTensorView & operator=(OrtTensorView &&) noexcept = default;
  OrtTensorView(const OrtTensorView &) = delete;
  OrtTensorView & operator=(const OrtTensorView &) = delete;

  Ort::Value & value() {return value_;}
  const Ort::Value & value() const {return value_;}

  /// False when the message carried no storage.
  explicit operator bool() const noexcept {return static_cast<bool>(managed_);}

private:
  dlpack_conversions::ManagedTensor managed_;
  Ort::Value value_{nullptr};
};

/// Backends installed in this process, in name order.
inline std::vector<std::string> available_backends()
{
  return dlpack_conversions::available_backends();
}

/// Backend used when a caller does not name one.
inline std::string default_backend()
{
  return dlpack_conversions::default_backend();
}

/// Allocates contiguous storage for a tensor message. An empty backend selects
/// default_backend().
inline std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & backend = {})
{
  return dlpack_conversions::allocate_tensor_msg(
    shape, detail::dl_dtype(dtype), backend);
}

namespace detail
{

inline OrtTensorView make_view(dlpack_conversions::ManagedTensor managed)
{
  if (!managed) {
    return {};
  }
  const DLTensor & tensor = managed.get()->dl_tensor;
  require_contiguous(tensor);
  // Ort::Value::CreateTensor takes a base pointer with no offset of its own,
  // so the core is expected to have folded byte_offset into the pointer.
  if (tensor.byte_offset != 0) {
    throw std::invalid_argument(
            "onnxruntime_conversions: DLPack byte_offset must be folded into "
            "the data pointer");
  }
  const auto dtype = onnx_dtype(tensor.dtype);
  auto memory_info = memory_info_for(tensor.device);
  auto value = Ort::Value::CreateTensor(
    memory_info, tensor.data, byte_count_of(tensor),
    tensor.shape, static_cast<size_t>(tensor.ndim), dtype);
  return OrtTensorView(std::move(managed), std::move(value));
}

}  // namespace detail

/// Reads message storage without copying. Pass the stream ONNX Runtime will
/// run on so the storage plugin can order access against it.
inline OrtTensorView from_input_tensor_msg(
  const TensorMsg & msg,
  void * execution_stream = nullptr)
{
  return detail::make_view(
    dlpack_conversions::from_input_tensor_msg(
      msg, reinterpret_cast<uintptr_t>(execution_stream)));
}

/// Exposes message storage for ONNX Runtime to write into.
inline OrtTensorView from_output_tensor_msg(
  TensorMsg & msg,
  void * execution_stream = nullptr)
{
  return detail::make_view(
    dlpack_conversions::from_output_tensor_msg(
      msg, reinterpret_cast<uintptr_t>(execution_stream)));
}

/// Copies a tensor into existing message storage and stamps its metadata.
inline void to_tensor_msg(
  TensorMsg & msg,
  const Ort::Value & value,
  void * execution_stream = nullptr)
{
  const auto source = detail::as_dl_source(value);
  dlpack_conversions::to_tensor_msg(
    msg, source.tensor, reinterpret_cast<uintptr_t>(execution_stream));
}

/// Allocates a message on the backend matching the value's device, then copies.
inline std::unique_ptr<TensorMsg> to_tensor_msg(
  const Ort::Value & value,
  void * execution_stream = nullptr)
{
  const auto source = detail::as_dl_source(value);
  return dlpack_conversions::to_tensor_msg(
    source.tensor, reinterpret_cast<uintptr_t>(execution_stream));
}

/// Appends the execution provider that runs where the given backend allocates,
/// so a session reads message storage in place.
///
/// Only the backends ONNX Runtime ships a provider for are handled. For any
/// other backend, append the provider yourself.
inline void configure_session_options(
  Ort::SessionOptions & session_options,
  const std::string & backend = {},
  int device_id = 0,
  void * execution_stream = nullptr)
{
  const std::string selected =
    backend.empty() ? dlpack_conversions::default_backend() : backend;
  if (selected == "cpu") {
    if (execution_stream != nullptr) {
      throw std::invalid_argument(
              "onnxruntime_conversions: host memory sessions take no execution stream");
    }
    return;
  }
  if (execution_stream == nullptr) {
    throw std::invalid_argument(
            "onnxruntime_conversions: backend '" + selected +
            "' requires an explicit execution stream");
  }
  if (selected == "cuda") {
    Ort::CUDAProviderOptions options;
    options.Update({{"device_id", std::to_string(device_id)}});
    options.UpdateWithValue("user_compute_stream", execution_stream);
    session_options.AppendExecutionProvider_CUDA_V2(*options);
    return;
  }
  if (selected == "rocm") {
    OrtROCMProviderOptions options{};
    options.device_id = device_id;
    options.has_user_compute_stream = 1;
    options.user_compute_stream = execution_stream;
    session_options.AppendExecutionProvider_ROCM(options);
    return;
  }
  throw std::invalid_argument(
          "onnxruntime_conversions: no ONNX Runtime execution provider is known for "
          "backend '" + selected + "'; append one yourself");
}

}  // namespace onnxruntime_conversions

#endif  // ONNXRUNTIME_CONVERSIONS__ONNXRUNTIME_CONVERSIONS_HPP_
