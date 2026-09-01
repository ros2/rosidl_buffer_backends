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

#ifndef ONNXRUNTIME_CONVERSIONS__DETAIL__ONNXRUNTIME_CONVERSIONS_IMPL_HPP_
#define ONNXRUNTIME_CONVERSIONS__DETAIL__ONNXRUNTIME_CONVERSIONS_IMPL_HPP_

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rosidl_buffer/buffer.hpp"

#ifdef ONNXRUNTIME_CONVERSIONS_HAS_CUDA
#include <cuda_runtime.h>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#endif

namespace onnxruntime_conversions
{
namespace detail
{

inline constexpr uint8_t kDlInt = 0;
inline constexpr uint8_t kDlUInt = 1;
inline constexpr uint8_t kDlFloat = 2;
inline constexpr uint8_t kDlBfloat = 4;
inline constexpr uint8_t kDlComplex = 5;
inline constexpr uint8_t kDlBool = 6;

struct TensorMetadata
{
  std::vector<int64_t> shape;
  ONNXTensorElementDataType dtype;
  size_t element_size;
  size_t byte_count;
};

inline size_t checked_multiply(size_t lhs, size_t rhs, const char * context)
{
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw std::overflow_error(context);
  }
  return lhs * rhs;
}

inline std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (size_t index = shape.size(); index > 0; --index) {
    strides[index - 1] = stride;
    if (shape[index - 1] != 0 &&
      stride > std::numeric_limits<int64_t>::max() / shape[index - 1])
    {
      throw std::overflow_error("Tensor strides overflow int64");
    }
    stride *= shape[index - 1];
  }
  return strides;
}

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
      throw std::invalid_argument("Unsupported ONNX tensor element type");
  }
}

inline ONNXTensorElementDataType dtype_from_msg(const TensorMsg & msg)
{
  if (msg.dtype_lanes != 1) {
    throw std::invalid_argument("ONNX Runtime tensors require dtype_lanes == 1");
  }

  switch (msg.dtype_code) {
    case kDlInt:
      switch (msg.dtype_bits) {
        case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      }
      break;
    case kDlUInt:
      switch (msg.dtype_bits) {
        case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
      }
      break;
    case kDlFloat:
      switch (msg.dtype_bits) {
        case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
      }
      break;
    case kDlBfloat:
      if (msg.dtype_bits == 16) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
      }
      break;
    case kDlComplex:
      switch (msg.dtype_bits) {
        case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
        case 128: return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
      }
      break;
    case kDlBool:
      if (msg.dtype_bits == 8) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
      }
      break;
  }
  throw std::invalid_argument("ExperimentalTensor dtype is unsupported by ONNX Runtime");
}

inline void set_msg_dtype(TensorMsg & msg, ONNXTensorElementDataType dtype)
{
  msg.dtype_lanes = 1;
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      msg.dtype_code = kDlInt;
      msg.dtype_bits = 8;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      msg.dtype_code = kDlInt;
      msg.dtype_bits = 16;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      msg.dtype_code = kDlInt;
      msg.dtype_bits = 32;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      msg.dtype_code = kDlInt;
      msg.dtype_bits = 64;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      msg.dtype_code = kDlUInt;
      msg.dtype_bits = 8;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      msg.dtype_code = kDlUInt;
      msg.dtype_bits = 16;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      msg.dtype_code = kDlUInt;
      msg.dtype_bits = 32;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      msg.dtype_code = kDlUInt;
      msg.dtype_bits = 64;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      msg.dtype_code = kDlFloat;
      msg.dtype_bits = 16;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      msg.dtype_code = kDlFloat;
      msg.dtype_bits = 32;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      msg.dtype_code = kDlFloat;
      msg.dtype_bits = 64;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      msg.dtype_code = kDlBfloat;
      msg.dtype_bits = 16;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      msg.dtype_code = kDlComplex;
      msg.dtype_bits = 64;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      msg.dtype_code = kDlComplex;
      msg.dtype_bits = 128;
      return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      msg.dtype_code = kDlBool;
      msg.dtype_bits = 8;
      return;
    default:
      throw std::invalid_argument("Unsupported ONNX tensor element type");
  }
}

inline TensorMetadata validate_metadata(const TensorMsg & msg)
{
  TensorMetadata metadata;
  metadata.shape.assign(msg.shape.begin(), msg.shape.end());
  metadata.dtype = dtype_from_msg(msg);
  metadata.element_size = element_size(metadata.dtype);

  size_t element_count = 1;
  for (const int64_t dimension : metadata.shape) {
    if (dimension < 0) {
      throw std::invalid_argument("Tensor shape dimensions must be nonnegative");
    }
    if (static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max()) {
      throw std::overflow_error("Tensor shape dimension exceeds size_t");
    }
    element_count = checked_multiply(
      element_count, static_cast<size_t>(dimension), "Tensor element count overflow");
  }

  const auto expected_strides = contiguous_strides(metadata.shape);
  if (!msg.strides.empty()) {
    if (msg.strides.size() != expected_strides.size() ||
      !std::equal(msg.strides.begin(), msg.strides.end(), expected_strides.begin()))
    {
      throw std::invalid_argument("ONNX Runtime conversion requires contiguous tensor strides");
    }
  }

  metadata.byte_count = checked_multiply(
    element_count, metadata.element_size, "Tensor byte count overflow");
  if (msg.byte_offset > std::numeric_limits<size_t>::max()) {
    throw std::overflow_error("Tensor byte_offset exceeds size_t");
  }
  const size_t byte_offset = static_cast<size_t>(msg.byte_offset);
  if (byte_offset % metadata.element_size != 0) {
    throw std::invalid_argument("Tensor byte_offset is not aligned to its element type");
  }
  if (byte_offset > msg.data.size() || metadata.byte_count > msg.data.size() - byte_offset) {
    throw std::out_of_range("Tensor view exceeds its backing buffer");
  }
  return metadata;
}

inline void validate_memory_info(
  const TensorMsg & msg,
  const Ort::MemoryInfo & memory_info)
{
  const auto device_type = memory_info.GetDeviceType();
  const std::string & backend = msg.data.get_backend_type();
  if (backend != "cpu") {
#ifdef ONNXRUNTIME_CONVERSIONS_HAS_CUDA
    if (backend == "cuda") {
      if (device_type != OrtMemoryInfoDeviceType_GPU ||
        memory_info.GetAllocatorName() != "Cuda")
      {
        throw std::invalid_argument("CUDA buffer requires CUDA Ort::MemoryInfo");
      }
      return;
    }
#endif
    throw std::invalid_argument("Unsupported rosidl buffer backend '" + backend + "'");
  }
  if (device_type != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("CPU buffer requires CPU Ort::MemoryInfo");
  }
}

inline uint8_t empty_storage = 0;

}  // namespace detail

struct OrtTensorView::Impl
{
  explicit Impl(std::shared_ptr<const TensorMsg> owner_arg)
  : owner(std::move(owner_arg)), value(nullptr) {}

  std::shared_ptr<const TensorMsg> owner;
#ifdef ONNXRUNTIME_CONVERSIONS_HAS_CUDA
  std::optional<cuda_buffer_backend::ReadHandle> read_handle;
  std::optional<cuda_buffer_backend::WriteHandle> write_handle;
#endif
  Ort::Value value;
};

inline OrtTensorView::OrtTensorView(std::unique_ptr<Impl> impl)
: impl_(std::move(impl)) {}

inline OrtTensorView::OrtTensorView(OrtTensorView &&) noexcept = default;

inline OrtTensorView & OrtTensorView::operator=(OrtTensorView &&) noexcept = default;

inline OrtTensorView::~OrtTensorView() = default;

inline Ort::Value & OrtTensorView::value()
{
  if (!impl_) {
    throw std::logic_error("OrtTensorView has been moved from");
  }
  return impl_->value;
}

inline const Ort::Value & OrtTensorView::value() const
{
  if (!impl_) {
    throw std::logic_error("OrtTensorView has been moved from");
  }
  return impl_->value;
}

inline std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & backend)
{
  size_t element_count = 1;
  for (const int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument("Tensor shape dimensions must be nonnegative");
    }
    if (static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max()) {
      throw std::overflow_error("Tensor shape dimension exceeds size_t");
    }
    element_count = detail::checked_multiply(
      element_count, static_cast<size_t>(dimension), "Tensor element count overflow");
  }
  const size_t byte_count = detail::checked_multiply(
    element_count, detail::element_size(dtype), "Tensor byte count overflow");

  auto msg = std::make_unique<TensorMsg>();
  detail::set_msg_dtype(*msg, dtype);
  msg->shape.assign(shape.begin(), shape.end());
  const auto strides = detail::contiguous_strides(shape);
  msg->strides.assign(strides.begin(), strides.end());
  msg->byte_offset = 0;

  if (backend == "cpu") {
    msg->data.resize(byte_count);
    return msg;
  }
#ifdef ONNXRUNTIME_CONVERSIONS_HAS_CUDA
  if (backend == "cuda") {
    auto cuda_impl =
      std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(byte_count);
    msg->data = rosidl::Buffer<uint8_t>(std::move(cuda_impl));
    return msg;
  }
#endif
  throw std::invalid_argument("Unsupported allocation backend '" + backend + "'");
}

inline OrtTensorView from_input_tensor_msg(
  std::shared_ptr<const TensorMsg> msg,
  const Ort::MemoryInfo & memory_info,
  void * execution_stream)
{
  if (!msg) {
    throw std::invalid_argument("Input tensor message must not be null");
  }
  const detail::TensorMetadata metadata = detail::validate_metadata(*msg);
  detail::validate_memory_info(*msg, memory_info);
  auto impl = std::make_unique<OrtTensorView::Impl>(msg);

  void * data = nullptr;
  if (msg->data.get_backend_type() == "cpu") {
    data = msg->data.empty() ?
      static_cast<void *>(&detail::empty_storage) :
      const_cast<void *>(static_cast<const void *>(msg->data.data()));
  } else {
#ifdef ONNXRUNTIME_CONVERSIONS_HAS_CUDA
    const auto * cuda_impl =
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      msg->data.get_impl());
    if (!cuda_impl) {
      throw std::runtime_error("CUDA backend is not a CudaBufferImpl");
    }
    if (memory_info.GetDeviceId() != cuda_impl->get_device_id()) {
      throw std::invalid_argument("CUDA buffer and Ort::MemoryInfo device IDs differ");
    }
    const auto stream = reinterpret_cast<cudaStream_t>(execution_stream);
    impl->read_handle.emplace(cuda_impl->get_cuda_buffer().get_read_handle(stream));
    data = const_cast<uint8_t *>(impl->read_handle->get_ptr());
#else
    (void)execution_stream;
    throw std::runtime_error("CUDA support is not compiled into onnxruntime_conversions");
#endif
  }

  data = static_cast<uint8_t *>(data) + msg->byte_offset;
  impl->value = Ort::Value::CreateTensor(
    memory_info, data, metadata.byte_count, metadata.shape.data(),
    metadata.shape.size(), metadata.dtype);
  return OrtTensorView(std::move(impl));
}

inline OrtTensorView from_output_tensor_msg(
  std::shared_ptr<TensorMsg> msg,
  const Ort::MemoryInfo & memory_info,
  void * execution_stream)
{
  if (!msg) {
    throw std::invalid_argument("Output tensor message must not be null");
  }
  const detail::TensorMetadata metadata = detail::validate_metadata(*msg);
  detail::validate_memory_info(*msg, memory_info);
  auto impl = std::make_unique<OrtTensorView::Impl>(msg);

  void * data = nullptr;
  if (msg->data.get_backend_type() == "cpu") {
    data = msg->data.empty() ?
      static_cast<void *>(&detail::empty_storage) :
      static_cast<void *>(msg->data.data());
  } else {
#ifdef ONNXRUNTIME_CONVERSIONS_HAS_CUDA
    auto * cuda_impl =
      dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      msg->data.get_impl());
    if (!cuda_impl) {
      throw std::runtime_error("CUDA backend is not a CudaBufferImpl");
    }
    if (memory_info.GetDeviceId() != cuda_impl->get_device_id()) {
      throw std::invalid_argument("CUDA buffer and Ort::MemoryInfo device IDs differ");
    }
    const auto stream = reinterpret_cast<cudaStream_t>(execution_stream);
    cuda_impl->set_stream(stream);
    impl->write_handle.emplace(cuda_impl->get_cuda_buffer().get_write_handle(stream));
    data = impl->write_handle->get_ptr();
#else
    (void)execution_stream;
    throw std::runtime_error("CUDA support is not compiled into onnxruntime_conversions");
#endif
  }

  data = static_cast<uint8_t *>(data) + msg->byte_offset;
  impl->value = Ort::Value::CreateTensor(
    memory_info, data, metadata.byte_count, metadata.shape.data(),
    metadata.shape.size(), metadata.dtype);
  return OrtTensorView(std::move(impl));
}

inline void to_tensor_msg(TensorMsg & msg, const Ort::Value & value)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("Ort::Value is not a tensor");
  }
  if (value.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("to_tensor_msg currently supports CPU Ort::Value tensors only");
  }
  if (msg.data.get_backend_type() != "cpu") {
    throw std::invalid_argument("to_tensor_msg requires a CPU destination buffer");
  }

  const auto type_info = value.GetTensorTypeAndShapeInfo();
  const auto dtype = type_info.GetElementType();
  const auto shape = type_info.GetShape();
  const size_t byte_count = detail::checked_multiply(
    type_info.GetElementCount(), detail::element_size(dtype), "Tensor byte count overflow");
  if (byte_count > msg.data.size()) {
    throw std::out_of_range("Ort::Value tensor exceeds the destination buffer");
  }

  if (byte_count != 0) {
    std::memcpy(msg.data.data(), value.GetTensorRawData(), byte_count);
  }
  detail::set_msg_dtype(msg, dtype);
  msg.shape.assign(shape.begin(), shape.end());
  const auto strides = detail::contiguous_strides(shape);
  msg.strides.assign(strides.begin(), strides.end());
  msg.byte_offset = 0;
}

inline std::unique_ptr<TensorMsg> to_tensor_msg(const Ort::Value & value)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("Ort::Value is not a tensor");
  }
  const auto type_info = value.GetTensorTypeAndShapeInfo();
  auto msg = allocate_tensor_msg(type_info.GetShape(), type_info.GetElementType());
  to_tensor_msg(*msg, value);
  return msg;
}

}  // namespace onnxruntime_conversions

#endif  // ONNXRUNTIME_CONVERSIONS__DETAIL__ONNXRUNTIME_CONVERSIONS_IMPL_HPP_
