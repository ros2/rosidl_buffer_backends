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

#ifndef TORCH_CONVERSIONS__TORCH_CONVERSIONS_HPP_
#define TORCH_CONVERSIONS__TORCH_CONVERSIONS_HPP_

#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <c10/core/StreamGuard.h>
#include <rcutils/logging_macros.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rosidl_buffer/buffer.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

#if __has_include("cuda_buffer/cuda_buffer_api.hpp")
#include <c10/cuda/CUDAStream.h>
#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#define TORCH_CONVERSIONS_HAS_CUDA
#endif

namespace torch_conversions
{

using TensorMsg = tensor_msgs::msg::ExperimentalTensor;
using ImageMsg = sensor_msgs::msg::Image;

namespace detail
{

// Internal DLPack type aliases and helpers. Users do not need to touch
// DLPack directly; the public torch-facing API (allocate_tensor_msg,
// from_input_tensor_msg, from_output_tensor_msg, to_tensor_msg) hides it.
using DLDataType = ::DLDataType;
using DLDevice = ::DLDevice;
using DLManagedTensor = ::DLManagedTensor;

inline DLDataType dl_dtype_from_scalar(at::ScalarType t)
{
  switch (t) {
    case at::kByte:     return DLDataType{kDLUInt, 8, 1};
    case at::kChar:     return DLDataType{kDLInt, 8, 1};
    case at::kShort:    return DLDataType{kDLInt, 16, 1};
    case at::kInt:      return DLDataType{kDLInt, 32, 1};
    case at::kLong:     return DLDataType{kDLInt, 64, 1};
    case at::kHalf:     return DLDataType{kDLFloat, 16, 1};
    case at::kBFloat16: return DLDataType{kDLBfloat, 16, 1};
    case at::kFloat:    return DLDataType{kDLFloat, 32, 1};
    case at::kDouble:   return DLDataType{kDLFloat, 64, 1};
    case at::kBool:     return DLDataType{kDLBool, 8, 1};
    default:
      throw std::runtime_error(
              "torch_conversions: unsupported at::ScalarType for DLPack encoding");
  }
}

inline at::ScalarType scalar_from_dl_dtype(DLDataType d)
{
  if (d.lanes != 1) {
    throw std::runtime_error(
            "torch_conversions: dtype_lanes != 1 not representable as at::ScalarType");
  }
  switch (d.code) {
    case kDLUInt:
      if (d.bits == 8) {return at::kByte;}
      break;
    case kDLInt:
      switch (d.bits) {
        case 8: return at::kChar;
        case 16: return at::kShort;
        case 32: return at::kInt;
        case 64: return at::kLong;
      }
      break;
    case kDLFloat:
      switch (d.bits) {
        case 16: return at::kHalf;
        case 32: return at::kFloat;
        case 64: return at::kDouble;
      }
      break;
    case kDLBfloat:
      if (d.bits == 16) {return at::kBFloat16;}
      break;
    case kDLBool:
      if (d.bits == 8) {return at::kBool;}
      break;
  }
  throw std::runtime_error(
          "torch_conversions: unsupported DLDataType (code=" +
          std::to_string(static_cast<int>(d.code)) +
          ", bits=" + std::to_string(static_cast<int>(d.bits)) +
          ", lanes=" + std::to_string(d.lanes) + ")");
}

inline size_t dl_dtype_bytesize(DLDataType d)
{
  return (static_cast<size_t>(d.bits) * d.lanes + 7) / 8;
}

inline void set_dtype(TensorMsg & m, DLDataType d)
{
  m.dtype_code = d.code;
  m.dtype_bits = d.bits;
  m.dtype_lanes = d.lanes;
}

// ---------------------------------------------------------------------------
// Stream helpers
// ---------------------------------------------------------------------------

inline std::optional<c10::Stream> select_torch_stream()
{
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  if (torch::cuda::is_available()) {
    return c10::cuda::getStreamFromPool();
  }
#endif
  return std::nullopt;
}

inline c10::DeviceType default_device()
{
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  if (torch::cuda::is_available()) {
    return c10::kCUDA;
  }
#endif
  return c10::kCPU;
}

inline std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> strides(shape.size());
  int64_t s = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = s;
    s *= shape[i];
  }
  return strides;
}

inline int64_t numel_of(const std::vector<int64_t> & shape)
{
  int64_t n = 1;
  for (auto d : shape) {
    if (d < 0) {
      throw std::runtime_error("torch_conversions: negative shape dimension");
    }
    n *= d;
  }
  return n;
}

struct ImageMetadata
{
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  at::ScalarType dtype;
  size_t byte_count;
};

inline size_t checked_multiply(size_t lhs, size_t rhs, const char * context)
{
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw std::runtime_error(
            std::string("torch_conversions::") + context + ": image size overflows size_t");
  }
  return lhs * rhs;
}

/// Resolve native Image metadata into an HWC torch view.
/// Only byte-oriented packed encodings are supported because PyTorch does
/// not have a portable dtype corresponding to every ROS image encoding.
inline ImageMetadata image_metadata(const ImageMsg & msg, const char * context)
{
  if (msg.encoding.empty()) {
    throw std::runtime_error(
            std::string("torch_conversions::") + context + ": Image.encoding is empty");
  }
  if (sensor_msgs::image_encodings::isPlanar(msg.encoding)) {
    throw std::runtime_error(
            std::string("torch_conversions::") + context +
            ": planar Image encoding " + msg.encoding + " cannot be represented as HWC");
  }

  int channels;
  int bits;
  try {
    channels = sensor_msgs::image_encodings::numChannels(msg.encoding);
    bits = sensor_msgs::image_encodings::bitDepth(msg.encoding);
  } catch (const std::runtime_error & error) {
    throw std::runtime_error(
            std::string("torch_conversions::") + context + ": " + error.what());
  }
  if (channels <= 0) {
    throw std::runtime_error(
            std::string("torch_conversions::") + context +
            ": encoding has a non-positive channel count");
  }
  if (bits != 8) {
    throw std::runtime_error(
            std::string("torch_conversions::") + context + ": Image encoding " +
            msg.encoding + " has " + std::to_string(bits) +
            "-bit channels; only byte-oriented encodings are supported");
  }

  const bool signed_bytes = msg.encoding.rfind("8SC", 0) == 0;
  const size_t packed_row = checked_multiply(
    static_cast<size_t>(msg.width), static_cast<size_t>(channels), context);
  if (msg.step < packed_row) {
    throw std::runtime_error(
            std::string("torch_conversions::") + context + ": Image.step (" +
            std::to_string(msg.step) + ") is smaller than the packed row size (" +
            std::to_string(packed_row) + ")");
  }
  const size_t byte_count = checked_multiply(
    static_cast<size_t>(msg.height), static_cast<size_t>(msg.step), context);
  if (msg.data.size() < byte_count) {
    throw std::runtime_error(
            std::string("torch_conversions::") + context + ": Image.data has " +
            std::to_string(msg.data.size()) + " bytes, but height * step requires " +
            std::to_string(byte_count));
  }

  return ImageMetadata{
    {static_cast<int64_t>(msg.height), static_cast<int64_t>(msg.width), channels},
    {static_cast<int64_t>(msg.step), channels, 1},
    signed_bytes ? at::kChar : at::kByte,
    byte_count};
}

#ifdef TORCH_CONVERSIONS_HAS_CUDA
inline cudaStream_t current_cuda_stream_or_warn(const char * context)
{
  cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
  if (s == nullptr) {
    RCUTILS_LOG_WARN_NAMED(
      "torch_conversions",
      "%s: current CUDA stream is the default stream. "
      "Set a non-default stream (e.g. via torch_conversions::set_stream()) "
      "for event-based synchronization.",
      context);
  }
  return s;
}
#endif


// ---------------------------------------------------------------------------
// DLPack hand-off helpers (framework-agnostic producer API)
// ---------------------------------------------------------------------------

/// Context held alive by the DLManagedTensor for the lifetime of the tensor
/// constructed from it. Owns:
///   - (CUDA builds only) a ReadHandle or WriteHandle that keeps the CUDA
///     storage alive and drives event-based synchronization in
///     cuda_buffer_backend, and
///   - stable int64_t storage for DLTensor::shape / DLTensor::strides.
struct DlpackContext
{
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  std::shared_ptr<cuda_buffer_backend::ReadHandle> rh;
  std::shared_ptr<cuda_buffer_backend::WriteHandle> wh;
#endif
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

/// C-style deleter used by every DLManagedTensor produced by the bridge.
/// Called exactly once by the DLPack consumer (e.g. at::fromDLPack, or
/// a framework's from_dlpack) when the imported tensor is destroyed.
inline void dlpack_deleter(DLManagedTensor * self)
{
  if (!self) {return;}
  delete static_cast<DlpackContext *>(self->manager_ctx);
  delete self;
}

/// Populate the DLTensor fields of `dlm` from `msg`, `ctx`, and the
/// resolved pointer/device. Callers set `dlm->manager_ctx` and `dlm->deleter` themselves.
///
/// Note: we bake `msg.byte_offset` into the `data` pointer and set
/// `dl_tensor.byte_offset = 0`. Per the DLPack spec both encodings describe
/// the same tensor, but several DLPack importers (including some versions
/// of torch's `at::fromDLPack`) ignore the `byte_offset` field and read
/// from `data` directly, so baking it in is the portable choice.
inline void populate_dl_tensor(
  DLManagedTensor & dlm,
  const TensorMsg & msg,
  const DlpackContext & ctx,
  void * data_ptr,
  int32_t dev_type,
  int32_t dev_id)
{
  auto * offset_ptr = static_cast<uint8_t *>(data_ptr) + msg.byte_offset;
  dlm.dl_tensor.data = static_cast<void *>(offset_ptr);
  dlm.dl_tensor.device = DLDevice{static_cast<DLDeviceType>(dev_type), dev_id};
  dlm.dl_tensor.ndim = static_cast<int32_t>(ctx.shape.size());
  dlm.dl_tensor.dtype = DLDataType{msg.dtype_code, msg.dtype_bits, msg.dtype_lanes};
  dlm.dl_tensor.shape = const_cast<int64_t *>(ctx.shape.data());
  dlm.dl_tensor.strides = ctx.strides.empty() ?
    nullptr : const_cast<int64_t *>(ctx.strides.data());
  dlm.dl_tensor.byte_offset = 0;
}

#ifdef TORCH_CONVERSIONS_HAS_CUDA

/// Build a DLManagedTensor for the input path.
/// For CUDA-backed `msg.data`, a ReadHandle is acquired on `consumer_stream`.
/// For CPU-backed data, `consumer_stream` is ignored.
inline DLManagedTensor * make_input_dlpack(
  const TensorMsg & msg,
  cudaStream_t consumer_stream = nullptr)
{
  auto ctx = std::make_unique<detail::DlpackContext>();
  ctx->shape.assign(msg.shape.begin(), msg.shape.end());
  ctx->strides.assign(msg.strides.begin(), msg.strides.end());

  void * data_ptr = nullptr;
  int32_t dev_type = kDLCPU;
  int32_t dev_id = 0;

  const std::string & backend = msg.data.get_backend_type();
  if (backend == "cuda") {
    const auto * cuda_impl =
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      msg.data.get_impl());
    if (!cuda_impl) {
      throw std::runtime_error(
              "torch_conversions::make_input_dlpack: cuda backend but not CudaBufferImpl");
    }
    ctx->rh = std::make_shared<cuda_buffer_backend::ReadHandle>(
      cuda_impl->get_cuda_buffer().get_read_handle(consumer_stream));
    data_ptr = const_cast<void *>(static_cast<const void *>(ctx->rh->get_ptr()));
    dev_type = kDLCUDA;
    dev_id = cuda_impl->get_device_id();
  } else if (backend == "cpu") {
    (void)consumer_stream;
    data_ptr = const_cast<void *>(static_cast<const void *>(msg.data.data()));
    dev_type = kDLCPU;
  } else {
    throw std::runtime_error(
            "torch_conversions::make_input_dlpack: unsupported backend '" +
            backend + "'");
  }

  auto * dlm = new DLManagedTensor;
  detail::populate_dl_tensor(*dlm, msg, *ctx, data_ptr, dev_type, dev_id);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = detail::dlpack_deleter;
  return dlm;
}

/// Build a DLManagedTensor for the output path.
/// For CUDA-backed `msg.data`, a WriteHandle is acquired on `consumer_stream`;
/// its destruction records the producer-side write event.
inline DLManagedTensor * make_output_dlpack(
  TensorMsg & msg,
  cudaStream_t consumer_stream = nullptr)
{
  auto ctx = std::make_unique<detail::DlpackContext>();
  ctx->shape.assign(msg.shape.begin(), msg.shape.end());
  ctx->strides.assign(msg.strides.begin(), msg.strides.end());

  void * data_ptr = nullptr;
  int32_t dev_type = kDLCPU;
  int32_t dev_id = 0;

  const std::string & backend = msg.data.get_backend_type();
  if (backend == "cuda") {
    auto * cuda_impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
        msg.data.get_impl()));
    if (!cuda_impl) {
      throw std::runtime_error(
              "torch_conversions::make_output_dlpack: cuda backend but not CudaBufferImpl");
    }
    cuda_impl->set_stream(consumer_stream);
    ctx->wh = std::make_shared<cuda_buffer_backend::WriteHandle>(
      cuda_impl->get_cuda_buffer().get_write_handle(consumer_stream));
    data_ptr = static_cast<void *>(ctx->wh->get_ptr());
    dev_type = kDLCUDA;
    dev_id = cuda_impl->get_device_id();
  } else if (backend == "cpu") {
    (void)consumer_stream;
    data_ptr = static_cast<void *>(msg.data.data());
    dev_type = kDLCPU;
  } else {
    throw std::runtime_error(
            "torch_conversions::make_output_dlpack: unsupported backend '" +
            backend + "'");
  }

  auto * dlm = new DLManagedTensor;
  detail::populate_dl_tensor(*dlm, msg, *ctx, data_ptr, dev_type, dev_id);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = detail::dlpack_deleter;
  return dlm;
}

#else  // TORCH_CONVERSIONS_HAS_CUDA

/// CPU-only build: only `backend == "cpu"` messages are supported.
inline DLManagedTensor * make_input_dlpack(const TensorMsg & msg)
{
  if (msg.data.get_backend_type() != "cpu") {
    throw std::runtime_error(
            "torch_conversions: CUDA not compiled in; cannot handle '" +
            msg.data.get_backend_type() + "' backend");
  }
  auto ctx = std::make_unique<detail::DlpackContext>();
  ctx->shape.assign(msg.shape.begin(), msg.shape.end());
  ctx->strides.assign(msg.strides.begin(), msg.strides.end());

  void * data_ptr = const_cast<void *>(static_cast<const void *>(msg.data.data()));

  auto * dlm = new DLManagedTensor;
  detail::populate_dl_tensor(*dlm, msg, *ctx, data_ptr, kDLCPU, 0);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = detail::dlpack_deleter;
  return dlm;
}

inline DLManagedTensor * make_output_dlpack(TensorMsg & msg)
{
  if (msg.data.get_backend_type() != "cpu") {
    throw std::runtime_error(
            "torch_conversions: CUDA not compiled in; cannot handle '" +
            msg.data.get_backend_type() + "' backend");
  }
  auto ctx = std::make_unique<detail::DlpackContext>();
  ctx->shape.assign(msg.shape.begin(), msg.shape.end());
  ctx->strides.assign(msg.strides.begin(), msg.strides.end());

  void * data_ptr = static_cast<void *>(msg.data.data());

  auto * dlm = new DLManagedTensor;
  detail::populate_dl_tensor(*dlm, msg, *ctx, data_ptr, kDLCPU, 0);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = detail::dlpack_deleter;
  return dlm;
}

#endif  // TORCH_CONVERSIONS_HAS_CUDA

inline void populate_image_dl_tensor(
  DLManagedTensor & dlm,
  const ImageMetadata & metadata,
  const DlpackContext & ctx,
  void * data_ptr,
  int32_t dev_type,
  int32_t dev_id)
{
  dlm.dl_tensor.data = data_ptr;
  dlm.dl_tensor.device = DLDevice{static_cast<DLDeviceType>(dev_type), dev_id};
  dlm.dl_tensor.ndim = static_cast<int32_t>(ctx.shape.size());
  dlm.dl_tensor.dtype = dl_dtype_from_scalar(metadata.dtype);
  dlm.dl_tensor.shape = const_cast<int64_t *>(ctx.shape.data());
  dlm.dl_tensor.strides = const_cast<int64_t *>(ctx.strides.data());
  dlm.dl_tensor.byte_offset = 0;
}

/// Export a read-only HWC DLPack view over Image.data.
inline DLManagedTensor * make_input_image_dlpack(
  const ImageMsg & msg
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  , cudaStream_t consumer_stream = nullptr
#endif
)
{
  const auto metadata = image_metadata(msg, "from_input_image_msg");
  auto ctx = std::make_unique<DlpackContext>();
  ctx->shape = metadata.shape;
  ctx->strides = metadata.strides;

  void * data_ptr = nullptr;
  int32_t dev_type = kDLCPU;
  int32_t dev_id = 0;
  const std::string & backend = msg.data.get_backend_type();
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  if (backend == "cuda") {
    const auto * cuda_impl =
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      msg.data.get_impl());
    if (!cuda_impl) {
      throw std::runtime_error(
              "torch_conversions::from_input_image_msg: cuda backend but not CudaBufferImpl");
    }
    ctx->rh = std::make_shared<cuda_buffer_backend::ReadHandle>(
      cuda_impl->get_cuda_buffer().get_read_handle(consumer_stream));
    data_ptr = const_cast<void *>(static_cast<const void *>(ctx->rh->get_ptr()));
    dev_type = kDLCUDA;
    dev_id = cuda_impl->get_device_id();
  } else  // NOLINT(readability/braces)
#endif
  if (backend == "cpu") {
    data_ptr = const_cast<void *>(static_cast<const void *>(msg.data.data()));
  } else {
    throw std::runtime_error(
            "torch_conversions::from_input_image_msg: unsupported backend " + backend);
  }

  auto * dlm = new DLManagedTensor;
  populate_image_dl_tensor(*dlm, metadata, *ctx, data_ptr, dev_type, dev_id);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = dlpack_deleter;
  return dlm;
}

/// Export a writable HWC DLPack view over Image.data.
inline DLManagedTensor * make_output_image_dlpack(
  ImageMsg & msg
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  , cudaStream_t consumer_stream = nullptr
#endif
)
{
  const auto metadata = image_metadata(msg, "from_output_image_msg");
  auto ctx = std::make_unique<DlpackContext>();
  ctx->shape = metadata.shape;
  ctx->strides = metadata.strides;

  void * data_ptr = nullptr;
  int32_t dev_type = kDLCPU;
  int32_t dev_id = 0;
  const std::string & backend = msg.data.get_backend_type();
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  if (backend == "cuda") {
    auto * cuda_impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
        msg.data.get_impl()));
    if (!cuda_impl) {
      throw std::runtime_error(
              "torch_conversions::from_output_image_msg: cuda backend but not CudaBufferImpl");
    }
    cuda_impl->set_stream(consumer_stream);
    ctx->wh = std::make_shared<cuda_buffer_backend::WriteHandle>(
      cuda_impl->get_cuda_buffer().get_write_handle(consumer_stream));
    data_ptr = static_cast<void *>(ctx->wh->get_ptr());
    dev_type = kDLCUDA;
    dev_id = cuda_impl->get_device_id();
  } else  // NOLINT(readability/braces)
#endif
  if (backend == "cpu") {
    data_ptr = static_cast<void *>(msg.data.data());
  } else {
    throw std::runtime_error(
            "torch_conversions::from_output_image_msg: unsupported backend " + backend);
  }

  auto * dlm = new DLManagedTensor;
  populate_image_dl_tensor(*dlm, metadata, *ctx, data_ptr, dev_type, dev_id);
  dlm->manager_ctx = ctx.release();
  dlm->deleter = dlpack_deleter;
  return dlm;
}

/// RAII wrapper for a DLManagedTensor. Useful when you're not immediately
/// handing the tensor off to a framework's `from_dlpack` (which would take
/// ownership itself). Calling `.release()` hands the raw pointer to such a
/// consumer.
struct DlpackDeleter
{
  void operator()(DLManagedTensor * p) const noexcept
  {
    if (p && p->deleter) {p->deleter(p);}
  }
};

using DlpackPtr = std::unique_ptr<DLManagedTensor, DlpackDeleter>;

inline DLManagedTensor * make_input_dlpack_current_stream(const TensorMsg & msg)
{
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  cudaStream_t s = nullptr;
  if (msg.data.get_backend_type() == "cuda") {
    s = current_cuda_stream_or_warn("from_input_tensor_msg");
  }
  return make_input_dlpack(msg, s);
#else
  return make_input_dlpack(msg);
#endif
}

inline DLManagedTensor * make_output_dlpack_current_stream(TensorMsg & msg)
{
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  cudaStream_t s = nullptr;
  if (msg.data.get_backend_type() == "cuda") {
    s = current_cuda_stream_or_warn("from_output_tensor_msg");
  }
  return make_output_dlpack(msg, s);
#else
  return make_output_dlpack(msg);
#endif
}


/// Populate shape / strides / dtype (plus byte_offset = 0) on `msg` from
/// a torch tensor, using `at::toDLPack` to derive the DLPack-form metadata.
/// Device fields on `msg` are intentionally NOT touched: they describe where
/// msg.data physically lives, which is fixed at allocation time and may
/// differ from the source tensor's device.
inline void populate_metadata_from_tensor(TensorMsg & msg, const at::Tensor & t)
{
  DLManagedTensor * dlm = at::toDLPack(t);
  const DLTensor & dt = dlm->dl_tensor;

  msg.shape.assign(dt.shape, dt.shape + dt.ndim);
  if (dt.strides) {
    msg.strides.assign(dt.strides, dt.strides + dt.ndim);
  } else {
    // DLPack null strides == row-major contiguous; materialize explicit ones.
    msg.strides = contiguous_strides(msg.shape);
  }
  msg.dtype_code = dt.dtype.code;
  msg.dtype_bits = dt.dtype.bits;
  msg.dtype_lanes = dt.dtype.lanes;
  msg.byte_offset = 0;

  if (dlm->deleter) {dlm->deleter(dlm);}
}


}  // namespace detail

/// RAII guard that sets a non-default CUDA stream for the current scope.
class StreamGuard
{
public:
  StreamGuard()
  : guard_(detail::select_torch_stream()) {}

private:
  c10::OptionalStreamGuard guard_;
};

inline StreamGuard set_stream() {return StreamGuard();}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

/// Allocate a Tensor message with metadata populated and a pre-sized
/// `data` buffer (CUDA-backed when available, else CPU). The storage is sized
/// exactly to hold `prod(shape) * dtype.bytesize` bytes and `byte_offset` is
/// left at zero; callers needing a view into larger storage can post-size
/// `msg.data` and set `byte_offset` manually.
inline std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::DeviceType> device = std::nullopt)
{
  c10::DeviceType dev = device.value_or(detail::default_device());
  detail::DLDataType dl = detail::dl_dtype_from_scalar(dtype);
  int64_t numel = detail::numel_of(shape);
  size_t byte_count = static_cast<size_t>(numel) * detail::dl_dtype_bytesize(dl);

  auto msg = std::make_unique<TensorMsg>();
  detail::set_dtype(*msg, dl);
  msg->shape.assign(shape.begin(), shape.end());
  auto strides = detail::contiguous_strides(shape);
  msg->strides.assign(strides.begin(), strides.end());
  msg->byte_offset = 0;

#ifdef TORCH_CONVERSIONS_HAS_CUDA
  if (dev == c10::kCUDA) {
    auto cuda_impl =
      std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(byte_count);
    msg->data = rosidl::Buffer<uint8_t>(std::move(cuda_impl));
    return msg;
  }
#endif
  if (dev == c10::kCPU) {
    msg->data.resize(byte_count);
    return msg;
  }
  throw std::runtime_error(
          "torch_conversions: unsupported device type " +
          std::to_string(static_cast<int>(dev)));
}


// ---------------------------------------------------------------------------
// sensor_msgs/Image allocation and conversion
// ---------------------------------------------------------------------------

/// Allocate a packed HWC Image with a pre-sized data buffer. The metadata is
/// initialized completely, and data uses CUDA storage when CUDA is selected.
inline std::unique_ptr<ImageMsg> allocate_image_msg(
  uint32_t height,
  uint32_t width,
  const std::string & encoding,
  std::optional<c10::DeviceType> device = std::nullopt)
{
  if (encoding.empty() || sensor_msgs::image_encodings::isPlanar(encoding)) {
    throw std::runtime_error(
            "torch_conversions::allocate_image_msg: encoding must be a packed byte encoding");
  }
  int channels;
  int bits;
  try {
    channels = sensor_msgs::image_encodings::numChannels(encoding);
    bits = sensor_msgs::image_encodings::bitDepth(encoding);
  } catch (const std::runtime_error & error) {
    throw std::runtime_error(
            std::string("torch_conversions::allocate_image_msg: ") + error.what());
  }
  if (channels <= 0 || bits != 8) {
    throw std::runtime_error(
            "torch_conversions::allocate_image_msg: only byte-oriented packed "
            "encodings are supported");
  }

  const size_t step = detail::checked_multiply(
    static_cast<size_t>(width), static_cast<size_t>(channels), "allocate_image_msg");
  if (step > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
            "torch_conversions::allocate_image_msg: packed row size exceeds Image.step");
  }
  const size_t byte_count = detail::checked_multiply(
    static_cast<size_t>(height), step, "allocate_image_msg");

  auto msg = std::make_unique<ImageMsg>();
  msg->height = height;
  msg->width = width;
  msg->encoding = encoding;
  msg->is_bigendian = 0;
  msg->step = static_cast<uint32_t>(step);

  const c10::DeviceType dev = device.value_or(detail::default_device());
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  if (dev == c10::kCUDA) {
    auto cuda_impl =
      std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(byte_count);
    msg->data = rosidl::Buffer<uint8_t>(std::move(cuda_impl));
    return msg;
  }
#endif
  if (dev == c10::kCPU) {
    msg->data.resize(byte_count);
    return msg;
  }
  throw std::runtime_error(
          "torch_conversions::allocate_image_msg: unsupported device type " +
          std::to_string(static_cast<int>(dev)));
}

/// Return a writable HWC tensor view over Image.data. The CUDA WriteHandle
/// remains alive until the returned tensor is destroyed.
inline at::Tensor from_output_image_msg(ImageMsg & msg)
{
  if (msg.data.empty() && msg.height == 0 && msg.width == 0) {return {};}
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  cudaStream_t stream = nullptr;
  if (msg.data.get_backend_type() == "cuda") {
    stream = detail::current_cuda_stream_or_warn("from_output_image_msg");
  }
  detail::DlpackPtr guard{detail::make_output_image_dlpack(msg, stream)};
#else
  detail::DlpackPtr guard{detail::make_output_image_dlpack(msg)};
#endif
  at::Tensor tensor = at::fromDLPack(guard.get());
  (void)guard.release();
  return tensor;
}

/// Return an HWC tensor for subscriber input. clone=false is a zero-copy,
/// read-only view whose CUDA ReadHandle remains alive with the tensor.
inline at::Tensor from_input_image_msg(const ImageMsg & msg, bool clone = true)
{
  if (msg.data.empty() && msg.height == 0 && msg.width == 0) {return {};}
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  cudaStream_t stream = nullptr;
  if (msg.data.get_backend_type() == "cuda") {
    stream = detail::current_cuda_stream_or_warn("from_input_image_msg");
  }
  detail::DlpackPtr guard{detail::make_input_image_dlpack(msg, stream)};
#else
  detail::DlpackPtr guard{detail::make_input_image_dlpack(msg)};
#endif
  at::Tensor tensor = at::fromDLPack(guard.get());
  (void)guard.release();
  return clone ? tensor.clone() : tensor;
}

/// Copy an HWC tensor into an existing Image allocation. Existing header and
/// image metadata are preserved; the tensor must match them exactly.
inline void to_image_msg(ImageMsg & msg, const at::Tensor & tensor)
{
  if (!tensor.defined()) {
    throw std::runtime_error("torch_conversions::to_image_msg: tensor is undefined");
  }
  const auto metadata = detail::image_metadata(msg, "to_image_msg");
  if (tensor.dim() != 3 ||
    !std::equal(tensor.sizes().begin(), tensor.sizes().end(), metadata.shape.begin()))
  {
    throw std::runtime_error(
            "torch_conversions::to_image_msg: tensor shape must match Image HWC metadata");
  }
  if (tensor.scalar_type() != metadata.dtype) {
    throw std::runtime_error(
            "torch_conversions::to_image_msg: tensor dtype does not match Image encoding");
  }
  at::Tensor output = from_output_image_msg(msg);
  // Keep CUDA work stream-ordered. The WriteHandle records the event when
  // output leaves scope; no device or stream synchronization is performed here.
  output.copy_(tensor, /*non_blocking=*/ true);
}

/// Allocate an Image matching an HWC tensor and copy the tensor into it.
inline std::unique_ptr<ImageMsg> to_image_msg(
  const at::Tensor & tensor, const std::string & encoding)
{
  if (!tensor.defined() || tensor.dim() != 3) {
    throw std::runtime_error(
            "torch_conversions::to_image_msg: tensor must be a defined HWC tensor");
  }
  if (tensor.size(0) < 0 || tensor.size(0) > std::numeric_limits<uint32_t>::max() ||
    tensor.size(1) < 0 || tensor.size(1) > std::numeric_limits<uint32_t>::max())
  {
    throw std::runtime_error(
            "torch_conversions::to_image_msg: tensor dimensions exceed Image metadata");
  }
  auto msg = allocate_image_msg(
    static_cast<uint32_t>(tensor.size(0)),
    static_cast<uint32_t>(tensor.size(1)),
    encoding,
    tensor.device().type());
  to_image_msg(*msg, tensor);
  return msg;
}


// ---------------------------------------------------------------------------
// at::Tensor <-> TensorMsg
// ---------------------------------------------------------------------------

/// Get a writable at::Tensor view over msg.data + msg.byte_offset.
/// Use on the publisher side: fill the returned tensor in place and then
/// publish msg. The view shares memory with msg.data; the caller must
/// ensure msg outlives the returned tensor.
inline at::Tensor from_output_tensor_msg(TensorMsg & msg)
{
  if (msg.data.empty()) {return {};}
  detail::DlpackPtr guard{detail::make_output_dlpack_current_stream(msg)};
  at::Tensor t = at::fromDLPack(guard.get());
  (void)guard.release();
  return t;
}

/// Get a read-only at::Tensor from msg.data + msg.byte_offset.
/// Use on the subscriber side.
/// \param clone If true (default), returns an independent copy safe to
/// mutate. If false, returns a zero-copy view that keeps a ReadHandle
/// alive for the tensor's lifetime; caller must treat the view as
/// read-only.
inline at::Tensor from_input_tensor_msg(const TensorMsg & msg, bool clone = true)
{
  if (msg.data.empty()) {return {};}
  detail::DlpackPtr guard{detail::make_input_dlpack_current_stream(msg)};
  at::Tensor t = at::fromDLPack(guard.get());
  (void)guard.release();
  return clone ? t.clone() : t;
}


/// Copy `tensor` into msg.data (pre-allocated by the selected buffer backend)
/// and refresh msg metadata to the contiguous form.
/// `byte_offset` is reset to 0; shape/strides/dtype are overwritten.
/// Device placement is derived from msg.data's backend when exported to DLPack.
inline void to_tensor_msg(TensorMsg & msg, const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }
  at::Tensor contig = tensor.contiguous();
  size_t byte_count = contig.numel() * contig.element_size();

  if (byte_count > msg.data.size()) {
    throw std::runtime_error(
            "torch_conversions::to_tensor_msg: tensor size (" +
            std::to_string(byte_count) + " bytes) exceeds allocated buffer (" +
            std::to_string(msg.data.size()) + " bytes)");
  }

  const std::string & backend = msg.data.get_backend_type();
#ifdef TORCH_CONVERSIONS_HAS_CUDA
  if (backend == "cuda") {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto wh = cuda_buffer_backend::from_output_buffer(msg.data, stream);
    cudaMemcpyKind kind = contig.is_cuda() ?
      cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cuda_buffer_backend::to_buffer(
      contig.data_ptr(), byte_count, wh, stream, kind);
  } else  // NOLINT(readability/braces)
#endif
  if (backend == "cpu") {
    at::Tensor cpu_contig = contig.to(torch::kCPU).contiguous();
    std::memcpy(msg.data.data(), cpu_contig.data_ptr(), byte_count);
  } else {
    throw std::runtime_error(
            "torch_conversions::to_tensor_msg: unsupported backend '" + backend + "'");
  }

  detail::populate_metadata_from_tensor(msg, contig);
}

/// Allocate a TensorMsg for `tensor`, copy the tensor contents into it, and
/// populate shape / strides / dtype metadata.
inline std::unique_ptr<TensorMsg> to_tensor_msg(const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return std::make_unique<TensorMsg>();
  }
  at::Tensor contig = tensor.contiguous();
  std::vector<int64_t> shape(contig.sizes().begin(), contig.sizes().end());
  auto msg = allocate_tensor_msg(
    shape, contig.scalar_type(), contig.device().type());
  to_tensor_msg(*msg, contig);
  return msg;
}

}  // namespace torch_conversions

#endif  // TORCH_CONVERSIONS__TORCH_CONVERSIONS_HPP_
