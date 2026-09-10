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
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// ATen/dlpack.h above wins the shared DLPACK_DLPACK_H_ include guard, so this
// translation unit and the core agree on DLManagedTensor. dlpack_conversions.hpp
// static_asserts the layout, which fails the build on a real version skew.
#include "dlpack_conversions/dlpack_conversions.hpp"
#include "torch_conversions/detail/stream.hpp"

namespace torch_conversions
{

using TensorMsg = dlpack_conversions::TensorMsg;

namespace detail
{

inline DLDataType dl_dtype(at::ScalarType type)
{
  switch (type) {
    case at::kByte: return {kDLUInt, 8, 1};
    case at::kChar: return {kDLInt, 8, 1};
    case at::kShort: return {kDLInt, 16, 1};
    case at::kInt: return {kDLInt, 32, 1};
    case at::kLong: return {kDLInt, 64, 1};
    case at::kHalf: return {kDLFloat, 16, 1};
    case at::kBFloat16: return {kDLBfloat, 16, 1};
    case at::kFloat: return {kDLFloat, 32, 1};
    case at::kDouble: return {kDLFloat, 64, 1};
    case at::kBool: return {kDLBool, 8, 1};
    default:
      throw std::runtime_error("torch_conversions: unsupported scalar type");
  }
}

// at::torchDeviceToDLDevice resolves the accelerator flavour of the LibTorch
// build, so a ROCm build reports kDLROCM even though its tensors are kCUDA.
inline std::string backend_for(c10::Device device)
{
  const auto dl_device = at::torchDeviceToDLDevice(device);
  std::string backend = dlpack_conversions::backend_for_device(
    static_cast<int32_t>(dl_device.device_type));
  if (backend.empty()) {
    throw std::runtime_error(
            "torch_conversions: no storage plugin serves device '" +
            device.str() + "'");
  }
  return backend;
}

inline DLTensor as_dl_tensor(const at::Tensor & tensor)
{
  DLTensor source{};
  source.data = tensor.data_ptr();
  source.device = at::torchDeviceToDLDevice(tensor.device());
  source.ndim = static_cast<int32_t>(tensor.dim());
  source.dtype = dl_dtype(tensor.scalar_type());
  source.shape = const_cast<int64_t *>(tensor.sizes().data());
  source.strides = const_cast<int64_t *>(tensor.strides().data());
  source.byte_offset = 0;
  return source;
}

inline uintptr_t stream_for(const TensorMsg & msg)
{
  return msg.data.get_backend_type() == "cpu" ? 0 : current_stream();
}

inline uintptr_t stream_for(const TensorMsg & msg, const at::Tensor & source)
{
  const bool host_only =
    source.device().is_cpu() && msg.data.get_backend_type() == "cpu";
  return host_only ? 0 : current_stream();
}

// Storage plugins and the LibTorch build are installed independently, so an
// accelerator plugin can be present alongside a LibTorch that has no kernels
// for it. Handing that storage back would only fail later inside ATen, so an
// unusable default falls back to host storage.
inline std::string default_backend()
{
  const std::string backend = dlpack_conversions::default_backend();
  const bool needs_accelerator = backend != "cpu";
  if (!needs_accelerator || at::hasCUDA() || at::hasHIP()) {
    return backend;
  }
  return dlpack_conversions::backend_available("cpu") ?
         std::string{"cpu"} : backend;
}

}  // namespace detail

inline std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::DeviceType> device = std::nullopt)
{
  const std::string backend = device ?
    detail::backend_for(c10::Device(*device, 0)) : detail::default_backend();
  return dlpack_conversions::allocate_tensor_msg(
    shape, detail::dl_dtype(dtype), backend);
}

inline at::Tensor from_output_tensor_msg(TensorMsg & msg)
{
  auto managed = dlpack_conversions::from_output_tensor_msg(
    msg, detail::stream_for(msg));
  if (!managed) {
    return {};
  }
  return at::fromDLPack(managed.release());
}

inline at::Tensor from_input_tensor_msg(const TensorMsg & msg, bool clone = true)
{
  auto managed = dlpack_conversions::from_input_tensor_msg(
    msg, detail::stream_for(msg));
  if (!managed) {
    return {};
  }
  auto tensor = at::fromDLPack(managed.release());
  return clone ? tensor.clone() : tensor;
}

inline void to_tensor_msg(TensorMsg & msg, const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }
  const auto contiguous = tensor.contiguous();
  dlpack_conversions::to_tensor_msg(
    msg,
    detail::as_dl_tensor(contiguous),
    detail::stream_for(msg, contiguous));
}

inline std::unique_ptr<TensorMsg> to_tensor_msg(const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return std::make_unique<TensorMsg>();
  }
  const auto contiguous = tensor.contiguous();
  return dlpack_conversions::to_tensor_msg(
    detail::as_dl_tensor(contiguous),
    contiguous.device().is_cpu() ? 0 : detail::current_stream());
}

}  // namespace torch_conversions

#endif  // TORCH_CONVERSIONS__TORCH_CONVERSIONS_HPP_
