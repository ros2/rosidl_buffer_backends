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

#include "torch_conversions/torch_conversions.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/dlpack.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pluginlib/class_loader.hpp>

#include "torch_conversions/conversion_plugin.hpp"

namespace torch_conversions
{
namespace
{

struct DlpackContext
{
  StorageView view;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

struct ConversionPluginHolder
{
  ConversionPluginHolder()
  : loader("torch_conversions", "torch_conversions::ConversionPlugin")
  {
    const auto classes = loader.getDeclaredClasses();
    if (classes.size() != 1) {
      throw std::runtime_error(
              "torch_conversions expected exactly one conversion plugin, found " +
              std::to_string(classes.size()));
    }
#if defined(__linux__)
    const std::string library_path = loader.getClassLibraryPath(classes.front());
    pinned_library = dlopen(
      library_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!pinned_library) {
      throw std::runtime_error(
              "torch_conversions failed to pin plugin library '" +
              library_path + "': " + dlerror());
    }
#endif
    plugin = loader.createSharedInstance(classes.front());
  }

  pluginlib::ClassLoader<ConversionPlugin> loader;
  std::shared_ptr<ConversionPlugin> plugin;
#if defined(__linux__)
  void * pinned_library{nullptr};
#endif
};

std::shared_ptr<ConversionPlugin> conversion_plugin()
{
  static ConversionPluginHolder holder;
  return holder.plugin;
}

DLDataType dl_dtype(at::ScalarType type)
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

std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (auto index = shape.size(); index > 0; --index) {
    strides[index - 1] = stride;
    stride *= shape[index - 1];
  }
  return strides;
}

size_t storage_size(
  const std::vector<int64_t> & shape,
  const DLDataType & dtype)
{
  size_t elements = 1;
  for (const auto dimension : shape) {
    if (dimension < 0) {
      throw std::runtime_error("torch_conversions: negative shape dimension");
    }
    elements *= static_cast<size_t>(dimension);
  }
  return elements * ((static_cast<size_t>(dtype.bits) * dtype.lanes + 7) / 8);
}

DeviceKind device_kind(c10::DeviceType type)
{
  if (type == c10::kCPU) {
    return DeviceKind::cpu;
  }
  if (type == c10::kCUDA) {
    return DeviceKind::cuda;
  }
  throw std::runtime_error("torch_conversions: unsupported device type");
}

uintptr_t current_stream()
{
  return reinterpret_cast<uintptr_t>(
    at::cuda::getCurrentCUDAStream().stream());
}

void dlpack_deleter(DLManagedTensor * tensor)
{
  if (tensor != nullptr) {
    delete static_cast<DlpackContext *>(tensor->manager_ctx);
    delete tensor;
  }
}

DLManagedTensor * make_dlpack(const TensorMsg & msg, StorageView view)
{
  auto context = std::make_unique<DlpackContext>();
  context->view = std::move(view);
  context->shape.assign(msg.shape.begin(), msg.shape.end());
  context->strides.assign(msg.strides.begin(), msg.strides.end());

  auto tensor = std::make_unique<DLManagedTensor>();
  tensor->manager_ctx = context.get();
  tensor->deleter = dlpack_deleter;
  tensor->dl_tensor.data =
    static_cast<uint8_t *>(context->view.data) + msg.byte_offset;
  tensor->dl_tensor.device = {
    static_cast<DLDeviceType>(context->view.dl_device_type),
    context->view.device_id,
  };
  tensor->dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
  tensor->dl_tensor.dtype = {
    msg.dtype_code,
    msg.dtype_bits,
    msg.dtype_lanes,
  };
  tensor->dl_tensor.shape = context->shape.data();
  tensor->dl_tensor.strides =
    context->strides.empty() ? nullptr : context->strides.data();
  tensor->dl_tensor.byte_offset = 0;
  context.release();
  return tensor.release();
}

void set_metadata(TensorMsg & msg, const at::Tensor & tensor)
{
  const auto dtype = dl_dtype(tensor.scalar_type());
  msg.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
  msg.strides.assign(tensor.strides().begin(), tensor.strides().end());
  msg.dtype_code = dtype.code;
  msg.dtype_bits = dtype.bits;
  msg.dtype_lanes = dtype.lanes;
  msg.byte_offset = 0;
}

}  // namespace

class StreamGuard::Impl
{
public:
  Impl()
  : guard_(torch::cuda::is_available() ?
      std::optional<c10::Stream>(c10::cuda::getStreamFromPool()) :
      std::nullopt) {}

private:
  c10::OptionalStreamGuard guard_;
};

StreamGuard::StreamGuard()
: impl_(std::make_unique<Impl>()) {}

StreamGuard::~StreamGuard() = default;
StreamGuard::StreamGuard(StreamGuard &&) noexcept = default;
StreamGuard & StreamGuard::operator=(StreamGuard &&) noexcept = default;

StreamGuard set_stream()
{
  return StreamGuard();
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::DeviceType> device)
{
  auto plugin = conversion_plugin();
  const DeviceKind selected =
    device ? device_kind(*device) : plugin->default_device();
  if (!plugin->device_available(selected)) {
    throw std::runtime_error(
            "torch_conversions: requested CUDA device is unavailable");
  }
  auto msg = std::make_unique<TensorMsg>();
  const auto encoded_dtype = dl_dtype(dtype);
  msg->shape.assign(shape.begin(), shape.end());
  msg->strides = contiguous_strides(shape);
  msg->dtype_code = encoded_dtype.code;
  msg->dtype_bits = encoded_dtype.bits;
  msg->dtype_lanes = encoded_dtype.lanes;
  msg->byte_offset = 0;
  plugin->allocate(*msg, storage_size(shape, encoded_dtype), selected);
  return msg;
}

at::Tensor from_output_tensor_msg(TensorMsg & msg)
{
  if (msg.data.empty()) {
    return {};
  }
  const auto stream =
    msg.data.get_backend_type() == "cuda" ? current_stream() : 0;
  return at::fromDLPack(make_dlpack(
      msg, conversion_plugin()->acquire_output(msg, stream)));
}

at::Tensor from_input_tensor_msg(const TensorMsg & msg, bool clone)
{
  if (msg.data.empty()) {
    return {};
  }
  const auto stream =
    msg.data.get_backend_type() == "cuda" ? current_stream() : 0;
  auto tensor = at::fromDLPack(make_dlpack(
      msg, conversion_plugin()->acquire_input(msg, stream)));
  return clone ? tensor.clone() : tensor;
}

void to_tensor_msg(TensorMsg & msg, const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }
  auto contiguous = tensor.contiguous();
  const size_t byte_count =
    static_cast<size_t>(contiguous.numel()) * contiguous.element_size();
  if (byte_count > msg.data.size()) {
    throw std::runtime_error(
            "torch_conversions: tensor exceeds allocated message storage");
  }
  const bool needs_cuda_stream =
    contiguous.is_cuda() || msg.data.get_backend_type() == "cuda";
  conversion_plugin()->copy_to(
    msg,
    contiguous.data_ptr(),
    byte_count,
    device_kind(contiguous.device().type()),
    needs_cuda_stream ? current_stream() : 0);
  set_metadata(msg, contiguous);
}

std::unique_ptr<TensorMsg> to_tensor_msg(const at::Tensor & tensor)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return std::make_unique<TensorMsg>();
  }
  auto contiguous = tensor.contiguous();
  std::vector<int64_t> shape(
    contiguous.sizes().begin(), contiguous.sizes().end());
  auto msg = allocate_tensor_msg(
    shape, contiguous.scalar_type(), contiguous.device().type());
  to_tensor_msg(*msg, contiguous);
  return msg;
}

}  // namespace torch_conversions
