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

#ifndef TORCH_CONVERSIONS__DETAIL__TORCH_CONVERSIONS_IMPL_HPP_
#define TORCH_CONVERSIONS__DETAIL__TORCH_CONVERSIONS_IMPL_HPP_

#include "torch_conversions/torch_conversions.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pluginlib/class_loader.hpp>

#include "torch_conversions/conversion_plugin.hpp"

namespace torch_conversions
{
namespace detail
{

// Defined by the runtime package including this header, together with
// StreamGuard::Impl and the StreamGuard special member functions.
uintptr_t current_stream();

}  // namespace detail

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

std::string backend_list(const ConversionPlugin & plugin)
{
  std::string names;
  for (const auto & backend : plugin.backends()) {
    names += names.empty() ? backend : ", " + backend;
  }
  return names;
}

// at::torchDeviceToDLDevice resolves the accelerator flavour of the LibTorch
// build, so a ROCm build reports kDLROCM even though its tensors are kCUDA.
std::string backend_for(c10::Device device, const ConversionPlugin & plugin)
{
  const auto dl_device = at::torchDeviceToDLDevice(device);
  const std::string backend =
    plugin.backend_for_device(static_cast<int32_t>(dl_device.device_type));
  if (backend.empty()) {
    throw std::runtime_error(
            "torch_conversions: no backend for device '" + device.str() +
            "'; the installed plugin provides " + backend_list(plugin));
  }
  return backend;
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

uintptr_t stream_for(const TensorMsg & msg)
{
  return msg.data.get_backend_type() == "cpu" ? 0 : detail::current_stream();
}

}  // namespace

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
  const std::string selected = device ?
    backend_for(c10::Device(*device, 0), *plugin) :
    plugin->default_backend();
  if (!plugin->backend_available(selected)) {
    throw std::runtime_error(
            "torch_conversions: backend '" + selected + "' is unavailable");
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
  return at::fromDLPack(make_dlpack(
      msg, conversion_plugin()->acquire_output(msg, stream_for(msg))));
}

at::Tensor from_input_tensor_msg(const TensorMsg & msg, bool clone)
{
  if (msg.data.empty()) {
    return {};
  }
  auto tensor = at::fromDLPack(make_dlpack(
      msg, conversion_plugin()->acquire_input(msg, stream_for(msg))));
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
  auto plugin = conversion_plugin();
  const std::string source_backend = backend_for(contiguous.device(), *plugin);
  const bool needs_stream =
    source_backend != "cpu" || msg.data.get_backend_type() != "cpu";
  plugin->copy_to(
    msg,
    contiguous.data_ptr(),
    byte_count,
    source_backend,
    needs_stream ? detail::current_stream() : 0);
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

#endif  // TORCH_CONVERSIONS__DETAIL__TORCH_CONVERSIONS_IMPL_HPP_
