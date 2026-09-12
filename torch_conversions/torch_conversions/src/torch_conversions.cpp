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

#include <ATen/dlpack.h>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <limits>
#include <map>
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

void pin_library(const std::string & path)
{
#if defined(__linux__)
  if (dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) == nullptr) {
    throw std::runtime_error(
            "torch_conversions failed to pin plugin library '" +
            path + "': " + dlerror());
  }
#else
  (void)path;
#endif
}

class Registry
{
public:
  static Registry & instance()
  {
    static Registry registry;
    return registry;
  }

  std::shared_ptr<ConversionPlugin> for_backend(
    const std::string & backend) const
  {
    const auto found = backends_.find(backend);
    if (found == backends_.end()) {
      throw std::runtime_error(
              "torch_conversions: no plugin provides backend '" +
              backend + "'; available backends are " + describe());
    }
    return found->second;
  }

  std::shared_ptr<ConversionPlugin> for_device(
    c10::DeviceType device_type) const
  {
    const auto found = devices_.find(device_type);
    if (found == devices_.end()) {
      throw std::runtime_error(
              "torch_conversions: no plugin serves Torch device type " +
              std::to_string(static_cast<int>(device_type)) +
              "; available backends are " + describe());
    }
    return found->second;
  }

  std::vector<std::string> names() const
  {
    std::vector<std::string> result;
    result.reserve(backends_.size());
    for (const auto & entry : backends_) {
      result.push_back(entry.first);
    }
    return result;
  }

  std::string default_name() const
  {
    const char * requested = std::getenv("ROSIDL_TENSOR_BACKEND");
    if (requested != nullptr && *requested != '\0') {
      (void)for_backend(requested);
      return requested;
    }

    std::string selected;
    int selected_priority = std::numeric_limits<int>::min();
    for (const auto & entry : backends_) {
      const int candidate_priority = entry.second->priority();
      if (selected.empty() || candidate_priority > selected_priority) {
        selected = entry.first;
        selected_priority = candidate_priority;
      }
    }
    if (selected.empty()) {
      throw std::runtime_error(
              "torch_conversions: no conversion plugin is installed");
    }
    return selected;
  }

  bool has(const std::string & backend) const
  {
    return backends_.count(backend) != 0;
  }

private:
  Registry()
  : loader_("torch_conversions", "torch_conversions::ConversionPlugin")
  {
    for (const auto & class_name : loader_.getDeclaredClasses()) {
      try {
        pin_library(loader_.getClassLibraryPath(class_name));
        auto plugin = loader_.createSharedInstance(class_name);
        if (!plugin->available()) {
          continue;
        }
        if (backends_.emplace(plugin->backend(), plugin).second) {
          devices_.emplace(plugin->device_type(), plugin);
        }
      } catch (const std::exception &) {
        // An independently installed accelerator plugin may lack its runtime.
      }
    }
  }

  std::string describe() const
  {
    std::string result;
    for (const auto & entry : backends_) {
      result += result.empty() ? entry.first : ", " + entry.first;
    }
    return result.empty() ? "[none installed]" : result;
  }

  pluginlib::ClassLoader<ConversionPlugin> loader_;
  std::map<std::string, std::shared_ptr<ConversionPlugin>> backends_;
  std::map<c10::DeviceType, std::shared_ptr<ConversionPlugin>> devices_;
};

struct DtypeDescription
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

DtypeDescription describe_dtype(at::ScalarType type)
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
      throw std::runtime_error(
              "torch_conversions: unsupported scalar type");
  }
}

size_t element_count(const std::vector<int64_t> & shape)
{
  size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension < 0) {
      throw std::runtime_error(
              "torch_conversions: negative shape dimension");
    }
    if (dimension != 0 && count >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension))
    {
      throw std::overflow_error("torch_conversions: tensor size overflow");
    }
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> result(shape.size());
  int64_t stride = 1;
  for (auto index = shape.size(); index > 0; --index) {
    const auto dimension = shape[index - 1];
    if (dimension < 0) {
      throw std::runtime_error(
              "torch_conversions: negative shape dimension");
    }
    result[index - 1] = stride;
    if (dimension != 0 && stride >
      std::numeric_limits<int64_t>::max() / dimension)
    {
      throw std::overflow_error("torch_conversions: tensor stride overflow");
    }
    stride *= dimension;
  }
  return result;
}

size_t byte_count(
  const std::vector<int64_t> & shape, at::ScalarType dtype)
{
  const size_t count = element_count(shape);
  const size_t item_size = c10::elementSize(dtype);
  if (count != 0 && item_size >
    std::numeric_limits<size_t>::max() / count)
  {
    throw std::overflow_error("torch_conversions: tensor size overflow");
  }
  return count * item_size;
}

void validate_view(const TensorMsg & msg)
{
  const auto strides = normalized_strides(msg);
  size_t span = 1;
  for (size_t index = 0; index < msg.shape.size(); ++index) {
    if (msg.shape[index] < 0 || strides[index] < 0) {
      throw std::runtime_error(
              "torch_conversions: negative shape or stride");
    }
    if (msg.shape[index] == 0) {
      span = 0;
      break;
    }
    const size_t extent = static_cast<size_t>(msg.shape[index] - 1);
    const size_t stride = static_cast<size_t>(strides[index]);
    if (stride != 0 && extent >
      std::numeric_limits<size_t>::max() / stride)
    {
      throw std::overflow_error("torch_conversions: tensor span overflow");
    }
    const size_t contribution = extent * stride;
    if (span > std::numeric_limits<size_t>::max() - contribution) {
      throw std::overflow_error("torch_conversions: tensor span overflow");
    }
    span += contribution;
  }

  const size_t item_size = c10::elementSize(scalar_type(msg));
  if (span != 0 && item_size >
    std::numeric_limits<size_t>::max() / span)
  {
    throw std::overflow_error("torch_conversions: tensor span overflow");
  }
  const size_t bytes = span * item_size;
  if (msg.byte_offset > std::numeric_limits<size_t>::max() - bytes) {
    throw std::overflow_error("torch_conversions: tensor span overflow");
  }
  const size_t required = msg.byte_offset + bytes;
  if (required > msg.data.size()) {
    throw std::runtime_error(
            "torch_conversions: tensor view exceeds message storage");
  }
}

void set_metadata(TensorMsg & msg, const at::Tensor & tensor)
{
  const auto dtype = describe_dtype(tensor.scalar_type());
  msg.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
  msg.strides = contiguous_strides(msg.shape);
  msg.dtype_code = dtype.code;
  msg.dtype_bits = dtype.bits;
  msg.dtype_lanes = dtype.lanes;
  msg.byte_offset = 0;
}

}  // namespace

std::vector<std::string> available_backends()
{
  return Registry::instance().names();
}

bool backend_available(const std::string & backend)
{
  return Registry::instance().has(backend);
}

std::string backend_for_device(c10::DeviceType device_type)
{
  return Registry::instance().for_device(device_type)->backend();
}

std::string default_backend()
{
  return Registry::instance().default_name();
}

StreamGuard::StreamGuard(std::optional<c10::Device> device)
{
  auto & registry = Registry::instance();
  auto plugin = device ? registry.for_device(device->type()) :
    registry.for_backend(registry.default_name());
  const auto stream = plugin->select_stream(device.value_or(c10::Device(plugin->device_type())));
  if (stream) {
    guard_.reset_stream(*stream);
  }
}

at::ScalarType scalar_type(const TensorMsg & msg)
{
  if (msg.dtype_lanes != 1) {
    throw std::runtime_error(
            "torch_conversions: vector-lane dtypes are unsupported");
  }
  if (msg.dtype_code == kDLUInt && msg.dtype_bits == 8) {
    return at::kByte;
  }
  if (msg.dtype_code == kDLInt) {
    switch (msg.dtype_bits) {
      case 8: return at::kChar;
      case 16: return at::kShort;
      case 32: return at::kInt;
      case 64: return at::kLong;
      default: break;
    }
  }
  if (msg.dtype_code == kDLFloat) {
    switch (msg.dtype_bits) {
      case 16: return at::kHalf;
      case 32: return at::kFloat;
      case 64: return at::kDouble;
      default: break;
    }
  }
  if (msg.dtype_code == kDLBfloat && msg.dtype_bits == 16) {
    return at::kBFloat16;
  }
  if (msg.dtype_code == kDLBool && msg.dtype_bits == 8) {
    return at::kBool;
  }
  throw std::runtime_error(
          "torch_conversions: unsupported message dtype");
}

std::vector<int64_t> normalized_strides(const TensorMsg & msg)
{
  if (msg.strides.empty()) {
    return contiguous_strides(
      std::vector<int64_t>(msg.shape.begin(), msg.shape.end()));
  }
  if (msg.strides.size() != msg.shape.size()) {
    throw std::runtime_error(
            "torch_conversions: strides must match shape rank");
  }
  return {msg.strides.begin(), msg.strides.end()};
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  at::ScalarType dtype,
  std::optional<c10::Device> device)
{
  auto & registry = Registry::instance();
  std::shared_ptr<ConversionPlugin> plugin;
  c10::Device selected_device(c10::kCPU);
  if (device) {
    selected_device = *device;
    plugin = registry.for_device(device->type());
  } else {
    plugin = registry.for_backend(registry.default_name());
    selected_device = c10::Device(plugin->device_type());
  }

  auto msg = std::make_unique<TensorMsg>();
  const auto strides = contiguous_strides(shape);
  const auto dtype_description = describe_dtype(dtype);
  msg->shape.assign(shape.begin(), shape.end());
  msg->strides.assign(strides.begin(), strides.end());
  msg->dtype_code = dtype_description.code;
  msg->dtype_bits = dtype_description.bits;
  msg->dtype_lanes = dtype_description.lanes;
  msg->byte_offset = 0;
  plugin->allocate(*msg, byte_count(shape, dtype), selected_device);
  return msg;
}

at::Tensor from_output_tensor_msg(TensorMsg & msg, void * execution_stream)
{
  if (msg.data.empty()) {
    return {};
  }
  validate_view(msg);
  return Registry::instance().for_backend(
    msg.data.get_backend_type())->from_output(msg, execution_stream);
}

at::Tensor from_input_tensor_msg(
  const TensorMsg & msg, bool clone, void * execution_stream)
{
  if (msg.data.empty()) {
    return {};
  }
  validate_view(msg);
  return Registry::instance().for_backend(
    msg.data.get_backend_type())->from_input(msg, clone, execution_stream);
}

void to_tensor_msg(
  TensorMsg & msg, const at::Tensor & tensor, void * execution_stream)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }
  const size_t required = tensor.nbytes();
  if (required > msg.data.size()) {
    throw std::runtime_error(
            "torch_conversions: tensor exceeds message storage");
  }

  auto & registry = Registry::instance();
  auto plugin = tensor.device().is_cpu() ?
    registry.for_backend(msg.data.get_backend_type()) :
    registry.for_device(tensor.device().type());
  plugin->copy_to(msg, tensor, execution_stream);
  set_metadata(msg, tensor);
}

std::unique_ptr<TensorMsg> to_tensor_msg(
  const at::Tensor & tensor, void * execution_stream)
{
  if (!tensor.defined() || tensor.numel() == 0) {
    return std::make_unique<TensorMsg>();
  }
  auto msg = allocate_tensor_msg(
    std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end()),
    tensor.scalar_type(), tensor.device());
  to_tensor_msg(*msg, tensor, execution_stream);
  return msg;
}

}  // namespace torch_conversions
