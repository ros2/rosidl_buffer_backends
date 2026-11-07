// Copyright 2026 Open Source Robotics Foundation, Inc.
// Copyright 2026 Meta Platforms, Inc. and affiliates.
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

#include "executorch_conversions/executorch_conversions.hpp"

#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include <pluginlib/class_loader.hpp>

#include "executorch_conversions/conversion_plugin.hpp"

namespace executorch_conversions
{
namespace
{

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;
using executorch::extension::TensorPtr;

// DLPack's DLDataTypeCode values, as used by tensor_msgs/ExperimentalTensor.
constexpr uint8_t kDLInt = 0;
constexpr uint8_t kDLUInt = 1;
constexpr uint8_t kDLFloat = 2;
constexpr uint8_t kDLBfloat = 4;
constexpr uint8_t kDLBool = 6;

void pin_library(const std::string & path)
{
#if defined(__linux__)
  if (dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) == nullptr) {
    throw std::runtime_error(
            "executorch_conversions failed to pin plugin library '" +
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
              "executorch_conversions: no plugin provides backend '" +
              backend + "'; available backends are " + describe());
    }
    return found->second;
  }

  std::shared_ptr<ConversionPlugin> for_device(DeviceType device_type) const
  {
    const auto found = devices_.find(device_type);
    if (found == devices_.end()) {
      throw std::runtime_error(
              "executorch_conversions: no plugin serves ExecuTorch device type " +
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
              "executorch_conversions: no conversion plugin is installed");
    }
    return selected;
  }

  bool has(const std::string & backend) const
  {
    return backends_.count(backend) != 0;
  }

private:
  Registry()
  : loader_("executorch_conversions", "executorch_conversions::ConversionPlugin")
  {
    for (const auto & class_name : loader_.getDeclaredClasses()) {
      try {
        auto plugin = loader_.createSharedInstance(class_name);
        // Register through pluginlib before pinning to preserve unloading of other libraries.
        pin_library(loader_.getClassLibraryPath(class_name));
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
  std::map<DeviceType, std::shared_ptr<ConversionPlugin>> devices_;
};

struct DtypeDescription
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

DtypeDescription describe_dtype(ScalarType type)
{
  switch (type) {
    case ScalarType::Byte: return {kDLUInt, 8, 1};
    case ScalarType::Char: return {kDLInt, 8, 1};
    case ScalarType::Short: return {kDLInt, 16, 1};
    case ScalarType::Int: return {kDLInt, 32, 1};
    case ScalarType::Long: return {kDLInt, 64, 1};
    case ScalarType::Half: return {kDLFloat, 16, 1};
    case ScalarType::BFloat16: return {kDLBfloat, 16, 1};
    case ScalarType::Float: return {kDLFloat, 32, 1};
    case ScalarType::Double: return {kDLFloat, 64, 1};
    case ScalarType::Bool: return {kDLBool, 8, 1};
    default:
      throw std::runtime_error(
              "executorch_conversions: unsupported scalar type");
  }
}

size_t element_count(const std::vector<int64_t> & shape)
{
  size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension < 0) {
      throw std::runtime_error(
              "executorch_conversions: negative shape dimension");
    }
    if (dimension != 0 && count >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension))
    {
      throw std::overflow_error("executorch_conversions: tensor size overflow");
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
              "executorch_conversions: negative shape dimension");
    }
    result[index - 1] = stride;
    // A size-0 dimension counts as 1, as in ExecuTorch's dim_order_to_stride.
    const auto extent = std::max<int64_t>(dimension, 1);
    if (stride > std::numeric_limits<int64_t>::max() / extent) {
      throw std::overflow_error("executorch_conversions: tensor stride overflow");
    }
    stride *= extent;
  }
  return result;
}

size_t byte_count(const std::vector<int64_t> & shape, ScalarType dtype)
{
  const size_t count = element_count(shape);
  const size_t item_size = executorch::runtime::elementSize(dtype);
  if (count != 0 && item_size >
    std::numeric_limits<size_t>::max() / count)
  {
    throw std::overflow_error("executorch_conversions: tensor size overflow");
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
              "executorch_conversions: negative shape or stride");
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
      throw std::overflow_error("executorch_conversions: tensor span overflow");
    }
    const size_t contribution = extent * stride;
    if (span > std::numeric_limits<size_t>::max() - contribution) {
      throw std::overflow_error("executorch_conversions: tensor span overflow");
    }
    span += contribution;
  }

  const size_t item_size = executorch::runtime::elementSize(scalar_type(msg));
  // Typed access needs element alignment. Message storage itself is allocated aligned.
  if (msg.byte_offset % item_size != 0) {
    throw std::runtime_error(
            "executorch_conversions: byte_offset is not a multiple of the element size");
  }
  if (span != 0 && item_size >
    std::numeric_limits<size_t>::max() / span)
  {
    throw std::overflow_error("executorch_conversions: tensor span overflow");
  }
  const size_t bytes = span * item_size;
  if (msg.byte_offset > std::numeric_limits<size_t>::max() - bytes) {
    throw std::overflow_error("executorch_conversions: tensor span overflow");
  }
  const size_t required = msg.byte_offset + bytes;
  if (required > msg.data.size()) {
    throw std::runtime_error(
            "executorch_conversions: tensor view exceeds message storage");
  }
}

// A message beyond these limits can never be viewed, so making one is refused too.
void check_executorch_limits(
  const std::vector<int64_t> & shape, const std::vector<int64_t> & contiguous)
{
  if (shape.size() > executorch::runtime::kTensorDimensionLimit) {
    throw std::runtime_error(
            "executorch_conversions: tensor rank exceeds the ExecuTorch limit");
  }
  for (size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] > std::numeric_limits<executorch::aten::SizesType>::max() ||
      contiguous[index] > std::numeric_limits<executorch::aten::StridesType>::max())
    {
      throw std::overflow_error(
              "executorch_conversions: shape or stride exceeds ExecuTorch limits");
    }
  }
}

// ExecuTorch's Method copies or shares input bytes without comparing dimension
// orders, so a view must have the default contiguous layout. Checked before the
// plugin hands out storage, because a CUDA buffer accepts one write handle per
// message and a rejected view must not use it up.
std::vector<executorch::aten::SizesType> view_sizes(const TensorMsg & msg)
{
  using SizesType = executorch::aten::SizesType;

  const auto rank = msg.shape.size();
  const auto contiguous = contiguous_strides(msg.shape);
  check_executorch_limits(msg.shape, contiguous);
  const auto strides = normalized_strides(msg);
  std::vector<SizesType> sizes(rank);
  for (size_t index = 0; index < rank; ++index) {
    if (msg.shape[index] != 1 && strides[index] != contiguous[index]) {
      throw std::runtime_error(
              "executorch_conversions: strides do not describe a contiguous tensor, "
              "which ExecuTorch requires");
    }
    sizes[index] = static_cast<SizesType>(msg.shape[index]);
  }
  return sizes;
}

TensorPtr make_tensor_view(
  const TensorMsg & msg, std::vector<executorch::aten::SizesType> sizes,
  StorageView storage)
{
  return executorch::extension::make_tensor_ptr(
    std::move(sizes),
    static_cast<uint8_t *>(storage.data) + msg.byte_offset,
    {},
    {},
    scalar_type(msg),
    storage.device,
    executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
    // The message owns the bytes; the captured lease holds the access until the view goes.
    [lease = std::move(storage.lease)](void *) {});
}

void set_metadata(
  TensorMsg & msg, const std::vector<int64_t> & shape, const std::vector<int64_t> & strides,
  const DtypeDescription & dtype)
{
  msg.shape.assign(shape.begin(), shape.end());
  msg.strides.assign(strides.begin(), strides.end());
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

std::string backend_for_device(DeviceType device_type)
{
  return Registry::instance().for_device(device_type)->backend();
}

std::string default_backend()
{
  return Registry::instance().default_name();
}

ScalarType scalar_type(const TensorMsg & msg)
{
  if (msg.dtype_lanes != 1) {
    throw std::runtime_error(
            "executorch_conversions: vector-lane dtypes are unsupported");
  }
  if (msg.dtype_code == kDLUInt && msg.dtype_bits == 8) {
    return ScalarType::Byte;
  }
  if (msg.dtype_code == kDLInt) {
    switch (msg.dtype_bits) {
      case 8: return ScalarType::Char;
      case 16: return ScalarType::Short;
      case 32: return ScalarType::Int;
      case 64: return ScalarType::Long;
      default: break;
    }
  }
  if (msg.dtype_code == kDLFloat) {
    switch (msg.dtype_bits) {
      case 16: return ScalarType::Half;
      case 32: return ScalarType::Float;
      case 64: return ScalarType::Double;
      default: break;
    }
  }
  if (msg.dtype_code == kDLBfloat && msg.dtype_bits == 16) {
    return ScalarType::BFloat16;
  }
  if (msg.dtype_code == kDLBool && msg.dtype_bits == 8) {
    return ScalarType::Bool;
  }
  throw std::runtime_error(
          "executorch_conversions: unsupported message dtype");
}

std::vector<int64_t> normalized_strides(const TensorMsg & msg)
{
  if (msg.strides.empty()) {
    return contiguous_strides(msg.shape);
  }
  if (msg.strides.size() != msg.shape.size()) {
    throw std::runtime_error(
            "executorch_conversions: strides must match shape rank");
  }
  return {msg.strides.begin(), msg.strides.end()};
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ScalarType dtype,
  std::optional<Device> device)
{
  auto & registry = Registry::instance();
  std::shared_ptr<ConversionPlugin> plugin;
  Device selected_device(DeviceType::CPU);
  if (device) {
    selected_device = *device;
    plugin = registry.for_device(device->type());
  } else {
    plugin = registry.for_backend(registry.default_name());
    selected_device = Device(plugin->device_type());
  }

  const auto strides = contiguous_strides(shape);
  check_executorch_limits(shape, strides);
  auto msg = std::make_unique<TensorMsg>();
  set_metadata(*msg, shape, strides, describe_dtype(dtype));
  plugin->allocate(*msg, byte_count(shape, dtype), selected_device);
  return msg;
}

TensorPtr from_output_tensor_msg(TensorMsg & msg, void * execution_stream)
{
  if (msg.data.empty()) {
    return {};
  }
  validate_view(msg);
  auto sizes = view_sizes(msg);
  auto storage = Registry::instance().for_backend(
    msg.data.get_backend_type())->from_output(msg, execution_stream);
  return make_tensor_view(msg, std::move(sizes), std::move(storage));
}

TensorPtr from_input_tensor_msg(const TensorMsg & msg, void * execution_stream)
{
  if (msg.data.empty()) {
    return {};
  }
  validate_view(msg);
  auto sizes = view_sizes(msg);
  auto storage = Registry::instance().for_backend(
    msg.data.get_backend_type())->from_input(msg, execution_stream);
  return make_tensor_view(msg, std::move(sizes), std::move(storage));
}

void to_tensor_msg(
  TensorMsg & msg, const executorch::aten::Tensor & tensor, void * execution_stream)
{
  if (tensor.numel() == 0) {
    return;
  }
  // Described first, so an unsupported dtype throws before the copy changes msg.
  const auto dtype = describe_dtype(tensor.scalar_type());
  if (tensor.const_data_ptr() == nullptr) {
    throw std::invalid_argument("executorch_conversions: tensor has no data");
  }
  // TensorImpl accepts null strides, and the message needs them. Asked without
  // strides(), which asserts on null strides in builds without NDEBUG.
  if (!tensor.unsafeGetTensorImpl()->has_layout_metadata()) {
    throw std::invalid_argument("executorch_conversions: tensor has no strides");
  }
  // The views accept only contiguous messages, so a permuted layout such as
  // channels-last is refused here too.
  const std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
  const auto contiguous = contiguous_strides(shape);
  check_executorch_limits(shape, contiguous);
  for (size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] != 1 && tensor.strides()[index] != contiguous[index]) {
      throw std::invalid_argument(
              "executorch_conversions: tensor is not contiguous, which messages must be");
    }
  }
  const size_t required = tensor.nbytes();
  if (required > msg.data.size()) {
    throw std::runtime_error(
            "executorch_conversions: tensor exceeds message storage");
  }

  auto & registry = Registry::instance();
  const auto device = tensor.device();
  auto plugin = device.is_cpu() ?
    registry.for_backend(msg.data.get_backend_type()) :
    registry.for_device(device.type());
  // A contiguous tensor's bytes copy as one block.
  plugin->copy_to(msg, tensor.const_data_ptr(), required, device, execution_stream);
  // The checked strides, not the tensor's: a size-1 dimension may carry any stride,
  // including a negative one that readers refuse.
  set_metadata(msg, shape, contiguous, dtype);
}

std::unique_ptr<TensorMsg> to_tensor_msg(
  const executorch::aten::Tensor & tensor, void * execution_stream)
{
  if (tensor.numel() == 0) {
    return std::make_unique<TensorMsg>();
  }
  auto msg = allocate_tensor_msg(
    std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end()),
    tensor.scalar_type(), tensor.device());
  to_tensor_msg(*msg, tensor, execution_stream);
  return msg;
}

}  // namespace executorch_conversions
