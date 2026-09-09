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

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pluginlib/class_loader.hpp>

#include "onnxruntime_conversions/detail/plugin_catalog.hpp"

namespace onnxruntime_conversions
{
namespace
{

constexpr uint8_t kDlInt = 0;
constexpr uint8_t kDlUInt = 1;
constexpr uint8_t kDlFloat = 2;
constexpr uint8_t kDlBfloat = 4;
constexpr uint8_t kDlComplex = 5;
constexpr uint8_t kDlBool = 6;

struct TensorMetadata
{
  std::vector<int64_t> shape;
  ONNXTensorElementDataType dtype;
  size_t element_size;
  size_t byte_count;
};

size_t checked_multiply(size_t lhs, size_t rhs, const char * context)
{
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw std::overflow_error(context);
  }
  return lhs * rhs;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (size_t index = shape.size(); index > 0; --index) {
    const int64_t dimension = shape[index - 1];
    strides[index - 1] = stride;
    if (dimension != 0 && stride > std::numeric_limits<int64_t>::max() / dimension) {
      throw std::overflow_error("Tensor strides overflow int64");
    }
    stride *= dimension;
  }
  return strides;
}

size_t element_size(ONNXTensorElementDataType dtype)
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

ONNXTensorElementDataType dtype_from_msg(const TensorMsg & msg)
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

void set_msg_dtype(TensorMsg & msg, ONNXTensorElementDataType dtype)
{
  msg.dtype_lanes = 1;
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      msg.dtype_code = kDlInt; msg.dtype_bits = 8; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      msg.dtype_code = kDlInt; msg.dtype_bits = 16; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      msg.dtype_code = kDlInt; msg.dtype_bits = 32; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      msg.dtype_code = kDlInt; msg.dtype_bits = 64; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      msg.dtype_code = kDlUInt; msg.dtype_bits = 8; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      msg.dtype_code = kDlUInt; msg.dtype_bits = 16; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      msg.dtype_code = kDlUInt; msg.dtype_bits = 32; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      msg.dtype_code = kDlUInt; msg.dtype_bits = 64; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      msg.dtype_code = kDlFloat; msg.dtype_bits = 16; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      msg.dtype_code = kDlFloat; msg.dtype_bits = 32; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      msg.dtype_code = kDlFloat; msg.dtype_bits = 64; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      msg.dtype_code = kDlBfloat; msg.dtype_bits = 16; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      msg.dtype_code = kDlComplex; msg.dtype_bits = 64; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      msg.dtype_code = kDlComplex; msg.dtype_bits = 128; return;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      msg.dtype_code = kDlBool; msg.dtype_bits = 8; return;
    default:
      throw std::invalid_argument("Unsupported ONNX tensor element type");
  }
}

TensorMetadata validate_metadata(const TensorMsg & msg)
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
  if (!msg.strides.empty() &&
    (msg.strides.size() != expected_strides.size() ||
    !std::equal(msg.strides.begin(), msg.strides.end(), expected_strides.begin())))
  {
    throw std::invalid_argument("ONNX Runtime conversion requires contiguous tensor strides");
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

void validate_storage(
  const StorageMetadata & storage,
  const Ort::MemoryInfo & memory_info,
  size_t required_size)
{
  if (storage.size_bytes < required_size) {
    throw std::out_of_range("Storage lease is smaller than the tensor backing buffer");
  }
  if (storage.device_type != memory_info.GetDeviceType()) {
    throw std::invalid_argument("Buffer backend and Ort::MemoryInfo device types differ");
  }
  if (storage.device_id != memory_info.GetDeviceId()) {
    throw std::invalid_argument("Buffer backend and Ort::MemoryInfo device IDs differ");
  }
  if (!storage.allocator_name.empty() &&
    storage.allocator_name != memory_info.GetAllocatorName())
  {
    throw std::invalid_argument("Buffer backend and Ort::MemoryInfo allocators differ");
  }
  if (!storage.data && required_size != 0) {
    throw std::runtime_error("Conversion plugin returned a null storage pointer");
  }
}

}  // namespace

StorageLease::~StorageLease() = default;
ConversionPlugin::~ConversionPlugin() = default;
AutomaticSelectionCapability::~AutomaticSelectionCapability() = default;

struct ConversionPluginRegistry::Impl
{
  Impl()
  : loader("onnxruntime_conversions", "onnxruntime_conversions::ConversionPlugin")
  {
    declared_classes = loader.getDeclaredClasses();
    std::sort(declared_classes.begin(), declared_classes.end());
    for (const auto & class_name : declared_classes) {
      std::shared_ptr<ConversionPlugin> plugin;
      std::vector<std::string> plugin_ids;
      try {
        plugin = loader.createSharedInstance(class_name);
        plugin_ids = plugin->backends();
        if (plugin_ids.empty()) {
          throw std::runtime_error("Plugin declared no backends");
        }
        if (std::find(plugin_ids.begin(), plugin_ids.end(), "") != plugin_ids.end()) {
          throw std::runtime_error("Plugin declared an empty backend name");
        }
      } catch (const std::exception & error) {
        load_failures.emplace(class_name, error.what());
        continue;
      }
      for (const auto & plugin_id : plugin_ids) {
        detail::register_plugin_id(plugin_classes, plugin_id, class_name);
        plugins.emplace(plugin_id, plugin);
      }
#if defined(__linux__)
      const std::string library_path = loader.getClassLibraryPath(class_name);
      void * library_handle =
        dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
      if (!library_handle) {
        throw std::runtime_error(
                "Failed to pin plugin library '" + library_path + "': " + dlerror());
      }
      pinned_library_handles.push_back(library_handle);
#endif
    }
    if (plugins.find("cpu") == plugins.end()) {
      throw std::runtime_error(diagnostic(
              "Required onnxruntime_conversions CPU plugin is unavailable."));
    }
  }

  std::string select_backend(const ConversionConfiguration & configuration) const
  {
    std::string selected;
    for (const auto & entry : plugins) {
      if (entry.first == "cpu") {
        continue;
      }
      const auto * capability =
        dynamic_cast<const AutomaticSelectionCapability *>(entry.second.get());
      if (!capability ||
        !capability->supports_automatic_selection(configuration))
      {
        continue;
      }
      if (!selected.empty()) {
        throw std::runtime_error(
                "Automatic ONNX Runtime plugin selection is ambiguous between '" +
                selected + "' and '" + entry.first + "'");
      }
      selected = entry.first;
    }
    if (selected.empty()) {
      return "cpu";
    }
    return selected;
  }

  std::string diagnostic(const std::string & subject) const
  {
    std::ostringstream message;
    message << subject << " Loaded plugin IDs: [";
    bool first = true;
    for (const auto & entry : plugins) {
      message << (first ? "" : ", ") << entry.first;
      first = false;
    }
    message << "]. Discoverable plugin classes: [";
    first = true;
    for (const auto & class_name : declared_classes) {
      message << (first ? "" : ", ") << class_name;
      first = false;
    }
    message << "]. Plugin load failures: [";
    first = true;
    for (const auto & failure : load_failures) {
      message << (first ? "" : "; ") << failure.first << ": " << failure.second;
      first = false;
    }
    if (first) {
      message << "none";
    }
    message << "].";
    return message.str();
  }

  mutable std::mutex mutex;
  pluginlib::ClassLoader<ConversionPlugin> loader;
  std::vector<std::string> declared_classes;
  std::map<std::string, std::shared_ptr<ConversionPlugin>> plugins;
  std::map<std::string, std::string> plugin_classes;
  std::map<std::string, std::string> load_failures;
  std::vector<void *> pinned_library_handles;
};

ConversionPluginRegistry::ConversionPluginRegistry()
: impl_(std::make_unique<Impl>()) {}

ConversionPluginRegistry::~ConversionPluginRegistry() = default;

ConversionPluginRegistry & ConversionPluginRegistry::instance()
{
  static ConversionPluginRegistry registry;
  return registry;
}

std::shared_ptr<ConversionPlugin> ConversionPluginRegistry::get_plugin(
  const std::string & plugin)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto found = impl_->plugins.find(plugin);
  if (found == impl_->plugins.end()) {
    throw std::invalid_argument(impl_->diagnostic(
            "ONNX Runtime conversion plugin '" + plugin + "' is unavailable."));
  }
  return found->second;
}

std::string ConversionPluginRegistry::select_backend(
  const ConversionConfiguration & configuration)
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->select_backend(configuration);
}

std::vector<std::string> ConversionPluginRegistry::available_plugins() const
{
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::vector<std::string> names;
  names.reserve(impl_->plugins.size());
  for (const auto & entry : impl_->plugins) {
    names.push_back(entry.first);
  }
  return names;
}

struct OrtTensorView::Impl
{
  explicit Impl(std::shared_ptr<const TensorMsg> owner_arg)
  : owner(std::move(owner_arg)), value(nullptr) {}

  std::shared_ptr<const TensorMsg> owner;
  std::shared_ptr<StorageLease> lease;
  Ort::Value value;
};

OrtTensorView::OrtTensorView(std::unique_ptr<Impl> impl)
: impl_(std::move(impl)) {}

OrtTensorView::OrtTensorView(OrtTensorView &&) noexcept = default;
OrtTensorView & OrtTensorView::operator=(OrtTensorView &&) noexcept = default;
OrtTensorView::~OrtTensorView() = default;

Ort::Value & OrtTensorView::value()
{
  if (!impl_) {
    throw std::logic_error("OrtTensorView has been moved from");
  }
  return impl_->value;
}

const Ort::Value & OrtTensorView::value() const
{
  if (!impl_) {
    throw std::logic_error("OrtTensorView has been moved from");
  }
  return impl_->value;
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & plugin)
{
  return allocate_tensor_msg(shape, dtype, plugin, {});
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const ConversionConfiguration & configuration)
{
  return allocate_tensor_msg(shape, dtype, "auto", configuration);
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & plugin,
  const ConversionConfiguration & configuration)
{
  size_t element_count = 1;
  for (const int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument("Tensor shape dimensions must be nonnegative");
    }
    if (static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max()) {
      throw std::overflow_error("Tensor shape dimension exceeds size_t");
    }
    element_count = checked_multiply(
      element_count, static_cast<size_t>(dimension), "Tensor element count overflow");
  }
  const size_t byte_count = checked_multiply(
    element_count, element_size(dtype), "Tensor byte count overflow");
  auto msg = std::make_unique<TensorMsg>();
  set_msg_dtype(*msg, dtype);
  msg->shape.assign(shape.begin(), shape.end());
  const auto strides = contiguous_strides(shape);
  msg->strides.assign(strides.begin(), strides.end());
  msg->byte_offset = 0;
  const auto selected_backend = plugin == "auto" ?
    ConversionPluginRegistry::instance().select_backend(configuration) :
    plugin;
  ConversionPluginRegistry::instance().get_plugin(selected_backend)->allocate_storage(
    *msg, byte_count, selected_backend);
  if (msg->data.get_backend_type() != selected_backend) {
    throw std::runtime_error(
            "Conversion plugin for backend '" + selected_backend + "' allocated '" +
            msg->data.get_backend_type() + "' storage");
  }
  return msg;
}

OrtTensorView from_input_tensor_msg(
  std::shared_ptr<const TensorMsg> msg,
  const Ort::MemoryInfo & memory_info,
  void * execution_stream)
{
  if (!msg) {
    throw std::invalid_argument("Input tensor message must not be null");
  }
  const auto metadata = validate_metadata(*msg);
  auto plugin =
    ConversionPluginRegistry::instance().get_plugin(msg->data.get_backend_type());
  auto impl = std::make_unique<OrtTensorView::Impl>(msg);
  impl->lease = plugin->acquire_input(msg, execution_stream);
  if (!impl->lease) {
    throw std::runtime_error("Conversion plugin returned a null input lease");
  }
  const auto & storage = impl->lease->metadata();
  validate_storage(storage, memory_info, msg->data.size());
  auto * data = static_cast<uint8_t *>(storage.data) + msg->byte_offset;
  impl->value = Ort::Value::CreateTensor(
    memory_info, data, metadata.byte_count, metadata.shape.data(),
    metadata.shape.size(), metadata.dtype);
  return OrtTensorView(std::move(impl));
}

OrtTensorView from_output_tensor_msg(
  std::shared_ptr<TensorMsg> msg,
  const Ort::MemoryInfo & memory_info,
  void * execution_stream)
{
  if (!msg) {
    throw std::invalid_argument("Output tensor message must not be null");
  }
  const auto metadata = validate_metadata(*msg);
  auto plugin =
    ConversionPluginRegistry::instance().get_plugin(msg->data.get_backend_type());
  auto impl = std::make_unique<OrtTensorView::Impl>(msg);
  impl->lease = plugin->acquire_output(msg, execution_stream);
  if (!impl->lease) {
    throw std::runtime_error("Conversion plugin returned a null output lease");
  }
  const auto & storage = impl->lease->metadata();
  validate_storage(storage, memory_info, msg->data.size());
  auto * data = static_cast<uint8_t *>(storage.data) + msg->byte_offset;
  impl->value = Ort::Value::CreateTensor(
    memory_info, data, metadata.byte_count, metadata.shape.data(),
    metadata.shape.size(), metadata.dtype);
  return OrtTensorView(std::move(impl));
}

void to_tensor_msg(
  TensorMsg & msg,
  const Ort::Value & value,
  void * execution_stream)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("Ort::Value is not a tensor");
  }
  const auto type_info = value.GetTensorTypeAndShapeInfo();
  const auto dtype = type_info.GetElementType();
  const auto shape = type_info.GetShape();
  const size_t byte_count = checked_multiply(
    type_info.GetElementCount(), element_size(dtype), "Tensor byte count overflow");
  if (byte_count > msg.data.size()) {
    throw std::out_of_range("Ort::Value tensor exceeds the destination buffer");
  }
  auto plugin =
    ConversionPluginRegistry::instance().get_plugin(msg.data.get_backend_type());
  plugin->copy_from_ort(msg, value, byte_count, execution_stream);
  set_msg_dtype(msg, dtype);
  msg.shape.assign(shape.begin(), shape.end());
  const auto strides = contiguous_strides(shape);
  msg.strides.assign(strides.begin(), strides.end());
  msg.byte_offset = 0;
}

std::unique_ptr<TensorMsg> to_tensor_msg(
  const Ort::Value & value,
  const std::string & plugin,
  void * execution_stream)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("Ort::Value is not a tensor");
  }
  const auto type_info = value.GetTensorTypeAndShapeInfo();
  ConversionConfiguration configuration;
  configuration.device_id = value.GetTensorMemoryInfo().GetDeviceId();
  configuration.execution_stream = execution_stream;
  auto msg = allocate_tensor_msg(
    type_info.GetShape(), type_info.GetElementType(), plugin, configuration);
  to_tensor_msg(*msg, value, execution_stream);
  return msg;
}

std::vector<std::string> available_plugins()
{
  return ConversionPluginRegistry::instance().available_plugins();
}

void configure_session_options(
  Ort::SessionOptions & session_options,
  const std::string & plugin,
  const ConversionConfiguration & configuration)
{
  const auto selected_backend = plugin == "auto" ?
    ConversionPluginRegistry::instance().select_backend(configuration) :
    plugin;
  ConversionPluginRegistry::instance().get_plugin(selected_backend)->configure_session(
    session_options, selected_backend, configuration);
}

}  // namespace onnxruntime_conversions

#endif  // ONNXRUNTIME_CONVERSIONS__DETAIL__ONNXRUNTIME_CONVERSIONS_IMPL_HPP_
