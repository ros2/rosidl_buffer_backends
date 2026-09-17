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

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include <pluginlib/class_loader.hpp>

#include "onnxruntime_conversions/conversion_plugin.hpp"

namespace onnxruntime_conversions
{
namespace
{

constexpr uint8_t kDtypeInt = 0;
constexpr uint8_t kDtypeUInt = 1;
constexpr uint8_t kDtypeFloat = 2;
constexpr uint8_t kDtypeBfloat = 4;
constexpr uint8_t kDtypeComplex = 5;
constexpr uint8_t kDtypeBool = 6;

void pin_library(const std::string & path)
{
#if defined(__linux__)
  if (dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) == nullptr) {
    throw std::runtime_error(
            "onnxruntime_conversions failed to pin plugin library '" +
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
              "onnxruntime_conversions: no plugin provides backend '" +
              backend + "'; available backends are " + describe());
    }
    return found->second;
  }

  std::shared_ptr<ConversionPlugin> for_device(
    const Ort::ConstMemoryInfo & memory) const
  {
    std::shared_ptr<ConversionPlugin> selected;
    for (const auto & entry : backends_) {
      if (entry.second->supports(memory)) {
        if (selected) {
          throw std::runtime_error("onnxruntime_conversions: ambiguous memory provider");
        }
        selected = entry.second;
      }
    }
    if (!selected) {
      throw std::runtime_error(
              "onnxruntime_conversions: no plugin serves allocator '" +
              memory.GetAllocatorName() + "'; available backends are " + describe());
    }
    return selected;
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
      const int priority = entry.second->priority();
      if (selected.empty() || priority > selected_priority) {
        selected = entry.first;
        selected_priority = priority;
      }
    }
    if (selected.empty()) {
      throw std::runtime_error(
              "onnxruntime_conversions: no conversion plugin is installed");
    }
    return selected;
  }

  bool has(const std::string & backend) const
  {
    return backends_.count(backend) != 0;
  }

private:
  Registry()
  : loader_("onnxruntime_conversions", "onnxruntime_conversions::ConversionPlugin")
  {
    for (const auto & class_name : loader_.getDeclaredClasses()) {
      try {
        pin_library(loader_.getClassLibraryPath(class_name));
        auto plugin = loader_.createSharedInstance(class_name);
        if (!plugin->available()) {
          continue;
        }
        backends_.emplace(plugin->backend(), plugin);
      } catch (const std::exception &) {
        // Independently installed accelerator plugins may lack their runtime.
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
};

struct DtypeDescription
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

size_t element_size(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return 16;
    default:
      throw std::invalid_argument(
              "onnxruntime_conversions: unsupported ONNX tensor element type");
  }
}

DtypeDescription describe_dtype(ONNXTensorElementDataType dtype)
{
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return {kDtypeInt, 8, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return {kDtypeInt, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return {kDtypeInt, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return {kDtypeInt, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return {kDtypeUInt, 8, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return {kDtypeUInt, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return {kDtypeUInt, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return {kDtypeUInt, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return {kDtypeFloat, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return {kDtypeFloat, 32, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return {kDtypeFloat, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return {kDtypeBfloat, 16, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return {kDtypeComplex, 64, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return {kDtypeComplex, 128, 1};
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return {kDtypeBool, 8, 1};
    default:
      throw std::invalid_argument(
              "onnxruntime_conversions: unsupported ONNX tensor element type");
  }
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> result(shape.size());
  int64_t stride = 1;
  for (size_t index = shape.size(); index > 0; --index) {
    const auto dimension = shape[index - 1];
    if (dimension < 0) {
      throw std::invalid_argument(
              "onnxruntime_conversions: shape dimensions must be nonnegative");
    }
    result[index - 1] = stride;
    if (dimension != 0 && stride >
      std::numeric_limits<int64_t>::max() / dimension)
    {
      throw std::overflow_error(
              "onnxruntime_conversions: tensor stride overflow");
    }
    stride *= dimension;
  }
  return result;
}

size_t element_count(const std::vector<int64_t> & shape)
{
  size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument(
              "onnxruntime_conversions: shape dimensions must be nonnegative");
    }
    if (dimension != 0 && count >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension))
    {
      throw std::overflow_error("onnxruntime_conversions: tensor size overflow");
    }
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

void validate_view(const TensorMsg & msg)
{
  const auto strides = normalized_strides(msg);
  if (strides != contiguous_strides(
      std::vector<int64_t>(msg.shape.begin(), msg.shape.end())))
  {
    throw std::invalid_argument(
            "onnxruntime_conversions: ONNX Runtime requires contiguous tensor strides");
  }
  const size_t bytes = tensor_byte_count(msg);
  if (msg.byte_offset > msg.data.size() || bytes > msg.data.size() - msg.byte_offset) {
    throw std::runtime_error(
            "onnxruntime_conversions: tensor view exceeds message storage");
  }
}

void set_metadata(TensorMsg & msg, const Ort::Value & value)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("onnxruntime_conversions: Ort::Value is not a tensor");
  }
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  const auto strides = contiguous_strides(shape);
  const auto dtype = describe_dtype(info.GetElementType());
  msg.shape.assign(shape.begin(), shape.end());
  msg.strides.assign(strides.begin(), strides.end());
  msg.dtype_code = dtype.code;
  msg.dtype_bits = dtype.bits;
  msg.dtype_lanes = dtype.lanes;
  msg.byte_offset = 0;
}

size_t value_byte_count(const Ort::Value & value)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("onnxruntime_conversions: Ort::Value is not a tensor");
  }
  const auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = info.GetElementCount();
  const size_t item_size = element_size(info.GetElementType());
  if (count != 0 && item_size > std::numeric_limits<size_t>::max() / count) {
    throw std::overflow_error("onnxruntime_conversions: tensor size overflow");
  }
  return count * item_size;
}

}  // namespace

Stream::Stream(std::string backend, int device_id, void * handle, Ort::SyncStream owner)
: backend_(std::move(backend)), device_id_(device_id), borrowed_handle_(handle),
  owner_(owner ? std::make_shared<Ort::SyncStream>(std::move(owner)) : nullptr) {}

void * Stream::handle() const
{
  return owner_ ? owner_->GetHandle() : borrowed_handle_;
}

Stream create_stream(Ort::Env & env, const std::string & backend, int device_id)
{
  auto & registry = Registry::instance();
  auto plugin = registry.for_backend(backend.empty() ? registry.default_name() : backend);
  auto owner = plugin->create_stream(env, device_id);
  plugin->validate_stream(device_id, owner ? owner.GetHandle() : nullptr);
  return Stream(plugin->backend(), device_id, nullptr, std::move(owner));
}

Stream borrow_stream(void * handle, const std::string & backend, int device_id)
{
  auto plugin = Registry::instance().for_backend(backend);
  plugin->validate_stream(device_id, handle);
  return Stream(plugin->backend(), device_id, handle, Ort::SyncStream{nullptr});
}

OrtTensorView::OrtTensorView() = default;

OrtTensorView::OrtTensorView(std::shared_ptr<void> lease, Ort::Value value)
: lease_(std::move(lease)), value_(std::move(value)) {}

OrtTensorView & OrtTensorView::operator=(OrtTensorView && other) noexcept
{
  if (this != &other) {
    value_ = Ort::Value{nullptr};
    lease_ = std::move(other.lease_);
    value_ = std::move(other.value_);
  }
  return *this;
}

Ort::Value & OrtTensorView::value() {return value_;}
const Ort::Value & OrtTensorView::value() const {return value_;}
OrtTensorView::operator bool() const noexcept {return value_ != nullptr;}

std::vector<std::string> available_backends()
{
  return Registry::instance().names();
}

bool backend_available(const std::string & backend)
{
  return Registry::instance().has(backend);
}

std::string default_backend()
{
  return Registry::instance().default_name();
}

ONNXTensorElementDataType element_type(const TensorMsg & msg)
{
  if (msg.dtype_lanes != 1) {
    throw std::invalid_argument(
            "onnxruntime_conversions: ONNX Runtime tensors require lanes == 1");
  }
  if (msg.dtype_code == kDtypeInt) {
    switch (msg.dtype_bits) {
      case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
      case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
      case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
  } else if (msg.dtype_code == kDtypeUInt) {
    switch (msg.dtype_bits) {
      case 8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
      case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
      case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    }
  } else if (msg.dtype_code == kDtypeFloat) {
    switch (msg.dtype_bits) {
      case 16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      case 32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      case 64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
  } else if (msg.dtype_code == kDtypeBfloat && msg.dtype_bits == 16) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  } else if (msg.dtype_code == kDtypeComplex) {
    if (msg.dtype_bits == 64) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    }
    if (msg.dtype_bits == 128) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
    }
  } else if (msg.dtype_code == kDtypeBool && msg.dtype_bits == 8) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  }
  throw std::invalid_argument(
          "onnxruntime_conversions: tensor dtype is unsupported by ONNX Runtime");
}

std::vector<int64_t> normalized_strides(const TensorMsg & msg)
{
  if (msg.strides.empty()) {
    return contiguous_strides(
      std::vector<int64_t>(msg.shape.begin(), msg.shape.end()));
  }
  if (msg.strides.size() != msg.shape.size()) {
    throw std::invalid_argument(
            "onnxruntime_conversions: strides must match shape rank");
  }
  return {msg.strides.begin(), msg.strides.end()};
}

size_t tensor_byte_count(const TensorMsg & msg)
{
  const auto shape = std::vector<int64_t>(msg.shape.begin(), msg.shape.end());
  const size_t count = element_count(shape);
  const size_t item_size = element_size(element_type(msg));
  if (count != 0 && item_size > std::numeric_limits<size_t>::max() / count) {
    throw std::overflow_error("onnxruntime_conversions: tensor size overflow");
  }
  return count * item_size;
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  ONNXTensorElementDataType dtype,
  const std::string & backend,
  int device_id)
{
  const auto description = describe_dtype(dtype);
  const auto strides = contiguous_strides(shape);
  auto msg = std::make_unique<TensorMsg>();
  msg->shape.assign(shape.begin(), shape.end());
  msg->strides.assign(strides.begin(), strides.end());
  msg->dtype_code = description.code;
  msg->dtype_bits = description.bits;
  msg->dtype_lanes = description.lanes;
  msg->byte_offset = 0;
  auto & registry = Registry::instance();
  auto plugin = registry.for_backend(
    backend.empty() ? registry.default_name() : backend);
  const size_t count = element_count(shape);
  const size_t item_size = element_size(dtype);
  if (count != 0 && item_size > std::numeric_limits<size_t>::max() / count) {
    throw std::overflow_error("onnxruntime_conversions: tensor size overflow");
  }
  plugin->allocate(*msg, count * item_size, device_id);
  return msg;
}

OrtTensorView from_input_tensor_msg(
  const TensorMsg & msg, void * execution_stream)
{
  if (msg.data.empty()) {
    return {};
  }
  validate_view(msg);
  auto view = Registry::instance().for_backend(
    msg.data.get_backend_type())->from_input(msg, execution_stream);
  return OrtTensorView(std::move(view.lease), std::move(view.value));
}

OrtTensorView from_output_tensor_msg(
  TensorMsg & msg, void * execution_stream)
{
  if (msg.data.empty()) {
    return {};
  }
  validate_view(msg);
  auto view = Registry::instance().for_backend(
    msg.data.get_backend_type())->from_output(msg, execution_stream);
  return OrtTensorView(std::move(view.lease), std::move(view.value));
}

void to_tensor_msg(
  TensorMsg & msg, const Ort::Value & value, void * execution_stream)
{
  const size_t bytes = value_byte_count(value);
  if (bytes > msg.data.size()) {
    throw std::runtime_error(
            "onnxruntime_conversions: tensor exceeds message storage");
  }
  const auto memory = value.GetTensorMemoryInfo();
  auto & registry = Registry::instance();
  auto plugin = memory.GetDeviceType() == OrtMemoryInfoDeviceType_CPU ?
    registry.for_backend(msg.data.get_backend_type()) :
    registry.for_device(memory);
  plugin->copy_to(msg, value, bytes, execution_stream);
  set_metadata(msg, value);
}

std::unique_ptr<TensorMsg> to_tensor_msg(
  const Ort::Value & value, void * execution_stream)
{
  if (!value.IsTensor()) {
    throw std::invalid_argument("onnxruntime_conversions: Ort::Value is not a tensor");
  }
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto memory = value.GetTensorMemoryInfo();
  auto plugin = Registry::instance().for_device(memory);
  auto msg = allocate_tensor_msg(
    info.GetShape(), info.GetElementType(), plugin->backend(), memory.GetDeviceId());
  to_tensor_msg(*msg, value, execution_stream);
  return msg;
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape, ONNXTensorElementDataType dtype, const Stream & stream)
{
  return allocate_tensor_msg(shape, dtype, stream.backend(), stream.device_id());
}

OrtTensorView from_input_tensor_msg(const TensorMsg & msg, const Stream & stream)
{
  return from_input_tensor_msg(msg, stream.handle());
}

OrtTensorView from_output_tensor_msg(TensorMsg & msg, const Stream & stream)
{
  return from_output_tensor_msg(msg, stream.handle());
}

void to_tensor_msg(TensorMsg & msg, const Ort::Value & value, const Stream & stream)
{
  to_tensor_msg(msg, value, stream.handle());
}

std::unique_ptr<TensorMsg> to_tensor_msg(const Ort::Value & value, const Stream & stream)
{
  return to_tensor_msg(value, stream.handle());
}

void configure_session_options(Ort::SessionOptions & session_options, const Stream & stream)
{
  configure_session_options(session_options, stream.backend(), stream.device_id(), stream.handle());
}

void configure_session_options(
  Ort::SessionOptions & session_options,
  const std::string & backend,
  int device_id,
  void * execution_stream)
{
  auto & registry = Registry::instance();
  const auto selected = backend.empty() ? registry.default_name() : backend;
  if (!registry.has(selected)) {
    if (selected != "cpu" && execution_stream == nullptr) {
      throw std::invalid_argument(
              "onnxruntime_conversions: backend '" + selected +
              "' requires an explicit execution stream");
    }
    throw std::invalid_argument(
            "onnxruntime_conversions: no ONNX Runtime execution provider is "
            "installed for backend '" + selected + "'");
  }
  registry.for_backend(selected)->configure_session(
    session_options, device_id, execution_stream);
}

}  // namespace onnxruntime_conversions
