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

#include "dlpack_conversions/dlpack_conversions.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pluginlib/class_loader.hpp>

#include "dlpack_conversions/storage_plugin.hpp"

namespace dlpack_conversions
{
namespace
{

struct DlpackContext
{
  StorageView view;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};

void pin_library(const std::string & path)
{
#if defined(__linux__)
  // RTLD_NODELETE keeps plugin statics alive for the life of the process.
  if (dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) == nullptr) {
    throw std::runtime_error(
            "dlpack_conversions failed to pin plugin library '" + path + "': " +
            dlerror());
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

  std::shared_ptr<StoragePlugin> for_backend(const std::string & backend) const
  {
    const auto found = backends_.find(backend);
    if (found == backends_.end()) {
      throw std::runtime_error(
              "dlpack_conversions: no storage plugin provides backend '" +
              backend + "'; available backends are " + describe());
    }
    return found->second;
  }

  std::string backend_for_device(int32_t dl_device_type) const
  {
    for (const auto & plugin : plugins_) {
      const std::string backend = plugin->backend_for_device(dl_device_type);
      if (!backend.empty() && backends_.count(backend) != 0) {
        return backend;
      }
    }
    return {};
  }

  // Per call, so an unusable ROSIDL_TENSOR_BACKEND fails here alone rather
  // than from every entry point, available_backends() included.
  std::string default_backend() const {return resolve_default();}

  bool available(const std::string & backend) const
  {
    return backends_.count(backend) != 0;
  }

  std::vector<std::string> names() const
  {
    std::vector<std::string> names;
    names.reserve(backends_.size());
    for (const auto & entry : backends_) {
      names.push_back(entry.first);
    }
    return names;
  }

  std::string describe() const
  {
    std::string text;
    for (const auto & entry : backends_) {
      text += text.empty() ? entry.first : ", " + entry.first;
    }
    return text.empty() ? "[none installed]" : text;
  }

private:
  Registry()
  : loader_("dlpack_conversions", "dlpack_conversions::StoragePlugin")
  {
    for (const auto & class_name : loader_.getDeclaredClasses()) {
      std::shared_ptr<StoragePlugin> plugin;
      try {
        pin_library(loader_.getClassLibraryPath(class_name));
        plugin = loader_.createSharedInstance(class_name);
      } catch (const std::exception &) {
        // An accelerator plugin whose runtime is absent is simply not offered.
        continue;
      }
      bool serves_anything = false;
      for (const auto & backend : plugin->backends()) {
        if (plugin->backend_available(backend)) {
          serves_anything |= backends_.emplace(backend, plugin).second;
        }
      }
      if (serves_anything) {
        plugins_.push_back(plugin);
      }
    }
  }

  std::string resolve_default() const
  {
    const char * requested = std::getenv("ROSIDL_TENSOR_BACKEND");
    if (requested != nullptr && *requested != '\0') {
      if (backends_.count(requested) == 0) {
        throw std::runtime_error(
                "dlpack_conversions: ROSIDL_TENSOR_BACKEND requests '" +
                std::string(requested) + "', which is not available; available "
                "backends are " + describe());
      }
      return requested;
    }
    // std::map iterates by name, so equal priorities resolve deterministically.
    std::string best;
    int best_priority = 0;
    for (const auto & entry : backends_) {
      const int priority = entry.second->priority();
      if (best.empty() || priority > best_priority) {
        best = entry.first;
        best_priority = priority;
      }
    }
    return best;
  }

  pluginlib::ClassLoader<StoragePlugin> loader_;
  std::vector<std::shared_ptr<StoragePlugin>> plugins_;
  std::map<std::string, std::shared_ptr<StoragePlugin>> backends_;
};

std::string require_backend_for_device(int32_t dl_device_type)
{
  const std::string backend = Registry::instance().backend_for_device(dl_device_type);
  if (backend.empty()) {
    throw std::runtime_error(
            "dlpack_conversions: no storage plugin serves DLPack device type " +
            std::to_string(dl_device_type) + "; available backends are " +
            Registry::instance().describe());
  }
  return backend;
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

size_t storage_size(const std::vector<int64_t> & shape, const DLDataType & dtype)
{
  size_t elements = 1;
  for (const auto dimension : shape) {
    if (dimension < 0) {
      throw std::runtime_error("dlpack_conversions: negative shape dimension");
    }
    elements *= static_cast<size_t>(dimension);
  }
  return elements * ((static_cast<size_t>(dtype.bits) * dtype.lanes + 7) / 8);
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
  tensor->dl_tensor.dtype = {msg.dtype_code, msg.dtype_bits, msg.dtype_lanes};
  tensor->dl_tensor.shape = context->shape.data();
  tensor->dl_tensor.strides =
    context->strides.empty() ? nullptr : context->strides.data();
  tensor->dl_tensor.byte_offset = 0;
  context.release();
  return tensor.release();
}

/// Rejects a message whose view reaches outside its own storage, before a
/// framework is handed a pointer into it.
void validate_view(const TensorMsg & msg)
{
  const std::vector<int64_t> shape(msg.shape.begin(), msg.shape.end());
  std::vector<int64_t> strides(msg.strides.begin(), msg.strides.end());
  if (strides.empty()) {
    strides = contiguous_strides(shape);
  } else if (strides.size() != shape.size()) {
    throw std::runtime_error(
            "dlpack_conversions: tensor strides must match the shape rank");
  }

  size_t span = 1;
  for (size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] < 0) {
      throw std::runtime_error("dlpack_conversions: negative shape dimension");
    }
    if (strides[index] < 0) {
      throw std::runtime_error("dlpack_conversions: negative tensor stride");
    }
    if (shape[index] == 0) {
      span = 0;
      break;
    }
    span += static_cast<size_t>(shape[index] - 1) *
      static_cast<size_t>(strides[index]);
  }

  const size_t item_size =
    (static_cast<size_t>(msg.dtype_bits) * msg.dtype_lanes + 7) / 8;
  const size_t required = msg.byte_offset + span * item_size;
  if (required > msg.data.size()) {
    throw std::runtime_error(
            "dlpack_conversions: tensor view needs " + std::to_string(required) +
            " bytes; buffer has " + std::to_string(msg.data.size()));
  }
}

void set_metadata(
  TensorMsg & msg,
  const DLTensor & source,
  const std::vector<int64_t> & shape)
{
  msg.shape.assign(shape.begin(), shape.end());
  if (source.strides != nullptr) {
    msg.strides.assign(source.strides, source.strides + source.ndim);
  } else {
    const auto strides = contiguous_strides(shape);
    msg.strides.assign(strides.begin(), strides.end());
  }
  msg.dtype_code = source.dtype.code;
  msg.dtype_bits = source.dtype.bits;
  msg.dtype_lanes = source.dtype.lanes;
  msg.byte_offset = 0;
}

}  // namespace

ManagedTensor::ManagedTensor(DLManagedTensor * tensor)
: tensor_(tensor) {}

ManagedTensor::~ManagedTensor()
{
  if (tensor_ != nullptr && tensor_->deleter != nullptr) {
    tensor_->deleter(tensor_);
  }
}

ManagedTensor::ManagedTensor(ManagedTensor && other) noexcept
: tensor_(other.tensor_)
{
  other.tensor_ = nullptr;
}

ManagedTensor & ManagedTensor::operator=(ManagedTensor && other) noexcept
{
  if (this != &other) {
    if (tensor_ != nullptr && tensor_->deleter != nullptr) {
      tensor_->deleter(tensor_);
    }
    tensor_ = other.tensor_;
    other.tensor_ = nullptr;
  }
  return *this;
}

DLManagedTensor * ManagedTensor::release() noexcept
{
  DLManagedTensor * released = tensor_;
  tensor_ = nullptr;
  return released;
}

std::string backend_for_device(int32_t dl_device_type)
{
  return Registry::instance().backend_for_device(dl_device_type);
}

std::string default_backend()
{
  return Registry::instance().default_backend();
}

bool backend_available(const std::string & backend)
{
  return Registry::instance().available(backend);
}

std::vector<std::string> available_backends()
{
  return Registry::instance().names();
}

std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  DLDataType dtype,
  const std::string & backend)
{
  auto & registry = Registry::instance();
  const std::string selected = backend.empty() ? registry.default_backend() : backend;
  if (selected.empty()) {
    throw std::runtime_error(
            "dlpack_conversions: no storage plugin is installed");
  }
  auto plugin = registry.for_backend(selected);

  auto msg = std::make_unique<TensorMsg>();
  const auto strides = contiguous_strides(shape);
  msg->shape.assign(shape.begin(), shape.end());
  msg->strides.assign(strides.begin(), strides.end());
  msg->dtype_code = dtype.code;
  msg->dtype_bits = dtype.bits;
  msg->dtype_lanes = dtype.lanes;
  msg->byte_offset = 0;
  plugin->allocate(*msg, storage_size(shape, dtype), selected);
  return msg;
}

ManagedTensor from_input_tensor_msg(const TensorMsg & msg, uintptr_t stream)
{
  if (msg.data.empty()) {
    return {};
  }
  validate_view(msg);
  auto plugin = Registry::instance().for_backend(msg.data.get_backend_type());
  return ManagedTensor(make_dlpack(msg, plugin->acquire_input(msg, stream)));
}

ManagedTensor from_output_tensor_msg(TensorMsg & msg, uintptr_t stream)
{
  if (msg.data.empty()) {
    return {};
  }
  validate_view(msg);
  auto plugin = Registry::instance().for_backend(msg.data.get_backend_type());
  return ManagedTensor(make_dlpack(msg, plugin->acquire_output(msg, stream)));
}

void to_tensor_msg(TensorMsg & msg, const DLTensor & source, uintptr_t stream)
{
  const std::vector<int64_t> shape(source.shape, source.shape + source.ndim);
  const size_t byte_count = storage_size(shape, source.dtype);
  if (byte_count == 0) {
    return;
  }
  if (byte_count > msg.data.size()) {
    throw std::runtime_error(
            "dlpack_conversions: tensor exceeds allocated message storage");
  }

  const std::string source_backend =
    require_backend_for_device(static_cast<int32_t>(source.device.device_type));
  // Only the accelerator's own plugin can read its device memory, so it also
  // performs device-to-host copies into host-backed messages.
  const std::string copier = source_backend == "cpu" ?
    msg.data.get_backend_type() : source_backend;
  Registry::instance().for_backend(copier)->copy_to(
    msg,
    static_cast<const uint8_t *>(source.data) + source.byte_offset,
    byte_count,
    source_backend,
    stream);
  set_metadata(msg, source, shape);
}

std::unique_ptr<TensorMsg> to_tensor_msg(const DLTensor & source, uintptr_t stream)
{
  const std::vector<int64_t> shape(source.shape, source.shape + source.ndim);
  if (storage_size(shape, source.dtype) == 0) {
    return std::make_unique<TensorMsg>();
  }
  auto msg = allocate_tensor_msg(
    shape,
    source.dtype,
    require_backend_for_device(static_cast<int32_t>(source.device.device_type)));
  to_tensor_msg(*msg, source, stream);
  return msg;
}

}  // namespace dlpack_conversions
