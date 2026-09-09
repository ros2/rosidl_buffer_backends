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

#ifndef DLPACK_CONVERSIONS__STORAGE_PLUGIN_HPP_
#define DLPACK_CONVERSIONS__STORAGE_PLUGIN_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace dlpack_conversions
{

/// Storage handed to DLPack, with a lease that keeps the buffer readable.
struct StorageView
{
  void * data{};
  int32_t dl_device_type{};
  int32_t device_id{};
  std::shared_ptr<void> lease;
};

/// Allocates and exposes rosidl buffer storage for one or more accelerators.
///
/// Implementations depend on their accelerator runtime and on a rosidl buffer
/// backend, never on a tensor framework.
class StoragePlugin
{
public:
  using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

  virtual ~StoragePlugin() = default;

  /// Buffer backend names this plugin allocates.
  virtual std::vector<std::string> backends() const = 0;

  /// Backend storing tensors for the given DLPack device type, or an empty
  /// string when this plugin does not serve that device.
  virtual std::string backend_for_device(int32_t dl_device_type) const = 0;

  /// Whether the backend can be used on this machine right now.
  virtual bool backend_available(const std::string & backend) const = 0;

  /// Preference when no backend is requested. Highest wins; host memory is 0.
  virtual int priority() const = 0;

  virtual void allocate(
    TensorMsg & msg,
    size_t byte_count,
    const std::string & backend) = 0;

  virtual StorageView acquire_input(const TensorMsg & msg, uintptr_t stream) = 0;

  virtual StorageView acquire_output(TensorMsg & msg, uintptr_t stream) = 0;

  /// Copies host or accelerator memory into the message storage. Accelerator
  /// plugins are also asked to copy into host-backed messages, because only
  /// they can read their own device memory.
  virtual void copy_to(
    TensorMsg & msg,
    const void * source,
    size_t byte_count,
    const std::string & source_backend,
    uintptr_t stream) = 0;
};

}  // namespace dlpack_conversions

#endif  // DLPACK_CONVERSIONS__STORAGE_PLUGIN_HPP_
