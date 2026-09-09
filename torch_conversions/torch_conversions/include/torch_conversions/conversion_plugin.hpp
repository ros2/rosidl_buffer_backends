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

#ifndef TORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_
#define TORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace torch_conversions
{

enum class DeviceKind
{
  cpu,
  cuda,
};

struct StorageView
{
  void * data{};
  int32_t dl_device_type{};
  int32_t device_id{};
  std::shared_ptr<void> lease;
};

class ConversionPlugin
{
public:
  using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

  virtual ~ConversionPlugin() = default;
  virtual DeviceKind default_device() const = 0;
  virtual bool device_available(DeviceKind device) const = 0;
  virtual void allocate(TensorMsg & msg, size_t byte_count, DeviceKind device) = 0;
  virtual StorageView acquire_input(const TensorMsg & msg, uintptr_t stream) = 0;
  virtual StorageView acquire_output(TensorMsg & msg, uintptr_t stream) = 0;
  virtual void copy_to(
    TensorMsg & msg,
    const void * source,
    size_t byte_count,
    DeviceKind source_device,
    uintptr_t stream) = 0;
};

}  // namespace torch_conversions

#endif  // TORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_
