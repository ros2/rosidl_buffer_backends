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
#include <vector>

#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace torch_conversions
{

// DLPack device type codes, as reported by StorageView::dl_device_type.
namespace dl_device
{
constexpr int32_t cpu = 1;
constexpr int32_t cuda = 2;
}  // namespace dl_device

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

  // Buffer backend names this plugin allocates, for diagnostics.
  virtual std::vector<std::string> backends() const = 0;
  // Backend storing tensors for the given DLPack device type, or an empty
  // string when this plugin does not serve that device.
  virtual std::string backend_for_device(int32_t dl_device_type) const = 0;
  virtual std::string default_backend() const = 0;
  virtual bool backend_available(const std::string & backend) const = 0;
  virtual void allocate(
    TensorMsg & msg,
    size_t byte_count,
    const std::string & backend) = 0;
  virtual StorageView acquire_input(const TensorMsg & msg, uintptr_t stream) = 0;
  virtual StorageView acquire_output(TensorMsg & msg, uintptr_t stream) = 0;
  virtual void copy_to(
    TensorMsg & msg,
    const void * source,
    size_t byte_count,
    const std::string & source_backend,
    uintptr_t stream) = 0;
};

}  // namespace torch_conversions

#endif  // TORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_
