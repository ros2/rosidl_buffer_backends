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

#ifndef EXECUTORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_
#define EXECUTORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_

#include <executorch/runtime/core/portable_type/device.h>

#include <cstddef>
#include <memory>
#include <string>

#include "tensor_msgs/msg/experimental_tensor.hpp"

namespace executorch_conversions
{

/// Message storage exposed to the core. `data` is the start of the storage: the
/// core adds the message's `byte_offset`. Releasing `lease` ends the backend's
/// access to `data`; the message keeps `data` allocated.
struct StorageView
{
  void * data{nullptr};
  executorch::runtime::etensor::Device device{
    executorch::runtime::etensor::DeviceType::CPU};
  std::shared_ptr<void> lease;
};

/// Storage ABI implemented by independently packaged DSOs. Plugins only move
/// bytes; the core builds the ExecuTorch tensors.
class ConversionPlugin
{
public:
  using TensorMsg = tensor_msgs::msg::ExperimentalTensor;
  using Device = executorch::runtime::etensor::Device;
  using DeviceType = executorch::runtime::etensor::DeviceType;

  virtual ~ConversionPlugin() = default;

  virtual std::string backend() const = 0;
  virtual DeviceType device_type() const = 0;
  virtual bool available() const = 0;
  virtual int priority() const = 0;

  virtual void allocate(TensorMsg & msg, size_t byte_count, Device device) = 0;

  virtual StorageView from_input(const TensorMsg & msg, void * execution_stream) = 0;

  virtual StorageView from_output(TensorMsg & msg, void * execution_stream) = 0;

  /// Copy `byte_count` dense bytes into message storage; returns after the copy completes.
  /// A plugin may refuse a source that overlaps the bytes it writes.
  virtual void copy_to(
    TensorMsg & msg,
    const void * source,
    size_t byte_count,
    Device source_device,
    void * execution_stream) = 0;
};

}  // namespace executorch_conversions

#endif  // EXECUTORCH_CONVERSIONS__CONVERSION_PLUGIN_HPP_
