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

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <pluginlib/class_list_macros.hpp>

#include "torch_conversions/conversion_plugin.hpp"

namespace torch_conversions_cpu
{

class CpuConversionPlugin final : public torch_conversions::ConversionPlugin
{
public:
  torch_conversions::DeviceKind default_device() const override
  {
    return torch_conversions::DeviceKind::cpu;
  }

  bool device_available(torch_conversions::DeviceKind device) const override
  {
    return device == torch_conversions::DeviceKind::cpu;
  }

  void allocate(
    TensorMsg & msg,
    size_t byte_count,
    torch_conversions::DeviceKind device) override
  {
    require_cpu(device);
    msg.data.resize(byte_count);
  }

  torch_conversions::StorageView acquire_input(
    const TensorMsg & msg, uintptr_t) override
  {
    require_cpu_backend(msg);
    return {
      const_cast<uint8_t *>(msg.data.data()),
      1,
      0,
      {},
    };
  }

  torch_conversions::StorageView acquire_output(
    TensorMsg & msg, uintptr_t) override
  {
    require_cpu_backend(msg);
    return {msg.data.data(), 1, 0, {}};
  }

  void copy_to(
    TensorMsg & msg,
    const void * source,
    size_t byte_count,
    torch_conversions::DeviceKind source_device,
    uintptr_t) override
  {
    require_cpu(source_device);
    require_cpu_backend(msg);
    std::memcpy(msg.data.data(), source, byte_count);
  }

private:
  static void require_cpu(torch_conversions::DeviceKind device)
  {
    if (device != torch_conversions::DeviceKind::cpu) {
      throw std::runtime_error(
              "torch_conversions_cpu does not support CUDA tensors");
    }
  }

  static void require_cpu_backend(const TensorMsg & msg)
  {
    if (msg.data.get_backend_type() != "cpu") {
      throw std::runtime_error(
              "torch_conversions_cpu cannot handle buffer backend '" +
              msg.data.get_backend_type() + "'");
    }
  }
};

}  // namespace torch_conversions_cpu

PLUGINLIB_EXPORT_CLASS(
  torch_conversions_cpu::CpuConversionPlugin,
  torch_conversions::ConversionPlugin)
