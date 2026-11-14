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

#include <cstring>
#include <stdexcept>

#include <pluginlib/class_list_macros.hpp>

#include "executorch_conversions/conversion_plugin.hpp"

namespace executorch_conversions_cpu
{

class CpuConversionPlugin final
  : public executorch_conversions::ConversionPlugin
{
public:
  std::string backend() const override
  {
    return "cpu";
  }

  DeviceType device_type() const override
  {
    return DeviceType::CPU;
  }

  bool available() const override
  {
    return true;
  }

  int priority() const override
  {
    return 0;
  }

  void allocate(TensorMsg & msg, size_t byte_count, Device device) override
  {
    if (!device.is_cpu()) {
      throw std::runtime_error("CPU plugin received a non-CPU device");
    }
    msg.data.resize(byte_count);
  }

  executorch_conversions::StorageView from_input(const TensorMsg & msg, void *) override
  {
    require_cpu_storage(msg);
    return {const_cast<uint8_t *>(msg.data.data()), DeviceType::CPU, nullptr};
  }

  executorch_conversions::StorageView from_output(TensorMsg & msg, void *) override
  {
    require_cpu_storage(msg);
    return {msg.data.data(), DeviceType::CPU, nullptr};
  }

  void copy_to(
    TensorMsg & msg, const void * source, size_t byte_count, Device source_device,
    void *) override
  {
    require_cpu_storage(msg);
    if (!source_device.is_cpu()) {
      throw std::runtime_error("CPU plugin cannot read a non-CPU tensor");
    }
    // The source can be a view of this same message.
    std::memmove(msg.data.data(), source, byte_count);
  }

private:
  static void require_cpu_storage(const TensorMsg & msg)
  {
    if (msg.data.get_backend_type() != "cpu") {
      throw std::runtime_error(
              "CPU plugin cannot handle buffer backend '" +
              msg.data.get_backend_type() + "'");
    }
  }
};

}  // namespace executorch_conversions_cpu

PLUGINLIB_EXPORT_CLASS(
  executorch_conversions_cpu::CpuConversionPlugin,
  executorch_conversions::ConversionPlugin)
