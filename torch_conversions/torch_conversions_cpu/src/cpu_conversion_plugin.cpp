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
#include <vector>

#include <pluginlib/class_list_macros.hpp>

#include "torch_conversions/conversion_plugin.hpp"

namespace torch_conversions_cpu
{

class CpuConversionPlugin final : public torch_conversions::ConversionPlugin
{
public:
  std::vector<std::string> backends() const override
  {
    return {"cpu"};
  }

  std::string backend_for_device(int32_t dl_device_type) const override
  {
    return dl_device_type == torch_conversions::dl_device::cpu ? "cpu" : "";
  }

  std::string default_backend() const override
  {
    return "cpu";
  }

  bool backend_available(const std::string & backend) const override
  {
    return backend == "cpu";
  }

  void allocate(
    TensorMsg & msg,
    size_t byte_count,
    const std::string & backend) override
  {
    require_cpu(backend);
    msg.data.resize(byte_count);
  }

  torch_conversions::StorageView acquire_input(
    const TensorMsg & msg, uintptr_t) override
  {
    require_cpu_backend(msg);
    return {
      const_cast<uint8_t *>(msg.data.data()),
      torch_conversions::dl_device::cpu,
      0,
      {},
    };
  }

  torch_conversions::StorageView acquire_output(
    TensorMsg & msg, uintptr_t) override
  {
    require_cpu_backend(msg);
    return {msg.data.data(), torch_conversions::dl_device::cpu, 0, {}};
  }

  void copy_to(
    TensorMsg & msg,
    const void * source,
    size_t byte_count,
    const std::string & source_backend,
    uintptr_t) override
  {
    require_cpu(source_backend);
    require_cpu_backend(msg);
    std::memcpy(msg.data.data(), source, byte_count);
  }

private:
  static void require_cpu(const std::string & backend)
  {
    if (backend != "cpu") {
      throw std::runtime_error(
              "torch_conversions_cpu does not support backend '" + backend + "'");
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
