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

#include <torch/torch.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include <pluginlib/class_list_macros.hpp>

#include "torch_conversions/conversion_plugin.hpp"
#include "torch_conversions/torch_conversions.hpp"

namespace torch_conversions_cpu
{

class CpuConversionPlugin final
  : public torch_conversions::ConversionPlugin
{
public:
  std::string backend() const override
  {
    return "cpu";
  }

  c10::DeviceType device_type() const override
  {
    return c10::kCPU;
  }

  bool available() const override
  {
    return true;
  }

  int priority() const override
  {
    return 0;
  }

  void allocate(
    TensorMsg & msg, size_t byte_count, c10::Device device) override
  {
    if (!device.is_cpu()) {
      throw std::runtime_error("CPU plugin received a non-CPU device");
    }
    msg.data.resize(byte_count);
  }

  at::Tensor from_input(const TensorMsg & msg, void *) override
  {
    require_cpu_storage(msg);
    return make_tensor(const_cast<uint8_t *>(msg.data.data()), msg);
  }

  at::Tensor from_output(TensorMsg & msg, void *) override
  {
    require_cpu_storage(msg);
    return make_tensor(msg.data.data(), msg);
  }

  void copy_to(
    TensorMsg & msg, const at::Tensor & source, void *) override
  {
    require_cpu_storage(msg);
    if (!source.device().is_cpu()) {
      throw std::runtime_error("CPU plugin cannot read a non-CPU tensor");
    }
    std::memcpy(msg.data.data(), source.data_ptr(), source.nbytes());
  }

private:
  static at::Tensor make_tensor(void * data, const TensorMsg & msg)
  {
    const auto strides =
      torch_conversions::normalized_strides(msg);
    const auto options = torch::TensorOptions()
      .dtype(torch_conversions::scalar_type(msg))
      .device(torch::kCPU);
    return at::from_blob(
      static_cast<uint8_t *>(data) + msg.byte_offset,
      msg.shape,
      strides,
      [](void *) {},
      options);
  }

  static void require_cpu_storage(const TensorMsg & msg)
  {
    if (msg.data.get_backend_type() != "cpu") {
      throw std::runtime_error(
              "CPU plugin cannot handle buffer backend '" +
              msg.data.get_backend_type() + "'");
    }
  }
};

}  // namespace torch_conversions_cpu

PLUGINLIB_EXPORT_CLASS(
  torch_conversions_cpu::CpuConversionPlugin,
  torch_conversions::ConversionPlugin)
