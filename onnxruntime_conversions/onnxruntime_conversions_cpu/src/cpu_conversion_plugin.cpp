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

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include <pluginlib/class_list_macros.hpp>

#include "onnxruntime_conversions/conversion_plugin.hpp"
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

namespace onnxruntime_conversions_cpu
{

class CpuConversionPlugin final
  : public onnxruntime_conversions::ConversionPlugin
{
public:
  std::string backend() const override {return "cpu";}
  bool supports(const Ort::ConstMemoryInfo & memory) const override
  {
    return memory.GetDeviceType() == OrtMemoryInfoDeviceType_CPU;
  }
  bool available() const override {return true;}
  int priority() const override {return 0;}

  Ort::SyncStream create_stream(Ort::Env &, int device_id) override
  {
    validate_stream(device_id, nullptr);
    return Ort::SyncStream{nullptr};
  }

  void validate_stream(int device_id, void * execution_stream) const override
  {
    if (device_id != 0 || execution_stream != nullptr) {
      throw std::invalid_argument("CPU streams require device 0 and a null handle");
    }
  }

  void allocate(TensorMsg & msg, size_t byte_count, int device_id) override
  {
    if (device_id != -1 && device_id != 0) {
      throw std::invalid_argument("CPU storage has no device index other than 0");
    }
    msg.data.resize(byte_count);
  }

  onnxruntime_conversions::ConversionView from_input(
    const TensorMsg & msg, void *) override
  {
    require_cpu_storage(msg);
    return make_view(const_cast<uint8_t *>(msg.data.data()), msg);
  }

  onnxruntime_conversions::ConversionView from_output(
    TensorMsg & msg, void *) override
  {
    require_cpu_storage(msg);
    return make_view(msg.data.data(), msg);
  }

  void copy_to(
    TensorMsg & msg, const Ort::Value & source, size_t byte_count,
    void *) override
  {
    require_cpu_storage(msg);
    if (source.GetTensorMemoryInfo().GetDeviceType() !=
      OrtMemoryInfoDeviceType_CPU)
    {
      throw std::runtime_error("CPU plugin cannot read a non-CPU Ort::Value");
    }
    std::memcpy(msg.data.data(), source.GetTensorRawData(), byte_count);
  }

  void configure_session(
    Ort::SessionOptions &, int, void * execution_stream) override
  {
    if (execution_stream != nullptr) {
      throw std::invalid_argument(
              "onnxruntime_conversions: host memory sessions take no execution stream");
    }
  }

private:
  static onnxruntime_conversions::ConversionView make_view(
    void * data, const TensorMsg & msg)
  {
    auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto value = Ort::Value::CreateTensor(
      memory_info,
      static_cast<uint8_t *>(data) + msg.byte_offset,
      onnxruntime_conversions::tensor_byte_count(msg),
      msg.shape.data(), msg.shape.size(),
      onnxruntime_conversions::element_type(msg));
    return {{}, std::move(value)};
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

}  // namespace onnxruntime_conversions_cpu

PLUGINLIB_EXPORT_CLASS(
  onnxruntime_conversions_cpu::CpuConversionPlugin,
  onnxruntime_conversions::ConversionPlugin)
