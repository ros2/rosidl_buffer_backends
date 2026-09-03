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
#include <utility>

#include <pluginlib/class_list_macros.hpp>
#include "onnxruntime_conversions/conversion_backend.hpp"

namespace onnxruntime_conversions
{
namespace
{

uint8_t empty_storage = 0;

class CpuStorageLease final : public StorageLease
{
public:
  CpuStorageLease(
    std::shared_ptr<const TensorMsg> owner,
    void * data,
    size_t size)
  : owner_(std::move(owner))
  {
    metadata_.data = data;
    metadata_.size_bytes = size;
    metadata_.device_type = OrtMemoryInfoDeviceType_CPU;
    metadata_.device_id = 0;
  }

  const StorageMetadata & metadata() const noexcept override
  {
    return metadata_;
  }

private:
  std::shared_ptr<const TensorMsg> owner_;
  StorageMetadata metadata_;
};

}  // namespace

class CpuConversionBackend final : public ConversionBackend
{
public:
  std::string backend_name() const override
  {
    return "cpu";
  }

  void allocate_storage(TensorMsg & msg, size_t byte_count) override
  {
    if (msg.data.get_backend_type() != "cpu") {
      throw std::invalid_argument("CPU plugin requires CPU message storage");
    }
    msg.data.resize(byte_count);
  }

  std::shared_ptr<StorageLease> acquire_input(
    std::shared_ptr<const TensorMsg> msg,
    void * execution_stream) override
  {
    validate(*msg, execution_stream);
    void * data = msg->data.empty() ?
      static_cast<void *>(&empty_storage) :
      const_cast<void *>(static_cast<const void *>(msg->data.data()));
    const size_t size = msg->data.size();
    return std::make_shared<CpuStorageLease>(std::move(msg), data, size);
  }

  std::shared_ptr<StorageLease> acquire_output(
    std::shared_ptr<TensorMsg> msg,
    void * execution_stream) override
  {
    validate(*msg, execution_stream);
    void * data = msg->data.empty() ?
      static_cast<void *>(&empty_storage) :
      static_cast<void *>(msg->data.data());
    const size_t size = msg->data.size();
    return std::make_shared<CpuStorageLease>(std::move(msg), data, size);
  }

  void copy_from_ort(
    TensorMsg & msg,
    const Ort::Value & value,
    size_t byte_count,
    void * execution_stream) override
  {
    validate(msg, execution_stream);
    if (value.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
      throw std::invalid_argument("CPU plugin requires a CPU Ort::Value");
    }
    if (byte_count != 0) {
      std::memcpy(msg.data.data(), value.GetTensorRawData(), byte_count);
    }
  }

  void configure_session(
    Ort::SessionOptions &,
    const BackendConfiguration & configuration) override
  {
    if (configuration.device_id != 0 || configuration.execution_stream != nullptr) {
      throw std::invalid_argument(
              "CPU backend requires device_id 0 and a null execution stream");
    }
  }

private:
  static void validate(const TensorMsg & msg, void * execution_stream)
  {
    if (msg.data.get_backend_type() != "cpu") {
      throw std::invalid_argument("CPU plugin received non-CPU message storage");
    }
    if (execution_stream != nullptr) {
      throw std::invalid_argument("CPU backend does not accept an execution stream");
    }
  }
};

}  // namespace onnxruntime_conversions

PLUGINLIB_EXPORT_CLASS(
  onnxruntime_conversions::CpuConversionBackend,
  onnxruntime_conversions::ConversionBackend)
