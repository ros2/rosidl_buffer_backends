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

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "onnxruntime_conversions/conversion_plugin.hpp"

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

void validate_cpu(const TensorMsg & msg, void * execution_stream)
{
  if (msg.data.get_backend_type() != "cpu") {
    throw std::invalid_argument("CPU conversion requires CPU message storage");
  }
  if (execution_stream != nullptr) {
    throw std::invalid_argument("CPU conversion does not accept an execution stream");
  }
}

std::shared_ptr<StorageLease> acquire_cpu_input(
  std::shared_ptr<const TensorMsg> msg,
  void * execution_stream)
{
  validate_cpu(*msg, execution_stream);
  void * data = msg->data.empty() ?
    static_cast<void *>(&empty_storage) :
    const_cast<void *>(static_cast<const void *>(msg->data.data()));
  const size_t size = msg->data.size();
  return std::make_shared<CpuStorageLease>(std::move(msg), data, size);
}

std::shared_ptr<StorageLease> acquire_cpu_output(
  std::shared_ptr<TensorMsg> msg,
  void * execution_stream)
{
  validate_cpu(*msg, execution_stream);
  void * data = msg->data.empty() ?
    static_cast<void *>(&empty_storage) :
    static_cast<void *>(msg->data.data());
  const size_t size = msg->data.size();
  return std::make_shared<CpuStorageLease>(std::move(msg), data, size);
}

void copy_cpu_from_ort(
  TensorMsg & msg,
  const Ort::Value & value,
  size_t byte_count,
  void * execution_stream)
{
  validate_cpu(msg, execution_stream);
  if (value.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
    throw std::invalid_argument("CPU conversion requires a CPU Ort::Value");
  }
  if (byte_count != 0) {
    std::memcpy(msg.data.data(), value.GetTensorRawData(), byte_count);
  }
}

cudaStream_t require_explicit_stream(void * execution_stream)
{
  const auto stream = reinterpret_cast<cudaStream_t>(execution_stream);
  if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
    throw std::invalid_argument(
            "CUDA plugin requires a non-null application-owned CUDA stream");
  }
  const cudaError_t status = cudaStreamQuery(stream);
  if (status != cudaSuccess && status != cudaErrorNotReady) {
    (void)cudaGetLastError();
    throw std::invalid_argument("CUDA plugin received an invalid CUDA stream");
  }
  return stream;
}

void validate_device_id(int device_id)
{
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_id < 0 || device_id >= device_count) {
    throw std::invalid_argument("CUDA device_id is outside the available device range");
  }
}

template<typename T>
T * require_cuda_impl(rosidl::Buffer<uint8_t> & buffer)
{
  auto * impl = dynamic_cast<T *>(buffer.get_impl());
  if (!impl) {
    throw std::invalid_argument("CUDA plugin requires cuda_buffer-backed message storage");
  }
  return impl;
}

const cuda_buffer_backend::CudaBufferImpl<uint8_t> * require_cuda_impl(
  const rosidl::Buffer<uint8_t> & buffer)
{
  const auto * impl =
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl());
  if (!impl) {
    throw std::invalid_argument("CUDA plugin requires cuda_buffer-backed message storage");
  }
  return impl;
}

void validate_device_pointer(const void * pointer, int expected_device)
{
  if (!pointer) {
    return;
  }
  cudaPointerAttributes attributes{};
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != expected_device) {
    throw std::invalid_argument("CUDA storage pointer is not on the reported device");
  }
}

StorageMetadata make_metadata(void * data, size_t size_bytes, int device_id)
{
  validate_device_pointer(data, device_id);
  StorageMetadata metadata;
  metadata.data = data;
  metadata.size_bytes = size_bytes;
  metadata.device_type = OrtMemoryInfoDeviceType_GPU;
  metadata.device_id = device_id;
  metadata.allocator_name = "Cuda";
  return metadata;
}

class CudaInputLease final : public StorageLease
{
public:
  CudaInputLease(
    std::shared_ptr<const TensorMsg> owner,
    cudaStream_t stream)
  : owner_(std::move(owner))
  {
    const auto * impl = require_cuda_impl(owner_->data);
    if (!owner_->data.empty()) {
      handle_.emplace(impl->get_cuda_buffer().get_read_handle(stream));
    }
    metadata_ = make_metadata(
      handle_ ? const_cast<uint8_t *>(handle_->get_ptr()) : nullptr,
      owner_->data.size(), impl->get_device_id());
  }

  const StorageMetadata & metadata() const noexcept override
  {
    return metadata_;
  }

private:
  std::shared_ptr<const TensorMsg> owner_;
  std::optional<cuda_buffer_backend::ReadHandle> handle_;
  StorageMetadata metadata_;
};

class CudaOutputLease final : public StorageLease
{
public:
  CudaOutputLease(
    std::shared_ptr<TensorMsg> owner,
    cudaStream_t stream)
  : owner_(std::move(owner))
  {
    auto * impl =
      require_cuda_impl<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(owner_->data);
    impl->set_stream(stream);
    if (!owner_->data.empty()) {
      handle_.emplace(impl->get_cuda_buffer().get_write_handle(stream));
    }
    metadata_ = make_metadata(
      handle_ ? handle_->get_ptr() : nullptr,
      owner_->data.size(), impl->get_device_id());
  }

  const StorageMetadata & metadata() const noexcept override
  {
    return metadata_;
  }

private:
  std::shared_ptr<TensorMsg> owner_;
  std::optional<cuda_buffer_backend::WriteHandle> handle_;
  StorageMetadata metadata_;
};

}  // namespace

class CudaConversionPlugin final
  : public ConversionPlugin, public AutomaticSelectionCapability
{
public:
  std::vector<std::string> backends() const override
  {
    return {"cpu", "cuda"};
  }

  bool supports_automatic_selection(
    const ConversionConfiguration & configuration) const noexcept override
  {
    const auto stream = reinterpret_cast<cudaStream_t>(configuration.execution_stream);
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
      return false;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
      configuration.device_id < 0 || configuration.device_id >= device_count)
    {
      (void)cudaGetLastError();
      return false;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
      current_device != configuration.device_id)
    {
      (void)cudaGetLastError();
      return false;
    }
    const cudaError_t stream_status = cudaStreamQuery(stream);
    if (stream_status != cudaSuccess && stream_status != cudaErrorNotReady) {
      (void)cudaGetLastError();
      return false;
    }
    return true;
  }

  void allocate_storage(
    TensorMsg & msg, size_t byte_count, const std::string & backend) override
  {
    if (backend == "cpu") {
      if (msg.data.get_backend_type() != "cpu") {
        throw std::invalid_argument("CPU conversion requires CPU message storage");
      }
      msg.data.resize(byte_count);
      return;
    }
    if (backend != "cuda") {
      throw std::invalid_argument(
              "CUDA plugin cannot allocate '" + backend + "' storage");
    }
    if (byte_count == 0) {
      throw std::invalid_argument(
              "CUDA plugin does not support zero-byte tensor storage");
    }
    msg.data = cuda_buffer_backend::allocate_buffer(byte_count);
    if (msg.data.get_backend_type() != "cuda") {
      throw std::runtime_error("cuda_buffer did not allocate CUDA message storage");
    }
  }

  std::shared_ptr<StorageLease> acquire_input(
    std::shared_ptr<const TensorMsg> msg,
    void * execution_stream) override
  {
    if (!msg) {
      throw std::invalid_argument("CUDA input message must not be null");
    }
    if (msg->data.get_backend_type() == "cpu") {
      return acquire_cpu_input(std::move(msg), execution_stream);
    }
    return std::make_shared<CudaInputLease>(
      std::move(msg), require_explicit_stream(execution_stream));
  }

  std::shared_ptr<StorageLease> acquire_output(
    std::shared_ptr<TensorMsg> msg,
    void * execution_stream) override
  {
    if (!msg) {
      throw std::invalid_argument("CUDA output message must not be null");
    }
    if (msg->data.get_backend_type() == "cpu") {
      return acquire_cpu_output(std::move(msg), execution_stream);
    }
    return std::make_shared<CudaOutputLease>(
      std::move(msg), require_explicit_stream(execution_stream));
  }

  void copy_from_ort(
    TensorMsg & msg,
    const Ort::Value & value,
    size_t byte_count,
    void * execution_stream) override
  {
    if (msg.data.get_backend_type() == "cpu") {
      copy_cpu_from_ort(msg, value, byte_count, execution_stream);
      return;
    }
    const auto stream = require_explicit_stream(execution_stream);
    auto * impl =
      require_cuda_impl<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(msg.data);
    const auto memory_info = value.GetTensorMemoryInfo();
    if (memory_info.GetDeviceType() != OrtMemoryInfoDeviceType_GPU ||
      memory_info.GetAllocatorName() != std::string("Cuda"))
    {
      throw std::invalid_argument("CUDA plugin requires a CUDA Ort::Value");
    }
    if (memory_info.GetDeviceId() != impl->get_device_id()) {
      throw std::invalid_argument("CUDA Ort::Value and message storage device IDs differ");
    }
    if (byte_count > msg.data.size()) {
      throw std::out_of_range("CUDA Ort::Value exceeds destination storage");
    }

    const void * source = value.GetTensorRawData();
    validate_device_pointer(source, impl->get_device_id());
    impl->set_stream(stream);
    if (byte_count != 0) {
      auto handle = impl->get_cuda_buffer().get_write_handle(stream);
      if (source != handle.get_ptr()) {
        CUDA_CHECK(cudaMemcpyAsync(
            handle.get_ptr(), source, byte_count, cudaMemcpyDeviceToDevice, stream));
      }
    }
  }

  void configure_session(
    Ort::SessionOptions & session_options,
    const std::string & backend,
    const ConversionConfiguration & configuration) override
  {
    if (backend == "cpu") {
      if (configuration.device_id != 0 || configuration.execution_stream != nullptr) {
        throw std::invalid_argument(
                "CPU conversion requires device_id 0 and a null execution stream");
      }
      return;
    }
    if (backend != "cuda") {
      throw std::invalid_argument(
              "CUDA plugin cannot configure a '" + backend + "' session");
    }
    validate_device_id(configuration.device_id);
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device != configuration.device_id) {
      throw std::invalid_argument(
              "CUDA device_id must match the stream's current CUDA device");
    }
    auto stream = require_explicit_stream(configuration.execution_stream);
    Ort::CUDAProviderOptions options;
    options.Update({{"device_id", std::to_string(configuration.device_id)}});
    options.UpdateWithValue("user_compute_stream", stream);
    session_options.AppendExecutionProvider_CUDA_V2(*options);
  }
};

}  // namespace onnxruntime_conversions

PLUGINLIB_EXPORT_CLASS(
  onnxruntime_conversions::CudaConversionPlugin,
  onnxruntime_conversions::ConversionPlugin)
