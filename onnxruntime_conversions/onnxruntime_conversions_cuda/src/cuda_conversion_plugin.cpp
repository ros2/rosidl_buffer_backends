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

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <pluginlib/class_list_macros.hpp>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "onnxruntime_conversions/conversion_plugin.hpp"
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

namespace onnxruntime_conversions_cuda
{

class CudaConversionPlugin final
  : public onnxruntime_conversions::ConversionPlugin
{
public:
  std::string backend() const override {return "cuda";}
  bool supports(const Ort::ConstMemoryInfo & memory) const override
  {
    return memory.GetDeviceType() == OrtMemoryInfoDeviceType_GPU &&
           memory.GetAllocatorName() == "Cuda";
  }
  bool available() const override
  {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  }
  int priority() const override {return 100;}

  Ort::SyncStream create_stream(Ort::Env & env, int device_id) override
  {
    DeviceGuard guard(device_id);
    static std::mutex registration_mutex;
    std::lock_guard<std::mutex> lock(registration_mutex);
    auto devices = env.GetEpDevices();
    bool registered = false;
    for (const auto & device : devices) {
      registered = registered || std::string(device.EpName()) == "CUDAExecutionProvider";
    }
    if (!registered) {
      env.RegisterExecutionProviderLibrary(
        "CUDAExecutionProvider", "libonnxruntime_providers_cuda.so");
      devices = env.GetEpDevices();
    }
    for (const auto & device : devices) {
      if (std::string(device.EpName()) == "CUDAExecutionProvider" &&
        device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT).GetDeviceId() == device_id)
      {
        return device.CreateSyncStream();
      }
    }
    throw std::runtime_error("ONNX Runtime has no stream provider for the requested CUDA device");
  }

  void validate_stream(int device_id, void * execution_stream) const override
  {
    if (execution_stream == nullptr) {
      throw std::invalid_argument("CUDA streams require a non-null native handle");
    }
    DeviceGuard guard(device_id);
    unsigned int flags = 0;
    check_cuda(cudaStreamGetFlags(stream(execution_stream), &flags));
  }

  void allocate(TensorMsg & msg, size_t byte_count, int device_id) override
  {
    if (device_id == -1) {
      check_cuda(cudaGetDevice(&device_id));
    }
    DeviceGuard guard(device_id);
    msg.data = cuda_buffer_backend::allocate_buffer(byte_count);
    if (byte_count > 0 && cuda_implementation(msg)->get_device_id() != device_id) {
      throw std::runtime_error("CUDA buffer pool cannot allocate on the requested device");
    }
  }

  onnxruntime_conversions::ConversionView from_input(
    const TensorMsg & msg, void * execution_stream) override
  {
    const auto * implementation = cuda_implementation(msg);
    DeviceGuard guard(implementation->get_device_id());
    const auto cuda_stream = stream(execution_stream);
    std::shared_ptr<void> lease =
      std::make_shared<cuda_buffer_backend::ReadHandle>(
      implementation->get_cuda_buffer().get_read_handle(cuda_stream));
    const auto * handle =
      static_cast<const cuda_buffer_backend::ReadHandle *>(lease.get());
    return make_view(
      const_cast<uint8_t *>(handle->get_ptr()), msg,
      implementation->get_device_id(), std::move(lease));
  }

  onnxruntime_conversions::ConversionView from_output(
    TensorMsg & msg, void * execution_stream) override
  {
    auto * implementation = cuda_implementation(msg);
    DeviceGuard guard(implementation->get_device_id());
    const auto cuda_stream = stream(execution_stream);
    implementation->set_stream(cuda_stream);
    std::shared_ptr<void> lease =
      std::make_shared<cuda_buffer_backend::WriteHandle>(
      implementation->get_cuda_buffer().get_write_handle(cuda_stream));
    auto * handle =
      static_cast<cuda_buffer_backend::WriteHandle *>(lease.get());
    return make_view(
      handle->get_ptr(), msg, implementation->get_device_id(),
      std::move(lease));
  }

  void copy_to(
    TensorMsg & msg, const Ort::Value & source, size_t byte_count,
    void * execution_stream) override
  {
    const auto cuda_stream = stream(execution_stream);
    const auto source_info = source.GetTensorMemoryInfo();
    const bool source_is_cpu =
      source_info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU;
    if (!source_is_cpu && !supports(source_info)) {
      throw std::invalid_argument("CUDA plugin cannot read this memory provider");
    }
    if (msg.data.get_backend_type() == "cpu") {
      if (source_is_cpu) {
        throw std::runtime_error(
                "CUDA plugin expected a CUDA source for a CPU destination");
      }
      DeviceGuard guard(source_info.GetDeviceId());
      check_cuda(cudaMemcpyAsync(
          msg.data.data(), source.GetTensorRawData(), byte_count,
          cudaMemcpyDeviceToHost, cuda_stream));
      check_cuda(cudaStreamSynchronize(cuda_stream));
      return;
    }

    auto * implementation = cuda_implementation(msg);
    if (!source_is_cpu && source_info.GetDeviceId() != implementation->get_device_id()) {
      throw std::invalid_argument("CUDA copies require matching source and destination devices");
    }
    DeviceGuard guard(implementation->get_device_id());
    auto write_handle = cuda_buffer_backend::from_output_buffer(
      msg.data, cuda_stream);
    const auto kind = source_is_cpu ?
      cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    cuda_buffer_backend::to_buffer(
      source.GetTensorRawData(), byte_count, write_handle, cuda_stream, kind);
    check_cuda(cudaStreamSynchronize(cuda_stream));
  }

  void configure_session(
    Ort::SessionOptions & options, int device_id,
    void * execution_stream) override
  {
    if (execution_stream == nullptr) {
      throw std::invalid_argument(
              "onnxruntime_conversions: backend 'cuda' requires an explicit execution stream");
    }
    Ort::CUDAProviderOptions provider_options;
    provider_options.Update({{"device_id", std::to_string(device_id)}});
    provider_options.UpdateWithValue("user_compute_stream", execution_stream);
    options.AppendExecutionProvider_CUDA_V2(*provider_options);
  }

private:
  class DeviceGuard
  {
public:
    explicit DeviceGuard(int device)
    {
      check_cuda(cudaGetDevice(&previous_));
      check_cuda(cudaSetDevice(device));
    }
    ~DeviceGuard() {cudaSetDevice(previous_);}

private:
    int previous_{0};
  };

  static onnxruntime_conversions::ConversionView make_view(
    void * data, const TensorMsg & msg, int device_id,
    std::shared_ptr<void> lease)
  {
    auto memory_info = Ort::MemoryInfo(
      "Cuda", OrtDeviceAllocator, device_id, OrtMemTypeDefault);
    auto value = Ort::Value::CreateTensor(
      memory_info,
      static_cast<uint8_t *>(data) + msg.byte_offset,
      onnxruntime_conversions::tensor_byte_count(msg),
      msg.shape.data(), msg.shape.size(),
      onnxruntime_conversions::element_type(msg));
    return {std::move(lease), std::move(value)};
  }

  static cudaStream_t stream(void * value)
  {
    return value ? reinterpret_cast<cudaStream_t>(value) : cudaStreamLegacy;
  }

  static void check_cuda(cudaError_t result)
  {
    if (result != cudaSuccess) {
      (void)cudaGetLastError();
      throw std::runtime_error(
              std::string("CUDA conversion plugin: ") +
              cudaGetErrorString(result));
    }
  }

  static const cuda_buffer_backend::CudaBufferImpl<uint8_t> *
  cuda_implementation(const TensorMsg & msg)
  {
    if (msg.data.get_backend_type() != "cuda") {
      throw std::runtime_error(
              "CUDA plugin cannot handle buffer backend '" +
              msg.data.get_backend_type() + "'");
    }
    const auto * implementation =
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      msg.data.get_impl());
    if (implementation == nullptr) {
      throw std::runtime_error("Invalid CUDA buffer implementation");
    }
    return implementation;
  }

  static cuda_buffer_backend::CudaBufferImpl<uint8_t> *
  cuda_implementation(TensorMsg & msg)
  {
    return const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      cuda_implementation(static_cast<const TensorMsg &>(msg)));
  }
};

}  // namespace onnxruntime_conversions_cuda

PLUGINLIB_EXPORT_CLASS(
  onnxruntime_conversions_cuda::CudaConversionPlugin,
  onnxruntime_conversions::ConversionPlugin)
