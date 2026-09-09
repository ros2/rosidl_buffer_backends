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

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <pluginlib/class_list_macros.hpp>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "torch_conversions/conversion_plugin.hpp"

namespace torch_conversions_cuda
{

class CudaConversionPlugin final : public torch_conversions::ConversionPlugin
{
public:
  torch_conversions::DeviceKind default_device() const override
  {
    return cuda_available() ?
           torch_conversions::DeviceKind::cuda :
           torch_conversions::DeviceKind::cpu;
  }

  bool device_available(torch_conversions::DeviceKind device) const override
  {
    return device == torch_conversions::DeviceKind::cpu ||
           (device == torch_conversions::DeviceKind::cuda && cuda_available());
  }

  void allocate(
    TensorMsg & msg,
    size_t byte_count,
    torch_conversions::DeviceKind device) override
  {
    if (device == torch_conversions::DeviceKind::cpu) {
      msg.data.resize(byte_count);
      return;
    }
    require_cuda();
    auto implementation =
      std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(byte_count);
    msg.data = rosidl::Buffer<uint8_t>(std::move(implementation));
  }

  torch_conversions::StorageView acquire_input(
    const TensorMsg & msg, uintptr_t stream_value) override
  {
    if (msg.data.get_backend_type() == "cpu") {
      return {const_cast<uint8_t *>(msg.data.data()), 1, 0, {}};
    }
    const auto * implementation = cuda_implementation(msg);
    auto lease = std::make_shared<cuda_buffer_backend::ReadHandle>(
      implementation->get_cuda_buffer().get_read_handle(stream(stream_value)));
    return {
      const_cast<uint8_t *>(lease->get_ptr()),
      2,
      implementation->get_device_id(),
      lease,
    };
  }

  torch_conversions::StorageView acquire_output(
    TensorMsg & msg, uintptr_t stream_value) override
  {
    if (msg.data.get_backend_type() == "cpu") {
      return {msg.data.data(), 1, 0, {}};
    }
    auto * implementation = cuda_implementation(msg);
    implementation->set_stream(stream(stream_value));
    auto lease = std::make_shared<cuda_buffer_backend::WriteHandle>(
      implementation->get_cuda_buffer().get_write_handle(stream(stream_value)));
    return {
      lease->get_ptr(),
      2,
      implementation->get_device_id(),
      lease,
    };
  }

  void copy_to(
    TensorMsg & msg,
    const void * source,
    size_t byte_count,
    torch_conversions::DeviceKind source_device,
    uintptr_t stream_value) override
  {
    const auto cuda_stream = stream(stream_value);
    if (msg.data.get_backend_type() == "cpu") {
      if (source_device == torch_conversions::DeviceKind::cpu) {
        std::memcpy(msg.data.data(), source, byte_count);
        return;
      }
      check_cuda(cudaMemcpyAsync(
          msg.data.data(), source, byte_count,
          cudaMemcpyDeviceToHost, cuda_stream));
      check_cuda(cudaStreamSynchronize(cuda_stream));
      return;
    }
    auto write_handle = cuda_buffer_backend::from_output_buffer(
      msg.data, cuda_stream);
    cuda_buffer_backend::to_buffer(
      source,
      byte_count,
      write_handle,
      cuda_stream,
      source_device == torch_conversions::DeviceKind::cuda ?
      cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
  }

private:
  static bool cuda_available()
  {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  }

  static void require_cuda()
  {
    if (!cuda_available()) {
      throw std::runtime_error(
              "torch_conversions_cuda: CUDA was requested but is unavailable");
    }
  }

  static cudaStream_t stream(uintptr_t value)
  {
    return reinterpret_cast<cudaStream_t>(value);
  }

  static void check_cuda(cudaError_t result)
  {
    if (result != cudaSuccess) {
      throw std::runtime_error(
              std::string("torch_conversions_cuda: ") +
              cudaGetErrorString(result));
    }
  }

  static const cuda_buffer_backend::CudaBufferImpl<uint8_t> *
  cuda_implementation(const TensorMsg & msg)
  {
    if (msg.data.get_backend_type() != "cuda") {
      throw std::runtime_error(
              "torch_conversions_cuda cannot handle buffer backend '" +
              msg.data.get_backend_type() + "'");
    }
    const auto * implementation =
      dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
      msg.data.get_impl());
    if (implementation == nullptr) {
      throw std::runtime_error(
              "torch_conversions_cuda: invalid CUDA buffer implementation");
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

}  // namespace torch_conversions_cuda

PLUGINLIB_EXPORT_CLASS(
  torch_conversions_cuda::CudaConversionPlugin,
  torch_conversions::ConversionPlugin)
