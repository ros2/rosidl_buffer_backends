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
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <memory>
#include <stdexcept>
#include <string>

#include <pluginlib/class_list_macros.hpp>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "torch_conversions/conversion_plugin.hpp"
#include "torch_conversions/torch_conversions.hpp"

namespace torch_conversions_cuda
{

class CudaConversionPlugin final
  : public torch_conversions::ConversionPlugin
{
public:
  std::string backend() const override
  {
    return "cuda";
  }

  c10::DeviceType device_type() const override
  {
    return c10::kCUDA;
  }

  bool available() const override
  {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  }

  int priority() const override
  {
    return 100;
  }

  void allocate(
    TensorMsg & msg, size_t byte_count, c10::Device device) override
  {
    if (!device.is_cuda()) {
      throw std::runtime_error("CUDA plugin received a non-CUDA device");
    }
    c10::cuda::CUDAGuard guard(device);
    msg.data = cuda_buffer_backend::allocate_buffer(byte_count);
  }

  at::Tensor from_input(
    const TensorMsg & msg, void * execution_stream) override
  {
    const auto * implementation = cuda_implementation(msg);
    const auto cuda_stream = stream(execution_stream);
    std::shared_ptr<void> lease =
      std::make_shared<cuda_buffer_backend::ReadHandle>(
      implementation->get_cuda_buffer().get_read_handle(cuda_stream));
    const auto * read_handle =
      static_cast<const cuda_buffer_backend::ReadHandle *>(lease.get());
    return make_tensor(
      const_cast<uint8_t *>(read_handle->get_ptr()),
      msg,
      implementation->get_device_id(),
      std::move(lease));
  }

  at::Tensor from_output(
    TensorMsg & msg, void * execution_stream) override
  {
    auto * implementation = cuda_implementation(msg);
    const auto cuda_stream = stream(execution_stream);
    implementation->set_stream(cuda_stream);
    std::shared_ptr<void> lease =
      std::make_shared<cuda_buffer_backend::WriteHandle>(
      implementation->get_cuda_buffer().get_write_handle(cuda_stream));
    auto * write_handle =
      static_cast<cuda_buffer_backend::WriteHandle *>(lease.get());
    return make_tensor(
      write_handle->get_ptr(),
      msg,
      implementation->get_device_id(),
      std::move(lease));
  }

  void copy_to(
    TensorMsg & msg,
    const at::Tensor & source,
    void * execution_stream) override
  {
    const auto cuda_stream = stream(execution_stream);
    if (msg.data.get_backend_type() == "cpu") {
      if (!source.device().is_cuda()) {
        throw std::runtime_error(
                "CUDA plugin expected a CUDA source for a CPU destination");
      }
      c10::cuda::CUDAGuard guard(source.device());
      check_cuda(cudaMemcpyAsync(
          msg.data.data(), source.data_ptr(), source.nbytes(),
          cudaMemcpyDeviceToHost, cuda_stream));
      check_cuda(cudaStreamSynchronize(cuda_stream));
      return;
    }

    auto * implementation = cuda_implementation(msg);
    c10::cuda::CUDAGuard guard(
      c10::Device(c10::kCUDA, implementation->get_device_id()));
    auto write_handle = cuda_buffer_backend::from_output_buffer(
      msg.data, cuda_stream);
    const auto kind = source.device().is_cpu() ?
      cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    cuda_buffer_backend::to_buffer(
      source.data_ptr(), source.nbytes(), write_handle, cuda_stream, kind);
  }

private:
  static at::Tensor make_tensor(
    void * data,
    const TensorMsg & msg,
    int device_id,
    std::shared_ptr<void> lease)
  {
    const auto strides =
      torch_conversions::normalized_strides(msg);
    const auto options = torch::TensorOptions()
      .dtype(torch_conversions::scalar_type(msg))
      .device(c10::Device(c10::kCUDA, device_id));
    return at::from_blob(
      static_cast<uint8_t *>(data) + msg.byte_offset,
      msg.shape,
      strides,
      [lease = std::move(lease)](void *) {},
      options);
  }

  static cudaStream_t stream(void * value)
  {
    return reinterpret_cast<cudaStream_t>(value);
  }

  static void check_cuda(cudaError_t result)
  {
    if (result != cudaSuccess) {
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

}  // namespace torch_conversions_cuda

PLUGINLIB_EXPORT_CLASS(
  torch_conversions_cuda::CudaConversionPlugin,
  torch_conversions::ConversionPlugin)
