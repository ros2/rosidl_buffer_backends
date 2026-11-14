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

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/cuda/device_guard.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include <pluginlib/class_list_macros.hpp>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "executorch_conversions/conversion_plugin.hpp"

namespace executorch_conversions_cuda
{

class CudaConversionPlugin final
  : public executorch_conversions::ConversionPlugin
{
public:
  std::string backend() const override
  {
    return "cuda";
  }

  DeviceType device_type() const override
  {
    return DeviceType::CUDA;
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

  void allocate(TensorMsg & msg, size_t byte_count, Device device) override
  {
    if (device.type() != DeviceType::CUDA) {
      throw std::runtime_error("CUDA plugin received a non-CUDA device");
    }
    auto guard = select_device(device.index());
    msg.data = cuda_buffer_backend::allocate_buffer(byte_count);
    if (byte_count > 0 && cuda_implementation(msg)->get_device_id() != device.index()) {
      throw std::runtime_error(
              "CUDA buffer pool is on device " +
              std::to_string(cuda_implementation(msg)->get_device_id()) +
              " and cannot allocate on device " + std::to_string(device.index()) +
              "; cuda_buffer serves one GPU per process");
    }
  }

  executorch_conversions::StorageView from_input(
    const TensorMsg & msg, void * execution_stream) override
  {
    const auto * implementation = cuda_implementation(msg);
    const int device = implementation->get_device_id();
    auto guard = select_device(device);
    const auto cuda_stream = resolve_stream(execution_stream, device);
    auto lease = lease_on(
      device, cuda_stream, implementation->get_cuda_buffer().get_read_handle(cuda_stream));
    const auto * read_handle =
      static_cast<const cuda_buffer_backend::ReadHandle *>(lease.get());
    return {
      const_cast<uint8_t *>(read_handle->get_ptr()),
      device_of(*implementation),
      std::move(lease)};
  }

  executorch_conversions::StorageView from_output(
    TensorMsg & msg, void * execution_stream) override
  {
    auto * implementation = cuda_implementation(msg);
    const int device = implementation->get_device_id();
    auto guard = select_device(device);
    const auto cuda_stream = resolve_stream(execution_stream, device);
    auto lease = lease_on(
      device, cuda_stream, implementation->get_cuda_buffer().get_write_handle(cuda_stream));
    auto * write_handle =
      static_cast<cuda_buffer_backend::WriteHandle *>(lease.get());
    return {write_handle->get_ptr(), device_of(*implementation), std::move(lease)};
  }

  void copy_to(
    TensorMsg & msg, const void * source, size_t byte_count, Device source_device,
    void * execution_stream) override
  {
    if (msg.data.get_backend_type() == "cpu") {
      if (source_device.is_cpu()) {
        throw std::runtime_error(
                "CUDA plugin expected a CUDA source for a CPU destination");
      }
      auto guard = select_device(source_device.index());
      const auto cuda_stream = resolve_stream(execution_stream, source_device.index());
      check_cuda(cudaMemcpyAsync(
          msg.data.data(), source, byte_count, cudaMemcpyDeviceToHost, cuda_stream));
      check_cuda(cudaStreamSynchronize(cuda_stream));
      return;
    }

    auto * implementation = cuda_implementation(msg);
    if (!source_device.is_cpu() && source_device.index() != implementation->get_device_id()) {
      throw std::invalid_argument("CUDA copies require matching source and destination devices");
    }
    // Unlike the CPU plugin's memmove, cudaMemcpyAsync does not handle overlap. Checked
    // before the write handle is taken, because a message accepts only one.
    const auto from = reinterpret_cast<std::uintptr_t>(source);
    const auto to = reinterpret_cast<std::uintptr_t>(
      implementation->get_cuda_buffer().get_device_ptr());
    if (!source_device.is_cpu() && from < to + byte_count && to < from + byte_count) {
      throw std::invalid_argument("CUDA plugin cannot copy bytes that overlap the ones it writes");
    }
    auto guard = select_device(implementation->get_device_id());
    const auto cuda_stream = resolve_stream(execution_stream, implementation->get_device_id());
    // Not from_output_buffer: it keeps the stream in the message, and a later copy of
    // the message would use it after the caller destroyed it.
    auto write_handle = implementation->get_cuda_buffer().get_write_handle(cuda_stream);
    const auto kind = source_device.is_cpu() ?
      cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    cuda_buffer_backend::to_buffer(source, byte_count, write_handle, cuda_stream, kind);
    check_cuda(cudaStreamSynchronize(cuda_stream));
  }

private:
  static executorch::extension::cuda::CUDAGuard select_device(int device)
  {
    auto guard = executorch::extension::cuda::CUDAGuard::create(device);
    if (!guard.ok()) {
      // Otherwise the next program run on this thread reports the failed switch.
      (void)cudaGetLastError();
      throw std::runtime_error("CUDA plugin cannot select device " + std::to_string(device));
    }
    return std::move(guard.get());
  }

  // A handle records its event on the current device when it is released,
  // which happens after the guard of the call that created it is gone.
  template<typename Handle>
  static std::shared_ptr<void> lease_on(int device, cudaStream_t stream, Handle handle)
  {
    return std::shared_ptr<void>(
      new Handle(std::move(handle)),
      [device, stream, owner = thread_token()](void * pointer) {
        // A deleter cannot throw, so a failed switch or wait is ignored here, as the
        // handle's own release ignores its errors.
        auto guard = executorch::extension::cuda::CUDAGuard::create(device);
        if (!guard.ok()) {
          (void)cudaGetLastError();
        }
        // The per-thread stream is a different stream on each thread, so on another
        // thread the event could complete before the work queued on the owner's stream.
        // CUDA refuses this wait while any thread records a CUDA graph, and the refusal
        // ends that recording, so the wait is retried until the recording stops.
        if (stream == cudaStreamPerThread && thread_token() != owner) {
          cudaError_t status;
          while ((status = cudaDeviceSynchronize()) == cudaErrorStreamCaptureUnsupported) {
            (void)cudaGetLastError();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
          if (status != cudaSuccess) {
            (void)cudaGetLastError();
          }
        }
        delete static_cast<Handle *>(pointer);
      });
  }

  // Unlike std::thread::id, a token is never given to a later thread.
  static std::uint64_t thread_token()
  {
    static std::atomic<std::uint64_t> next{0};
    thread_local const std::uint64_t token = next++;
    return token;
  }

  // ExecuTorch's CUDA backend runs on the caller stream when one is set and on
  // the per-thread stream otherwise, so the read and write events follow the
  // same rule and order against the program without extra synchronization.
  // A null caller stream selects the legacy default stream, while cuda_buffer
  // reads null as "no stream" and would skip its events.
  static cudaStream_t resolve_stream(void * value, int device)
  {
    cudaStream_t stream = static_cast<cudaStream_t>(value);
    if (stream == nullptr) {
      const auto caller = executorch::extension::cuda::getCallerStream();
      if (!caller) {
        stream = cudaStreamPerThread;
      } else {
        stream = *caller != nullptr ? *caller : cudaStreamLegacy;
      }
    }
    // The message's events can be recorded only on a stream of its GPU, and
    // cuda_buffer ignores the failure when the view is released. The two default
    // streams belong to the current GPU, which the caller has just selected.
    if (stream != cudaStreamPerThread && stream != cudaStreamLegacy) {
      int stream_device = -1;
      check_cuda(cudaStreamGetDevice(stream, &stream_device));
      if (stream_device != device) {
        throw std::invalid_argument(
                "CUDA plugin got a stream on device " + std::to_string(stream_device) +
                " for a message on device " + std::to_string(device));
      }
    }
    return stream;
  }

  static Device device_of(const cuda_buffer_backend::CudaBufferImpl<uint8_t> & implementation)
  {
    return Device(
      DeviceType::CUDA,
      static_cast<executorch::runtime::etensor::DeviceIndex>(implementation.get_device_id()));
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

}  // namespace executorch_conversions_cuda

PLUGINLIB_EXPORT_CLASS(
  executorch_conversions_cuda::CudaConversionPlugin,
  executorch_conversions::ConversionPlugin)
