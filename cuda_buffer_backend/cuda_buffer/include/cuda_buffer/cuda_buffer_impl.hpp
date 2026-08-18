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

#ifndef CUDA_BUFFER__CUDA_BUFFER_IMPL_HPP_
#define CUDA_BUFFER__CUDA_BUFFER_IMPL_HPP_

#include <rcutils/logging_macros.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/cuda_memory_pool.hpp"
#include "cuda_buffer/visibility_control.h"
#include "rosidl_buffer/buffer.hpp"
#include "rosidl_buffer/buffer_impl_base.hpp"
#include "rosidl_buffer/cpu_buffer_impl.hpp"

namespace cuda_buffer_backend
{

/// Process-wide VMM allocation pool. Defined out-of-line so backend plugins
/// and application DSOs resolve allocations through the same pool instance.
CUDA_BUFFER_PUBLIC std::shared_ptr<CudaMemoryPool> get_or_create_global_pool();

// Process-wide stream for internal ops (clone, to_cpu, resize).
// Intentionally leaked; destroyed by cudaDeviceReset in ~CudaMemoryPool.
inline cudaStream_t get_internal_stream()
{
  static cudaStream_t s = [] {
      cudaStream_t stream = nullptr;
      cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      if (err != cudaSuccess) {
        RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
          "Failed to create internal CUDA stream (%s); "
          "clone/resize/to_cpu will use the default (synchronizing) stream",
          cudaGetErrorName(err));
        (void)cudaGetLastError();
      }
      return stream;
    }();
  return s;
}

template<typename T>
class CudaBufferImpl : public rosidl::BufferImplBase<T>
{
public:
  CudaBufferImpl()
  : size_(0) {}

  explicit CudaBufferImpl(size_t size)
  : size_(size)
  {
    if (size_ > 0) {
      allocate_buffer(size_);
    }
  }

  explicit CudaBufferImpl(CudaBuffer && buffer, size_t size)
  : size_(size), cuda_buffer_(std::move(buffer)) {}

  ~CudaBufferImpl() = default;

  CudaBufferImpl(const CudaBufferImpl &) = delete;
  CudaBufferImpl & operator=(const CudaBufferImpl &) = delete;
  CudaBufferImpl(CudaBufferImpl &&) = delete;
  CudaBufferImpl & operator=(CudaBufferImpl &&) = delete;

  std::string get_backend_type() const override {return "cuda";}

  size_t size() const override {return size_;}

  void resize(size_t n)
  {
    if (n == size_) {
      return;
    }

    if (n == 0) {
      clear();
      return;
    }

    CudaBuffer new_buffer;
    allocate_buffer_internal(new_buffer, n);

    if (size_ > 0 && cuda_buffer_.size() > 0) {
      cudaStream_t s = stream_ ? stream_ : get_internal_stream();
      size_t copy_size = std::min(n, size_) * sizeof(T);
      ReadHandle rh = cuda_buffer_.get_read_handle(s);
      WriteHandle wh = new_buffer.get_write_handle(s);
      CUDA_CHECK(cudaMemcpyAsync(
        wh.get_ptr(), rh.get_ptr(),
        copy_size, cudaMemcpyDeviceToDevice, s));
    }

    cuda_buffer_ = std::move(new_buffer);
    size_ = n;
  }

  void clear()
  {
    cuda_buffer_ = CudaBuffer();
    size_ = 0;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> to_cpu() const override
  {
    auto cpu = std::make_unique<rosidl::CpuBufferImpl<T>>();
    cpu->get_storage().resize(size_);

    if (size_ > 0 && cuda_buffer_.size() > 0) {
      cudaStream_t s = stream_ ? stream_ : get_internal_stream();
      ReadHandle rh = cuda_buffer_.get_read_handle(s);
      CUDA_CHECK(cudaMemcpyAsync(
        cpu->get_storage().data(), rh.get_ptr(),
        size_ * sizeof(T), cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
    }

    return cpu;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> clone() const override
  {
    auto copy = std::make_unique<CudaBufferImpl<T>>(size_);

    if (size_ > 0 && cuda_buffer_.size() > 0) {
      cudaStream_t s = stream_ ? stream_ : get_internal_stream();
      ReadHandle rh = cuda_buffer_.get_read_handle(s);
      WriteHandle wh = copy->cuda_buffer_.get_write_handle(s);
      CUDA_CHECK(cudaMemcpyAsync(
        wh.get_ptr(), rh.get_ptr(),
        size_ * sizeof(T), cudaMemcpyDeviceToDevice, s));
    }

    return copy;
  }

  CudaBuffer & get_cuda_buffer() {return cuda_buffer_;}
  const CudaBuffer & get_cuda_buffer() const {return cuda_buffer_;}

  void set_stream(cudaStream_t stream) {stream_ = stream;}
  cudaStream_t get_stream() const {return stream_;}
  int get_device_id() const {return cuda_buffer_.get_device_id();}

  static std::shared_ptr<CudaMemoryPool> get_or_create_global_pool()
  {
    return cuda_buffer_backend::get_or_create_global_pool();
  }

  static bool is_pool_ipc_capable()
  {
    auto pool = get_or_create_global_pool();
    return pool && pool->is_ipc_capable();
  }

private:
  void allocate_buffer(size_t n)
  {
    allocate_buffer_internal(cuda_buffer_, n);
  }

  void allocate_buffer_internal(CudaBuffer & buffer, size_t n)
  {
    auto pool = get_or_create_global_pool();
    size_t byte_size = n * sizeof(T);

    VmmBlock * block = pool->allocate(byte_size);

    cudaEvent_t ev = nullptr;
    cudaError_t ev_err = cudaEventCreateWithFlags(&ev, CUDA_BUFFER_EVENT_FLAGS);
    if (ev_err != cudaSuccess) {
      (void)cudaGetLastError();
      ev_err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
      if (ev_err != cudaSuccess) {
        ev = nullptr;
        (void)cudaGetLastError();
        RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
          "Failed to create CUDA event; stream ordering disabled for this buffer");
      }
    }

    buffer = CudaBuffer(
      reinterpret_cast<void *>(block->va), byte_size, pool->get_device_id(),
      pool->deleter(block));

    if (ev) {
      buffer.set_write_event(ev, true);
    }
  }

  size_t size_;
  CudaBuffer cuda_buffer_;
  cudaStream_t stream_{nullptr};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_IMPL_HPP_
