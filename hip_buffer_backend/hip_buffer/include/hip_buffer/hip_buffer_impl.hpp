// Copyright 2026 Open Source Robotics Foundation, Inc.
// Modified for the HIP port by ObeliskGate, 2026.
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

#ifndef HIP_BUFFER__HIP_BUFFER_IMPL_HPP_
#define HIP_BUFFER__HIP_BUFFER_IMPL_HPP_

#include <rcutils/logging_macros.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "hip_buffer/hip_buffer.hpp"
#include "hip_buffer/hip_error.hpp"
#include "hip_buffer/hip_memory_pool.hpp"
#include "hip_buffer/visibility_control.h"
#include "rosidl_buffer/buffer.hpp"
#include "rosidl_buffer/buffer_impl_base.hpp"
#include "rosidl_buffer/cpu_buffer_impl.hpp"

namespace hip_buffer_backend
{

/// Process-wide VMM allocation pool. Defined out-of-line so backend plugins
/// and application DSOs resolve allocations through the same pool instance.
HIP_BUFFER_PUBLIC std::shared_ptr<HipMemoryPool> get_or_create_global_pool();

/// Configure the process-wide pool before its first successful initialization.
/// Throws std::logic_error after the global pool has been initialized.
HIP_BUFFER_PUBLIC void configure_global_pool(const HipPoolConfig & config);

// Process-wide stream for internal ops (clone, to_cpu, resize).
// Intentionally leaked; the HIP runtime releases it at process exit.
inline hipStream_t get_internal_stream()
{
  static hipStream_t s = [] {
      hipStream_t stream = nullptr;
      hipError_t err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
      if (err != hipSuccess) {
        RCUTILS_LOG_WARN_NAMED("hip_buffer_backend",
          "Failed to create internal HIP stream (%s); "
          "clone/resize/to_cpu will use the default (synchronizing) stream",
          hipGetErrorName(err));
        (void)hipGetLastError();
      }
      return stream;
    }();
  return s;
}

template<typename T>
class HipBufferImpl : public rosidl::BufferImplBase<T>
{
public:
  HipBufferImpl()
  : size_(0) {}

  explicit HipBufferImpl(size_t size)
  : size_(size)
  {
    if (size_ > 0) {
      allocate_buffer(size_);
    }
  }

  explicit HipBufferImpl(HipBuffer && buffer, size_t size)
  : size_(size), hip_buffer_(std::move(buffer)) {}

  ~HipBufferImpl() = default;

  HipBufferImpl(const HipBufferImpl &) = delete;
  HipBufferImpl & operator=(const HipBufferImpl &) = delete;
  HipBufferImpl(HipBufferImpl &&) = delete;
  HipBufferImpl & operator=(HipBufferImpl &&) = delete;

  std::string get_backend_type() const override {return "hip";}

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

    HipBuffer new_buffer;
    allocate_buffer_internal(new_buffer, n);

    if (size_ > 0 && hip_buffer_.size() > 0) {
      hipStream_t s = stream_ ? stream_ : get_internal_stream();
      size_t copy_size = std::min(n, size_) * sizeof(T);
      ReadHandle rh = hip_buffer_.get_read_handle(s);
      WriteHandle wh = new_buffer.get_write_handle(s);
      HIP_CHECK(hipMemcpyAsync(
        wh.get_ptr(), rh.get_ptr(),
        copy_size, hipMemcpyDeviceToDevice, s));
    }

    hip_buffer_ = std::move(new_buffer);
    size_ = n;
  }

  void clear()
  {
    hip_buffer_ = HipBuffer();
    size_ = 0;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> to_cpu() const override
  {
    auto cpu = std::make_unique<rosidl::CpuBufferImpl<T>>();
    cpu->get_storage().resize(size_);

    if (size_ > 0 && hip_buffer_.size() > 0) {
      hipStream_t s = stream_ ? stream_ : get_internal_stream();
      ReadHandle rh = hip_buffer_.get_read_handle(s);
      HIP_CHECK(hipMemcpyAsync(
        cpu->get_storage().data(), rh.get_ptr(),
        size_ * sizeof(T), hipMemcpyDeviceToHost, s));
      HIP_CHECK(hipStreamSynchronize(s));
    }

    return cpu;
  }

  std::unique_ptr<rosidl::BufferImplBase<T>> clone() const override
  {
    auto copy = std::make_unique<HipBufferImpl<T>>(size_);

    if (size_ > 0 && hip_buffer_.size() > 0) {
      hipStream_t s = stream_ ? stream_ : get_internal_stream();
      ReadHandle rh = hip_buffer_.get_read_handle(s);
      WriteHandle wh = copy->hip_buffer_.get_write_handle(s);
      HIP_CHECK(hipMemcpyAsync(
        wh.get_ptr(), rh.get_ptr(),
        size_ * sizeof(T), hipMemcpyDeviceToDevice, s));
    }

    return copy;
  }

  HipBuffer & get_hip_buffer() {return hip_buffer_;}
  const HipBuffer & get_hip_buffer() const {return hip_buffer_;}

  void set_stream(hipStream_t stream) {stream_ = stream;}
  hipStream_t get_stream() const {return stream_;}
  int get_device_id() const {return hip_buffer_.get_device_id();}

  static std::shared_ptr<HipMemoryPool> get_or_create_global_pool()
  {
    return hip_buffer_backend::get_or_create_global_pool();
  }

  static bool is_pool_ipc_capable()
  {
    try {
      auto pool = get_or_create_global_pool();
      return pool && pool->is_ipc_capable();
    } catch (const HipError &) {
      return false;
    }
  }

private:
  void allocate_buffer(size_t n)
  {
    allocate_buffer_internal(hip_buffer_, n);
  }

  void allocate_buffer_internal(HipBuffer & buffer, size_t n)
  {
    auto pool = get_or_create_global_pool();
    size_t byte_size = n * sizeof(T);

    VmmBlock * block = pool->allocate(byte_size);

    hipEvent_t ev = nullptr;
    hipError_t ev_err = hipEventCreateWithFlags(&ev, HIP_BUFFER_DEFAULT_EVENT_FLAGS);
    if (ev_err != hipSuccess) {
      const hipError_t interprocess_event_error = ev_err;
      (void)hipGetLastError();
      ev_err = hipEventCreateWithFlags(&ev, HIP_BUFFER_MINIMUM_EVENT_FLAGS);
      if (ev_err != hipSuccess) {
        ev = nullptr;
        (void)hipGetLastError();
        RCUTILS_LOG_WARN_NAMED("hip_buffer_backend",
          "Failed to create HIP event; stream ordering disabled for this buffer");
      } else {
        RCUTILS_LOG_WARN_ONCE_NAMED(
          "hip_buffer_backend",
          "Failed to create interprocess HIP event (%s); falling back to a local event",
          hipGetErrorName(interprocess_event_error));
      }
    }

    buffer = HipBuffer(
      block->va, byte_size, pool->get_device_id(),
      pool->deleter(block));

    if (ev) {
      buffer.set_write_event(ev, true);
    }
  }

  size_t size_;
  HipBuffer hip_buffer_;
  hipStream_t stream_{nullptr};
};

}  // namespace hip_buffer_backend

#endif  // HIP_BUFFER__HIP_BUFFER_IMPL_HPP_
