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
#include "cuda_buffer/external_memory_pool.hpp"
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

/// \brief Create the write event a CudaBuffer uses to order its readers.
///
/// Interprocess-capable when the driver allows it, local otherwise, and null
/// when neither works: a buffer with no event still reads and writes correctly,
/// it just cannot order a later reader behind an earlier writer.
///
/// Shared by the pool allocator and by adoption, so that memory this backend
/// did not allocate is ordered exactly like memory it did.
inline cudaEvent_t make_buffer_write_event()
{
  cudaEvent_t ev = nullptr;
  cudaError_t ev_err = cudaEventCreateWithFlags(&ev, CUDA_BUFFER_DEFAULT_EVENT_FLAGS);
  if (ev_err != cudaSuccess) {
    const cudaError_t interprocess_event_error = ev_err;
    (void)cudaGetLastError();
    ev_err = cudaEventCreateWithFlags(&ev, CUDA_BUFFER_MINIMUM_EVENT_FLAGS);
    if (ev_err != cudaSuccess) {
      ev = nullptr;
      (void)cudaGetLastError();
      RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
        "Failed to create CUDA event; stream ordering disabled for this buffer");
    } else {
      RCUTILS_LOG_WARN_ONCE_NAMED(
        "cuda_buffer_backend",
        "Failed to create interprocess CUDA event (%s); falling back to a local event",
        cudaGetErrorName(interprocess_event_error));
    }
  }
  return ev;
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

  /// \brief Allocate \p size elements out of a caller-owned device region.
  ///
  /// The pool-allocating constructor above takes its memory from this process's
  /// global VMM pool. This one takes it from an ExternalMemoryPool -- a region
  /// somebody else allocated and mapped, an NvSciBuf transport arena being the
  /// motivating case -- and is otherwise identical: same events, same handles,
  /// same backend name.
  ///
  /// The pool reference is kept so that a later resize() can grow *within the
  /// same region*. That is the one behavioural difference from adoption, and it
  /// is the reason a pool is worth having: adopted storage is a single block
  /// with nowhere to grow, whereas a pool has the rest of its arena.
  ///
  /// \throw CudaError if \p pool is null, or has no run of \p size elements
  ///   left.
  CudaBufferImpl(std::shared_ptr<ExternalMemoryPool> pool, size_t size)
  : size_(size), external_pool_(std::move(pool))
  {
    if (!external_pool_) {
      throw CudaError("CudaBufferImpl: null ExternalMemoryPool");
    }
    if (size_ > 0) {
      allocate_buffer(size_);
    }
  }

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

    // Adopted storage is not ours to reallocate. Growing would need memory to
    // grow into, which is exactly what a buffer in this position does not have;
    // shrinking is only a change to the reported count, so it is allowed and
    // costs nothing. Reallocating here instead would silently swap the caller's
    // storage -- a transport slot, say -- for a pool block, and every later
    // "zero copy" would quietly be a copy.
    if (cuda_buffer_.is_adopted()) {
      if (!shrink(n)) {
        throw CudaError(
                "CudaBufferImpl::resize: cannot grow adopted storage; "
                "the allocation belongs to whoever lent it");
      }
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

  /// \brief Report fewer elements than the storage holds, without touching it.
  ///
  /// For a producer handed fixed-size storage that filled less than all of it.
  /// Neither reallocates nor copies, so the device pointer, the event state and
  /// any adoption survive unchanged; only the count this buffer reports moves.
  ///
  /// Growing is refused rather than reallocating, because the caller that needs
  /// this is precisely the one whose storage is not its own.
  ///
  /// \return true if the count is now \p n; false if \p n exceeds the current
  ///         size, leaving the buffer untouched.
  bool shrink(size_t n)
  {
    if (n > size_) {
      return false;
    }
    size_ = n;
    return true;
  }

  /// \brief True when this buffer wraps storage allocated outside the backend.
  bool is_adopted() const {return cuda_buffer_.is_adopted();}

  /// \brief True when this buffer was suballocated from an ExternalMemoryPool.
  ///
  /// Distinct from is_adopted(): both name memory the backend did not allocate
  /// from the driver, but an adopted buffer is a single block it cannot grow,
  /// whereas this one has the rest of its pool to grow into.
  bool is_external() const {return external_pool_ != nullptr;}

  /// \brief The pool this buffer was suballocated from, or null.
  const std::shared_ptr<ExternalMemoryPool> & get_external_pool() const
  {
    return external_pool_;
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

  /// \brief Copy into storage of our own.
  ///
  /// Deliberately the *global* pool even when this buffer came from an external
  /// one. A clone is what a caller asks for when it needs the data to outlive
  /// the original, and the original's arena is precisely what it needs to
  /// outlive: cloning back into a transport's slot pool would consume a slot to
  /// escape a slot. The copy is ordinary CUDA memory with no borrowed lifetime.
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
    size_t byte_size = n * sizeof(T);

    if (external_pool_) {
      allocate_from_external(buffer, byte_size);
      return;
    }

    auto pool = get_or_create_global_pool();

    VmmBlock * block = pool->allocate(byte_size);

    cudaEvent_t ev = make_buffer_write_event();

    buffer = CudaBuffer(
      reinterpret_cast<void *>(block->va), byte_size, pool->get_device_id(),
      pool->deleter(block));

    if (ev) {
      buffer.set_write_event(ev, true);
    }
  }

  /// Carve \p byte_size out of the external pool and wrap it.
  ///
  /// The resulting CudaBuffer is *not* marked adopted. Adoption means "this
  /// block is not mine to reallocate"; a pool block is, because the pool can
  /// hand out a different one. What both share is that the deleter frees
  /// nothing to the driver -- here it returns the block to the free list.
  void allocate_from_external(CudaBuffer & buffer, size_t byte_size)
  {
    ExternalBlock * block = external_pool_->allocate(byte_size);
    if (block == nullptr) {
      // Exhaustion is reported by the pool as null because it is ordinary for a
      // fixed arena, but by the time a buffer is being constructed the caller
      // has asked for storage and there is no half-answer to give it.
      throw CudaError(
              "CudaBufferImpl: external memory pool is exhausted; "
              "no contiguous run left for this allocation");
    }

    cudaEvent_t ev = make_buffer_write_event();

    buffer = CudaBuffer(
      block->ptr, byte_size, external_pool_->get_device_id(),
      external_pool_->deleter(block));

    if (ev) {
      buffer.set_write_event(ev, true);
    }
  }

  size_t size_;
  CudaBuffer cuda_buffer_;
  cudaStream_t stream_{nullptr};

  /// Null for an ordinary buffer; set when this one was suballocated from a
  /// caller-owned region. Held rather than merely consulted at construction so
  /// that resize() can grow within the same region.
  std::shared_ptr<ExternalMemoryPool> external_pool_;
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_IMPL_HPP_
