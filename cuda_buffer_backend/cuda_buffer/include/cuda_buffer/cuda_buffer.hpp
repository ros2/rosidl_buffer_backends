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

#ifndef CUDA_BUFFER__CUDA_BUFFER_HPP_
#define CUDA_BUFFER__CUDA_BUFFER_HPP_

#include <cuda_runtime.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_handle.hpp"
#include "cuda_buffer/cuda_error.hpp"

namespace cuda_buffer_backend
{

/// \brief Internal CUDA storage object owned by CudaBufferImpl.
///
/// The public rosidl buffer hierarchy is:
///   rosidl::Buffer<T> -> rosidl::BufferImplBase<T> -> CudaBufferImpl<T>.
/// CudaBuffer is not part of that hierarchy. It is the
/// low-level RAII holder for the CUDA device pointer, device id, write/read
/// event state, and deferred recycler used by CudaBufferImpl.
class CudaBuffer
{
public:
  CudaBuffer() = default;

  CudaBuffer(CudaBuffer && other) noexcept
  : device_ptr_(std::move(other.device_ptr_)),
    size_(other.size_),
    device_id_(other.device_id_),
    write_event_(other.write_event_),
    owns_write_event_(other.owns_write_event_),
    read_events_(std::move(other.read_events_)),
    handle_state_(std::move(other.handle_state_)),
    recycler_(std::move(other.recycler_)),
    adopted_(other.adopted_)
  {
    other.size_ = 0;
    other.device_id_ = 0;
    other.write_event_ = nullptr;
    other.owns_write_event_ = false;
    other.adopted_ = false;
  }

  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer & operator=(const CudaBuffer &) = delete;

  CudaBuffer & operator=(CudaBuffer && other) noexcept
  {
    if (this != &other) {
      std::swap(device_ptr_, other.device_ptr_);
      std::swap(size_, other.size_);
      std::swap(device_id_, other.device_id_);
      std::swap(write_event_, other.write_event_);
      std::swap(owns_write_event_, other.owns_write_event_);
      std::swap(read_events_, other.read_events_);
      std::swap(handle_state_, other.handle_state_);
      std::swap(recycler_, other.recycler_);
      std::swap(adopted_, other.adopted_);
    }
    return *this;
  }

  ~CudaBuffer();

  CudaBuffer(
    void * ptr, size_t size, int device_id,
    std::function<void(uint8_t *)> custom_deleter);

  /// \brief Wrap device memory that belongs to somebody else.
  ///
  /// For memory allocated outside this backend and mapped into CUDA by its
  /// owner -- a zero-copy transport's shared slot, an NvSciBuf attachment, a
  /// graphics interop surface. The result behaves exactly like a pool-allocated
  /// CudaBuffer for every read and write, and differs in what it will not do:
  /// it never frees the storage and never reallocates it.
  ///
  /// \param keepalive Held until the GPU is finished with this memory, then
  ///   released. Whatever the owner uses to mean "still in use" goes here -- a
  ///   slot refcount pin, a shared_ptr to the owning object. May be null when
  ///   the storage outlives this buffer by construction.
  static CudaBuffer adopt(
    void * ptr, size_t size, int device_id, std::shared_ptr<void> keepalive);

  /// \brief True when the storage came from adopt() rather than the pool.
  ///
  /// The one thing callers have to branch on: adopted storage cannot be
  /// reallocated, because it is not this object's to reallocate.
  bool is_adopted() const {return adopted_;}

  ReadHandle get_read_handle(cudaStream_t stream) const;
  WriteHandle get_write_handle(cudaStream_t stream);

  void set_write_event(cudaEvent_t event, bool owns_event = false)
  {
    write_event_ = event;
    owns_write_event_ = owns_event;
  }

  cudaEvent_t get_write_event() const {return write_event_;}

  void finalize_write_handle() const;

  size_t size() const {return size_;}
  int get_device_id() const {return device_id_;}
  uint8_t * get_device_ptr() {return device_ptr_.get();}
  const uint8_t * get_device_ptr() const {return device_ptr_.get();}

private:
  class BufferRecycler;

  void reap_completed_read_events() const;
  void finalize_write_handle_locked() const;
  static void default_cuda_free(uint8_t * p);

  std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> device_ptr_{
    nullptr, default_cuda_free
  };
  size_t size_{0};
  int device_id_{0};

  mutable cudaEvent_t write_event_{nullptr};
  bool owns_write_event_{false};
  mutable std::vector<cudaEvent_t> read_events_;
  mutable std::mutex events_mutex_;

  std::shared_ptr<HandleState> handle_state_{nullptr};
  std::shared_ptr<BufferRecycler> recycler_;

  /// Metadata only -- the keepalive that makes adoption safe rides in the
  /// deleter, not here. See CudaBuffer::adopt().
  bool adopted_{false};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_HPP_
