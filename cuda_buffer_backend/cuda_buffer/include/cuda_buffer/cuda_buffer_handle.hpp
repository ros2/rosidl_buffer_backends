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

#ifndef CUDA_BUFFER__CUDA_BUFFER_HANDLE_HPP_
#define CUDA_BUFFER__CUDA_BUFFER_HANDLE_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>

#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/visibility_control.h"
#include "rosidl_buffer/buffer.hpp"

namespace cuda_buffer_backend
{

class CudaBuffer;

constexpr unsigned int CUDA_BUFFER_MINIMUM_EVENT_FLAGS =
  cudaEventBlockingSync | cudaEventDisableTiming;
constexpr unsigned int CUDA_BUFFER_DEFAULT_EVENT_FLAGS =
  CUDA_BUFFER_MINIMUM_EVENT_FLAGS | cudaEventInterprocess;

inline bool cuda_is_stream_usable(cudaStream_t s)
{
  if (s == nullptr) {return false;}
  const cudaError_t st = cudaStreamQuery(s);
  return (st == cudaSuccess) || (st == cudaErrorNotReady);
}

struct HandleState
{
  enum class State
  {
    Unset,
    InUse,
    Finalized
  };

  std::mutex mtx;
  State state{State::Unset};
  cudaStream_t write_stream{nullptr};
};

/// \brief Scoped read access to a CudaBuffer.
///
/// ReadHandle is created by CudaBuffer::get_read_handle(), which makes the
/// reader's stream wait for the producer's write before exposing a const
/// device pointer. On destruction the handle tells the buffer that the read
/// has been enqueued, so the storage backend can keep the memory alive until
/// that work completes.
/// Must not outlive the CudaBuffer that created it. CudaBufferImpl::resize()
/// and CudaBufferImpl::clear() replace that CudaBuffer, so the handle must be
/// released before either is called.
class ReadHandle
{
public:
  ReadHandle() = default;
  ReadHandle(const ReadHandle &) = delete;
  ReadHandle & operator=(const ReadHandle &) = delete;

  ReadHandle(ReadHandle && other) noexcept
  : data_ptr_(other.data_ptr_), owner_(other.owner_), stream_(other.stream_),
    promoted_buffer_(std::move(other.promoted_buffer_))
  {
    other.data_ptr_ = nullptr;
    other.owner_ = nullptr;
    other.stream_ = nullptr;
  }

  ReadHandle & operator=(ReadHandle && other) noexcept
  {
    if (this != &other) {
      release();
      data_ptr_ = other.data_ptr_;
      owner_ = other.owner_;
      stream_ = other.stream_;
      promoted_buffer_ = std::move(other.promoted_buffer_);
      other.data_ptr_ = nullptr;
      other.owner_ = nullptr;
      other.stream_ = nullptr;
    }
    return *this;
  }

  ~ReadHandle()
  {
    release();
  }

  const uint8_t * get_ptr() const {return data_ptr_;}

  /// \brief If set, this handle owns a newly-allocated CUDA buffer that was created
  /// to promote a non-CUDA source buffer (e.g. CPU-backed) in from_buffer.
  /// The promoted buffer stays alive as long as the handle (or a shared copy) lives.
  std::shared_ptr<rosidl::Buffer<uint8_t>> get_promoted_buffer() const
  {
    return promoted_buffer_;
  }

  void set_promoted_buffer(std::shared_ptr<rosidl::Buffer<uint8_t>> buffer)
  {
    promoted_buffer_ = std::move(buffer);
  }

private:
  /// Report the end of this read to the owning buffer. Defined in cuda_buffer.cpp.
  CUDA_BUFFER_PUBLIC void release() noexcept;

  friend class CudaBuffer;

  explicit ReadHandle(
    const uint8_t * data_ptr,
    const CudaBuffer * owner,
    cudaStream_t stream)
  : data_ptr_(data_ptr), owner_(owner), stream_(stream)
  {
  }

  const uint8_t * data_ptr_{nullptr};
  const CudaBuffer * owner_{nullptr};
  cudaStream_t stream_{nullptr};
  std::shared_ptr<rosidl::Buffer<uint8_t>> promoted_buffer_{};
};

/// \brief Scoped write access to a CudaBuffer.
///
/// WriteHandle is created by CudaBuffer::get_write_handle(). It exposes a
/// mutable device pointer and, on destruction, tells the buffer that the
/// producer's work has been enqueued, making later ReadHandles wait for the
/// correct CUDA stream work. Only one write handle may be active for a buffer.
/// Must not outlive the CudaBuffer that created it, unless the write was
/// already finalized (for example by publishing the buffer). CudaBufferImpl::resize()
/// and CudaBufferImpl::clear() replace that CudaBuffer, so the handle must be
/// released before either is called.
class WriteHandle
{
public:
  WriteHandle(const WriteHandle &) = delete;
  WriteHandle & operator=(const WriteHandle &) = delete;

  WriteHandle(WriteHandle && other) noexcept
  : data_ptr_(other.data_ptr_), owner_(other.owner_),
    stream_(other.stream_), state_(std::move(other.state_)),
    promoted_buffer_(std::move(other.promoted_buffer_))
  {
    other.data_ptr_ = nullptr;
    other.owner_ = nullptr;
    other.stream_ = nullptr;
  }

  WriteHandle & operator=(WriteHandle && other) noexcept
  {
    if (this != &other) {
      release();
      data_ptr_ = other.data_ptr_;
      owner_ = other.owner_;
      stream_ = other.stream_;
      state_ = std::move(other.state_);
      promoted_buffer_ = std::move(other.promoted_buffer_);
      other.data_ptr_ = nullptr;
      other.owner_ = nullptr;
      other.stream_ = nullptr;
    }
    return *this;
  }

  ~WriteHandle()
  {
    release();
  }

  uint8_t * get_ptr() {return data_ptr_;}

  /// \brief If set, this handle owns a newly-allocated CUDA buffer that was created
  /// to promote a non-CUDA source buffer (e.g. CPU-backed) in from_buffer.
  /// The promoted buffer stays alive as long as the handle (or a shared copy) lives.
  std::shared_ptr<rosidl::Buffer<uint8_t>> get_promoted_buffer() const
  {
    return promoted_buffer_;
  }

  void set_promoted_buffer(std::shared_ptr<rosidl::Buffer<uint8_t>> buffer)
  {
    promoted_buffer_ = std::move(buffer);
  }

private:
  /// Finalize this write on the owning buffer. Defined in cuda_buffer.cpp.
  CUDA_BUFFER_PUBLIC void release() noexcept;

  friend class CudaBuffer;

  explicit WriteHandle(
    uint8_t * data_ptr,
    const CudaBuffer * owner,
    cudaStream_t stream,
    std::shared_ptr<HandleState> state)
  : data_ptr_(data_ptr), owner_(owner), stream_(stream),
    state_(state)
  {
  }

  uint8_t * data_ptr_{nullptr};
  const CudaBuffer * owner_{nullptr};
  cudaStream_t stream_{nullptr};
  std::shared_ptr<HandleState> state_{nullptr};
  std::shared_ptr<rosidl::Buffer<uint8_t>> promoted_buffer_{};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_HANDLE_HPP_
