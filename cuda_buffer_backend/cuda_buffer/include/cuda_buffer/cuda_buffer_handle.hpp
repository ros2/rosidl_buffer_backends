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
#include <vector>

#include "cuda_buffer/cuda_error.hpp"
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
/// ReadHandle is created by CudaBuffer::get_read_handle(). It waits on the
/// producer write event before exposing a const device pointer, then records a
/// read event on destruction. CudaBuffer's recycler later waits on those read
/// events before releasing or reusing the underlying storage.
/// Must not outlive the CudaBuffer that created it.
class ReadHandle
{
public:
  ReadHandle() = default;
  ReadHandle(const ReadHandle &) = delete;
  ReadHandle & operator=(const ReadHandle &) = delete;

  ReadHandle(ReadHandle && other) noexcept
  : data_ptr_(other.data_ptr_), read_events_(other.read_events_),
    events_mutex_(other.events_mutex_), stream_(other.stream_),
    promoted_buffer_(std::move(other.promoted_buffer_))
  {
    other.data_ptr_ = nullptr;
    other.read_events_ = nullptr;
    other.events_mutex_ = nullptr;
    other.stream_ = nullptr;
  }

  ReadHandle & operator=(ReadHandle && other) noexcept
  {
    if (this != &other) {
      release();
      data_ptr_ = other.data_ptr_;
      read_events_ = other.read_events_;
      events_mutex_ = other.events_mutex_;
      stream_ = other.stream_;
      promoted_buffer_ = std::move(other.promoted_buffer_);
      other.data_ptr_ = nullptr;
      other.read_events_ = nullptr;
      other.events_mutex_ = nullptr;
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
  void release() noexcept
  {
    if (!read_events_ || !cuda_is_stream_usable(stream_)) {return;}

    cudaEvent_t ev{nullptr};
    CUDA_CHECK_NOTHROW(
      cudaEventCreateWithFlags(&ev, CUDA_BUFFER_DEFAULT_EVENT_FLAGS),
      return);
    CUDA_CHECK_NOTHROW(cudaEventRecord(ev, stream_), {cudaEventDestroy(ev); return;});
    std::lock_guard<std::mutex> lg(*events_mutex_);
    read_events_->push_back(ev);
  }

  friend class CudaBuffer;

  explicit ReadHandle(
    const uint8_t * data_ptr,
    cudaEvent_t write_event,
    std::vector<cudaEvent_t> * read_events,
    std::mutex * events_mutex,
    cudaStream_t stream)
  : data_ptr_(data_ptr), read_events_(read_events),
    events_mutex_(events_mutex), stream_(stream)
  {
    if (write_event != nullptr && stream_ != nullptr) {
      CUDA_CHECK(cudaStreamWaitEvent(stream_, write_event, 0));
    }
  }

  const uint8_t * data_ptr_{nullptr};
  std::vector<cudaEvent_t> * read_events_{nullptr};
  std::mutex * events_mutex_{nullptr};
  cudaStream_t stream_{nullptr};
  std::shared_ptr<rosidl::Buffer<uint8_t>> promoted_buffer_{};
};

/// \brief Scoped write access to a CudaBuffer.
///
/// WriteHandle is created by CudaBuffer::get_write_handle(). It exposes a
/// mutable device pointer and records the producer write event on destruction,
/// making later ReadHandles wait for the correct CUDA stream work. Only one
/// write handle may be active for a buffer.
/// Must not outlive the CudaBuffer that created it.
class WriteHandle
{
public:
  WriteHandle(const WriteHandle &) = delete;
  WriteHandle & operator=(const WriteHandle &) = delete;

  WriteHandle(WriteHandle && other) noexcept
  : data_ptr_(other.data_ptr_), write_event_ptr_(other.write_event_ptr_),
    stream_(other.stream_), state_(std::move(other.state_)),
    promoted_buffer_(std::move(other.promoted_buffer_))
  {
    other.data_ptr_ = nullptr;
    other.write_event_ptr_ = nullptr;
    other.stream_ = nullptr;
  }

  WriteHandle & operator=(WriteHandle && other) noexcept
  {
    if (this != &other) {
      release();
      data_ptr_ = other.data_ptr_;
      write_event_ptr_ = other.write_event_ptr_;
      stream_ = other.stream_;
      state_ = std::move(other.state_);
      promoted_buffer_ = std::move(other.promoted_buffer_);
      other.data_ptr_ = nullptr;
      other.write_event_ptr_ = nullptr;
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
  void release() noexcept
  {
    if (!state_) {return;}
    std::lock_guard<std::mutex> lock(state_->mtx);
    if (state_->state == HandleState::State::Finalized) {return;}
    cudaStream_t ws = state_->write_stream ? state_->write_stream : stream_;
    if (cuda_is_stream_usable(ws) && write_event_ptr_ && *write_event_ptr_) {
      CUDA_CHECK_NOTHROW(cudaEventRecord(*write_event_ptr_, ws), (void)0);
    }
    state_->state = HandleState::State::Finalized;
    state_->write_stream = nullptr;
  }

  friend class CudaBuffer;

  explicit WriteHandle(
    uint8_t * data_ptr,
    cudaEvent_t * write_event_ptr,
    cudaStream_t stream,
    std::shared_ptr<HandleState> state)
  : data_ptr_(data_ptr), write_event_ptr_(write_event_ptr), stream_(stream),
    state_(state)
  {
  }

  uint8_t * data_ptr_{nullptr};
  cudaEvent_t * write_event_ptr_{nullptr};
  cudaStream_t stream_{nullptr};
  std::shared_ptr<HandleState> state_{nullptr};
  std::shared_ptr<rosidl::Buffer<uint8_t>> promoted_buffer_{};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_HANDLE_HPP_
