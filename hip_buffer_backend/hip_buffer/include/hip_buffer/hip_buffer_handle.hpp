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

#ifndef HIP_BUFFER__HIP_BUFFER_HANDLE_HPP_
#define HIP_BUFFER__HIP_BUFFER_HANDLE_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "hip_buffer/hip_error.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace hip_buffer_backend
{

class HipBuffer;

constexpr unsigned int HIP_BUFFER_MINIMUM_EVENT_FLAGS =
  hipEventBlockingSync | hipEventDisableTiming;
constexpr unsigned int HIP_BUFFER_DEFAULT_EVENT_FLAGS =
  HIP_BUFFER_MINIMUM_EVENT_FLAGS | hipEventInterprocess;

inline bool hip_is_stream_usable(hipStream_t s)
{
  if (s == nullptr) {return false;}
  const hipError_t st = hipStreamQuery(s);
  return (st == hipSuccess) || (st == hipErrorNotReady);
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
  hipStream_t write_stream{nullptr};
};

/// \brief Scoped read access to a HipBuffer.
///
/// ReadHandle is created by HipBuffer::get_read_handle(). It waits on the
/// producer write event before exposing a const device pointer, then records a
/// read event on destruction. HipBuffer's recycler later waits on those read
/// events before releasing or reusing the underlying storage.
/// Must not outlive the HipBuffer that created it.
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

  /// \brief If set, this handle owns a newly-allocated HIP buffer that was created
  /// to promote a non-HIP source buffer (e.g. CPU-backed) in from_buffer.
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
    if (!read_events_ || !hip_is_stream_usable(stream_)) {return;}

    // Read events are only queried and synchronized in this process, never exported.
    hipEvent_t ev{nullptr};
    HIP_CHECK_NOTHROW(
      hipEventCreateWithFlags(&ev, HIP_BUFFER_MINIMUM_EVENT_FLAGS),
      return);
    HIP_CHECK_NOTHROW(hipEventRecord(ev, stream_), {hipEventDestroy(ev); return;});
    std::lock_guard<std::mutex> lg(*events_mutex_);
    read_events_->push_back(ev);
  }

  friend class HipBuffer;

  explicit ReadHandle(
    const uint8_t * data_ptr,
    hipEvent_t write_event,
    std::vector<hipEvent_t> * read_events,
    std::mutex * events_mutex,
    hipStream_t stream)
  : data_ptr_(data_ptr), read_events_(read_events),
    events_mutex_(events_mutex), stream_(stream)
  {
    if (write_event != nullptr && stream_ != nullptr) {
      HIP_CHECK(hipStreamWaitEvent(stream_, write_event, 0));
    }
  }

  const uint8_t * data_ptr_{nullptr};
  std::vector<hipEvent_t> * read_events_{nullptr};
  std::mutex * events_mutex_{nullptr};
  hipStream_t stream_{nullptr};
  std::shared_ptr<rosidl::Buffer<uint8_t>> promoted_buffer_{};
};

/// \brief Scoped write access to a HipBuffer.
///
/// WriteHandle is created by HipBuffer::get_write_handle(). It exposes a
/// mutable device pointer and records the producer write event on destruction,
/// making later ReadHandles wait for the correct HIP stream work. Only one
/// write handle may be active for a buffer.
/// Must not outlive the HipBuffer that created it.
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

  /// \brief If set, this handle owns a newly-allocated HIP buffer that was created
  /// to promote a non-HIP source buffer (e.g. CPU-backed) in from_buffer.
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
    hipStream_t ws = state_->write_stream ? state_->write_stream : stream_;
    if (hip_is_stream_usable(ws) && write_event_ptr_ && *write_event_ptr_) {
      HIP_CHECK_NOTHROW(hipEventRecord(*write_event_ptr_, ws), (void)0);
    }
    state_->state = HandleState::State::Finalized;
    state_->write_stream = nullptr;
  }

  friend class HipBuffer;

  explicit WriteHandle(
    uint8_t * data_ptr,
    hipEvent_t * write_event_ptr,
    hipStream_t stream,
    std::shared_ptr<HandleState> state)
  : data_ptr_(data_ptr), write_event_ptr_(write_event_ptr), stream_(stream),
    state_(state)
  {
  }

  uint8_t * data_ptr_{nullptr};
  hipEvent_t * write_event_ptr_{nullptr};
  hipStream_t stream_{nullptr};
  std::shared_ptr<HandleState> state_{nullptr};
  std::shared_ptr<rosidl::Buffer<uint8_t>> promoted_buffer_{};
};

}  // namespace hip_buffer_backend

#endif  // HIP_BUFFER__HIP_BUFFER_HANDLE_HPP_
