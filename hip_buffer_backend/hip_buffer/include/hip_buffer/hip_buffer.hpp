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

#ifndef HIP_BUFFER__HIP_BUFFER_HPP_
#define HIP_BUFFER__HIP_BUFFER_HPP_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "hip_buffer/hip_buffer_handle.hpp"
#include "hip_buffer/hip_error.hpp"

namespace hip_buffer_backend
{

/// \brief Internal HIP storage object owned by HipBufferImpl.
///
/// The public rosidl buffer hierarchy is:
///   rosidl::Buffer<T> -> rosidl::BufferImplBase<T> -> HipBufferImpl<T>.
/// HipBuffer is not part of that hierarchy. It is the
/// low-level RAII holder for the HIP device pointer, device id, write/read
/// event state, and deferred recycler used by HipBufferImpl.
class HipBuffer
{
public:
  HipBuffer() = default;

  HipBuffer(HipBuffer && other) noexcept
  : device_ptr_(std::move(other.device_ptr_)),
    size_(other.size_),
    device_id_(other.device_id_),
    write_event_(other.write_event_),
    owns_write_event_(other.owns_write_event_),
    read_events_(std::move(other.read_events_)),
    handle_state_(std::move(other.handle_state_)),
    recycler_(std::move(other.recycler_))
  {
    other.size_ = 0;
    other.device_id_ = 0;
    other.write_event_ = nullptr;
    other.owns_write_event_ = false;
  }

  HipBuffer(const HipBuffer &) = delete;
  HipBuffer & operator=(const HipBuffer &) = delete;

  HipBuffer & operator=(HipBuffer && other) noexcept
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
    }
    return *this;
  }

  ~HipBuffer();

  HipBuffer(
    void * ptr, size_t size, int device_id,
    std::function<void(uint8_t *)> custom_deleter);

  ReadHandle get_read_handle(hipStream_t stream) const;
  WriteHandle get_write_handle(hipStream_t stream);

  void set_write_event(hipEvent_t event, bool owns_event = false)
  {
    write_event_ = event;
    owns_write_event_ = owns_event;
  }

  hipEvent_t get_write_event() const {return write_event_;}

  void release_write_event_ownership()
  {
    std::lock_guard<std::mutex> lock(events_mutex_);
    owns_write_event_ = false;
  }

  void finalize_write_handle() const;

  size_t size() const {return size_;}
  int get_device_id() const {return device_id_;}
  uint8_t * get_device_ptr() {return device_ptr_.get();}
  const uint8_t * get_device_ptr() const {return device_ptr_.get();}

private:
  class BufferRecycler;

  void reap_completed_read_events() const;
  void finalize_write_handle_locked() const;
  static void default_hip_free(uint8_t * p);

  std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> device_ptr_{
    nullptr, default_hip_free
  };
  size_t size_{0};
  int device_id_{0};

  mutable hipEvent_t write_event_{nullptr};
  bool owns_write_event_{false};
  mutable std::vector<hipEvent_t> read_events_;
  mutable std::mutex events_mutex_;

  std::shared_ptr<HandleState> handle_state_{nullptr};
  std::shared_ptr<BufferRecycler> recycler_;
};

}  // namespace hip_buffer_backend

#endif  // HIP_BUFFER__HIP_BUFFER_HPP_
