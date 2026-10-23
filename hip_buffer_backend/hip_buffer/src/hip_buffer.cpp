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

#include "hip_buffer/hip_buffer.hpp"

#include <rcutils/logging_macros.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>

namespace hip_buffer_backend
{

// -- HipBuffer::BufferRecycler (private nested class) --

class HipBuffer::BufferRecycler
{
public:
  // Storage initializes HIP before creating this service. Owning the service
  // lets its destructor drain pending work and join the worker before the pool and
  // HIP runtime are destroyed, rather than leaving GPU cleanup running at exit.
  static std::shared_ptr<BufferRecycler> get_instance()
  {
    static const std::shared_ptr<BufferRecycler> recycler(new BufferRecycler());
    return recycler;
  }

  ~BufferRecycler()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  BufferRecycler(const BufferRecycler &) = delete;
  BufferRecycler & operator=(const BufferRecycler &) = delete;

  void enqueue(
    std::vector<hipEvent_t> events,
    std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> ptr)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back({std::move(events), std::move(ptr)});
    ++enqueue_epoch_;
    cv_.notify_one();
  }

private:
  struct PendingWork
  {
    std::vector<hipEvent_t> events;
    std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> device_ptr;
    bool query_error_reported{false};
  };

  BufferRecycler()
  {
    thread_ = std::thread(&BufferRecycler::run, this);
  }

  static bool is_complete(PendingWork & work)
  {
    bool complete = true;
    for (hipEvent_t event : work.events) {
      if (!event) {continue;}
      const hipError_t status = hipEventQuery(event);
      if (status == hipErrorNotReady) {
        complete = false;
      } else if (status != hipSuccess) {
        // An error does not prove that work using the storage has completed.
        // Keep the entry quarantined rather than risking premature reuse.
        complete = false;
        if (!work.query_error_reported) {
          RCUTILS_LOG_WARN_NAMED(
            "hip_buffer_backend", "hipEventQuery failed in recycler; retaining buffer: %s",
            hipGetErrorName(status));
          work.query_error_reported = true;
        }
      }
      if (status != hipSuccess) {
        (void)hipGetLastError();
      }
    }
    return complete;
  }

  static void release(PendingWork & work)
  {
    for (hipEvent_t event : work.events) {
      if (event) {
        hipEventDestroy(event);
        (void)hipGetLastError();
      }
    }
    work.device_ptr.reset();
  }

  void run()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      cv_.wait(lock, [this] {return !queue_.empty() || !running_;});
      if (queue_.empty() && !running_) {break;}

      std::deque<PendingWork> completed;
      for (auto work = queue_.begin(); work != queue_.end(); ) {
        if (is_complete(*work)) {
          completed.push_back(std::move(*work));
          work = queue_.erase(work);
        } else {
          ++work;
        }
      }
      const uint64_t observed_epoch = enqueue_epoch_;
      const bool has_pending = !queue_.empty();

      lock.unlock();
      for (auto & work : completed) {
        release(work);
      }
      lock.lock();

      if (has_pending && !queue_.empty()) {
        cv_.wait_for(
          lock, std::chrono::milliseconds(1),
          [this, observed_epoch] {
            return !running_ || enqueue_epoch_ != observed_epoch;
          });
      }
    }
  }

  std::deque<PendingWork> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  uint64_t enqueue_epoch_{0};
  bool running_{true};
};

// -- HipBuffer --

HipBuffer::HipBuffer(
  void * ptr, size_t size, int device_id,
  std::function<void(uint8_t *)> custom_deleter)
: device_ptr_(static_cast<uint8_t *>(ptr), std::move(custom_deleter)), size_(size),
  device_id_(device_id),
  recycler_(BufferRecycler::get_instance())
{
}

HipBuffer::~HipBuffer()
{
  std::vector<hipEvent_t> events_to_sync;

  if (owns_write_event_ && write_event_) {
    events_to_sync.push_back(write_event_);
    write_event_ = nullptr;
  }

  if (!read_events_.empty()) {
    events_to_sync.insert(
      events_to_sync.end(), read_events_.begin(), read_events_.end());
    read_events_.clear();
  }

  if (!events_to_sync.empty() && recycler_) {
    bool completed = true;
    for (hipEvent_t event : events_to_sync) {
      if (event && hipEventQuery(event) != hipSuccess) {
        completed = false;
        break;
      }
    }
    if (completed) {
      // Return completed storage immediately instead of letting a busy cleanup
      // worker retain it while the pool allocates replacement blocks.
      for (hipEvent_t event : events_to_sync) {
        if (event) {
          hipEventDestroy(event);
          (void)hipGetLastError();
        }
      }
    } else {
      recycler_->enqueue(std::move(events_to_sync), std::move(device_ptr_));
    }
  }
}

ReadHandle HipBuffer::get_read_handle(hipStream_t stream) const
{
  reap_completed_read_events();

  if (handle_state_) {
    std::lock_guard<std::mutex> lk(handle_state_->mtx);
    if (handle_state_->state == HandleState::State::InUse) {
      finalize_write_handle_locked();
    }
  }

  return ReadHandle(
    device_ptr_.get(), write_event_, &read_events_, &events_mutex_, stream);
}

WriteHandle HipBuffer::get_write_handle(hipStream_t stream)
{
  reap_completed_read_events();
  std::lock_guard<std::mutex> lg(events_mutex_);

  if (handle_state_) {
    std::lock_guard<std::mutex> lk(handle_state_->mtx);
    if (handle_state_->state == HandleState::State::InUse) {
      throw HipError("HipBuffer: write handle already in use; cannot acquire second handle");
    }
    if (handle_state_->state == HandleState::State::Finalized) {
      throw HipError("HipBuffer: write already finalized; cannot acquire write handle");
    }
  }

  if (!read_events_.empty()) {
    throw HipError("HipBuffer: read events exist; cannot re-acquire write handle");
  }

  if (!handle_state_) {handle_state_ = std::make_shared<HandleState>();}
  handle_state_->state = HandleState::State::InUse;
  handle_state_->write_stream = stream;
  return WriteHandle(device_ptr_.get(), &write_event_, stream, handle_state_);
}

void HipBuffer::reap_completed_read_events() const
{
  std::lock_guard<std::mutex> lock(events_mutex_);
  auto event = read_events_.begin();
  while (event != read_events_.end()) {
    const hipError_t status = hipEventQuery(*event);
    if (status == hipSuccess) {
      HIP_CHECK_NOTHROW(hipEventDestroy(*event), (void)0);
      event = read_events_.erase(event);
      continue;
    }
    if (status != hipErrorNotReady) {
      RCUTILS_LOG_WARN_NAMED(
        "hip_buffer_backend", "hipEventQuery failed: %s", hipGetErrorName(status));
    }
    (void)hipGetLastError();
    ++event;
  }
}

void HipBuffer::finalize_write_handle() const
{
  if (handle_state_ == nullptr) {return;}
  std::lock_guard<std::mutex> lock(handle_state_->mtx);
  finalize_write_handle_locked();
}

void HipBuffer::finalize_write_handle_locked() const
{
  if (handle_state_->state != HandleState::State::InUse) {return;}
  if (hip_is_stream_usable(handle_state_->write_stream) && write_event_) {
    if (hipEventRecord(write_event_, handle_state_->write_stream) != hipSuccess) {
      (void)hipGetLastError();
    }
  }
  handle_state_->state = HandleState::State::Finalized;
  handle_state_->write_stream = nullptr;
}

void HipBuffer::default_hip_free(uint8_t * p)
{
  if (p) {
    hipError_t e = hipFree(p);
    if (!hip_error_is_safe(e)) {
      RCUTILS_LOG_WARN_NAMED("hip_buffer_backend",
        "hipFree failed: %s", hipGetErrorName(e));
    }
    (void)hipGetLastError();
  }
}

}  // namespace hip_buffer_backend
