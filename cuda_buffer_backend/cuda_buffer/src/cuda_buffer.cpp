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

#include "cuda_buffer/cuda_buffer.hpp"

#include <rcutils/logging_macros.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>

namespace cuda_buffer_backend
{

// -- CudaBuffer::BufferRecycler (private nested class) --

class CudaBuffer::BufferRecycler
{
public:
  // Process-lifetime CUDA cleanup service. Intentionally allocated with
  // a no-op deleter so its worker thread is not torn down during static
  // destruction, when the CUDA runtime/driver may already be unloading.
  static std::shared_ptr<BufferRecycler> get_instance()
  {
    static const std::shared_ptr<BufferRecycler> recycler(
      new BufferRecycler(), [](BufferRecycler *) {});
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
    std::vector<cudaEvent_t> events,
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
    std::vector<cudaEvent_t> events;
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
    for (cudaEvent_t event : work.events) {
      if (!event) {continue;}
      const cudaError_t status = cudaEventQuery(event);
      if (status == cudaErrorNotReady) {
        complete = false;
      } else if (status != cudaSuccess) {
        // An error does not prove that work using the storage has completed.
        // Keep the entry quarantined rather than risking premature reuse.
        complete = false;
        if (!work.query_error_reported) {
          RCUTILS_LOG_WARN_NAMED(
            "cuda_buffer_backend", "cudaEventQuery failed in recycler; retaining buffer: %s",
            cudaGetErrorName(status));
          work.query_error_reported = true;
        }
      }
      if (status != cudaSuccess) {
        (void)cudaGetLastError();
      }
    }
    return complete;
  }

  static void release(PendingWork & work)
  {
    for (cudaEvent_t event : work.events) {
      if (event) {
        cudaEventDestroy(event);
        (void)cudaGetLastError();
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

// -- CudaBuffer --

CudaBuffer::CudaBuffer(
  void * ptr, size_t size, int device_id,
  std::function<void(uint8_t *)> custom_deleter)
: device_ptr_(static_cast<uint8_t *>(ptr), std::move(custom_deleter)), size_(size),
  device_id_(device_id),
  recycler_(BufferRecycler::get_instance())
{
}

CudaBuffer::~CudaBuffer()
{
  std::vector<cudaEvent_t> events_to_sync;

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
    for (cudaEvent_t event : events_to_sync) {
      if (event && cudaEventQuery(event) != cudaSuccess) {
        completed = false;
        break;
      }
    }
    if (completed) {
      // Return completed storage immediately instead of letting a busy cleanup
      // worker retain it while the pool allocates replacement blocks.
      for (cudaEvent_t event : events_to_sync) {
        if (event) {
          cudaEventDestroy(event);
          (void)cudaGetLastError();
        }
      }
    } else {
      recycler_->enqueue(std::move(events_to_sync), std::move(device_ptr_));
    }
  }
}

ReadHandle CudaBuffer::get_read_handle(cudaStream_t stream) const
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

WriteHandle CudaBuffer::get_write_handle(cudaStream_t stream)
{
  reap_completed_read_events();
  std::lock_guard<std::mutex> lg(events_mutex_);

  if (handle_state_) {
    std::lock_guard<std::mutex> lk(handle_state_->mtx);
    if (handle_state_->state == HandleState::State::InUse) {
      throw CudaError("CudaBuffer: write handle already in use; cannot acquire second handle");
    }
    if (handle_state_->state == HandleState::State::Finalized) {
      throw CudaError("CudaBuffer: write already finalized; cannot acquire write handle");
    }
  }

  if (!read_events_.empty()) {
    throw CudaError("CudaBuffer: read events exist; cannot re-acquire write handle");
  }

  if (!handle_state_) {handle_state_ = std::make_shared<HandleState>();}
  handle_state_->state = HandleState::State::InUse;
  handle_state_->write_stream = stream;
  return WriteHandle(device_ptr_.get(), &write_event_, stream, handle_state_);
}

void CudaBuffer::reap_completed_read_events() const
{
  std::lock_guard<std::mutex> lock(events_mutex_);
  auto event = read_events_.begin();
  while (event != read_events_.end()) {
    const cudaError_t status = cudaEventQuery(*event);
    if (status == cudaSuccess) {
      CUDA_CHECK_NOTHROW(cudaEventDestroy(*event), (void)0);
      event = read_events_.erase(event);
      continue;
    }
    if (status != cudaErrorNotReady) {
      RCUTILS_LOG_WARN_NAMED(
        "cuda_buffer_backend", "cudaEventQuery failed: %s", cudaGetErrorName(status));
    }
    (void)cudaGetLastError();
    ++event;
  }
}

void CudaBuffer::finalize_write_handle() const
{
  if (handle_state_ == nullptr) {return;}
  std::lock_guard<std::mutex> lock(handle_state_->mtx);
  finalize_write_handle_locked();
}

void CudaBuffer::finalize_write_handle_locked() const
{
  if (handle_state_->state != HandleState::State::InUse) {return;}
  if (cuda_is_stream_usable(handle_state_->write_stream) && write_event_) {
    if (cudaEventRecord(write_event_, handle_state_->write_stream) != cudaSuccess) {
      (void)cudaGetLastError();
    }
  }
  handle_state_->state = HandleState::State::Finalized;
  handle_state_->write_stream = nullptr;
}

void CudaBuffer::default_cuda_free(uint8_t * p)
{
  if (p) {
    cudaError_t e = cudaFree(p);
    if (!cuda_error_is_safe(e)) {
      RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
        "cudaFree failed: %s", cudaGetErrorName(e));
    }
    (void)cudaGetLastError();
  }
}

}  // namespace cuda_buffer_backend
