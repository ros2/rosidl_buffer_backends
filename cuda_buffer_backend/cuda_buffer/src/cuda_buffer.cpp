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
    cv_.notify_one();
  }

private:
  struct PendingWork
  {
    std::vector<cudaEvent_t> events;
    std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> device_ptr;
  };

  BufferRecycler()
  {
    thread_ = std::thread(&BufferRecycler::run, this);
  }

  void run()
  {
    while (true) {
      PendingWork work;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {return !queue_.empty() || !running_;});
        if (queue_.empty() && !running_) {break;}
        if (queue_.empty()) {continue;}
        work = std::move(queue_.front());
        queue_.pop_front();
      }
      for (cudaEvent_t ev : work.events) {
        if (ev) {
          cudaEventSynchronize(ev);
          cudaEventDestroy(ev);
          (void)cudaGetLastError();
        }
      }
      work.device_ptr.reset();
    }
  }

  std::deque<PendingWork> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
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

CudaBuffer CudaBuffer::adopt(
  void * ptr, size_t size, int device_id, std::shared_ptr<void> keepalive)
{
  // The keepalive rides in the deleter rather than in a member of its own, and
  // that placement is the whole contract. ~CudaBuffer hands device_ptr_ to the
  // BufferRecycler, which destroys it only after synchronizing every
  // outstanding read and write event -- so a keepalive captured here is
  // released once the GPU is finished, not when the last C++ reference drops.
  // A member would be destroyed at the wrong moment: the owner would be told
  // "done" while a kernel was still reading its memory.
  //
  // The deleter itself frees nothing. Adopted storage is not ours; the owner
  // reclaims it when its keepalive tells it to.
  CudaBuffer buffer(
    ptr, size, device_id,
    [keepalive = std::move(keepalive)](uint8_t *) {});
  buffer.adopted_ = true;
  return buffer;
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
    recycler_->enqueue(std::move(events_to_sync), std::move(device_ptr_));
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
