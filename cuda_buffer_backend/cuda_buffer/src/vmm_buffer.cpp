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

#include "cuda_buffer/vmm_buffer.hpp"

#include <rcutils/logging_macros.h>

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/cuda_memory_pool.hpp"

namespace cuda_buffer_backend
{

// -- VmmBuffer::BufferRecycler (private nested class) --

class VmmBuffer::BufferRecycler
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

// -- VmmBuffer --

VmmBuffer::VmmBuffer() = default;

VmmBuffer::VmmBuffer(
  void * ptr, size_t size, int device_id,
  std::function<void(uint8_t *)> custom_deleter)
: CudaBuffer(size, device_id),
  device_ptr_(static_cast<uint8_t *>(ptr), std::move(custom_deleter)),
  recycler_(BufferRecycler::get_instance())
{
}

VmmBuffer::~VmmBuffer()
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

std::unique_ptr<VmmBuffer> VmmBuffer::allocate(size_t byte_size)
{
  if (byte_size == 0) {
    return std::make_unique<VmmBuffer>();
  }

  auto pool = get_or_create_global_pool();

  VmmBlock * block = pool->allocate(byte_size);

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

  auto buffer = std::make_unique<VmmBuffer>(
    reinterpret_cast<void *>(block->va), byte_size, pool->get_device_id(),
    pool->deleter(block));

  if (ev) {
    buffer->set_write_event(ev, true);
  }

  return buffer;
}

void VmmBuffer::wait_for_write(cudaStream_t stream) const
{
  if (write_event_ != nullptr && stream != nullptr) {
    CUDA_CHECK(cudaStreamWaitEvent(stream, write_event_, 0));
  }
}

void VmmBuffer::signal_write(cudaStream_t stream) const noexcept
{
  if (write_event_) {
    CUDA_CHECK_NOTHROW(cudaEventRecord(write_event_, stream), (void)0);
  }
}

void VmmBuffer::signal_read(cudaStream_t stream) const noexcept
{
  // Read events are only queried and synchronised in this process (see
  // VmmBuffer::reap_completed_reads), never exported with
  // cudaIpcGetEventHandle. cudaEventInterprocess is therefore not needed, and
  // it makes the CUDA driver create a per-process /dev/shm/cuda.shm.* segment
  // that it does not remove at process exit.
  cudaEvent_t ev{nullptr};
  CUDA_CHECK_NOTHROW(
    cudaEventCreateWithFlags(&ev, CUDA_BUFFER_MINIMUM_EVENT_FLAGS),
    return);
  CUDA_CHECK_NOTHROW(cudaEventRecord(ev, stream), {cudaEventDestroy(ev); return;});
  std::lock_guard<std::mutex> lg(events_mutex_);
  read_events_.push_back(ev);
}

size_t VmmBuffer::reap_completed_reads() const
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
  return read_events_.size();
}

void VmmBuffer::default_cuda_free(uint8_t * p)
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
