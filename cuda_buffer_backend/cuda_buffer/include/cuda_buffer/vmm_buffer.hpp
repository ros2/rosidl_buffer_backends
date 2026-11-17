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

#ifndef CUDA_BUFFER__VMM_BUFFER_HPP_
#define CUDA_BUFFER__VMM_BUFFER_HPP_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/visibility_control.h"

namespace cuda_buffer_backend
{

/// \brief CudaBuffer backed by CUDA Virtual Memory Management (VMM).
///
/// VmmBuffer is the RAII holder for a CUDA device pointer together with the
/// CUDA events that order producer and consumer streams against each other.
/// Locally allocated buffers wrap a block of the process-wide CudaMemoryPool;
/// buffers received from another process wrap an imported VMM mapping. In both
/// cases the deleter decides what releasing the storage means, and a deferred
/// recycler runs it only once all recorded stream work has completed.
///
/// This header is for storage and transport code. Code that only reads or
/// writes buffers should use the CudaBuffer interface and cuda_buffer_api.hpp.
class CUDA_BUFFER_PUBLIC VmmBuffer final : public CudaBuffer
{
public:
  /// \brief Create an empty buffer that owns no device memory.
  VmmBuffer();

  /// \brief Wrap \p size bytes of device memory at \p ptr on \p device_id.
  /// \param custom_deleter invoked when the storage is released.
  VmmBuffer(
    void * ptr, size_t size, int device_id,
    std::function<void(uint8_t *)> custom_deleter);

  ~VmmBuffer() override;

  /// \brief Allocate \p byte_size bytes from the process-wide CudaMemoryPool.
  /// \details A \p byte_size of zero returns an empty buffer without creating
  /// the pool.
  static std::unique_ptr<VmmBuffer> allocate(size_t byte_size);

  StorageKind storage_kind() const noexcept override {return StorageKind::Vmm;}

  void set_write_event(cudaEvent_t event, bool owns_event = false)
  {
    write_event_ = event;
    owns_write_event_ = owns_event;
  }

  cudaEvent_t get_write_event() const {return write_event_;}

protected:
  uint8_t * device_ptr() const noexcept override {return device_ptr_.get();}
  void wait_for_write(cudaStream_t stream) const override;
  void signal_write(cudaStream_t stream) const noexcept override;
  void signal_read(cudaStream_t stream) const noexcept override;
  size_t reap_completed_reads() const override;

private:
  class BufferRecycler;

  static void default_cuda_free(uint8_t * p);

  std::unique_ptr<uint8_t, std::function<void(uint8_t *)>> device_ptr_{
    nullptr, default_cuda_free
  };

  mutable cudaEvent_t write_event_{nullptr};
  bool owns_write_event_{false};
  mutable std::vector<cudaEvent_t> read_events_;
  mutable std::mutex events_mutex_;

  std::shared_ptr<BufferRecycler> recycler_;
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__VMM_BUFFER_HPP_
