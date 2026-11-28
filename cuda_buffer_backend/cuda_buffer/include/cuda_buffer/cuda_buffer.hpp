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
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_handle.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/visibility_control.h"

namespace cuda_buffer_backend
{

/// \brief The storage backend behind a CudaBuffer.
/// \details Code that treats one backend specially, such as the plugin's
/// transport, switches on CudaBuffer::storage_kind() and then static_casts to
/// the class the enumerator names. Each new backend adds an enumerator here.
enum class StorageKind
{
  /// VmmBuffer: CUDA Virtual Memory Management.
  Vmm,
};

/// \brief Abstract CUDA storage object owned by CudaBufferImpl.
///
/// The public rosidl buffer hierarchy is:
///   rosidl::Buffer<T> -> rosidl::BufferImplBase<T> -> CudaBufferImpl<T>.
/// CudaBuffer is not part of that hierarchy. It is the byte-level storage
/// that CudaBufferImpl owns, and the interface through which read and write
/// handles are acquired.
///
/// This base class owns everything that is the same for every storage
/// backend: the size, the device id, and the write-handle state machine
/// (one writer, no write after a read, finalize on publish). How the device
/// memory is obtained and released, and how producer and consumer streams are
/// ordered against each other, is left to the storage backend through the
/// protected hooks below. VmmBuffer is the CUDA VMM implementation.
class CUDA_BUFFER_PUBLIC CudaBuffer
{
public:
  virtual ~CudaBuffer();

  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer & operator=(const CudaBuffer &) = delete;
  CudaBuffer(CudaBuffer &&) = delete;
  CudaBuffer & operator=(CudaBuffer &&) = delete;

  /// \brief Allocate \p byte_size bytes from this process's storage backend.
  /// \details This is where the storage backend for new allocations is chosen.
  /// Buffers received from another process are constructed by the plugin. A
  /// \p byte_size of zero returns an empty buffer without touching the device.
  static std::unique_ptr<CudaBuffer> allocate(size_t byte_size);

  /// \brief The storage backend this buffer belongs to.
  /// \details Every subclass returns its own enumerator, so a static_cast to
  /// the class that enumerator names is valid.
  virtual StorageKind storage_kind() const noexcept = 0;

  ReadHandle get_read_handle(cudaStream_t stream) const;
  WriteHandle get_write_handle(cudaStream_t stream);

  void finalize_write_handle() const;

  size_t size() const {return size_;}
  int get_device_id() const {return device_id_;}
  uint8_t * get_device_ptr() {return device_ptr();}
  const uint8_t * get_device_ptr() const {return device_ptr();}

protected:
  CudaBuffer() = default;
  CudaBuffer(size_t size, int device_id)
  : size_(size), device_id_(device_id) {}

  /// \brief Device pointer of the storage, or nullptr for an empty buffer.
  virtual uint8_t * device_ptr() const noexcept = 0;

  /// \brief Make work submitted to \p stream wait for the producer's write.
  /// \details Called before a read handle is handed out. \p stream may be
  /// nullptr. May throw CudaError.
  virtual void wait_for_write(cudaStream_t stream) const = 0;

  /// \brief Mark the end of the producer's work on \p stream.
  /// \details Called once per write, either when the write handle is released
  /// or when the write is finalized on publish. \p stream is usable.
  virtual void signal_write(cudaStream_t stream) const noexcept = 0;

  /// \brief Mark the end of a consumer's work on \p stream.
  /// \details Called when a read handle is released. \p stream is usable.
  virtual void signal_read(cudaStream_t stream) const noexcept = 0;

  /// \brief Drop bookkeeping for reads that have completed.
  /// \return the number of reads that are still in flight.
  virtual size_t reap_completed_reads() const = 0;

private:
  friend class ReadHandle;
  friend class WriteHandle;

  void finalize_write_handle_locked() const;

  size_t size_{0};
  int device_id_{0};

  /// Serializes write-handle acquisition.
  std::mutex write_acquire_mutex_;
  std::shared_ptr<HandleState> handle_state_{nullptr};
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_BUFFER_HPP_
