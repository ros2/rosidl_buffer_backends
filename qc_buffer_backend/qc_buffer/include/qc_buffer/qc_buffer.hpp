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

#ifndef QC_BUFFER__QC_BUFFER_HPP_
#define QC_BUFFER__QC_BUFFER_HPP_

#include <cstddef>
#include <cstdint>

#include "qc_buffer/visibility_control.h"

namespace qc_buffer_backend
{

/// \brief Internal RAII holder for a single ION/dma-buf allocation.
///
/// Two construction paths:
///   kRpcmem (default): publisher allocates via rpcmem_alloc; reset() calls
///                      rpcmem_free.
///   kMmap: cross-process subscriber receives a dma-buf fd via SCM_RIGHTS and
///          mmap()s it; reset() calls munmap() + close(fd).
class QC_BUFFER_PUBLIC QcBuffer
{
public:
  QcBuffer() = default;

  /// Allocate \p byte_size bytes via rpcmem. Throws QcError on failure.
  explicit QcBuffer(size_t byte_size);

  /// Cross-process subscriber path: adopt an externally mmap-ed pointer and
  /// the fd it was mapped from. Takes ownership of both (munmap + close on
  /// destruction). \p uid is assigned by next_uid().
  QcBuffer(uint8_t * ptr, size_t size, int fd);

  ~QcBuffer();

  QcBuffer(QcBuffer && other) noexcept;
  QcBuffer & operator=(QcBuffer && other) noexcept;

  QcBuffer(const QcBuffer &) = delete;
  QcBuffer & operator=(const QcBuffer &) = delete;

  uint8_t * data() {return ptr_;}
  const uint8_t * data() const {return ptr_;}

  int dmabuf_fd() const {return fd_;}
  size_t size() const {return size_;}
  uint64_t uid() const {return uid_;}

private:
  void reset() noexcept;

  enum class Origin { kRpcmem, kMmap };

  uint8_t * ptr_{nullptr};
  int fd_{-1};
  size_t size_{0};
  uint64_t uid_{0};
  Origin origin_{Origin::kRpcmem};
};

}  // namespace qc_buffer_backend

#endif  // QC_BUFFER__QC_BUFFER_HPP_
