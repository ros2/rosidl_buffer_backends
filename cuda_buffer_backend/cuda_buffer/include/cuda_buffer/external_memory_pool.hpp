// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
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

#ifndef CUDA_BUFFER__EXTERNAL_MEMORY_POOL_HPP_
#define CUDA_BUFFER__EXTERNAL_MEMORY_POOL_HPP_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>

#include "cuda_buffer/visibility_control.h"

namespace cuda_buffer_backend
{

/// \brief One suballocation handed out by ExternalMemoryPool.
///
/// Owned by the pool, not by the buffer using it. The buffer holds only the
/// deleter that returns this block, which is why the pointer stays valid for
/// exactly as long as the buffer does.
struct ExternalBlock
{
  uint8_t * ptr{nullptr};
  size_t offset{0};
  size_t size{0};
};

/// \brief A pool over device memory that somebody else allocated and mapped.
///
/// The counterpart to CudaMemoryPool for storage this backend did not create.
/// CudaMemoryPool owns its memory: it calls cuMemCreate, maps a VA, and exports
/// an fd so its blocks can cross a process boundary. ExternalMemoryPool owns
/// nothing. It is handed one contiguous, already-CUDA-addressable region and
/// suballocates it, so that a middleware which has its own allocator -- an
/// NvSciBuf pool behind a zero-copy transport is the motivating case -- can
/// hand ROS applications ordinary rosidl::Buffers that live in *its* memory.
///
/// The result is indistinguishable from a pool-allocated buffer at every call
/// site that matters: backend "cuda", the same read and write handles, the same
/// event ordering, the same promotion rules in from_input_buffer() and
/// from_output_buffer(). What differs is where the bytes came from and who
/// reclaims them.
///
/// \par Relationship to adopt_buffer()
/// CudaBuffer::adopt() wraps *one* pointer that the caller has already carved
/// out. This wraps a *region* and does the carving, with reuse: blocks return
/// to a free list on destruction and are handed out again. Use adoption for a
/// single transport slot handed to you; use a pool when you own the arena and
/// want many buffers out of it.
///
/// \par Lifetime
/// Blocks keep the pool alive -- each block's deleter holds a shared_ptr back
/// here -- so the region cannot be released while a buffer still names part of
/// it. The keepalive passed to create() is released when the last block has
/// been returned *and* the last external reference to the pool has dropped, and
/// because a block is returned by CudaBuffer's deleter, that happens only after
/// the recycler has synchronized every read and write event on it. The owner is
/// therefore told "done" when the GPU is done, not when the last C++ reference
/// drops.
///
/// \par What this deliberately does not do
/// No IPC metadata, no refcount, no publish grace window. CudaMemoryPool needs
/// those because it exports its blocks to other processes and has to decide
/// when a subscriber is finished. Here the region's owner runs cross-process
/// lifetime -- Halos pins a slot until its readers release it -- and this pool
/// is responsible only for the local half. Adding a second, weaker refcount
/// underneath the owner's would not make anything safer.
///
/// Thread-safe: allocate() and free() may be called concurrently.
class CUDA_BUFFER_PUBLIC ExternalMemoryPool
  : public std::enable_shared_from_this<ExternalMemoryPool>
{
public:
  /// Suballocation granularity when the caller does not specify one. 256 bytes
  /// is CUDA's texture-alignment guarantee and a common NvSciBuf constraint, so
  /// it is safe for the kernels most likely to read these buffers.
  static constexpr size_t kDefaultAlignment = 256;

  ~ExternalMemoryPool();

  ExternalMemoryPool(const ExternalMemoryPool &) = delete;
  ExternalMemoryPool & operator=(const ExternalMemoryPool &) = delete;
  ExternalMemoryPool(ExternalMemoryPool &&) = delete;
  ExternalMemoryPool & operator=(ExternalMemoryPool &&) = delete;

  /// \brief Build a pool over an already-mapped device region.
  ///
  /// \param base_device_ptr Start of the region, already mapped into this
  ///   process's CUDA context by whoever owns it. The pool aligns upward from
  ///   here, so an unaligned base costs a few bytes rather than failing.
  /// \param size Bytes at \p base_device_ptr.
  /// \param device_id CUDA device the region belongs to; -1 asks the current
  ///   device, which is right whenever the caller mapped it itself.
  /// \param keepalive Released once every block has come back and the pool
  ///   itself is dropped. Whatever the owner uses to mean "still in use" goes
  ///   here. May be null when the region outlives the pool by construction.
  /// \param alignment Suballocation granularity; must be a power of two.
  /// \throw CudaError if \p base_device_ptr is null, \p size is zero,
  ///   \p alignment is not a power of two, \p size is too small to hold one
  ///   aligned block, or \p device_id is -1 and no current device exists.
  static std::shared_ptr<ExternalMemoryPool> create(
    void * base_device_ptr, size_t size, int device_id = -1,
    std::shared_ptr<void> keepalive = nullptr,
    size_t alignment = kDefaultAlignment);

  /// \brief Build a pool over a pre-existing NvSciBuf allocation.
  ///
  /// Imports \p nvscibuf_obj into CUDA and pools `[offset, offset + size)` of
  /// it. This is the path for a middleware that allocated its transport memory
  /// as NvSciBuf -- rmw_halos and its Halos slot pool -- and wants ROS
  /// applications to get cuda_buffer-backed payloads out of that same memory
  /// instead of a private cudaMalloc arena.
  ///
  /// The pool owns the import and tears it down (cudaFree of the mapped view,
  /// then cudaDestroyExternalMemory) once every block is back. It does not own
  /// the NvSciBufObj: the caller keeps that alive, and \p keepalive is how to
  /// say so if its lifetime is not already guaranteed.
  ///
  /// \param nvscibuf_obj An \c NvSciBufObj, taken as \c const void* so this
  ///   header -- and the whole package -- needs no NvSci dependency. CUDA's own
  ///   \c cudaExternalMemoryHandleDesc types it the same way.
  /// \param offset Start of the poolable region within the object. The import
  ///   covers `offset + size` and the mapped view narrows to the region, which
  ///   is what lets a caller pool one slice of a larger object.
  /// \param size Bytes to pool.
  /// \param keepalive Held for the pool's lifetime. Use it to pin the
  ///   NvSciBufObj, or the allocator that produced it.
  /// \param alignment Suballocation granularity; must be a power of two. Pass
  ///   the object's own alignment constraint when it is stricter than the
  ///   default.
  /// \throw CudaError if the import or the mapping fails -- including on a
  ///   platform whose CUDA driver has no NvSciBuf support, which surfaces here
  ///   as a failed import rather than at build time.
  static std::shared_ptr<ExternalMemoryPool> import_nvscibuf(
    const void * nvscibuf_obj, size_t offset, size_t size,
    std::shared_ptr<void> keepalive = nullptr,
    size_t alignment = kDefaultAlignment);

  /// \brief Carve \p byte_size bytes out of the region.
  ///
  /// First fit over a free list that coalesces on release, so a pool cycling
  /// same-sized slots -- the transport case -- keeps reusing the same offsets
  /// and does not fragment.
  ///
  /// \return null when the region has no run of \p byte_size free bytes left.
  ///   Null rather than an exception because exhaustion is an ordinary
  ///   condition for a fixed arena: the caller is expected to fall back, and
  ///   allocate_buffer_from() is what turns it into an error when it cannot.
  ExternalBlock * allocate(size_t byte_size);

  /// \brief Return a block. Called by the deleter; not usually called directly.
  void free(ExternalBlock * block);

  /// \brief The deleter that returns \p block to this pool.
  ///
  /// Captures a shared_ptr to the pool, which is what keeps the region mapped
  /// for as long as any buffer still points into it.
  std::function<void(uint8_t *)> deleter(ExternalBlock * block);

  int get_device_id() const {return device_id_;}
  size_t alignment() const {return alignment_;}

  /// Poolable bytes, after aligning the base and rounding the tail down.
  size_t capacity() const {return capacity_;}

  size_t bytes_in_use() const;
  size_t bytes_free() const;

  /// Largest single allocation this pool could still satisfy. Below
  /// bytes_free() exactly when the free space is fragmented.
  size_t largest_free_block() const;

private:
  ExternalMemoryPool(
    uint8_t * base, size_t capacity, int device_id,
    std::shared_ptr<void> keepalive, size_t alignment);

  uint8_t * base_{nullptr};
  size_t capacity_{0};
  int device_id_{0};
  size_t alignment_{kDefaultAlignment};
  std::shared_ptr<void> keepalive_;

  /// Free regions by offset, coalesced and non-overlapping. Keyed by offset
  /// rather than size so that free() can find and merge its neighbours; the
  /// scan in allocate() is linear, which is the right trade for a pool whose
  /// free list stays short because same-sized blocks keep recombining.
  std::map<size_t, size_t> free_by_offset_;
  std::map<size_t, std::unique_ptr<ExternalBlock>> live_by_offset_;
  mutable std::mutex mutex_;
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__EXTERNAL_MEMORY_POOL_HPP_
