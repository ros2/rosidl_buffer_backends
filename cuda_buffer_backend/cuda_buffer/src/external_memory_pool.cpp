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

#include "cuda_buffer/external_memory_pool.hpp"

#include <rcutils/logging_macros.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "cuda_buffer/cuda_error.hpp"

namespace cuda_buffer_backend
{

namespace
{

bool is_power_of_two(size_t v)
{
  return v != 0 && (v & (v - 1)) == 0;
}

size_t align_up(size_t v, size_t alignment)
{
  return (v + alignment - 1) & ~(alignment - 1);
}

/// Owns a CUDA import of an NvSciBuf object, and tears it down in the order the
/// driver requires: the mapped view first, then the external memory handle.
///
/// Held as the pool's keepalive rather than as a member, so that the one
/// mechanism which already means "release when every block is back" also covers
/// the import. A member would need its own teardown ordering against the pool's
/// destructor; this needs none.
struct NvSciImport
{
  cudaExternalMemory_t memory{nullptr};
  void * mapped{nullptr};
  std::shared_ptr<void> user_keepalive;

  ~NvSciImport()
  {
    if (mapped != nullptr) {
      CUDA_CHECK_NOTHROW(cudaFree(mapped), (void)0);
    }
    if (memory != nullptr) {
      CUDA_CHECK_NOTHROW(cudaDestroyExternalMemory(memory), (void)0);
    }
  }
};

int resolve_device_id(int device_id)
{
  if (device_id >= 0) {
    return device_id;
  }
  int current = 0;
  CUDA_CHECK(cudaGetDevice(&current));
  return current;
}

}  // namespace

ExternalMemoryPool::ExternalMemoryPool(
  uint8_t * base, size_t capacity, int device_id,
  std::shared_ptr<void> keepalive, size_t alignment)
: base_(base), capacity_(capacity), device_id_(device_id), alignment_(alignment),
  keepalive_(std::move(keepalive))
{
  free_by_offset_.emplace(0, capacity_);
}

ExternalMemoryPool::~ExternalMemoryPool()
{
  // Unreachable by construction: every live block's deleter holds a shared_ptr
  // to this pool, so the last block must be returned before the refcount can
  // reach zero. Logged rather than asserted because a destructor is the wrong
  // place to abort, and because the only way here is a caller that bypassed
  // deleter() and freed a block by hand.
  if (!live_by_offset_.empty()) {
    RCUTILS_LOG_ERROR_NAMED("cuda_buffer_backend",
      "ExternalMemoryPool destroyed with %zu block(s) still outstanding; "
      "the region's owner may reclaim memory a buffer is still using",
      live_by_offset_.size());
  }
}

std::shared_ptr<ExternalMemoryPool> ExternalMemoryPool::create(
  void * base_device_ptr, size_t size, int device_id,
  std::shared_ptr<void> keepalive, size_t alignment)
{
  if (base_device_ptr == nullptr) {
    throw CudaError("ExternalMemoryPool::create called with a null base pointer");
  }
  if (size == 0) {
    throw CudaError("ExternalMemoryPool::create called with a zero size");
  }
  if (!is_power_of_two(alignment)) {
    throw CudaError("ExternalMemoryPool::create requires a power-of-two alignment");
  }

  // Align the base up and the tail down, once, so that every offset the
  // allocator subsequently hands out is aligned by construction and no block
  // needs padding bookkeeping of its own.
  const uintptr_t raw = reinterpret_cast<uintptr_t>(base_device_ptr);
  const uintptr_t aligned = align_up(raw, alignment);
  const size_t lost = static_cast<size_t>(aligned - raw);
  if (lost >= size) {
    throw CudaError(
            "ExternalMemoryPool::create: region is too small to hold an aligned block");
  }
  const size_t capacity = (size - lost) & ~(alignment - 1);
  if (capacity == 0) {
    throw CudaError(
            "ExternalMemoryPool::create: region is smaller than one alignment unit");
  }

  const int resolved = resolve_device_id(device_id);

  return std::shared_ptr<ExternalMemoryPool>(
    new ExternalMemoryPool(
      reinterpret_cast<uint8_t *>(aligned), capacity, resolved,
      std::move(keepalive), alignment));
}

std::shared_ptr<ExternalMemoryPool> ExternalMemoryPool::import_nvscibuf(
  const void * nvscibuf_obj, size_t offset, size_t size,
  std::shared_ptr<void> keepalive, size_t alignment)
{
  if (nvscibuf_obj == nullptr) {
    throw CudaError("ExternalMemoryPool::import_nvscibuf called with a null object");
  }
  if (size == 0) {
    throw CudaError("ExternalMemoryPool::import_nvscibuf called with a zero size");
  }

  auto import = std::make_shared<NvSciImport>();
  import->user_keepalive = std::move(keepalive);

  cudaExternalMemoryHandleDesc handle_desc{};
  handle_desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
  handle_desc.handle.nvSciBufObject = nvscibuf_obj;
  // The import must span everything up to the end of the region we want; the
  // mapped view below is what narrows it to [offset, offset + size).
  handle_desc.size = offset + size;

  CUDA_CHECK(cudaImportExternalMemory(&import->memory, &handle_desc));

  cudaExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = offset;
  buffer_desc.size = size;
  CUDA_CHECK(
    cudaExternalMemoryGetMappedBuffer(&import->mapped, import->memory, &buffer_desc));

  // -1: the mapping was just made against the current device, so that is the
  // device the region belongs to. Resolved inside create().
  //
  // The import becomes the keepalive, which is what makes teardown ordering
  // automatic -- the pool cannot outlive its blocks, and the import cannot
  // outlive the pool.
  void * mapped = import->mapped;
  return create(mapped, size, -1, std::move(import), alignment);
}

ExternalBlock * ExternalMemoryPool::allocate(size_t byte_size)
{
  // A zero-byte request still needs a distinct address, or two of them would
  // alias and the second free() would corrupt the free list.
  const size_t need = align_up(byte_size == 0 ? 1 : byte_size, alignment_);

  std::lock_guard<std::mutex> lock(mutex_);

  for (auto it = free_by_offset_.begin(); it != free_by_offset_.end(); ++it) {
    if (it->second < need) {
      continue;
    }
    const size_t offset = it->first;
    const size_t remaining = it->second - need;
    free_by_offset_.erase(it);
    if (remaining > 0) {
      free_by_offset_.emplace(offset + need, remaining);
    }

    auto block = std::make_unique<ExternalBlock>();
    block->ptr = base_ + offset;
    block->offset = offset;
    block->size = need;
    ExternalBlock * raw = block.get();
    live_by_offset_.emplace(offset, std::move(block));
    return raw;
  }

  return nullptr;
}

void ExternalMemoryPool::free(ExternalBlock * block)
{
  if (block == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto live = live_by_offset_.find(block->offset);
  if (live == live_by_offset_.end() || live->second.get() != block) {
    RCUTILS_LOG_ERROR_NAMED("cuda_buffer_backend",
      "ExternalMemoryPool::free called with a block this pool did not hand out");
    return;
  }

  size_t offset = block->offset;
  size_t size = block->size;
  live_by_offset_.erase(live);

  // Coalesce with the successor first, then the predecessor. lower_bound gives
  // the first free region at or after this one; because free regions never
  // overlap and this one is not in the map yet, that is exactly the successor.
  auto next = free_by_offset_.lower_bound(offset);
  if (next != free_by_offset_.end() && offset + size == next->first) {
    size += next->second;
    next = free_by_offset_.erase(next);
  }
  if (next != free_by_offset_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == offset) {
      // Merged into the predecessor, which already sits at the right offset.
      prev->second += size;
      return;
    }
  }
  free_by_offset_.emplace(offset, size);
}

std::function<void(uint8_t *)> ExternalMemoryPool::deleter(ExternalBlock * block)
{
  auto self = shared_from_this();
  return [self, block](uint8_t *) {
           self->free(block);
         };
}

size_t ExternalMemoryPool::bytes_free() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  size_t total = 0;
  for (const auto & region : free_by_offset_) {
    total += region.second;
  }
  return total;
}

size_t ExternalMemoryPool::bytes_in_use() const
{
  return capacity_ - bytes_free();
}

size_t ExternalMemoryPool::largest_free_block() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  size_t largest = 0;
  for (const auto & region : free_by_offset_) {
    if (region.second > largest) {
      largest = region.second;
    }
  }
  return largest;
}

}  // namespace cuda_buffer_backend
