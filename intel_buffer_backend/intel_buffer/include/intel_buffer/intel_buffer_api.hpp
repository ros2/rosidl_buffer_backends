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

#ifndef INTEL_BUFFER__INTEL_BUFFER_API_HPP_
#define INTEL_BUFFER__INTEL_BUFFER_API_HPP_

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "intel_buffer/intel_buffer_impl.hpp"
#include "intel_memory_core/intel_memory_buffer_pool.hpp"
#include "intel_memory_core/l0_external_memmap.hpp"
#include "intel_memory_core/intel_memory_ipc_manager.hpp"
#include "intel_memory_core/l0_usm_ipc.hpp"
#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/gpu_cl_dmabuf_import.hpp"
#include "rosidl_buffer/buffer.hpp"

namespace intel_buffer_backend
{

// intel_memory_core (untouched by this refactor) declares its types in
// namespace intel_memory_core; bridge the ones this backend needs.
using intel_memory_core::PoolBlock;
using intel_memory_core::TransportMode;
using intel_memory_core::SharedSysmem;
using intel_memory_core::IntelMemoryIPCManager;
using intel_memory_core::L0UsmIpc;
using intel_memory_core::DmaBufIpc;
using intel_memory_core::import_dmabuf_as_cl_mem;
using intel_memory_core::release_cl_mem;
using intel_memory_core::get_or_import_cl_mem;
using intel_memory_core::get_or_wrap_host_ptr_cl_mem;
using intel_memory_core::sync_host_ptr_cl_mem;
using intel_memory_core::release_all_cl_mem;
using intel_memory_core::cl_dmabuf_import_available;
using intel_memory_core::L0ExternalMemMap;
using intel_memory_core::ExternalMemMapImporter;
using intel_memory_core::kExternalMemMapGenerationBytes;
using intel_memory_core::external_memmap_generation_ptr;

// ============================================================================
// Buffer creation — choose the path that matches your data source
// ============================================================================


inline rosidl::Buffer<uint8_t> allocate_buffer(size_t count)
{
  return rosidl::Buffer<uint8_t>(
    std::make_unique<IntelBufferImpl<uint8_t>>(count));
}

/// Wrap an existing DMA-BUF file descriptor as a zero-copy buffer.

inline rosidl::Buffer<uint8_t> wrap_dmabuf(int dmabuf_fd, size_t size, bool owns_fd = true)
{
  void * ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
  if (ptr == MAP_FAILED) {
    if (owns_fd) {
      close(dmabuf_fd);
    }
    return rosidl::Buffer<uint8_t>(
      std::make_unique<IntelBufferImpl<uint8_t>>());
  }

  auto block = std::make_unique<PoolBlock>();
  block->ptr = ptr;
  block->size = size;
  block->transport = TransportMode::DMA_BUF;
  block->dmabuf_fd = dmabuf_fd;

  auto impl = std::make_unique<IntelBufferImpl<uint8_t>>(
    block.release(), size, owns_fd, /*standalone=*/true, /*owns_mapping=*/true);
  return rosidl::Buffer<uint8_t>(std::move(impl));
}

/// Wrap an existing Level Zero USM pointer as a zero-copy buffer.

inline rosidl::Buffer<uint8_t> wrap_usm_ptr(void * usm_ptr, size_t size, bool owns_ptr = false)
{
  auto block = std::make_unique<PoolBlock>();
  block->ptr = usm_ptr;
  block->size = size;
  block->transport = TransportMode::LEVEL_ZERO_USM;

  auto impl = std::make_unique<IntelBufferImpl<uint8_t>>(
    block.release(), size, owns_ptr, /*standalone=*/true);
  return rosidl::Buffer<uint8_t>(std::move(impl));
}

/// Wrap producer-owned system memory that is already mapped to the device via
/// the Level Zero external-memmap extension.

inline rosidl::Buffer<uint8_t> wrap_external_memmap(
  void * usm_ptr, size_t size, const std::string & shm_name, uint32_t generation = 0)
{
  auto block = std::make_unique<PoolBlock>();
  block->ptr = usm_ptr;
  block->size = size;
  block->transport = TransportMode::EXTERNAL_MEMMAP;
  block->extmap_shm_name = shm_name;
  block->extmap_generation = generation;

  auto impl = std::make_unique<IntelBufferImpl<uint8_t>>(
    block.release(), size, false, /*standalone=*/true);
  return rosidl::Buffer<uint8_t>(std::move(impl));
}


class ExternalMemMapRing
{
public:
  static bool available() {return true;}

  ExternalMemMapRing() = default;
  ~ExternalMemMapRing() {close();}

  ExternalMemMapRing(const ExternalMemMapRing &) = delete;
  ExternalMemMapRing & operator=(const ExternalMemMapRing &) = delete;

  bool create(const std::string & name_prefix, size_t bytes, size_t slots = 4)
  {
    if (bytes == 0 || slots == 0) {
      return false;
    }
    close();

    static std::atomic<uint32_t> next_ring_id{0};
    const uint32_t ring_id = next_ring_id.fetch_add(1, std::memory_order_relaxed);
    const bool use_l0 = L0ExternalMemMap::supported();
    shm_.resize(slots);
    usm_.assign(slots, nullptr);
    mapped_.assign(slots, false);
    generation_.assign(slots, nullptr);
    for (size_t i = 0; i < slots; ++i) {
      const std::string name = name_prefix + "_" + std::to_string(getpid()) +
        "_" + std::to_string(ring_id) + "_" + std::to_string(i);

      if (!shm_[i].create(name, bytes + kExternalMemMapGenerationBytes)) {
        close();
        return false;
      }
      if (use_l0) {
        usm_[i] = L0ExternalMemMap::map(shm_[i].ptr(), shm_[i].size());
        if (!usm_[i]) {
          close();
          return false;
        }
        mapped_[i] = true;
      } else {
        usm_[i] = shm_[i].ptr();
      }
      generation_[i] = external_memmap_generation_ptr(shm_[i].ptr(), shm_[i].size());
      if (generation_[i]) {
        generation_[i]->store(0, std::memory_order_relaxed);
      }
    }
    next_ = slots - 1;
    return true;
  }


  size_t advance()
  {
    if (usm_.empty()) {
      return 0;
    }
    next_ = (next_ + 1) % usm_.size();
    if (next_ < generation_.size() && generation_[next_]) {
      generation_[next_]->fetch_add(1, std::memory_order_release);
    }
    return next_;
  }


  uint8_t * data(size_t slot)
  {
    return slot < usm_.size() ? static_cast<uint8_t *>(usm_[slot]) : nullptr;
  }


  rosidl::Buffer<uint8_t> wrap(size_t slot, size_t size)
  {
    if (slot >= usm_.size()) {
      return rosidl::Buffer<uint8_t>(
        std::make_unique<IntelBufferImpl<uint8_t>>());
    }
    const uint32_t gen = (slot < generation_.size() && generation_[slot]) ?
      generation_[slot]->load(std::memory_order_relaxed) : 0;
    return wrap_external_memmap(usm_[slot], size, shm_[slot].name(), gen);
  }

  size_t slots() const {return usm_.size();}

  size_t slot_size() const {return shm_.empty() ? 0 : shm_.front().size();}


  void close()
  {
    for (size_t i = 0; i < usm_.size(); ++i) {
      if (i < mapped_.size() && mapped_[i]) {
        L0ExternalMemMap::unmap(usm_[i]);
      }
    }
    usm_.clear();
    mapped_.clear();
    generation_.clear();
    shm_.clear();
    next_ = 0;
  }

private:
  std::vector<SharedSysmem> shm_;
  std::vector<void *> usm_;
  std::vector<bool> mapped_;
  std::vector<std::atomic<uint32_t> *> generation_;
  size_t next_{0};
};

// ============================================================================
// Buffer access — subscriber side
// ============================================================================

inline uint8_t * get_usm_ptr(rosidl::Buffer<uint8_t> & buffer)
{
  auto * impl = dynamic_cast<IntelBufferImpl<uint8_t> *>(buffer.get_impl());
  return impl ? impl->data() : nullptr;
}

/// Get the raw USM/DMA-BUF pointer from a const buffer (read access).
inline const uint8_t * get_usm_ptr(const rosidl::Buffer<uint8_t> & buffer)
{
  const auto * impl =
    dynamic_cast<const IntelBufferImpl<uint8_t> *>(buffer.get_impl());
  return impl ? impl->data() : nullptr;
}

// ============================================================================
// Data ingestion — simulation / software source only
// ============================================================================


inline bool write_to_buffer(
  rosidl::Buffer<uint8_t> & buffer,
  const void * src,
  size_t byte_count)
{
  uint8_t * ptr = get_usm_ptr(buffer);
  if (!ptr || byte_count > buffer.size()) {
    return false;
  }
  if (byte_count > 0) {
    std::memcpy(ptr, src, byte_count);
  }
  return true;
}

}  // namespace intel_buffer_backend

#endif  // INTEL_BUFFER__INTEL_BUFFER_API_HPP_
