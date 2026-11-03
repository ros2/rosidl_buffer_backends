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

// Level Zero External Memory Mapping — map an application-owned, page-aligned
// system-memory buffer as USM host memory with no dedicated pool and no copy.
#ifndef INTEL_MEMORY_CORE__L0_EXTERNAL_MEMMAP_HPP_
#define INTEL_MEMORY_CORE__L0_EXTERNAL_MEMMAP_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include "intel_memory_core/intel_memory_buffer_pool.hpp"

namespace intel_memory_core
{

// An external-memmap ring slot region (page-aligned, shared by producer and
// consumers) ends in a trailer:
//   [payload ...][pad][IPCMetadata: consumer refcount + holders][uint32 generation]
// The generation word is the region's last 4 bytes; the producer bumps it each
// time it reuses the slot. The refcount keeps the producer off slots consumers
// still read.
constexpr size_t kExternalMemMapGenerationBytes = sizeof(uint32_t);
constexpr size_t kExternalMemMapTrailerBytes =
  sizeof(IPCMetadata) + alignof(IPCMetadata) + kExternalMemMapGenerationBytes;

inline std::atomic<uint32_t> * external_memmap_generation_ptr(
  void * region, size_t region_size)
{
  if (region == nullptr || region_size < kExternalMemMapGenerationBytes) {
    return nullptr;
  }
  return reinterpret_cast<std::atomic<uint32_t> *>(
    static_cast<uint8_t *>(region) + region_size - kExternalMemMapGenerationBytes);
}

inline IPCMetadata * external_memmap_slot_meta_ptr(void * region, size_t region_size)
{
  if (region == nullptr || region_size < kExternalMemMapTrailerBytes) {
    return nullptr;
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(region) + region_size -
    kExternalMemMapGenerationBytes - sizeof(IPCMetadata);
  addr &= ~(static_cast<uintptr_t>(alignof(IPCMetadata)) - 1);
  return reinterpret_cast<IPCMetadata *>(addr);
}

class L0ExternalMemMap
{
public:
  static void * map(void * ptr, size_t size, bool read_only = false);

  static void unmap(void * usm_ptr);

  static bool supported();
};

class SharedSysmem
{
public:
  SharedSysmem() = default;
  ~SharedSysmem();

  SharedSysmem(const SharedSysmem &) = delete;
  SharedSysmem & operator=(const SharedSysmem &) = delete;
  SharedSysmem(SharedSysmem && other) noexcept;
  SharedSysmem & operator=(SharedSysmem && other) noexcept;

  bool create(const std::string & name, size_t size);

  bool open(const std::string & name);

  void * ptr() const {return ptr_;}
  size_t size() const {return size_;}
  const std::string & name() const {return name_;}

  void close();

private:
  void * ptr_{nullptr};
  size_t size_{0};
  std::string name_;
  bool owner_{false};
};

/// Regions this process produces (ExternalMemMapRing slots): shm name -> the
/// pointer published for it as a descriptor's local_ptr. Lets an intra-process
/// import verify local_ptr instead of trusting wire data.
class ExternalMemMapLocalRegions
{
public:
  static void add(const std::string & shm_name, const void * ptr, size_t capacity);

  static void remove(const std::string & shm_name);

  /// True if this process produces `shm_name` at `ptr`, with room for `size` bytes.
  static bool owns(const std::string & shm_name, const void * ptr, size_t size);
};

class ExternalMemMapImporter
{
public:
  static void * import(
    const std::string & shm_name, size_t size, int32_t publisher_pid,
    size_t * out_region_size = nullptr);

  static bool generation_unchanged(
    const void * region, size_t region_size, uint32_t published_generation);
};

}  // namespace intel_memory_core
#endif  // INTEL_MEMORY_CORE__L0_EXTERNAL_MEMMAP_HPP_
