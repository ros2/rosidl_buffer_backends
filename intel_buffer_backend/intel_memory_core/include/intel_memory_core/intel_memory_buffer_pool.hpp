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

#ifndef INTEL_MEMORY_CORE__INTEL_MEMORY_BUFFER_POOL_HPP_
#define INTEL_MEMORY_CORE__INTEL_MEMORY_BUFFER_POOL_HPP_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace intel_memory_core
{

static constexpr int kZeIpcHandleBytes = 64;

static constexpr int kZeDeviceUuidBytes = 16;

static constexpr int kMaxImportHolders = 16;

struct IPCMetadata
{
  std::atomic<int32_t> refcount{0};
  std::atomic<uint64_t> uid{0};
  std::atomic<uint64_t> publish_timestamp_us{0};
  std::atomic<int32_t> holder_pids[kMaxImportHolders]{};
};

enum class TransportMode : uint8_t
{
  LEVEL_ZERO_USM = 0,
  DMA_BUF = 1,
  EXTERNAL_MEMMAP = 2
};

/// A single allocation block in the memory pool.
struct PoolBlock
{
  void * ptr{nullptr};             // Virtual address (USM or mmap'd DMA-BUF)
  size_t size{0};                  // Allocation size in bytes
  uint32_t block_id{0};            // Unique block identifier
  TransportMode transport{TransportMode::LEVEL_ZERO_USM};

  // Level Zero USM fields
  void * ze_context{nullptr};      // ze_context_handle_t (opaque)
  int32_t ze_device_ordinal{-1};
  unsigned char ze_device_uuid[kZeDeviceUuidBytes]{};
  unsigned char ze_ipc_handle[kZeIpcHandleBytes]{};
  bool ze_ipc_handle_valid{false};

  // DMA-BUF fields
  int dmabuf_fd{-1};               // DMA-BUF file descriptor
  int dmabuf_heap_fd{-1};          // /dev/dma_heap fd used for allocation

  std::string extmap_shm_name;
  uint32_t extmap_generation{0};

  // IPC metadata (shared memory backed)
  IPCMetadata * ipc_meta{nullptr};
  int shm_fd{-1};
  std::string shm_name;
  uint64_t current_uid{0};
};

/// Configuration for pool growth and block reuse.
struct PoolConfig
{
  size_t max_blocks{64};
  uint64_t grace_period_us{250000};    // 250 ms

};

class IntelMemoryBufferPool : public std::enable_shared_from_this<IntelMemoryBufferPool>
{
public:
  IntelMemoryBufferPool();
  ~IntelMemoryBufferPool();

  IntelMemoryBufferPool(const IntelMemoryBufferPool &) = delete;
  IntelMemoryBufferPool & operator=(const IntelMemoryBufferPool &) = delete;
  IntelMemoryBufferPool(IntelMemoryBufferPool &&) = delete;
  IntelMemoryBufferPool & operator=(IntelMemoryBufferPool &&) = delete;

  bool initialize(const PoolConfig & config = PoolConfig{});

  PoolBlock * allocate(size_t byte_size);

  /// Return a block to the pool for reuse.
  void release(PoolBlock * block);

  /// Assign a unique publish UID to a block (for staleness detection).
  uint64_t assign_uid(PoolBlock * block);

  /// Create a custom deleter for use with CudaBuffer-style RAII.
  std::function<void(uint8_t *)> deleter(PoolBlock * block);

  /// Get pool statistics.
  size_t total_blocks() const;
  size_t free_blocks() const;
  size_t active_blocks() const;
  uint32_t generation() const { return generation_.load(std::memory_order_relaxed); }
  uint32_t pool_id() const { return pool_id_; }

  /// Query transport mode chosen at initialization.
  TransportMode transport_mode() const { return transport_mode_; }

  /// Check if Level Zero IPC is supported.
  bool is_level_zero_capable() const { return level_zero_capable_; }

  /// Check if DMA-BUF is supported.
  bool is_dmabuf_capable() const { return dmabuf_capable_; }

  /// Get the Level Zero device ordinal.
  int32_t ze_device_ordinal() const { return ze_device_ordinal_; }

  /// Find a block by its virtual address.
  PoolBlock * find_block_for_ptr(void * ptr) const;

  /// Register a block for IPC export (returns socket path for DMA-BUF mode).
  std::string register_block_for_ipc(PoolBlock * block);

private:
  bool probe_level_zero();
  bool probe_dmabuf();
  PoolBlock * create_block(size_t aligned_size);
  void destroy_block(PoolBlock * block);
  bool is_block_ready(PoolBlock * block) const;
  size_t align_size(size_t size) const;

  PoolConfig config_;
  TransportMode transport_mode_{TransportMode::LEVEL_ZERO_USM};
  bool level_zero_capable_{false};
  bool dmabuf_capable_{false};
  bool initialized_{false};
  int32_t ze_device_ordinal_{-1};
  unsigned char ze_device_uuid_[kZeDeviceUuidBytes]{};  // stable device identity
  void * ze_driver_{nullptr};      // ze_driver_handle_t
  void * ze_device_{nullptr};      // ze_device_handle_t
  void * ze_context_{nullptr};     // ze_context_handle_t
  size_t alignment_{65536};        // 64KiB default alignment

  uint32_t pool_id_{0};
  uint32_t next_block_id_{0};
  std::atomic<uint32_t> generation_{0};

  std::mt19937_64 uid_rng_{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> uid_dist_{1, UINT64_MAX};

  std::map<size_t, std::vector<PoolBlock *>> free_blocks_;
  std::vector<std::unique_ptr<PoolBlock>> all_blocks_;
  mutable std::mutex mutex_;

  // DMA-BUF IPC server state
  struct DmaBufServer;
  std::unique_ptr<DmaBufServer> dmabuf_server_;
};

}  // namespace intel_memory_core

#endif  // INTEL_MEMORY_CORE__INTEL_MEMORY_BUFFER_POOL_HPP_
