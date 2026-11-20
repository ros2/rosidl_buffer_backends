// Copyright 2026 Open Source Robotics Foundation, Inc.
// Modified for the HIP port by ObeliskGate, 2026.
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

#ifndef HIP_BUFFER__HIP_MEMORY_POOL_HPP_
#define HIP_BUFFER__HIP_MEMORY_POOL_HPP_

#include <hip/hip_runtime_api.h>

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

#include "hip_buffer/hip_buffer_ipc_manager.hpp"
#include "hip_buffer/hip_error.hpp"

namespace hip_buffer_backend
{

/// \brief Grace-period configuration for a HIP allocation pool.
struct HipPoolConfig
{
  uint64_t grace_period_us{250000};
};

struct IPCMetadata
{
  std::atomic<int32_t> refcount{0};
  // Zero closes the block to new imports. A nonzero value identifies the
  // currently published generation.
  std::atomic<uint64_t> uid{0};
  std::atomic<uint64_t> publish_timestamp_us{0};
};

/// \brief VMM allocation block with optional pre-exported IPC handle.
struct VmmBlock
{
  hipMemGenericAllocationHandle_t handle{};
  void * va{nullptr};
  size_t size{0};
  int exported_fd{-1};
  uint32_t block_id{0};

  IPCMetadata * ipc_meta{nullptr};
  int shm_fd{-1};
  std::string shm_name;
  uint64_t current_uid{0};
  hipEvent_t exported_event{nullptr};
  // A terminal event error permanently quarantines this block from reuse.
  bool exported_event_retirement_failed{false};
};

/// \brief Publisher-side VMM allocation pool used by HipBufferImpl.
///
/// HipMemoryPool owns the real HIP VMM allocations. HipBuffer instances
/// represent temporary uses of blocks from this pool and return blocks via the
/// deleter produced by HipMemoryPool::deleter(). The pool decides whether a
/// returned block is ready for reuse by checking per-block IPC metadata
/// (subscriber refcount, UID, and publish grace window).
class HipMemoryPool : public std::enable_shared_from_this<HipMemoryPool>
{
public:
  explicit HipMemoryPool(const HipPoolConfig & config = {});
  ~HipMemoryPool();

  HipMemoryPool(const HipMemoryPool &) = delete;
  HipMemoryPool & operator=(const HipMemoryPool &) = delete;
  HipMemoryPool(HipMemoryPool &&) = delete;
  HipMemoryPool & operator=(HipMemoryPool &&) = delete;

  hipError_t create();
  VmmBlock * allocate(size_t byte_size);
  void free(VmmBlock * block);
  void retain_exported_event(VmmBlock * block, hipEvent_t event, uint64_t uid);

  uint64_t assign_uid(VmmBlock * block)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (block->current_uid != 0) {
      return block->current_uid;
    }
    uint64_t uid = uid_dist_(uid_rng_);
    block->current_uid = uid;
    if (block->ipc_meta) {
      block->ipc_meta->publish_timestamp_us.store(
        static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()),
        std::memory_order_release);
      // Publish the generation only after all of its metadata is initialized.
      block->ipc_meta->uid.store(uid, std::memory_order_seq_cst);
    }
    return uid;
  }

  std::function<void(uint8_t *)> deleter(VmmBlock * block)
  {
    auto self = shared_from_this();
    return [self, block](uint8_t *) {
             self->free(block);
           };
  }

  std::string register_block_for_ipc(VmmBlock * block)
  {
    if (!ipc_capable_ || !ipc_manager_ || block->exported_fd < 0 || !block->ipc_meta) {
      return "";
    }
    return ipc_manager_->register_block(block);
  }

  VmmBlock * find_block_for_va(void * va) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto address = reinterpret_cast<std::uintptr_t>(va);
    for (auto & block : all_blocks_) {
      const auto base = reinterpret_cast<std::uintptr_t>(block->va);
      if (address >= base && address - base < block->size) {
        return block.get();
      }
    }
    return nullptr;
  }

  bool is_ipc_capable() const {return ipc_capable_;}
  int get_device_id() const {return device_id_;}
  const HipPoolConfig & get_config() const noexcept {return config_;}

private:
  enum class BlockReadiness
  {
    Ready,
    Pending,
    TerminalError,
  };

  BlockReadiness get_block_readiness(VmmBlock * block);
  BlockReadiness retire_exported_event(VmmBlock * block);
  bool probe_vmm_ipc();
  VmmBlock * create_block(size_t aligned_size);

  size_t round_to_granularity(size_t size) const
  {
    return ((size + granularity_ - 1) / granularity_) * granularity_;
  }

  hipMemAllocationProp prop_{};
  size_t granularity_{0};
  int device_id_{0};
  bool ipc_capable_{false};
  bool initialized_{false};
  uint32_t next_block_id_{0};
  const HipPoolConfig config_;
  std::mt19937_64 uid_rng_{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> uid_dist_{1, UINT64_MAX};

  std::map<size_t, std::vector<VmmBlock *>> free_blocks_;
  std::vector<std::unique_ptr<VmmBlock>> all_blocks_;
  std::unique_ptr<HipVmmIPCManager> ipc_manager_;
  mutable std::mutex mutex_;
};

}  // namespace hip_buffer_backend

#endif  // HIP_BUFFER__HIP_MEMORY_POOL_HPP_
