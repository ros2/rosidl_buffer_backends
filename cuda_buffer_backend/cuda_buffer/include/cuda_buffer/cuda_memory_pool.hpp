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

#ifndef CUDA_BUFFER__CUDA_MEMORY_POOL_HPP_
#define CUDA_BUFFER__CUDA_MEMORY_POOL_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

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

#include "cuda_buffer/cuda_buffer_ipc_manager.hpp"
#include "cuda_buffer/cuda_error.hpp"

namespace cuda_buffer_backend
{

struct IPCMetadata
{
  std::atomic<int32_t> refcount{0};
  std::atomic<uint64_t> uid{0};
  std::atomic<uint64_t> publish_timestamp_us{0};
};

/// \brief VMM allocation block with optional pre-exported IPC handle.
struct VmmBlock
{
  CUmemGenericAllocationHandle handle{0};
  CUdeviceptr va{0};
  size_t size{0};
  int exported_fd{-1};
  uint32_t block_id{0};
  cudaEvent_t local_event{nullptr};

  IPCMetadata * ipc_meta{nullptr};
  int shm_fd{-1};
  std::string shm_name;
  uint64_t current_uid{0};
};

/// \brief Publisher-side VMM allocation pool used by CudaBufferImpl.
///
/// CudaMemoryPool owns the real CUDA VMM allocations. CudaBuffer instances
/// represent temporary uses of blocks from this pool and return blocks via the
/// deleter produced by CudaMemoryPool::deleter(). The pool decides whether a
/// returned block is ready for reuse by checking per-block IPC metadata
/// (subscriber refcount, UID, and publish grace window).
class CudaMemoryPool : public std::enable_shared_from_this<CudaMemoryPool>
{
public:
  CudaMemoryPool() = default;
  ~CudaMemoryPool();

  CudaMemoryPool(const CudaMemoryPool &) = delete;
  CudaMemoryPool & operator=(const CudaMemoryPool &) = delete;
  CudaMemoryPool(CudaMemoryPool &&) = delete;
  CudaMemoryPool & operator=(CudaMemoryPool &&) = delete;

  CUresult create();
  VmmBlock * allocate(size_t byte_size);
  void free(VmmBlock * block);

  uint64_t assign_uid(VmmBlock * block)
  {
    if (block->current_uid != 0) {
      return block->current_uid;
    }
    uint64_t uid = uid_dist_(uid_rng_);
    block->current_uid = uid;
    if (block->ipc_meta) {
      block->ipc_meta->uid.store(uid, std::memory_order_release);
      block->ipc_meta->publish_timestamp_us.store(
        static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()),
        std::memory_order_release);
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
    if (!ipc_capable_ || !ipc_manager_ || block->exported_fd < 0) {
      return "";
    }
    return ipc_manager_->register_block(block);
  }

  VmmBlock * find_block_for_va(CUdeviceptr va) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & block : all_blocks_) {
      if (va >= block->va && va < block->va + block->size) {
        return block.get();
      }
    }
    return nullptr;
  }

  bool is_ipc_capable() const {return ipc_capable_;}
  int get_device_id() const {return device_id_;}

private:
  bool is_block_ready(VmmBlock * block) const;
  bool probe_vmm_ipc();
  VmmBlock * create_block(size_t aligned_size);

  size_t round_to_granularity(size_t size) const
  {
    return ((size + granularity_ - 1) / granularity_) * granularity_;
  }

  CUmemAllocationProp prop_{};
  size_t granularity_{0};
  int device_id_{0};
  bool ipc_capable_{false};
  bool initialized_{false};
  uint32_t next_block_id_{0};
  static constexpr uint64_t grace_period_us_{100000};
  std::mt19937_64 uid_rng_{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> uid_dist_{1, UINT64_MAX};

  std::map<size_t, std::vector<VmmBlock *>> free_blocks_;
  std::vector<std::unique_ptr<VmmBlock>> all_blocks_;
  std::unique_ptr<CudaVmmIPCManager> ipc_manager_;
  mutable std::mutex mutex_;
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__CUDA_MEMORY_POOL_HPP_
