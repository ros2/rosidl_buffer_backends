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

#include "hip_buffer/hip_memory_pool.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <rcutils/logging_macros.h>

#include <cerrno>
#include <chrono>
#include <string>
#include <thread>

namespace hip_buffer_backend
{

HipMemoryPool::HipMemoryPool(const HipPoolConfig & config)
: config_(config) {}

HipMemoryPool::~HipMemoryPool()
{
  for (;; ) {
    bool has_pending_block = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Pending work can make progress. A terminal HIP error cannot, so the
      // final cleanup below handles it once instead of retrying forever.
      for (auto & block : all_blocks_) {
        if (get_block_readiness(block.get()) == BlockReadiness::Pending) {
          has_pending_block = true;
        }
      }
    }
    if (!has_pending_block) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto & block : all_blocks_) {
    if (block->exported_event) {
      hipEventDestroy(block->exported_event);
      block->exported_event = nullptr;
      (void)hipGetLastError();
    }
    if (block->ipc_meta) {
      munmap(block->ipc_meta, sizeof(IPCMetadata));
      block->ipc_meta = nullptr;
    }
    if (block->shm_fd >= 0) {
      close(block->shm_fd);
      shm_unlink(block->shm_name.c_str());
    }
    if (block->exported_fd >= 0) {
      close(block->exported_fd);
    }
    if (block->va != nullptr) {
      hipMemUnmap(block->va, block->size);
      hipMemAddressFree(block->va, block->size);
    }
    if (block->handle != 0) {
      hipMemRelease(block->handle);
    }
  }
}

hipError_t HipMemoryPool::create()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return hipErrorAlreadyAcquired;
  }

  hipError_t r = hipInit(0);
  if (r != hipSuccess) {
    return r;
  }

  hipError_t rt = hipGetDevice(&device_id_);
  if (rt != hipSuccess) {
    return hipErrorInvalidDevice;
  }

  rt = hipFree(nullptr);
  if (rt != hipSuccess && rt != hipErrorInvalidValue) {
    return hipErrorNotInitialized;
  }

  prop_ = {};
  prop_.type = hipMemAllocationTypePinned;
  prop_.location.type = hipMemLocationTypeDevice;
  prop_.location.id = device_id_;
  prop_.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;

  r = hipMemGetAllocationGranularity(
    &granularity_, &prop_, hipMemAllocationGranularityMinimum);
  if (r != hipSuccess || granularity_ == 0) {
    prop_.requestedHandleTypes = hipMemHandleTypeNone;
    r = hipMemGetAllocationGranularity(
      &granularity_, &prop_, hipMemAllocationGranularityMinimum);
    if (r != hipSuccess || granularity_ == 0) {
      return hipErrorNotSupported;
    }
    ipc_capable_ = false;
  } else {
    ipc_capable_ = probe_vmm_ipc();
    if (!ipc_capable_) {
      prop_.requestedHandleTypes = hipMemHandleTypeNone;
    }
  }

  if (ipc_capable_) {
    ipc_manager_ = std::make_unique<HipVmmIPCManager>();
  } else {
    RCUTILS_LOG_INFO_NAMED("hip_memory_pool",
      "VMM IPC not supported on device %d; "
      "cross-process transport will use CPU fallback", device_id_);
  }

  initialized_ = true;
  return hipSuccess;
}

VmmBlock * HipMemoryPool::allocate(size_t byte_size)
{
  size_t aligned = round_to_granularity(byte_size);

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = free_blocks_.lower_bound(aligned);
  if (it != free_blocks_.end() && !it->second.empty()) {
    auto & vec = it->second;
    // Linear scan for a ready block. Order within the bucket is irrelevant,
    // so swap-and-pop gives O(1) removal without the shift std::vector::erase
    // would incur. A std::list would also give O(1) erase, but its scattered
    // heap nodes cost more per-iteration than the cache-friendly vector scan.
    for (size_t i = 0; i < vec.size(); ++i) {
      if (get_block_readiness(vec[i]) == BlockReadiness::Ready) {
        VmmBlock * block = vec[i];
        vec[i] = vec.back();
        vec.pop_back();
        if (vec.empty()) {
          free_blocks_.erase(it);
        }
        block->current_uid = 0;
        if (block->ipc_meta) {
          block->ipc_meta->publish_timestamp_us.store(0, std::memory_order_release);
        }
        return block;
      }
    }
  }

  return create_block(aligned);
}

void HipMemoryPool::free(VmmBlock * block)
{
  std::lock_guard<std::mutex> lock(mutex_);
  free_blocks_[block->size].push_back(block);
}

void HipMemoryPool::retain_exported_event(
  VmmBlock * block, hipEvent_t event, uint64_t uid)
{
  if (!block || !event || uid == 0) {
    throw HipError("Cannot retain an invalid exported HIP event");
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (block->current_uid != uid) {
    throw HipError("Cannot retain a HIP event for a stale block generation");
  }
  if (block->exported_event && block->exported_event != event) {
    throw HipError("A different HIP event is already exported for this block generation");
  }
  block->exported_event = event;
  block->exported_event_retirement_failed = false;
}

HipMemoryPool::BlockReadiness HipMemoryPool::retire_exported_event(VmmBlock * block)
{
  if (block->exported_event_retirement_failed) {
    return BlockReadiness::TerminalError;
  }
  if (!block->exported_event) {
    return BlockReadiness::Ready;
  }

  const hipError_t status = hipEventQuery(block->exported_event);
  if (status == hipErrorNotReady) {
    (void)hipGetLastError();
    return BlockReadiness::Pending;
  }
  if (status != hipSuccess) {
    RCUTILS_LOG_WARN_NAMED(
      "hip_memory_pool",
      "hipEventQuery failed while retiring IPC event; block will not be reused: %s",
      hipGetErrorName(status));
    (void)hipGetLastError();
    block->exported_event_retirement_failed = true;
    return BlockReadiness::TerminalError;
  }

  const hipError_t destroy_status = hipEventDestroy(block->exported_event);
  block->exported_event = nullptr;
  (void)hipGetLastError();
  if (destroy_status != hipSuccess) {
    RCUTILS_LOG_WARN_NAMED(
      "hip_memory_pool",
      "hipEventDestroy failed while retiring IPC event; block will not be reused: %s",
      hipGetErrorName(destroy_status));
    block->exported_event_retirement_failed = true;
    return BlockReadiness::TerminalError;
  }
  return BlockReadiness::Ready;
}

HipMemoryPool::BlockReadiness HipMemoryPool::get_block_readiness(VmmBlock * block)
{
  if (!block->ipc_meta) {
    return retire_exported_event(block);
  }

  uint64_t publish_us = block->ipc_meta->publish_timestamp_us.load(
    std::memory_order_acquire);
  if (publish_us == 0) {
    return retire_exported_event(block);
  }

  uint64_t now = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  if (now < publish_us || (now - publish_us) < config_.grace_period_us) {
    return BlockReadiness::Pending;
  }

  // UID zero closes the generation. A consumer either holds a counted lease
  // before this store, or observes a closed/stale UID before opening its IPC
  // event. Using one atomic for identity and state also prevents a close/reopen
  // race from accepting a descriptor for the previous generation.
  block->ipc_meta->uid.store(0, std::memory_order_seq_cst);
  if (block->ipc_meta->refcount.load(std::memory_order_seq_cst) > 0) {
    return BlockReadiness::Pending;
  }

  return retire_exported_event(block);
}

bool HipMemoryPool::probe_vmm_ipc()
{
  hipMemGenericAllocationHandle_t h{};
  hipError_t r = hipMemCreate(&h, granularity_, &prop_, 0);
  if (r != hipSuccess) {
    return false;
  }
  int fd = -1;
  r = hipMemExportToShareableHandle(
    &fd, h, hipMemHandleTypePosixFileDescriptor, 0);
  if (fd >= 0) {
    close(fd);
  }
  hipMemRelease(h);
  return r == hipSuccess;
}

VmmBlock * HipMemoryPool::create_block(size_t aligned_size)
{
  auto block = std::make_unique<VmmBlock>();
  block->size = aligned_size;
  block->block_id = next_block_id_++;

  hipError_t r = hipMemCreate(&block->handle, aligned_size, &prop_, 0);
  if (r != hipSuccess) {
    throw HipError(__FILE__, __LINE__, "hipMemCreate", r);
  }

  r = hipMemAddressReserve(&block->va, aligned_size, 0, nullptr, 0);
  if (r != hipSuccess) {
    hipMemRelease(block->handle);
    throw HipError(__FILE__, __LINE__, "hipMemAddressReserve", r);
  }

  r = hipMemMap(block->va, aligned_size, 0, block->handle, 0);
  if (r != hipSuccess) {
    hipMemAddressFree(block->va, aligned_size);
    hipMemRelease(block->handle);
    throw HipError(__FILE__, __LINE__, "hipMemMap", r);
  }

  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = device_id_;
  access.flags = hipMemAccessFlagsProtReadWrite;
  r = hipMemSetAccess(block->va, aligned_size, &access, 1);
  if (r != hipSuccess) {
    hipMemUnmap(block->va, aligned_size);
    hipMemAddressFree(block->va, aligned_size);
    hipMemRelease(block->handle);
    throw HipError(__FILE__, __LINE__, "hipMemSetAccess", r);
  }

  if (ipc_capable_) {
    int fd = -1;
    r = hipMemExportToShareableHandle(
      &fd, block->handle, hipMemHandleTypePosixFileDescriptor, 0);
    if (r == hipSuccess) {
      block->exported_fd = fd;
    }

    std::string shm_name = "/hip_vmm_" + std::to_string(getpid()) +
      "_" + std::to_string(block->block_id);
    shm_unlink(shm_name.c_str());
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd >= 0) {
      if (ftruncate(shm_fd, sizeof(IPCMetadata)) != 0) {
        RCUTILS_LOG_WARN_NAMED("hip_memory_pool",
          "ftruncate failed for IPC metadata segment '%s' (errno=%d); "
          "block will be created without IPC metadata",
          shm_name.c_str(), errno);
        close(shm_fd);
        shm_unlink(shm_name.c_str());
      } else {
        void * ptr = mmap(nullptr, sizeof(IPCMetadata),
          PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (ptr != MAP_FAILED) {
          block->ipc_meta = new(ptr) IPCMetadata();
          block->shm_fd = shm_fd;
          block->shm_name = shm_name;
        } else {
          close(shm_fd);
          shm_unlink(shm_name.c_str());
        }
      }
    }
  }

  VmmBlock * ptr = block.get();
  all_blocks_.push_back(std::move(block));
  return ptr;
}

}  // namespace hip_buffer_backend
