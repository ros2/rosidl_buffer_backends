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

#include "cuda_buffer/cuda_memory_pool.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <rcutils/logging_macros.h>

#include <cerrno>
#include <chrono>
#include <string>
#include <thread>

namespace cuda_buffer_backend
{

CudaMemoryPool::~CudaMemoryPool()
{
  for (;; ) {
    bool all_ready = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto & block : all_blocks_) {
        if (!is_block_ready(block.get())) {
          all_ready = false;
          break;
        }
      }
    }
    if (all_ready) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto & block : all_blocks_) {
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
    if (block->local_event) {
      cudaEventDestroy(block->local_event);
      block->local_event = nullptr;
    }
    if (block->va != 0) {
      cuMemUnmap(block->va, block->size);
      cuMemAddressFree(block->va, block->size);
    }
    if (block->handle != 0) {
      cuMemRelease(block->handle);
    }
  }
}

CUresult CudaMemoryPool::create()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return CUDA_ERROR_ALREADY_ACQUIRED;
  }

  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) {
    return r;
  }

  cudaError_t rt = cudaGetDevice(&device_id_);
  if (rt != cudaSuccess) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  rt = cudaFree(nullptr);
  if (rt != cudaSuccess && rt != cudaErrorInvalidValue) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }

  prop_ = {};
  prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop_.location.id = device_id_;
  prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  r = cuMemGetAllocationGranularity(
    &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (r != CUDA_SUCCESS || granularity_ == 0) {
    prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    r = cuMemGetAllocationGranularity(
      &granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (r != CUDA_SUCCESS || granularity_ == 0) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    ipc_capable_ = false;
  } else {
    ipc_capable_ = probe_vmm_ipc();
    if (!ipc_capable_) {
      prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    }
  }

  if (ipc_capable_) {
    ipc_manager_ = std::make_unique<CudaVmmIPCManager>();
  } else {
    RCUTILS_LOG_INFO_NAMED("cuda_memory_pool",
      "VMM IPC not supported on device %d; "
      "cross-process transport will use CPU fallback", device_id_);
  }

  initialized_ = true;
  return CUDA_SUCCESS;
}

VmmBlock * CudaMemoryPool::allocate(size_t byte_size)
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
      if (is_block_ready(vec[i])) {
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

void CudaMemoryPool::free(VmmBlock * block)
{
  std::lock_guard<std::mutex> lock(mutex_);
  free_blocks_[block->size].push_back(block);
}

bool CudaMemoryPool::is_block_ready(VmmBlock * block) const
{
  if (!block->ipc_meta) {
    return true;
  }

  int32_t rc = block->ipc_meta->refcount.load(std::memory_order_acquire);
  if (rc > 0) {
    return false;
  }

  uint64_t publish_us = block->ipc_meta->publish_timestamp_us.load(
    std::memory_order_acquire);
  if (publish_us == 0) {
    return true;
  }

  uint64_t now = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  if (now < publish_us) {
    return false;
  }
  return (now - publish_us) >= grace_period_us_;
}

bool CudaMemoryPool::probe_vmm_ipc()
{
  CUmemGenericAllocationHandle h = 0;
  CUresult r = cuMemCreate(&h, granularity_, &prop_, 0);
  if (r != CUDA_SUCCESS) {
    return false;
  }
  int fd = -1;
  r = cuMemExportToShareableHandle(
    &fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (fd >= 0) {
    close(fd);
  }
  cuMemRelease(h);
  return r == CUDA_SUCCESS;
}

VmmBlock * CudaMemoryPool::create_block(size_t aligned_size)
{
  auto block = std::make_unique<VmmBlock>();
  block->size = aligned_size;
  block->block_id = next_block_id_++;

  CUresult r = cuMemCreate(&block->handle, aligned_size, &prop_, 0);
  if (r != CUDA_SUCCESS) {
    throw CudaError(__FILE__, __LINE__, "cuMemCreate", r);
  }

  r = cuMemAddressReserve(&block->va, aligned_size, 0, 0, 0);
  if (r != CUDA_SUCCESS) {
    cuMemRelease(block->handle);
    throw CudaError(__FILE__, __LINE__, "cuMemAddressReserve", r);
  }

  r = cuMemMap(block->va, aligned_size, 0, block->handle, 0);
  if (r != CUDA_SUCCESS) {
    cuMemAddressFree(block->va, aligned_size);
    cuMemRelease(block->handle);
    throw CudaError(__FILE__, __LINE__, "cuMemMap", r);
  }

  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device_id_;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  r = cuMemSetAccess(block->va, aligned_size, &access, 1);
  if (r != CUDA_SUCCESS) {
    cuMemUnmap(block->va, aligned_size);
    cuMemAddressFree(block->va, aligned_size);
    cuMemRelease(block->handle);
    throw CudaError(__FILE__, __LINE__, "cuMemSetAccess", r);
  }

  cudaError_t event_result = cudaEventCreateWithFlags(
    &block->local_event, cudaEventDisableTiming);
  if (event_result != cudaSuccess) {
    block->local_event = nullptr;
    (void)cudaGetLastError();
    RCUTILS_LOG_WARN_NAMED(
      "cuda_memory_pool",
      "Failed to create same-process CUDA event; local transport may synchronize");
  }

  if (ipc_capable_) {
    int fd = -1;
    r = cuMemExportToShareableHandle(
      &fd, block->handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    if (r == CUDA_SUCCESS) {
      block->exported_fd = fd;
    }

    std::string shm_name = "/cuda_vmm_" + std::to_string(getpid()) +
      "_" + std::to_string(block->block_id);
    shm_unlink(shm_name.c_str());
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd >= 0) {
      if (ftruncate(shm_fd, sizeof(IPCMetadata)) != 0) {
        RCUTILS_LOG_WARN_NAMED("cuda_memory_pool",
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

}  // namespace cuda_buffer_backend
