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

#include "intel_memory_core/intel_memory_buffer_pool.hpp"
#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/intel_memory_ipc_manager.hpp"
#include "intel_memory_core/l0_usm_ipc.hpp"

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <csignal>

#include <level_zero/ze_api.h>

#include "intel_memory_core/logging.hpp"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>

namespace intel_memory_core
{

// DMA-BUF IPC server (AF_UNIX fd export via SCM_RIGHTS)
struct IntelMemoryBufferPool::DmaBufServer
{
};

IntelMemoryBufferPool::IntelMemoryBufferPool() = default;

IntelMemoryBufferPool::~IntelMemoryBufferPool()
{
  for (;;) {
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
    destroy_block(block.get());
  }
  all_blocks_.clear();
  free_blocks_.clear();

  // Destroy Level Zero context
  if (ze_context_) {
    zeContextDestroy(static_cast<ze_context_handle_t>(ze_context_));
    ze_context_ = nullptr;
  }
}

bool IntelMemoryBufferPool::initialize(const PoolConfig & config)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return true;
  }

  config_ = config;
  pool_id_ = std::random_device{}();

  level_zero_capable_ = probe_level_zero();

  dmabuf_capable_ = probe_dmabuf();

  const char * env_transport = std::getenv("INTEL_MEMORY_BUFFER_TYPE");
  bool force_dmabuf = false;
  if (env_transport) {
    std::string t(env_transport);
    if (t == "dmabuf") force_dmabuf = true;
  }

  if (level_zero_capable_ && !force_dmabuf) {
    transport_mode_ = TransportMode::LEVEL_ZERO_USM;
    RCUTILS_LOG_INFO_NAMED("intel_memory_buffer_pool",
      "Level Zero USM (device ordinal %d); "
      "zero-copy across CPU/iGPU/NPU via zeMemAllocHost IPC",
      ze_device_ordinal_);
  } else if (dmabuf_capable_) {
    transport_mode_ = TransportMode::DMA_BUF;
    RCUTILS_LOG_INFO_NAMED("intel_memory_buffer_pool",
      "Using DMA-BUF transport for cross-process zero-copy");
  } else {
    RCUTILS_LOG_WARN_NAMED("intel_memory_buffer_pool",
      "Neither Level Zero nor DMA-BUF available; "
      "buffer backend will fall back to CPU copies");
    initialized_ = true;
    return true;
  }

  if (dmabuf_capable_) {
    dmabuf_server_ = std::make_unique<DmaBufServer>();
  }

  initialized_ = true;
  return true;
}

PoolBlock * IntelMemoryBufferPool::allocate(size_t byte_size)
{
  size_t aligned = align_size(byte_size);

  std::lock_guard<std::mutex> lock(mutex_);

  // Try to reuse a free block of suitable size
  auto it = free_blocks_.lower_bound(aligned);
  if (it != free_blocks_.end() && !it->second.empty()) {
    auto & vec = it->second;
    for (size_t i = 0; i < vec.size(); ++i) {
      if (is_block_ready(vec[i])) {
        PoolBlock * block = vec[i];
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

  if (all_blocks_.size() >= config_.max_blocks) {
    auto has_live_holder = [](PoolBlock * b) {
      if (!b->ipc_meta) {
        return false;
      }
      for (int i = 0; i < kMaxImportHolders; ++i) {
        int32_t pid = b->ipc_meta->holder_pids[i].load(std::memory_order_acquire);
        // Live if the PID is set and kill(pid, 0) does not report ESRCH.
        if (pid != 0 && !(kill(pid, 0) != 0 && errno == ESRCH)) {
          return true;
        }
      }
      return false;
    };

    auto pick = free_blocks_.end();
    size_t pick_idx = 0;
    for (auto it = free_blocks_.lower_bound(aligned); it != free_blocks_.end(); ++it) {
      for (size_t i = 0; i < it->second.size(); ++i) {
        if (!has_live_holder(it->second[i])) {
          pick = it;
          pick_idx = i;
          break;
        }
      }
      if (pick != free_blocks_.end()) {
        break;
      }
    }
    if (pick == free_blocks_.end()) {
      auto it = free_blocks_.lower_bound(aligned);
      if (it != free_blocks_.end() && !it->second.empty()) {
        pick = it;
        pick_idx = 0;
      }
    }

    if (pick != free_blocks_.end()) {
      PoolBlock * block = pick->second[pick_idx];
      pick->second.erase(pick->second.begin() + pick_idx);
      if (pick->second.empty()) {
        free_blocks_.erase(pick);
      }
      block->current_uid = 0;
      if (block->ipc_meta) {
        for (int i = 0; i < kMaxImportHolders; ++i) {
          block->ipc_meta->holder_pids[i].store(0, std::memory_order_release);
        }
        block->ipc_meta->refcount.store(0, std::memory_order_release);
        block->ipc_meta->publish_timestamp_us.store(0, std::memory_order_release);
      }
      return block;
    }

    RCUTILS_LOG_WARN_NAMED("intel_memory_buffer_pool",
      "Memory pool exhausted (%zu blocks, all in-flight); frame dropped",
      config_.max_blocks);
    return nullptr;
  }

  PoolBlock * block = create_block(aligned);
  generation_.fetch_add(1, std::memory_order_relaxed);
  return block;
}

void IntelMemoryBufferPool::release(PoolBlock * block)
{
  std::lock_guard<std::mutex> lock(mutex_);
  free_blocks_[block->size].push_back(block);
}

uint64_t IntelMemoryBufferPool::assign_uid(PoolBlock * block)
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

std::function<void(uint8_t *)> IntelMemoryBufferPool::deleter(PoolBlock * block)
{
  auto self = shared_from_this();
  return [self, block](uint8_t *) {
           self->release(block);
         };
}

size_t IntelMemoryBufferPool::total_blocks() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return all_blocks_.size();
}

size_t IntelMemoryBufferPool::free_blocks() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (auto & [sz, vec] : free_blocks_) {
    count += vec.size();
  }
  return count;
}

size_t IntelMemoryBufferPool::active_blocks() const
{
  return total_blocks() - free_blocks();
}

PoolBlock * IntelMemoryBufferPool::find_block_for_ptr(void * ptr) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto & block : all_blocks_) {
    uint8_t * base = static_cast<uint8_t *>(block->ptr);
    if (ptr >= base && ptr < base + block->size) {
      return block.get();
    }
  }
  return nullptr;
}

std::string IntelMemoryBufferPool::register_block_for_ipc(PoolBlock * block)
{
  if (transport_mode_ == TransportMode::LEVEL_ZERO_USM) {
    int fd = -1;
    if (!L0UsmIpc::export_fd(block, fd)) {
      return "";
    }
    return IntelMemoryIPCManager::register_fd(block->block_id, fd);
  }

  if (!dmabuf_capable_ || block->dmabuf_fd < 0) {
    return "";
  }

  return IntelMemoryIPCManager::register_block(block);
}

bool IntelMemoryBufferPool::probe_level_zero()
{
  ::setenv("NEOReadDebugKeys", "1", 0);
  ::setenv("EnableHostUsmAllocationPool", "0", 0);

  ze_result_t result = zeInit(0);
  if (result != ZE_RESULT_SUCCESS) {
    return false;
  }

  // Enumerate drivers
  uint32_t driver_count = 0;
  result = zeDriverGet(&driver_count, nullptr);
  if (result != ZE_RESULT_SUCCESS || driver_count == 0) {
    return false;
  }

  std::vector<ze_driver_handle_t> drivers(driver_count);
  result = zeDriverGet(&driver_count, drivers.data());
  if (result != ZE_RESULT_SUCCESS) {
    return false;
  }

  // Find an Intel GPU device
  for (auto driver : drivers) {
    uint32_t device_count = 0;
    zeDeviceGet(driver, &device_count, nullptr);
    if (device_count == 0) {
      continue;
    }

    std::vector<ze_device_handle_t> devices(device_count);
    zeDeviceGet(driver, &device_count, devices.data());

    for (uint32_t i = 0; i < device_count; ++i) {
      ze_device_properties_t props{};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      zeDeviceGetProperties(devices[i], &props);

      if (props.type == ZE_DEVICE_TYPE_GPU &&
        props.vendorId == 0x8086)  // Intel vendor ID
      {
        ze_driver_ = driver;
        ze_device_ = devices[i];
        ze_device_ordinal_ = static_cast<int32_t>(i);
        static_assert(sizeof(props.uuid.id) == kZeDeviceUuidBytes,
          "ze_device_uuid_t size mismatch");
        std::memcpy(ze_device_uuid_, props.uuid.id, kZeDeviceUuidBytes);

        // Verify GPU can access host memory (required for zero-copy IPC)
        ze_device_memory_access_properties_t mem_access{};
        mem_access.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
        result = zeDeviceGetMemoryAccessProperties(devices[i], &mem_access);
        if (result == ZE_RESULT_SUCCESS) {
          RCUTILS_LOG_INFO_NAMED("intel_memory_buffer_pool",
            "[L0 probe] Memory access: hostAlloc=0x%x, deviceAlloc=0x%x, "
            "sharedSingle=0x%x, sharedCross=0x%x",
            mem_access.hostAllocCapabilities,
            mem_access.deviceAllocCapabilities,
            mem_access.sharedSingleDeviceAllocCapabilities,
            mem_access.sharedCrossDeviceAllocCapabilities);

          if (!(mem_access.hostAllocCapabilities &
            ZE_MEMORY_ACCESS_CAP_FLAG_RW))
          {
            RCUTILS_LOG_WARN_NAMED("intel_memory_buffer_pool",
              "GPU cannot read/write host allocations — "
              "host USM IPC will not provide GPU access");
          }
        }

        // Check P2P access for multi-device configurations
        for (uint32_t j = 0; j < device_count; ++j) {
          if (j == i) continue;
          ze_bool_t can_access = 0;
          ze_result_t p2p_res = zeDeviceCanAccessPeer(
            devices[i], devices[j], &can_access);
          if (p2p_res == ZE_RESULT_SUCCESS) {
            ze_device_p2p_properties_t p2p_props{};
            p2p_props.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
            zeDeviceGetP2PProperties(devices[i], devices[j], &p2p_props);
            RCUTILS_LOG_INFO_NAMED("intel_memory_buffer_pool",
              "P2P device[%u]->device[%u]: access=%s, flags=0x%x",
              i, j, can_access ? "YES" : "NO", p2p_props.flags);
          }
        }

        // Create context for USM allocations
        ze_context_desc_t ctx_desc{};
        ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
        ze_context_handle_t ctx = nullptr;
        result = zeContextCreate(driver, &ctx_desc, &ctx);
        if (result != ZE_RESULT_SUCCESS) {
          return false;
        }
        ze_context_ = ctx;

        ze_host_mem_alloc_desc_t host_desc{};
        host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        host_desc.flags = 0;

        void * test_ptr = nullptr;
        result = zeMemAllocHost(ctx, &host_desc,
          alignment_, alignment_, &test_ptr);
        if (result != ZE_RESULT_SUCCESS || !test_ptr) {
          RCUTILS_LOG_ERROR_NAMED("intel_memory_buffer_pool",
            "zeMemAllocHost failed (result=0x%x)", result);
          zeContextDestroy(ctx);
          ze_context_ = nullptr;
          return false;
        }

        // Verify IPC handle can be obtained
        ze_ipc_mem_handle_t ipc_handle{};
        result = zeMemGetIpcHandle(ctx, test_ptr, &ipc_handle);
        zeMemFree(ctx, test_ptr);

        if (result != ZE_RESULT_SUCCESS) {
          RCUTILS_LOG_ERROR_NAMED("intel_memory_buffer_pool",
            "zeMemGetIpcHandle failed (result=0x%x) — IPC not available",
            result);
          zeContextDestroy(ctx);
          ze_context_ = nullptr;
          return false;
        }

        RCUTILS_LOG_INFO_NAMED("intel_memory_buffer_pool",
          "[L0 probe] Level Zero USM capability available "
          "(zeMemAllocHost + zeMemGetIpcHandle OK). Active transport is "
          "chosen below based on INTEL_MEMORY_BUFFER_TYPE.");
        return true;
      }
    }
  }

  return false;
}

bool IntelMemoryBufferPool::probe_dmabuf()
{
  int fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
  if (fd >= 0) {
    close(fd);
    return true;
  }
  RCUTILS_LOG_WARN_NAMED("intel_memory_buffer_pool",
    "DMA-BUF unavailable: open(/dev/dma_heap/system, O_RDWR) failed "
    "(errno=%d: %s). If the device exists but is not accessible, install the "
    "udev rule from the README (udev/60-intel-memory-dmabuf.rules) so the "
    "'render' group can allocate DMA-BUFs.",
    errno, std::strerror(errno));
  return false;
}

PoolBlock * IntelMemoryBufferPool::create_block(size_t aligned_size)
{
  auto block = std::make_unique<PoolBlock>();
  block->size = aligned_size;
  block->block_id = next_block_id_++;
  block->transport = transport_mode_;

  if (transport_mode_ == TransportMode::LEVEL_ZERO_USM) {
    ze_host_mem_alloc_desc_t host_desc{};
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    host_desc.flags = 0;

    ze_result_t result = zeMemAllocHost(
      static_cast<ze_context_handle_t>(ze_context_),
      &host_desc,
      aligned_size, alignment_,
      &block->ptr);
    if (result != ZE_RESULT_SUCCESS) {
      RCUTILS_LOG_ERROR_NAMED("intel_memory_buffer_pool",
        "zeMemAllocHost failed (size=%zu, result=0x%x)",
        aligned_size, result);
      return nullptr;
    }
    block->ze_context = ze_context_;
    block->ze_device_ordinal = ze_device_ordinal_;
    std::memcpy(block->ze_device_uuid, ze_device_uuid_, kZeDeviceUuidBytes);
  } else {
    int heap_fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
      RCUTILS_LOG_ERROR_NAMED("intel_memory_buffer_pool",
        "open(/dev/dma_heap/system) failed (errno=%d: %s); cannot allocate "
        "DMA-BUF block. Ensure the dma-heap is accessible — install the udev "
        "rule described in the README (udev/60-intel-memory-dmabuf.rules).",
        errno, std::strerror(errno));
      return nullptr;
    }

    struct dma_heap_allocation_data alloc_data{};
    alloc_data.len = aligned_size;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
      RCUTILS_LOG_ERROR_NAMED("intel_memory_buffer_pool",
        "DMA_HEAP_IOCTL_ALLOC failed (size=%zu, errno=%d: %s)",
        aligned_size, errno, std::strerror(errno));
      close(heap_fd);
      return nullptr;
    }
    block->dmabuf_fd = alloc_data.fd;

    close(heap_fd);
    block->dmabuf_heap_fd = -1;

    // mmap the fd for CPU access — same physical pages shared cross-process
    block->ptr = mmap(nullptr, aligned_size,
      PROT_READ | PROT_WRITE, MAP_SHARED,
      block->dmabuf_fd, 0);
    if (block->ptr == MAP_FAILED) {
      RCUTILS_LOG_ERROR_NAMED("intel_memory_buffer_pool",
        "mmap of DMA-BUF fd failed (size=%zu, errno=%d: %s)",
        aligned_size, errno, std::strerror(errno));
      close(block->dmabuf_fd);
      block->dmabuf_fd = -1;
      return nullptr;
    }
  }

  // Create shared IPC metadata segment
  std::string shm_name = "/intel_memory_buffer_" + std::to_string(getpid()) +
    "_" + std::to_string(block->block_id);
  shm_unlink(shm_name.c_str());
  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (shm_fd >= 0) {
    if (ftruncate(shm_fd, sizeof(IPCMetadata)) == 0) {
      void * meta_ptr = mmap(nullptr, sizeof(IPCMetadata),
        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
      if (meta_ptr != MAP_FAILED) {
        block->ipc_meta = new (meta_ptr) IPCMetadata();
        block->shm_fd = shm_fd;
        block->shm_name = shm_name;
      } else {
        close(shm_fd);
        shm_unlink(shm_name.c_str());
      }
    } else {
      close(shm_fd);
      shm_unlink(shm_name.c_str());
    }
  }

  PoolBlock * ptr = block.get();
  all_blocks_.push_back(std::move(block));
  return ptr;
}

void IntelMemoryBufferPool::destroy_block(PoolBlock * block)
{
  DmaBufIpc::unregister_block(block->block_id);

  if (block->ipc_meta) {
    munmap(block->ipc_meta, sizeof(IPCMetadata));
    block->ipc_meta = nullptr;
  }
  if (block->shm_fd >= 0) {
    close(block->shm_fd);
    shm_unlink(block->shm_name.c_str());
  }

  if (block->transport == TransportMode::LEVEL_ZERO_USM) {
    L0UsmIpc::put_handle(block);
    if (block->ptr && block->ze_context) {
      zeMemFree(
        static_cast<ze_context_handle_t>(block->ze_context),
        block->ptr);
    }
  } else {
    if (block->ptr && block->ptr != MAP_FAILED) {
      munmap(block->ptr, block->size);
    }
    if (block->dmabuf_fd >= 0) {
      close(block->dmabuf_fd);
    }
    if (block->dmabuf_heap_fd >= 0) {
      close(block->dmabuf_heap_fd);
    }
  }
  block->ptr = nullptr;
}

bool IntelMemoryBufferPool::is_block_ready(PoolBlock * block) const
{
  if (!block->ipc_meta) {
    return true;
  }

  for (int i = 0; i < kMaxImportHolders; ++i) {
    int32_t holder = block->ipc_meta->holder_pids[i].load(std::memory_order_acquire);
    if (holder != 0 && kill(holder, 0) != 0 && errno == ESRCH) {
      // CAS the slot to 0 so only one sweeper reclaims this dead holder's ref.
      if (block->ipc_meta->holder_pids[i].compare_exchange_strong(
          holder, 0, std::memory_order_acq_rel)) {
        block->ipc_meta->refcount.fetch_sub(1, std::memory_order_release);
        RCUTILS_LOG_WARN_NAMED("intel_memory_buffer_pool",
          "reclaimed leaked reference from dead consumer pid=%d (block %u)",
          holder, block->block_id);
      }
    }
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

  uint64_t now_us = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  if (now_us < publish_us) {
    return false;
  }
  return (now_us - publish_us) >= config_.grace_period_us;
}

size_t IntelMemoryBufferPool::align_size(size_t size) const
{
  return ((size + alignment_ - 1) / alignment_) * alignment_;
}

}  // namespace intel_memory_core
