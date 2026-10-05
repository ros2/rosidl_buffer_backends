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

#include "intel_memory_core/intel_memory_ipc_manager.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <csignal>

#include "intel_memory_core/logging.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/l0_usm_ipc.hpp"
#include "intel_memory_core/intel_memory_buffer_pool.hpp"

namespace intel_memory_core
{

std::string IntelMemoryIPCManager::register_block(PoolBlock * block)
{
  return DmaBufIpc::register_block(block);
}

std::string IntelMemoryIPCManager::register_fd(uint32_t block_id, int fd)
{
  return DmaBufIpc::register_fd(block_id, fd);
}

namespace
{
void register_holder_pid(IPCMetadata * meta)
{
  int32_t self = static_cast<int32_t>(getpid());
  for (int i = 0; i < kMaxImportHolders; ++i) {
    int32_t expected = 0;
    if (meta->holder_pids[i].compare_exchange_strong(
        expected, self, std::memory_order_acq_rel)) {
      return;
    }
  }
}

bool clear_holder_pid(IPCMetadata * meta)
{
  int32_t self = static_cast<int32_t>(getpid());
  for (int i = 0; i < kMaxImportHolders; ++i) {
    int32_t expected = self;
    if (meta->holder_pids[i].compare_exchange_strong(
        expected, 0, std::memory_order_acq_rel)) {
      return true;
    }
  }
  return false;
}

// One cached refcount mapping: the pointer plus whose it is.
struct CachedMeta
{
  IPCMetadata * meta{nullptr};
  int32_t pid{0};
};

std::unordered_map<uint64_t, CachedMeta> & meta_cache()
{
  thread_local std::unordered_map<uint64_t, CachedMeta> cache;
  return cache;
}

IPCMetadata * open_shared_meta(int32_t pid, uint32_t block_id, uint32_t pool_id)
{
  auto & cache = meta_cache();
  const uint64_t key = import_cache_key(pool_id, block_id);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second.meta;
  }

  for (auto e = cache.begin(); e != cache.end(); ) {
    if (!producer_process_alive(e->second.pid)) {
      munmap(e->second.meta, sizeof(IPCMetadata));
      e = cache.erase(e);
    } else {
      ++e;
    }
  }

  std::string shm_name = "/intel_memory_buffer_" + std::to_string(pid) +
    "_" + std::to_string(block_id);
  int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (shm_fd < 0) {
    return nullptr;
  }
  void * meta_ptr = mmap(nullptr, sizeof(IPCMetadata),
    PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  close(shm_fd);
  if (meta_ptr == MAP_FAILED) {
    return nullptr;
  }
  IPCMetadata * meta = static_cast<IPCMetadata *>(meta_ptr);
  cache[key] = CachedMeta{meta, pid};
  return meta;
}

// Drop this thread's cached refcount mappings for one producer instance.
void evict_meta_pool(uint32_t pool_id)
{
  auto & cache = meta_cache();
  for (auto it = cache.begin(); it != cache.end(); ) {
    if (static_cast<uint32_t>(it->first >> 32) == pool_id) {
      munmap(it->second.meta, sizeof(IPCMetadata));
      it = cache.erase(it);
    } else {
      ++it;
    }
  }
}

bool acquire_and_validate(IPCMetadata * meta, uint64_t expected_uid, uint32_t block_id)
{
  meta->refcount.fetch_add(1, std::memory_order_acquire);
  register_holder_pid(meta);

  if (expected_uid != 0) {
    uint64_t live_uid = meta->uid.load(std::memory_order_acquire);
    if (live_uid != expected_uid) {
      clear_holder_pid(meta);
      meta->refcount.fetch_sub(1, std::memory_order_release);
      RCUTILS_LOG_WARN_NAMED("intel_memory_ipc",
        "stale import rejected: expected uid=%lu, live uid=%lu (block %u)",
        static_cast<unsigned long>(expected_uid),
        static_cast<unsigned long>(live_uid), block_id);
      return false;
    }
  }
  return true;
}
}  // namespace

IntelMemoryIPCManager::ImportResult IntelMemoryIPCManager::import_dmabuf_block(
  const std::string & socket_path,
  int32_t pid,
  uint32_t block_id,
  uint64_t size,
  uint64_t expected_uid,
  uint32_t pool_id)
{
  IpcImportResult ipc_result;
  try {
    ipc_result = DmaBufIpc::import_block(socket_path, pid, block_id, size, pool_id);
  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_NAMED("intel_memory_ipc",
      "dmabuf import failed (producer gone or stale socket): %s", e.what());
    DmaBufIpc::evict_pool(pool_id);
    evict_meta_pool(pool_id);
    return ImportResult{nullptr, nullptr};
  }

  IPCMetadata * meta = open_shared_meta(pid, block_id, pool_id);
  if (!meta) {
    return ImportResult{ipc_result.ptr, nullptr, ipc_result.dmabuf_fd};
  }

  if (!acquire_and_validate(meta, expected_uid, block_id)) {
    return ImportResult{nullptr, meta};   // stale; reference already undone
  }
  return ImportResult{ipc_result.ptr, meta, ipc_result.dmabuf_fd};
}

void IntelMemoryIPCManager::release_block(IPCMetadata * meta)
{
  if (!meta) {
    return;
  }

  if (clear_holder_pid(meta)) {
    meta->refcount.fetch_sub(1, std::memory_order_release);
  }
}

IntelMemoryIPCManager::ImportResult IntelMemoryIPCManager::import_level_zero_block(
  const uint8_t * ipc_handle_data,
  size_t ipc_handle_size,
  int32_t device_ordinal,
  const uint8_t * device_uuid,
  uint64_t size,
  int32_t pid,
  uint32_t block_id,
  uint64_t expected_uid,
  const std::string & l0_socket_path,
  uint32_t pool_id)
{
  IpcImportResult ipc_result;
  try {
    ipc_result = L0UsmIpc::import_handle(
      ipc_handle_data, ipc_handle_size, device_ordinal, device_uuid,
      size, pid, block_id, l0_socket_path, pool_id);
  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_NAMED("intel_memory_ipc",
      "level-zero import failed (producer gone): %s", e.what());
    L0UsmIpc::evict_pool(pool_id);
    evict_meta_pool(pool_id);
    return ImportResult{nullptr, nullptr};
  }

  IPCMetadata * meta = open_shared_meta(pid, block_id, pool_id);
  if (!meta) {
    // best effort, no refcount
    return ImportResult{ipc_result.ptr, nullptr, ipc_result.dmabuf_fd};
  }
  if (!acquire_and_validate(meta, expected_uid, block_id)) {
    return ImportResult{nullptr, meta};
  }
  return ImportResult{ipc_result.ptr, meta, ipc_result.dmabuf_fd};
}

}  // namespace intel_memory_core
