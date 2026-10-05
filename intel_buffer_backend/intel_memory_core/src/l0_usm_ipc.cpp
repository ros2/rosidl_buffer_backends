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

#include "intel_memory_core/l0_usm_ipc.hpp"

#include <level_zero/ze_api.h>
#include <unistd.h>

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/logging.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace intel_memory_core
{

struct L0State {
  ze_driver_handle_t driver{nullptr};
  ze_context_handle_t ctx{nullptr};
  std::vector<ze_device_handle_t> devices;
  bool initialized{false};

  bool init()
  {
    if (initialized) return (ctx != nullptr);
    initialized = true;

    ze_result_t r = zeInit(0);
    if (r != ZE_RESULT_SUCCESS) return false;

    uint32_t drv_count = 0;
    zeDriverGet(&drv_count, nullptr);
    if (drv_count == 0) return false;
    std::vector<ze_driver_handle_t> drivers(drv_count);
    zeDriverGet(&drv_count, drivers.data());

    // Find Intel GPU driver (same selection as publisher's pool)
    for (auto drv : drivers) {
      uint32_t dev_count = 0;
      zeDeviceGet(drv, &dev_count, nullptr);
      if (dev_count == 0) continue;
      std::vector<ze_device_handle_t> devs(dev_count);
      zeDeviceGet(drv, &dev_count, devs.data());

      for (uint32_t i = 0; i < dev_count; ++i) {
        ze_device_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        zeDeviceGetProperties(devs[i], &props);
        if (props.type == ZE_DEVICE_TYPE_GPU && props.vendorId == 0x8086) {
          driver = drv;
          devices = devs;
          ze_context_desc_t ctx_desc{};
          ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
          r = zeContextCreate(drv, &ctx_desc, &ctx);
          return (r == ZE_RESULT_SUCCESS);
        }
      }
    }
    return false;
  }

  ~L0State()
  {
    if (ctx) {
      zeContextDestroy(ctx);
      ctx = nullptr;
    }
  }
};

bool L0UsmIpc::export_handle(
  PoolBlock * block,
  uint8_t * out_handle_data,
  uint64_t & out_handle_size,
  uint64_t & out_context_id)
{
  static_assert(sizeof(ze_ipc_mem_handle_t) <= kZeIpcHandleBytes,
    "ze_ipc_mem_handle_t exceeds PoolBlock cache size");

  if (!block->ze_ipc_handle_valid) {
    ze_ipc_mem_handle_t ipc_handle{};
    ze_result_t result = zeMemGetIpcHandle(
      static_cast<ze_context_handle_t>(block->ze_context),
      block->ptr,
      &ipc_handle);
    if (result != ZE_RESULT_SUCCESS) {
      RCUTILS_LOG_WARN_NAMED("l0_usm_ipc",
        "zeMemGetIpcHandle failed (result=0x%x)", result);
      return false;
    }
    std::memcpy(block->ze_ipc_handle, &ipc_handle, sizeof(ipc_handle));
    block->ze_ipc_handle_valid = true;
  }

  std::memcpy(out_handle_data, block->ze_ipc_handle, sizeof(ze_ipc_mem_handle_t));
  out_handle_size = sizeof(ze_ipc_mem_handle_t);
  out_context_id = reinterpret_cast<uint64_t>(block->ze_context);
  return true;
}

bool L0UsmIpc::export_fd(PoolBlock * block, int & out_fd)
{
  if (!block->ze_ipc_handle_valid || !block->ze_context) {
    return false;
  }
  ze_ipc_mem_handle_t ipc_handle{};
  std::memcpy(&ipc_handle, block->ze_ipc_handle, sizeof(ipc_handle));

  uint64_t fd = 0;
  ze_result_t result = zeMemGetFileDescriptorFromIpcHandleExp(
    static_cast<ze_context_handle_t>(block->ze_context), ipc_handle, &fd);
  if (result != ZE_RESULT_SUCCESS) {
    RCUTILS_LOG_WARN_NAMED("l0_usm_ipc",
      "zeMemGetFileDescriptorFromIpcHandleExp failed (result=0x%x); "
      "cross-process L0 fd passing unavailable", result);
    return false;
  }
  out_fd = static_cast<int>(fd);
  return true;
}

void L0UsmIpc::put_handle(PoolBlock * block)
{
  if (!block->ze_ipc_handle_valid || !block->ze_context) {
    return;
  }
  ze_ipc_mem_handle_t ipc_handle{};
  std::memcpy(&ipc_handle, block->ze_ipc_handle, sizeof(ipc_handle));
  ze_result_t result = zeMemPutIpcHandle(
    static_cast<ze_context_handle_t>(block->ze_context), ipc_handle);
  if (result != ZE_RESULT_SUCCESS) {
    RCUTILS_LOG_WARN_NAMED("l0_usm_ipc",
      "zeMemPutIpcHandle failed (result=0x%x)", result);
  }
  block->ze_ipc_handle_valid = false;
}

namespace
{
// One L0 state per importing thread (see L0State above).
L0State & l0_state()
{
  thread_local L0State state;
  return state;
}

// One cached import: the USM pointer plus whose it is, for the dead-producer reap.
struct CachedL0Import
{
  IpcImportResult result;
  int32_t pid{0};
};

std::unordered_map<uint64_t, CachedL0Import> & l0_import_cache()
{
  thread_local std::unordered_map<uint64_t, CachedL0Import> cache;
  return cache;
}

void drop_l0_import(const CachedL0Import & entry)
{
  L0State & l0 = l0_state();
  if (entry.result.ptr && l0.ctx) {
    zeMemFree(l0.ctx, entry.result.ptr);
  }
  if (entry.result.dmabuf_fd >= 0) {
    close(entry.result.dmabuf_fd);
  }
}
}  // namespace

void L0UsmIpc::evict_pool(uint32_t pool_id)
{
  auto & cache = l0_import_cache();
  for (auto it = cache.begin(); it != cache.end(); ) {
    if (static_cast<uint32_t>(it->first >> 32) == pool_id) {
      drop_l0_import(it->second);
      it = cache.erase(it);
    } else {
      ++it;
    }
  }
}

IpcImportResult L0UsmIpc::import_handle(
  const uint8_t * handle_data,
  size_t handle_size,
  int32_t device_ordinal,
  const uint8_t * device_uuid,
  uint64_t size,
  int32_t pid,
  uint32_t block_id,
  const std::string & l0_socket_path,
  uint32_t pool_id)
{
  auto & cache = l0_import_cache();
  const uint64_t cache_key = import_cache_key(pool_id, block_id);
  auto cache_it = cache.find(cache_key);
  if (cache_it != cache.end()) {
    return cache_it->second.result;
  }

  try {

  L0State & l0 = l0_state();
  if (!l0.init()) {
    throw std::runtime_error("Level Zero initialization failed on import side");
  }

  for (auto e = cache.begin(); e != cache.end(); ) {
    if (!producer_process_alive(e->second.pid)) {
      drop_l0_import(e->second);
      e = cache.erase(e);
    } else {
      ++e;
    }
  }

  uint32_t selected = 0;
  bool selected_by_uuid = false;
  if (device_uuid != nullptr) {
    bool uuid_set = false;
    for (int b = 0; b < kZeDeviceUuidBytes; ++b) {
      if (device_uuid[b] != 0) { uuid_set = true; break; }
    }
    if (uuid_set) {
      for (uint32_t d = 0; d < l0.devices.size(); ++d) {
        ze_device_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        zeDeviceGetProperties(l0.devices[d], &props);
        if (std::memcmp(props.uuid.id, device_uuid, kZeDeviceUuidBytes) == 0) {
          selected = d;
          selected_by_uuid = true;
          break;
        }
      }
      if (!selected_by_uuid) {
        RCUTILS_LOG_WARN_NAMED("l0_usm_ipc",
          "producer device UUID not found among %zu local devices; "
          "falling back to ordinal %d", l0.devices.size(), device_ordinal);
      }
    }
  }

  if (!selected_by_uuid) {
    if (device_ordinal < 0 ||
      static_cast<uint32_t>(device_ordinal) >= l0.devices.size())
    {
      throw std::runtime_error("Invalid device ordinal for L0 import");
    }
    selected = static_cast<uint32_t>(device_ordinal);
  }

  void * ptr = nullptr;
  int recv_fd = -1;
  if (!l0_socket_path.empty()) {
    recv_fd = DmaBufIpc::receive_fd(l0_socket_path);

    int keep_fd = dup(recv_fd);

    ze_external_memory_import_fd_t import_fd{};
    import_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    import_fd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    import_fd.fd = recv_fd;

    ze_host_mem_alloc_desc_t host_desc{};
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    host_desc.pNext = &import_fd;

    ze_result_t r = zeMemAllocHost(l0.ctx, &host_desc, size, 0, &ptr);
    if (r != ZE_RESULT_SUCCESS) {
      close(recv_fd);
      if (keep_fd >= 0) {
        close(keep_fd);
      }
      RCUTILS_LOG_ERROR_NAMED("l0_usm_ipc",
        "zeMemAllocHost(import dma_buf fd) failed (result=0x%x, size=%lu)",
        r, size);
      throw std::runtime_error(
        "zeMemAllocHost(import dma_buf fd) failed (result=0x" +
        std::to_string(static_cast<unsigned>(r)) + ")");
    }
    recv_fd = keep_fd;   // the one the driver did not take
    RCUTILS_LOG_INFO_NAMED("l0_usm_ipc",
      "L0 dma_buf-fd import succeeded: ptr=%p fd=%d (host USM, CPU+GPU accessible)",
      ptr, recv_fd);
  } else {
    ze_ipc_mem_handle_t ipc_handle{};
    if (handle_size > sizeof(ipc_handle)) {
      handle_size = sizeof(ipc_handle);
    }
    std::memcpy(&ipc_handle, handle_data, handle_size);

    ze_result_t result = zeMemOpenIpcHandle(
      l0.ctx, l0.devices[selected],
      ipc_handle, 0, &ptr);
    if (result != ZE_RESULT_SUCCESS) {
      RCUTILS_LOG_ERROR_NAMED("l0_usm_ipc",
        "zeMemOpenIpcHandle failed (result=0x%x, device=%u, by_uuid=%d, size=%lu)",
        result, selected, selected_by_uuid, size);
      throw std::runtime_error(
        "zeMemOpenIpcHandle failed (result=0x" +
        std::to_string(static_cast<unsigned>(result)) + ")");
    }
    RCUTILS_LOG_INFO_NAMED("l0_usm_ipc",
      "L0 IPC import succeeded: ptr=%p (host USM, CPU+GPU accessible)", ptr);
  }

  IpcImportResult import_result{ptr, recv_fd};
  cache[cache_key] = CachedL0Import{import_result, pid};
  return import_result;

  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_NAMED("l0_usm_ipc",
      "level-zero import failed (producer gone or invalid handle): %s", e.what());
    return IpcImportResult{};
  }
}

}  // namespace intel_memory_core
