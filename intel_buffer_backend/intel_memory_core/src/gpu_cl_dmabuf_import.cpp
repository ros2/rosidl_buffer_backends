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

#include "intel_memory_core/gpu_cl_dmabuf_import.hpp"

#include <mutex>
#include <unordered_map>
#include <utility>

#include "intel_memory_core/logging.hpp"

namespace intel_memory_core
{

#ifdef INTEL_MEMORY_CORE_HAVE_OPENCL

}  // namespace intel_memory_core

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

namespace intel_memory_core
{

bool cl_dmabuf_import_available() { return true; }

void * import_dmabuf_as_cl_mem(void * context_handle, int fd, uint64_t size)
{
  if (!context_handle || fd < 0 || size == 0) {
    return nullptr;
  }
  cl_context ctx = static_cast<cl_context>(context_handle);

  cl_mem_properties props[] = {
    static_cast<cl_mem_properties>(CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR),
    static_cast<cl_mem_properties>(fd),
    0
  };

  cl_int err = CL_SUCCESS;
  cl_mem buffer = clCreateBufferWithProperties(
    ctx, props, CL_MEM_READ_ONLY, static_cast<size_t>(size), nullptr, &err);
  if (err != CL_SUCCESS || !buffer) {
    RCUTILS_LOG_WARN_NAMED("gpu_cl_dmabuf_import",
      "clCreateBufferWithProperties(dma_buf fd=%d, size=%llu) failed (cl_int=%d)",
      fd, static_cast<unsigned long long>(size), static_cast<int>(err));
    return nullptr;
  }
  return static_cast<void *>(buffer);
}

void release_cl_mem(void * mem)
{
  if (mem) {
    clReleaseMemObject(static_cast<cl_mem>(mem));
  }
}

namespace
{
std::mutex g_cl_mem_cache_mutex;
std::unordered_map<uint64_t, void *> g_cl_mem_cache;

uint64_t cache_key(void * ctx, int fd)
{
  return (reinterpret_cast<uint64_t>(ctx) << 20) ^ static_cast<uint64_t>(fd);
}

uint64_t host_ptr_cache_key(void * ctx, void * host_ptr)
{
  return ((reinterpret_cast<uint64_t>(ctx) << 20) ^
         reinterpret_cast<uint64_t>(host_ptr)) ^ 0x484f5354'50545200ull;  // "HOSTPTR"
}

std::mutex g_queue_cache_mutex;
std::unordered_map<void *, cl_command_queue> g_queue_cache;

cl_command_queue get_or_create_sync_queue(cl_context ctx)
{
  std::lock_guard<std::mutex> lock(g_queue_cache_mutex);
  auto it = g_queue_cache.find(ctx);
  if (it != g_queue_cache.end()) {
    return it->second;
  }
  cl_device_id device = nullptr;
  cl_int err = clGetContextInfo(
    ctx, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr);
  if (err != CL_SUCCESS || !device) {
    RCUTILS_LOG_WARN_NAMED("gpu_cl_dmabuf_import",
      "sync_host_ptr_cl_mem: clGetContextInfo(CL_CONTEXT_DEVICES) failed (cl_int=%d)",
      static_cast<int>(err));
    return nullptr;
  }
  cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);
  if (err != CL_SUCCESS || !queue) {
    RCUTILS_LOG_WARN_NAMED("gpu_cl_dmabuf_import",
      "sync_host_ptr_cl_mem: clCreateCommandQueueWithProperties failed (cl_int=%d)",
      static_cast<int>(err));
    return nullptr;
  }
  g_queue_cache[ctx] = queue;
  return queue;
}
}  // namespace

bool sync_host_ptr_cl_mem(void * context_handle, void * mem, uint64_t size)
{
  if (!context_handle || !mem || size == 0) {
    return false;
  }
  cl_command_queue queue = get_or_create_sync_queue(static_cast<cl_context>(context_handle));
  if (!queue) {
    return false;
  }
  cl_int err = CL_SUCCESS;
  void * mapped = clEnqueueMapBuffer(
    queue, static_cast<cl_mem>(mem), CL_TRUE, CL_MAP_READ, 0,
    static_cast<size_t>(size), 0, nullptr, nullptr, &err);
  if (err != CL_SUCCESS || !mapped) {
    RCUTILS_LOG_WARN_NAMED("gpu_cl_dmabuf_import",
      "sync_host_ptr_cl_mem: clEnqueueMapBuffer failed (cl_int=%d)", static_cast<int>(err));
    return false;
  }
  err = clEnqueueUnmapMemObject(
    queue, static_cast<cl_mem>(mem), mapped, 0, nullptr, nullptr);
  if (err == CL_SUCCESS) {
    err = clFinish(queue);
  }
  if (err != CL_SUCCESS) {
    RCUTILS_LOG_WARN_NAMED("gpu_cl_dmabuf_import",
      "sync_host_ptr_cl_mem: unmap/finish failed (cl_int=%d)", static_cast<int>(err));
    return false;
  }
  return true;
}

void * get_or_import_cl_mem(void * cl_context, int fd, uint64_t size)
{
  if (!cl_context || fd < 0 || size == 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(g_cl_mem_cache_mutex);
  uint64_t key = cache_key(cl_context, fd);
  auto it = g_cl_mem_cache.find(key);
  if (it != g_cl_mem_cache.end()) {
    return it->second;
  }
  void * mem = import_dmabuf_as_cl_mem(cl_context, fd, size);
  if (mem) {
    g_cl_mem_cache[key] = mem;
  }
  return mem;
}

void * get_or_wrap_host_ptr_cl_mem(void * cl_context, void * host_ptr, uint64_t size)
{
  if (!cl_context || !host_ptr || size == 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(g_cl_mem_cache_mutex);
  uint64_t key = host_ptr_cache_key(cl_context, host_ptr);
  auto it = g_cl_mem_cache.find(key);
  if (it != g_cl_mem_cache.end()) {
    return it->second;
  }

  cl_int err = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(
    static_cast<::cl_context>(cl_context),
    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
    static_cast<size_t>(size), host_ptr, &err);
  if (err != CL_SUCCESS || !buffer) {
    RCUTILS_LOG_WARN_NAMED("gpu_cl_dmabuf_import",
      "clCreateBuffer(USE_HOST_PTR, ptr=%p, size=%llu) failed (cl_int=%d)",
      host_ptr, static_cast<unsigned long long>(size), static_cast<int>(err));
    return nullptr;   // not cached — a later call may retry
  }
  g_cl_mem_cache[key] = static_cast<void *>(buffer);
  return static_cast<void *>(buffer);
}

void release_all_cl_mem()
{
  std::lock_guard<std::mutex> lock(g_cl_mem_cache_mutex);
  for (auto & kv : g_cl_mem_cache) {
    clReleaseMemObject(static_cast<cl_mem>(kv.second));
  }
  g_cl_mem_cache.clear();
}

#else  // !INTEL_MEMORY_CORE_HAVE_OPENCL

bool cl_dmabuf_import_available() { return false; }

void * import_dmabuf_as_cl_mem(void * /*cl_context*/, int /*fd*/, uint64_t /*size*/)
{
  return nullptr;
}

void release_cl_mem(void * /*mem*/) {}

void * get_or_import_cl_mem(void * /*cl_context*/, int /*fd*/, uint64_t /*size*/)
{
  return nullptr;
}

void * get_or_wrap_host_ptr_cl_mem(
  void * /*cl_context*/, void * /*host_ptr*/, uint64_t /*size*/)
{
  return nullptr;
}

bool sync_host_ptr_cl_mem(void * /*cl_context*/, void * /*mem*/, uint64_t /*size*/)
{
  return false;
}

void release_all_cl_mem() {}

#endif  // INTEL_MEMORY_CORE_HAVE_OPENCL

}  // namespace intel_memory_core
