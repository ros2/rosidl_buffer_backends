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

#include "intel_memory_core/l0_external_memmap.hpp"

#include <level_zero/ze_api.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/logging.hpp"

#include <cerrno>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace intel_memory_core
{

namespace
{

struct MemMapL0State {
  ze_driver_handle_t driver{nullptr};
  ze_context_handle_t ctx{nullptr};
  bool initialized{false};

  bool init()
  {
    if (initialized) {return ctx != nullptr;}
    initialized = true;

    if (zeInit(0) != ZE_RESULT_SUCCESS) {return false;}

    uint32_t drv_count = 0;
    zeDriverGet(&drv_count, nullptr);
    if (drv_count == 0) {return false;}
    std::vector<ze_driver_handle_t> drivers(drv_count);
    zeDriverGet(&drv_count, drivers.data());

    for (auto drv : drivers) {
      uint32_t dev_count = 0;
      zeDeviceGet(drv, &dev_count, nullptr);
      if (dev_count == 0) {continue;}
      std::vector<ze_device_handle_t> devs(dev_count);
      zeDeviceGet(drv, &dev_count, devs.data());

      for (uint32_t i = 0; i < dev_count; ++i) {
        ze_device_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        zeDeviceGetProperties(devs[i], &props);
        if (props.type == ZE_DEVICE_TYPE_GPU && props.vendorId == 0x8086) {
          driver = drv;
          ze_context_desc_t ctx_desc{};
          ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
          return zeContextCreate(drv, &ctx_desc, &ctx) == ZE_RESULT_SUCCESS;
        }
      }
    }
    return false;
  }

  ~MemMapL0State()
  {
    if (ctx) {
      zeContextDestroy(ctx);
      ctx = nullptr;
    }
  }
};

MemMapL0State g_l0;
std::mutex g_l0_mutex;

MemMapL0State * l0_state()
{
  std::lock_guard<std::mutex> lock(g_l0_mutex);
  return g_l0.init() ? &g_l0 : nullptr;
}

size_t page_size()
{
  long ps = ::sysconf(_SC_PAGESIZE);
  return ps > 0 ? static_cast<size_t>(ps) : 4096;
}

}  // namespace

bool L0ExternalMemMap::supported()
{
  MemMapL0State * l0 = l0_state();
  if (!l0) {return false;}

  uint32_t count = 0;
  if (zeDriverGetExtensionProperties(l0->driver, &count, nullptr) !=
    ZE_RESULT_SUCCESS || count == 0)
  {
    return false;
  }
  std::vector<ze_driver_extension_properties_t> exts(count);
  if (zeDriverGetExtensionProperties(l0->driver, &count, exts.data()) !=
    ZE_RESULT_SUCCESS)
  {
    return false;
  }
  for (const auto & e : exts) {
    if (std::strcmp(e.name, ZE_EXTERNAL_MEMORY_MAPPING_EXT_NAME) == 0) {
      return true;
    }
  }
  return false;
}

void * L0ExternalMemMap::map(void * ptr, size_t size, bool read_only)
{
  if (ptr == nullptr || size == 0) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "map: null pointer or zero size");
    return nullptr;
  }

  const size_t ps = page_size();
  if ((reinterpret_cast<uintptr_t>(ptr) % ps) != 0) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "map: pointer %p is not page-aligned (page size %zu)", ptr, ps);
    return nullptr;
  }
  if ((size % ps) != 0) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "map: size %zu is not a multiple of page size %zu", size, ps);
    return nullptr;
  }

  MemMapL0State * l0 = l0_state();
  if (!l0) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "map: Level Zero initialization failed");
    return nullptr;
  }

  ze_external_memmap_sysmem_ext_desc_t memmap_desc{};
  memmap_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC;
  memmap_desc.pSystemMemory = ptr;
  memmap_desc.size = size;

  ze_host_mem_alloc_desc_t host_desc{};
  host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  host_desc.pNext = &memmap_desc;
  if (read_only) {
    host_desc.flags = ZE_HOST_MEM_ALLOC_FLAG_MEM_READ_ONLY;
  }

  void * usm_ptr = nullptr;
  ze_result_t r = zeMemAllocHost(l0->ctx, &host_desc, size, ps, &usm_ptr);
  if (r != ZE_RESULT_SUCCESS) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "map: zeMemAllocHost(external-memmap) failed (result=0x%x, ptr=%p, "
      "size=%zu)", r, ptr, size);
    return nullptr;
  }

  if (usm_ptr != ptr) {
    RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
      "map: USM pointer %p differs from input %p (extension expects equality)",
      usm_ptr, ptr);
  }
  RCUTILS_LOG_INFO_NAMED("l0_external_memmap",
    "map: mapped app buffer as USM host (ptr=%p size=%zu%s)",
    usm_ptr, size, read_only ? ", read-only" : "");
  return usm_ptr;
}

void L0ExternalMemMap::unmap(void * usm_ptr)
{
  if (usm_ptr == nullptr) {return;}
  MemMapL0State * l0 = l0_state();
  if (!l0) {return;}
  ze_result_t r = zeMemFree(l0->ctx, usm_ptr);
  if (r != ZE_RESULT_SUCCESS) {
    RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
      "unmap: zeMemFree failed (result=0x%x, ptr=%p)", r, usm_ptr);
  }
}

// --- SharedSysmem -----------------------------------------------------------

SharedSysmem::~SharedSysmem()
{
  close();
}

SharedSysmem::SharedSysmem(SharedSysmem && other) noexcept
: ptr_(other.ptr_), size_(other.size_),
  name_(std::move(other.name_)), owner_(other.owner_)
{
  other.ptr_ = nullptr;
  other.size_ = 0;
  other.owner_ = false;
}

SharedSysmem & SharedSysmem::operator=(SharedSysmem && other) noexcept
{
  if (this != &other) {
    close();
    ptr_ = other.ptr_;
    size_ = other.size_;
    name_ = std::move(other.name_);
    owner_ = other.owner_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.owner_ = false;
  }
  return *this;
}

bool SharedSysmem::create(const std::string & name, size_t size)
{
  if (ptr_ != nullptr) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "SharedSysmem::create: already open (%s)", name_.c_str());
    return false;
  }
  if (size == 0) {return false;}

  const size_t ps = page_size();
  const size_t aligned = ((size + ps - 1) / ps) * ps;

  int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
  if (fd < 0) {
    RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
      "SharedSysmem::create: shm_open(%s) failed (errno=%d: %s)",
      name.c_str(), errno, std::strerror(errno));
    return false;
  }
  if (ftruncate(fd, static_cast<off_t>(aligned)) != 0) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "SharedSysmem::create: ftruncate(%zu) failed (errno=%d: %s)",
      aligned, errno, std::strerror(errno));
    ::close(fd);
    shm_unlink(name.c_str());
    return false;
  }

  void * p = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "SharedSysmem::create: mmap(%zu) failed (errno=%d: %s)",
      aligned, errno, std::strerror(errno));
    shm_unlink(name.c_str());
    return false;
  }

  ptr_ = p;
  size_ = aligned;
  name_ = name;
  owner_ = true;
  return true;
}

bool SharedSysmem::open(const std::string & name)
{
  if (ptr_ != nullptr) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "SharedSysmem::open: already open (%s)", name_.c_str());
    return false;
  }

  int fd = shm_open(name.c_str(), O_RDWR, 0666);
  if (fd < 0) {
    RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
      "SharedSysmem::open: shm_open(%s) failed (errno=%d: %s)",
      name.c_str(), errno, std::strerror(errno));
    return false;
  }

  // The peer does not know the size a priori; read it back from the segment.
  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
      "SharedSysmem::open: fstat(%s) failed or empty (errno=%d: %s)",
      name.c_str(), errno, std::strerror(errno));
    ::close(fd);
    return false;
  }
  const size_t sz = static_cast<size_t>(st.st_size);

  void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) {
    RCUTILS_LOG_ERROR_NAMED("l0_external_memmap",
      "SharedSysmem::open: mmap(%zu) failed (errno=%d: %s)",
      sz, errno, std::strerror(errno));
    return false;
  }

  ptr_ = p;
  size_ = sz;
  name_ = name;
  owner_ = false;
  return true;
}

void SharedSysmem::close()
{
  if (ptr_ != nullptr) {
    if (mmap(ptr_, size_, PROT_NONE,
      MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0) == MAP_FAILED)
    {
      RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
        "SharedSysmem::close: failed to reserve freed VA range %p "
        "(errno=%d: %s); a later L0 mapping may reuse it and crash at exit",
        ptr_, errno, std::strerror(errno));
    }
    ptr_ = nullptr;
  }
  if (owner_ && !name_.empty()) {
    shm_unlink(name_.c_str());
  }
  size_ = 0;
  name_.clear();
  owner_ = false;
}

// --- ExternalMemMapImporter -------------------------------------------------

namespace
{

struct ImportedRegion
{
  SharedSysmem shm;
  int32_t pid{0};
};

std::mutex g_import_mutex;
std::map<std::string, std::unique_ptr<ImportedRegion>> g_imports;

}  // namespace

void * ExternalMemMapImporter::import(
  const std::string & shm_name, size_t size, int32_t publisher_pid,
  size_t * out_region_size)
{
  if (shm_name.empty() || size == 0) {
    RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
      "import: empty shm name or zero size");
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(g_import_mutex);

  auto it = g_imports.find(shm_name);
  if (it != g_imports.end()) {
    if (size > it->second->shm.size()) {
      RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
        "import: payload %zu exceeds cached region %s (%zu bytes)",
        size, shm_name.c_str(), it->second->shm.size());
      return nullptr;
    }
    if (out_region_size) {*out_region_size = it->second->shm.size();}
    return it->second->shm.ptr();
  }

  for (auto e = g_imports.begin(); e != g_imports.end(); ) {
    if (!producer_process_alive(e->second->pid)) {
      e = g_imports.erase(e);
    } else {
      ++e;
    }
  }

  auto entry = std::make_unique<ImportedRegion>();
  entry->pid = publisher_pid;
  if (!entry->shm.open(shm_name)) {
    return nullptr;  // open() logs the reason
  }
  if (size > entry->shm.size()) {
    // Same rationale as the cache-hit check above -- untrusted wire size.
    RCUTILS_LOG_WARN_NAMED("l0_external_memmap",
      "import: payload %zu exceeds region %s (%zu bytes)",
      size, shm_name.c_str(), entry->shm.size());
    return nullptr;
  }

  void * ptr = entry->shm.ptr();
  size_t region_size = entry->shm.size();
  RCUTILS_LOG_INFO_NAMED("l0_external_memmap",
    "import: opened producer region %s as raw host pointer (ptr=%p size=%zu)",
    shm_name.c_str(), ptr, region_size);
  g_imports.emplace(shm_name, std::move(entry));
  if (out_region_size) {*out_region_size = region_size;}
  return ptr;
}

bool ExternalMemMapImporter::generation_unchanged(
  const void * region, size_t region_size, uint32_t published_generation)
{
  auto * gen = external_memmap_generation_ptr(const_cast<void *>(region), region_size);
  if (!gen) {
    return true;
  }
  return gen->load(std::memory_order_acquire) == published_generation;
}

}  // namespace intel_memory_core
