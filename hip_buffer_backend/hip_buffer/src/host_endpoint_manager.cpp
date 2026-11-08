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

#include "hip_buffer/host_endpoint_manager.hpp"

#include <fcntl.h>
#include <semaphore.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>
#include <rcutils/logging_macros.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace hip_host_endpoint_manager
{

static std::optional<int64_t> parse_override_int_env(
  const char * name, int64_t min, int64_t max)
{
  const char * value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  char * end = nullptr;
  errno = 0;
  const int64_t parsed = std::strtoll(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' ||
    parsed < min || parsed > max)
  {
    RCUTILS_LOG_WARN_NAMED(
      "hip_host_endpoint_manager",
      "Invalid value for %s='%s' (expected integer in [%s, %s]); ignoring",
      name, value, std::to_string(min).c_str(), std::to_string(max).c_str());
    return std::nullopt;
  }
  return parsed;
}

static bool sem_wait_timed(sem_t * sem, int timeout_sec = 2)
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += timeout_sec;
  if (sem_timedwait(sem, &ts) != 0) {
    RCUTILS_LOG_ERROR_NAMED(
      "hip_host_endpoint_manager", "sem_timedwait timed out after %ds (errno=%d)",
      timeout_sec, errno);
    return false;
  }
  return true;
}

constexpr size_t MAX_ENDPOINTS = 512;

struct EndpointEntry
{
  uint64_t instance_id;
  uint8_t gid[RMW_GID_STORAGE_SIZE];
  EntityType entity_type;
  int32_t device_id;
  uint32_t uid;
  bool active;
  bool ipc_capable;
  uint8_t padding[2];
  char topic_or_service_name[256];
};

struct EndpointRegistry
{
  uint32_t version;
  char hostname[256];
  uint32_t max_entries;
  uint32_t num_entries;
  EndpointEntry entries[MAX_ENDPOINTS];
};

static_assert(
  sizeof(EndpointEntry) % 8 == 0,
  "EndpointEntry must be 8-byte aligned for efficient array access");
static_assert(
  alignof(EndpointEntry) >= 8,
  "EndpointEntry must have at least 8-byte alignment");

static size_t purge_dead_entries(EndpointRegistry * registry)
{
  size_t purged = 0;
  for (size_t i = 0; i < registry->num_entries; ++i) {
    if (!registry->entries[i].active) {
      continue;
    }
    pid_t pid = static_cast<pid_t>(registry->entries[i].instance_id >> 32);
    if (pid > 0 && kill(pid, 0) != 0 && errno == ESRCH) {
      registry->entries[i].active = false;
      ++purged;
    }
  }
  return purged;
}

static bool has_active_entries(const EndpointRegistry * registry)
{
  for (size_t i = 0; i < registry->num_entries; ++i) {
    if (registry->entries[i].active) {
      return true;
    }
  }
  return false;
}

std::mutex HostEndpointManager::instances_mutex_;
std::unordered_map<size_t, std::shared_ptr<HostEndpointManager>>
HostEndpointManager::instances_;

std::shared_ptr<HostEndpointManager>
HostEndpointManager::get_instance(size_t domain_id)
{
  std::lock_guard<std::mutex> lock(instances_mutex_);

  auto it = instances_.find(domain_id);
  if (it != instances_.end()) {
    return it->second;
  }

  auto instance = std::shared_ptr<HostEndpointManager>(
    new HostEndpointManager(domain_id)
  );

  instances_[domain_id] = instance;

  return instance;
}

HostEndpointManager::HostEndpointManager(size_t domain_id)
: domain_id_(domain_id),
  instance_id_(0),
  shm_fd_(-1),
  shm_ptr_(nullptr),
  shm_size_(0),
  shm_mutex_(nullptr),
  local_device_id_(-1),
  local_uid_(0)
{
  static std::atomic<uint64_t> next_id{1};
  instance_id_ =
    (static_cast<uint64_t>(getpid()) << 32) |
    next_id.fetch_add(1, std::memory_order_relaxed);

  int device = -1;
  hipError_t err = hipGetDevice(&device);
  if (err == hipSuccess) {
    local_device_id_ = device;
  }

  // Env-var names for testing IPC fallback paths without requiring multiple
  // GPUs or users.
  // Not intended for production configuration — use only from test launches.
  constexpr const char * kDeviceIdOverrideEnv = "HIP_BUFFER_DEVICE_ID_OVERRIDE";
  constexpr const char * kUidOverrideEnv = "HIP_BUFFER_UID_OVERRIDE";

  if (auto v = parse_override_int_env(
      kDeviceIdOverrideEnv, 0, std::numeric_limits<int>::max()))
  {
    local_device_id_ = static_cast<int>(*v);
  }

  local_uid_ = static_cast<uint32_t>(getuid());

  if (auto v = parse_override_int_env(
      kUidOverrideEnv, 0, std::numeric_limits<uint32_t>::max()))
  {
    local_uid_ = static_cast<uint32_t>(*v);
  }

  char hostname_buf[256];
  if (gethostname(hostname_buf, sizeof(hostname_buf)) != 0) {
    throw std::runtime_error("Failed to get hostname");
  }
  hostname_ = hostname_buf;

  uint32_t host_hash = hash_string(hostname_);
  shm_name_ = "/ros2_hip_hem_d" + std::to_string(domain_id) +
    "_h" + std::to_string(host_hash);

  shm_size_ = sizeof(EndpointRegistry);

  shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
  if (shm_fd_ < 0) {
    throw std::runtime_error(
            "Failed to open shared memory: " + std::string(strerror(errno)));
  }

  if (ftruncate(shm_fd_, shm_size_) != 0) {
    close(shm_fd_);
    throw std::runtime_error(
            "Failed to resize shared memory: " + std::string(strerror(errno)));
  }

  shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  if (shm_ptr_ == MAP_FAILED) {
    close(shm_fd_);
    throw std::runtime_error(
            "Failed to map shared memory: " + std::string(strerror(errno)));
  }

  std::string mutex_name = shm_name_ + "_mtx";
  shm_mutex_ = sem_open(mutex_name.c_str(), O_CREAT, 0666, 1);
  if (shm_mutex_ == SEM_FAILED) {
    munmap(shm_ptr_, shm_size_);
    close(shm_fd_);
    throw std::runtime_error(
            "Failed to open mutex: " + std::string(strerror(errno)));
  }

  auto * registry = static_cast<EndpointRegistry *>(shm_ptr_);

  sem_t * sem = static_cast<sem_t *>(shm_mutex_);
  if (!sem_wait_timed(sem)) {
    throw std::runtime_error("Failed to acquire shared memory mutex during initialization");
  }

  if (registry->version == 0) {
    registry->version = 1;
    strncpy(registry->hostname, hostname_.c_str(), 255);
    registry->hostname[255] = '\0';
    registry->max_entries = MAX_ENDPOINTS;
    registry->num_entries = 0;

    for (size_t i = 0; i < MAX_ENDPOINTS; ++i) {
      registry->entries[i].active = false;
    }

    RCUTILS_LOG_INFO_NAMED(
      "hip_host_endpoint_manager",
      "Initialized shared memory registry for domain %zu on host %s",
      domain_id, hostname_.c_str());
  } else {
    size_t purged = purge_dead_entries(registry);
    if (purged > 0) {
      RCUTILS_LOG_INFO_NAMED(
        "hip_host_endpoint_manager",
        "Purged %zu stale entries from dead processes in domain %zu",
        purged, domain_id);
    }
    RCUTILS_LOG_INFO_NAMED(
      "hip_host_endpoint_manager",
      "Attached to existing shared memory registry for domain %zu", domain_id);
  }

  sem_post(sem);
}

HostEndpointManager::~HostEndpointManager()
{
  bool should_unlink = false;

  if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
    auto * registry = static_cast<EndpointRegistry *>(shm_ptr_);
    sem_t * sem = static_cast<sem_t *>(shm_mutex_);

    if (sem != nullptr && sem != SEM_FAILED) {
      if (!sem_wait_timed(sem)) {
        sem_close(sem);
        munmap(shm_ptr_, shm_size_);
        if (shm_fd_ >= 0) {close(shm_fd_);}
        return;
      }

      for (size_t i = 0; i < registry->num_entries; ++i) {
        if (registry->entries[i].active &&
          registry->entries[i].instance_id == instance_id_)
        {
          registry->entries[i].active = false;
        }
      }
      purge_dead_entries(registry);
      should_unlink = !has_active_entries(registry);

      sem_post(sem);
      sem_close(sem);
    }

    munmap(shm_ptr_, shm_size_);
  }

  if (shm_fd_ >= 0) {
    close(shm_fd_);
  }

  if (should_unlink) {
    shm_unlink(shm_name_.c_str());
    sem_unlink((shm_name_ + "_mtx").c_str());
  }
}

bool HostEndpointManager::register_publisher(
  const rmw_gid_t & gid,
  const char * topic_name)
{
  return register_endpoint(gid, topic_name, EntityType::PUBLISHER);
}

bool HostEndpointManager::register_subscription(
  const rmw_gid_t & gid,
  const char * topic_name)
{
  return register_endpoint(gid, topic_name, EntityType::SUBSCRIPTION);
}

bool HostEndpointManager::register_endpoint(
  const rmw_gid_t & gid,
  const char * name,
  EntityType type)
{
  auto * registry = static_cast<EndpointRegistry *>(shm_ptr_);
  sem_t * sem = static_cast<sem_t *>(shm_mutex_);

  if (!sem_wait_timed(sem)) {
    return false;
  }

  EndpointEntry * slot = nullptr;

  for (size_t i = 0; i < registry->num_entries; ++i) {
    if (memcmp(registry->entries[i].gid, gid.data, RMW_GID_STORAGE_SIZE) == 0) {
      slot = &registry->entries[i];
      break;
    }
  }

  if (slot == nullptr) {
    for (size_t i = 0; i < registry->num_entries; ++i) {
      if (!registry->entries[i].active) {
        slot = &registry->entries[i];
        break;
      }
    }
    if (!slot) {
      if (registry->num_entries >= registry->max_entries) {
        sem_post(sem);
        RCUTILS_LOG_ERROR_NAMED(
          "hip_host_endpoint_manager",
          "Shared memory registry full (max %u entries)", registry->max_entries);
        return false;
      }

      slot = &registry->entries[registry->num_entries];
      registry->num_entries++;
    }
  }

  slot->instance_id = instance_id_;
  memcpy(slot->gid, gid.data, RMW_GID_STORAGE_SIZE);
  slot->entity_type = type;
  slot->device_id = local_device_id_;
  slot->uid = local_uid_;
  slot->ipc_capable = local_ipc_capable_;
  slot->active = true;
  if (name != nullptr) {
    strncpy(slot->topic_or_service_name, name, 255);
    slot->topic_or_service_name[255] = '\0';
  } else {
    slot->topic_or_service_name[0] = '\0';
  }

  sem_post(sem);

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    local_cache_[GidKey(gid)] = {
      EndpointLocality::INTRA_PROCESS,
      instance_id_,
      type,
      local_device_id_,
      local_uid_,
      local_ipc_capable_
    };
  }

  return true;
}

LocalityInfo HostEndpointManager::query_endpoint_locality(const rmw_gid_t & gid) const
{
  std::lock_guard<std::mutex> lock(cache_mutex_);
  auto it = local_cache_.find(GidKey(gid));

  if (it == local_cache_.end()) {
    return {
      EndpointLocality::UNDEFINED,
      EntityType::PUBLISHER,
      0,
      -1,
      0,
      false,
      false
    };
  }

  return {
    it->second.locality,
    it->second.entity_type,
    it->second.instance_id,
    it->second.device_id,
    it->second.uid,
    it->second.ipc_capable,
    true
  };
}

void HostEndpointManager::refresh_from_remote()
{
  auto * registry = static_cast<EndpointRegistry *>(shm_ptr_);
  sem_t * sem = static_cast<sem_t *>(shm_mutex_);

  if (!sem_wait_timed(sem)) {
    return;
  }

  std::unordered_map<GidKey, CachedEndpointInfo, GidKeyHash> updates;

  for (size_t i = 0; i < registry->num_entries; ++i) {
    if (!registry->entries[i].active) {
      continue;
    }

    const auto & entry = registry->entries[i];

    EndpointLocality locality =
      (entry.instance_id == instance_id_) ?
      EndpointLocality::INTRA_PROCESS :
      EndpointLocality::INTER_PROCESS_SAME_HOST;

    updates[GidKey(entry.gid)] = {
      locality,
      entry.instance_id,
      entry.entity_type,
      entry.device_id,
      entry.uid,
      entry.ipc_capable
    };
  }

  sem_post(sem);

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (const auto & [key, info] : updates) {
      local_cache_[key] = info;
    }
  }
}

int HostEndpointManager::get_device_id() const
{
  return local_device_id_;
}

uint32_t HostEndpointManager::get_uid() const
{
  return local_uid_;
}

uint32_t HostEndpointManager::hash_string(const std::string & str)
{
  uint32_t hash = 0;
  for (char c : str) {
    hash = hash * 31 + static_cast<uint32_t>(c);
  }
  return hash;
}

}  // namespace hip_host_endpoint_manager
