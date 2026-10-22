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

#include "cuda_buffer/host_endpoint_manager.hpp"

#include <fcntl.h>
#include <linux/futex.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <rcutils/logging_macros.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace host_endpoint_manager
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
      "host_endpoint_manager",
      "Invalid value for %s='%s' (expected integer in [%s, %s]); ignoring",
      name, value, std::to_string(min).c_str(), std::to_string(max).c_str());
    return std::nullopt;
  }
  return parsed;
}

// flock is released by the kernel on process exit. The local mutex also
// serializes threads sharing this open file description.
class RegistryLock
{
public:
  RegistryLock(std::mutex & mutex, int fd)
  : lock_(mutex), fd_(fd)
  {
    while (flock(fd_, LOCK_EX) != 0) {
      if (errno != EINTR) {
        throw std::runtime_error("Failed to lock host registry");
      }
    }
  }
  ~RegistryLock() {flock(fd_, LOCK_UN);}

private:
  std::unique_lock<std::mutex> lock_;
  int fd_;
};

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
  alignas(4) uint32_t generation;
  char hostname[256];
  uint32_t max_entries;
  uint32_t num_entries;
  EndpointEntry entries[MAX_ENDPOINTS];
};

static_assert(__atomic_always_lock_free(sizeof(uint32_t), nullptr),
  "Registry generation requires lock-free shared atomic operations");

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

// Accessed atomically because futex observes this word outside the file lock.
static uint32_t generation(const EndpointRegistry * registry)
{
  return __atomic_load_n(&registry->generation, __ATOMIC_ACQUIRE);
}

static void advance_generation(EndpointRegistry * registry)
{
  __atomic_add_fetch(&registry->generation, 1u, __ATOMIC_SEQ_CST);
}

static void wake_listeners(EndpointRegistry * registry)
{
  if (syscall(SYS_futex, &registry->generation, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0) < 0) {
    RCUTILS_LOG_ERROR_NAMED("host_endpoint_manager", "Registry wake failed: %d", errno);
  }
}

static void deliver(
  const HostEndpointManager::UpdateSubscription & callback,
  const GidKey & key, const LocalityInfo & info) noexcept
{
  try {
    (*callback)(key, info);
  } catch (...) {
    RCUTILS_LOG_ERROR_NAMED("host_endpoint_manager", "Endpoint subscriber threw an exception");
  }
}

std::mutex HostEndpointManager::instances_mutex_;
std::unordered_map<size_t, std::weak_ptr<HostEndpointManager>>
HostEndpointManager::instances_;

std::shared_ptr<HostEndpointManager>
HostEndpointManager::get_instance(size_t domain_id)
{
  std::lock_guard<std::mutex> lock(instances_mutex_);

  auto it = instances_.find(domain_id);
  if (it != instances_.end()) {
    if (auto instance = it->second.lock()) {
      return instance;
    }
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
  local_device_id_(-1),
  local_uid_(0)
{
  static std::atomic<uint64_t> next_id{1};
  instance_id_ =
    (static_cast<uint64_t>(getpid()) << 32) |
    next_id.fetch_add(1, std::memory_order_relaxed);

  int device = -1;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    local_device_id_ = device;
  }

  // Env-var names for testing IPC fallback paths without requiring multiple
  // GPUs or users.
  // Not intended for production configuration — use only from test launches.
  constexpr const char * kDeviceIdOverrideEnv = "CUDA_BUFFER_DEVICE_ID_OVERRIDE";
  constexpr const char * kUidOverrideEnv = "CUDA_BUFFER_UID_OVERRIDE";

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
  hostname_buf[sizeof(hostname_buf) - 1] = '\0';
  hostname_ = hostname_buf;

  uint32_t host_hash = hash_string(hostname_);
  shm_name_ = "/ros2_hem_v2_d" + std::to_string(domain_id) +
    "_h" + std::to_string(host_hash);

  shm_size_ = sizeof(EndpointRegistry);

  shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
  if (shm_fd_ < 0) {
    throw std::runtime_error(
            "Failed to open shared memory: " + std::string(strerror(errno)));
  }

  try {
    {
      RegistryLock lock(registry_mutex_, shm_fd_);
      struct stat status;
      if (fstat(shm_fd_, &status) != 0) {
        throw std::runtime_error("Failed to stat host registry");
      }
      if (status.st_size == 0) {
        if (ftruncate(shm_fd_, shm_size_) != 0) {
          throw std::runtime_error("Failed to size host registry");
        }
      } else if (static_cast<size_t>(status.st_size) != shm_size_) {
        throw std::runtime_error("Incompatible host registry size");
      }
      shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
      if (shm_ptr_ == MAP_FAILED) {
        throw std::runtime_error("Failed to map host registry");
      }
      auto * registry = static_cast<EndpointRegistry *>(shm_ptr_);
      if (registry->version == 0) {
        // The version is committed last; an interrupted initializer is retried.
        std::memset(registry, 0, shm_size_);
        std::strncpy(registry->hostname, hostname_.c_str(), 255);
        registry->max_entries = MAX_ENDPOINTS;
        __atomic_store_n(&registry->version, 2u, __ATOMIC_RELEASE);
      }
      if (registry->version != 2 || registry->max_entries != MAX_ENDPOINTS ||
        registry->num_entries > MAX_ENDPOINTS)
      {
        throw std::runtime_error("Incompatible or corrupt host registry");
      }
      advance_generation(registry);
      purge_dead_entries(registry);
      wake_listeners(registry);
    }
    refresh_from_remote();
    listener_ = std::thread(&HostEndpointManager::listen, this);
  } catch (...) {
    if (shm_ptr_ && shm_ptr_ != MAP_FAILED) {
      munmap(shm_ptr_, shm_size_);
    }
    close(shm_fd_);
    throw;
  }
}

HostEndpointManager::~HostEndpointManager()
{
  auto * registry = static_cast<EndpointRegistry *>(shm_ptr_);
  stopping_.store(true);
  // Change the futex word as well as waking it: shutdown between the listener's
  // stop check and FUTEX_WAIT must cause EAGAIN rather than a lost wakeup.
  advance_generation(registry);
  wake_listeners(registry);
  if (listener_.joinable()) {
    listener_.join();
  }
  try {
    RegistryLock lock(registry_mutex_, shm_fd_);
    advance_generation(registry);
    for (size_t i = 0; i < registry->num_entries; ++i) {
      if (registry->entries[i].instance_id == instance_id_) {
        registry->entries[i].active = false;
      }
    }
    wake_listeners(registry);
  } catch (...) {
    RCUTILS_LOG_ERROR_NAMED("host_endpoint_manager", "Failed to retire host endpoints");
  }
  munmap(shm_ptr_, shm_size_);
  close(shm_fd_);
  // Persist the named object. Unlinking based on endpoint count can split the
  // registry while another manager with no endpoints still has it mapped.
}

void HostEndpointManager::listen() noexcept
{
  auto * registry = static_cast<EndpointRegistry *>(shm_ptr_);
  uint32_t observed = generation(registry);
  bool initial = true;
  while (!stopping_.load()) {
    try {
      const auto current = generation(registry);
      if (initial || current != observed) {
        refresh_from_remote();
        observed = current;
        initial = false;
      }
      if (stopping_.load()) {
        break;
      }
      // A writer can die after committing a record but before FUTEX_WAKE.
      // Recovery checks only this word once a second; unchanged generations
      // never cause scans. FUTEX_WAIT atomically compares and sleeps.
      const timespec timeout{1, 0};
      const auto result = syscall(
        SYS_futex, &registry->generation, FUTEX_WAIT, observed, &timeout, nullptr, 0);
      if (result < 0 && errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT) {
        RCUTILS_LOG_ERROR_NAMED("host_endpoint_manager", "Registry wait failed: %d", errno);
        return;
      }
    } catch (...) {
      RCUTILS_LOG_ERROR_NAMED("host_endpoint_manager", "Host registry listener failed");
      return;
    }
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
  RegistryLock lock(registry_mutex_, shm_fd_);
  // Increment before modifying records so a crashed writer is observable.
  advance_generation(registry);
  purge_dead_entries(registry);

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
        wake_listeners(registry);
        RCUTILS_LOG_ERROR_NAMED(
          "host_endpoint_manager",
          "Shared memory registry full (max %u entries)", registry->max_entries);
        return false;
      }

      slot = &registry->entries[registry->num_entries];
      registry->num_entries++;
    }
  }

  __atomic_store_n(&slot->active, false, __ATOMIC_SEQ_CST);
  slot->instance_id = instance_id_;
  memcpy(slot->gid, gid.data, RMW_GID_STORAGE_SIZE);
  slot->entity_type = type;
  slot->device_id = local_device_id_;
  slot->uid = local_uid_;
  slot->ipc_capable = local_ipc_capable_;
  if (name != nullptr) {
    strncpy(slot->topic_or_service_name, name, 255);
    slot->topic_or_service_name[255] = '\0';
  } else {
    slot->topic_or_service_name[0] = '\0';
  }

  // Commit last. An interrupted replacement stays inactive, never half valid.
  __atomic_store_n(&slot->active, true, __ATOMIC_RELEASE);
  wake_listeners(registry);
  // The listener handles local registrations too, after releasing this lock.

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

HostEndpointManager::UpdateSubscription HostEndpointManager::subscribe_endpoint_updates(
  EndpointUpdateCallback callback)
{
  auto token = std::make_shared<const EndpointUpdateCallback>(std::move(callback));
  std::vector<std::pair<GidKey, LocalityInfo>> snapshot;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (const auto & [key, info] : local_cache_) {
      snapshot.emplace_back(key, LocalityInfo{info.locality, info.entity_type,
          info.instance_id, info.device_id, info.uid, info.ipc_capable, true});
    }
    subscribers_.erase(std::remove_if(subscribers_.begin(), subscribers_.end(),
      [](const auto & subscriber) {return subscriber.expired();}), subscribers_.end());
    subscribers_.push_back(token);
  }
  for (const auto & [key, info] : snapshot) {
    deliver(token, key, info);
  }
  return token;
}

void HostEndpointManager::refresh_from_remote()
{
  std::vector<std::pair<GidKey, LocalityInfo>> updates;
  std::vector<UpdateSubscription> subscribers;
  {
    // Lock order: registry, then cache. Publish/query only take cache_mutex_.
    RegistryLock registry_lock(registry_mutex_, shm_fd_);
    auto * registry = static_cast<EndpointRegistry *>(shm_ptr_);
    std::unordered_map<GidKey, CachedEndpointInfo, GidKeyHash> snapshot;
    for (size_t i = 0; i < registry->num_entries; ++i) {
      const auto & entry = registry->entries[i];
      if (!entry.active) {
        continue;
      }
      const auto locality = entry.instance_id == instance_id_ ?
        EndpointLocality::INTRA_PROCESS : EndpointLocality::INTER_PROCESS_SAME_HOST;
      snapshot[GidKey(entry.gid)] = {locality, entry.instance_id, entry.entity_type,
        entry.device_id, entry.uid, entry.ipc_capable};
    }
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    for (const auto & [key, info] : snapshot) {
      const auto old = local_cache_.find(key);
      if (old == local_cache_.end() || old->second.instance_id != info.instance_id ||
        old->second.entity_type != info.entity_type || old->second.device_id != info.device_id ||
        old->second.uid != info.uid || old->second.ipc_capable != info.ipc_capable)
      {
        updates.emplace_back(key, LocalityInfo{info.locality, info.entity_type,
            info.instance_id, info.device_id, info.uid, info.ipc_capable, true});
      }
    }
    local_cache_.swap(snapshot);
    for (auto it = subscribers_.begin(); it != subscribers_.end(); ) {
      if (auto subscriber = it->lock()) {
        subscribers.push_back(std::move(subscriber));
        ++it;
      } else {
        it = subscribers_.erase(it);
      }
    }
  }
  for (const auto & [key, info] : updates) {
    for (const auto & subscriber : subscribers) {
      deliver(subscriber, key, info);
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

}  // namespace host_endpoint_manager
