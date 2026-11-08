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

#ifndef HIP_BUFFER__HOST_ENDPOINT_MANAGER_HPP_
#define HIP_BUFFER__HOST_ENDPOINT_MANAGER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "hip_buffer/visibility_control.h"
#include "rmw/types.h"

namespace hip_host_endpoint_manager
{

enum class EntityType : uint8_t
{
  PUBLISHER = 0,
  SUBSCRIPTION = 1,
  SERVICE_CLIENT = 2,
  SERVICE_SERVER = 3
};

enum class EndpointLocality : uint8_t
{
  UNDEFINED = 0,
  INTRA_PROCESS = 1,
  INTER_PROCESS_SAME_HOST = 2,
  INTER_HOST = 3
};

struct LocalityInfo
{
  EndpointLocality locality;
  EntityType entity_type;
  uint64_t remote_instance_id;
  int device_id;
  uint32_t uid;
  bool ipc_capable;
  bool found;
};

struct GidKey
{
  std::array<uint8_t, RMW_GID_STORAGE_SIZE> data;

  explicit GidKey(const rmw_gid_t & gid)
  {
    std::memcpy(data.data(), gid.data, RMW_GID_STORAGE_SIZE);
  }

  explicit GidKey(const uint8_t * raw)
  {
    std::memcpy(data.data(), raw, RMW_GID_STORAGE_SIZE);
  }

  bool operator==(const GidKey & other) const
  {
    return data == other.data;
  }
};

struct GidKeyHash
{
  size_t operator()(const GidKey & key) const
  {
    std::hash<std::string> hasher;
    return hasher(
      std::string(reinterpret_cast<const char *>(key.data.data()), RMW_GID_STORAGE_SIZE));
  }
};

class HIP_BUFFER_PUBLIC HostEndpointManager
{
public:
  static std::shared_ptr<HostEndpointManager> get_instance(size_t domain_id);

  ~HostEndpointManager();

  HostEndpointManager(const HostEndpointManager &) = delete;
  HostEndpointManager & operator=(const HostEndpointManager &) = delete;
  HostEndpointManager(HostEndpointManager &&) = delete;
  HostEndpointManager & operator=(HostEndpointManager &&) = delete;

  void set_ipc_capable(bool capable) {local_ipc_capable_ = capable;}

  bool register_publisher(const rmw_gid_t & gid, const char * topic_name);
  bool register_subscription(const rmw_gid_t & gid, const char * topic_name);
  LocalityInfo query_endpoint_locality(const rmw_gid_t & gid) const;

  void refresh_from_remote();

  int get_device_id() const;
  uint32_t get_uid() const;

private:
  explicit HostEndpointManager(size_t domain_id);

  bool register_endpoint(
    const rmw_gid_t & gid,
    const char * name,
    EntityType type);

  static uint32_t hash_string(const std::string & str);

  static std::mutex instances_mutex_;
  static std::unordered_map<size_t, std::shared_ptr<HostEndpointManager>> instances_;

  const size_t domain_id_;
  uint64_t instance_id_;
  std::string hostname_;

  int shm_fd_;
  void * shm_ptr_;
  size_t shm_size_;
  void * shm_mutex_;
  std::string shm_name_;

  struct CachedEndpointInfo
  {
    EndpointLocality locality;
    uint64_t instance_id;
    EntityType entity_type;
    int device_id;
    uint32_t uid;
    bool ipc_capable;
  };
  mutable std::unordered_map<GidKey, CachedEndpointInfo, GidKeyHash> local_cache_;
  mutable std::mutex cache_mutex_;

  int local_device_id_;
  uint32_t local_uid_;
  bool local_ipc_capable_{false};
};

}  // namespace hip_host_endpoint_manager

#endif  // HIP_BUFFER__HOST_ENDPOINT_MANAGER_HPP_
