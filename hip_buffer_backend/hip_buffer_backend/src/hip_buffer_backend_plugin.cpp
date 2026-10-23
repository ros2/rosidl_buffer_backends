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

#include "hip_buffer_backend/hip_buffer_backend.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <typeinfo>

#include "hip_buffer/hip_buffer_ipc_manager.hpp"
#include "hip_buffer/hip_error.hpp"
#include "hip_buffer/hip_memory_pool.hpp"
#include "hip_buffer_backend_msgs/msg/hip_buffer_descriptor.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rcutils/logging_macros.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
// HIP adaptation of upstream
// cuda_buffer_backend/cuda_buffer_backend/src/cuda_buffer_backend_plugin.cpp.
// Unknown-subscriber handling is based on ros2/rosidl_buffer_backends PR #26,
// with HIP locality and device/UID compatibility checks.

namespace hip_buffer_backend
{

HipBufferBackend::HipBufferBackend()
{
}

const rosidl_message_type_support_t *
HipBufferBackend::get_descriptor_type_support() const
{
  return rosidl_typesupport_cpp::get_message_type_support_handle<
    hip_buffer_backend_msgs::msg::HipBufferDescriptor>();
}

std::shared_ptr<void>
HipBufferBackend::create_empty_descriptor() const
{
  return std::make_shared<hip_buffer_backend_msgs::msg::HipBufferDescriptor>();
}

std::shared_ptr<hip_host_endpoint_manager::HostEndpointManager>
HipBufferBackend::get_endpoint_manager() const
{
  std::lock_guard<std::mutex> lock(manager_mutex_);

  if (!endpoint_manager_) {
    size_t domain_id = 0;
    endpoint_manager_ = hip_host_endpoint_manager::HostEndpointManager::get_instance(domain_id);
    endpoint_manager_->set_ipc_capable(HipBufferImpl<uint8_t>::is_pool_ipc_capable());
  }

  return endpoint_manager_;
}

void HipBufferBackend::on_creating_endpoint(
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  auto manager = get_endpoint_manager();

  rmw_gid_t gid;
  gid.implementation_identifier = "";
  std::memcpy(gid.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);

  if (endpoint_info.endpoint_type == RMW_ENDPOINT_PUBLISHER) {
    manager->register_publisher(gid, endpoint_info.topic_type);
  } else if (endpoint_info.endpoint_type == RMW_ENDPOINT_SUBSCRIPTION) {
    manager->register_subscription(gid, endpoint_info.topic_type);
  }
}

std::pair<bool, std::vector<std::set<uint32_t>>>
HipBufferBackend::on_discovering_endpoint(
  const rmw_topic_endpoint_info_t & endpoint_info,
  const std::vector<rmw_topic_endpoint_info_t> & existing_endpoints,
  const std::unordered_map<std::string, std::string> & endpoint_supported_backends)
{
  (void)existing_endpoints;

  bool supports_hip = endpoint_supported_backends.find("hip") !=
    endpoint_supported_backends.end();
  if (!supports_hip) {
    return {false, {}};
  }

  if (!HipBufferImpl<uint8_t>::is_pool_ipc_capable()) {
    return {false, {}};
  }

  rmw_gid_t gid;
  gid.implementation_identifier = "";
  std::memcpy(gid.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);

  auto manager = get_endpoint_manager();
  manager->refresh_from_remote();

  auto locality_info = manager->query_endpoint_locality(gid);

  bool is_ipc_capable = false;

  if (locality_info.found && locality_info.ipc_capable) {
    auto loc = locality_info.locality;
    if (loc == hip_host_endpoint_manager::EndpointLocality::INTRA_PROCESS) {
      is_ipc_capable = true;
    } else if (loc == hip_host_endpoint_manager::EndpointLocality::INTER_PROCESS_SAME_HOST) {
      is_ipc_capable =
        (locality_info.device_id == manager->get_device_id()) &&
        (locality_info.uid == manager->get_uid());
    }
  }

  if (locality_info.found) {
    std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
    hip_host_endpoint_manager::GidKey key(endpoint_info.endpoint_gid);
    ipc_decision_cache_[key] = is_ipc_capable;
  }

  return {is_ipc_capable, {}};
}

std::shared_ptr<void> HipBufferBackend::create_descriptor_with_endpoint(
  const void * impl,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  hip_host_endpoint_manager::GidKey key(endpoint_info.endpoint_gid);

  bool decided = false;
  {
    std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
    auto it = ipc_decision_cache_.find(key);
    if (it != ipc_decision_cache_.end() && !it->second) {
      return nullptr;
    }
    decided = it != ipc_decision_cache_.end();
  }

  if (!decided) {
    // Ported from ros2/rosidl_buffer_backends PR #26, with the HIP locality
    // and device/UID compatibility checks required for this backend.
    rmw_gid_t peer;
    peer.implementation_identifier = "";
    std::memcpy(peer.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);
    auto manager = get_endpoint_manager();
    manager->refresh_from_remote();
    auto peer_locality = manager->query_endpoint_locality(peer);

    bool is_ipc_capable = false;
    if (peer_locality.found && peer_locality.ipc_capable) {
      auto loc = peer_locality.locality;
      if (loc == hip_host_endpoint_manager::EndpointLocality::INTRA_PROCESS) {
        is_ipc_capable = true;
      } else if (loc == hip_host_endpoint_manager::EndpointLocality::INTER_PROCESS_SAME_HOST) {
        is_ipc_capable =
          peer_locality.device_id == manager->get_device_id() &&
          peer_locality.uid == manager->get_uid();
      }
    }
    if (!is_ipc_capable) {
      return nullptr;
    }
  }

  auto * hip_impl = dynamic_cast<HipBufferImpl<uint8_t> *>(
    const_cast<rosidl::BufferImplBase<uint8_t> *>(
      static_cast<const rosidl::BufferImplBase<uint8_t> *>(impl)));
  if (!hip_impl) {
    return nullptr;
  }

  auto pool = HipBufferImpl<uint8_t>::get_or_create_global_pool();
  if (!pool || !pool->is_ipc_capable()) {
    return nullptr;
  }

  auto descriptor = std::make_shared<hip_buffer_backend_msgs::msg::HipBufferDescriptor>();
  static_assert(
    sizeof(descriptor->ipc_event_handle) == sizeof(hipIpcEventHandle_t),
    "HipBufferDescriptor event field must match hipIpcEventHandle_t");
  descriptor->size = hip_impl->size();
  descriptor->element_type_name = typeid(uint8_t).name();

  descriptor->device_id = hip_impl->get_device_id();

  void * dev_ptr = static_cast<void *>(
    hip_impl->get_hip_buffer().get_device_ptr());

  VmmBlock * block = pool->find_block_for_va(dev_ptr);
  if (!block || block->exported_fd < 0) {
    return nullptr;
  }

  std::string socket_path = pool->register_block_for_ipc(block);
  if (socket_path.empty()) {
    return nullptr;
  }

  uint64_t uid = pool->assign_uid(block);

  descriptor->use_ipc = true;
  descriptor->vmm_pid = static_cast<int32_t>(getpid());
  descriptor->vmm_block_id = block->block_id;
  descriptor->vmm_block_size = block->size;
  descriptor->vmm_socket_path = socket_path;
  descriptor->ipc_uid = uid;

  hip_impl->get_hip_buffer().finalize_write_handle();

  hipEvent_t write_event = hip_impl->get_hip_buffer().get_write_event();

  rmw_gid_t gid;
  gid.implementation_identifier = "";
  std::memcpy(gid.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);
  auto locality = get_endpoint_manager()->query_endpoint_locality(gid);
  if (locality.found &&
    locality.locality == hip_host_endpoint_manager::EndpointLocality::INTRA_PROCESS &&
    write_event)
  {
    RCUTILS_LOG_WARN_ONCE_NAMED("hip_buffer_backend",
      "Same-process HIP IPC requires hipEventSynchronize on the publish path. "
      "Enable intra-process communication to avoid this overhead.");
    HIP_CHECK(hipEventSynchronize(write_event));
  }

  if (write_event) {
    hipIpcEventHandle_t event_handle;
    HIP_CHECK(hipIpcGetEventHandle(&event_handle, write_event));
    pool->retain_exported_event(block, write_event, uid);
    hip_impl->get_hip_buffer().release_write_event_ownership();
    std::memcpy(
      descriptor->ipc_event_handle.data(), &event_handle, sizeof(hipIpcEventHandle_t));
  } else {
    std::memset(descriptor->ipc_event_handle.data(), 0, sizeof(hipIpcEventHandle_t));
  }

  return descriptor;
}

std::unique_ptr<void, void (*)(void *)> HipBufferBackend::from_descriptor_with_endpoint(
  const void * descriptor_ptr,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  (void)endpoint_info;

  const auto * descriptor =
    static_cast<const hip_buffer_backend_msgs::msg::HipBufferDescriptor *>(descriptor_ptr);

  int local_device_id = 0;
  HIP_CHECK(hipGetDevice(&local_device_id));

  if (descriptor->element_type_name != typeid(uint8_t).name()) {
    throw HipError(
            "HipBufferDescriptor element type mismatch: expected " +
            std::string(typeid(uint8_t).name()) + ", got " +
            descriptor->element_type_name);
  }

  size_t byte_size = descriptor->size * sizeof(uint8_t);

  if (descriptor->use_ipc && !descriptor->vmm_socket_path.empty()) {
    try {
      if (descriptor->device_id != local_device_id) {
        throw HipError("Cross-device IPC not implemented");
      }

      auto import_result = HipVmmIPCManager::import_block(
        descriptor->vmm_socket_path,
        descriptor->vmm_pid,
        descriptor->vmm_block_id,
        descriptor->vmm_block_size,
        descriptor->ipc_uid);

      IPCMetadata * meta = import_result.ipc_meta;
      auto deleter = [meta](uint8_t *) {
          if (meta) {
            meta->refcount.fetch_sub(1, std::memory_order_release);
          }
        };
      HipBuffer imported_buffer(
        reinterpret_cast<void *>(import_result.va), byte_size,
        descriptor->device_id, deleter);

      bool has_event = false;
      for (size_t i = 0; i < descriptor->ipc_event_handle.size(); ++i) {
        if (descriptor->ipc_event_handle[i] != 0) {
          has_event = true;
          break;
        }
      }

      if (has_event && descriptor->vmm_pid != static_cast<int32_t>(getpid())) {
        hipIpcEventHandle_t event_handle;
        std::memcpy(&event_handle, descriptor->ipc_event_handle.data(),
            sizeof(hipIpcEventHandle_t));

        hipEvent_t imported_event = nullptr;
        hipError_t ev_err = hipIpcOpenEventHandle(&imported_event, event_handle);
        if (ev_err == hipSuccess) {
          imported_buffer.set_write_event(imported_event, true);
        } else {
          throw HipError(__FILE__, __LINE__, "hipIpcOpenEventHandle", ev_err);
        }
      }

      auto result = std::make_unique<HipBufferImpl<uint8_t>>(
        std::move(imported_buffer), descriptor->size);
      return {result.release(), [](void * p) {
          delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
        }};
    } catch (const std::exception & e) {
      RCUTILS_LOG_WARN_NAMED("hip_buffer_backend",
        "IPC import failed; dropping descriptor: %s", e.what());
    }
  }

  RCUTILS_LOG_WARN_NAMED("hip_buffer_backend",
    "Dropping stale descriptor: IPC failed");
  auto empty = std::make_unique<HipBufferImpl<uint8_t>>();
  return {empty.release(), [](void * p) {
      delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
    }};
}
}  // namespace hip_buffer_backend

PLUGINLIB_EXPORT_CLASS(
  hip_buffer_backend::HipBufferBackend,
  rosidl::BufferBackend)
