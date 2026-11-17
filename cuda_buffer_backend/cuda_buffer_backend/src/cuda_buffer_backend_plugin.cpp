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

#include "cuda_buffer_backend/cuda_buffer_backend.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>

#include "cuda_buffer/cuda_buffer.hpp"
#include "cuda_buffer/cuda_buffer_ipc_manager.hpp"
#include "cuda_buffer/cuda_error.hpp"
#include "cuda_buffer/cuda_memory_pool.hpp"
#include "cuda_buffer/vmm_buffer.hpp"
#include "cuda_buffer_backend_msgs/msg/cuda_buffer_descriptor.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rcutils/logging_macros.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"


namespace cuda_buffer_backend
{

namespace
{

using cuda_buffer_backend_msgs::msg::CudaBufferDescriptor;

// -- CUDA VMM transport --
//
// Everything that is specific to sharing a VmmBuffer between processes lives
// in these two functions: exporting a pool block and its write event into a
// descriptor, and mapping a described block back into a VmmBuffer.

/// \brief Fill the VMM IPC fields of \p descriptor for \p buffer.
/// \param peer_is_same_process queried only once the block is known to be shareable.
/// \return false if \p buffer cannot be shared over VMM IPC.
bool export_vmm_buffer(
  VmmBuffer & buffer,
  const std::function<bool()> & peer_is_same_process,
  CudaBufferDescriptor & descriptor)
{
  auto pool = get_or_create_global_pool();
  if (!pool || !pool->is_ipc_capable()) {
    return false;
  }

  CUdeviceptr dev_ptr = reinterpret_cast<CUdeviceptr>(buffer.get_device_ptr());

  VmmBlock * block = pool->find_block_for_va(dev_ptr);
  if (!block || block->exported_fd < 0) {
    return false;
  }

  std::string socket_path = pool->register_block_for_ipc(block);
  if (socket_path.empty()) {
    return false;
  }

  uint64_t uid = pool->assign_uid(block);

  descriptor.use_ipc = true;
  descriptor.vmm_pid = static_cast<int32_t>(getpid());
  descriptor.vmm_block_id = block->block_id;
  descriptor.vmm_block_size = block->size;
  descriptor.vmm_socket_path = socket_path;
  descriptor.ipc_uid = uid;

  buffer.finalize_write_handle();

  cudaEvent_t write_event = buffer.get_write_event();

  if (peer_is_same_process() && write_event) {
    RCUTILS_LOG_WARN_ONCE_NAMED("cuda_buffer_backend",
      "Same-process CUDA IPC requires cudaEventSynchronize on the publish path. "
      "Enable intra-process communication to avoid this overhead.");
    cudaEventSynchronize(write_event);
  }

  if (write_event) {
    cudaIpcEventHandle_t event_handle;
    CUDA_CHECK(cudaIpcGetEventHandle(&event_handle, write_event));
    std::memcpy(
      descriptor.ipc_event_handle.data(), &event_handle, sizeof(cudaIpcEventHandle_t));
  } else {
    std::memset(descriptor.ipc_event_handle.data(), 0, sizeof(cudaIpcEventHandle_t));
  }

  return true;
}

/// \brief Map the VMM block described by \p descriptor into this process.
/// \throws std::exception if the block cannot be imported.
std::unique_ptr<CudaBuffer> import_vmm_buffer(
  const CudaBufferDescriptor & descriptor,
  int local_device_id)
{
  if (descriptor.device_id != local_device_id) {
    throw CudaError("Cross-device IPC not implemented");
  }

  auto import_result = CudaVmmIPCManager::import_block(
    descriptor.vmm_socket_path,
    descriptor.vmm_pid,
    descriptor.vmm_block_id,
    descriptor.vmm_block_size,
    descriptor.ipc_uid);

  IPCMetadata * meta = import_result.ipc_meta;
  auto deleter = [meta](uint8_t *) {
      if (meta) {
        meta->refcount.fetch_sub(1, std::memory_order_release);
      }
    };
  auto imported_buffer = std::make_unique<VmmBuffer>(
    reinterpret_cast<void *>(import_result.va), descriptor.size * sizeof(uint8_t),
    descriptor.device_id, deleter);

  bool has_event = false;
  for (size_t i = 0; i < descriptor.ipc_event_handle.size(); ++i) {
    if (descriptor.ipc_event_handle[i] != 0) {
      has_event = true;
      break;
    }
  }

  if (has_event) {
    cudaIpcEventHandle_t event_handle;
    std::memcpy(&event_handle, descriptor.ipc_event_handle.data(),
        sizeof(cudaIpcEventHandle_t));

    cudaEvent_t imported_event = nullptr;
    cudaError_t ev_err = cudaIpcOpenEventHandle(&imported_event, event_handle);
    if (ev_err == cudaSuccess) {
      imported_buffer->set_write_event(imported_event, true);
    } else {
      // Same-process: IPC event handles can't be opened in the originating
      // process.
      (void)cudaGetLastError();
    }
  }

  return imported_buffer;
}

}  // namespace

CudaBufferBackend::CudaBufferBackend()
{
}

const rosidl_message_type_support_t *
CudaBufferBackend::get_descriptor_type_support() const
{
  return rosidl_typesupport_cpp::get_message_type_support_handle<CudaBufferDescriptor>();
}

std::shared_ptr<void>
CudaBufferBackend::create_empty_descriptor() const
{
  return std::make_shared<CudaBufferDescriptor>();
}

std::shared_ptr<host_endpoint_manager::HostEndpointManager>
CudaBufferBackend::get_endpoint_manager() const
{
  std::lock_guard<std::mutex> lock(manager_mutex_);

  if (!endpoint_manager_) {
    size_t domain_id = 0;
    endpoint_manager_ = host_endpoint_manager::HostEndpointManager::get_instance(domain_id);
    endpoint_manager_->set_ipc_capable(CudaBufferImpl<uint8_t>::is_pool_ipc_capable());
  }

  return endpoint_manager_;
}

bool CudaBufferBackend::can_import_descriptor(
  const host_endpoint_manager::LocalityInfo & peer,
  const host_endpoint_manager::HostEndpointManager & manager)
{
  if (!peer.found || !peer.ipc_capable) {
    return false;
  }
  if (peer.locality == host_endpoint_manager::EndpointLocality::INTRA_PROCESS) {
    return true;
  }
  if (peer.locality == host_endpoint_manager::EndpointLocality::INTER_PROCESS_SAME_HOST) {
    return peer.device_id == manager.get_device_id() && peer.uid == manager.get_uid();
  }
  return false;
}

void CudaBufferBackend::on_creating_endpoint(
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
CudaBufferBackend::on_discovering_endpoint(
  const rmw_topic_endpoint_info_t & endpoint_info,
  const std::vector<rmw_topic_endpoint_info_t> & existing_endpoints,
  const std::unordered_map<std::string, std::string> & endpoint_supported_backends)
{
  (void)existing_endpoints;

  bool supports_cuda = endpoint_supported_backends.find("cuda") !=
    endpoint_supported_backends.end();
  if (!supports_cuda) {
    return {false, {}};
  }

  if (!CudaBufferImpl<uint8_t>::is_pool_ipc_capable()) {
    return {false, {}};
  }

  rmw_gid_t gid;
  gid.implementation_identifier = "";
  std::memcpy(gid.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);

  auto manager = get_endpoint_manager();
  manager->refresh_from_remote();

  auto locality_info = manager->query_endpoint_locality(gid);

  bool is_ipc_capable = can_import_descriptor(locality_info, *manager);

  if (locality_info.found) {
    std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
    host_endpoint_manager::GidKey key(endpoint_info.endpoint_gid);
    ipc_decision_cache_[key] = is_ipc_capable;
  }

  return {is_ipc_capable, {}};
}

std::shared_ptr<void> CudaBufferBackend::create_descriptor_with_endpoint(
  const void * impl,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  host_endpoint_manager::GidKey key(endpoint_info.endpoint_gid);

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
    // The subscriber was never classified, so classify it the way
    // on_discovering_endpoint does. If it is missing from the local endpoint
    // registry it is on another host or has not registered yet; if it is
    // present it may have no usable device, or use another device or uid.
    // None of these can import a CUDA descriptor, so use the host path.
    // A peer found in the registry is cached, as in on_discovering_endpoint.
    // A miss is not cached, so a same-host peer gets a CUDA descriptor once
    // it registers.
    rmw_gid_t peer;
    peer.implementation_identifier = "";
    std::memcpy(peer.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);
    auto peer_manager = get_endpoint_manager();
    peer_manager->refresh_from_remote();
    auto peer_locality = peer_manager->query_endpoint_locality(peer);
    bool can_import = can_import_descriptor(peer_locality, *peer_manager);
    if (peer_locality.found) {
      std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
      ipc_decision_cache_[key] = can_import;
    }
    if (!can_import) {
      return nullptr;
    }
  }

  auto * cuda_impl = dynamic_cast<CudaBufferImpl<uint8_t> *>(
    const_cast<rosidl::BufferImplBase<uint8_t> *>(
      static_cast<const rosidl::BufferImplBase<uint8_t> *>(impl)));
  if (!cuda_impl) {
    return nullptr;
  }

  CudaBuffer & storage = cuda_impl->get_cuda_buffer();

  auto descriptor = std::make_shared<CudaBufferDescriptor>();

  descriptor->size = cuda_impl->size();
  descriptor->element_type_name = typeid(uint8_t).name();

  descriptor->device_id = cuda_impl->get_device_id();

  auto peer_is_same_process = [this, &endpoint_info]() {
      rmw_gid_t gid;
      gid.implementation_identifier = "";
      std::memcpy(gid.data, endpoint_info.endpoint_gid, RMW_GID_STORAGE_SIZE);
      auto locality = get_endpoint_manager()->query_endpoint_locality(gid);
      return locality.found &&
             locality.locality == host_endpoint_manager::EndpointLocality::INTRA_PROCESS;
    };

  // The storage backend of the buffer decides how it is shared. A backend
  // whose storage cannot be exported falls back to the CPU path. There is no
  // default case, so -Wswitch points here when a backend is added.
  bool exported = false;
  switch (storage.storage_kind()) {
    case StorageKind::Vmm:
      exported = export_vmm_buffer(
        static_cast<VmmBuffer &>(storage), peer_is_same_process, *descriptor);
      break;
  }
  if (!exported) {
    return nullptr;
  }

  return descriptor;
}

std::unique_ptr<void, void (*)(void *)> CudaBufferBackend::from_descriptor_with_endpoint(
  const void * descriptor_ptr,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  (void)endpoint_info;

  const auto * descriptor = static_cast<const CudaBufferDescriptor *>(descriptor_ptr);

  int local_device_id = 0;
  CUDA_CHECK(cudaGetDevice(&local_device_id));

  if (descriptor->element_type_name != typeid(uint8_t).name()) {
    throw CudaError(
            "CudaBufferDescriptor element type mismatch: expected " +
            std::string(typeid(uint8_t).name()) + ", got " +
            descriptor->element_type_name);
  }

  if (descriptor->use_ipc && !descriptor->vmm_socket_path.empty()) {
    try {
      auto imported_buffer = import_vmm_buffer(*descriptor, local_device_id);

      auto result = std::make_unique<CudaBufferImpl<uint8_t>>(
        std::move(imported_buffer), descriptor->size);
      return {result.release(), [](void * p) {
          delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
        }};
    } catch (const std::exception & e) {
      RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
        "IPC import failed, falling back to CPU: %s", e.what());
    }
  }

  RCUTILS_LOG_WARN_NAMED("cuda_buffer_backend",
    "Dropping stale descriptor: IPC failed");
  auto empty = std::make_unique<CudaBufferImpl<uint8_t>>();
  return {empty.release(), [](void * p) {
      delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
    }};
}
}  // namespace cuda_buffer_backend

PLUGINLIB_EXPORT_CLASS(
  cuda_buffer_backend::CudaBufferBackend,
  rosidl::BufferBackend)
