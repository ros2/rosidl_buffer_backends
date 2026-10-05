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

#include "intel_buffer_backend/intel_buffer_backend.hpp"

#include <unistd.h>

#include <cstring>
#include <typeinfo>

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/intel_memory_ipc_manager.hpp"
#include "intel_memory_core/intel_memory_buffer_pool.hpp"
#include "intel_memory_core/l0_external_memmap.hpp"
#include "intel_memory_core/l0_usm_ipc.hpp"
#include "intel_buffer_backend_msgs/msg/intel_buffer_descriptor.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rcutils/logging_macros.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"

namespace intel_buffer_backend
{

using IntelBufferDescriptor = intel_buffer_backend_msgs::msg::IntelBufferDescriptor;

using intel_memory_core::TransportMode;
using intel_memory_core::IntelMemoryIPCManager;
using intel_memory_core::L0UsmIpc;
using intel_memory_core::ExternalMemMapImporter;

IntelBufferBackend::IntelBufferBackend()
{
}

std::string IntelBufferBackend::get_backend_metadata() const
{
  auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
  if (!pool) {
    return "transport=extmemmap;extmemmap=1";
  }
  std::string meta = "transport=";
  if (pool->is_level_zero_capable()) {
    meta += "l0usm";
  } else if (pool->is_dmabuf_capable()) {
    meta += "dmabuf";
  } else {
    meta += "none";
  }
  meta += ";device=" + std::to_string(pool->ze_device_ordinal());
  meta += ";extmemmap=1";
  return meta;
}

const rosidl_message_type_support_t *
IntelBufferBackend::get_descriptor_type_support() const
{
  return rosidl_typesupport_cpp::get_message_type_support_handle<IntelBufferDescriptor>();
}

std::shared_ptr<void>
IntelBufferBackend::create_empty_descriptor() const
{
  return std::make_shared<IntelBufferDescriptor>();
}

void IntelBufferBackend::on_creating_endpoint(
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  (void)endpoint_info;
  IntelBufferImpl<uint8_t>::get_or_create_global_pool();
}

std::pair<bool, std::vector<std::set<uint32_t>>>
IntelBufferBackend::on_discovering_endpoint(
  const rmw_topic_endpoint_info_t & endpoint_info,
  const std::vector<rmw_topic_endpoint_info_t> & existing_endpoints,
  const std::unordered_map<std::string, std::string> & endpoint_supported_backends)
{
  (void)existing_endpoints;

  auto it = endpoint_supported_backends.find("intel_buffer");
  if (it == endpoint_supported_backends.end()) {
    return {false, {}};
  }

  const std::string & remote_meta = it->second;

  bool compatible = remote_meta.find("transport=extmemmap") != std::string::npos &&
    remote_meta.find("extmemmap=1") != std::string::npos;

  if (!compatible) {
    if (!IntelBufferImpl<uint8_t>::is_ipc_capable()) {
      return {false, {}};
    }

    auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
    if (!pool) {
      return {false, {}};
    }

    if (pool->is_level_zero_capable() &&
      remote_meta.find("transport=l0usm") != std::string::npos)
    {
      compatible = true;
    } else if (pool->is_dmabuf_capable() &&
      remote_meta.find("transport=dmabuf") != std::string::npos)
    {
      compatible = true;
    }
  }

  {
    uint32_t gid_hash = 0;
    for (size_t i = 0; i < RMW_GID_STORAGE_SIZE && i < 4; ++i) {
      gid_hash = (gid_hash << 8) | endpoint_info.endpoint_gid[i];
    }
    std::lock_guard<std::mutex> lock(cache_mutex_);
    ipc_decision_cache_[gid_hash] = compatible;
  }

  return {compatible, {}};
}

std::shared_ptr<void> IntelBufferBackend::create_descriptor_with_endpoint(
  const void * impl,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  {
    uint32_t gid_hash = 0;
    for (size_t i = 0; i < RMW_GID_STORAGE_SIZE && i < 4; ++i) {
      gid_hash = (gid_hash << 8) | endpoint_info.endpoint_gid[i];
    }
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = ipc_decision_cache_.find(gid_hash);
    if (it != ipc_decision_cache_.end() && !it->second) {
      return nullptr;
    }
  }

  auto * usm_impl = dynamic_cast<IntelBufferImpl<uint8_t> *>(
    const_cast<rosidl::BufferImplBase<uint8_t> *>(
      static_cast<const rosidl::BufferImplBase<uint8_t> *>(impl)));
  if (!usm_impl) {
    return nullptr;
  }

  PoolBlock * block = usm_impl->get_block();
  if (!block) {
    return nullptr;
  }

  if (block->transport == TransportMode::EXTERNAL_MEMMAP) {
    if (block->extmap_shm_name.empty()) {
      RCUTILS_LOG_WARN_NAMED("intel_buffer_backend",
        "external-memmap block has no shm name; falling back");
      return nullptr;
    }
    auto frame = std::make_shared<IntelBufferDescriptor>();
    frame->transport_mode = IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP;
    auto & desc = frame->extmap_desc;
    desc.size = usm_impl->size();
    desc.shm_name = block->extmap_shm_name;
    desc.local_ptr = reinterpret_cast<uint64_t>(block->ptr);
    desc.publisher_pid = static_cast<int32_t>(getpid());
    desc.slot_generation = block->extmap_generation;
    return frame;
  }

  auto pool = IntelBufferImpl<uint8_t>::get_or_create_global_pool();
  if (!pool) {
    return nullptr;
  }

  auto frame = std::make_shared<IntelBufferDescriptor>();

  if (block->transport == TransportMode::LEVEL_ZERO_USM) {
    frame->transport_mode = IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM;
    auto & desc = frame->l0_desc;
    desc.size = usm_impl->size();
    desc.pool_id = pool->pool_id();
    desc.pool_generation = pool->generation();
    desc.ipc_uid = pool->assign_uid(block);
    desc.ze_device_ordinal = block->ze_device_ordinal;
    std::memcpy(desc.ze_device_uuid.data(), block->ze_device_uuid,
      sizeof(block->ze_device_uuid));
    desc.block_id = block->block_id;
    desc.local_ptr = reinterpret_cast<uint64_t>(block->ptr);
    desc.publisher_pid = static_cast<int32_t>(getpid());

    uint64_t handle_size = 0;
    uint64_t context_id = 0;
    if (!L0UsmIpc::export_handle(
        block,
        desc.ze_ipc_handle.data(),
        handle_size,
        context_id))
    {
      RCUTILS_LOG_WARN_NAMED("intel_buffer_backend",
        "L0UsmIpc::export_handle failed; falling back");
      return nullptr;
    }
    desc.ze_ipc_handle_size = handle_size;
    desc.ze_context_id = context_id;

    desc.l0_socket_path = pool->register_block_for_ipc(block);

  } else {
    frame->transport_mode = IntelBufferDescriptor::TRANSPORT_DMA_BUF;
    auto & desc = frame->dmabuf_desc;
    desc.size = usm_impl->size();
    desc.pool_id = pool->pool_id();
    desc.pool_generation = pool->generation();
    desc.ipc_uid = pool->assign_uid(block);
    desc.local_ptr = reinterpret_cast<uint64_t>(block->ptr);
    desc.publisher_pid = static_cast<int32_t>(getpid());
    desc.dmabuf_pid = static_cast<int32_t>(getpid());
    desc.dmabuf_pool_block_id = block->block_id;
    desc.dmabuf_block_size = block->size;

    std::string socket_path = pool->register_block_for_ipc(block);
    if (socket_path.empty()) {
      return nullptr;
    }
    desc.dmabuf_socket_path = socket_path;
  }

  return frame;
}

std::unique_ptr<void, void (*)(void *)> IntelBufferBackend::from_descriptor_with_endpoint(
  const void * descriptor_ptr,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  (void)endpoint_info;

  const auto * frame = static_cast<const IntelBufferDescriptor *>(descriptor_ptr);

  try {
    IntelMemoryIPCManager::ImportResult import_result{};

    if (frame->transport_mode == IntelBufferDescriptor::TRANSPORT_EXTERNAL_MEMMAP) {
      const auto & d = frame->extmap_desc;
      if (d.size == 0) {
        throw std::runtime_error("external-memmap descriptor has zero size");
      }
      void * ptr = nullptr;
      if (d.local_ptr != 0 && d.publisher_pid == static_cast<int32_t>(getpid())) {
        // Intra-process: the producer's mapping is already valid here.
        ptr = reinterpret_cast<void *>(d.local_ptr);
      } else {
        ptr = ExternalMemMapImporter::import(d.shm_name, d.size, d.publisher_pid);
        if (!ptr) {
          throw std::runtime_error("external-memmap import failed for " + d.shm_name);
        }
      }
      auto imported_block = std::make_unique<PoolBlock>();
      imported_block->ptr = ptr;
      imported_block->size = d.size;
      imported_block->transport = TransportMode::EXTERNAL_MEMMAP;
      imported_block->extmap_shm_name = d.shm_name;

      auto result = std::make_unique<IntelBufferImpl<uint8_t>>(
        imported_block.release(), d.size, false, /*standalone=*/true);
      return {result.release(), [](void * p) {
          delete static_cast<IntelBufferImpl<uint8_t> *>(p);
        }};
    }

    if (frame->transport_mode == IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM) {
      const auto & d = frame->l0_desc;
      // Intra-process shortcut
      if (d.local_ptr != 0 && d.publisher_pid == static_cast<int32_t>(getpid())) {
        auto imported_block = std::make_unique<PoolBlock>();
        imported_block->ptr = reinterpret_cast<void *>(d.local_ptr);
        imported_block->size = d.size;
        imported_block->transport = TransportMode::LEVEL_ZERO_USM;

        auto result = std::make_unique<IntelBufferImpl<uint8_t>>(
          imported_block.release(), d.size, false, /*standalone=*/true);
        return {result.release(), [](void * p) {
            delete static_cast<IntelBufferImpl<uint8_t> *>(p);
          }};
      }
      import_result = IntelMemoryIPCManager::import_level_zero_block(
        d.ze_ipc_handle.data(), d.ze_ipc_handle_size,
        d.ze_device_ordinal, d.ze_device_uuid.data(), d.size,
        d.publisher_pid, d.block_id, d.ipc_uid, d.l0_socket_path, d.pool_id);
    } else {
      const auto & d = frame->dmabuf_desc;
      // Intra-process shortcut
      if (d.local_ptr != 0 && d.publisher_pid == static_cast<int32_t>(getpid())) {
        auto imported_block = std::make_unique<PoolBlock>();
        imported_block->ptr = reinterpret_cast<void *>(d.local_ptr);
        imported_block->size = d.size;
        imported_block->transport = TransportMode::DMA_BUF;

        auto result = std::make_unique<IntelBufferImpl<uint8_t>>(
          imported_block.release(), d.size, false, /*standalone=*/true);
        return {result.release(), [](void * p) {
            delete static_cast<IntelBufferImpl<uint8_t> *>(p);
          }};
      }
      import_result = IntelMemoryIPCManager::import_dmabuf_block(
        d.dmabuf_socket_path, d.dmabuf_pid,
        d.dmabuf_pool_block_id, d.dmabuf_block_size, d.ipc_uid, d.pool_id);
    }

    if (!import_result.ptr) {
      throw std::runtime_error("IPC import returned no mapping (stale or failed)");
    }

    const uint64_t block_size =
      (frame->transport_mode == IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM)
      ? frame->l0_desc.size : frame->dmabuf_desc.size;

    auto imported_block = std::make_unique<PoolBlock>();
    imported_block->ptr = import_result.ptr;
    imported_block->size = block_size;
    imported_block->ipc_meta = import_result.ipc_meta;
    imported_block->transport = (frame->transport_mode == IntelBufferDescriptor::TRANSPORT_LEVEL_ZERO_USM)
      ? TransportMode::LEVEL_ZERO_USM : TransportMode::DMA_BUF;

    imported_block->dmabuf_fd = import_result.dmabuf_fd;

    auto result = std::make_unique<IntelBufferImpl<uint8_t>>(
      imported_block.release(), block_size, false, /*standalone=*/true);
    return {result.release(), [](void * p) {
        auto * impl = static_cast<IntelBufferImpl<uint8_t> *>(p);
        if (impl->get_block() && impl->get_block()->ipc_meta) {
          IntelMemoryIPCManager::release_block(impl->get_block()->ipc_meta);
        }
        delete impl;
      }};
  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_NAMED("intel_buffer_backend",
      "IPC import failed, returning empty buffer: %s", e.what());
  }

  auto empty = std::make_unique<IntelBufferImpl<uint8_t>>();
  return {empty.release(), [](void * p) {
      delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
    }};
}

}  // namespace intel_buffer_backend

PLUGINLIB_EXPORT_CLASS(
  intel_buffer_backend::IntelBufferBackend,
  rosidl::BufferBackend)
