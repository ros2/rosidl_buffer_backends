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

// Level Zero Host USM IPC — export and import
#ifndef INTEL_MEMORY_CORE__L0_USM_IPC_HPP_
#define INTEL_MEMORY_CORE__L0_USM_IPC_HPP_

#include <cstdint>
#include <string>
#include <unordered_map>

#include "intel_memory_core/intel_memory_buffer_pool.hpp"

namespace intel_memory_core
{

struct IpcImportResult {
  void * ptr{nullptr};
  int dmabuf_fd{-1};
};

class L0UsmIpc
{
public:
  static bool export_handle(
    PoolBlock * block,
    uint8_t * out_handle_data,
    uint64_t & out_handle_size,
    uint64_t & out_context_id);

  static bool export_fd(PoolBlock * block, int & out_fd);

  static void put_handle(PoolBlock * block);

  static IpcImportResult import_handle(
    const uint8_t * handle_data,
    size_t handle_size,
    int32_t device_ordinal,
    const uint8_t * device_uuid,
    uint64_t size,
    int32_t pid,
    uint32_t block_id,
    const std::string & l0_socket_path,
    uint32_t pool_id);

  static void evict_pool(uint32_t pool_id);
};

}  // namespace intel_memory_core
#endif  // INTEL_MEMORY_CORE__L0_USM_IPC_HPP_
