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

#ifndef INTEL_MEMORY_CORE__INTEL_MEMORY_IPC_MANAGER_HPP_
#define INTEL_MEMORY_CORE__INTEL_MEMORY_IPC_MANAGER_HPP_

#include <cstdint>
#include <string>

namespace intel_memory_core
{

struct PoolBlock;
struct IPCMetadata;

class IntelMemoryIPCManager
{
public:
  IntelMemoryIPCManager() = delete;

  static std::string register_block(PoolBlock * block);

  static std::string register_fd(uint32_t block_id, int fd);

  struct ImportResult
  {
    void * ptr;
    IPCMetadata * ipc_meta;
    int dmabuf_fd{-1};
    uint64_t region_size{0};
  };

  static ImportResult import_dmabuf_block(
    const std::string & socket_path,
    int32_t pid,
    uint32_t block_id,
    uint64_t size,
    uint64_t expected_uid,
    uint32_t pool_id);

  static void release_block(IPCMetadata * meta);

  static ImportResult import_level_zero_block(
    const uint8_t * ipc_handle_data,
    size_t ipc_handle_size,
    int32_t device_ordinal,
    const uint8_t * device_uuid,
    uint64_t size,
    int32_t pid,
    uint32_t block_id,
    uint64_t expected_uid,
    const std::string & l0_socket_path,
    uint32_t pool_id);
};

}  // namespace intel_memory_core

#endif  // INTEL_MEMORY_CORE__INTEL_MEMORY_IPC_MANAGER_HPP_
