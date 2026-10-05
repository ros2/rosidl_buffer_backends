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

// DMA-BUF IPC — export and import dma-heap fds via AF_UNIX SCM_RIGHTS
#ifndef INTEL_MEMORY_CORE__DMABUF_IPC_HPP_
#define INTEL_MEMORY_CORE__DMABUF_IPC_HPP_

#include <signal.h>

#include <cerrno>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace intel_memory_core
{
struct PoolBlock;

struct IpcImportResult;

inline uint64_t import_cache_key(uint32_t pool_id, uint32_t block_id)
{
  return (static_cast<uint64_t>(pool_id) << 32) | block_id;
}

inline bool producer_process_alive(int32_t pid)
{
  if (pid <= 0) {
    return true;
  }
  return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
}

class DmaBufIpc
{
public:
  static std::string register_block(PoolBlock * block);

  static IpcImportResult import_block(
    const std::string & socket_path,
    int32_t pid,
    uint32_t block_id,
    uint64_t size,
    uint32_t pool_id);

  static void evict_pool(uint32_t pool_id);

  static std::string register_fd(uint32_t block_id, int fd);
  static int receive_fd(const std::string & socket_path);

  static void unregister_block(uint32_t block_id);
};

}  // namespace intel_memory_core
#endif  // INTEL_MEMORY_CORE__DMABUF_IPC_HPP_
