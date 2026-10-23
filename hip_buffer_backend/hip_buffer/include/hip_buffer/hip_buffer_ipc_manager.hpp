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

#ifndef HIP_BUFFER__HIP_BUFFER_IPC_MANAGER_HPP_
#define HIP_BUFFER__HIP_BUFFER_IPC_MANAGER_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace hip_buffer_backend
{

struct VmmBlock;
struct IPCMetadata;

/// \brief VMM IPC transport helper used by HipMemoryPool and the backend plugin.
///
/// On the publisher side, HipMemoryPool registers VMM blocks here so their
/// exported file descriptors can be served over Unix-domain sockets. On the
/// subscriber side, import_block() receives the FD, maps the VMM block into
/// the local process, atomically acquires the open block generation,
/// and caches imports by (publisher pid, block id).
class HipVmmIPCManager
{
public:
  HipVmmIPCManager();
  ~HipVmmIPCManager();

  HipVmmIPCManager(const HipVmmIPCManager &) = delete;
  HipVmmIPCManager & operator=(const HipVmmIPCManager &) = delete;
  HipVmmIPCManager(HipVmmIPCManager &&) = delete;
  HipVmmIPCManager & operator=(HipVmmIPCManager &&) = delete;

  std::string register_block(VmmBlock * block);

  struct ImportResult
  {
    void * va;
    IPCMetadata * ipc_meta;
  };

  static ImportResult import_block(
    const std::string & socket_path,
    int32_t pid,
    uint32_t block_id,
    uint64_t size,
    uint64_t expected_uid);

private:
  static int create_fd_server_socket(const std::string & socket_path);
  static int receive_fd_from_socket(const std::string & socket_path);

  struct FDDispatcher;
  static std::shared_ptr<FDDispatcher> get_dispatcher();

  std::shared_ptr<FDDispatcher> dispatcher_;

  struct RegisteredBlockInfo
  {
    int server_socket;
    std::string socket_path;
  };
  std::unordered_map<uint32_t, RegisteredBlockInfo> registered_blocks_;
  std::mutex blocks_mutex_;
};

}  // namespace hip_buffer_backend

#endif  // HIP_BUFFER__HIP_BUFFER_IPC_MANAGER_HPP_
