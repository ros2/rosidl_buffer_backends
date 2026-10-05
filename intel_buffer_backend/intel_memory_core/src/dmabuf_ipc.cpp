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

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/l0_usm_ipc.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "intel_memory_core/logging.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include "intel_memory_core/intel_memory_buffer_pool.hpp"

namespace intel_memory_core
{

static std::mutex s_blocks_mutex;
struct RegisteredBlockInfo {
  int server_socket;
  std::string socket_path;
  std::thread server_thread;
};
static std::unordered_map<uint32_t, RegisteredBlockInfo> s_registered_blocks;

std::string DmaBufIpc::register_fd(uint32_t block_id, int fd)
{
  std::lock_guard<std::mutex> lock(s_blocks_mutex);

  // Check if already registered
  auto it = s_registered_blocks.find(block_id);
  if (it != s_registered_blocks.end()) {
    return it->second.socket_path;
  }

  // Create AF_UNIX socket for fd export
  std::string socket_path = "/tmp/intel_memory_buffer_" +
    std::to_string(getpid()) + "_blk" +
    std::to_string(block_id) + ".sock";

  unlink(socket_path.c_str());

  int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    RCUTILS_LOG_ERROR_NAMED("dmabuf_ipc",
      "socket() failed (errno=%d)", errno);
    return "";
  }

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(sock_fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
    RCUTILS_LOG_ERROR_NAMED("dmabuf_ipc",
      "bind() failed for %s (errno=%d)", socket_path.c_str(), errno);
    close(sock_fd);
    return "";
  }

  if (listen(sock_fd, 16) < 0) {
    close(sock_fd);
    unlink(socket_path.c_str());
    return "";
  }

  std::thread server_thread([sock_fd, shared_fd = fd]() {
    for (;;) {
      int peer = accept(sock_fd, nullptr, nullptr);
      if (peer < 0) {
        break;
      }

      // Send the fd via SCM_RIGHTS
      struct msghdr msg{};
      struct iovec iov{};
      char buf[1] = {0};
      iov.iov_base = buf;
      iov.iov_len = 1;
      msg.msg_iov = &iov;
      msg.msg_iovlen = 1;

      union {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
      } cmsg_buf{};

      msg.msg_control = cmsg_buf.control;
      msg.msg_controllen = sizeof(cmsg_buf.control);

      struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = SCM_RIGHTS;
      cmsg->cmsg_len = CMSG_LEN(sizeof(int));
      // Copy the fd 4-byte(int) into the metadata buffer for transmission
      std::memcpy(CMSG_DATA(cmsg), &shared_fd, sizeof(int));

      sendmsg(peer, &msg, MSG_NOSIGNAL);
      close(peer);
    }
  });

  s_registered_blocks[block_id] =
    RegisteredBlockInfo{sock_fd, socket_path, std::move(server_thread)};
  return socket_path;
}

void DmaBufIpc::unregister_block(uint32_t block_id)
{
  RegisteredBlockInfo info;
  {
    std::lock_guard<std::mutex> lock(s_blocks_mutex);
    auto it = s_registered_blocks.find(block_id);
    if (it == s_registered_blocks.end()) {
      return;
    }
    info = std::move(it->second);
    s_registered_blocks.erase(it);
  }

  shutdown(info.server_socket, SHUT_RDWR);
  if (info.server_thread.joinable()) {
    info.server_thread.join();
  }
  close(info.server_socket);
  unlink(info.socket_path.c_str());
}

std::string DmaBufIpc::register_block(PoolBlock * block)
{
  return register_fd(block->block_id, block->dmabuf_fd);
}

int DmaBufIpc::receive_fd(const std::string & socket_path)
{
  int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    throw std::runtime_error("socket() failed for SCM_RIGHTS import");
  }

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (connect(sock_fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
    close(sock_fd);
    throw std::runtime_error("connect() failed for SCM_RIGHTS import: " + socket_path);
  }

  // Receive fd via SCM_RIGHTS
  struct msghdr msg{};
  struct iovec iov{};
  char buf[1];
  iov.iov_base = buf;
  iov.iov_len = 1;
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } cmsg_buf{};

  msg.msg_control = cmsg_buf.control;
  msg.msg_controllen = sizeof(cmsg_buf.control);

  ssize_t n = recvmsg(sock_fd, &msg, 0);
  close(sock_fd);

  if (n <= 0) {
    throw std::runtime_error("recvmsg() failed for SCM_RIGHTS import");
  }

  struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_type != SCM_RIGHTS) {
    throw std::runtime_error("No SCM_RIGHTS in import message");
  }

  int recv_fd = -1;
  // Copy the received fd from the control message into an integer variable
  std::memcpy(&recv_fd, CMSG_DATA(cmsg), sizeof(int));
  if (recv_fd < 0) {
    throw std::runtime_error("Invalid fd received via SCM_RIGHTS");
  }
  return recv_fd;
}

namespace
{
struct CachedImport
{
  IpcImportResult result;
  uint64_t size{0};   // for munmap
  int32_t pid{0};     // producer process, for the dead-producer reap
};

std::unordered_map<uint64_t, CachedImport> & import_cache()
{
  thread_local std::unordered_map<uint64_t, CachedImport> cache;
  return cache;
}

void drop_import(const CachedImport & entry)
{
  if (entry.result.ptr) {
    munmap(entry.result.ptr, entry.size);
  }
  if (entry.result.dmabuf_fd >= 0) {
    close(entry.result.dmabuf_fd);
  }
}
}  // namespace

void DmaBufIpc::evict_pool(uint32_t pool_id)
{
  auto & cache = import_cache();
  for (auto it = cache.begin(); it != cache.end(); ) {
    if (static_cast<uint32_t>(it->first >> 32) == pool_id) {
      drop_import(it->second);
      it = cache.erase(it);
    } else {
      ++it;
    }
  }
}

IpcImportResult DmaBufIpc::import_block(
  const std::string & socket_path,
  int32_t pid,
  uint32_t block_id,
  uint64_t size,
  uint32_t pool_id)
{
  auto & cache = import_cache();
  const uint64_t key = import_cache_key(pool_id, block_id);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second.result;
  }

  for (auto e = cache.begin(); e != cache.end(); ) {
    if (!producer_process_alive(e->second.pid)) {
      drop_import(e->second);
      e = cache.erase(e);
    } else {
      ++e;
    }
  }

  int dmabuf_fd = -1;
  void * ptr = nullptr;
  try {
    dmabuf_fd = receive_fd(socket_path);
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    if (ptr == MAP_FAILED) {
      close(dmabuf_fd);
      throw std::runtime_error("mmap() failed for imported DMA-BUF");
    }
  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_NAMED("dmabuf_ipc",
      "dmabuf import failed (producer gone or stale socket): %s", e.what());
    return IpcImportResult{};
  }

  IpcImportResult result{ptr, dmabuf_fd};
  cache[key] = CachedImport{result, size, pid};
  return result;
}

}  // namespace intel_memory_core