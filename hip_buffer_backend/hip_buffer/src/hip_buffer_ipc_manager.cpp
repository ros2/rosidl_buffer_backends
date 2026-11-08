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

#include "hip_buffer/hip_buffer_ipc_manager.hpp"

#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <rcutils/logging_macros.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <thread>

#include "hip_buffer/hip_error.hpp"
#include "hip_buffer/hip_memory_pool.hpp"

namespace hip_buffer_backend
{

namespace
{

void acquire_ipc_lease(
  IPCMetadata * meta,
  uint32_t block_id,
  int32_t pid,
  uint64_t expected_uid)
{
  if (!meta || expected_uid == 0) {
    throw HipError("IPC block is missing generation metadata");
  }

  uint64_t current = meta->uid.load(std::memory_order_seq_cst);
  if (current != expected_uid) {
    RCUTILS_LOG_WARN_NAMED("hip_ipc",
      "Stale or closed block %u from pid %d: descriptor uid %lu, current uid %lu; dropping",
      block_id, pid, expected_uid, current);
    throw HipError("IPC block generation is stale or closed");
  }

  meta->refcount.fetch_add(1, std::memory_order_seq_cst);
  current = meta->uid.load(std::memory_order_seq_cst);
  if (current != expected_uid) {
    meta->refcount.fetch_sub(1, std::memory_order_seq_cst);
    RCUTILS_LOG_WARN_NAMED("hip_ipc",
      "Stale or closed block %u from pid %d: descriptor uid %lu, current uid %lu; dropping",
      block_id, pid, expected_uid, current);
    throw HipError("IPC block recycled or closed before import");
  }
}

struct UnixAddress
{
  struct sockaddr_un addr;
  socklen_t length;
};

UnixAddress make_abstract_unix_addr(const std::string & name)
{
  UnixAddress address{};
  if (name.empty() || name.size() > sizeof(address.addr.sun_path) - 1) {
    throw HipError("Invalid abstract Unix socket name");
  }

  address.addr.sun_family = AF_UNIX;
  memcpy(address.addr.sun_path + 1, name.data(), name.size());
  address.length = static_cast<socklen_t>(
    offsetof(sockaddr_un, sun_path) + 1 + name.size());
  return address;
}

}  // namespace

struct ImportCacheKey
{
  int32_t pid;
  uint32_t block_id;

  bool operator==(const ImportCacheKey & other) const
  {
    return pid == other.pid && block_id == other.block_id;
  }
};

struct ImportCacheKeyHash
{
  size_t operator()(const ImportCacheKey & key) const
  {
    return std::hash<int64_t>()(
      (static_cast<int64_t>(key.pid) << 32) | key.block_id);
  }
};

struct CachedImport
{
  hipMemGenericAllocationHandle_t handle{};
  void * va{nullptr};
  uint64_t size{0};
  IPCMetadata * ipc_meta{nullptr};
  ~CachedImport();
  CachedImport() = default;
  CachedImport(const CachedImport &) = delete;
  CachedImport & operator=(const CachedImport &) = delete;
  CachedImport(CachedImport && other) noexcept;
  CachedImport & operator=(CachedImport && other) noexcept;
};

static std::unordered_map<ImportCacheKey, CachedImport, ImportCacheKeyHash> &
get_import_cache()
{
  static auto * cache =
    new std::unordered_map<ImportCacheKey, CachedImport, ImportCacheKeyHash>();
  return *cache;
}

static std::mutex & get_import_cache_mutex()
{
  static auto * mtx = new std::mutex();
  return *mtx;
}

CachedImport::~CachedImport()
{
  if (ipc_meta != nullptr) {
    munmap(ipc_meta, sizeof(IPCMetadata));
  }
  if (va != nullptr) {
    hipMemUnmap(va, size);
    hipMemAddressFree(va, size);
  }
  if (handle != 0) {
    hipMemRelease(handle);
  }
}

CachedImport::CachedImport(CachedImport && other) noexcept
: handle(other.handle), va(other.va), size(other.size),
  ipc_meta(other.ipc_meta)
{
  other.handle = {};
  other.va = nullptr;
  other.size = 0;
  other.ipc_meta = nullptr;
}

CachedImport & CachedImport::operator=(CachedImport && other) noexcept
{
  if (this != &other) {
    if (ipc_meta != nullptr) {munmap(ipc_meta, sizeof(IPCMetadata));}
    if (va != nullptr) {hipMemUnmap(va, size); hipMemAddressFree(va, size);}
    if (handle != 0) {hipMemRelease(handle);}
    handle = other.handle;
    va = other.va;
    size = other.size;
    ipc_meta = other.ipc_meta;
    other.handle = {};
    other.va = nullptr;
    other.size = 0;
    other.ipc_meta = nullptr;
  }
  return *this;
}

struct HipVmmIPCManager::FDDispatcher
{
  FDDispatcher();
  ~FDDispatcher();

  FDDispatcher(const FDDispatcher &) = delete;
  FDDispatcher & operator=(const FDDispatcher &) = delete;

  void add_socket(int server_socket, int fd_to_serve);
  void remove_socket(int server_socket);
  void stop();

private:
  void run();
  void handle_client(int server_socket, int fd_to_serve);

  int epoll_fd_{-1};
  int event_fd_{-1};
  std::unordered_map<int, int> socket_to_fd_;
  std::mutex map_mutex_;
  std::thread dispatcher_thread_;
  std::atomic<bool> running_{false};
};

HipVmmIPCManager::FDDispatcher::FDDispatcher()
{
  epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
  if (epoll_fd_ < 0) {
    return;
  }

  event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (event_fd_ < 0) {
    close(epoll_fd_);
    epoll_fd_ = -1;
    return;
  }

  struct epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.fd = event_fd_;
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, event_fd_, &ev) < 0) {
    close(event_fd_);
    close(epoll_fd_);
    epoll_fd_ = -1;
    event_fd_ = -1;
    return;
  }

  running_.store(true);
  dispatcher_thread_ = std::thread(&FDDispatcher::run, this);
}

HipVmmIPCManager::FDDispatcher::~FDDispatcher()
{
  stop();
}

void HipVmmIPCManager::FDDispatcher::stop()
{
  if (!running_.load()) {
    return;
  }

  running_.store(false);

  if (event_fd_ >= 0) {
    uint64_t val = 1;
    [[maybe_unused]] auto ret = write(event_fd_, &val, sizeof(val));
  }

  if (dispatcher_thread_.joinable()) {
    dispatcher_thread_.join();
  }

  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    for (auto & [sock, fd] : socket_to_fd_) {
      close(sock);
    }
    socket_to_fd_.clear();
  }

  if (event_fd_ >= 0) {
    close(event_fd_);
    event_fd_ = -1;
  }
  if (epoll_fd_ >= 0) {
    close(epoll_fd_);
    epoll_fd_ = -1;
  }
}

void HipVmmIPCManager::FDDispatcher::add_socket(int server_socket, int fd_to_serve)
{
  int duped_fd = dup(fd_to_serve);
  if (duped_fd < 0) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    socket_to_fd_[server_socket] = duped_fd;
  }

  struct epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.fd = server_socket;
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, server_socket, &ev) < 0) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    socket_to_fd_.erase(server_socket);
    close(duped_fd);
  }
}

void HipVmmIPCManager::FDDispatcher::remove_socket(int server_socket)
{
  int duped_fd = -1;
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = socket_to_fd_.find(server_socket);
    if (it != socket_to_fd_.end()) {
      duped_fd = it->second;
      socket_to_fd_.erase(it);
    }
  }

  epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, server_socket, nullptr);
  close(server_socket);

  if (duped_fd >= 0) {
    close(duped_fd);
  }
}

void HipVmmIPCManager::FDDispatcher::run()
{
  constexpr int MAX_EVENTS = 16;
  struct epoll_event events[MAX_EVENTS];

  while (running_.load()) {
    int n = epoll_wait(epoll_fd_, events, MAX_EVENTS, 1000);

    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }

    for (int i = 0; i < n; i++) {
      int fd = events[i].data.fd;

      if (fd == event_fd_) {
        uint64_t val;
        (void)read(event_fd_, &val, sizeof(val));
        continue;
      }

      int fd_to_serve = -1;
      {
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = socket_to_fd_.find(fd);
        if (it != socket_to_fd_.end()) {
          fd_to_serve = it->second;
        }
      }

      if (fd_to_serve >= 0) {
        handle_client(fd, fd_to_serve);
      }
    }
  }
}

void HipVmmIPCManager::FDDispatcher::handle_client(int server_socket, int fd_to_serve)
{
  int client_socket = accept(server_socket, nullptr, nullptr);
  if (client_socket < 0) {
    return;
  }

  struct msghdr msg;
  struct iovec iov[1];
  char ctrl_buf[CMSG_SPACE(sizeof(int))];
  char data[1] = {'X'};

  memset(&msg, 0, sizeof(msg));
  memset(ctrl_buf, 0, sizeof(ctrl_buf));

  iov[0].iov_base = data;
  iov[0].iov_len = sizeof(data);

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl_buf;
  msg.msg_controllen = sizeof(ctrl_buf);

  struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &fd_to_serve, sizeof(int));

  if (sendmsg(client_socket, &msg, 0) < 0) {
    RCUTILS_LOG_WARN_NAMED(
      "hip_buffer_backend",
      "FDDispatcher: sendmsg failed (errno=%d): FD will not be delivered "
      "to client; subscriber import will fall back to CPU", errno);
  }

  close(client_socket);
}

std::shared_ptr<HipVmmIPCManager::FDDispatcher>
HipVmmIPCManager::get_dispatcher()
{
  // Process-lifetime FD dispatcher. It owns a worker thread and should not
  // be torn down during static destruction while subscribers may still be
  // importing VMM handles.
  static auto * dispatcher = new std::shared_ptr<FDDispatcher>(
    new FDDispatcher());
  return *dispatcher;
}

HipVmmIPCManager::HipVmmIPCManager()
: dispatcher_(get_dispatcher())
{
}

HipVmmIPCManager::~HipVmmIPCManager()
{
  std::lock_guard<std::mutex> lock(blocks_mutex_);
  for (auto & [id, info] : registered_blocks_) {
    if (info.server_socket >= 0 && dispatcher_) {
      dispatcher_->remove_socket(info.server_socket);
    }
  }
  registered_blocks_.clear();
}

std::string HipVmmIPCManager::register_block(VmmBlock * block)
{
  std::lock_guard<std::mutex> lock(blocks_mutex_);

  auto it = registered_blocks_.find(block->block_id);
  if (it != registered_blocks_.end()) {
    return it->second.socket_path;
  }

  std::stringstream ss;
  ss << "hip_vmm_" << getpid() << "_" << block->block_id;
  std::string socket_path = ss.str();

  int server_socket = create_fd_server_socket(socket_path);
  if (server_socket < 0) {
    throw HipError("Failed to create FD server socket for block");
  }

  dispatcher_->add_socket(server_socket, block->exported_fd);

  RegisteredBlockInfo info;
  info.server_socket = server_socket;
  info.socket_path = socket_path;
  registered_blocks_[block->block_id] = info;

  return socket_path;
}

HipVmmIPCManager::ImportResult HipVmmIPCManager::import_block(
  const std::string & socket_path,
  int32_t pid,
  uint32_t block_id,
  uint64_t size,
  uint64_t expected_uid)
{
  ImportCacheKey cache_key{pid, block_id};

  auto & cache = get_import_cache();
  std::lock_guard<std::mutex> lock(get_import_cache_mutex());

  auto it = cache.find(cache_key);
  if (it != cache.end()) {
    IPCMetadata * meta = it->second.ipc_meta;
    acquire_ipc_lease(meta, block_id, pid, expected_uid);
    return {it->second.va, meta};
  }

  IPCMetadata * ipc_meta = nullptr;
  std::string shm_name = "/hip_vmm_" + std::to_string(pid) +
    "_" + std::to_string(block_id);
  int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (shm_fd >= 0) {
    void * ptr = mmap(nullptr, sizeof(IPCMetadata),
      PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);
    if (ptr != MAP_FAILED) {
      ipc_meta = static_cast<IPCMetadata *>(ptr);
    }
  }

  try {
    acquire_ipc_lease(ipc_meta, block_id, pid, expected_uid);
  } catch (...) {
    if (ipc_meta) {
      munmap(ipc_meta, sizeof(IPCMetadata));
    }
    throw;
  }

  hipMemGenericAllocationHandle_t handle{};
  void * va = nullptr;
  bool handle_imported = false;
  bool address_reserved = false;
  bool mapped = false;

  auto cleanup_failed_import = [&]() {
      if (mapped) {
        hipMemUnmap(va, size);
      }
      if (address_reserved) {
        hipMemAddressFree(va, size);
      }
      if (handle_imported) {
        hipMemRelease(handle);
      }
      if (ipc_meta) {
        ipc_meta->refcount.fetch_sub(1, std::memory_order_release);
        munmap(ipc_meta, sizeof(IPCMetadata));
      }
    };

  int fd = receive_fd_from_socket(socket_path);
  if (fd < 0) {
    cleanup_failed_import();
    throw HipError("Failed to receive VMM FD from socket: " + socket_path);
  }

  hipError_t r = hipMemImportFromShareableHandle(
    &handle,
    reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
    hipMemHandleTypePosixFileDescriptor);

  close(fd);

  if (r != hipSuccess) {
    cleanup_failed_import();
    throw HipError(__FILE__, __LINE__, "hipMemImportFromShareableHandle", r);
  }
  handle_imported = true;

  r = hipMemAddressReserve(&va, size, 0, nullptr, 0);
  if (r != hipSuccess) {
    cleanup_failed_import();
    throw HipError(__FILE__, __LINE__, "hipMemAddressReserve", r);
  }
  address_reserved = true;

  r = hipMemMap(va, size, 0, handle, 0);
  if (r != hipSuccess) {
    cleanup_failed_import();
    throw HipError(__FILE__, __LINE__, "hipMemMap", r);
  }
  mapped = true;

  int device_id = 0;
  hipError_t rt = hipGetDevice(&device_id);
  if (rt != hipSuccess) {
    cleanup_failed_import();
    throw HipError(__FILE__, __LINE__, "hipGetDevice", rt);
  }
  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = device_id;
  access.flags = hipMemAccessFlagsProtReadWrite;
  r = hipMemSetAccess(va, size, &access, 1);
  if (r != hipSuccess) {
    cleanup_failed_import();
    throw HipError(__FILE__, __LINE__, "hipMemSetAccess", r);
  }

  auto & cached = cache[cache_key];
  cached.handle = handle;
  cached.va = va;
  cached.size = size;
  cached.ipc_meta = ipc_meta;

  return {va, ipc_meta};
}

int HipVmmIPCManager::create_fd_server_socket(const std::string & socket_path)
{
  int server_socket = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (server_socket < 0) {
    return -1;
  }

  auto address = make_abstract_unix_addr(socket_path);
  if (bind(
      server_socket, reinterpret_cast<struct sockaddr *>(&address.addr), address.length) < 0)
  {
    close(server_socket);
    return -1;
  }

  if (listen(server_socket, 5) < 0) {
    close(server_socket);
    return -1;
  }

  return server_socket;
}

int HipVmmIPCManager::receive_fd_from_socket(const std::string & socket_path)
{
  int client_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (client_socket < 0) {
    return -1;
  }

  auto address = make_abstract_unix_addr(socket_path);
  if (connect(
      client_socket, reinterpret_cast<struct sockaddr *>(&address.addr), address.length) < 0)
  {
    close(client_socket);
    return -1;
  }

  struct msghdr msg;
  struct iovec iov[1];
  char ctrl_buf[CMSG_SPACE(sizeof(int))];
  char data[1];

  memset(&msg, 0, sizeof(msg));
  memset(ctrl_buf, 0, sizeof(ctrl_buf));

  iov[0].iov_base = data;
  iov[0].iov_len = sizeof(data);

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl_buf;
  msg.msg_controllen = sizeof(ctrl_buf);

  ssize_t n = recvmsg(client_socket, &msg, 0);
  const int recvmsg_errno = errno;
  close(client_socket);

  if (n <= 0) {
    RCUTILS_LOG_WARN_NAMED(
      "hip_buffer_backend",
      "recvmsg on '%s' returned %zd (errno=%d): VMM FD not delivered",
      socket_path.c_str(), n, n < 0 ? recvmsg_errno : 0);
    return -1;
  }

  struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg == nullptr || cmsg->cmsg_level != SOL_SOCKET ||
    cmsg->cmsg_type != SCM_RIGHTS)
  {
    return -1;
  }

  int received_fd;
  memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));

  return received_fd;
}

}  // namespace hip_buffer_backend
