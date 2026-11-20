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

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/l0_usm_ipc.hpp"

using intel_memory_core::DmaBufIpc;
using intel_memory_core::import_cache_key;
using intel_memory_core::producer_process_alive;

TEST(ImportCacheKey, BitPackRoundTrip)
{
  uint64_t key = import_cache_key(0x12345678u, 0x9abcdef0u);
  EXPECT_EQ(key, (static_cast<uint64_t>(0x12345678u) << 32) | 0x9abcdef0u);
  EXPECT_EQ(import_cache_key(0, 0), 0u);
}

TEST(ImportCacheKey, DistinctPoolBlockPairsDoNotCollide)
{
  EXPECT_NE(import_cache_key(1, 2), import_cache_key(2, 1));
}

TEST(ProducerProcessAlive, ZeroOrNegativePidAssumedAlive)
{
  EXPECT_TRUE(producer_process_alive(0));
  EXPECT_TRUE(producer_process_alive(-1));
}

TEST(ProducerProcessAlive, SelfPidIsAlive)
{
  EXPECT_TRUE(producer_process_alive(static_cast<int32_t>(getpid())));
}

TEST(ProducerProcessAlive, ReapedChildPidIsDead)
{
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  EXPECT_FALSE(producer_process_alive(static_cast<int32_t>(child)));
}

TEST(DmaBufIpcRegisterFd, RoundTripOverScmRights)
{
  int fd = memfd_create("test_dmabuf_ipc", 0);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, 4096), 0);
  const char payload[] = "hello-scm-rights";
  ASSERT_EQ(write(fd, payload, sizeof(payload)), static_cast<ssize_t>(sizeof(payload)));

  std::string socket_path = DmaBufIpc::register_fd(12345u, fd);
  ASSERT_FALSE(socket_path.empty());

  int received = DmaBufIpc::receive_fd(socket_path);
  ASSERT_GE(received, 0);

  char readback[sizeof(payload)] = {};
  ASSERT_EQ(lseek(received, 0, SEEK_SET), 0);
  ASSERT_EQ(read(received, readback, sizeof(readback)), static_cast<ssize_t>(sizeof(readback)));
  EXPECT_STREQ(readback, payload);

  close(received);
  DmaBufIpc::unregister_block(12345u);
  close(fd);
}

TEST(DmaBufIpcRegisterFd, RepeatedRegisterReturnsSameSocket)
{
  int fd = memfd_create("test_dmabuf_ipc_repeat", 0);
  ASSERT_GE(fd, 0);

  std::string first = DmaBufIpc::register_fd(999u, fd);
  std::string second = DmaBufIpc::register_fd(999u, fd);
  ASSERT_FALSE(first.empty());
  EXPECT_EQ(first, second);

  DmaBufIpc::unregister_block(999u);
  close(fd);
}

TEST(DmaBufIpcRegisterFd, StillRegisteredAtProcessExitExitsCleanly)
{
  // e.g. wrap_dmabuf() camera buffers, which no pool ever unregisters.
  pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    int fd = memfd_create("test_dmabuf_ipc_exit", 0);
    if (fd < 0 || DmaBufIpc::register_fd(4242u, fd).empty()) {
      _exit(2);
    }
    exit(0);  // runs static destructors, unlike _exit()
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "child terminated by signal " << WTERMSIG(status);
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(DmaBufIpcBufferSize, ReportsTheReceivedBuffersSize)
{
  int fd = memfd_create("test_dmabuf_ipc_size", 0);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, 3 * 4096), 0);
  EXPECT_EQ(DmaBufIpc::buffer_size(fd), 3 * 4096);
  close(fd);
  EXPECT_EQ(DmaBufIpc::buffer_size(-1), -1);
}

TEST(DmaBufIpcImportBlock, RequestLargerThanExportedBufferIsRejected)
{
  int fd = memfd_create("test_dmabuf_ipc_import_size", 0);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, 4096), 0);
  const std::string socket_path = DmaBufIpc::register_fd(777u, fd);
  ASSERT_FALSE(socket_path.empty());
  const int32_t self = static_cast<int32_t>(getpid());

  // Mapping 16 KiB of a 4 KiB buffer would SIGBUS on access past 4 KiB.
  EXPECT_EQ(DmaBufIpc::import_block(socket_path, self, 777u, 4 * 4096, 0x5e1fu).ptr, nullptr);
  auto ok = DmaBufIpc::import_block(socket_path, self, 777u, 4096, 0x5e1fu);
  EXPECT_NE(ok.ptr, nullptr);
  // Served from the import cache: still bounded by what was mapped.
  EXPECT_EQ(DmaBufIpc::import_block(socket_path, self, 777u, 2 * 4096, 0x5e1fu).ptr, nullptr);

  DmaBufIpc::evict_pool(0x5e1fu);
  DmaBufIpc::unregister_block(777u);
  close(fd);
}

// --- receive_fd against a peer that sends arbitrary control messages --------
// The socket path comes from an untrusted descriptor, so whoever serves it
// decides what arrives: receive_fd must accept exactly one fd and never leave
// other received descriptors open in this process.

namespace
{
size_t open_fd_count()
{
  size_t n = 0;
  if (DIR * dir = opendir("/proc/self/fd")) {
    while (readdir(dir) != nullptr) {
      ++n;
    }
    closedir(dir);
  }
  return n;
}

// Serves one connection on `path`, sending `fds` in a single SCM_RIGHTS
// message (no control message at all if empty).
class FakeFdServer
{
public:
  FakeFdServer(const std::string & path, std::vector<int> fds)
  : path_(path)
  {
    unlink(path_.c_str());
    listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);
    ok_ = listen_fd_ >= 0 &&
      bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0 &&
      listen(listen_fd_, 1) == 0;
    thread_ = std::thread([this, fds]() {
        int peer = accept(listen_fd_, nullptr, nullptr);
        if (peer < 0) {
          return;
        }
        char byte = 0;
        iovec iov{&byte, 1};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        std::vector<char> control(CMSG_SPACE(sizeof(int) * fds.size()));
        if (!fds.empty()) {
          msg.msg_control = control.data();
          msg.msg_controllen = control.size();
          cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
          cmsg->cmsg_level = SOL_SOCKET;
          cmsg->cmsg_type = SCM_RIGHTS;
          cmsg->cmsg_len = CMSG_LEN(sizeof(int) * fds.size());
          std::memcpy(CMSG_DATA(cmsg), fds.data(), sizeof(int) * fds.size());
        }
        (void)sendmsg(peer, &msg, MSG_NOSIGNAL);
        close(peer);
      });
  }
  ~FakeFdServer()
  {
    if (thread_.joinable()) {
      if (!ok_) {
        shutdown(listen_fd_, SHUT_RDWR);
      }
      thread_.join();
    }
    close(listen_fd_);
    unlink(path_.c_str());
  }
  void wait() {thread_.join();}
  bool ok() const {return ok_;}

private:
  std::string path_;
  int listen_fd_{-1};
  bool ok_{false};
  std::thread thread_;
};

std::string fake_socket_path(const char * tag)
{
  return "/tmp/ibb_test_recvfd_" + std::to_string(getpid()) + "_" + tag + ".sock";
}
}  // namespace

TEST(DmaBufIpcReceiveFd, SingleFdIsReturnedCloseOnExec)
{
  int fd = memfd_create("test_recv_single", 0);
  ASSERT_GE(fd, 0);
  FakeFdServer server(fake_socket_path("single"), {fd});
  ASSERT_TRUE(server.ok());

  int received = DmaBufIpc::receive_fd(fake_socket_path("single"));
  server.wait();
  ASSERT_GE(received, 0);
  EXPECT_TRUE(fcntl(received, F_GETFD) & FD_CLOEXEC)
    << "received buffer fd would leak into exec'd child processes";
  close(received);
  close(fd);
}

TEST(DmaBufIpcReceiveFd, ExtraFdsAreRejectedWithoutLeaking)
{
  // Two fds fit the receive buffer and are both installed by the kernel;
  // three or more truncate the control data. Either way: reject, leak nothing.
  for (size_t count : {2u, 3u}) {
    std::vector<int> fds;
    for (size_t i = 0; i < count; ++i) {
      fds.push_back(memfd_create("test_recv_extra", 0));
      ASSERT_GE(fds.back(), 0);
    }
    const std::string path = fake_socket_path("extra");
    const size_t before = open_fd_count();
    {
      FakeFdServer server(path, fds);
      ASSERT_TRUE(server.ok());
      int received = -1;
      EXPECT_THROW(received = DmaBufIpc::receive_fd(path), std::runtime_error)
        << count << " fds accepted";
      server.wait();
      if (received >= 0) {
        close(received);
      }
    }
    EXPECT_EQ(open_fd_count(), before) << count << " fds sent: descriptors leaked";
    for (int fd : fds) {
      close(fd);
    }
  }
}

TEST(DmaBufIpcReceiveFd, MessageWithoutFdThrows)
{
  FakeFdServer server(fake_socket_path("none"), {});
  ASSERT_TRUE(server.ok());
  EXPECT_THROW(DmaBufIpc::receive_fd(fake_socket_path("none")), std::runtime_error);
}

TEST(DmaBufIpcReceiveFd, BogusSocketPathThrows)
{
  EXPECT_THROW(DmaBufIpc::receive_fd("/tmp/does_not_exist.sock"), std::runtime_error);
}

TEST(DmaBufIpcUnregisterBlock, UnknownBlockIdIsNoOp)
{
  EXPECT_NO_THROW(DmaBufIpc::unregister_block(0xdeadbeefu));
}
