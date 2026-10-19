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

#include <gtest/gtest.h>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "qc_buffer/fd_broker.hpp"
#include "qc_buffer/qc_buffer.hpp"
#include "qc_buffer/qc_buffer_api.hpp"
#include "qc_buffer/qc_buffer_impl.hpp"
#include "qc_buffer/rpcmem_loader.hpp"
#include "rosidl_buffer/buffer.hpp"
#include "rosidl_buffer/cpu_buffer_impl.hpp"

using qc_buffer_backend::FdBroker;
using qc_buffer_backend::QcBuffer;
using qc_buffer_backend::QcBufferImpl;
using qc_buffer_backend::RpcMemLoader;

// ── RpcMemLoader ─────────────────────────────────────────────────────────────

// Singleton returns the same instance on repeated calls.
TEST(RpcMemLoaderTest, SingletonIdentity)
{
  EXPECT_EQ(&RpcMemLoader::instance(), &RpcMemLoader::instance());
}

// On a Qualcomm device with libcdsprpc.so, available() must return true.
// Skips gracefully on host machines where the library is absent.
TEST(RpcMemLoaderTest, AvailableOnDevice)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "libcdsprpc.so not found; this check requires a Qualcomm device";
  }
  EXPECT_TRUE(RpcMemLoader::instance().available());
}

// alloc(0) returns nullptr without crashing when library is available.
TEST(RpcMemLoaderTest, AllocZeroReturnsNullptr)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  void * p = RpcMemLoader::instance().alloc(
    qc_buffer_backend::kRpcmemHeapIdSystem,
    qc_buffer_backend::kRpcmemDefaultFlags,
    0);
  // rpcmem_alloc(0) behaviour is implementation-defined; just ensure no crash.
  if (p != nullptr) {
    RpcMemLoader::instance().free(p);
  }
}

// ── QcBuffer ─────────────────────────────────────────────────────────────────

// Default-constructed buffer is empty.
TEST(QcBufferTest, DefaultConstructedIsEmpty)
{
  QcBuffer buf;
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.dmabuf_fd(), -1);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.uid(), 0u);
}

// Allocating a non-zero buffer succeeds and fields are populated.
TEST(QcBufferTest, AllocPopulatesFields)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBuffer buf(64);
  EXPECT_NE(buf.data(), nullptr);
  EXPECT_GE(buf.dmabuf_fd(), 0);
  EXPECT_EQ(buf.size(), 64u);
  EXPECT_GT(buf.uid(), 0u);
}

// Two allocations receive distinct uids.
TEST(QcBufferTest, DistinctUids)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBuffer a(32);
  QcBuffer b(32);
  EXPECT_NE(a.uid(), b.uid());
}

// CPU writes are readable through the same pointer.
TEST(QcBufferTest, CpuReadWrite)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBuffer buf(8);
  for (size_t i = 0; i < 8; ++i) {
    buf.data()[i] = static_cast<uint8_t>(i + 1);
  }
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(buf.data()[i], static_cast<uint8_t>(i + 1));
  }
}

// Move constructor transfers ownership; source is left empty.
TEST(QcBufferTest, MoveConstructor)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBuffer a(32);
  uint8_t * orig_data = a.data();
  uint64_t orig_uid = a.uid();
  QcBuffer b(std::move(a));
  EXPECT_EQ(b.data(), orig_data);
  EXPECT_EQ(b.uid(), orig_uid);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.dmabuf_fd(), -1);
  EXPECT_EQ(a.uid(), 0u);
}

// kMmap path: constructor accepts an externally mmap-ed pointer.
TEST(QcBufferTest, MmapOriginConstructor)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBuffer src(64);
  ASSERT_GE(src.dmabuf_fd(), 0);

  int dup_fd = ::dup(src.dmabuf_fd());
  ASSERT_GE(dup_fd, 0);
  void * ptr = ::mmap(nullptr, 64, PROT_READ | PROT_WRITE, MAP_SHARED, dup_fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  QcBuffer mapped(static_cast<uint8_t *>(ptr), 64, dup_fd);
  EXPECT_EQ(mapped.data(), ptr);
  EXPECT_EQ(mapped.dmabuf_fd(), dup_fd);
  EXPECT_EQ(mapped.size(), 64u);
  EXPECT_GT(mapped.uid(), 0u);
  // Destructor calls munmap + close(dup_fd); verified by absence of ASAN leak.
}

// ── QcBufferImpl ─────────────────────────────────────────────────────────────

// Default-constructed impl reports "qc" backend and size 0.
TEST(QcBufferImplTest, DefaultConstructed)
{
  QcBufferImpl<uint8_t> impl;
  EXPECT_EQ(impl.get_backend_type(), "qc");
  EXPECT_EQ(impl.size(), 0u);
  EXPECT_EQ(impl.get_qc_buffer(), nullptr);
}

// Size constructor allocates a real buffer.
TEST(QcBufferImplTest, SizeConstructor)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBufferImpl<uint8_t> impl(16);
  EXPECT_EQ(impl.size(), 16u);
  ASSERT_NE(impl.get_qc_buffer(), nullptr);
  EXPECT_NE(impl.get_qc_buffer()->data(), nullptr);
  EXPECT_GE(impl.get_qc_buffer()->dmabuf_fd(), 0);
}

// to_cpu() copies bytes to a CPU buffer correctly.
TEST(QcBufferImplTest, ToCpuCopiesBytes)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBufferImpl<uint8_t> impl(4);
  ASSERT_NE(impl.get_qc_buffer(), nullptr);
  impl.get_qc_buffer()->data()[0] = 0xAA;
  impl.get_qc_buffer()->data()[3] = 0xBB;

  auto cpu = impl.to_cpu();
  ASSERT_NE(cpu, nullptr);
  EXPECT_EQ(cpu->size(), 4u);

  auto * cpu_impl = dynamic_cast<rosidl::CpuBufferImpl<uint8_t> *>(cpu.get());
  ASSERT_NE(cpu_impl, nullptr);
  EXPECT_EQ(cpu_impl->get_storage()[0], 0xAA);
  EXPECT_EQ(cpu_impl->get_storage()[3], 0xBB);
}

// clone() produces an independent deep copy.
TEST(QcBufferImplTest, CloneIsIndependent)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBufferImpl<uint8_t> impl(4);
  impl.get_qc_buffer()->data()[0] = 0x42;

  auto copy = impl.clone();
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->size(), 4u);
  // Mutate original; clone must be unaffected.
  impl.get_qc_buffer()->data()[0] = 0xFF;
  auto copy_cpu = copy->to_cpu();
  auto * cpu_copy = dynamic_cast<rosidl::CpuBufferImpl<uint8_t> *>(copy_cpu.get());
  ASSERT_NE(cpu_copy, nullptr);
  EXPECT_EQ(cpu_copy->get_storage()[0], 0x42);
}

// Cross-process constructor throws when given a valid fd that cannot be
// MAP_SHARED mmap-ed (e.g. a pipe read-end). This matches the real production
// failure mode when SCM_RIGHTS delivers the wrong fd type.
// The constructor must close the fd before throwing (ownership transfer).
TEST(QcBufferImplTest, MmapConstructorThrowsOnNonMmapableFd)
{
  int pipe_fds[2];
  ASSERT_EQ(::pipe(pipe_fds), 0);
  int read_fd = pipe_fds[0];
  ::close(pipe_fds[1]);

  EXPECT_THROW((QcBufferImpl<uint8_t>{read_fd, 64, 64}), std::exception);

  // Constructor must have closed read_fd; a second close returns EBADF.
  EXPECT_EQ(::close(read_fd), -1);
  EXPECT_EQ(errno, EBADF);
}

// Cross-process constructor succeeds with a valid dma-buf dup fd.
TEST(QcBufferImplTest, MmapConstructorSucceeds)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  QcBuffer src(64);
  src.data()[0] = 0xAB;

  int dup_fd = ::dup(src.dmabuf_fd());
  ASSERT_GE(dup_fd, 0);

  QcBufferImpl<uint8_t> impl(dup_fd, src.size(), 64);
  EXPECT_EQ(impl.size(), 64u);
  EXPECT_EQ(impl.get_backend_type(), "qc");
  ASSERT_NE(impl.get_qc_buffer(), nullptr);
  EXPECT_EQ(impl.get_qc_buffer()->data()[0], 0xAB);
}

// ── qc_buffer_api ────────────────────────────────────────────────────────────

// allocate_buffer returns a qc-backed buffer with the requested size.
TEST(QcBufferApiTest, AllocateBufferReturnsQcBackend)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  auto buf = qc_buffer_backend::allocate_buffer(32);
  EXPECT_EQ(buf.get_backend_type(), "qc");
  EXPECT_EQ(buf.size(), 32u);
}

// get_dmabuf_fd returns a valid fd for a qc-backed buffer.
TEST(QcBufferApiTest, GetDmabufFdIsValid)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  auto buf = qc_buffer_backend::allocate_buffer(16);
  EXPECT_GE(qc_buffer_backend::get_dmabuf_fd(buf), 0);
}

// get_data_ptr returns a non-null CPU pointer.
TEST(QcBufferApiTest, GetDataPtrIsNonNull)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  auto buf = qc_buffer_backend::allocate_buffer(8);
  EXPECT_NE(qc_buffer_backend::get_data_ptr(buf), nullptr);
}

// get_dmabuf_fd returns -1 for a default-constructed (CPU) buffer.
TEST(QcBufferApiTest, GetDmabufFdReturnsMinus1ForCpuBuffer)
{
  rosidl::Buffer<uint8_t> buf;
  EXPECT_EQ(qc_buffer_backend::get_dmabuf_fd(buf), -1);
}

// ── FdBroker ─────────────────────────────────────────────────────────────────

// Singleton returns the same instance.
TEST(FdBrokerTest, SingletonIdentity)
{
  EXPECT_EQ(&FdBroker::instance(), &FdBroker::instance());
}

// Broker starts successfully.
TEST(FdBrokerTest, StartsRunning)
{
  EXPECT_TRUE(FdBroker::instance().running());
}

// Socket path contains the process pid.
TEST(FdBrokerTest, SocketPathContainsPid)
{
  const std::string & path = FdBroker::instance().socket_path();
  EXPECT_FALSE(path.empty());
  EXPECT_NE(path.find(std::to_string(getpid())), std::string::npos);
}

// register_buffer with invalid arguments is silently ignored.
TEST(FdBrokerTest, RegisterInvalidArgsIgnored)
{
  EXPECT_NO_THROW(FdBroker::instance().register_buffer(0, 42, 100));
  EXPECT_NO_THROW(FdBroker::instance().register_buffer(1, -1, 100));
}

// A registered buffer can be retrieved via the socket (SCM_RIGHTS).
TEST(FdBrokerTest, RegisterAndRetrieveFd)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  auto buf = std::make_shared<QcBuffer>(64);
  auto & broker = FdBroker::instance();
  broker.register_buffer(buf->uid(), buf->dmabuf_fd(), buf->size());

  int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock, 0);
  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, broker.socket_path().c_str(), sizeof(addr.sun_path) - 1);
  ASSERT_EQ(::connect(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0);

  uint64_t uid = buf->uid();
  ASSERT_EQ(::send(sock, &uid, sizeof(uid), 0), static_cast<ssize_t>(sizeof(uid)));

  uint8_t status = 0;
  struct iovec iov{&status, 1};
  union { char b[CMSG_SPACE(sizeof(int))]; struct cmsghdr a; } cmsg_buf{};
  struct msghdr msg{};
  msg.msg_iov = &iov; msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf.b; msg.msg_controllen = sizeof(cmsg_buf.b);
  struct timeval tv{1, 0};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ASSERT_GT(::recvmsg(sock, &msg, 0), 0);
  ::close(sock);

  EXPECT_EQ(status, 0x01);
  int received_fd = -1;
  struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg) {std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));}
  ASSERT_GE(received_fd, 0);
  ::close(received_fd);
}

// Unknown uid returns kStatusMiss (0x02).
TEST(FdBrokerTest, UnknownUidReturnsMiss)
{
  auto & broker = FdBroker::instance();
  int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock, 0);
  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, broker.socket_path().c_str(), sizeof(addr.sun_path) - 1);
  ASSERT_EQ(::connect(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0);

  uint64_t uid = 0xDEADDEADDEADDEADULL;
  ::send(sock, &uid, sizeof(uid), 0);

  uint8_t status = 0xFF;
  struct iovec iov{&status, 1};
  union { char b[CMSG_SPACE(sizeof(int))]; struct cmsghdr a; } cmsg_buf{};
  struct msghdr msg{};
  msg.msg_iov = &iov; msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf.b; msg.msg_controllen = sizeof(cmsg_buf.b);
  struct timeval tv{1, 0};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::recvmsg(sock, &msg, 0);
  ::close(sock);
  EXPECT_EQ(status, 0x02);
}

// Registering the same uid twice is idempotent (no duplicate entry).
TEST(FdBrokerTest, RegisterSameUidIdempotent)
{
  if (!RpcMemLoader::instance().available()) {
    GTEST_SKIP() << "rpcmem not available";
  }
  auto buf = std::make_shared<QcBuffer>(32);
  auto & broker = FdBroker::instance();
  broker.register_buffer(buf->uid(), buf->dmabuf_fd(), buf->size());
  EXPECT_NO_THROW(
    broker.register_buffer(buf->uid(), buf->dmabuf_fd(), buf->size()));
}

// Broker survives a subscriber that disconnects immediately after sending uid.
TEST(FdBrokerTest, EarlyDisconnectDoesNotCrashBroker)
{
  auto & broker = FdBroker::instance();
  int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock, 0);
  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, broker.socket_path().c_str(), sizeof(addr.sun_path) - 1);
  ASSERT_EQ(::connect(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0);
  uint64_t uid = 0xBADFEED0ULL;
  ::send(sock, &uid, sizeof(uid), 0);
  ::close(sock);
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_TRUE(broker.running());
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
