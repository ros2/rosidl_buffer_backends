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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "intel_memory_core/dmabuf_ipc.hpp"

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

TEST(DmaBufIpcReceiveFd, BogusSocketPathThrows)
{
  EXPECT_THROW(DmaBufIpc::receive_fd("/tmp/does_not_exist.sock"), std::runtime_error);
}

TEST(DmaBufIpcUnregisterBlock, UnknownBlockIdIsNoOp)
{
  EXPECT_NO_THROW(DmaBufIpc::unregister_block(0xdeadbeefu));
}
