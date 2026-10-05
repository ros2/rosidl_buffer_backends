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

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "intel_memory_core/dmabuf_ipc.hpp"
#include "intel_memory_core/intel_memory_ipc_manager.hpp"

using intel_memory_core::DmaBufIpc;
using intel_memory_core::IntelMemoryIPCManager;

TEST(IntelMemoryIpcManager, ImportDmabufBlockWithBogusInputsNeverThrows)
{
  IntelMemoryIPCManager::ImportResult result;
  ASSERT_NO_THROW(
    result = IntelMemoryIPCManager::import_dmabuf_block(
      "/tmp/does_not_exist.sock", 999999, 42, 4096, 0, 1));
  EXPECT_EQ(result.ptr, nullptr);
  EXPECT_EQ(result.dmabuf_fd, -1);
}

TEST(IntelMemoryIpcManager, ImportLevelZeroBlockWithBogusInputsNeverThrows)
{
  uint8_t fake_handle[8] = {};
  uint8_t fake_uuid[16] = {};
  IntelMemoryIPCManager::ImportResult result;
  ASSERT_NO_THROW(
    result = IntelMemoryIPCManager::import_level_zero_block(
      fake_handle, sizeof(fake_handle),
      -1, fake_uuid, 4096,
      999999, 42, 0,
      "/tmp/does_not_exist_l0.sock", 1));
  EXPECT_EQ(result.ptr, nullptr);
}

TEST(IntelMemoryIpcManager, ImportLevelZeroBlockWithNullUuidFallsBackToOrdinal)
{
  uint8_t fake_handle[8] = {};
  IntelMemoryIPCManager::ImportResult result;
  ASSERT_NO_THROW(
    result = IntelMemoryIPCManager::import_level_zero_block(
      fake_handle, sizeof(fake_handle),
      -1, nullptr, 4096,
      999999, 42, 0,
      "", 1));
  EXPECT_EQ(result.ptr, nullptr);
}

TEST(IntelMemoryIpcManager, ReleaseBlockWithNullMetaIsNoOp)
{
  EXPECT_NO_THROW(IntelMemoryIPCManager::release_block(nullptr));
}

TEST(IntelMemoryIpcManager, RegisterFdDelegatesToDmaBufIpc)
{
  std::string path;
  ASSERT_NO_THROW(path = IntelMemoryIPCManager::register_fd(0xabcdu, -1));
  EXPECT_FALSE(path.empty());
  DmaBufIpc::unregister_block(0xabcdu);
}
