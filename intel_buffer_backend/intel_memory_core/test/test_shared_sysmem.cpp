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

#include <unistd.h>

#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "intel_memory_core/l0_external_memmap.hpp"

using intel_memory_core::SharedSysmem;

namespace
{
std::string unique_name(const char * suffix)
{
  return "/imc_test_" + std::to_string(getpid()) + "_" + suffix;
}
}  // namespace

TEST(SharedSysmem, CreateRoundsUpToPageSize)
{
  SharedSysmem shm;
  const std::string name = unique_name("create_round");
  ASSERT_TRUE(shm.create(name, 1));
  long page_size = sysconf(_SC_PAGESIZE);
  EXPECT_EQ(shm.size() % static_cast<size_t>(page_size), 0u);
  EXPECT_GE(shm.size(), 1u);
  EXPECT_NE(shm.ptr(), nullptr);
  EXPECT_EQ(shm.name(), name);
}

TEST(SharedSysmem, WriteThenOpenFromPeerSeesSameBytes)
{
  const std::string name = unique_name("peer_view");
  SharedSysmem owner;
  ASSERT_TRUE(owner.create(name, 4096));
  std::memset(owner.ptr(), 0xAB, owner.size());

  SharedSysmem peer;
  ASSERT_TRUE(peer.open(name));
  EXPECT_EQ(peer.size(), owner.size());
  EXPECT_EQ(std::memcmp(owner.ptr(), peer.ptr(), owner.size()), 0);
}

TEST(SharedSysmem, DoubleCloseIsSafe)
{
  SharedSysmem shm;
  ASSERT_TRUE(shm.create(unique_name("double_close"), 4096));
  shm.close();
  EXPECT_NO_THROW(shm.close());
  EXPECT_EQ(shm.ptr(), nullptr);
}

TEST(SharedSysmem, OwnerUnlinksRegionOnClose)
{
  const std::string name = unique_name("owner_unlink");
  SharedSysmem owner;
  ASSERT_TRUE(owner.create(name, 4096));
  owner.close();

  SharedSysmem peer;
  EXPECT_FALSE(peer.open(name));
}

TEST(SharedSysmem, PeerCloseDoesNotUnlink)
{
  const std::string name = unique_name("peer_no_unlink");
  SharedSysmem owner;
  ASSERT_TRUE(owner.create(name, 4096));

  SharedSysmem peer;
  ASSERT_TRUE(peer.open(name));
  peer.close();

  SharedSysmem peer2;
  EXPECT_TRUE(peer2.open(name));

  owner.close();
}

TEST(SharedSysmem, OpenNonexistentNameFails)
{
  SharedSysmem shm;
  EXPECT_FALSE(shm.open(unique_name("never_created")));
}
