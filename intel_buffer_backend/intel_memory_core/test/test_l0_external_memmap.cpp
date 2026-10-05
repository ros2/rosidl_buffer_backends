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

#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "intel_memory_core/l0_external_memmap.hpp"

using intel_memory_core::ExternalMemMapImporter;
using intel_memory_core::L0ExternalMemMap;
using intel_memory_core::SharedSysmem;

TEST(L0ExternalMemMap, SupportedIsQueryable)
{
  (void)L0ExternalMemMap::supported();
  SUCCEED();
}

TEST(L0ExternalMemMap, MapUnmapRoundTripOrUnsupportedFailsCleanly)
{
  SharedSysmem shm;
  ASSERT_TRUE(shm.create(
      "/imc_test_" + std::to_string(getpid()) + "_l0_memmap", 4096));
  std::memset(shm.ptr(), 0x5A, shm.size());

  if (!L0ExternalMemMap::supported()) {
    GTEST_SKIP() << "ZE_extension_external_memmap_sysmem not advertised by "
                    "the running Level Zero driver on this machine";
  }

  void * usm_ptr = L0ExternalMemMap::map(shm.ptr(), shm.size());
  ASSERT_NE(usm_ptr, nullptr);
  EXPECT_EQ(usm_ptr, shm.ptr());
  EXPECT_EQ(static_cast<uint8_t *>(usm_ptr)[0], 0x5A);

  L0ExternalMemMap::unmap(usm_ptr);
}

TEST(L0ExternalMemMap, UnmapNullptrIsNoOp)
{
  EXPECT_NO_THROW(L0ExternalMemMap::unmap(nullptr));
}

TEST(ExternalMemMapImporter, ImportUnknownRegionFailsCleanly)
{
  void * ptr = ExternalMemMapImporter::import(
    "/imc_test_never_created_region", 4096, 0);
  EXPECT_EQ(ptr, nullptr);
}

TEST(ExternalMemMapImporter, ImportReturnsRawShmPointerSeeingProducerWrites)
{
  const std::string name = "/imc_test_import_alias_" + std::to_string(getpid());
  SharedSysmem owner;
  ASSERT_TRUE(owner.create(name, 4096));
  std::memset(owner.ptr(), 0x3C, owner.size());

  void * imported = ExternalMemMapImporter::import(
    name, 4096, static_cast<int32_t>(getpid()));
  ASSERT_NE(imported, nullptr);
  EXPECT_EQ(static_cast<uint8_t *>(imported)[0], 0x3C);
  EXPECT_EQ(static_cast<uint8_t *>(imported)[4095], 0x3C);
}

TEST(ExternalMemMapImporter, OutRegionSizeIsPageAlignedNotPayloadSize)
{
  const std::string name = "/imc_test_region_size_" + std::to_string(getpid());
  SharedSysmem owner;
  ASSERT_TRUE(owner.create(name, 1));  // rounds up to a full page
  ASSERT_GT(owner.size(), static_cast<size_t>(1));

  size_t region_size = 0;
  void * ptr = ExternalMemMapImporter::import(
    name, 1, static_cast<int32_t>(getpid()), &region_size);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(region_size, owner.size());

  // Cache hit path.
  region_size = 0;
  ptr = ExternalMemMapImporter::import(name, 1, static_cast<int32_t>(getpid()), &region_size);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(region_size, owner.size());
}

TEST(ExternalMemMapImporter, GenerationUnchangedDetectsMismatchAndMatch)
{
  const std::string name = "/imc_test_generation_" + std::to_string(getpid());
  SharedSysmem owner;
  ASSERT_TRUE(owner.create(name, 4096));

  auto * gen = intel_memory_core::external_memmap_generation_ptr(
    owner.ptr(), owner.size());
  ASSERT_NE(gen, nullptr);
  gen->store(5, std::memory_order_relaxed);

  size_t region_size = 0;
  void * ptr = ExternalMemMapImporter::import(
    name, 4096, static_cast<int32_t>(getpid()), &region_size);
  ASSERT_NE(ptr, nullptr);

  EXPECT_TRUE(ExternalMemMapImporter::generation_unchanged(ptr, region_size, 5));
  EXPECT_FALSE(ExternalMemMapImporter::generation_unchanged(ptr, region_size, 6));

  // Producer reclaims the slot mid-read: bumps the counter.
  gen->store(6, std::memory_order_relaxed);
  EXPECT_FALSE(ExternalMemMapImporter::generation_unchanged(ptr, region_size, 5));
  EXPECT_TRUE(ExternalMemMapImporter::generation_unchanged(ptr, region_size, 6));
}

TEST(ExternalMemMapImporter, GenerationUnchangedIsTrueWhenRegionTooSmall)
{
  uint8_t byte = 0;
  EXPECT_TRUE(ExternalMemMapImporter::generation_unchanged(&byte, 0, 0));
}

TEST(ExternalMemMapImporter, DeadProducerEntryIsReapedOnNextMiss)
{
  const std::string pid_str = std::to_string(getpid());
  SharedSysmem owner1;
  ASSERT_TRUE(owner1.create("/imc_test_reap_shm1_" + pid_str, 4096));

  pid_t dead_pid = fork();
  ASSERT_GE(dead_pid, 0);
  if (dead_pid == 0) {
    _exit(0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(dead_pid, &status, 0), dead_pid);  // reap the zombie

  void * ptr1 = ExternalMemMapImporter::import(owner1.name(), 4096, dead_pid);
  ASSERT_NE(ptr1, nullptr);

  // An unrelated import from a live producer (this process) triggers the reap.
  SharedSysmem owner2;
  ASSERT_TRUE(owner2.create("/imc_test_reap_shm2_" + pid_str, 4096));
  void * ptr2 = ExternalMemMapImporter::import(
    owner2.name(), 4096, static_cast<int32_t>(getpid()));
  ASSERT_NE(ptr2, nullptr);

  void * ptr1_again = ExternalMemMapImporter::import(owner1.name(), 4096, dead_pid);
  ASSERT_NE(ptr1_again, nullptr);
  EXPECT_NE(ptr1_again, ptr1);
}
