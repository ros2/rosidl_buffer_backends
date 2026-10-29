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

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "intel_buffer/intel_buffer_api.hpp"

using intel_buffer_backend::ExternalMemMapRing;
using intel_buffer_backend::get_usm_ptr;
using intel_buffer_backend::wrap_dmabuf;
using intel_buffer_backend::wrap_usm_ptr;
using intel_buffer_backend::write_to_buffer;

namespace
{

int make_memfd(size_t size)
{
  int fd = memfd_create("test_dmabuf", 0);
  if (fd >= 0) {
    EXPECT_EQ(ftruncate(fd, static_cast<off_t>(size)), 0);
  }
  return fd;
}
}  // namespace

TEST(IntelBufferApi, WrapUsmPtrOverHeapBufferIsAccessible)
{
  const size_t size = 64;
  void * heap = std::malloc(size);
  std::memset(heap, 0xAB, size);

  auto buf = wrap_usm_ptr(heap, size, false);
  ASSERT_EQ(buf.size(), size);
  uint8_t * ptr = get_usm_ptr(buf);
  ASSERT_EQ(ptr, heap);
  EXPECT_EQ(ptr[0], 0xAB);

  std::free(heap);
}

TEST(IntelBufferApi, WrapUsmPtrOwnsPtrFreesOnDestruction)
{
  const size_t size = 32;
  void * heap = std::malloc(size);
  {
    auto buf = wrap_usm_ptr(heap, size, true);
    ASSERT_EQ(get_usm_ptr(buf), heap);
  }
  SUCCEED();
}

TEST(IntelBufferApi, WrapDmabufOverMemfdRoundTripsBytes)
{
  const size_t size = 4096;
  int fd = make_memfd(size);
  ASSERT_GE(fd, 0);

  auto buf = wrap_dmabuf(fd, size, true);
  ASSERT_EQ(buf.size(), size);
  uint8_t * ptr = get_usm_ptr(buf);
  ASSERT_NE(ptr, nullptr);

  ptr[0] = 0x42;
  ptr[size - 1] = 0x24;
  EXPECT_EQ(ptr[0], 0x42);
  EXPECT_EQ(ptr[size - 1], 0x24);

}

TEST(IntelBufferApi, WrapDmabufWithOwnsFdFalseLeavesFdOpen)
{
  const size_t size = 4096;
  int fd = make_memfd(size);
  ASSERT_GE(fd, 0);

  {
    auto buf = wrap_dmabuf(fd, size, false);
    ASSERT_EQ(buf.size(), size);
  }

  struct stat st;
  EXPECT_EQ(fstat(fd, &st), 0);
  close(fd);
}

TEST(IntelBufferApi, WrapDmabufWithOwnsFdFalseReleasesMapping)
{
  const size_t size = 4096;
  int fd = make_memfd(size);
  ASSERT_GE(fd, 0);

  void * mapped = nullptr;
  {
    auto buf = wrap_dmabuf(fd, size, false);
    mapped = get_usm_ptr(buf);
    ASSERT_NE(mapped, nullptr);
    unsigned char vec = 0;
    ASSERT_EQ(mincore(mapped, size, &vec), 0);
  }
  unsigned char vec = 0;
  EXPECT_EQ(mincore(mapped, size, &vec), -1);  // unmapped
  EXPECT_EQ(errno, ENOMEM);
  close(fd);
}

TEST(IntelBufferApi, WrapDmabufWithInvalidFdReturnsEmptyBuffer)
{
  auto buf = wrap_dmabuf(-1, 4096, false);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(get_usm_ptr(buf), nullptr);
}

TEST(IntelBufferApi, GetUsmPtrOnNonIntelBufferImplReturnsNullptr)
{
  rosidl::Buffer<uint8_t> cpu_buf(std::make_unique<rosidl::CpuBufferImpl<uint8_t>>());
  EXPECT_EQ(get_usm_ptr(cpu_buf), nullptr);
}

TEST(IntelBufferApi, WriteToBufferCopiesInBoundsData)
{
  auto buf = wrap_usm_ptr(std::malloc(16), 16, true);
  const char src[] = "0123456789012345";
  EXPECT_TRUE(write_to_buffer(buf, src, 16));
  EXPECT_EQ(std::memcmp(get_usm_ptr(buf), src, 16), 0);
}

TEST(IntelBufferApi, WriteToBufferRejectsOversizeCopy)
{
  auto buf = wrap_usm_ptr(std::malloc(8), 8, true);
  const char src[16] = {};
  EXPECT_FALSE(write_to_buffer(buf, src, 16));
}

TEST(IntelBufferApi, WriteToBufferRejectsNullBackedBuffer)
{
  rosidl::Buffer<uint8_t> empty_buf(
    std::make_unique<intel_buffer_backend::IntelBufferImpl<uint8_t>>());
  const char src[4] = {};
  EXPECT_FALSE(write_to_buffer(empty_buf, src, 4));
}

TEST(ExternalMemMapRingTest, AvailableIsAlwaysTrue)
{
  EXPECT_TRUE(ExternalMemMapRing::available());
}

TEST(ExternalMemMapRingTest, CreateFailsOnZeroArgs)
{
  ExternalMemMapRing ring;
  EXPECT_FALSE(ring.create("/imb_ring_test_zero", 0));
  EXPECT_FALSE(ring.create("/imb_ring_test_zero_slots", 4096, 0));
}

TEST(ExternalMemMapRingTest, CreateAdvanceWriteWrapRoundTrip)
{
  ExternalMemMapRing ring;
  ASSERT_TRUE(ring.create("/imb_ring_test", 4096, 3));
  EXPECT_EQ(ring.slots(), 3u);
  EXPECT_GE(ring.slot_size(), 4096u);

  size_t slot = ring.advance();
  uint8_t * ptr = ring.data(slot);
  ASSERT_NE(ptr, nullptr);
  ptr[0] = 0x55;

  auto buf = ring.wrap(slot, 4096);
  ASSERT_EQ(buf.size(), 4096u);
  uint8_t * wrapped_ptr = get_usm_ptr(buf);
  ASSERT_NE(wrapped_ptr, nullptr);
  EXPECT_EQ(wrapped_ptr[0], 0x55);

  size_t next_slot = ring.advance();
  EXPECT_NE(next_slot, slot);
}

TEST(ExternalMemMapRingTest, AdvanceBumpsGenerationAndWrapPropagatesIt)
{
  ExternalMemMapRing ring;
  ASSERT_TRUE(ring.create("/imb_ring_generation_test", 4096, 2));

  const size_t slot_a = ring.advance();
  auto buf_a = ring.wrap(slot_a, 4096);
  auto * impl_a = dynamic_cast<intel_buffer_backend::IntelBufferImpl<uint8_t> *>(
    buf_a.get_impl());
  ASSERT_NE(impl_a, nullptr);
  const uint32_t gen_a = impl_a->get_block()->extmap_generation;

  const size_t slot_b = ring.advance();
  ASSERT_NE(slot_b, slot_a);
  const size_t slot_a_again = ring.advance();
  ASSERT_EQ(slot_a_again, slot_a);

  auto buf_a2 = ring.wrap(slot_a_again, 4096);
  auto * impl_a2 = dynamic_cast<intel_buffer_backend::IntelBufferImpl<uint8_t> *>(
    buf_a2.get_impl());
  ASSERT_NE(impl_a2, nullptr);
  EXPECT_NE(impl_a2->get_block()->extmap_generation, gen_a);
}

TEST(ExternalMemMapRingTest, SecondRingWithSamePrefixInSameProcessSucceedsWithDistinctNames)
{
  ExternalMemMapRing first;
  ASSERT_TRUE(first.create("/imb_ring_collision_test", 4096, 2));

  size_t slot = first.advance();
  uint8_t * ptr = first.data(slot);
  ASSERT_NE(ptr, nullptr);
  ptr[0] = 0x77;

  ExternalMemMapRing second;
  ASSERT_TRUE(second.create("/imb_ring_collision_test", 4096, 2));

  size_t second_slot = second.advance();
  uint8_t * second_ptr = second.data(second_slot);
  ASSERT_NE(second_ptr, nullptr);
  second_ptr[0] = 0x88;

  EXPECT_EQ(ptr[0], 0x77);
  EXPECT_EQ(second_ptr[0], 0x88);
}

TEST(SharedSysmemTest, GenuineNameCollisionFailsInsteadOfStealing)
{
  intel_buffer_backend::SharedSysmem first;
  ASSERT_TRUE(first.create("/imb_shm_collision_test", 4096));

  intel_buffer_backend::SharedSysmem second;
  EXPECT_FALSE(second.create("/imb_shm_collision_test", 4096));
}

TEST(ExternalMemMapRingTest, DataAndWrapOutOfRangeOnClosedRingAreSafe)
{
  ExternalMemMapRing ring;
  EXPECT_EQ(ring.data(0), nullptr);
  auto buf = ring.wrap(0, 4096);
  EXPECT_EQ(buf.size(), 0u);
}
