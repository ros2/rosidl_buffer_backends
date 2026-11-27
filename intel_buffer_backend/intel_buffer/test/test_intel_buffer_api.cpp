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

  auto buf = wrap_usm_ptr(heap, size);
  ASSERT_EQ(buf.size(), size);
  uint8_t * ptr = get_usm_ptr(buf);
  ASSERT_EQ(ptr, heap);
  EXPECT_EQ(ptr[0], 0xAB);

  std::free(heap);
}

// The backend can't free a USM pointer itself (it doesn't know the allocator):
// the caller's deleter frees it, once, when the last holder of the buffer goes.
TEST(IntelBufferApi, WrapUsmPtrDeleterFreesPointerOnceWhenBufferIsDestroyed)
{
  const size_t size = 32;
  void * heap = std::malloc(size);
  std::vector<void *> freed;
  {
    auto buf = wrap_usm_ptr(heap, size, [&freed](void * p) {freed.push_back(p); std::free(p);});
    ASSERT_EQ(get_usm_ptr(buf), heap);
    rosidl::Buffer<uint8_t> moved(std::move(buf));
    EXPECT_TRUE(freed.empty());
  }
  ASSERT_EQ(freed.size(), 1u);
  EXPECT_EQ(freed[0], heap);
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

// Every wrap of one dma-buf shares a single pool block (one block id, one
// refcount for subscribers), which outlives the application's fd.
TEST(IntelBufferApi, WrapDmabufSharesOnePoolBlockPerBuffer)
{
  if (!intel_buffer_backend::IntelBufferImpl<uint8_t>::get_or_create_global_pool()) {
    GTEST_SKIP() << "no buffer pool on this machine";
  }
  const size_t size = 4096;
  int fd = make_memfd(size);
  int other_fd = make_memfd(size);
  ASSERT_GE(fd, 0);
  ASSERT_GE(other_fd, 0);
  const int dup_fd = dup(fd);   // another fd for the same buffer

  auto first = wrap_dmabuf(fd, size, /*owns_fd=*/true);
  auto again = wrap_dmabuf(dup_fd, size, /*owns_fd=*/true);
  auto other = wrap_dmabuf(other_fd, size, /*owns_fd=*/true);
  auto block_of = [](rosidl::Buffer<uint8_t> & buf) {
      return dynamic_cast<intel_buffer_backend::IntelBufferImpl<uint8_t> *>(
        buf.get_impl())->get_block();
    };
  ASSERT_NE(block_of(first), nullptr);
  EXPECT_EQ(block_of(again), block_of(first));
  EXPECT_NE(block_of(other)->block_id, block_of(first)->block_id);
  EXPECT_NE(block_of(first)->ipc_meta, nullptr);

  // Both fds of the buffer were closed: the pool's own dup keeps it mapped.
  get_usm_ptr(first)[0] = 0x5e;
  EXPECT_EQ(get_usm_ptr(again)[0], 0x5e);
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
  auto buf = wrap_usm_ptr(std::malloc(16), 16, [](void * p) {std::free(p);});
  const char src[] = "0123456789012345";
  EXPECT_TRUE(write_to_buffer(buf, src, 16));
  EXPECT_EQ(std::memcmp(get_usm_ptr(buf), src, 16), 0);
}

TEST(IntelBufferApi, WriteToBufferRejectsOversizeCopy)
{
  auto buf = wrap_usm_ptr(std::malloc(8), 8, [](void * p) {std::free(p);});
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

// Another process with this pid (another PID namespace sharing /dev/shm)
// already has rings with this prefix: creating ours must still succeed.
TEST(ExternalMemMapRingTest, RingOfAnotherProcessWithTheSamePidDoesNotBlockCreate)
{
  const std::string prefix = "/imb_ring_ns_test";
  std::vector<std::string> theirs;
  for (int ring = 0; ring < 256; ++ring) {   // whichever ring id ours would get
    theirs.push_back(prefix + "_" + std::to_string(getpid()) + "_" + std::to_string(ring) + "_0");
    const int fd = shm_open(theirs.back().c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    close(fd);
  }
  ExternalMemMapRing ring;
  EXPECT_TRUE(ring.create(prefix, 4096, 1));
  for (const auto & name : theirs) {
    shm_unlink(name.c_str());
  }
}
