// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
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

// Suballocating a region this backend does not own.
//
// Most of these are deliberately fixture-free and run without a GPU. The pool
// is pure address arithmetic over a base pointer -- it makes no CUDA calls of
// its own -- so every allocation, recycling, coalescing and lifetime rule below
// can be checked on a machine with no device. The tests that move bytes take a
// device and skip themselves when there is none.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/external_memory_pool.hpp"

namespace
{

/// Stands in for a middleware's already-mapped arena -- an NvSciBuf pool
/// imported into CUDA looks exactly like this from the pool's side: one base
/// address and a length.
alignas(4096) uint8_t g_arena[64 * 1024];

constexpr size_t kAlign = cuda_buffer_backend::ExternalMemoryPool::kDefaultAlignment;

std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool> make_pool(
  size_t size = sizeof(g_arena), std::shared_ptr<void> keepalive = nullptr)
{
  // device_id 0 rather than -1: no CUDA device is needed for these, and asking
  // for the current one would fail on a GPU-less machine.
  return cuda_buffer_backend::ExternalMemoryPool::create(
    g_arena, size, 0, std::move(keepalive), kAlign);
}

}  // namespace

// ---------------------------------------------------------------------------
// The allocator itself
// ---------------------------------------------------------------------------

TEST(ExternalMemoryPoolTest, PoolsTheRegionItWasGiven)
{
  auto pool = make_pool();
  ASSERT_NE(pool, nullptr);

  EXPECT_EQ(pool->capacity(), sizeof(g_arena)) << "a 4096-aligned arena loses nothing";
  EXPECT_EQ(pool->bytes_free(), sizeof(g_arena));
  EXPECT_EQ(pool->bytes_in_use(), 0u);
  EXPECT_EQ(pool->get_device_id(), 0);
  EXPECT_EQ(pool->alignment(), kAlign);
}

TEST(ExternalMemoryPoolTest, HandsOutAlignedNonOverlappingBlocks)
{
  auto pool = make_pool();

  auto * a = pool->allocate(100);
  auto * b = pool->allocate(100);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  // Rounded up to the granularity, so two 100-byte requests cannot overlap.
  EXPECT_EQ(a->size, kAlign);
  EXPECT_EQ(b->size, kAlign);
  EXPECT_NE(a->ptr, b->ptr);
  EXPECT_GE(b->ptr, a->ptr + a->size);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(a->ptr) % kAlign, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(b->ptr) % kAlign, 0u);
  EXPECT_EQ(pool->bytes_in_use(), 2 * kAlign);

  pool->free(a);
  pool->free(b);
  EXPECT_EQ(pool->bytes_in_use(), 0u);
}

TEST(ExternalMemoryPoolTest, ReusesReturnedBlocks)
{
  auto pool = make_pool();

  // The transport case: the same slot size cycling forever. It must reuse
  // storage rather than walking down the arena.
  uint8_t * first = nullptr;
  for (int i = 0; i < 100; ++i) {
    auto * block = pool->allocate(1024);
    ASSERT_NE(block, nullptr) << "iteration " << i;
    if (i == 0) {
      first = block->ptr;
    } else {
      EXPECT_EQ(block->ptr, first) << "iteration " << i << " should reuse the same offset";
    }
    pool->free(block);
  }
  EXPECT_EQ(pool->bytes_in_use(), 0u);
}

TEST(ExternalMemoryPoolTest, CoalescesAdjacentFreeBlocks)
{
  auto pool = make_pool();

  auto * a = pool->allocate(kAlign);
  auto * b = pool->allocate(kAlign);
  auto * c = pool->allocate(kAlign);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);

  // Free the outer two first, then the middle. Without coalescing the arena
  // would be left with three separate holes and could not satisfy a request
  // spanning them.
  pool->free(a);
  pool->free(c);
  pool->free(b);

  EXPECT_EQ(pool->bytes_free(), pool->capacity());
  EXPECT_EQ(pool->largest_free_block(), pool->capacity())
    << "three adjacent frees must merge back into one region";
}

TEST(ExternalMemoryPoolTest, CoalescesWithPredecessorAndSuccessor)
{
  // An arena of exactly four blocks, so the free tail cannot mask the merge:
  // whatever largest_free_block() reports has to come from the three blocks
  // under test.
  auto pool = make_pool(4 * kAlign);
  ASSERT_EQ(pool->capacity(), 4 * kAlign);

  auto * a = pool->allocate(kAlign);
  auto * b = pool->allocate(kAlign);
  auto * c = pool->allocate(kAlign);
  auto * tail = pool->allocate(kAlign);  // holds the arena tail out of the merge
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);
  ASSERT_NE(tail, nullptr);
  ASSERT_EQ(pool->bytes_free(), 0u);

  pool->free(a);  // a free region before b
  pool->free(c);  // a free region after b
  EXPECT_EQ(pool->largest_free_block(), kAlign) << "two separate one-block holes";

  pool->free(b);  // must merge with both neighbours in one release
  EXPECT_EQ(pool->largest_free_block(), 3 * kAlign)
    << "a, b and c must become a single region, not three";
  EXPECT_EQ(pool->bytes_free(), 3 * kAlign);

  // And the merged region is genuinely usable as one run.
  auto * big = pool->allocate(3 * kAlign);
  ASSERT_NE(big, nullptr);
  EXPECT_EQ(big->offset, 0u);
}

TEST(ExternalMemoryPoolTest, ReportsExhaustionAsNull)
{
  // One block's worth of arena, so the second request cannot be satisfied.
  auto pool = make_pool(kAlign);
  ASSERT_EQ(pool->capacity(), kAlign);

  auto * a = pool->allocate(kAlign);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(pool->allocate(1), nullptr) << "exhaustion is null, not an exception";
  EXPECT_EQ(pool->bytes_free(), 0u);

  pool->free(a);
  EXPECT_NE(pool->allocate(1), nullptr) << "and it recovers once the block is back";
}

TEST(ExternalMemoryPoolTest, RefusesRequestsLargerThanTheArena)
{
  auto pool = make_pool();
  EXPECT_EQ(pool->allocate(sizeof(g_arena) + 1), nullptr);
  EXPECT_EQ(pool->bytes_in_use(), 0u) << "a refused request must reserve nothing";
}

TEST(ExternalMemoryPoolTest, GivesDistinctAddressesForZeroSizedRequests)
{
  auto pool = make_pool();
  auto * a = pool->allocate(0);
  auto * b = pool->allocate(0);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a->ptr, b->ptr) << "aliasing here would corrupt the free list on release";
}

TEST(ExternalMemoryPoolTest, AlignsAnUnalignedBase)
{
  // A base one byte into the arena: the pool must align up rather than hand out
  // misaligned device pointers.
  auto pool = cuda_buffer_backend::ExternalMemoryPool::create(
    g_arena + 1, sizeof(g_arena) - 1, 0, nullptr, kAlign);
  ASSERT_NE(pool, nullptr);

  auto * block = pool->allocate(16);
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(block->ptr) % kAlign, 0u);
  EXPECT_GT(block->ptr, g_arena);
  EXPECT_LE(block->ptr + block->size, g_arena + sizeof(g_arena));
}

TEST(ExternalMemoryPoolTest, RejectsNonsenseArguments)
{
  EXPECT_THROW(
    cuda_buffer_backend::ExternalMemoryPool::create(nullptr, 1024, 0),
    cuda_buffer_backend::CudaError);
  EXPECT_THROW(
    cuda_buffer_backend::ExternalMemoryPool::create(g_arena, 0, 0),
    cuda_buffer_backend::CudaError);
  EXPECT_THROW(
    cuda_buffer_backend::ExternalMemoryPool::create(g_arena, 1024, 0, nullptr, 300),
    cuda_buffer_backend::CudaError) << "alignment must be a power of two";
  EXPECT_THROW(
    cuda_buffer_backend::ExternalMemoryPool::create(g_arena, 8, 0, nullptr, kAlign),
    cuda_buffer_backend::CudaError) << "region smaller than one alignment unit";
  EXPECT_THROW(
    cuda_buffer_backend::ExternalMemoryPool::import_nvscibuf(nullptr, 0, 1024),
    cuda_buffer_backend::CudaError);
}

TEST(ExternalMemoryPoolTest, HoldsTheKeepaliveUntilEveryBlockIsBack)
{
  auto pin = std::make_shared<int>(1);
  std::weak_ptr<int> observer = pin;

  auto pool = make_pool(sizeof(g_arena), pin);
  pin.reset();
  EXPECT_FALSE(observer.expired());

  auto buffer = cuda_buffer_backend::allocate_buffer_from<uint8_t>(pool, 1024);
  pool.reset();
  EXPECT_FALSE(observer.expired())
    << "a live buffer must keep the pool, and so the owner's pin, alive";

  buffer = rosidl::Buffer<uint8_t>();
  EXPECT_TRUE(observer.expired()) << "dropping the last buffer must release the pin";
}

// ---------------------------------------------------------------------------
// Buffers allocated from the pool
// ---------------------------------------------------------------------------

TEST(ExternalMemoryPoolBufferTest, LooksLikeAnyOtherCudaBuffer)
{
  auto pool = make_pool();
  auto buffer = cuda_buffer_backend::allocate_buffer_from<uint8_t>(pool, 1024);

  EXPECT_EQ(buffer.get_backend_type(), "cuda");
  EXPECT_EQ(buffer.size(), 1024u);

  auto * impl = dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl());
  ASSERT_NE(impl, nullptr);
  EXPECT_TRUE(impl->is_external());
  EXPECT_FALSE(impl->is_adopted()) << "a pool block is reallocatable; an adopted one is not";
  EXPECT_EQ(impl->get_external_pool(), pool);

  // The pointer really is inside the caller's arena.
  const uint8_t * ptr = impl->get_cuda_buffer().get_device_ptr();
  EXPECT_GE(ptr, g_arena);
  EXPECT_LE(ptr + 1024, g_arena + sizeof(g_arena));
}

TEST(ExternalMemoryPoolBufferTest, ReturnsItsBlockWhenDropped)
{
  auto pool = make_pool();
  const size_t free_before = pool->bytes_free();

  {
    auto buffer = cuda_buffer_backend::allocate_buffer_from<uint8_t>(pool, 4096);
    EXPECT_LT(pool->bytes_free(), free_before);
  }

  EXPECT_EQ(pool->bytes_free(), free_before) << "the block must come back to the pool";
}

TEST(ExternalMemoryPoolBufferTest, CountsElementsNotBytes)
{
  auto pool = make_pool();
  auto buffer = cuda_buffer_backend::allocate_buffer_from<float>(pool, 256);

  EXPECT_EQ(buffer.size(), 256u);
  // 256 floats is 1024 bytes; the pool must have reserved bytes, not elements.
  EXPECT_GE(pool->bytes_in_use(), 256 * sizeof(float));
}

TEST(ExternalMemoryPoolBufferTest, ThrowsWhenThePoolCannotSatisfyTheRequest)
{
  auto pool = make_pool(kAlign);
  EXPECT_THROW(
    cuda_buffer_backend::allocate_buffer_from<uint8_t>(pool, sizeof(g_arena)),
    cuda_buffer_backend::CudaError)
    << "a buffer that cannot be allocated is an error, not an empty buffer";

  EXPECT_THROW(
    cuda_buffer_backend::allocate_buffer_from<uint8_t>(nullptr, 16),
    cuda_buffer_backend::CudaError);
}

TEST(ExternalMemoryPoolBufferTest, EmptyRequestAllocatesNothing)
{
  auto pool = make_pool();
  auto buffer = cuda_buffer_backend::allocate_buffer_from<uint8_t>(pool, 0);
  EXPECT_EQ(buffer.size(), 0u);
  EXPECT_EQ(pool->bytes_in_use(), 0u);
}

TEST(ExternalMemoryPoolBufferTest, GrowsWithinTheSamePool)
{
  // Needs a device: growing reallocates and copies the old contents forward,
  // and that copy is a real cudaMemcpyAsync.
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "no usable CUDA device";
  }

  constexpr size_t kArena = 1 << 20;
  void * arena = nullptr;
  ASSERT_EQ(cudaMalloc(&arena, kArena), cudaSuccess);

  {
    auto pool = cuda_buffer_backend::ExternalMemoryPool::create(arena, kArena, 0);
    auto buffer = cuda_buffer_backend::allocate_buffer_from<uint8_t>(pool, 1024);
    auto * impl =
      dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl());
    ASSERT_NE(impl, nullptr);

    // The difference from adoption: adopted storage cannot grow, a pool block
    // can -- into the rest of the pool's own arena.
    impl->resize(8192);
    EXPECT_EQ(impl->size(), 8192u);
    EXPECT_TRUE(impl->is_external()) << "growth must not fall back to the global pool";

    const uint8_t * ptr = impl->get_cuda_buffer().get_device_ptr();
    EXPECT_GE(ptr, static_cast<const uint8_t *>(arena));
    EXPECT_LE(ptr + 8192, static_cast<const uint8_t *>(arena) + kArena)
      << "growth must stay inside the caller's region";
  }

  EXPECT_EQ(cudaFree(arena), cudaSuccess);
}

TEST(ExternalMemoryPoolBufferTest, ShrinksWithoutReallocating)
{
  auto pool = make_pool();
  auto buffer = cuda_buffer_backend::allocate_buffer_from<uint8_t>(pool, 4096);
  auto * impl = dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl());
  ASSERT_NE(impl, nullptr);
  const uint8_t * before = impl->get_cuda_buffer().get_device_ptr();

  ASSERT_TRUE(cuda_buffer_backend::shrink_buffer(buffer, 100));
  EXPECT_EQ(buffer.size(), 100u);
  EXPECT_EQ(impl->get_cuda_buffer().get_device_ptr(), before);
}

// ---------------------------------------------------------------------------
// The one test that needs a device
// ---------------------------------------------------------------------------

TEST(ExternalMemoryPoolDeviceTest, RoundTripsBytesThroughAPooledBuffer)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "no usable CUDA device";
  }

  // cudaMalloc stands in for the middleware's already-mapped arena; an
  // NvSciBuf object imported via import_nvscibuf() produces the same thing.
  constexpr size_t kArena = 1 << 20;
  void * arena = nullptr;
  ASSERT_EQ(cudaMalloc(&arena, kArena), cudaSuccess);

  {
    auto pool = cuda_buffer_backend::ExternalMemoryPool::create(arena, kArena, 0);
    ASSERT_NE(pool, nullptr);

    constexpr size_t kCount = 4096;
    std::vector<uint8_t> expected(kCount);
    for (size_t i = 0; i < kCount; ++i) {
      expected[i] = static_cast<uint8_t>(i % 251);
    }

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto buffer = cuda_buffer_backend::allocate_buffer_from<uint8_t>(pool, kCount);

    // Written and read through the ordinary handle API, with no hint that the
    // storage came from somebody else's allocator.
    {
      cuda_buffer_backend::WriteHandle wh =
        cuda_buffer_backend::from_output_buffer(buffer, stream);
      ASSERT_EQ(
        cudaMemcpyAsync(wh.get_ptr(), expected.data(), kCount,
        cudaMemcpyHostToDevice, stream), cudaSuccess);
    }
    {
      cuda_buffer_backend::ReadHandle rh =
        cuda_buffer_backend::from_input_buffer(buffer, stream);
      // No promotion: the read must hand back the pooled pointer itself.
      EXPECT_GE(rh.get_ptr(), static_cast<const uint8_t *>(arena));
      EXPECT_LE(rh.get_ptr() + kCount, static_cast<const uint8_t *>(arena) + kArena);
    }

    const std::vector<uint8_t> host = buffer.to_vector();
    ASSERT_EQ(host.size(), kCount);
    EXPECT_EQ(std::memcmp(host.data(), expected.data(), kCount), 0);

    // A copy escapes the arena: rosidl::Buffer's copy constructor deep-copies
    // through impl->clone(), and that clone must not be pooled storage, because
    // the whole point of copying is to outlive the region.
    rosidl::Buffer<uint8_t> cloned = buffer;
    auto * clone_impl =
      dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(cloned.get_impl());
    ASSERT_NE(clone_impl, nullptr);
    EXPECT_FALSE(clone_impl->is_external())
      << "a copy must not consume a slot in the arena it is escaping";
    EXPECT_EQ(cloned.size(), kCount);
    EXPECT_EQ(std::memcmp(cloned.to_vector().data(), expected.data(), kCount), 0);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  }

  // The pool is gone. If it had taken ownership this would double-free.
  EXPECT_EQ(cudaFree(arena), cudaSuccess);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
