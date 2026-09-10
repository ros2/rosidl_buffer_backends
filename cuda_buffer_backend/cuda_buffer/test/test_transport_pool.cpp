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

// Finding a transport's device-memory pool through the RMW that is loaded.
//
// The mechanism this covers is easy to get subtly wrong and fails silently when
// it is: the consequence of a bad lookup is not a crash but a process that
// quietly copies every payload forever. Two properties in particular:
//
//   * `dlsym(RTLD_DEFAULT, ...)` cannot see the RMW, because rmw_implementation
//     loads it with `dlopen(path, RTLD_LAZY)` and glibc defaults that to
//     RTLD_LOCAL. The test asserts the negative so that a future change to a
//     global-scope lookup fails here rather than in production.
//   * A negative result must not be memoized before an RMW is loaded, or a
//     process that asks early never sees a pool again.
//
// One test for all of that, not several, because the resolution is memoized per
// process and the interesting properties are about *order*. The second test
// touches no loader state at all.

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstdint>
#include <memory>

// Both, and in this order deliberately: transport_pool.hpp no longer includes
// cuda_buffer_api.hpp -- the dependency runs the other way now -- so a header
// that stopped compiling on its own would otherwise be hidden here.
#include "cuda_buffer/transport_pool.hpp"
#include "cuda_buffer/cuda_buffer_api.hpp"

// What rmw_implementation would answer. Names the companion .so built beside
// this test; the client turns it into "librmw_faketransport.so".
extern "C" const char * rmw_get_implementation_identifier()
{
  return "rmw_faketransport";
}

namespace
{

/// The fallback path allocates from the process-wide VMM pool, which needs a
/// real driver. Everything about transport memory runs without one, so the
/// assertions that do not are guarded rather than the whole test skipped.
bool have_cuda_device()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace

TEST(TransportPoolTest, ResolvesThroughTheLoadedRmwAndAllocatesFromIt)
{
  // Nothing is loaded yet. The answer must be "no pool" *without* becoming
  // permanent -- that is the regression this ordering exists to catch.
  EXPECT_FALSE(cuda_buffer_backend::shared_transport_pool_available());
  EXPECT_EQ(cuda_buffer_backend::shared_transport_pool(1024), nullptr);

  // With no shared pool to be had, allocate_buffer() must be exactly what it
  // was before it learned about one.
  if (have_cuda_device()) {
    rosidl::Buffer<uint8_t> plain = cuda_buffer_backend::allocate_buffer(256);
    EXPECT_EQ(plain.size(), 256u);
    EXPECT_EQ(plain.get_backend_type(), "cuda");
    auto * impl =
      dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(plain.get_impl());
    ASSERT_NE(impl, nullptr);
    EXPECT_FALSE(impl->is_external())
      << "allocate_buffer used transport memory with no transport loaded";
  }

  // rmw_implementation's exact call.
  void * rmw = dlopen("librmw_faketransport.so", RTLD_LAZY);
  ASSERT_NE(rmw, nullptr) << "dlopen failed: " << dlerror();

  EXPECT_EQ(dlsym(RTLD_DEFAULT, "rosidl_cuda_transport_pool_acquire_shared_v1"), nullptr)
    << "the RMW's symbols are in the global namespace, so this build no longer "
       "reproduces the RTLD_LOCAL case the client is written for";

  // Now it must resolve -- the earlier "no" must not have stuck.
  EXPECT_TRUE(cuda_buffer_backend::shared_transport_pool_available());
  auto shared = cuda_buffer_backend::shared_transport_pool(1024);
  ASSERT_NE(shared, nullptr);
  EXPECT_EQ(shared->capacity(), 32u * 1024u);

  // Same pool each time: a provider caches one per participant, and a fresh
  // allocator per call would hand out overlapping regions of one arena.
  auto again = cuda_buffer_backend::shared_transport_pool(1024);
  EXPECT_EQ(again, shared);

  // The payoff, and the reason this exists at all: an unmodified
  // allocate_buffer() call lands in transport memory.
  const size_t shared_before = shared->bytes_in_use();
  {
    rosidl::Buffer<uint8_t> buffer = cuda_buffer_backend::allocate_buffer(2048);
    EXPECT_EQ(buffer.size(), 2048u);
    EXPECT_EQ(buffer.get_backend_type(), "cuda");
    EXPECT_GE(shared->bytes_in_use(), shared_before + 2048)
      << "allocate_buffer did not come from the transport pool";

    auto * impl =
      dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl());
    ASSERT_NE(impl, nullptr);
    EXPECT_TRUE(impl->is_external());
  }
  EXPECT_EQ(shared->bytes_in_use(), shared_before)
    << "dropping the buffer must return its block";

  // The hint is invisible from this side otherwise, so nothing would notice if
  // allocate_buffer() stopped passing the byte count through and a provider
  // with size classes started guessing.
  auto * last_hint =
    static_cast<size_t *>(dlsym(rmw, "rmw_faketransport_last_shared_hint"));
  ASSERT_NE(last_hint, nullptr) << "dlsym failed: " << dlerror();
  EXPECT_EQ(*last_hint, 2048u);

  // A provider that reports success with an empty holder. The client absorbs
  // it: the caller must see "no pool" and get an ordinary buffer, not a null
  // pool that throws from the next allocation, a long way from the cause.
  auto * force_empty =
    static_cast<int *>(dlsym(rmw, "rmw_faketransport_shared_empty"));
  ASSERT_NE(force_empty, nullptr) << "dlsym failed: " << dlerror();
  *force_empty = 1;
  EXPECT_EQ(cuda_buffer_backend::shared_transport_pool(1024), nullptr);
  EXPECT_TRUE(cuda_buffer_backend::shared_transport_pool_available())
    << "a provider bug is not a missing capability";
  if (have_cuda_device()) {
    rosidl::Buffer<uint8_t> fallback = cuda_buffer_backend::allocate_buffer(256);
    EXPECT_EQ(fallback.size(), 256u);
    auto * impl =
      dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(fallback.get_impl());
    ASSERT_NE(impl, nullptr);
    EXPECT_FALSE(impl->is_external());
  }
  *force_empty = 0;
}

// Half a pair is not a capability. Where the fake RMW cannot reach: it exports
// both symbols, so only a hand-built api can show what an RMW that exports one
// of them gets. Touches no loader state, so it neither depends on nor disturbs
// the ordering above.
TEST(TransportPoolTest, HalfAPairIsNotACapability)
{
  cuda_buffer_backend::detail::TransportPoolApi api;
  EXPECT_FALSE(api.shared_valid());

  api.acquire_shared = [](size_t, void **) {return -1;};
  EXPECT_FALSE(api.shared_valid())
    << "acquiring with no way to release leaks the holder every time";

  api.release_shared = [](void *) {};
  EXPECT_TRUE(api.shared_valid());

  api.acquire_shared = nullptr;
  EXPECT_FALSE(api.shared_valid());
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
