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
// One test, not several, because the resolution is memoized per process and the
// interesting properties are about *order*.

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstdint>
#include <memory>

#include "cuda_buffer/transport_pool.hpp"

// What rmw_implementation would answer. Names the companion .so built beside
// this test; the client turns it into "librmw_faketransport.so".
extern "C" const char * rmw_get_implementation_identifier()
{
  return "rmw_faketransport";
}

TEST(TransportPoolTest, ResolvesThroughTheLoadedRmwAndAllocatesFromIt)
{
  // Nothing is loaded yet. The answer must be "no pool" *without* becoming
  // permanent -- that is the regression this ordering exists to catch.
  EXPECT_FALSE(cuda_buffer_backend::transport_pool_available());

  int publisher = 0;  // stands in for an rmw_publisher_t; never dereferenced
  EXPECT_EQ(cuda_buffer_backend::transport_pool_for_publisher(&publisher), nullptr);

  // rmw_implementation's exact call.
  void * rmw = dlopen("librmw_faketransport.so", RTLD_LAZY);
  ASSERT_NE(rmw, nullptr) << "dlopen failed: " << dlerror();

  EXPECT_EQ(dlsym(RTLD_DEFAULT, "rosidl_cuda_transport_pool_acquire_v1"), nullptr)
    << "the RMW's symbols are in the global namespace, so this build no longer "
       "reproduces the RTLD_LOCAL case the client is written for";

  // Now it must resolve -- the earlier "no" must not have stuck.
  EXPECT_TRUE(cuda_buffer_backend::transport_pool_available());
  auto pool = cuda_buffer_backend::transport_pool_for_publisher(&publisher);
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->capacity(), 64u * 1024u);

  // Same pool each time: a provider caches one per publisher, and a fresh
  // allocator per call would hand out overlapping regions of one arena.
  auto again = cuda_buffer_backend::transport_pool_for_publisher(&publisher);
  EXPECT_EQ(again, pool);

  // The payoff: an allocation that lands in the transport's memory.
  const size_t before = pool->bytes_in_use();
  {
    rosidl::Buffer<uint8_t> buffer =
      cuda_buffer_backend::allocate_for_publisher<uint8_t>(&publisher, 1024);
    EXPECT_EQ(buffer.size(), 1024u);
    EXPECT_EQ(buffer.get_backend_type(), "cuda");
    EXPECT_GE(pool->bytes_in_use(), before + 1024)
      << "the buffer did not come from the transport pool";

    auto * impl =
      dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl());
    ASSERT_NE(impl, nullptr);
    EXPECT_TRUE(impl->is_external());
  }
  EXPECT_EQ(pool->bytes_in_use(), before) << "dropping the buffer must return its block";

  // A null publisher is not a transport allocation, so it takes the ordinary
  // path -- which is the global VMM pool and needs a real driver.
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
    rosidl::Buffer<uint8_t> fallback =
      cuda_buffer_backend::allocate_for_publisher<uint8_t>(nullptr, 256);
    EXPECT_EQ(fallback.size(), 256u);
    auto * impl =
      dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(fallback.get_impl());
    ASSERT_NE(impl, nullptr);
    EXPECT_FALSE(impl->is_external()) << "the fallback must not use transport memory";
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
