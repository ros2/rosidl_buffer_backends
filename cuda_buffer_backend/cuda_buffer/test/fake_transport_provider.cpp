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

// An RMW that implements the transport-pool protocol, for test_transport_pool.
//
// Built as librmw_faketransport.so so that the client's
// "lib<rmw_get_implementation_identifier()>.so" lookup finds it, and compiled
// -fvisibility=hidden like a real RMW so that the explicit default visibility
// on the entry points is actually load-bearing here.
//
// Implements the protocol out of a static arena, so a test can tell that an
// allocation came from the transport by looking at where the bytes went.
//
// It also exports two variables no real RMW would. The client memoizes symbol
// resolution for the life of the process, so a provider misbehaviour that has
// to be observed *after* resolution cannot be a second .so; it has to be a
// switch on this one.

#include <cstddef>
#include <memory>
#include <new>

#include "cuda_buffer/external_memory_pool.hpp"

namespace
{
/// Stands in for a per-participant arena, owned by no particular publisher.
alignas(256) unsigned char g_shared_arena[32 * 1024];
}  // namespace

extern "C" {

/// Test control: when non-zero, acquire_shared reports success and hands back a
/// holder with nothing in it -- the one provider bug the client is written to
/// absorb rather than propagate.
__attribute__((visibility("default")))
int rmw_faketransport_shared_empty = 0;

/// Test observation: the size_hint of the last acquire_shared call. The hint is
/// otherwise invisible from the client side, so without this nothing would
/// notice if allocate_buffer() stopped passing it through.
__attribute__((visibility("default")))
size_t rmw_faketransport_last_shared_hint = 0;

__attribute__((visibility("default")))
int rosidl_cuda_transport_pool_acquire_shared_v1(size_t size_hint, void ** out_pool)
{
  if (out_pool == nullptr) {
    return -1;
  }
  *out_pool = nullptr;
  rmw_faketransport_last_shared_hint = size_hint;

  // One pool for the participant. Built once and handed out by copy -- moving
  // out of it would empty the cache and every later acquire would report
  // success with nothing in it.
  static std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool> pool =
    cuda_buffer_backend::ExternalMemoryPool::create(
    g_shared_arena, sizeof(g_shared_arena), 0, nullptr, 256);
  if (!pool) {
    return -1;
  }

  if (rmw_faketransport_shared_empty != 0) {
    *out_pool =
      new (std::nothrow) std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool>();
    return *out_pool != nullptr ? 0 : -1;
  }

  *out_pool =
    new (std::nothrow) std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool>(pool);
  return *out_pool != nullptr ? 0 : -1;
}

__attribute__((visibility("default")))
void rosidl_cuda_transport_pool_release_shared_v1(void * out_pool)
{
  delete static_cast<std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool> *>(
    out_pool);
}

}  // extern "C"
