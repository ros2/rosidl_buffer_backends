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
// on the two entry points is actually load-bearing here.

#include <cstddef>
#include <memory>
#include <new>

#include "cuda_buffer/external_memory_pool.hpp"

namespace
{
/// Stands in for a transport's shared-memory arena.
alignas(256) unsigned char g_arena[64 * 1024];
}  // namespace

extern "C" {

__attribute__((visibility("default")))
int rosidl_cuda_transport_pool_acquire_v1(const void * rmw_publisher, void ** out_pool)
{
  if (out_pool == nullptr) {
    return -1;
  }
  *out_pool = nullptr;
  if (rmw_publisher == nullptr) {
    return -1;
  }

  // One pool for the arena, as a real provider has one per publisher slot.
  // Built once and handed out by copy -- moving out of it would empty the cache
  // and every later acquire would report success with nothing in it.
  static std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool> pool =
    cuda_buffer_backend::ExternalMemoryPool::create(
    g_arena, sizeof(g_arena), 0, nullptr, 256);
  if (!pool) {
    return -1;
  }

  *out_pool =
    new (std::nothrow) std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool>(pool);
  return *out_pool != nullptr ? 0 : -1;
}

__attribute__((visibility("default")))
void rosidl_cuda_transport_pool_release_v1(void * out_pool)
{
  delete static_cast<std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool> *>(
    out_pool);
}

}  // extern "C"
