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

#include "qc_buffer/rpcmem_loader.hpp"

#include <dlfcn.h>

#include <rcutils/logging_macros.h>

namespace qc_buffer_backend
{

RpcMemLoader & RpcMemLoader::instance()
{
  static RpcMemLoader loader;
  return loader;
}

RpcMemLoader::RpcMemLoader()
{
  lib_ = ::dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (lib_ == nullptr) {
    RCUTILS_LOG_WARN_NAMED("qc_buffer_backend",
      "libcdsprpc.so not available (%s); qc buffers will fall back to CPU",
      ::dlerror());
    return;
  }

  init_ = reinterpret_cast<RpcMemInitFn>(::dlsym(lib_, "rpcmem_init"));
  deinit_ = reinterpret_cast<RpcMemDeinitFn>(::dlsym(lib_, "rpcmem_deinit"));
  alloc_ = reinterpret_cast<RpcMemAllocFn>(::dlsym(lib_, "rpcmem_alloc"));
  free_ = reinterpret_cast<RpcMemFreeFn>(::dlsym(lib_, "rpcmem_free"));
  to_fd_ = reinterpret_cast<RpcMemToFdFn>(::dlsym(lib_, "rpcmem_to_fd"));

  if (alloc_ == nullptr || free_ == nullptr || to_fd_ == nullptr) {
    RCUTILS_LOG_WARN_NAMED("qc_buffer_backend",
      "rpcmem symbols missing in libcdsprpc.so; qc buffers will fall back to CPU");
    ::dlclose(lib_);
    lib_ = nullptr;
    return;
  }

  if (init_ != nullptr) {
    init_();
  }

  available_ = true;
  RCUTILS_LOG_INFO_NAMED("qc_buffer_backend", "rpcmem initialized");
}

RpcMemLoader::~RpcMemLoader()
{
  if (lib_ != nullptr) {
    if (deinit_ != nullptr) {
      deinit_();
    }
    ::dlclose(lib_);
    lib_ = nullptr;
  }
}

void * RpcMemLoader::alloc(int heap_id, uint32_t flags, size_t size)
{
  if (!available_) {
    return nullptr;
  }
  return alloc_(heap_id, flags, static_cast<int>(size));
}

void RpcMemLoader::free(void * ptr)
{
  if (available_ && ptr != nullptr) {
    free_(ptr);
  }
}

int RpcMemLoader::to_fd(void * ptr)
{
  if (!available_ || ptr == nullptr) {
    return -1;
  }
  return to_fd_(ptr);
}

}  // namespace qc_buffer_backend
