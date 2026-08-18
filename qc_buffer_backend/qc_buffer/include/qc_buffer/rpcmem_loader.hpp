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

#ifndef QC_BUFFER__RPCMEM_LOADER_HPP_
#define QC_BUFFER__RPCMEM_LOADER_HPP_

#include <cstddef>
#include <cstdint>

#include "qc_buffer/visibility_control.h"

namespace qc_buffer_backend
{

/// ION heap id used for FastRPC-mappable system memory (matches the value
/// used by qrb_ros_*_with_dma).
inline constexpr int kRpcmemHeapIdSystem = 25;

/// Default ION allocation flags (matches qrb_ros_*_with_dma).
inline constexpr uint32_t kRpcmemDefaultFlags = 1;

/// \brief Process-wide loader for the Qualcomm rpcmem (FastRPC/ION) library.
///
/// Wraps dlopen("libcdsprpc.so") and resolves the rpcmem_* symbols. On
/// devices where the library is missing (e.g. developer hosts), available()
/// returns false and the backend transparently falls back to the CPU path.
/// The instance is created lazily and lives for the process lifetime.
class QC_BUFFER_PUBLIC RpcMemLoader
{
public:
  static RpcMemLoader & instance();

  /// True if the library loaded and every required symbol resolved.
  bool available() const {return available_;}

  /// Allocate an ION/dma-buf buffer. Returns nullptr on failure or when the
  /// library is unavailable.
  void * alloc(int heap_id, uint32_t flags, size_t size);

  /// Free a buffer previously returned by alloc().
  void free(void * ptr);

  /// Return the dma-buf fd for a buffer, or -1 on failure/unavailable.
  int to_fd(void * ptr);

private:
  RpcMemLoader();
  ~RpcMemLoader();

  RpcMemLoader(const RpcMemLoader &) = delete;
  RpcMemLoader & operator=(const RpcMemLoader &) = delete;

  using RpcMemInitFn = void (*)(void);
  using RpcMemDeinitFn = void (*)(void);
  using RpcMemAllocFn = void * (*)(int, uint32_t, int);
  using RpcMemFreeFn = void (*)(void *);
  using RpcMemToFdFn = int (*)(void *);

  void * lib_{nullptr};
  bool available_{false};

  RpcMemInitFn init_{nullptr};
  RpcMemDeinitFn deinit_{nullptr};
  RpcMemAllocFn alloc_{nullptr};
  RpcMemFreeFn free_{nullptr};
  RpcMemToFdFn to_fd_{nullptr};
};

}  // namespace qc_buffer_backend

#endif  // QC_BUFFER__RPCMEM_LOADER_HPP_
