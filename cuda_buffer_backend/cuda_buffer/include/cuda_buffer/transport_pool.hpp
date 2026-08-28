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

#ifndef CUDA_BUFFER__TRANSPORT_POOL_HPP_
#define CUDA_BUFFER__TRANSPORT_POOL_HPP_

#include <dlfcn.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/external_memory_pool.hpp"
#include "rmw/rmw.h"
#include "rosidl_buffer/buffer.hpp"

namespace cuda_buffer_backend
{

/// \file
/// \brief Allocating a buffer the *current* RMW can send with the fewest copies.
///
/// Some transports own device memory a publisher can write directly -- a
/// zero-copy shared-memory transport whose slots are NvSciBuf, say. Writing a
/// payload there means it never has to be copied to be sent. Every other
/// transport wants an ordinary buffer and will copy it into whatever it sends.
///
/// An application should not have to know which kind it is talking to, and
/// certainly should not link the RMW to find out. So this asks, portably:
///
///     msg.data = allocate_for_publisher<uint8_t>(rmw_pub, count);
///
/// and gets transport memory when the transport offers it, an ordinary pooled
/// buffer when it does not. The result is a `rosidl::Buffer` either way, with
/// the same backend name, handles and event ordering, so nothing downstream
/// changes.
///
/// \par The protocol
/// An RMW opts in by exporting two C symbols with default visibility:
///
/// \code
///   /// \param rmw_publisher an `rmw_publisher_t *`.
///   /// \param out_pool receives a heap-allocated
///   ///   `std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool> *`.
///   /// \return 0 on success; anything else means "no pool for this
///   ///   publisher", which is an ordinary answer and not an error.
///   int rosidl_cuda_transport_pool_acquire_v1(const void * rmw_publisher,
///                                             void ** out_pool);
///
///   /// Destroy the holder from a successful acquire. The caller has copied
///   /// the shared_ptr out of it by then, so the pool itself survives.
///   void rosidl_cuda_transport_pool_release_v1(void * out_pool);
/// \endcode
///
/// The two-call shape exists because a `shared_ptr` cannot cross a C boundary.
/// The `_v1` suffix is the ABI version: a future revision adds new symbols
/// rather than changing these, so an old client and a new RMW keep working.
///
/// \par Why the RMW is found this way
/// `rmw_implementation` loads the RMW with `dlopen(path, RTLD_LAZY)`, and glibc
/// defaults that to `RTLD_LOCAL` -- so the symbols are *not* in the global
/// namespace and `dlsym(RTLD_DEFAULT, ...)` cannot see them. Reopening the
/// library by name with `RTLD_NOLOAD` gets a handle to the copy already loaded
/// without loading a second one, which is the whole point: linking the RMW
/// directly would risk two copies of its static state in one process.
///
/// The name comes from `rmw_get_implementation_identifier()` and ROS's
/// `lib<identifier>.so` convention, so no RMW is named here.

namespace detail
{

/// The RMW's transport-pool entry points.
///
/// Absent for most RMWs, which is the expected case and not a failure.
struct TransportPoolApi
{
  int (* acquire)(const void *, void **) = nullptr;
  void (* release)(void *) = nullptr;

  bool valid() const {return acquire != nullptr && release != nullptr;}
};

/// Resolve the protocol, memoizing only once the answer is *definitive*.
///
/// The distinction matters and is easy to get wrong. Caching "absent" on the
/// first call would be wrong whenever that call happens before the RMW is
/// loaded -- the process would then never see a transport pool, silently, for
/// its whole life. So a failed `dlopen` means "not loaded yet, ask again",
/// while a successful one means the RMW is present and its answer -- symbols or
/// no symbols -- is final.
///
/// The cost of that care is one loader lookup per call until an RMW is loaded.
/// After that it is an atomic read, including for the common case of an RMW
/// that does not implement the protocol at all.
inline const TransportPoolApi & transport_pool_api()
{
  static std::atomic<bool> decided{false};
  static std::mutex mutex;
  static TransportPoolApi api;

  if (decided.load(std::memory_order_acquire)) {
    return api;
  }

  std::lock_guard<std::mutex> lock(mutex);
  if (decided.load(std::memory_order_relaxed)) {
    return api;
  }

  const char * id = rmw_get_implementation_identifier();
  if (id == nullptr) {
    return api;  // no RMW yet; undecided
  }

  // RTLD_NOLOAD: resolve against the copy rmw_implementation already loaded,
  // or fail. Never load one ourselves -- a second copy of an RMW in a process
  // is its own kind of bug.
  //
  // Deliberately never dlclose'd. The handle is process-lifetime state, and
  // closing it while a pool from that library is still alive would unmap the
  // code its deleter runs in.
  const std::string soname = "lib" + std::string(id) + ".so";
  void * handle = dlopen(soname.c_str(), RTLD_LAZY | RTLD_NOLOAD);
  if (handle == nullptr) {
    return api;  // not loaded yet; undecided
  }

  api.acquire = reinterpret_cast<int (*)(const void *, void **)>(
    dlsym(handle, "rosidl_cuda_transport_pool_acquire_v1"));
  api.release = reinterpret_cast<void (*)(void *)>(
    dlsym(handle, "rosidl_cuda_transport_pool_release_v1"));
  if (!api.valid()) {
    // Half a protocol is no protocol; treat it as absent rather than calling
    // acquire with no way to release.
    api = TransportPoolApi{};
  }
  decided.store(true, std::memory_order_release);
  return api;
}

}  // namespace detail

/// \brief The transport's device-memory pool for \p rmw_publisher, or null.
///
/// Null whenever the transport has nothing to offer: an RMW that does not
/// implement the protocol, a publisher whose type carries no buffer field, no
/// usable GPU, or no free slot right now. All ordinary answers -- allocate a
/// normal buffer instead, which is what allocate_for_publisher() does.
///
/// \param rmw_publisher an `rmw_publisher_t *`. Taken as `void *` so this
///   header imposes no opinion about how the caller reached it; from rclcpp it
///   is `rcl_publisher_get_rmw_handle(pub->get_publisher_handle().get())`.
inline std::shared_ptr<ExternalMemoryPool> transport_pool_for_publisher(
  const void * rmw_publisher)
{
  if (rmw_publisher == nullptr) {
    return nullptr;
  }
  const detail::TransportPoolApi & api = detail::transport_pool_api();
  if (!api.valid()) {
    return nullptr;
  }
  void * holder = nullptr;
  if (api.acquire(rmw_publisher, &holder) != 0 || holder == nullptr) {
    return nullptr;
  }
  auto * typed = static_cast<std::shared_ptr<ExternalMemoryPool> *>(holder);
  // Copied, not moved: the holder is the provider's to destroy, and it may be
  // handing out a reference to a pool it keeps.
  std::shared_ptr<ExternalMemoryPool> pool = *typed;
  api.release(holder);
  // A provider that reports success with an empty holder is buggy, but the
  // useful response is the same as "no pool": fall back. Returning the null
  // through would turn a provider's bug into an exception from the caller's
  // next allocation, a long way from the cause.
  return pool ? pool : nullptr;
}

/// \brief Allocate \p count elements of device memory suited to \p rmw_publisher.
///
/// Transport-owned when the transport offers it -- so publishing copies nothing
/// -- and an ordinary pooled buffer otherwise. Identical to use either way.
///
/// Two caveats, both inherited from the transport pool and both irrelevant to
/// the fallback:
///   * A transport buffer is tied to the publisher's current slot. Allocate,
///     fill and publish one message at a time, and allocate again for the next.
///   * After publishing, a transport buffer names memory the transport owns.
///     Do not write it, and do not expect it to stay valid indefinitely.
///
/// Falling back is never silent about being *wrong*, only about being ordinary:
/// if the transport had a pool and it failed to satisfy the request, that
/// throws rather than quietly costing a copy.
template<typename T = uint8_t>
rosidl::Buffer<T> allocate_for_publisher(const void * rmw_publisher, size_t count)
{
  if (std::shared_ptr<ExternalMemoryPool> pool =
    transport_pool_for_publisher(rmw_publisher))
  {
    return allocate_buffer_from<T>(std::move(pool), count);
  }
  return rosidl::Buffer<T>(std::make_unique<CudaBufferImpl<T>>(count));
}

/// \brief True when this process's RMW implements the transport-pool protocol.
///
/// For diagnostics and tests -- "am I actually getting zero copy?" -- not for
/// branching, since allocate_for_publisher() already handles both cases.
inline bool transport_pool_available()
{
  return detail::transport_pool_api().valid();
}

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__TRANSPORT_POOL_HPP_
