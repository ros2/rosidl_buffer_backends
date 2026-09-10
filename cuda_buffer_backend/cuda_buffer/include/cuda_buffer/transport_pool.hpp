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

// Nothing from cuda_buffer_api.hpp, deliberately: allocate_buffer() consults
// the pool declared below, so that header includes *this* one. The dependency
// has to run one way, and this is the end of it that hands back a pool rather
// than a buffer, so it can do without.
#include "cuda_buffer/external_memory_pool.hpp"
#include "rmw/rmw.h"

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
/// certainly should not link the RMW to find out. It should not have to hold a
/// publisher either -- most allocation sites never see one -- so there is
/// nothing portable to ask here at all. Ordinary `allocate_buffer(n)` consults
/// the pool below and lands in transport memory where one exists; every
/// existing call site gets that unmodified, and none of them names a writer.
///
/// \par The protocol
/// An RMW opts in by exporting two C symbols with default visibility:
///
/// \code
///   /// \param size_hint bytes the caller is about to allocate, so a provider
///   ///   keeping several size classes can pick one. 0 means "no hint".
///   /// \param out_pool receives a heap-allocated
///   ///   `std::shared_ptr<cuda_buffer_backend::ExternalMemoryPool> *`.
///   /// \return 0 on success; anything else means "no shared pool", which is
///   ///   an ordinary answer and not an error.
///   int rosidl_cuda_transport_pool_acquire_shared_v1(size_t size_hint,
///                                                    void ** out_pool);
///
///   /// Destroy the holder from a successful acquire. The caller has copied
///   /// the shared_ptr out of it by then, so the pool itself survives.
///   void rosidl_cuda_transport_pool_release_shared_v1(void * out_pool);
/// \endcode
///
/// The two-call shape exists because a `shared_ptr` cannot cross a C boundary.
/// The `_v1` suffix is the ABI version: a future revision adds new symbols
/// rather than changing these, so an old client and a new RMW keep working.
///
/// The `_shared` in the names is historical and load-bearing only as an ABI
/// string. There was once a second, publisher-scoped pair -- `acquire_v1` /
/// `release_v1`, reached through an `allocate_for_publisher()` that took an
/// `rmw_publisher_t *` -- and these were "the participant-wide variant" of it.
/// That pair is gone: an allocation that has to name a writer cannot serve the
/// call sites this exists for, and keeping both meant two ways to ask one
/// question. Do not reintroduce a publisher-scoped symbol under a new name
/// without first showing that a caller which cannot reach `allocate_buffer()`
/// actually exists.
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
  int (* acquire_shared)(size_t, void **) = nullptr;
  void (* release_shared)(void *) = nullptr;

  bool shared_valid() const
  {
    return acquire_shared != nullptr && release_shared != nullptr;
  }
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

  api.acquire_shared = reinterpret_cast<int (*)(size_t, void **)>(
    dlsym(handle, "rosidl_cuda_transport_pool_acquire_shared_v1"));
  api.release_shared = reinterpret_cast<void (*)(void *)>(
    dlsym(handle, "rosidl_cuda_transport_pool_release_shared_v1"));
  // Half a protocol is no protocol; treat it as absent rather than calling
  // acquire with no way to release.
  if (!api.shared_valid()) {
    api.acquire_shared = nullptr;
    api.release_shared = nullptr;
  }
  decided.store(true, std::memory_order_release);
  return api;
}

}  // namespace detail

/// \brief The transport's participant-wide device-memory pool, or null.
///
/// Null whenever the transport has nothing to offer: an RMW that does not
/// implement the protocol, no usable GPU, or nothing free right now. All
/// ordinary answers -- allocate a normal buffer instead, which is what
/// allocate_buffer() does.
///
/// It names no writer, and that is what it is for: a call site which was never
/// handed a publisher -- most of them -- can still put its payload where the
/// transport can send it from.
///
/// \param size_hint Bytes the caller is about to allocate, so a provider that
///   keeps several size classes can pick one. 0 asks it to choose for itself.
///   Only ever a hint: the pool that comes back is not promised to fit.
inline std::shared_ptr<ExternalMemoryPool> shared_transport_pool(size_t size_hint)
{
  const detail::TransportPoolApi & api = detail::transport_pool_api();
  if (!api.shared_valid()) {
    return nullptr;
  }
  void * holder = nullptr;
  if (api.acquire_shared(size_hint, &holder) != 0 || holder == nullptr) {
    return nullptr;
  }
  auto * typed = static_cast<std::shared_ptr<ExternalMemoryPool> *>(holder);
  // Copied, not moved: the holder is the provider's to destroy, and it may be
  // handing out a reference to a pool it keeps.
  std::shared_ptr<ExternalMemoryPool> pool = *typed;
  api.release_shared(holder);
  // A provider that reports success with an empty holder is buggy, but the
  // useful response is the same as "no pool": fall back. Returning the null
  // through would turn a provider's bug into an exception from the caller's
  // next allocation, a long way from the cause.
  return pool ? pool : nullptr;
}

/// \brief True when this process's RMW offers a participant-wide shared pool.
///
/// For diagnostics and tests -- "is allocate_buffer() actually getting zero
/// copy?" -- not for branching, since allocate_buffer() already handles both
/// cases.
inline bool shared_transport_pool_available()
{
  return detail::transport_pool_api().shared_valid();
}

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER__TRANSPORT_POOL_HPP_
