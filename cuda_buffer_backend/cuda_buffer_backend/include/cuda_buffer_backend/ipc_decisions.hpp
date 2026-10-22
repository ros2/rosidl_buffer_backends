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

#ifndef CUDA_BUFFER_BACKEND__IPC_DECISIONS_HPP_
#define CUDA_BUFFER_BACKEND__IPC_DECISIONS_HPP_

#include <mutex>
#include <unordered_map>

#include "cuda_buffer/host_endpoint_manager.hpp"

namespace cuda_buffer_backend
{
namespace detail
{

// Only backend discovery may enroll a peer. Missing records stay pending until
// a manager update arrives. No endpoint-removal hook exists in BufferBackend;
// all state (including permanently remote peers) lives until backend teardown.
class IpcDecisions
{
public:
  using GidKey = host_endpoint_manager::GidKey;
  using LocalityInfo = host_endpoint_manager::LocalityInfo;

  template<typename Lookup>
  bool discover(const GidKey & key, bool eligible, int device, uint32_t uid, Lookup lookup)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!eligible) {
      peers_[key] = State::Rejected;
      return false;
    }
    auto [it, inserted] = peers_.emplace(key, State::Pending);
    (void)inserted;
    if (it->second == State::Pending) {
      // Enroll before lookup under the same lock used by notifications. Manager
      // updates its cache before calling us, with no manager locks held: either
      // lookup sees the update or its callback resolves this pending entry later.
      resolve(it->second, lookup(), device, uid);
    }
    return it->second == State::Approved;
  }

  void update(const GidKey & key, const LocalityInfo & info, int device, uint32_t uid)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = peers_.find(key);
    if (it != peers_.end() && it->second == State::Pending) {
      resolve(it->second, info, device, uid);
    }
  }

  bool approved(const GidKey & key) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = peers_.find(key);
    return it != peers_.end() && it->second == State::Approved;
  }

private:
  enum class State {Pending, Rejected, Approved};

  static void resolve(State & state, const LocalityInfo & info, int device, uint32_t uid)
  {
    if (!info.found) {
      return;
    }
    using host_endpoint_manager::EndpointLocality;
    const bool compatible = info.ipc_capable &&
      (info.locality == EndpointLocality::INTRA_PROCESS ||
      (info.locality == EndpointLocality::INTER_PROCESS_SAME_HOST &&
      info.device_id == device && info.uid == uid));
    state = compatible ? State::Approved : State::Rejected;
  }

  mutable std::mutex mutex_;
  std::unordered_map<GidKey, State, host_endpoint_manager::GidKeyHash> peers_;
};

}  // namespace detail
}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER_BACKEND__IPC_DECISIONS_HPP_
