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

#ifndef CUDA_BUFFER_BACKEND__IPC_METADATA_HPP_
#define CUDA_BUFFER_BACKEND__IPC_METADATA_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace cuda_buffer_backend
{
namespace detail
{

// Immutable process/pool capabilities copied into each endpoint's advertisement.
// Environment: boot ID, /dev/shm identity, network/PID/user/IPC namespace identities.
struct IpcMetadata
{
  bool capable{false};
  std::array<std::string, 6> environment;
  std::string process;
  std::string gpu;
  uint32_t device{0};
  uint32_t uid{0};
};

struct IpcDecision
{
  bool approved{false};
  bool same_process{false};
};

std::string serialize_ipc_metadata(const IpcMetadata & metadata);
std::optional<IpcMetadata> parse_ipc_metadata(const std::string & wire);
IpcDecision decide_ipc(const IpcMetadata & local, const IpcMetadata & peer);
IpcMetadata collect_ipc_metadata();

}  // namespace detail
}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER_BACKEND__IPC_METADATA_HPP_
