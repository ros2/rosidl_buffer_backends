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

#ifndef CUDA_BUFFER_BACKEND__CUDA_BUFFER_BACKEND_HPP_
#define CUDA_BUFFER_BACKEND__CUDA_BUFFER_BACKEND_HPP_

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer_backend/ipc_metadata.hpp"
#include "rosidl_buffer_backend/buffer_backend.hpp"

namespace cuda_buffer_backend
{

/// \brief CUDA buffer backend plugin for zero-copy GPU memory sharing via VMM IPC.
class CudaBufferBackend : public rosidl::BufferBackend
{
public:
  CudaBufferBackend();
  ~CudaBufferBackend() override = default;

  std::string get_backend_type() const override
  {
    return "cuda";
  }

  const rosidl_message_type_support_t * get_descriptor_type_support() const override;

  std::shared_ptr<void> create_empty_descriptor() const override;

  std::shared_ptr<void> create_descriptor_with_endpoint(
    const void * impl,
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  std::unique_ptr<void, void (*)(void *)> from_descriptor_with_endpoint(
    const void * descriptor,
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  std::pair<bool, std::vector<std::set<uint32_t>>> on_discovering_endpoint(
    const rmw_topic_endpoint_info_t & endpoint_info,
    const std::vector<rmw_topic_endpoint_info_t> & existing_endpoints,
    const std::unordered_map<std::string, std::string> & endpoint_supported_backends) override;

  std::string get_backend_metadata() const override;

private:
  const detail::IpcMetadata & local_metadata() const;
  mutable std::once_flag metadata_once_;
  mutable detail::IpcMetadata local_metadata_;
  mutable std::string metadata_wire_;

  using EndpointKey = std::array<uint8_t, RMW_GID_STORAGE_SIZE>;
  mutable std::map<EndpointKey, detail::IpcDecision> ipc_decision_cache_;
  mutable std::mutex ipc_cache_mutex_;
};

}  // namespace cuda_buffer_backend

#endif  // CUDA_BUFFER_BACKEND__CUDA_BUFFER_BACKEND_HPP_
