// Copyright 2026 Open Source Robotics Foundation, Inc.
// Modified for the HIP port by ObeliskGate, 2026.
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

#ifndef HIP_BUFFER_BACKEND__HIP_BUFFER_BACKEND_HPP_
#define HIP_BUFFER_BACKEND__HIP_BUFFER_BACKEND_HPP_

#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hip_buffer/hip_buffer_impl.hpp"
#include "hip_buffer/host_endpoint_manager.hpp"
#include "rosidl_buffer_backend/buffer_backend.hpp"

namespace hip_buffer_backend
{

/// \brief HIP buffer backend plugin for zero-copy GPU memory sharing via VMM IPC.
class HipBufferBackend : public rosidl::BufferBackend
{
public:
  HipBufferBackend();
  ~HipBufferBackend() override = default;

  std::string get_backend_type() const override
  {
    return "hip";
  }

  const rosidl_message_type_support_t * get_descriptor_type_support() const override;

  std::shared_ptr<void> create_empty_descriptor() const override;

  std::shared_ptr<void> create_descriptor_with_endpoint(
    const void * impl,
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  std::unique_ptr<void, void (*)(void *)> from_descriptor_with_endpoint(
    const void * descriptor,
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  void on_creating_endpoint(
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  std::pair<bool, std::vector<std::set<uint32_t>>> on_discovering_endpoint(
    const rmw_topic_endpoint_info_t & endpoint_info,
    const std::vector<rmw_topic_endpoint_info_t> & existing_endpoints,
    const std::unordered_map<std::string, std::string> & endpoint_supported_backends) override;

  std::string get_backend_metadata() const override
  {
    return "";
  }

private:
  std::shared_ptr<hip_host_endpoint_manager::HostEndpointManager> get_endpoint_manager() const;

  mutable std::shared_ptr<hip_host_endpoint_manager::HostEndpointManager> endpoint_manager_;
  mutable std::mutex manager_mutex_;

  mutable std::unordered_map<
    hip_host_endpoint_manager::GidKey, bool,
    hip_host_endpoint_manager::GidKeyHash> ipc_decision_cache_;
  mutable std::mutex ipc_cache_mutex_;
};

}  // namespace hip_buffer_backend

#endif  // HIP_BUFFER_BACKEND__HIP_BUFFER_BACKEND_HPP_
