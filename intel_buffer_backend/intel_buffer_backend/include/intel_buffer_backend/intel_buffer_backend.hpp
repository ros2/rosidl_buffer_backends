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

#ifndef INTEL_BUFFER_BACKEND__INTEL_BUFFER_BACKEND_HPP_
#define INTEL_BUFFER_BACKEND__INTEL_BUFFER_BACKEND_HPP_

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "intel_buffer/intel_buffer_impl.hpp"
#include "rosidl_buffer_backend/buffer_backend.hpp"

namespace intel_buffer_backend
{

class IntelBufferBackend : public rosidl::BufferBackend
{
public:
  IntelBufferBackend();
  ~IntelBufferBackend() override = default;

  std::string get_backend_type() const override
  {
    return "intel_buffer";
  }

  std::string get_backend_metadata() const override;

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

private:
  mutable std::unordered_map<uint32_t, bool> ipc_decision_cache_;
  mutable std::mutex cache_mutex_;
};

}  // namespace intel_buffer_backend

#endif  // INTEL_BUFFER_BACKEND__INTEL_BUFFER_BACKEND_HPP_
