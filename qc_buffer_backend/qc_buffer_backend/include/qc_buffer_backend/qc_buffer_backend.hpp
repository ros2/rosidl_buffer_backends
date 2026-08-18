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

#ifndef QC_BUFFER_BACKEND__QC_BUFFER_BACKEND_HPP_
#define QC_BUFFER_BACKEND__QC_BUFFER_BACKEND_HPP_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rmw/topic_endpoint_info.h"
#include "rosidl_buffer_backend/buffer_backend.hpp"

namespace qc_buffer_backend
{

/// \brief Qualcomm buffer backend plugin for zero-copy sharing of ION/dma-buf
/// memory between CPU and HTP accelerator.
///
/// Both same-process and cross-process subscribers use the same FdBroker +
/// SCM_RIGHTS + mmap path. The publisher registers each buffer's dma-buf fd
/// with FdBroker; any subscriber (same or different process) connects to the
/// broker socket, receives a fd copy via SCM_RIGHTS, and mmap()s it to access
/// the same physical memory without copying.
class QcBufferBackend : public rosidl::BufferBackend
{
public:
  QcBufferBackend() = default;
  ~QcBufferBackend() override = default;

  std::string get_backend_type() const override {return "qc";}

  const rosidl_message_type_support_t * get_descriptor_type_support() const override;
  std::shared_ptr<void> create_empty_descriptor() const override;

  std::shared_ptr<void> create_descriptor_with_endpoint(
    const void * impl,
    const rmw_topic_endpoint_info_t & endpoint_info) const override;

  std::unique_ptr<void, void (*)(void *)> from_descriptor_with_endpoint(
    const void * descriptor,
    const rmw_topic_endpoint_info_t & endpoint_info) const override;
};

}  // namespace qc_buffer_backend

#endif  // QC_BUFFER_BACKEND__QC_BUFFER_BACKEND_HPP_
