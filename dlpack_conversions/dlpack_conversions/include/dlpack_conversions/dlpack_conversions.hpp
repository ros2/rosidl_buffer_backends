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

#ifndef DLPACK_CONVERSIONS__DLPACK_CONVERSIONS_HPP_
#define DLPACK_CONVERSIONS__DLPACK_CONVERSIONS_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Shares the DLPACK_DLPACK_H_ include guard with the DLPack header a framework
// adapter may have included first, so both describe one DLManagedTensor and no
// translation is needed. The asserts below fail the build if that header ever
// disagrees with the layout this library was compiled against, rather than
// letting the mismatch corrupt memory at run time.
#include "dlpack_conversions/dlpack.h"
#include "dlpack_conversions/visibility_control.hpp"
#include "tensor_msgs/msg/experimental_tensor.hpp"

static_assert(DLPACK_MAJOR_VERSION == 1, "dlpack_conversions needs DLPack 1.x");
static_assert(sizeof(DLDevice) == 8, "unexpected DLDevice layout");
static_assert(sizeof(DLDataType) == 4, "unexpected DLDataType layout");
static_assert(sizeof(DLTensor) == 48, "unexpected DLTensor layout");
static_assert(offsetof(DLTensor, data) == 0, "unexpected DLTensor layout");
static_assert(offsetof(DLTensor, device) == 8, "unexpected DLTensor layout");
static_assert(offsetof(DLTensor, ndim) == 16, "unexpected DLTensor layout");
static_assert(offsetof(DLTensor, dtype) == 20, "unexpected DLTensor layout");
static_assert(offsetof(DLTensor, shape) == 24, "unexpected DLTensor layout");
static_assert(offsetof(DLTensor, strides) == 32, "unexpected DLTensor layout");
static_assert(
  offsetof(DLTensor, byte_offset) == 40, "unexpected DLTensor layout");
static_assert(
  sizeof(DLManagedTensor) == 64, "unexpected DLManagedTensor layout");
static_assert(
  offsetof(DLManagedTensor, manager_ctx) == 48,
  "unexpected DLManagedTensor layout");
static_assert(
  offsetof(DLManagedTensor, deleter) == 56,
  "unexpected DLManagedTensor layout");

namespace dlpack_conversions
{

using TensorMsg = tensor_msgs::msg::ExperimentalTensor;

/// Owning handle to a DLManagedTensor produced from a tensor message.
///
/// Destroying the handle drops the storage lease. Call release() to hand the
/// tensor to a framework that takes over the DLPack deleter.
class DLPACK_CONVERSIONS_PUBLIC ManagedTensor
{
public:
  ManagedTensor() = default;
  explicit ManagedTensor(DLManagedTensor * tensor);
  ~ManagedTensor();

  ManagedTensor(ManagedTensor && other) noexcept;
  ManagedTensor & operator=(ManagedTensor && other) noexcept;
  ManagedTensor(const ManagedTensor &) = delete;
  ManagedTensor & operator=(const ManagedTensor &) = delete;

  DLManagedTensor * get() const noexcept {return tensor_;}
  DLManagedTensor * release() noexcept;
  explicit operator bool() const noexcept {return tensor_ != nullptr;}

private:
  DLManagedTensor * tensor_{nullptr};
};

/// Backend serving the given DLPack device type, or an empty string when no
/// installed plugin provides it.
DLPACK_CONVERSIONS_PUBLIC std::string backend_for_device(int32_t dl_device_type);

/// Backend used when a caller does not name one. Honours the
/// ROSIDL_TENSOR_BACKEND environment variable, otherwise prefers an
/// accelerator over host memory.
DLPACK_CONVERSIONS_PUBLIC std::string default_backend();

DLPACK_CONVERSIONS_PUBLIC bool backend_available(const std::string & backend);

DLPACK_CONVERSIONS_PUBLIC std::vector<std::string> available_backends();

/// Allocates contiguous storage for a tensor message. An empty backend selects
/// default_backend().
DLPACK_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> allocate_tensor_msg(
  const std::vector<int64_t> & shape,
  DLDataType dtype,
  const std::string & backend = {});

DLPACK_CONVERSIONS_PUBLIC ManagedTensor from_input_tensor_msg(
  const TensorMsg & msg,
  uintptr_t stream);

DLPACK_CONVERSIONS_PUBLIC ManagedTensor from_output_tensor_msg(
  TensorMsg & msg,
  uintptr_t stream);

/// Copies a contiguous source tensor into existing message storage and stamps
/// the message metadata.
DLPACK_CONVERSIONS_PUBLIC void to_tensor_msg(
  TensorMsg & msg,
  const DLTensor & source,
  uintptr_t stream);

/// Allocates a message on the backend matching the source device, then copies.
DLPACK_CONVERSIONS_PUBLIC std::unique_ptr<TensorMsg> to_tensor_msg(
  const DLTensor & source,
  uintptr_t stream);

}  // namespace dlpack_conversions

#endif  // DLPACK_CONVERSIONS__DLPACK_CONVERSIONS_HPP_
