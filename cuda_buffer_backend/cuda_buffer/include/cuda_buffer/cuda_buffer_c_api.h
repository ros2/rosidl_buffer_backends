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

#ifndef CUDA_BUFFER__CUDA_BUFFER_C_API_H_
#define CUDA_BUFFER__CUDA_BUFFER_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "cuda_buffer/visibility_control.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define CUDA_BUFFER_C_API_VERSION 1U
/* Zero selects CUDA's default stream; this value selects the internal stream. */
#define CUDA_BUFFER_STREAM_INTERNAL UINTPTR_MAX

typedef struct cuda_buffer_lease cuda_buffer_lease;

typedef enum cuda_buffer_status
{
  CUDA_BUFFER_STATUS_OK = 0,
  CUDA_BUFFER_STATUS_INVALID_ARGUMENT = 1,
  CUDA_BUFFER_STATUS_INCOMPATIBLE_ABI = 2,
  CUDA_BUFFER_STATUS_NOT_CUDA = 3,
  CUDA_BUFFER_STATUS_EMPTY_BUFFER = 4,
  CUDA_BUFFER_STATUS_OPERATION_FAILED = 5
} cuda_buffer_status;

typedef struct cuda_buffer_api_v1
{
  uint32_t abi_version;
  uint32_t struct_size;

  /*
   * buffer must point to a rosidl::Buffer<uint8_t> created by the same ROS
   * distribution. The pointer remains owned by the caller.
   */
  cuda_buffer_status (*acquire_read)(
    const void * buffer,
    uintptr_t stream,
    cuda_buffer_lease ** lease,
    const void ** device_data,
    int32_t * device_id);

  cuda_buffer_status (*acquire_write)(
    void * buffer,
    uintptr_t stream,
    cuda_buffer_lease ** lease,
    void ** device_data,
    int32_t * device_id);

  void (*release)(cuda_buffer_lease * lease);
  const char * (*get_last_error)(void);
} cuda_buffer_api_v1;

CUDA_BUFFER_PUBLIC cuda_buffer_status cuda_buffer_get_api(
  uint32_t requested_version,
  size_t api_size,
  cuda_buffer_api_v1 * api);

#ifdef __cplusplus
}
#endif

#endif  // CUDA_BUFFER__CUDA_BUFFER_C_API_H_
