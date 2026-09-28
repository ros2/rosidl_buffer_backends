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

#ifndef CUDA_BUFFER__CUDA_BUFFER_C_H_
#define CUDA_BUFFER__CUDA_BUFFER_C_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cuda_buffer/visibility_control.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Fallible operations return a code and store error text on failure.
typedef enum cuda_buffer_ret_t
{
  CUDA_BUFFER_RET_OK = 0,
  CUDA_BUFFER_RET_INVALID_ARGUMENT = 1,
  CUDA_BUFFER_RET_BAD_ALLOC = 2,
  CUDA_BUFFER_RET_CUDA_ERROR = 3,
  CUDA_BUFFER_RET_ERROR = 4
} cuda_buffer_ret_t;

/// Source memory kinds supported by copies into a CUDA buffer.
typedef enum cuda_buffer_copy_kind_t
{
  CUDA_BUFFER_COPY_HOST_TO_DEVICE = 1,
  CUDA_BUFFER_COPY_DEVICE_TO_DEVICE = 3
} cuda_buffer_copy_kind_t;

/// Opaque scoped read access to a CUDA-backed buffer.
typedef struct cuda_buffer_read_handle_t cuda_buffer_read_handle_t;

/// Opaque scoped write access to a CUDA-backed buffer.
typedef struct cuda_buffer_write_handle_t cuda_buffer_write_handle_t;

/// Non-null, thread-local error text. Copy it before the next fallible ABI call.
CUDA_BUFFER_PUBLIC
const char * cuda_buffer_error_message(void);

/// Store the backend's internal cudaStream_t in cuda_stream.
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_internal_stream(void ** cuda_stream);

/// Allocate byte_count uninitialized bytes; zero creates an empty buffer.
/// The output is a rosidl::Buffer<uint8_t> *, released with rosidl_buffer_uint8_destroy().
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_allocate(size_t byte_count, void ** buffer);

/// Test a rosidl::Buffer<uint8_t> * for CUDA storage. NULL returns false.
CUDA_BUFFER_PUBLIC
bool cuda_buffer_is_cuda_backed(const void * buffer);

/// Get the device ordinal. Rejects CPU-backed and empty buffers without copying.
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_device_id(const void * buffer, int * device_id);

/// Read a rosidl::Buffer<uint8_t> *; NULL selects CUDA default stream 0.
/// The stream must match the buffer's device/context and outlive the handle.
/// Obtain the backend's internal stream with cuda_buffer_internal_stream().
/// Non-CUDA input is copied to a CUDA allocation retained by the read handle.
/// Keep the source buffer alive until cuda_buffer_read_handle_destroy().
/// Submit all reads on cuda_stream before destroying the handle.
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_acquire_read(
  const void * buffer,
  void * cuda_stream,
  cuda_buffer_read_handle_t ** handle);

/// Write a rosidl::Buffer<uint8_t> *; NULL selects CUDA default stream 0.
/// The stream must match the buffer's device/context and outlive the handle.
/// Obtain the backend's internal stream with cuda_buffer_internal_stream().
/// Non-CUDA input is replaced on success with an uninitialized CUDA allocation
/// of the same length. The caller owns both pointers and must release each with
/// rosidl_buffer_uint8_destroy(). Failure leaves *buffer unchanged.
/// Submit all writes on cuda_stream before the buffer is read, published, or
/// destroyed. Release the handle with cuda_buffer_write_handle_destroy(); cleanup
/// remains valid after the buffer finalizes the write and is destroyed.
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_acquire_write(
  void ** buffer,
  void * cuda_stream,
  cuda_buffer_write_handle_t ** handle);

/// Copy bytes asynchronously on the handle's acquisition stream (NULL means 0).
/// Source memory must match kind, remain readable until completion, and not
/// overlap the destination. Rejects oversized copies; zero bytes is a no-op.
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_to_buffer_on_stream(
  const void * source,
  size_t byte_count,
  cuda_buffer_write_handle_t * handle,
  void * cuda_stream,
  cuda_buffer_copy_kind_t kind);

/// Get the device pointer exposed by a read handle, or NULL.
CUDA_BUFFER_PUBLIC
const uint8_t * cuda_buffer_read_handle_data(const cuda_buffer_read_handle_t * handle);

/// Get the device pointer exposed by a write handle, or NULL.
CUDA_BUFFER_PUBLIC
uint8_t * cuda_buffer_write_handle_data(cuda_buffer_write_handle_t * handle);

/// Get the number of accessible bytes behind a read handle, or zero.
CUDA_BUFFER_PUBLIC
size_t cuda_buffer_read_handle_size(const cuda_buffer_read_handle_t * handle);

/// Get the number of accessible bytes behind a write handle, or zero.
CUDA_BUFFER_PUBLIC
size_t cuda_buffer_write_handle_size(const cuda_buffer_write_handle_t * handle);

/// Destroy a read handle, recording its read event. Accepts NULL.
CUDA_BUFFER_PUBLIC
void cuda_buffer_read_handle_destroy(cuda_buffer_read_handle_t * handle);

/// Destroy a write handle, finalizing any outstanding write. Accepts NULL.
CUDA_BUFFER_PUBLIC
void cuda_buffer_write_handle_destroy(cuda_buffer_write_handle_t * handle);

#ifdef __cplusplus
}
#endif

#endif  // CUDA_BUFFER__CUDA_BUFFER_C_H_
