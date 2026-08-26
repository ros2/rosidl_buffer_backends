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

/// Return codes for the CUDA buffer C ABI.
/**
 * Every function in this header is an exception boundary: C++ exceptions raised
 * by the underlying `cuda_buffer` implementation are converted to one of these
 * codes and the associated text is available from
 * `cuda_buffer_error_message()`.
 */
typedef enum cuda_buffer_ret_t
{
  CUDA_BUFFER_RET_OK = 0,
  CUDA_BUFFER_RET_INVALID_ARGUMENT = 1,
  CUDA_BUFFER_RET_BAD_ALLOC = 2,
  CUDA_BUFFER_RET_CUDA_ERROR = 3,
  CUDA_BUFFER_RET_ERROR = 4
} cuda_buffer_ret_t;

/// Opaque scoped read access to a CUDA-backed buffer.
typedef struct cuda_buffer_read_handle_t cuda_buffer_read_handle_t;

/// Opaque scoped write access to a CUDA-backed buffer.
typedef struct cuda_buffer_write_handle_t cuda_buffer_write_handle_t;

/// Message describing the most recent failure on the calling thread.
/**
 * The returned string is owned by the library, is thread-local, and stays valid
 * until the next call into this ABI on the same thread. Never NULL.
 */
CUDA_BUFFER_PUBLIC
const char * cuda_buffer_error_message(void);

/// Get the process-wide internal CUDA stream used when no stream is supplied.
/**
 * \param[out] cuda_stream Receives a `cudaStream_t`.
 */
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_internal_stream(void ** cuda_stream);

/// Allocate a CUDA-backed rosidl::Buffer<uint8_t> of \p byte_count bytes.
/**
 * The caller takes ownership of the returned pointer and must release it with
 * `rosidl_buffer_uint8_destroy()`, the canonical Buffer destruction function.
 *
 * \param[in] byte_count Number of bytes to allocate. Zero yields an empty
 *   CUDA-backed buffer.
 * \param[out] buffer Receives an opaque `rosidl::Buffer<uint8_t> *`.
 */
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_allocate(size_t byte_count, void ** buffer);

/// Report whether \p buffer is backed by the CUDA implementation.
/**
 * \param[in] buffer Opaque `rosidl::Buffer<uint8_t> *`, or NULL.
 */
CUDA_BUFFER_PUBLIC
bool cuda_buffer_is_cuda_backed(const void * buffer);

/// Acquire scoped read access to \p buffer on \p cuda_stream.
/**
 * A non-CUDA buffer is promoted to CUDA memory with a host-to-device copy. The
 * promoted allocation is retained by the returned handle, so the caller never
 * owns it.
 *
 * \param[in] buffer Opaque `rosidl::Buffer<uint8_t> *`.
 * \param[in] cuda_stream `cudaStream_t` to order access on, or NULL for the
 *   internal stream.
 * \param[out] handle Receives a handle that must be released with
 *   `cuda_buffer_read_handle_destroy()`.
 */
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_acquire_read(
  const void * buffer,
  void * cuda_stream,
  cuda_buffer_read_handle_t ** handle);

/// Acquire scoped write access to `*buffer` on \p cuda_stream.
/**
 * When `*buffer` is not CUDA-backed it is promoted: a CUDA-backed buffer of the
 * same length is allocated and, only after handle creation succeeds, `*buffer`
 * is replaced with the new pointer. The caller then owns both the new pointer
 * and the pointer it passed in, and must destroy each with
 * `rosidl_buffer_uint8_destroy()`. The promoted contents are uninitialized;
 * the caller is expected to overwrite them.
 *
 * On failure `*buffer` is left unchanged.
 *
 * \param[inout] buffer Address of an opaque `rosidl::Buffer<uint8_t> *`.
 * \param[in] cuda_stream `cudaStream_t` to order access on, or NULL for the
 *   internal stream.
 * \param[out] handle Receives a handle that must be released with
 *   `cuda_buffer_write_handle_destroy()`.
 */
CUDA_BUFFER_PUBLIC
cuda_buffer_ret_t cuda_buffer_acquire_write(
  void ** buffer,
  void * cuda_stream,
  cuda_buffer_write_handle_t ** handle);

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

/// Destroy a write handle, recording its write event. Accepts NULL.
CUDA_BUFFER_PUBLIC
void cuda_buffer_write_handle_destroy(cuda_buffer_write_handle_t * handle);

#ifdef __cplusplus
}
#endif

#endif  // CUDA_BUFFER__CUDA_BUFFER_C_H_
