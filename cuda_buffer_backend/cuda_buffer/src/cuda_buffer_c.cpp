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

#include "cuda_buffer/cuda_buffer_c.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rosidl_buffer/buffer.hpp"

using cuda_buffer_backend::CudaError;
using CudaBufferHandle = rosidl::Buffer<uint8_t>;

struct cuda_buffer_read_handle_t
{
  cuda_buffer_read_handle_t(cuda_buffer_backend::ReadHandle && h, size_t n)
  : handle(std::move(h)), byte_count(n) {}

  cuda_buffer_backend::ReadHandle handle;
  size_t byte_count;
};

struct cuda_buffer_write_handle_t
{
  cuda_buffer_write_handle_t(
    cuda_buffer_backend::WriteHandle && h, size_t n, cudaStream_t s)
  : handle(std::move(h)), byte_count(n), stream(s) {}

  cuda_buffer_backend::WriteHandle handle;
  size_t byte_count;
  cudaStream_t stream;
};

namespace
{

thread_local char g_error_message[1024]{};

cuda_buffer_ret_t fail(cuda_buffer_ret_t code, const char * message) noexcept
{
  std::snprintf(g_error_message, sizeof(g_error_message), "%s", message);
  return code;
}

/// Run \p fn inside the ABI exception boundary, mapping C++ exceptions to codes.
template<typename Callable>
cuda_buffer_ret_t guarded(Callable && fn)
{
  g_error_message[0] = '\0';
  try {
    return fn();
  } catch (const std::bad_alloc & e) {
    return fail(CUDA_BUFFER_RET_BAD_ALLOC, e.what());
  } catch (const CudaError & e) {
    return fail(CUDA_BUFFER_RET_CUDA_ERROR, e.what());
  } catch (const std::invalid_argument & e) {
    return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, e.what());
  } catch (const std::exception & e) {
    return fail(CUDA_BUFFER_RET_ERROR, e.what());
  } catch (...) {
    return fail(CUDA_BUFFER_RET_ERROR, "unknown exception");
  }
}

}  // namespace

const char * cuda_buffer_error_message(void)
{
  return g_error_message;
}

cuda_buffer_ret_t cuda_buffer_internal_stream(void ** cuda_stream)
{
  return guarded(
    [cuda_stream]() {
      if (!cuda_stream) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "cuda_stream output must not be null");
      }
      *cuda_stream = cuda_buffer_backend::get_internal_stream();
      return CUDA_BUFFER_RET_OK;
    });
}

cuda_buffer_ret_t cuda_buffer_allocate(size_t byte_count, void ** buffer)
{
  return guarded(
    [byte_count, buffer]() {
      if (!buffer) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "buffer output must not be null");
      }
      *buffer = nullptr;
      auto allocated = std::make_unique<CudaBufferHandle>(
        cuda_buffer_backend::allocate_buffer(byte_count));
      *buffer = allocated.release();
      return CUDA_BUFFER_RET_OK;
    });
}

bool cuda_buffer_is_cuda_backed(const void * buffer)
{
  if (!buffer) {
    return false;
  }
  const auto * typed = static_cast<const CudaBufferHandle *>(buffer);
  return dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    typed->get_impl()) != nullptr;
}

cuda_buffer_ret_t cuda_buffer_device_id(const void * buffer, int * device_id)
{
  return guarded(
    [buffer, device_id]() {
      if (!buffer || !device_id) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "buffer and device_id must not be null");
      }
      const auto * typed = static_cast<const CudaBufferHandle *>(buffer);
      const auto * impl = dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
        typed->get_impl());
      if (!impl || typed->size() == 0) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "expected a nonempty CUDA-backed buffer");
      }
      *device_id = impl->get_device_id();
      return CUDA_BUFFER_RET_OK;
    });
}

cuda_buffer_ret_t cuda_buffer_acquire_read(
  const void * buffer,
  void * cuda_stream,
  cuda_buffer_read_handle_t ** handle)
{
  return guarded(
    [buffer, cuda_stream, handle]() {
      if (!handle) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "handle output must not be null");
      }
      *handle = nullptr;
      if (!buffer) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "buffer must not be null");
      }
      const auto * typed = static_cast<const CudaBufferHandle *>(buffer);
      if (typed->size() == 0) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "cannot read an empty buffer");
      }
      auto acquired = std::make_unique<cuda_buffer_read_handle_t>(
        cuda_buffer_backend::from_input_buffer(
          *typed, static_cast<cudaStream_t>(cuda_stream)),
        typed->size());
      *handle = acquired.release();
      return CUDA_BUFFER_RET_OK;
    });
}

cuda_buffer_ret_t cuda_buffer_acquire_write(
  void ** buffer,
  void * cuda_stream,
  cuda_buffer_write_handle_t ** handle)
{
  return guarded(
    [buffer, cuda_stream, handle]() {
      if (!handle) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "handle output must not be null");
      }
      *handle = nullptr;
      if (!buffer || !*buffer) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "buffer must not be null");
      }
      auto * source = static_cast<CudaBufferHandle *>(*buffer);
      if (source->size() == 0) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "cannot write an empty buffer");
      }

      // A promoted allocation transfers to the caller with the write handle.
      std::unique_ptr<CudaBufferHandle> promoted;
      CudaBufferHandle * target = source;
      if (!cuda_buffer_is_cuda_backed(source)) {
        promoted = std::make_unique<CudaBufferHandle>(
          cuda_buffer_backend::allocate_buffer(source->size()));
        target = promoted.get();
      }

      auto stream = static_cast<cudaStream_t>(cuda_stream);
      auto acquired = std::make_unique<cuda_buffer_write_handle_t>(
        cuda_buffer_backend::from_output_buffer(*target, stream), target->size(), stream);

      *handle = acquired.release();
      if (promoted) {
        *buffer = promoted.release();
      }
      return CUDA_BUFFER_RET_OK;
    });
}

cuda_buffer_ret_t cuda_buffer_to_buffer_on_stream(
  const void * source,
  size_t byte_count,
  cuda_buffer_write_handle_t * handle,
  void * cuda_stream,
  cuda_buffer_copy_kind_t kind)
{
  return guarded(
    [source, byte_count, handle, cuda_stream, kind]() {
      if (byte_count == 0) {
        return CUDA_BUFFER_RET_OK;
      }
      if (!source || !handle || byte_count > handle->byte_count) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "invalid copy source, handle, or byte count");
      }
      auto stream = static_cast<cudaStream_t>(cuda_stream);
      if (stream != handle->stream) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "copy stream must match the write handle");
      }
      if (kind != CUDA_BUFFER_COPY_HOST_TO_DEVICE && kind != CUDA_BUFFER_COPY_DEVICE_TO_DEVICE) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "unsupported CUDA copy kind");
      }
      cuda_buffer_backend::to_buffer(
        source, byte_count, handle->handle, stream, static_cast<cudaMemcpyKind>(kind));
      return CUDA_BUFFER_RET_OK;
    });
}

const uint8_t * cuda_buffer_read_handle_data(const cuda_buffer_read_handle_t * handle)
{
  return handle ? handle->handle.get_ptr() : nullptr;
}

uint8_t * cuda_buffer_write_handle_data(cuda_buffer_write_handle_t * handle)
{
  return handle ? handle->handle.get_ptr() : nullptr;
}

size_t cuda_buffer_read_handle_size(const cuda_buffer_read_handle_t * handle)
{
  return handle ? handle->byte_count : 0;
}

size_t cuda_buffer_write_handle_size(const cuda_buffer_write_handle_t * handle)
{
  return handle ? handle->byte_count : 0;
}

void cuda_buffer_read_handle_destroy(cuda_buffer_read_handle_t * handle)
{
  delete handle;
}

void cuda_buffer_write_handle_destroy(cuda_buffer_write_handle_t * handle)
{
  delete handle;
}
