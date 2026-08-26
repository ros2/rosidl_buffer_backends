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

#include <memory>
#include <new>
#include <stdexcept>
#include <string>
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
  cuda_buffer_write_handle_t(cuda_buffer_backend::WriteHandle && h, size_t n)
  : handle(std::move(h)), byte_count(n) {}

  cuda_buffer_backend::WriteHandle handle;
  size_t byte_count;
};

namespace
{

thread_local std::string g_error_message;

cuda_buffer_ret_t fail(cuda_buffer_ret_t code, const char * message)
{
  g_error_message = message;
  return code;
}

/// Run \p fn inside the ABI exception boundary, mapping C++ exceptions to codes.
template<typename Callable>
cuda_buffer_ret_t guarded(Callable && fn)
{
  g_error_message.clear();
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

cudaStream_t resolve_stream(void * cuda_stream)
{
  if (cuda_stream) {
    return static_cast<cudaStream_t>(cuda_stream);
  }
  return cuda_buffer_backend::get_internal_stream();
}

}  // namespace

const char * cuda_buffer_error_message(void)
{
  return g_error_message.c_str();
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
      if (!buffer) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "buffer must not be null");
      }
      const auto * typed = static_cast<const CudaBufferHandle *>(buffer);
      if (typed->size() == 0) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "cannot read an empty buffer");
      }
      auto acquired = std::make_unique<cuda_buffer_read_handle_t>(
        cuda_buffer_backend::from_input_buffer(*typed, resolve_stream(cuda_stream)),
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
      if (!buffer || !*buffer) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "buffer must not be null");
      }
      auto * source = static_cast<CudaBufferHandle *>(*buffer);
      if (source->size() == 0) {
        return fail(CUDA_BUFFER_RET_INVALID_ARGUMENT, "cannot write an empty buffer");
      }

      // Promote here rather than through from_output_buffer() so the new
      // allocation has a single, caller-visible owner instead of being held only
      // by the handle's shared promoted buffer.
      std::unique_ptr<CudaBufferHandle> promoted;
      CudaBufferHandle * target = source;
      if (!cuda_buffer_is_cuda_backed(source)) {
        promoted = std::make_unique<CudaBufferHandle>(
          cuda_buffer_backend::allocate_buffer(source->size()));
        target = promoted.get();
      }

      auto acquired = std::make_unique<cuda_buffer_write_handle_t>(
        cuda_buffer_backend::from_output_buffer(*target, resolve_stream(cuda_stream)),
        target->size());

      *handle = acquired.release();
      if (promoted) {
        *buffer = promoted.release();
      }
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
