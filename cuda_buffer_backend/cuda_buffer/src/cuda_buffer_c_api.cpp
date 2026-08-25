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

#include "cuda_buffer/cuda_buffer_c_api.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "cuda_buffer/cuda_buffer_handle.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "rosidl_buffer/buffer.hpp"

struct cuda_buffer_lease
{
  virtual ~cuda_buffer_lease() = default;
};

namespace
{

thread_local std::array<char, 512> last_error{};

class ReadLease final : public cuda_buffer_lease
{
public:
  explicit ReadLease(cuda_buffer_backend::ReadHandle && handle)
  : handle_(std::move(handle)) {}

  const void * data() const {return handle_.get_ptr();}

private:
  cuda_buffer_backend::ReadHandle handle_;
};

class WriteLease final : public cuda_buffer_lease
{
public:
  explicit WriteLease(cuda_buffer_backend::WriteHandle && handle)
  : handle_(std::move(handle)) {}

  void * data() {return handle_.get_ptr();}

private:
  cuda_buffer_backend::WriteHandle handle_;
};

cudaStream_t resolve_stream(uintptr_t stream)
{
  if (stream == CUDA_BUFFER_STREAM_INTERNAL) {
    return cuda_buffer_backend::get_internal_stream();
  }
  return reinterpret_cast<cudaStream_t>(stream);
}

cuda_buffer_status set_error(cuda_buffer_status status, const char * message) noexcept
{
  std::snprintf(last_error.data(), last_error.size(), "%s", message);
  return status;
}

cuda_buffer_status validate_buffer(
  const rosidl::Buffer<uint8_t> * buffer)
{
  if (!buffer) {
    return set_error(CUDA_BUFFER_STATUS_INVALID_ARGUMENT, "buffer is null");
  }
  if (!buffer->get_impl()) {
    return set_error(CUDA_BUFFER_STATUS_INVALID_ARGUMENT, "buffer implementation is null");
  }
  if (buffer->size() == 0U) {
    return set_error(CUDA_BUFFER_STATUS_EMPTY_BUFFER, "buffer is empty");
  }
  return CUDA_BUFFER_STATUS_OK;
}

cuda_buffer_status acquire_read(
  const void * opaque_buffer,
  uintptr_t stream,
  cuda_buffer_lease ** lease,
  const void ** device_data,
  int32_t * device_id)
{
  if (!lease || !device_data || !device_id) {
    return set_error(CUDA_BUFFER_STATUS_INVALID_ARGUMENT, "output argument is null");
  }
  *lease = nullptr;
  *device_data = nullptr;
  *device_id = -1;

  auto * buffer = static_cast<const rosidl::Buffer<uint8_t> *>(opaque_buffer);
  cuda_buffer_status status = validate_buffer(buffer);
  if (status != CUDA_BUFFER_STATUS_OK) {
    return status;
  }
  const auto * cuda_impl =
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer->get_impl());
  if (!cuda_impl) {
    return set_error(CUDA_BUFFER_STATUS_NOT_CUDA, "buffer is not CUDA-backed");
  }

  try {
    auto result = std::make_unique<ReadLease>(
      cuda_impl->get_cuda_buffer().get_read_handle(resolve_stream(stream)));
    *device_data = result->data();
    *device_id = cuda_impl->get_device_id();
    *lease = result.release();
    last_error[0] = '\0';
    return CUDA_BUFFER_STATUS_OK;
  } catch (const std::exception & error) {
    return set_error(CUDA_BUFFER_STATUS_OPERATION_FAILED, error.what());
  } catch (...) {
    return set_error(CUDA_BUFFER_STATUS_OPERATION_FAILED, "unknown read acquisition error");
  }
}

cuda_buffer_status acquire_write(
  void * opaque_buffer,
  uintptr_t stream,
  cuda_buffer_lease ** lease,
  void ** device_data,
  int32_t * device_id)
{
  if (!lease || !device_data || !device_id) {
    return set_error(CUDA_BUFFER_STATUS_INVALID_ARGUMENT, "output argument is null");
  }
  *lease = nullptr;
  *device_data = nullptr;
  *device_id = -1;

  auto * buffer = static_cast<rosidl::Buffer<uint8_t> *>(opaque_buffer);
  cuda_buffer_status status = validate_buffer(buffer);
  if (status != CUDA_BUFFER_STATUS_OK) {
    return status;
  }
  auto * cuda_impl =
    dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer->get_impl());
  if (!cuda_impl) {
    return set_error(CUDA_BUFFER_STATUS_NOT_CUDA, "buffer is not CUDA-backed");
  }

  try {
    cudaStream_t cuda_stream = resolve_stream(stream);
    cuda_impl->set_stream(cuda_stream);
    auto result = std::make_unique<WriteLease>(
      cuda_impl->get_cuda_buffer().get_write_handle(cuda_stream));
    *device_data = result->data();
    *device_id = cuda_impl->get_device_id();
    *lease = result.release();
    last_error[0] = '\0';
    return CUDA_BUFFER_STATUS_OK;
  } catch (const std::exception & error) {
    return set_error(CUDA_BUFFER_STATUS_OPERATION_FAILED, error.what());
  } catch (...) {
    return set_error(CUDA_BUFFER_STATUS_OPERATION_FAILED, "unknown write acquisition error");
  }
}

void release(cuda_buffer_lease * lease)
{
  delete lease;
}

const char * get_last_error()
{
  return last_error.data();
}

}  // namespace

cuda_buffer_status cuda_buffer_get_api(
  uint32_t requested_version,
  size_t api_size,
  cuda_buffer_api_v1 * api)
{
  if (!api) {
    return set_error(CUDA_BUFFER_STATUS_INVALID_ARGUMENT, "API output is null");
  }
  if (requested_version != CUDA_BUFFER_C_API_VERSION ||
    api_size < sizeof(cuda_buffer_api_v1))
  {
    return set_error(CUDA_BUFFER_STATUS_INCOMPATIBLE_ABI, "unsupported CUDA buffer C ABI");
  }

  cuda_buffer_api_v1 resolved_api{};
  resolved_api.abi_version = CUDA_BUFFER_C_API_VERSION;
  resolved_api.struct_size = sizeof(cuda_buffer_api_v1);
  resolved_api.acquire_read = acquire_read;
  resolved_api.acquire_write = acquire_write;
  resolved_api.release = release;
  resolved_api.get_last_error = get_last_error;
  std::memcpy(api, &resolved_api, sizeof(resolved_api));
  last_error[0] = '\0';
  return CUDA_BUFFER_STATUS_OK;
}
