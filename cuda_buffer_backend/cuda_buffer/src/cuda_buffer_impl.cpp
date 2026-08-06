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

#include "cuda_buffer/cuda_buffer_impl.hpp"

#include <rcutils/logging_macros.h>

#include <memory>

namespace cuda_buffer_backend
{

cudaStream_t get_internal_stream()
{
  static cudaStream_t stream = [] {
      cudaStream_t result = nullptr;
      cudaError_t error = cudaStreamCreateWithFlags(&result, cudaStreamNonBlocking);
      if (error != cudaSuccess) {
        RCUTILS_LOG_WARN_NAMED(
          "cuda_buffer_backend",
          "Failed to create internal CUDA stream (%s); "
          "clone/resize/to_cpu will use the default (synchronizing) stream",
          cudaGetErrorName(error));
        (void)cudaGetLastError();
      }
      return result;
    }();
  return stream;
}

std::shared_ptr<CudaMemoryPool> get_or_create_global_pool()
{
  static std::shared_ptr<CudaMemoryPool> global_pool = [] {
      auto pool = std::make_shared<CudaMemoryPool>();
      const CUresult result = pool->create();
      if (result != CUDA_SUCCESS) {
        throw CudaError(__FILE__, __LINE__, "CudaMemoryPool::create", result);
      }
      return pool;
    }();
  return global_pool;
}

}  // namespace cuda_buffer_backend
