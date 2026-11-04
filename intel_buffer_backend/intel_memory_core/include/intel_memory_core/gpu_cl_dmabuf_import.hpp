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

#ifndef INTEL_MEMORY_CORE__GPU_CL_DMABUF_IMPORT_HPP_
#define INTEL_MEMORY_CORE__GPU_CL_DMABUF_IMPORT_HPP_

#include <cstdint>

namespace intel_memory_core
{

void * import_dmabuf_as_cl_mem(void * cl_context, int fd, uint64_t size);

void release_cl_mem(void * cl_mem);

void * get_or_import_cl_mem(void * cl_context, int fd, uint64_t size);

void * get_or_wrap_host_ptr_cl_mem(void * cl_context, void * host_ptr, uint64_t size);

bool sync_host_ptr_cl_mem(void * cl_context, void * cl_mem, uint64_t size);

void release_all_cl_mem();

bool cl_dmabuf_import_available();

}  // namespace intel_memory_core
#endif  // INTEL_MEMORY_CORE__GPU_CL_DMABUF_IMPORT_HPP_
