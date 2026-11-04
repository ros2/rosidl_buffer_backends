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

#include <gtest/gtest.h>

#include "intel_memory_core/gpu_cl_dmabuf_import.hpp"

using namespace intel_memory_core;  // NOLINT

TEST(GpuClDmabufImport, InvalidInputsReturnNullptrRegardlessOfAvailability)
{
  EXPECT_EQ(import_dmabuf_as_cl_mem(nullptr, -1, 4096), nullptr);
  EXPECT_EQ(get_or_import_cl_mem(nullptr, -1, 4096), nullptr);
  EXPECT_EQ(get_or_wrap_host_ptr_cl_mem(nullptr, nullptr, 4096), nullptr);
}

TEST(GpuClDmabufImport, ReleaseNullptrIsNoOp)
{
  EXPECT_NO_THROW(release_cl_mem(nullptr));
  EXPECT_NO_THROW(release_all_cl_mem());
}

TEST(GpuClDmabufImport, AvailabilityMatchesImportBehavior)
{
  if (cl_dmabuf_import_available()) {
    GTEST_SKIP() << "OpenCL dma_buf import compiled in on this machine; "
                    "success path requires a real cl_context and dma_buf fd, "
                    "exercised by the integration pipeline, not this unit test";
  }
  EXPECT_EQ(import_dmabuf_as_cl_mem(reinterpret_cast<void *>(0x1), 3, 4096), nullptr);
}
