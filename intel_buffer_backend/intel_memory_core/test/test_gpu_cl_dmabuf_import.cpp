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

#ifdef INTEL_MEMORY_CORE_HAVE_OPENCL

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>

namespace
{
// A real dma-buf (udmabuf over a sealed memfd) of `size` bytes, or -1.
int make_dmabuf(size_t size)
{
  const int memfd = memfd_create("cl_cache_test", MFD_ALLOW_SEALING);
  const int dev = open("/dev/udmabuf", O_RDWR);
  int fd = -1;
  if (memfd >= 0 && dev >= 0 && ftruncate(memfd, static_cast<off_t>(size)) == 0 &&
    fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK) == 0)
  {
    struct udmabuf_create create{};
    create.memfd = static_cast<uint32_t>(memfd);
    create.flags = UDMABUF_FLAGS_CLOEXEC;
    create.size = size;
    fd = ioctl(dev, UDMABUF_CREATE, &create);
  }
  if (dev >= 0) {close(dev);}
  if (memfd >= 0) {close(memfd);}
  return fd;
}

class ClMemCache : public ::testing::Test
{
protected:
  void SetUp() override
  {
    cl_uint num_platforms = 0;
    cl_platform_id platforms[8];
    clGetPlatformIDs(8, platforms, &num_platforms);
    cl_device_id device = nullptr;
    for (cl_uint i = 0; i < num_platforms && !device; ++i) {
      if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) {
        device = nullptr;
      }
    }
    if (device) {
      ctx_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    }
    if (!ctx_) {
      GTEST_SKIP() << "no OpenCL GPU context on this machine";
    }
  }

  void TearDown() override
  {
    release_all_cl_mem();
    if (ctx_) {clReleaseContext(ctx_);}
  }

  cl_context ctx_{nullptr};
};
}  // namespace

// An fd number is reused after close; the cache must not hand out the cl_mem of
// the buffer the number used to refer to.
TEST_F(ClMemCache, ReusedFdNumberForAnotherDmabufGetsItsOwnClMem)
{
  const int fd = make_dmabuf(4096);
  if (fd < 0) {
    GTEST_SKIP() << "udmabuf unavailable on this machine";
  }
  void * first = get_or_import_cl_mem(ctx_, fd, 4096);
  if (!first) {
    close(fd);
    GTEST_SKIP() << "OpenCL dma-buf import unavailable on this machine";
  }

  const int other = make_dmabuf(4096);
  ASSERT_GE(other, 0);
  ASSERT_EQ(dup2(other, fd), fd);   // fd now names a different buffer
  close(other);

  void * second = get_or_import_cl_mem(ctx_, fd, 4096);
  EXPECT_NE(second, nullptr);
  EXPECT_NE(second, first);

  // Another fd for the same buffer is still a cache hit.
  const int same = dup(fd);
  EXPECT_EQ(get_or_import_cl_mem(ctx_, same, 4096), second);
  close(same);
  close(fd);
}

TEST_F(ClMemCache, DmabufCacheHitNeverReturnsASmallerClMem)
{
  const int fd = make_dmabuf(8192);
  if (fd < 0) {
    GTEST_SKIP() << "udmabuf unavailable on this machine";
  }
  void * small = get_or_import_cl_mem(ctx_, fd, 4096);
  if (!small) {
    close(fd);
    GTEST_SKIP() << "OpenCL dma-buf import unavailable on this machine";
  }
  void * whole = get_or_import_cl_mem(ctx_, fd, 8192);
  EXPECT_NE(whole, nullptr);
  EXPECT_NE(whole, small);
  close(fd);
}

TEST_F(ClMemCache, HostPtrCacheHitNeverReturnsASmallerClMem)
{
  void * host = std::aligned_alloc(4096, 8192);
  ASSERT_NE(host, nullptr);
  void * small = get_or_wrap_host_ptr_cl_mem(ctx_, host, 4096);
  if (!small) {
    std::free(host);
    GTEST_SKIP() << "OpenCL USE_HOST_PTR wrap unavailable on this machine";
  }
  void * whole = get_or_wrap_host_ptr_cl_mem(ctx_, host, 8192);
  EXPECT_NE(whole, nullptr);
  EXPECT_NE(whole, small);
  release_all_cl_mem();   // before the memory behind the cl_mems is freed
  std::free(host);
}

#endif  // INTEL_MEMORY_CORE_HAVE_OPENCL
