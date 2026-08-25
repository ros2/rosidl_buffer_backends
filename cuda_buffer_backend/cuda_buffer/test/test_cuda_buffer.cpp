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
#include <cuda_runtime.h>

#include <cstring>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_c_api.h"
#include "rosidl_buffer/buffer.hpp"

class CudaBufferTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    cudaStreamCreate(&stream1_);
    cudaStreamCreate(&stream2_);
  }

  void TearDown() override
  {
    cudaStreamDestroy(stream1_);
    cudaStreamDestroy(stream2_);
  }

  void allocate_buffer(rosidl::Buffer<uint8_t> & buffer, size_t count)
  {
    auto impl = std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(count);
    buffer = rosidl::Buffer<uint8_t>(std::move(impl));
  }

  void write_pattern(uint8_t * device_ptr, size_t count, uint8_t offset, cudaStream_t stream)
  {
    std::vector<uint8_t> host(count);
    for (size_t i = 0; i < count; ++i) {
      host[i] = static_cast<uint8_t>((offset + i) % 256);
    }
    cudaMemcpyAsync(device_ptr, host.data(), count, cudaMemcpyHostToDevice, stream);
  }

  std::vector<uint8_t> read_to_host(const uint8_t * device_ptr, size_t count, cudaStream_t stream)
  {
    std::vector<uint8_t> host(count);
    cudaMemcpyAsync(host.data(), device_ptr, count, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return host;
  }

  cudaStream_t stream1_{nullptr};
  cudaStream_t stream2_{nullptr};
};

TEST_F(CudaBufferTest, AllocateAndWriteHandle)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 1024);

  cuda_buffer_backend::WriteHandle handle =
    cuda_buffer_backend::from_output_buffer(buffer, stream1_);

  EXPECT_NE(nullptr, handle.get_ptr());
  EXPECT_EQ("cuda", buffer.get_backend_type());
  EXPECT_EQ(1024u, buffer.size());
}

TEST_F(CudaBufferTest, CApiRejectsIncompatibleVersion)
{
  cuda_buffer_api_v1 api{};

  EXPECT_EQ(
    CUDA_BUFFER_STATUS_INCOMPATIBLE_ABI,
    cuda_buffer_get_api(CUDA_BUFFER_C_API_VERSION + 1U, sizeof(api), &api));
  EXPECT_EQ(nullptr, api.acquire_read);
}

TEST_F(CudaBufferTest, CApiAcquiresWriteAndReadLeases)
{
  cuda_buffer_api_v1 api{};
  ASSERT_EQ(
    CUDA_BUFFER_STATUS_OK,
    cuda_buffer_get_api(CUDA_BUFFER_C_API_VERSION, sizeof(api), &api));
  ASSERT_EQ(CUDA_BUFFER_C_API_VERSION, api.abi_version);
  ASSERT_EQ(sizeof(api), api.struct_size);

  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 64);

  cuda_buffer_lease * write_lease = nullptr;
  void * write_data = nullptr;
  int32_t write_device_id = -1;
  ASSERT_EQ(
    CUDA_BUFFER_STATUS_OK,
    api.acquire_write(
      &buffer, reinterpret_cast<uintptr_t>(stream1_), &write_lease,
      &write_data, &write_device_id));
  ASSERT_NE(nullptr, write_lease);
  ASSERT_NE(nullptr, write_data);
  EXPECT_GE(write_device_id, 0);
  write_pattern(static_cast<uint8_t *>(write_data), 64, 17, stream1_);
  api.release(write_lease);

  cuda_buffer_lease * read_lease = nullptr;
  const void * read_data = nullptr;
  int32_t read_device_id = -1;
  ASSERT_EQ(
    CUDA_BUFFER_STATUS_OK,
    api.acquire_read(
      &buffer, reinterpret_cast<uintptr_t>(stream2_), &read_lease,
      &read_data, &read_device_id));
  ASSERT_NE(nullptr, read_lease);
  ASSERT_NE(nullptr, read_data);
  EXPECT_EQ(write_device_id, read_device_id);
  auto result = read_to_host(
    static_cast<const uint8_t *>(read_data), 64, stream2_);
  api.release(read_lease);

  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_EQ(static_cast<uint8_t>((17 + i) % 256), result[i]);
  }
}

TEST_F(CudaBufferTest, CApiPreservesCudaDefaultStream)
{
  cuda_buffer_api_v1 api{};
  ASSERT_EQ(
    CUDA_BUFFER_STATUS_OK,
    cuda_buffer_get_api(CUDA_BUFFER_C_API_VERSION, sizeof(api), &api));

  constexpr size_t size = 4U * 1024U * 1024U;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, size);

  cuda_buffer_lease * write_lease = nullptr;
  void * write_data = nullptr;
  int32_t device_id = -1;
  ASSERT_EQ(
    CUDA_BUFFER_STATUS_OK,
    api.acquire_write(
      &buffer, 0U, &write_lease, &write_data, &device_id));
  ASSERT_EQ(cudaSuccess, cudaMemsetAsync(write_data, 0xAB, size, nullptr));
  api.release(write_lease);

  cuda_buffer_lease * read_lease = nullptr;
  const void * read_data = nullptr;
  ASSERT_EQ(
    CUDA_BUFFER_STATUS_OK,
    api.acquire_read(
      &buffer, reinterpret_cast<uintptr_t>(stream2_), &read_lease,
      &read_data, &device_id));
  auto result = read_to_host(
    static_cast<const uint8_t *>(read_data), size, stream2_);
  api.release(read_lease);

  EXPECT_EQ(size, result.size());
  EXPECT_EQ(0xAB, result.front());
  EXPECT_EQ(0xAB, result.back());
}

TEST_F(CudaBufferTest, CApiRejectsCpuAndEmptyBuffers)
{
  cuda_buffer_api_v1 api{};
  ASSERT_EQ(
    CUDA_BUFFER_STATUS_OK,
    cuda_buffer_get_api(CUDA_BUFFER_C_API_VERSION, sizeof(api), &api));

  rosidl::Buffer<uint8_t> cpu_buffer(16);
  cuda_buffer_lease * lease = nullptr;
  const void * data = nullptr;
  int32_t device_id = -1;
  EXPECT_EQ(
    CUDA_BUFFER_STATUS_NOT_CUDA,
    api.acquire_read(&cpu_buffer, 0U, &lease, &data, &device_id));
  EXPECT_NE(nullptr, api.get_last_error());

  rosidl::Buffer<uint8_t> empty_buffer;
  EXPECT_EQ(
    CUDA_BUFFER_STATUS_EMPTY_BUFFER,
    api.acquire_read(&empty_buffer, 0U, &lease, &data, &device_id));
}

TEST_F(CudaBufferTest, FromBuffer_PromotesCpuBufferForRead)
{
  rosidl::Buffer<uint8_t> buffer(64);
  for (size_t i = 0; i < 64; ++i) {
    buffer[i] = static_cast<uint8_t>(0xA0 + (i % 16));
  }
  const rosidl::Buffer<uint8_t> & cbuf = buffer;

  cuda_buffer_backend::ReadHandle rh =
    cuda_buffer_backend::from_input_buffer(cbuf, stream1_);
  auto promoted = rh.get_promoted_buffer();
  ASSERT_NE(nullptr, promoted);
  EXPECT_EQ(promoted->get_backend_type(), "cuda");
  EXPECT_EQ(promoted->size(), 64u);

  std::vector<uint8_t> readback = read_to_host(rh.get_ptr(), 64, stream1_);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(readback[i], static_cast<uint8_t>(0xA0 + (i % 16)));
  }
}

TEST_F(CudaBufferTest, FromBuffer_PromotesCpuBufferForWrite)
{
  rosidl::Buffer<uint8_t> buffer(64);
  cuda_buffer_backend::WriteHandle wh =
    cuda_buffer_backend::from_output_buffer(buffer, stream1_);
  auto promoted = wh.get_promoted_buffer();
  ASSERT_NE(nullptr, promoted);
  EXPECT_EQ(promoted->get_backend_type(), "cuda");
  EXPECT_EQ(promoted->size(), 64u);
  EXPECT_NE(nullptr, wh.get_ptr());

  write_pattern(wh.get_ptr(), 64, 123, stream1_);
}

TEST_F(CudaBufferTest, FromBuffer_ThrowsOnEmptyBuffer)
{
  rosidl::Buffer<uint8_t> buffer;
  const rosidl::Buffer<uint8_t> & cbuf = buffer;

  EXPECT_THROW(
    cuda_buffer_backend::from_input_buffer(cbuf, stream1_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferTest, ToBuffer_CopiesThroughWriteHandle)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 64);

  std::vector<uint8_t> host_data(64, 0xCD);
  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_output_buffer(buffer, stream1_);
    cuda_buffer_backend::to_buffer(
      host_data.data(), 64, wh, stream1_, cudaMemcpyHostToDevice);
  }

  const auto & cbuf = buffer;
  auto rh = cuda_buffer_backend::from_input_buffer(cbuf, stream1_);
  std::vector<uint8_t> readback = read_to_host(rh.get_ptr(), 64, stream1_);
  EXPECT_EQ(readback[0], 0xCD);
  EXPECT_EQ(readback[63], 0xCD);
}

TEST_F(CudaBufferTest, EventSync_WriteOnStream1_ReadOnStream2)
{
  constexpr size_t N = 1024;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_output_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 42, stream1_);
  }

  std::vector<uint8_t> result;
  {
    const rosidl::Buffer<uint8_t> & cbuf = buffer;
    cuda_buffer_backend::ReadHandle rh =
      cuda_buffer_backend::from_input_buffer(cbuf, stream2_);
    result = read_to_host(rh.get_ptr(), N, stream2_);
  }

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((42 + i) % 256), result[i])
      << "Mismatch at index " << i;
  }
}

TEST_F(CudaBufferTest, DoubleWriteHandle_Throws)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  cuda_buffer_backend::WriteHandle wh =
    cuda_buffer_backend::from_output_buffer(buffer, stream1_);

  EXPECT_THROW(
    cuda_buffer_backend::from_output_buffer(buffer, stream2_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferTest, WriteAfterFinalized_Throws)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_output_buffer(buffer, stream1_);
  }

  EXPECT_THROW(
    cuda_buffer_backend::from_output_buffer(buffer, stream1_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferTest, ReadAfterReadEvents_BlocksWrite)
{
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, 256);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_output_buffer(buffer, stream1_);
  }

  {
    const rosidl::Buffer<uint8_t> & cbuf = buffer;
    cuda_buffer_backend::ReadHandle rh =
      cuda_buffer_backend::from_input_buffer(cbuf, stream2_);
  }

  EXPECT_THROW(
    cuda_buffer_backend::from_output_buffer(buffer, stream1_),
    cuda_buffer_backend::CudaError);
}

TEST_F(CudaBufferTest, Clone_PreservesData)
{
  constexpr size_t N = 512;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_output_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 77, stream1_);
  }

  auto * impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl()));
  impl->set_stream(nullptr);

  auto cloned_impl = impl->clone();
  auto * cloned_cuda = dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    cloned_impl.get());
  ASSERT_NE(nullptr, cloned_cuda);

  const auto & cbuf = cloned_cuda->get_cuda_buffer();
  cuda_buffer_backend::ReadHandle rh = cbuf.get_read_handle(stream2_);
  std::vector<uint8_t> result = read_to_host(rh.get_ptr(), N, stream2_);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((77 + i) % 256), result[i])
      << "Clone mismatch at index " << i;
  }
}

TEST_F(CudaBufferTest, ToCpu_PreservesData)
{
  constexpr size_t N = 1024;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_output_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 55, stream1_);
  }

  auto * impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl()));
  impl->set_stream(nullptr);

  auto cpu_impl = buffer.get_impl()->to_cpu();
  ASSERT_NE(nullptr, cpu_impl);

  auto * cpu = dynamic_cast<rosidl::CpuBufferImpl<uint8_t> *>(cpu_impl.get());
  ASSERT_NE(nullptr, cpu);
  ASSERT_EQ(N, cpu->get_storage().size());

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((55 + i) % 256), cpu->get_storage()[i])
      << "to_cpu mismatch at index " << i;
  }
}

TEST_F(CudaBufferTest, Resize_PreservesPrefix)
{
  constexpr size_t N = 512;
  rosidl::Buffer<uint8_t> buffer;
  allocate_buffer(buffer, N);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_output_buffer(buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 88, stream1_);
  }

  auto * impl = const_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    dynamic_cast<const cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(buffer.get_impl()));
  impl->set_stream(nullptr);
  impl->resize(N * 2);

  EXPECT_EQ(N * 2, impl->size());

  const auto & cbuf = impl->get_cuda_buffer();
  cuda_buffer_backend::ReadHandle rh = cbuf.get_read_handle(stream2_);
  std::vector<uint8_t> result = read_to_host(rh.get_ptr(), N, stream2_);

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((88 + i) % 256), result[i])
      << "Resize prefix mismatch at index " << i;
  }
}

TEST_F(CudaBufferTest, GpuPipeline_NoIntermediateCpuSync)
{
  constexpr size_t N = 4096;
  rosidl::Buffer<uint8_t> src_buffer;
  rosidl::Buffer<uint8_t> dst_buffer;
  allocate_buffer(src_buffer, N);
  allocate_buffer(dst_buffer, N);

  {
    cuda_buffer_backend::WriteHandle wh =
      cuda_buffer_backend::from_output_buffer(src_buffer, stream1_);
    write_pattern(wh.get_ptr(), N, 99, stream1_);
  }

  {
    const rosidl::Buffer<uint8_t> & cbuf = src_buffer;
    cuda_buffer_backend::ReadHandle rh =
      cuda_buffer_backend::from_input_buffer(cbuf, stream2_);
    cuda_buffer_backend::WriteHandle wh2 =
      cuda_buffer_backend::from_output_buffer(dst_buffer, stream2_);
    cudaMemcpyAsync(
      wh2.get_ptr(), rh.get_ptr(), N, cudaMemcpyDeviceToDevice, stream2_);
  }

  std::vector<uint8_t> result;
  {
    const rosidl::Buffer<uint8_t> & cbuf = dst_buffer;
    cuda_buffer_backend::ReadHandle rh =
      cuda_buffer_backend::from_input_buffer(cbuf, stream1_);
    result = read_to_host(rh.get_ptr(), N, stream1_);
  }

  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(static_cast<uint8_t>((99 + i) % 256), result[i])
      << "Mismatch at index " << i;
  }
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
