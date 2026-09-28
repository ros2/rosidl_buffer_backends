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

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <string>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_c.h"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "rosidl_buffer/buffer.hpp"
#include "rosidl_buffer/c_helpers.h"

class CudaBufferCApiTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override
  {
    cudaStreamDestroy(stream_);
  }

  static std::vector<uint8_t> pattern(size_t count, uint8_t offset)
  {
    std::vector<uint8_t> host(count);
    for (size_t i = 0; i < count; ++i) {
      host[i] = static_cast<uint8_t>((offset + i) % 256);
    }
    return host;
  }

  std::vector<uint8_t> read_to_host(const uint8_t * device_ptr, size_t count)
  {
    std::vector<uint8_t> host(count);
    EXPECT_EQ(
      cudaMemcpyAsync(host.data(), device_ptr, count, cudaMemcpyDeviceToHost, stream_),
      cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    return host;
  }

  cudaStream_t stream_{nullptr};
};

TEST_F(CudaBufferCApiTest, WriteThenReadRoundTripsDeviceData)
{
  void * buffer = nullptr;
  ASSERT_EQ(cuda_buffer_allocate(512, &buffer), CUDA_BUFFER_RET_OK);
  ASSERT_NE(buffer, nullptr);
  EXPECT_TRUE(cuda_buffer_is_cuda_backed(buffer));
  const auto * typed = static_cast<const rosidl::Buffer<uint8_t> *>(buffer);
  EXPECT_EQ(typed->size(), 512u);
  EXPECT_EQ(typed->get_backend_type(), "cuda");

  const std::vector<uint8_t> expected = pattern(512, 7);

  void * unchanged = buffer;
  cuda_buffer_write_handle_t * write_handle = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_write(&buffer, stream_, &write_handle), CUDA_BUFFER_RET_OK);
  ASSERT_NE(write_handle, nullptr);
  EXPECT_EQ(buffer, unchanged);
  EXPECT_EQ(cuda_buffer_write_handle_size(write_handle), 512u);

  uint8_t * device_ptr = cuda_buffer_write_handle_data(write_handle);
  ASSERT_NE(device_ptr, nullptr);
  ASSERT_EQ(
    cudaMemcpyAsync(device_ptr, expected.data(), expected.size(), cudaMemcpyHostToDevice, stream_),
    cudaSuccess);
  cuda_buffer_write_handle_destroy(write_handle);

  cuda_buffer_read_handle_t * read_handle = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_read(buffer, stream_, &read_handle), CUDA_BUFFER_RET_OK);
  ASSERT_NE(read_handle, nullptr);
  EXPECT_EQ(cuda_buffer_read_handle_size(read_handle), 512u);

  const uint8_t * read_ptr = cuda_buffer_read_handle_data(read_handle);
  ASSERT_NE(read_ptr, nullptr);
  EXPECT_EQ(read_to_host(read_ptr, expected.size()), expected);

  cuda_buffer_read_handle_destroy(read_handle);
  rosidl_buffer_uint8_destroy(buffer);
}

TEST_F(CudaBufferCApiTest, AcquireWritePromotesCpuBufferAndTransfersOwnership)
{
  const std::vector<uint8_t> source = pattern(128, 3);
  auto * cpu_buffer = new rosidl::Buffer<uint8_t>(source);
  void * buffer = cpu_buffer;

  cuda_buffer_write_handle_t * write_handle = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_write(&buffer, stream_, &write_handle), CUDA_BUFFER_RET_OK);
  ASSERT_NE(write_handle, nullptr);
  ASSERT_NE(buffer, static_cast<void *>(cpu_buffer));
  EXPECT_TRUE(cuda_buffer_is_cuda_backed(buffer));
  EXPECT_EQ(static_cast<rosidl::Buffer<uint8_t> *>(buffer)->size(), source.size());
  EXPECT_EQ(cpu_buffer->get_backend_type(), "cpu");

  const std::vector<uint8_t> replacement = pattern(128, 100);
  uint8_t * device_ptr = cuda_buffer_write_handle_data(write_handle);
  ASSERT_NE(device_ptr, nullptr);
  ASSERT_EQ(
    cudaMemcpyAsync(
      device_ptr, replacement.data(), replacement.size(), cudaMemcpyHostToDevice, stream_),
    cudaSuccess);
  cuda_buffer_write_handle_destroy(write_handle);

  cuda_buffer_read_handle_t * read_handle = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_read(buffer, stream_, &read_handle), CUDA_BUFFER_RET_OK);
  EXPECT_EQ(read_to_host(cuda_buffer_read_handle_data(read_handle), replacement.size()),
    replacement);
  cuda_buffer_read_handle_destroy(read_handle);

  rosidl_buffer_uint8_destroy(buffer);
  rosidl_buffer_uint8_destroy(cpu_buffer);
}

TEST_F(CudaBufferCApiTest, AcquireReadPromotesCpuBufferWithoutTransferringOwnership)
{
  const std::vector<uint8_t> source = pattern(64, 42);
  rosidl::Buffer<uint8_t> cpu_buffer(source);
  EXPECT_FALSE(cuda_buffer_is_cuda_backed(&cpu_buffer));

  cuda_buffer_read_handle_t * read_handle = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_read(&cpu_buffer, stream_, &read_handle), CUDA_BUFFER_RET_OK);
  ASSERT_NE(read_handle, nullptr);
  EXPECT_EQ(cuda_buffer_read_handle_size(read_handle), source.size());

  const uint8_t * device_ptr = cuda_buffer_read_handle_data(read_handle);
  ASSERT_NE(device_ptr, nullptr);
  EXPECT_EQ(read_to_host(device_ptr, source.size()), source);

  cuda_buffer_read_handle_destroy(read_handle);
  EXPECT_EQ(cpu_buffer.get_backend_type(), "cpu");
}

TEST_F(CudaBufferCApiTest, InternalStreamCanBePassedExplicitly)
{
  void * internal_stream = nullptr;
  ASSERT_EQ(cuda_buffer_internal_stream(&internal_stream), CUDA_BUFFER_RET_OK);
  EXPECT_NE(internal_stream, nullptr);

  void * buffer = nullptr;
  ASSERT_EQ(cuda_buffer_allocate(32, &buffer), CUDA_BUFFER_RET_OK);

  cuda_buffer_write_handle_t * write_handle = nullptr;
  ASSERT_EQ(
    cuda_buffer_acquire_write(&buffer, internal_stream, &write_handle), CUDA_BUFFER_RET_OK);
  EXPECT_NE(cuda_buffer_write_handle_data(write_handle), nullptr);
  cuda_buffer_write_handle_destroy(write_handle);

  rosidl_buffer_uint8_destroy(buffer);
}

TEST_F(CudaBufferCApiTest, InvalidArgumentsReportErrorMessages)
{
  void * buffer = nullptr;
  EXPECT_EQ(cuda_buffer_allocate(16, nullptr), CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_FALSE(std::string(cuda_buffer_error_message()).empty());

  cuda_buffer_read_handle_t * read_handle = nullptr;
  EXPECT_EQ(
    cuda_buffer_acquire_read(nullptr, stream_, &read_handle),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(read_handle, nullptr);

  cuda_buffer_write_handle_t * write_handle = nullptr;
  EXPECT_EQ(
    cuda_buffer_acquire_write(&buffer, stream_, &write_handle),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(write_handle, nullptr);

  EXPECT_FALSE(cuda_buffer_is_cuda_backed(nullptr));
  EXPECT_EQ(cuda_buffer_read_handle_data(nullptr), nullptr);
  EXPECT_EQ(cuda_buffer_write_handle_data(nullptr), nullptr);
  EXPECT_EQ(cuda_buffer_read_handle_size(nullptr), 0u);
  EXPECT_EQ(cuda_buffer_write_handle_size(nullptr), 0u);
  cuda_buffer_read_handle_destroy(nullptr);
  cuda_buffer_write_handle_destroy(nullptr);
  rosidl_buffer_uint8_destroy(nullptr);
}

TEST_F(CudaBufferCApiTest, EmptyBufferRejectsHandleAcquisition)
{
  void * buffer = nullptr;
  ASSERT_EQ(cuda_buffer_allocate(0, &buffer), CUDA_BUFFER_RET_OK);
  ASSERT_NE(buffer, nullptr);

  cuda_buffer_read_handle_t * read_handle = nullptr;
  EXPECT_EQ(
    cuda_buffer_acquire_read(buffer, stream_, &read_handle),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(read_handle, nullptr);

  void * unchanged = buffer;
  cuda_buffer_write_handle_t * write_handle = nullptr;
  EXPECT_EQ(
    cuda_buffer_acquire_write(&buffer, stream_, &write_handle),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(write_handle, nullptr);
  EXPECT_EQ(buffer, unchanged);

  rosidl_buffer_uint8_destroy(buffer);
}

TEST_F(CudaBufferCApiTest, RepeatedAcquireReleaseCyclesAreStable)
{
  for (int i = 0; i < 64; ++i) {
    void * buffer = nullptr;
    ASSERT_EQ(cuda_buffer_allocate(4096, &buffer), CUDA_BUFFER_RET_OK);

    cuda_buffer_write_handle_t * write_handle = nullptr;
    ASSERT_EQ(cuda_buffer_acquire_write(&buffer, stream_, &write_handle), CUDA_BUFFER_RET_OK);
    ASSERT_EQ(cudaMemsetAsync(cuda_buffer_write_handle_data(write_handle), i, 4096, stream_),
      cudaSuccess);
    cuda_buffer_write_handle_destroy(write_handle);

    cuda_buffer_read_handle_t * read_handle = nullptr;
    ASSERT_EQ(cuda_buffer_acquire_read(buffer, stream_, &read_handle), CUDA_BUFFER_RET_OK);
    const auto host = read_to_host(cuda_buffer_read_handle_data(read_handle), 4096);
    EXPECT_EQ(host.front(), static_cast<uint8_t>(i));
    EXPECT_EQ(host.back(), static_cast<uint8_t>(i));
    cuda_buffer_read_handle_destroy(read_handle);

    rosidl_buffer_uint8_destroy(buffer);
  }
}

namespace
{
struct StreamGate
{
  std::atomic<bool> open{false};
  static void CUDART_CB wait(void * raw)
  {
    auto * gate = static_cast<StreamGate *>(raw);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!gate->open.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
};
}  // namespace

TEST_F(CudaBufferCApiTest, ExactDefaultStreamRecordsPendingWriteEvent)
{
  void * buffer = nullptr;
  ASSERT_EQ(cuda_buffer_allocate(128, &buffer), CUDA_BUFFER_RET_OK);
  cuda_buffer_write_handle_t * handle = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_write(&buffer, nullptr, &handle), CUDA_BUFFER_RET_OK);
  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(nullptr, StreamGate::wait, &gate), cudaSuccess);
  EXPECT_EQ(cudaMemsetAsync(cuda_buffer_write_handle_data(handle), 42, 128, nullptr), cudaSuccess);
  cuda_buffer_write_handle_destroy(handle);
  auto * impl = dynamic_cast<cuda_buffer_backend::CudaBufferImpl<uint8_t> *>(
    static_cast<rosidl::Buffer<uint8_t> *>(buffer)->get_impl());
  const auto state = cudaEventQuery(impl->get_cuda_buffer().get_write_event());
  gate.open.store(true);
  EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  EXPECT_EQ(state, cudaErrorNotReady);
  rosidl_buffer_uint8_destroy(buffer);
}

TEST_F(CudaBufferCApiTest, ExactDefaultStreamWaitsForNonblockingProducer)
{
  void * buffer = nullptr;
  ASSERT_EQ(cuda_buffer_allocate(128, &buffer), CUDA_BUFFER_RET_OK);
  cudaStream_t producer = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
  cudaEvent_t marker = nullptr;
  ASSERT_EQ(cudaEventCreateWithFlags(&marker, cudaEventDisableTiming), cudaSuccess);
  cuda_buffer_write_handle_t * writer = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_write(&buffer, producer, &writer), CUDA_BUFFER_RET_OK);
  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(producer, StreamGate::wait, &gate), cudaSuccess);
  EXPECT_EQ(cudaMemsetAsync(cuda_buffer_write_handle_data(writer), 73, 128, producer), cudaSuccess);
  cuda_buffer_write_handle_destroy(writer);
  cuda_buffer_read_handle_t * reader = nullptr;
  EXPECT_EQ(cuda_buffer_acquire_read(buffer, nullptr, &reader), CUDA_BUFFER_RET_OK);
  EXPECT_EQ(cudaEventRecord(marker, nullptr), cudaSuccess);
  const auto state = cudaEventQuery(marker);
  gate.open.store(true);
  EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  EXPECT_EQ(cudaStreamSynchronize(producer), cudaSuccess);
  EXPECT_EQ(state, cudaErrorNotReady);
  EXPECT_EQ(read_to_host(cuda_buffer_read_handle_data(reader), 128), std::vector<uint8_t>(128, 73));
  cuda_buffer_read_handle_destroy(reader);
  cudaEventDestroy(marker);
  cudaStreamDestroy(producer);
  rosidl_buffer_uint8_destroy(buffer);
}

TEST_F(CudaBufferCApiTest, ExactStreamRejectsInvalidArgumentsAndClearsOutputs)
{
  cuda_buffer_read_handle_t * read = reinterpret_cast<cuda_buffer_read_handle_t *>(1);
  cuda_buffer_write_handle_t * write = reinterpret_cast<cuda_buffer_write_handle_t *>(1);
  EXPECT_EQ(cuda_buffer_acquire_read(nullptr, nullptr, &read),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(read, nullptr);
  EXPECT_EQ(cuda_buffer_acquire_write(nullptr, nullptr, &write),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(write, nullptr);
  int device = -1;
  EXPECT_EQ(cuda_buffer_device_id(nullptr, &device), CUDA_BUFFER_RET_INVALID_ARGUMENT);
  rosidl::Buffer<uint8_t> cpu(std::vector<uint8_t>(16));
  EXPECT_EQ(cuda_buffer_device_id(&cpu, &device), CUDA_BUFFER_RET_INVALID_ARGUMENT);
}

TEST_F(CudaBufferCApiTest, ExactDefaultStreamReadDefersRecycling)
{
  void * pointer = nullptr;
  ASSERT_EQ(cudaMalloc(&pointer, 128), cudaSuccess);
  auto released = std::make_shared<std::atomic<bool>>(false);
  cuda_buffer_backend::CudaBuffer storage(pointer, 128, 0,
    [released](uint8_t * p) {
      (void)cudaFree(p);
      released->store(true);
    });
  auto * buffer = new rosidl::Buffer<uint8_t>(
    std::make_unique<cuda_buffer_backend::CudaBufferImpl<uint8_t>>(std::move(storage), 128));
  cuda_buffer_read_handle_t * reader = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_read(buffer, nullptr, &reader), CUDA_BUFFER_RET_OK);
  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(nullptr, StreamGate::wait, &gate), cudaSuccess);
  cuda_buffer_read_handle_destroy(reader);
  rosidl_buffer_uint8_destroy(buffer);
  const bool released_before_completion = released->load();
  gate.open.store(true);
  EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!released->load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_FALSE(released_before_completion);
  EXPECT_TRUE(released->load());
}

TEST_F(CudaBufferCApiTest, CopyIntoOutputUsesTheExactDefaultStream)
{
  void * buffer = nullptr;
  ASSERT_EQ(cuda_buffer_allocate(16, &buffer), CUDA_BUFFER_RET_OK);
  std::unique_ptr<void, decltype(& rosidl_buffer_uint8_destroy)> owner(
    buffer, rosidl_buffer_uint8_destroy);
  cuda_buffer_write_handle_t * raw = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_write(&buffer, nullptr, &raw), CUDA_BUFFER_RET_OK);
  std::unique_ptr<cuda_buffer_write_handle_t, decltype(& cuda_buffer_write_handle_destroy)> handle(
    raw, cuda_buffer_write_handle_destroy);
  const auto source = pattern(16, 7);
  EXPECT_EQ(cuda_buffer_to_buffer_on_stream(
      source.data(), source.size(), raw, nullptr, CUDA_BUFFER_COPY_HOST_TO_DEVICE),
    CUDA_BUFFER_RET_OK);
  EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  EXPECT_EQ(read_to_host(cuda_buffer_write_handle_data(raw), 16), source);
}

TEST_F(CudaBufferCApiTest, CopyIntoOutputRejectsInvalidArguments)
{
  void * buffer = nullptr;
  ASSERT_EQ(cuda_buffer_allocate(16, &buffer), CUDA_BUFFER_RET_OK);
  std::unique_ptr<void, decltype(& rosidl_buffer_uint8_destroy)> owner(
    buffer, rosidl_buffer_uint8_destroy);
  cuda_buffer_write_handle_t * raw = nullptr;
  ASSERT_EQ(cuda_buffer_acquire_write(&buffer, stream_, &raw), CUDA_BUFFER_RET_OK);
  std::unique_ptr<cuda_buffer_write_handle_t, decltype(& cuda_buffer_write_handle_destroy)> handle(
    raw, cuda_buffer_write_handle_destroy);
  const auto source = pattern(16, 7);
  EXPECT_EQ(cuda_buffer_to_buffer_on_stream(
      nullptr, 0, nullptr, nullptr, CUDA_BUFFER_COPY_HOST_TO_DEVICE), CUDA_BUFFER_RET_OK);
  EXPECT_EQ(cuda_buffer_to_buffer_on_stream(
      source.data(), 16, nullptr, stream_, CUDA_BUFFER_COPY_HOST_TO_DEVICE),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(cuda_buffer_to_buffer_on_stream(
      nullptr, 16, raw, stream_, CUDA_BUFFER_COPY_HOST_TO_DEVICE),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(cuda_buffer_to_buffer_on_stream(
      source.data(), 17, raw, stream_, CUDA_BUFFER_COPY_HOST_TO_DEVICE),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(cuda_buffer_to_buffer_on_stream(
      source.data(), 16, raw, nullptr, CUDA_BUFFER_COPY_HOST_TO_DEVICE),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(cuda_buffer_to_buffer_on_stream(
      source.data(), 16, raw, stream_, static_cast<cuda_buffer_copy_kind_t>(2)),
    CUDA_BUFFER_RET_INVALID_ARGUMENT);
  EXPECT_EQ(cuda_buffer_to_buffer_on_stream(
      source.data(), 16, raw, stream_, CUDA_BUFFER_COPY_HOST_TO_DEVICE), CUDA_BUFFER_RET_OK);
  EXPECT_EQ(read_to_host(cuda_buffer_write_handle_data(raw), 16), source);
}
