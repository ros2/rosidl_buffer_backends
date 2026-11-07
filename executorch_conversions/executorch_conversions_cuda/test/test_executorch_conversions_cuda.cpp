// Copyright 2026 Open Source Robotics Foundation, Inc.
// Copyright 2026 Meta Platforms, Inc. and affiliates.
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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/module/module.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "executorch_conversions/executorch_conversions.hpp"

using executorch::aten::Device;
using executorch::aten::DeviceType;
using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::cuda::CallerStreamGuard;
using executorch::extension::make_tensor_ptr;

namespace
{

const Device kCpu(DeviceType::CPU);
const Device kCuda(DeviceType::CUDA);

// A device without a loadable plugin is a failure, not a skip.
bool cuda_available()
{
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    return false;
  }
  EXPECT_TRUE(executorch_conversions::backend_available("cuda"));
  return executorch_conversions::backend_available("cuda");
}

template<typename T>
std::vector<T> host_values(const executorch::aten::Tensor & tensor)
{
  std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
  auto host = executorch_conversions::allocate_tensor_msg(shape, tensor.scalar_type(), kCpu);
  executorch_conversions::to_tensor_msg(*host, tensor);
  const auto * data = reinterpret_cast<const T *>(host->data.data());
  return {data, data + tensor.numel()};
}

void sleep_on(cudaStream_t stream, std::atomic<bool> * slept = nullptr)
{
  ASSERT_EQ(cudaLaunchHostFunc(stream, [](void * flag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (flag != nullptr) {
        static_cast<std::atomic<bool> *>(flag)->store(true);
      }
    }, slept), cudaSuccess);
}

// Puts a new message on the freed block of a read that still waits on reader, and
// overwrites it. A read that still holds its block keeps the next message off it.
void overwrite_block_during_read(const void * block, cudaStream_t reader, int64_t count)
{
  while (cudaStreamQuery(reader) == cudaErrorNotReady) {
    auto next = executorch_conversions::allocate_tensor_msg({count}, ScalarType::Int, kCuda);
    auto output = executorch_conversions::from_output_tensor_msg(*next);
    if (output->const_data_ptr() == block) {
      ASSERT_EQ(
        cudaMemsetAsync(output->mutable_data_ptr(), 9, 16, cudaStreamPerThread), cudaSuccess);
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// The pool hands the block out again once the read is done, so a test that saw no reuse
// during the read could have seen one. cuda_buffer returns the block on its recycler
// thread, and other free blocks of the size may come first, so they are held until it
// shows up.
void expect_block_reused(const void * block, int64_t count)
{
  std::vector<std::unique_ptr<executorch_conversions::TensorMsg>> others;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (others.size() < 1024 && std::chrono::steady_clock::now() < deadline) {
    auto next = executorch_conversions::allocate_tensor_msg({count}, ScalarType::Int, kCuda);
    if (executorch_conversions::from_output_tensor_msg(*next)->const_data_ptr() == block) {
      return;
    }
    others.push_back(std::move(next));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ADD_FAILURE() << "the block of the finished read was not handed out again";
}

class Stream
{
public:
  Stream() {EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);}
  ~Stream() {cudaStreamDestroy(stream_);}
  cudaStream_t get() const {return stream_;}

private:
  cudaStream_t stream_{nullptr};
};

}  // namespace

TEST(ExecuTorchConversionsCuda, AllocatesCudaStorage)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  EXPECT_EQ(msg->data.get_backend_type(), "cuda");

  auto tensor = executorch_conversions::from_output_tensor_msg(*msg);
  EXPECT_EQ(tensor->device().type(), DeviceType::CUDA);
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, tensor->const_data_ptr()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
}

TEST(ExecuTorchConversionsCuda, DefaultBackendIsCuda)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  EXPECT_EQ(executorch_conversions::default_backend(), "cuda");
  EXPECT_EQ(executorch_conversions::backend_for_device(DeviceType::CUDA), "cuda");
}

TEST(ExecuTorchConversionsCuda, RoundTripsThroughDeviceStorage)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto source = make_tensor_ptr({2, 3}, std::vector<float>{0, 1, 2, 3, 4, 5});
  auto msg = executorch_conversions::allocate_tensor_msg({2, 3}, ScalarType::Float, kCuda);
  executorch_conversions::to_tensor_msg(*msg, *source);

  auto view = executorch_conversions::from_input_tensor_msg(*msg);
  EXPECT_EQ(view->device().type(), DeviceType::CUDA);
  EXPECT_EQ(host_values<float>(*view), (std::vector<float>{0, 1, 2, 3, 4, 5}));
}

TEST(ExecuTorchConversionsCuda, ReadWaitsOnTheProducerStream)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  Stream producer;
  Stream consumer;
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  {
    auto output = executorch_conversions::from_output_tensor_msg(*msg, producer.get());
    ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 0, 16, producer.get()), cudaSuccess);
    sleep_on(producer.get());
    ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 7, 16, producer.get()), cudaSuccess);
  }
  std::vector<int32_t> result(4);
  {
    auto input = executorch_conversions::from_input_tensor_msg(*msg, consumer.get());
    ASSERT_EQ(cudaMemcpyAsync(result.data(), input->const_data_ptr(), 16,
      cudaMemcpyDeviceToHost, consumer.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer.get()), cudaSuccess);
  }
  EXPECT_EQ(result, std::vector<int32_t>(4, 0x07070707));
}

TEST(ExecuTorchConversionsCuda, FollowsTheExecuTorchCallerStream)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  Stream caller;
  Stream consumer;
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  {
    CallerStreamGuard guard(caller.get());
    auto output = executorch_conversions::from_output_tensor_msg(*msg);
    sleep_on(caller.get());
    ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 9, 16, caller.get()), cudaSuccess);
  }
  std::vector<int32_t> result(4);
  {
    auto input = executorch_conversions::from_input_tensor_msg(*msg, consumer.get());
    ASSERT_EQ(cudaMemcpyAsync(result.data(), input->const_data_ptr(), 16,
      cudaMemcpyDeviceToHost, consumer.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer.get()), cudaSuccess);
  }
  EXPECT_EQ(result, std::vector<int32_t>(4, 0x09090909));
}

TEST(ExecuTorchConversionsCuda, InputViewFollowsTheExecuTorchCallerStream)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  Stream producer;
  Stream caller;
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  {
    auto output = executorch_conversions::from_output_tensor_msg(*msg, producer.get());
    ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 0, 16, producer.get()), cudaSuccess);
    sleep_on(producer.get());
    ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 4, 16, producer.get()), cudaSuccess);
  }
  std::vector<int32_t> result(4);
  {
    CallerStreamGuard guard(caller.get());
    auto input = executorch_conversions::from_input_tensor_msg(*msg);
    ASSERT_EQ(cudaMemcpyAsync(result.data(), input->const_data_ptr(), 16,
      cudaMemcpyDeviceToHost, caller.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(caller.get()), cudaSuccess);
  }
  EXPECT_EQ(result, std::vector<int32_t>(4, 0x04040404));
}

TEST(ExecuTorchConversionsCuda, CopyFollowsTheExecuTorchCallerStream)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  Stream caller;
  auto first = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  auto second = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  {
    CallerStreamGuard guard(caller.get());
    auto source = executorch_conversions::from_output_tensor_msg(*first);
    ASSERT_EQ(cudaMemsetAsync(source->mutable_data_ptr(), 0, 16, caller.get()), cudaSuccess);
    sleep_on(caller.get());
    ASSERT_EQ(cudaMemsetAsync(source->mutable_data_ptr(), 3, 16, caller.get()), cudaSuccess);
    executorch_conversions::to_tensor_msg(*second, *source);
  }
  EXPECT_EQ(
    host_values<int32_t>(*executorch_conversions::from_input_tensor_msg(*second)),
    std::vector<int32_t>(4, 0x03030303));
}

TEST(ExecuTorchConversionsCuda, OrdersANullCallerStreamAsTheLegacyStream)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  Stream consumer;
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  {
    CallerStreamGuard guard(nullptr);
    auto output = executorch_conversions::from_output_tensor_msg(*msg);
    ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 0, 16, cudaStreamLegacy), cudaSuccess);
    sleep_on(cudaStreamLegacy);
    ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 5, 16, cudaStreamLegacy), cudaSuccess);
  }
  std::vector<int32_t> result(4);
  {
    auto input = executorch_conversions::from_input_tensor_msg(*msg, consumer.get());
    ASSERT_EQ(cudaMemcpyAsync(result.data(), input->const_data_ptr(), 16,
      cudaMemcpyDeviceToHost, consumer.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer.get()), cudaSuccess);
  }
  EXPECT_EQ(result, std::vector<int32_t>(4, 0x05050505));
}

TEST(ExecuTorchConversionsCuda, RejectedOutputViewKeepsTheMessageWritable)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto msg = executorch_conversions::allocate_tensor_msg({2, 2}, ScalarType::Float, kCuda);
  msg->strides = {1, 2};
  EXPECT_THROW(executorch_conversions::from_output_tensor_msg(*msg), std::runtime_error);

  auto source = make_tensor_ptr({2, 2}, std::vector<float>{0, 1, 2, 3});
  executorch_conversions::to_tensor_msg(*msg, *source);
  EXPECT_EQ(
    host_values<float>(*executorch_conversions::from_input_tensor_msg(*msg)),
    (std::vector<float>{0, 1, 2, 3}));
}

TEST(ExecuTorchConversionsCuda, RejectsASecondWrite)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  auto first = make_tensor_ptr({4}, std::vector<float>{0, 1, 2, 3});
  executorch_conversions::to_tensor_msg(*msg, *first);
  auto other = make_tensor_ptr({4}, std::vector<float>{4, 5, 6, 7});
  EXPECT_THROW(executorch_conversions::to_tensor_msg(*msg, *other), std::exception);
  EXPECT_THROW(executorch_conversions::from_output_tensor_msg(*msg), std::exception);
  EXPECT_EQ(
    host_values<float>(*executorch_conversions::from_input_tensor_msg(*msg)),
    (std::vector<float>{0, 1, 2, 3}));
}

TEST(ExecuTorchConversionsCuda, ReadKeepsStorageUntilTheReaderFinishes)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  // Larger than the pool granularity, so no other test leaves a free block of this size.
  constexpr int64_t kCount = 5 << 18;
  Stream reader;
  // Pinned, so the read stays queued; a copy into pageable memory waits for it.
  void * pinned = nullptr;
  ASSERT_EQ(cudaMallocHost(&pinned, 16), cudaSuccess);
  std::unique_ptr<void, cudaError_t (*)(void *)> pinned_owner(pinned, cudaFreeHost);
  auto * seen = static_cast<int32_t *>(pinned);
  const void * block = nullptr;
  {
    auto msg = executorch_conversions::allocate_tensor_msg({kCount}, ScalarType::Int, kCuda);
    executorch_conversions::to_tensor_msg(
      *msg, *make_tensor_ptr({kCount}, std::vector<int32_t>(kCount, 3)));
    auto input = executorch_conversions::from_input_tensor_msg(*msg, reader.get());
    block = input->const_data_ptr();
    sleep_on(reader.get());
    ASSERT_EQ(cudaMemcpyAsync(seen, input->const_data_ptr(), 16,
      cudaMemcpyDeviceToHost, reader.get()), cudaSuccess);
  }
  overwrite_block_during_read(block, reader.get(), kCount);
  ASSERT_EQ(cudaStreamSynchronize(reader.get()), cudaSuccess);
  expect_block_reused(block, kCount);
  EXPECT_EQ(std::vector<int32_t>(seen, seen + 4), std::vector<int32_t>(4, 3));
}

TEST(ExecuTorchConversionsCuda, ReadWaitsForAWriteOnTheDefaultStream)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  {
    auto output = executorch_conversions::from_output_tensor_msg(*msg);
    ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 0, 16, cudaStreamPerThread), cudaSuccess);
    sleep_on(cudaStreamPerThread);
    ASSERT_EQ(
      cudaMemsetAsync(output->mutable_data_ptr(), 0x5a, 16, cudaStreamPerThread), cudaSuccess);
  }
  Stream reader;
  std::vector<int32_t> seen(4);
  {
    auto input = executorch_conversions::from_input_tensor_msg(*msg, reader.get());
    ASSERT_EQ(cudaMemcpyAsync(seen.data(), input->const_data_ptr(), 16,
      cudaMemcpyDeviceToHost, reader.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(reader.get()), cudaSuccess);
  }
  EXPECT_EQ(seen, std::vector<int32_t>(4, 0x5a5a5a5a));
}

TEST(ExecuTorchConversionsCuda, CopiesAfterPendingWorkOnTheGivenStream)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  Stream stream;
  auto first = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  auto second = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  {
    auto source = executorch_conversions::from_output_tensor_msg(*first, stream.get());
    ASSERT_EQ(cudaMemsetAsync(source->mutable_data_ptr(), 0, 16, stream.get()), cudaSuccess);
    sleep_on(stream.get());
    ASSERT_EQ(cudaMemsetAsync(source->mutable_data_ptr(), 6, 16, stream.get()), cudaSuccess);
    executorch_conversions::to_tensor_msg(*second, *source, stream.get());
  }
  EXPECT_EQ(
    host_values<int32_t>(*executorch_conversions::from_input_tensor_msg(*second)),
    std::vector<int32_t>(4, 0x06060606));
}

TEST(ExecuTorchConversionsCuda, CopiesADeviceTensorIntoADeviceMessage)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto source = make_tensor_ptr({4}, std::vector<float>{1, 2, 3, 4});
  auto first = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  executorch_conversions::to_tensor_msg(*first, *source);
  auto second = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  executorch_conversions::to_tensor_msg(
    *second, *executorch_conversions::from_input_tensor_msg(*first));
  EXPECT_EQ(
    host_values<float>(*executorch_conversions::from_input_tensor_msg(*second)),
    (std::vector<float>{1, 2, 3, 4}));
}

TEST(ExecuTorchConversionsCuda, NewMessagesAreOnTheTensorDevice)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto source = make_tensor_ptr({4}, std::vector<int32_t>{1, 2, 3, 4});
  EXPECT_EQ(executorch_conversions::to_tensor_msg(*source)->data.get_backend_type(), "cpu");
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  executorch_conversions::to_tensor_msg(*msg, *source);
  auto copy = executorch_conversions::to_tensor_msg(
    *executorch_conversions::from_input_tensor_msg(*msg));
  EXPECT_EQ(copy->data.get_backend_type(), "cuda");
  EXPECT_EQ(
    host_values<int32_t>(*executorch_conversions::from_input_tensor_msg(*copy)),
    (std::vector<int32_t>{1, 2, 3, 4}));
}

TEST(ExecuTorchConversionsCuda, CopyIntoADeviceMessageFinishesBeforeReturning)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  Stream stream;
  auto first = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  auto second = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  std::atomic<bool> slept{false};
  {
    auto source = executorch_conversions::from_input_tensor_msg(*first, stream.get());
    sleep_on(stream.get(), &slept);
    executorch_conversions::to_tensor_msg(*second, *source, stream.get());
    // The copy is queued behind the sleep, so it has finished only if the sleep has.
    EXPECT_TRUE(slept);
  }
  // The callback writes slept, so it has to finish before slept goes away.
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
}

TEST(ExecuTorchConversionsCuda, MessageOutlivesTheStreamThatWroteIt)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto source = make_tensor_ptr({4}, std::vector<float>{1, 2, 3, 4});
  auto viewed = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  auto copied = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  {
    Stream stream;
    {
      auto output = executorch_conversions::from_output_tensor_msg(*viewed, stream.get());
      ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 0, 16, stream.get()), cudaSuccess);
    }
    executorch_conversions::to_tensor_msg(*copied, *source, stream.get());
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  }
  const auto viewed_copy = *viewed;
  const auto copied_copy = *copied;
  EXPECT_EQ(
    host_values<float>(*executorch_conversions::from_input_tensor_msg(viewed_copy)),
    std::vector<float>(4, 0));
  EXPECT_EQ(
    host_values<float>(*executorch_conversions::from_input_tensor_msg(copied_copy)),
    (std::vector<float>{1, 2, 3, 4}));
}

TEST(ExecuTorchConversionsCuda, CopiesIntoStorageOnACallerSuppliedStream)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  Stream stream;
  auto source = make_tensor_ptr({32}, std::vector<float>(32, 5));
  auto msg = executorch_conversions::to_tensor_msg(*source, stream.get());
  EXPECT_EQ(msg->data.get_backend_type(), "cpu");

  auto device_msg = executorch_conversions::allocate_tensor_msg({32}, ScalarType::Float, kCuda);
  executorch_conversions::to_tensor_msg(*device_msg, *source, stream.get());
  auto readback = executorch_conversions::from_input_tensor_msg(*device_msg, stream.get());
  EXPECT_EQ(host_values<float>(*readback), std::vector<float>(32, 5));
}

TEST(ExecuTorchConversionsCuda, ViewsAByteOffset)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto msg = executorch_conversions::allocate_tensor_msg({8}, ScalarType::Int, kCuda);
  executorch_conversions::to_tensor_msg(
    *msg, *make_tensor_ptr({8}, std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  msg->shape = {6};
  msg->byte_offset = 8;
  EXPECT_EQ(
    host_values<int32_t>(*executorch_conversions::from_input_tensor_msg(*msg)),
    (std::vector<int32_t>{2, 3, 4, 5, 6, 7}));
}

TEST(ExecuTorchConversionsCuda, RejectsACopyIntoItsOwnStorage)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto msg = executorch_conversions::allocate_tensor_msg({8}, ScalarType::Int, kCuda);
  msg->shape = {6};
  msg->byte_offset = 8;
  auto view = executorch_conversions::from_input_tensor_msg(*msg);
  EXPECT_THROW(executorch_conversions::to_tensor_msg(*msg, *view), std::invalid_argument);
  view.reset();
  ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  // The rejected copy must leave the message's one write unused.
  executorch_conversions::to_tensor_msg(*msg, *make_tensor_ptr({6}, std::vector<int32_t>(6, 7)));
  EXPECT_EQ(
    host_values<int32_t>(*executorch_conversions::from_input_tensor_msg(*msg)),
    std::vector<int32_t>(6, 7));
}

TEST(ExecuTorchConversionsCuda, RejectsACopyFromAnotherDevice)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto source = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  auto view = executorch_conversions::from_input_tensor_msg(*source);
  // Refused before either buffer is touched, so the device index alone reaches it.
  auto stamped = make_tensor_ptr(
    {4}, view->mutable_data_ptr(), {}, {}, ScalarType::Int, Device(DeviceType::CUDA, 1));
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  EXPECT_THROW(executorch_conversions::to_tensor_msg(*msg, *stamped), std::invalid_argument);
}

TEST(ExecuTorchConversionsCuda, ReleasingAViewOnAnotherThreadKeepsItsOrder)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  auto output = executorch_conversions::from_output_tensor_msg(*msg);
  ASSERT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 0, 16, cudaStreamPerThread), cudaSuccess);
  sleep_on(cudaStreamPerThread);
  ASSERT_EQ(
    cudaMemsetAsync(output->mutable_data_ptr(), 0x5a, 16, cudaStreamPerThread), cudaSuccess);
  std::thread([&output] {output.reset();}).join();
  Stream reader;
  std::vector<int32_t> seen(4);
  {
    auto input = executorch_conversions::from_input_tensor_msg(*msg, reader.get());
    ASSERT_EQ(cudaMemcpyAsync(seen.data(), input->const_data_ptr(), 16,
      cudaMemcpyDeviceToHost, reader.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(reader.get()), cudaSuccess);
  }
  EXPECT_EQ(seen, std::vector<int32_t>(4, 0x5a5a5a5a));
}

TEST(ExecuTorchConversionsCuda, ReleasingAReadOnAnotherThreadWaitsOutAGraphRecording)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  // A size no other test uses, so the freed block is the only free one of its size.
  constexpr int64_t kCount = 9 << 18;
  void * pinned = nullptr;
  ASSERT_EQ(cudaMallocHost(&pinned, 16), cudaSuccess);
  std::unique_ptr<void, cudaError_t (*)(void *)> pinned_owner(pinned, cudaFreeHost);
  auto * seen = static_cast<int32_t *>(pinned);
  auto msg = executorch_conversions::allocate_tensor_msg({kCount}, ScalarType::Int, kCuda);
  executorch_conversions::to_tensor_msg(
    *msg, *make_tensor_ptr({kCount}, std::vector<int32_t>(kCount, 3)));
  // The ExecuTorch CUDA backend records its graphs in relaxed mode.
  std::atomic<int> phase{0};
  cudaError_t began = cudaSuccess;
  cudaError_t ended = cudaSuccess;
  std::thread recorder([&phase, &began, &ended] {
      Stream stream;
      began = cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeRelaxed);
      phase = 1;
      while (phase.load() != 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      cudaGraph_t graph = nullptr;
      ended = cudaStreamEndCapture(stream.get(), &graph);
      if (graph != nullptr) {
        cudaGraphDestroy(graph);
      }
      (void)cudaGetLastError();
    });
  while (phase.load() != 1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto input = executorch_conversions::from_input_tensor_msg(*msg);
  const void * block = input->const_data_ptr();
  sleep_on(cudaStreamPerThread);
  // Not ASSERT: returning before the recorder is joined would end the test binary.
  EXPECT_EQ(cudaMemcpyAsync(seen, input->const_data_ptr(), 16,
    cudaMemcpyDeviceToHost, cudaStreamPerThread), cudaSuccess);
  std::thread([&input, &phase] {
      phase = 2;
      input.reset();
    }).join();
  // A release that did not wait for the read lets the block go back while the read is queued.
  msg.reset();
  Stream writer;
  bool reused = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (!reused && std::chrono::steady_clock::now() < deadline) {
    auto next = executorch_conversions::allocate_tensor_msg({kCount}, ScalarType::Int, kCuda);
    auto output = executorch_conversions::from_output_tensor_msg(*next, writer.get());
    if (output->const_data_ptr() == block) {
      EXPECT_EQ(cudaMemsetAsync(output->mutable_data_ptr(), 9, 16, writer.get()), cudaSuccess);
      reused = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  recorder.join();
  ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(writer.get()), cudaSuccess);
  ASSERT_EQ(began, cudaSuccess);
  // The release was refused while the recording was open, which ended the recording.
  EXPECT_EQ(ended, cudaErrorStreamCaptureInvalidated);
  EXPECT_TRUE(reused);
  EXPECT_EQ(std::vector<int32_t>(seen, seen + 4), std::vector<int32_t>(4, 3));
}

TEST(ExecuTorchConversionsCuda, ValidatesDeviceIndices)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg(
      {1}, ScalarType::Float, Device(DeviceType::CUDA, static_cast<int8_t>(count))),
    std::runtime_error);
  EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
  auto msg = executorch_conversions::allocate_tensor_msg(
    {1}, ScalarType::Float, Device(DeviceType::CUDA, 0));
  EXPECT_EQ(executorch_conversions::from_input_tensor_msg(*msg)->device().index(), 0);
}

// cuda_buffer keeps one memory pool per process, on the first device it uses, so
// with device 0 used first a message on device 1 is refused.
TEST(ExecuTorchConversionsCuda, RestoresTheCurrentDevice)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "Needs 2 GPUs";
  }
  auto msg = executorch_conversions::allocate_tensor_msg({1}, ScalarType::Float, kCuda);
  int current = -1;
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  EXPECT_THROW(
    executorch_conversions::allocate_tensor_msg(
      {1}, ScalarType::Float, Device(DeviceType::CUDA, 1)),
    std::runtime_error);
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 0);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  auto other = executorch_conversions::allocate_tensor_msg({1}, ScalarType::Float, kCuda);
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 1);
  executorch_conversions::from_input_tensor_msg(*msg).reset();
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 1);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}

TEST(ExecuTorchConversionsCuda, RejectsAStreamOnAnotherDevice)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "Needs 2 GPUs";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto msg = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  auto source = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Int, kCuda);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  {
    Stream other;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW(
      executorch_conversions::from_output_tensor_msg(*msg, other.get()), std::invalid_argument);
    EXPECT_THROW(
      executorch_conversions::from_input_tensor_msg(*msg, other.get()), std::invalid_argument);
    EXPECT_THROW(
      executorch_conversions::to_tensor_msg(
        *msg, *executorch_conversions::from_input_tensor_msg(*source), other.get()),
      std::invalid_argument);
    CallerStreamGuard guard(other.get());
    EXPECT_THROW(executorch_conversions::from_input_tensor_msg(*msg), std::invalid_argument);
  }
  // Refused before the write handle is taken, so the message still accepts its write.
  executorch_conversions::to_tensor_msg(*msg, *make_tensor_ptr({4}, std::vector<int32_t>(4, 2)));
  EXPECT_EQ(
    host_values<int32_t>(*executorch_conversions::from_input_tensor_msg(*msg)),
    std::vector<int32_t>(4, 2));
}

TEST(ExecuTorchConversionsCuda, ReleasingAViewWithAnotherCurrentDeviceKeepsTheRead)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "Needs 2 GPUs";
  }
  // A size no other test uses, so the freed block is the only free one of its size.
  constexpr int64_t kCount = 7 << 18;
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  Stream reader;
  void * pinned = nullptr;
  ASSERT_EQ(cudaMallocHost(&pinned, 16), cudaSuccess);
  std::unique_ptr<void, cudaError_t (*)(void *)> pinned_owner(pinned, cudaFreeHost);
  auto * seen = static_cast<int32_t *>(pinned);
  const void * block = nullptr;
  {
    auto msg = executorch_conversions::allocate_tensor_msg({kCount}, ScalarType::Int, kCuda);
    executorch_conversions::to_tensor_msg(
      *msg, *make_tensor_ptr({kCount}, std::vector<int32_t>(kCount, 3)));
    auto input = executorch_conversions::from_input_tensor_msg(*msg, reader.get());
    block = input->const_data_ptr();
    sleep_on(reader.get());
    ASSERT_EQ(cudaMemcpyAsync(seen, input->const_data_ptr(), 16,
      cudaMemcpyDeviceToHost, reader.get()), cudaSuccess);
    // The release must record the read on the message's device, not the current one.
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    input.reset();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
  overwrite_block_during_read(block, reader.get(), kCount);
  ASSERT_EQ(cudaStreamSynchronize(reader.get()), cudaSuccess);
  expect_block_reused(block, kCount);
  EXPECT_EQ(std::vector<int32_t>(seen, seen + 4), std::vector<int32_t>(4, 3));
}

// Set EXECUTORCH_CONVERSIONS_TEST_CUDA_ADD_ONE_PTE to a CUDA backend program
// computing x + 1 for a float tensor of shape [4] whose inputs and outputs stay
// in device memory and are not memory planned, and
// EXECUTORCH_CONVERSIONS_TEST_CUDA_ADD_ONE_PTD to its data file. On a machine with
// a CUDA GPU, python3 executorch_conversions/executorch_conversions_cpu/test/export_add_one.py
// --cuda add_one.pte writes both. It is the only test that runs the plugin with
// the CUDA backend's stream, so run it whenever the ExecuTorch pin changes and
// before relying on a reused ExecuTorch install.
TEST(ExecuTorchConversionsCuda, CudaBackendReadsAndWritesMessageStorage)
{
  if (!cuda_available()) {
    GTEST_SKIP() << "CUDA driver or plugin is unavailable";
  }
  const char * path = std::getenv("EXECUTORCH_CONVERSIONS_TEST_CUDA_ADD_ONE_PTE");
  const char * data_path = std::getenv("EXECUTORCH_CONVERSIONS_TEST_CUDA_ADD_ONE_PTD");
  if (path == nullptr || data_path == nullptr) {
    GTEST_SKIP() << "EXECUTORCH_CONVERSIONS_TEST_CUDA_ADD_ONE_PTE or _PTD is not set";
  }
  Module module(path, data_path);
  const auto method_meta = module.method_meta("forward");
  ASSERT_TRUE(method_meta.ok());
  const auto input_meta = method_meta->input_tensor_meta(0);
  ASSERT_TRUE(input_meta.ok());
  // The runtime copies a planned input, so the program would not read message storage.
  EXPECT_FALSE(input_meta->is_memory_planned());
  auto values = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  // Values no other test writes, so a reused block cannot already hold them.
  executorch_conversions::to_tensor_msg(
    *values, *make_tensor_ptr({4}, std::vector<float>{10, 20, 30, 40}));
  const auto run = [&](const auto & request, auto & reply) {
      auto input = executorch_conversions::from_input_tensor_msg(request);
      auto output = executorch_conversions::from_output_tensor_msg(reply);
      ASSERT_EQ(module.set_output(*output), executorch::runtime::Error::Ok);
      const auto result = module.forward(*input);
      ASSERT_TRUE(result.ok()) << static_cast<int>(result.error());
      EXPECT_EQ(result->at(0).toTensor().const_data_ptr(), output->const_data_ptr());
    };
  // The first run waits for the whole GPU while it allocates, which would hide a missing order.
  auto warmup = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  executorch_conversions::to_tensor_msg(*warmup, *make_tensor_ptr({4}, std::vector<float>(4)));
  auto warmup_reply = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  run(*warmup, *warmup_reply);
  auto request = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  auto reply = executorch_conversions::allocate_tensor_msg({4}, ScalarType::Float, kCuda);
  {
    // Written on another stream with no host wait, so only the message's events
    // order the program after it.
    Stream producer;
    CallerStreamGuard guard(producer.get());
    auto source = executorch_conversions::from_input_tensor_msg(*values);
    auto output = executorch_conversions::from_output_tensor_msg(*request);
    sleep_on(producer.get());
    ASSERT_EQ(cudaMemcpyAsync(output->mutable_data_ptr(), source->const_data_ptr(), 16,
      cudaMemcpyDeviceToDevice, producer.get()), cudaSuccess);
  }
  run(*request, *reply);
  auto view = executorch_conversions::from_input_tensor_msg(*reply);
  EXPECT_EQ(host_values<float>(*view), (std::vector<float>{11, 21, 31, 41}));
}
