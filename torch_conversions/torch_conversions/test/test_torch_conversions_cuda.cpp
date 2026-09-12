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
#include <torch/torch.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <stdexcept>
#include <thread>

#include "torch_conversions/torch_conversions.hpp"

namespace
{

void * current_stream()
{
  return c10::cuda::getCurrentCUDAStream().stream();
}

}  // namespace

TEST(TorchConversionsCuda, CloneWaitsOnTheSuppliedConsumerStream)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }
  const auto producer = c10::cuda::getStreamFromPool();
  const auto consumer = c10::cuda::getStreamFromPool();
  auto msg = torch_conversions::allocate_tensor_msg({4}, at::kFloat, c10::kCUDA);
  {
    c10::cuda::CUDAStreamGuard guard(producer);
    auto output = torch_conversions::from_output_tensor_msg(*msg, producer.stream());
    output.zero_();
    producer.synchronize();
    ASSERT_EQ(cudaLaunchHostFunc(producer.stream(), [](void *) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }, nullptr), cudaSuccess);
    output.fill_(7);
  }
  auto clone = torch_conversions::from_input_tensor_msg(*msg, true, consumer.stream());
  consumer.synchronize();
  EXPECT_TRUE(torch::equal(clone.cpu(), torch::full({4}, 7.0)));
}

TEST(TorchConversionsCuda, NonContiguousCopyUsesTheSuppliedStream)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }
  const auto stream = c10::cuda::getStreamFromPool();
  at::Tensor source;
  {
    c10::cuda::CUDAStreamGuard guard(stream);
    source = torch::zeros({2, 3}, at::device(at::kCUDA).dtype(at::kFloat));
    stream.synchronize();
    ASSERT_EQ(cudaLaunchHostFunc(stream.stream(), [](void *) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }, nullptr), cudaSuccess);
    source.fill_(5);
  }
  auto msg = torch_conversions::to_tensor_msg(source.transpose(0, 1), stream.stream());
  source = at::Tensor{};
  auto value = torch_conversions::from_input_tensor_msg(*msg, true, stream.stream());
  stream.synchronize();
  EXPECT_TRUE(torch::equal(value.cpu(), torch::full({3, 2}, 5.0)));
}

TEST(TorchConversionsCuda, InputCloneSurvivesMessageDestruction)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }
  const auto stream = c10::cuda::getStreamFromPool();
  c10::cuda::CUDAStreamGuard guard(stream);
  auto msg = torch_conversions::allocate_tensor_msg({4}, at::kFloat, c10::kCUDA);
  auto output = torch_conversions::from_output_tensor_msg(*msg, stream.stream());
  auto * pointer = output.data_ptr();
  output.fill_(9);
  output = at::Tensor{};
  auto input = torch_conversions::from_input_tensor_msg(*msg, true, stream.stream());
  msg.reset();
  EXPECT_NE(input.data_ptr(), pointer);
  EXPECT_TRUE(torch::equal(input.cpu(), torch::full({4}, 9.0)));
}

TEST(TorchConversionsCuda, PreservesAndValidatesDeviceIndices)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  EXPECT_THROW(torch_conversions::set_stream(c10::Device(c10::kCUDA, count)), c10::Error);
  EXPECT_THROW(torch_conversions::allocate_tensor_msg(
      {1}, at::kFloat, c10::Device(c10::kCUDA, count)), c10::Error);
  const int current = c10::cuda::current_device();
  for (int device = 0; device < count; ++device) {
    if (device != current) {
      EXPECT_THROW(torch_conversions::allocate_tensor_msg(
          {1}, at::kFloat, c10::Device(c10::kCUDA, device)), std::runtime_error);
      continue;
    }
    auto msg = torch_conversions::allocate_tensor_msg(
      {1}, at::kFloat, c10::Device(c10::kCUDA, device));
    auto value = torch_conversions::from_output_tensor_msg(*msg);
    EXPECT_EQ(value.device().index(), device);
  }
}

TEST(TorchConversionsCuda, AllocatesCudaStorage)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }

  auto msg = torch_conversions::allocate_tensor_msg(
    {4}, at::kFloat, c10::kCUDA);
  EXPECT_EQ(msg->data.get_backend_type(), "cuda");

  auto tensor = torch_conversions::from_output_tensor_msg(*msg);
  EXPECT_TRUE(tensor.is_cuda());
  tensor.fill_(1.0);
}

TEST(TorchConversionsCuda, RoundTripsOnACallerSuppliedStream)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }

  auto msg = torch_conversions::allocate_tensor_msg(
    {64}, at::kFloat, c10::kCUDA);

  {
    auto output = torch_conversions::from_output_tensor_msg(
      *msg, current_stream());
    ASSERT_TRUE(output.is_cuda());
    output.fill_(3.0);
  }

  auto input = torch_conversions::from_input_tensor_msg(
    *msg, /*clone=*/true, current_stream());
  ASSERT_TRUE(input.is_cuda());
  EXPECT_FLOAT_EQ(input.sum().item<float>(), 192.0f);
}

TEST(TorchConversionsCuda, CopiesIntoStorageOnACallerSuppliedStream)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }

  const auto source = at::full({32}, 5.0, at::device(at::kCUDA).dtype(at::kFloat));

  auto msg = torch_conversions::to_tensor_msg(source, current_stream());
  ASSERT_EQ(msg->data.get_backend_type(), "cuda");

  auto readback = torch_conversions::from_input_tensor_msg(
    *msg, /*clone=*/true, current_stream());
  EXPECT_FLOAT_EQ(readback.sum().item<float>(), 160.0f);
}

TEST(TorchConversionsCuda, ScopedStreamRestoresState)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA driver is unavailable";
  }

  const auto original = c10::cuda::getCurrentCUDAStream();
  {
    auto guard = torch_conversions::set_stream();
    EXPECT_EQ(
      c10::cuda::getCurrentCUDAStream() != original,
      torch_conversions::default_backend() == "cuda");
  }
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(), original);
  {
    auto guard = torch_conversions::set_stream(original.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    EXPECT_NE(stream, original);
    EXPECT_NE(stream, c10::cuda::getDefaultCUDAStream());
    {
      auto cpu_guard = torch_conversions::set_stream(c10::kCPU);
      EXPECT_EQ(c10::cuda::getCurrentCUDAStream(), stream);
    }
    try {
      torch_conversions::StreamGuard nested(original.device());
      EXPECT_NE(c10::cuda::getCurrentCUDAStream(), stream);
      throw std::runtime_error("unwind stream guard");
    } catch (const std::runtime_error &) {
      EXPECT_EQ(c10::cuda::getCurrentCUDAStream(), stream);
    }

    auto msg = torch_conversions::allocate_tensor_msg({32}, at::kFloat, original.device());
    {
      auto output = torch_conversions::from_output_tensor_msg(*msg);
      output.fill_(7.0);
    }
    auto input = torch_conversions::from_input_tensor_msg(*msg);
    EXPECT_FLOAT_EQ(input.sum().item<float>(), 224.0f);
  }
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(), original);
  EXPECT_EQ(c10::cuda::current_device(), original.device_index());
}
