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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

namespace
{

using onnxruntime_conversions::BackendConfiguration;
using onnxruntime_conversions::ConversionBackendRegistry;
using onnxruntime_conversions::TensorMsg;
using onnxruntime_conversions::allocate_tensor_msg;
using onnxruntime_conversions::available_backends;
using onnxruntime_conversions::configure_session_options;
using onnxruntime_conversions::from_input_tensor_msg;
using onnxruntime_conversions::from_output_tensor_msg;
using onnxruntime_conversions::to_tensor_msg;

const uint8_t identity_model[] = {
  8, 10, 58, 88, 10, 25, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111,
  117, 116, 112, 117, 116, 34, 8, 73, 100, 101, 110, 116, 105, 116, 121,
  18, 8, 105, 100, 101, 110, 116, 105, 116, 121, 90, 23, 10, 5, 105,
  110, 112, 117, 116, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 2, 10,
  2, 8, 3, 98, 24, 10, 6, 111, 117, 116, 112, 117, 116, 18, 14, 10,
  12, 8, 1, 18, 8, 10, 2, 8, 2, 10, 2, 8, 3, 66, 4, 10, 0, 16, 18};

class CudaPluginTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
      (void)cudaGetLastError();
      GTEST_SKIP() << "CUDA device is unavailable";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
  }

  void TearDown() override
  {
    if (stream_) {
      EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
    }
  }

  Ort::MemoryInfo memory_info_{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
  cudaStream_t stream_{nullptr};
};

TEST_F(CudaPluginTest, DiscoversAndAllocatesCudaStorage)
{
  const auto backends = available_backends();
  EXPECT_NE(std::find(backends.begin(), backends.end(), "cuda"), backends.end());
  auto msg = allocate_tensor_msg(
    {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda");
  EXPECT_EQ(msg->data.get_backend_type(), "cuda");
  EXPECT_EQ(msg->data.size(), 6u * sizeof(float));

  std::shared_ptr<TensorMsg> owner(std::move(msg));
  auto lease = ConversionBackendRegistry::instance().get_backend("cuda")->acquire_output(
    owner, stream_);
  EXPECT_NE(lease->metadata().data, nullptr);
  EXPECT_EQ(lease->metadata().size_bytes, 6u * sizeof(float));
  EXPECT_EQ(lease->metadata().device_type, OrtMemoryInfoDeviceType_GPU);
  EXPECT_EQ(lease->metadata().device_id, 0);
  EXPECT_EQ(lease->metadata().allocator_name, "Cuda");
}

TEST_F(CudaPluginTest, AutoSelectsCudaOnlyWithExplicitUsableStream)
{
  BackendConfiguration configuration;
  configuration.device_id = 0;
  configuration.execution_stream = stream_;
  auto cuda_msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, configuration);
  EXPECT_EQ(cuda_msg->data.get_backend_type(), "cuda");

  BackendConfiguration no_stream;
  auto cpu_msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, no_stream);
  EXPECT_EQ(cpu_msg->data.get_backend_type(), "cpu");
}

TEST_F(CudaPluginTest, ExplicitCpuOverridesStreamAndAutoDoesNotFallback)
{
  BackendConfiguration configuration;
  configuration.execution_stream = stream_;
  auto cpu_msg = allocate_tensor_msg(
    {4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu", configuration);
  EXPECT_EQ(cpu_msg->data.get_backend_type(), "cpu");

  EXPECT_THROW(
    allocate_tensor_msg(
      {0}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, configuration),
    std::invalid_argument);
}

TEST_F(CudaPluginTest, RejectsMissingOrImplicitStream)
{
  std::shared_ptr<TensorMsg> msg(
    allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"));
  EXPECT_THROW(
    from_output_tensor_msg(msg, memory_info_, nullptr),
    std::invalid_argument);
  EXPECT_THROW(
    from_output_tensor_msg(
      msg, memory_info_, reinterpret_cast<void *>(cudaStreamPerThread)),
    std::invalid_argument);

  Ort::SessionOptions options;
  EXPECT_THROW(configure_session_options(options, "cuda"), std::invalid_argument);
}

TEST_F(CudaPluginTest, InputAndOutputViewsAliasCudaStorage)
{
  std::shared_ptr<TensorMsg> msg(
    allocate_tensor_msg({4}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"));
  void * pointer = nullptr;
  {
    auto output = from_output_tensor_msg(msg, memory_info_, stream_);
    pointer = output.value().GetTensorMutableRawData();
  }
  std::shared_ptr<const TensorMsg> input = msg;
  auto view = from_input_tensor_msg(input, memory_info_, stream_);
  EXPECT_EQ(view.value().GetTensorRawData(), pointer);
}

TEST_F(CudaPluginTest, RejectsZeroByteCudaTensors)
{
  EXPECT_THROW(
    allocate_tensor_msg(
      {2, 0, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"),
    std::invalid_argument);
}

TEST_F(CudaPluginTest, LeasePreservesOwnerAndOrdersAcrossStreams)
{
  cudaStream_t consumer_stream = nullptr;
  ASSERT_EQ(
    cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking),
    cudaSuccess);
  std::shared_ptr<TensorMsg> msg(
    allocate_tensor_msg({1024}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "cuda"));
  std::weak_ptr<TensorMsg> weak = msg;
  void * pointer = nullptr;
  {
    auto output = from_output_tensor_msg(msg, memory_info_, stream_);
    pointer = output.value().GetTensorMutableRawData();
    ASSERT_EQ(cudaMemsetAsync(pointer, 0x5a, 1024, stream_), cudaSuccess);
    msg.reset();
    EXPECT_FALSE(weak.expired());
  }
  EXPECT_TRUE(weak.expired());

  std::shared_ptr<TensorMsg> ordered(
    allocate_tensor_msg({1024}, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "cuda"));
  {
    auto output = from_output_tensor_msg(ordered, memory_info_, stream_);
    ASSERT_EQ(
      cudaMemsetAsync(output.value().GetTensorMutableRawData(), 0xa5, 1024, stream_),
      cudaSuccess);
  }
  std::vector<uint8_t> host(1024);
  {
    std::shared_ptr<const TensorMsg> input = ordered;
    auto view = from_input_tensor_msg(input, memory_info_, consumer_stream);
    ASSERT_EQ(
      cudaMemcpyAsync(
        host.data(), view.value().GetTensorRawData(), host.size(),
        cudaMemcpyDeviceToHost, consumer_stream),
      cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(consumer_stream), cudaSuccess);
  EXPECT_TRUE(std::all_of(host.begin(), host.end(), [](uint8_t value) {
      return value == 0xa5;
    }));
  EXPECT_EQ(cudaStreamDestroy(consumer_stream), cudaSuccess);
}

TEST_F(CudaPluginTest, CopiesCudaOrtValueWithoutHostStaging)
{
  const std::vector<int64_t> shape{4};
  float * source = nullptr;
  ASSERT_EQ(cudaMalloc(&source, shape[0] * sizeof(float)), cudaSuccess);
  const std::vector<float> expected{1.0F, 2.0F, 3.0F, 4.0F};
  ASSERT_EQ(
    cudaMemcpyAsync(
      source, expected.data(), expected.size() * sizeof(float),
      cudaMemcpyHostToDevice, stream_),
    cudaSuccess);
  auto value = Ort::Value::CreateTensor<float>(
    memory_info_, source, expected.size(), shape.data(), shape.size());
  auto msg = to_tensor_msg(value, "cuda", stream_);

  std::vector<float> actual(expected.size());
  {
    std::shared_ptr<const TensorMsg> owner(std::move(msg));
    auto view = from_input_tensor_msg(owner, memory_info_, stream_);
    ASSERT_EQ(
      cudaMemcpyAsync(
        actual.data(), view.value().GetTensorRawData(),
        actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
      cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  ASSERT_EQ(cudaFree(source), cudaSuccess);
  EXPECT_EQ(actual, expected);
}

TEST_F(CudaPluginTest, CopyFromOrtReturnsBeforeStreamCompletes)
{
  struct CallbackState
  {
    std::mutex mutex;
    std::condition_variable condition;
    bool released{false};
  } state;

  const std::vector<int64_t> shape{4};
  float * source = nullptr;
  ASSERT_EQ(cudaMalloc(&source, shape[0] * sizeof(float)), cudaSuccess);
  auto value = Ort::Value::CreateTensor<float>(
    memory_info_, source, shape[0], shape.data(), shape.size());
  ASSERT_EQ(
    cudaLaunchHostFunc(
      stream_, [](void * data) {
        auto * callback_state = static_cast<CallbackState *>(data);
        std::unique_lock<std::mutex> lock(callback_state->mutex);
        callback_state->condition.wait(
          lock, [callback_state]() {return callback_state->released;});
      }, &state),
    cudaSuccess);

  auto result = std::async(std::launch::async, [&]() {
        return to_tensor_msg(value, "cuda", stream_);
    });
  const auto status = result.wait_for(std::chrono::milliseconds(200));
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.released = true;
  }
  state.condition.notify_one();
  EXPECT_EQ(status, std::future_status::ready);
  auto msg = result.get();
  ASSERT_NE(msg, nullptr);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  ASSERT_EQ(cudaFree(source), cudaSuccess);
}

TEST_F(CudaPluginTest, ConfiguresProviderAndRunsCudaInference)
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_conversions_cuda_test");
  Ort::SessionOptions options;
  BackendConfiguration configuration;
  configuration.device_id = 0;
  configuration.execution_stream = stream_;
  ASSERT_NO_THROW(configure_session_options(options, "cuda", configuration));
  Ort::Session session(env, identity_model, sizeof(identity_model), options);
  Ort::IoBinding binding(session);

  std::shared_ptr<TensorMsg> input(
    allocate_tensor_msg({2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"));
  std::shared_ptr<TensorMsg> output(
    allocate_tensor_msg({2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"));
  const std::vector<float> expected{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  {
    auto view = from_output_tensor_msg(input, memory_info_, stream_);
    ASSERT_EQ(
      cudaMemcpyAsync(
        view.value().GetTensorMutableRawData(), expected.data(),
        expected.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
      cudaSuccess);
  }
  {
    std::shared_ptr<const TensorMsg> const_input = input;
    auto input_view = from_input_tensor_msg(const_input, memory_info_, stream_);
    auto output_view = from_output_tensor_msg(output, memory_info_, stream_);
    binding.BindInput("input", input_view.value());
    binding.BindOutput("output", output_view.value());
    Ort::RunOptions run_options;
    session.Run(run_options, binding);
  }

  std::vector<float> actual(expected.size());
  {
    std::shared_ptr<const TensorMsg> result = output;
    auto view = from_input_tensor_msg(result, memory_info_, stream_);
    ASSERT_EQ(
      cudaMemcpyAsync(
        actual.data(), view.value().GetTensorRawData(),
        actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
      cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  EXPECT_EQ(actual, expected);
}

}  // namespace
