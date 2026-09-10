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

// Mirrors the host plugin suite over device storage. Every case that touches
// the GPU skips when no device is present so the suite stays meaningful on
// CPU-only build machines.

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "dlpack_conversions/dlpack_conversions.hpp"

namespace
{

constexpr DLDataType kFloat32{kDLFloat, 32, 1};
constexpr DLDataType kUint8{kDLUInt, 8, 1};

bool cuda_present()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<int64_t> contiguous(const std::vector<int64_t> & shape)
{
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (size_t index = shape.size(); index > 0; --index) {
    strides[index - 1] = stride;
    stride *= shape[index - 1];
  }
  return strides;
}

DLTensor host_source(
  void * data, std::vector<int64_t> & shape, std::vector<int64_t> & strides,
  DLDataType dtype)
{
  DLTensor tensor{};
  tensor.data = data;
  tensor.device = {kDLCPU, 0};
  tensor.ndim = static_cast<int32_t>(shape.size());
  tensor.dtype = dtype;
  tensor.shape = shape.data();
  tensor.strides = strides.data();
  tensor.byte_offset = 0;
  return tensor;
}

std::vector<float> read_back(const void * device_pointer, size_t count)
{
  std::vector<float> host(count);
  const auto result = cudaMemcpy(
    host.data(), device_pointer, count * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_EQ(result, cudaSuccess);
  return host;
}

class DeviceStorage : public testing::Test
{
protected:
  void SetUp() override
  {
    if (!cuda_present()) {
      GTEST_SKIP() << "no CUDA device is available";
    }
  }
};

}  // namespace

// Discovery does not need a device, so it runs everywhere and proves the
// plugin description file and library are installed where pluginlib looks.
TEST(DeviceStorageDiscovery, ThePluginIsDiscoveredThroughPluginlib)
{
  const auto backends = dlpack_conversions::available_backends();

  EXPECT_NE(
    std::find(backends.begin(), backends.end(), "cuda"), backends.end());
}

TEST(DeviceStorageDiscovery, TheDeviceBackendServesTheCudaDeviceType)
{
  EXPECT_EQ(dlpack_conversions::backend_for_device(kDLCUDA), "cuda");
}

// backend_available answers for the host, so it must report false rather than
// throw when the plugin is installed but the machine has no device.
TEST(DeviceStorageDiscovery, AvailabilityTracksTheHardware)
{
  EXPECT_EQ(dlpack_conversions::backend_available("cuda"), cuda_present());
}

TEST_F(DeviceStorage, AllocationStampsContiguousMetadata)
{
  const auto msg = dlpack_conversions::allocate_tensor_msg(
    {2, 3, 4}, kFloat32, "cuda");

  EXPECT_EQ(
    std::vector<int64_t>(msg->shape.begin(), msg->shape.end()),
    std::vector<int64_t>({2, 3, 4}));
  EXPECT_EQ(
    std::vector<int64_t>(msg->strides.begin(), msg->strides.end()),
    std::vector<int64_t>({12, 4, 1}));
  EXPECT_EQ(msg->dtype_code, kDLFloat);
  EXPECT_EQ(msg->dtype_bits, 32);
  EXPECT_EQ(msg->byte_offset, 0u);
  EXPECT_EQ(msg->data.size(), 2u * 3u * 4u * 4u);
  EXPECT_EQ(msg->data.get_backend_type(), "cuda");
}

TEST_F(DeviceStorage, ViewsCarryTheCudaDeviceAndAliasOneAllocation)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cuda");

  const auto output = dlpack_conversions::from_output_tensor_msg(*msg, 0);
  const auto input = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  ASSERT_TRUE(output);
  ASSERT_TRUE(input);
  EXPECT_EQ(output.get()->dl_tensor.data, input.get()->dl_tensor.data);
  EXPECT_EQ(output.get()->dl_tensor.device.device_type, kDLCUDA);
  EXPECT_GE(output.get()->dl_tensor.device.device_id, 0);
  EXPECT_EQ(output.get()->dl_tensor.ndim, 1);
  EXPECT_EQ(output.get()->dl_tensor.byte_offset, 0u);
}

// A device pointer must not be dereferenced on the host, so the check that it
// really is device memory goes through the CUDA allocator itself.
TEST_F(DeviceStorage, AllocationsLiveInDeviceMemory)
{
  const auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cuda");
  const auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);
  ASSERT_TRUE(view);

  cudaPointerAttributes attributes{};
  ASSERT_EQ(
    cudaPointerGetAttributes(&attributes, view.get()->dl_tensor.data),
    cudaSuccess);

  EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
}

TEST_F(DeviceStorage, AByteOffsetIsFoldedIntoThePointer)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({8}, kUint8, "cuda");
  const auto base = dlpack_conversions::from_input_tensor_msg(*msg, 0);
  ASSERT_TRUE(base);
  auto * const base_pointer = static_cast<uint8_t *>(base.get()->dl_tensor.data);
  msg->shape = {4};
  msg->strides = {1};
  msg->byte_offset = 4;

  const auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  ASSERT_TRUE(view);
  EXPECT_EQ(
    static_cast<uint8_t *>(view.get()->dl_tensor.data), base_pointer + 4);
  EXPECT_EQ(view.get()->dl_tensor.byte_offset, 0u);
}

TEST_F(DeviceStorage, ScalarShapesAreAccepted)
{
  const auto msg = dlpack_conversions::allocate_tensor_msg({}, kFloat32, "cuda");

  const auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  ASSERT_TRUE(view);
  EXPECT_EQ(view.get()->dl_tensor.ndim, 0);
  EXPECT_EQ(msg->data.size(), 4u);
}

TEST_F(DeviceStorage, EmptyStridesFallBackToRowMajor)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({2, 3}, kFloat32, "cuda");
  msg->strides.clear();

  const auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  ASSERT_TRUE(view);
  EXPECT_EQ(view.get()->dl_tensor.strides, nullptr);
}

TEST_F(DeviceStorage, ReleasingHandsOffTheDeleter)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cuda");
  auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);

  DLManagedTensor * raw = view.release();

  ASSERT_NE(raw, nullptr);
  EXPECT_FALSE(view);
  ASSERT_NE(raw->deleter, nullptr);
  raw->deleter(raw);
}

// The bounds checks live in the core rather than the plugin, so they must hold
// for device storage too.
TEST_F(DeviceStorage, ViewsReachingPastTheirStorageAreRejected)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cuda");
  msg->shape = {5};
  msg->strides = {1};

  EXPECT_THROW(
    dlpack_conversions::from_input_tensor_msg(*msg, 0), std::runtime_error);
  EXPECT_THROW(
    dlpack_conversions::from_output_tensor_msg(*msg, 0), std::runtime_error);
}

TEST_F(DeviceStorage, OverlongStridesAreRejected)
{
  auto msg = dlpack_conversions::allocate_tensor_msg({2, 2}, kFloat32, "cuda");
  msg->strides = {8, 1};

  EXPECT_THROW(
    dlpack_conversions::from_input_tensor_msg(*msg, 0), std::runtime_error);
}

// Copying a host tensor into device storage is the path a framework takes when
// its own tensor is not already in a shareable buffer.
TEST_F(DeviceStorage, CopyingFromTheHostLandsTheBytesOnTheDevice)
{
  std::vector<float> source(6);
  std::iota(source.begin(), source.end(), 1.0f);
  std::vector<int64_t> shape{2, 3};
  auto strides = contiguous(shape);
  const auto tensor = host_source(source.data(), shape, strides, kFloat32);
  auto msg = dlpack_conversions::allocate_tensor_msg({2, 3}, kFloat32, "cuda");

  dlpack_conversions::to_tensor_msg(*msg, tensor, 0);

  const auto view = dlpack_conversions::from_input_tensor_msg(*msg, 0);
  ASSERT_TRUE(view);
  const auto copied = read_back(view.get()->dl_tensor.data, source.size());
  EXPECT_TRUE(std::equal(source.begin(), source.end(), copied.begin()));
}

TEST_F(DeviceStorage, CopyingIntoAPreSizedMessageReusesItsStorage)
{
  std::vector<float> source{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape{4};
  auto strides = contiguous(shape);
  const auto tensor = host_source(source.data(), shape, strides, kFloat32);
  auto msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cuda");
  const auto before = dlpack_conversions::from_input_tensor_msg(*msg, 0);
  ASSERT_TRUE(before);
  const void * const before_pointer = before.get()->dl_tensor.data;

  dlpack_conversions::to_tensor_msg(*msg, tensor, 0);

  const auto after = dlpack_conversions::from_input_tensor_msg(*msg, 0);
  ASSERT_TRUE(after);
  EXPECT_EQ(after.get()->dl_tensor.data, before_pointer);
  const auto copied = read_back(after.get()->dl_tensor.data, source.size());
  EXPECT_TRUE(std::equal(source.begin(), source.end(), copied.begin()));
}

TEST_F(DeviceStorage, CopyingIntoATooSmallMessageIsRejected)
{
  std::vector<float> source(6, 1.0f);
  std::vector<int64_t> shape{6};
  auto strides = contiguous(shape);
  const auto tensor = host_source(source.data(), shape, strides, kFloat32);
  auto msg = dlpack_conversions::allocate_tensor_msg({2}, kFloat32, "cuda");

  EXPECT_THROW(
    dlpack_conversions::to_tensor_msg(*msg, tensor, 0), std::runtime_error);
}

// Only the accelerator's plugin can read its own device memory, so a
// host-backed destination still routes through the CUDA plugin. This is the
// path a CPU consumer takes to read a device-backed message.
TEST_F(DeviceStorage, DeviceStorageCopiesBackIntoAHostMessage)
{
  std::vector<float> source(4);
  std::iota(source.begin(), source.end(), 10.0f);
  std::vector<int64_t> shape{4};
  auto strides = contiguous(shape);
  const auto tensor = host_source(source.data(), shape, strides, kFloat32);
  auto device_msg =
    dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cuda");
  dlpack_conversions::to_tensor_msg(*device_msg, tensor, 0);
  const auto device_view =
    dlpack_conversions::from_input_tensor_msg(*device_msg, 0);
  ASSERT_TRUE(device_view);
  auto host_msg = dlpack_conversions::allocate_tensor_msg({4}, kFloat32, "cpu");

  dlpack_conversions::to_tensor_msg(
    *host_msg, device_view.get()->dl_tensor, 0);

  ASSERT_EQ(host_msg->data.size(), source.size() * sizeof(float));
  const auto * landed =
    reinterpret_cast<const float *>(host_msg->data.data());
  EXPECT_TRUE(std::equal(source.begin(), source.end(), landed));
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
