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

// The core cannot depend on a storage plugin without making the dependency
// graph circular, so these cases stay independent of which plugins happen to
// be installed. Conversions over real storage are covered by the plugin
// packages, which may depend on the core.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "dlpack_conversions/dlpack_conversions.hpp"

namespace
{

constexpr DLDataType kFloat32{kDLFloat, 32, 1};

}  // namespace

TEST(DlpackConversions, BackendsAreReportedInNameOrder)
{
  const auto backends = dlpack_conversions::available_backends();

  EXPECT_TRUE(std::is_sorted(backends.begin(), backends.end()));
}

TEST(DlpackConversions, UnknownDeviceTypesHaveNoBackend)
{
  EXPECT_TRUE(dlpack_conversions::backend_for_device(kDLVulkan).empty());
  EXPECT_TRUE(dlpack_conversions::backend_for_device(kDLMetal).empty());
}

TEST(DlpackConversions, UninstalledBackendsAreNotAvailable)
{
  EXPECT_FALSE(dlpack_conversions::backend_available("no_such_backend"));
}

// Naming a backend that is not installed has to fail loudly rather than fall
// back to whatever storage happens to be present.
TEST(DlpackConversions, AllocatingOnAnUninstalledBackendThrows)
{
  EXPECT_THROW(
    dlpack_conversions::allocate_tensor_msg({2}, kFloat32, "no_such_backend"),
    std::runtime_error);
}

TEST(DlpackConversions, AnEmptyMessageYieldsNoTensor)
{
  dlpack_conversions::TensorMsg msg;

  EXPECT_FALSE(dlpack_conversions::from_input_tensor_msg(msg, 0));
  EXPECT_FALSE(dlpack_conversions::from_output_tensor_msg(msg, 0));
}

namespace
{

int g_deleter_calls = 0;

void counting_deleter(DLManagedTensor * tensor)
{
  ++g_deleter_calls;
  delete tensor;
}

DLManagedTensor * make_tensor()
{
  auto * tensor = new DLManagedTensor{};
  tensor->deleter = counting_deleter;
  return tensor;
}

}  // namespace

TEST(ManagedTensor, DestructionRunsTheDlpackDeleter)
{
  g_deleter_calls = 0;
  {
    const dlpack_conversions::ManagedTensor owned(make_tensor());
    EXPECT_TRUE(owned);
    EXPECT_EQ(g_deleter_calls, 0);
  }

  EXPECT_EQ(g_deleter_calls, 1);
}

TEST(ManagedTensor, ADefaultHandleOwnsNothing)
{
  const dlpack_conversions::ManagedTensor empty;

  EXPECT_FALSE(empty);
  EXPECT_EQ(empty.get(), nullptr);
}

TEST(ManagedTensor, ReleasingHandsTheDeleterToTheCaller)
{
  g_deleter_calls = 0;
  dlpack_conversions::ManagedTensor owned(make_tensor());

  DLManagedTensor * raw = owned.release();

  ASSERT_NE(raw, nullptr);
  EXPECT_FALSE(owned);
  EXPECT_EQ(g_deleter_calls, 0);
  raw->deleter(raw);
  EXPECT_EQ(g_deleter_calls, 1);
}

TEST(ManagedTensor, MovingTransfersOwnershipExactlyOnce)
{
  g_deleter_calls = 0;
  {
    dlpack_conversions::ManagedTensor source(make_tensor());
    const auto * expected = source.get();

    const dlpack_conversions::ManagedTensor moved(std::move(source));

    EXPECT_EQ(moved.get(), expected);
    EXPECT_FALSE(source);
    EXPECT_EQ(g_deleter_calls, 0);
  }

  EXPECT_EQ(g_deleter_calls, 1);
}

TEST(ManagedTensor, MoveAssignmentReleasesThePreviousTensor)
{
  g_deleter_calls = 0;
  dlpack_conversions::ManagedTensor first(make_tensor());
  dlpack_conversions::ManagedTensor second(make_tensor());

  first = std::move(second);

  EXPECT_EQ(g_deleter_calls, 1);
  EXPECT_TRUE(first);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
