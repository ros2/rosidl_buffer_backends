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

#include <limits>
#include <string>

#include "cuda_buffer_backend/ipc_metadata.hpp"

using cuda_buffer_backend::detail::IpcMetadata;
using cuda_buffer_backend::detail::parse_ipc_metadata;
using cuda_buffer_backend::detail::serialize_ipc_metadata;
using cuda_buffer_backend::detail::decide_ipc;

namespace
{
IpcMetadata local()
{
  return {true, {std::string(32, 'c'), "1.2", "3.4", "5.6", "7.8", "9.10"},
    std::string(32, 'a'), std::string(32, 'b'), 0, 1000};
}
}  // namespace

TEST(IpcMetadata, RoundTripAndMiddlewareSafeEncoding)
{
  auto expected = local();
  expected.uid = std::numeric_limits<uint32_t>::max();
  const auto wire = serialize_ipc_metadata(expected);
  EXPECT_LT(wire.size(), 1024u);
  EXPECT_EQ(wire.find_first_of(",:;=/\\"), std::string::npos);
  const auto result = parse_ipc_metadata(wire);
  ASSERT_TRUE(result);
  EXPECT_TRUE(decide_ipc(expected, *result).approved);
  EXPECT_TRUE(decide_ipc(expected, *result).same_process);
  EXPECT_EQ(result->uid, expected.uid);
}

TEST(IpcMetadata, SeparateProcessOnSameIpcEnvironment)
{
  auto peer = local();
  peer.process = std::string(32, 'd');
  const auto result = decide_ipc(local(), peer);
  EXPECT_TRUE(result.approved);
  EXPECT_FALSE(result.same_process);
}

TEST(IpcMetadata, EveryIpcEnvironmentBoundaryRejects)
{
  for (size_t i = 0; i < local().environment.size(); ++i) {
    auto peer = local();
    peer.environment[i] += "0";
    EXPECT_FALSE(decide_ipc(local(), peer).approved) << i;
  }
}

TEST(IpcMetadata, CapabilityDeviceAndUserMismatchesReject)
{
  auto peer = local();
  peer.capable = false;
  EXPECT_FALSE(decide_ipc(local(), peer).approved);
  EXPECT_FALSE(decide_ipc(peer, local()).approved);
  peer = local();
  peer.gpu = std::string(32, 'e');
  EXPECT_FALSE(decide_ipc(local(), peer).approved);  // Equal ordinal, different GPU.
  peer = local();
  ++peer.device;
  EXPECT_FALSE(decide_ipc(local(), peer).approved);  // Remapped ordinals not supported.
  peer = local();
  ++peer.uid;
  EXPECT_FALSE(decide_ipc(local(), peer).approved);
}

TEST(IpcMetadata, DisabledAndUnidentifiableCapabilitiesFailClosed)
{
  EXPECT_EQ(serialize_ipc_metadata({}), "cuda-ipc-v1|0");
  const auto disabled = parse_ipc_metadata("cuda-ipc-v1|0");
  ASSERT_TRUE(disabled);
  EXPECT_FALSE(disabled->capable);
  auto invalid = local();
  invalid.environment[2].clear();
  EXPECT_EQ(serialize_ipc_metadata(invalid), "cuda-ipc-v1|0");
  EXPECT_FALSE(decide_ipc(invalid, invalid).approved);
}

TEST(IpcMetadata, MalformedLegacyAndFutureVersionsReject)
{
  const auto good = serialize_ipc_metadata(local());
  for (const auto & bad : {std::string{}, std::string("cuda-ipc-v2|0"),
      std::string("cuda-ipc-v1|2"), std::string("cuda-ipc-v1|0|"),
      std::string("cuda"), good + "|", good + "|extra", std::string(1025, 'a')})
  {
    EXPECT_FALSE(parse_ipc_metadata(bad)) << bad;
  }
  for (size_t length = 0; length < good.size() - 4; ++length) {
    EXPECT_FALSE(parse_ipc_metadata(good.substr(0, length))) << length;
  }
  for (const auto & bad_number : {"-1", "+1", " 1", "4294967296", "1x", ""}) {
    const auto prefix = good.substr(0, good.rfind('|') + 1);
    EXPECT_FALSE(parse_ipc_metadata(prefix + bad_number));
  }
  auto invalid = good;
  invalid[invalid.find('|', invalid.find('|') + 1) + 1] = ';';
  EXPECT_FALSE(parse_ipc_metadata(invalid));
}
