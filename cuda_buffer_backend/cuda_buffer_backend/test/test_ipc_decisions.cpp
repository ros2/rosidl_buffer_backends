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

#include <chrono>
#include <future>

#include "cuda_buffer_backend/ipc_decisions.hpp"

using cuda_buffer_backend::detail::IpcDecisions;
using host_endpoint_manager::EndpointLocality;
using host_endpoint_manager::EntityType;
using host_endpoint_manager::GidKey;
using host_endpoint_manager::LocalityInfo;

namespace
{
GidKey key()
{
  rmw_gid_t gid{};
  return GidKey(gid);
}
LocalityInfo compatible()
{
  return {EndpointLocality::INTER_PROCESS_SAME_HOST, EntityType::PUBLISHER,
    1, 2, 3, true, true};
}
}  // namespace

TEST(IpcDecisions, PendingThenNotificationWithoutRediscovery)
{
  IpcDecisions decisions;
  int lookups = 0;
  EXPECT_FALSE(decisions.discover(key(), true, 2, 3, [&]() {
      ++lookups;
      return LocalityInfo{};
  }));
  for (int i = 0; i < 1000; ++i) {
    EXPECT_FALSE(decisions.approved(key()));
                                                                          }
  decisions.update(key(), compatible(), 2, 3);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(decisions.approved(key()));
                                                                         }
  EXPECT_EQ(lookups, 1);
  auto changed = compatible();
  changed.ipc_capable = false;
  decisions.update(key(), changed, 2, 3);
  EXPECT_TRUE(decisions.approved(key()));  // Resolved peers are stable.
}

TEST(IpcDecisions, SnapshotBeforeDiscoveryAndNoUnsolicitedApproval)
{
  IpcDecisions decisions;
  decisions.update(key(), compatible(), 2, 3);
  EXPECT_FALSE(decisions.approved(key()));
  EXPECT_TRUE(decisions.discover(key(), true, 2, 3, compatible));
}

TEST(IpcDecisions, RegistrationBetweenEnrollmentAndLookup)
{
  IpcDecisions decisions;
  std::promise<void> lookup_entered, cache_updated;
  auto entered = lookup_entered.get_future();
  auto updated = cache_updated.get_future();
  auto discovery = std::async(std::launch::async, [&]() {
        return decisions.discover(key(), true, 2, 3, [&]() {
                 lookup_entered.set_value();
                 updated.wait();
                 return compatible();
    });
  });
  entered.wait();
  auto notification = std::async(std::launch::async, [&]() {
        cache_updated.set_value();
        decisions.update(key(), compatible(), 2, 3);
  });
  EXPECT_TRUE(discovery.get());
  notification.get();
  EXPECT_TRUE(decisions.approved(key()));
}

TEST(IpcDecisions, NotificationAfterMissingLookupCannotBeOverwritten)
{
  IpcDecisions decisions;
  std::promise<void> lookup_entered, notification_ready;
  auto entered = lookup_entered.get_future();
  auto ready = notification_ready.get_future();
  auto discovery = std::async(std::launch::async, [&]() {
        return decisions.discover(key(), true, 2, 3, [&]() {
                 lookup_entered.set_value();
                 ready.wait();
                 return LocalityInfo{};
    });
  });
  entered.wait();
  auto notification = std::async(std::launch::async, [&]() {
        notification_ready.set_value();
        decisions.update(key(), compatible(), 2, 3);
  });
  EXPECT_FALSE(discovery.get());
  notification.get();
  EXPECT_TRUE(decisions.approved(key()));
}

TEST(IpcDecisions, IneligiblePeersNeverEnrolled)
{
  IpcDecisions decisions;
  EXPECT_FALSE(decisions.discover(key(), false, 2, 3, []() {
      ADD_FAILURE() << "Ineligible peer must not perform lookup";
      return compatible();
  }));
  decisions.update(key(), compatible(), 2, 3);
  EXPECT_FALSE(decisions.approved(key()));
}

TEST(IpcDecisions, CompatibilityPredicateMatchesDiscoveryAndNotification)
{
  for (int scenario = 0; scenario < 7; ++scenario) {
    auto info = compatible();
    switch (scenario) {
      case 0: info.ipc_capable = false; break;
      case 1: info.device_id = 8; break;
      case 2: info.uid = 8; break;
      case 3: info.locality = EndpointLocality::INTER_HOST; break;
      case 4: info.locality = EndpointLocality::UNDEFINED; break;
      case 5: info.found = false; break;
      case 6:
        info.locality = EndpointLocality::INTRA_PROCESS;
        info.device_id = 8;
        info.uid = 8;
        break;
    }
    IpcDecisions discovery, notification;
    const bool expected = scenario == 6;
    EXPECT_EQ(discovery.discover(key(), true, 2, 3, [&]() {return info;}), expected);
    EXPECT_FALSE(notification.discover(key(), true, 2, 3, []() {return LocalityInfo{};}));
    notification.update(key(), info, 2, 3);
    EXPECT_EQ(notification.approved(key()), expected);
  }
}
