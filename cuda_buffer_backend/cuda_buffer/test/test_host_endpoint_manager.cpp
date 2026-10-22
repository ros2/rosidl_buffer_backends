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

#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>

#include "cuda_buffer/host_endpoint_manager.hpp"

extern char ** environ;
using host_endpoint_manager::HostEndpointManager;
using host_endpoint_manager::EndpointLocality;

namespace
{
rmw_gid_t gid(uint8_t value)
{
  rmw_gid_t result{};
  result.data[0] = value;
  return result;
}

size_t domain()
{
  static std::atomic<size_t> next{static_cast<size_t>(getpid()) * 100};
  return next++;
}

std::string registry_name(size_t id)
{
  char hostname[256]{};
  gethostname(hostname, sizeof(hostname));
  uint32_t hash = 0;
  for (char c : std::string(hostname)) {
    hash = hash * 31 + static_cast<uint32_t>(c);
  }
  return "/ros2_hem_v2_d" + std::to_string(id) + "_h" + std::to_string(hash);
}

struct Registry
{
  const size_t id = domain();
  ~Registry() {shm_unlink(registry_name(id).c_str());}
};

bool read_byte(int fd, char expected)
{
  pollfd event{fd, POLLIN, 0};
  if (poll(&event, 1, 5000) != 1) {return false;}
  char value = 0;
  return read(fd, &value, 1) == 1 && value == expected;
}

class Worker
{
public:
  Worker(size_t domain_id, const char * mode)
  {
    int input[2], output[2];
    if (pipe(input) || pipe(output)) {throw std::runtime_error("pipe");}
    input_ = input[1];
    output_ = output[0];
    const auto id = std::to_string(domain_id);
    const auto in = std::to_string(input[0]);
    const auto out = std::to_string(output[1]);
    // Child configuration is passed as argv, never through shared environment.
    char * args[] = {const_cast<char *>("/proc/self/exe"),
      const_cast<char *>("--hem-worker"), const_cast<char *>(mode),
      const_cast<char *>(id.c_str()), const_cast<char *>(in.c_str()),
      const_cast<char *>(out.c_str()), nullptr};
    const int result = posix_spawn(&pid_, args[0], nullptr, nullptr, args, environ);
    close(input[0]);
    close(output[1]);
    if (result != 0) {throw std::runtime_error("spawn");}
  }
  ~Worker()
  {
    kill(pid_, SIGKILL);
    waitpid(pid_, nullptr, 0);
    close(input_);
    close(output_);
  }
  bool expect(char value) {return read_byte(output_, value);}
  void send(char value) {ASSERT_EQ(write(input_, &value, 1), 1);}

private:
  pid_t pid_ = -1;
  int input_ = -1;
  int output_ = -1;
};

int worker_main(char ** argv)
{
  const size_t id = std::stoull(argv[3]);
  const int input = std::stoi(argv[4]);
  const int output = std::stoi(argv[5]);
  if (std::string(argv[2]) == "lock") {
    const int fd = shm_open(registry_name(id).c_str(), O_RDWR, 0600);
    if (fd < 0 || flock(fd, LOCK_EX) != 0) {return 1;}
    if (write(output, "L", 1) != 1) {return 2;}
    char command;
    return read(input, &command, 1) == 1 ? 0 : 3;
  }
  auto manager = HostEndpointManager::get_instance(id);
  auto token = manager->subscribe_endpoint_updates(
    [output](const auto & key, const auto & info) {
      if (key.data[0] == 42 && info.found &&
      info.locality == EndpointLocality::INTER_PROCESS_SAME_HOST)
      {
        (void)write(output, "N", 1);
      }
    });
  if (write(output, "R", 1) != 1) {return 4;}
  char command;
  if (read(input, &command, 1) != 1) {return 5;}
  return 0;
}
}  // namespace

TEST(HostEndpointManager, LocalUpdateSnapshotAndExceptionIsolation)
{
  Registry registry;
  auto manager = HostEndpointManager::get_instance(registry.id);
  manager->set_ipc_capable(true);
  auto bad = manager->subscribe_endpoint_updates([](const auto &, const auto &) {
        throw std::runtime_error("subscriber failure");
  });
  std::promise<void> delivered;
  auto token = manager->subscribe_endpoint_updates([&](const auto & key, const auto & info) {
        EXPECT_EQ(key, host_endpoint_manager::GidKey(gid(1)));
        EXPECT_TRUE(info.ipc_capable);
        EXPECT_EQ(info.locality, EndpointLocality::INTRA_PROCESS);
        EXPECT_TRUE(manager->query_endpoint_locality(gid(1)).found);
        delivered.set_value();
  });
  ASSERT_TRUE(manager->register_publisher(gid(1), "test"));
  ASSERT_EQ(delivered.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
  token.reset();
  bad.reset();
  bool replayed = false;
  auto replay = manager->subscribe_endpoint_updates([&](const auto &, const auto &) {
        replayed = true;
  });
  EXPECT_TRUE(replayed);
}

TEST(HostEndpointManager, BroadcastAndRegistrationBeforeStartup)
{
  Registry registry;
  Worker first(registry.id, "listen");
  Worker second(registry.id, "listen");
  ASSERT_TRUE(first.expect('R'));
  ASSERT_TRUE(second.expect('R'));
  auto manager = HostEndpointManager::get_instance(registry.id);
  ASSERT_TRUE(manager->register_subscription(gid(42), "test"));
  ASSERT_TRUE(first.expect('N'));
  ASSERT_TRUE(second.expect('N'));
  Worker late(registry.id, "listen");
  ASSERT_TRUE(late.expect('N'));  // Initial snapshot is replayed before ready.
  ASSERT_TRUE(late.expect('R'));
}

TEST(HostEndpointManager, CoalescedRegistrationsAndChangedRecords)
{
  Registry registry;
  auto manager = HostEndpointManager::get_instance(registry.id);
  std::mutex mutex;
  std::condition_variable condition;
  size_t seen = 0;
  bool changed = false;
  auto token = manager->subscribe_endpoint_updates([&](const auto &, const auto & info) {
        std::lock_guard<std::mutex> lock(mutex);
        if (info.ipc_capable) {
          changed = true;
        } else {
          ++seen;
        }
        condition.notify_all();
  });
  for (uint8_t i = 1; i <= 64; ++i) {
    ASSERT_TRUE(manager->register_subscription(gid(i), "batch"));
  }
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&]() {return seen == 64;}));
  }
  manager->set_ipc_capable(true);
  ASSERT_TRUE(manager->register_subscription(gid(1), "changed"));
  {
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(condition.wait_for(lock, std::chrono::seconds(5), [&]() {return changed;}));
    EXPECT_EQ(seen, 64u);
  }
}

TEST(HostEndpointManager, SubscriptionSetupCannotLoseConcurrentRegistration)
{
  Registry registry;
  auto manager = HostEndpointManager::get_instance(registry.id);
  ASSERT_TRUE(manager->register_subscription(gid(1), "first"));
  manager->refresh_from_remote();
  std::promise<void> snapshot_entered, release_snapshot, live_update;
  auto release = release_snapshot.get_future().share();
  auto subscribing = std::async(std::launch::async, [&]() {
        return manager->subscribe_endpoint_updates([&](const auto & key, const auto &) {
                 if (key.data[0] == 1) {
                   snapshot_entered.set_value();
                   release.wait();
                 } else if (key.data[0] == 2) {
                   live_update.set_value();
                 }
    });
  });
  const auto entered = snapshot_entered.get_future().wait_for(std::chrono::seconds(5));
  EXPECT_EQ(entered, std::future_status::ready);
  EXPECT_TRUE(manager->register_subscription(gid(2), "second"));
  const auto updated = live_update.get_future().wait_for(std::chrono::seconds(5));
  release_snapshot.set_value();
  EXPECT_EQ(updated, std::future_status::ready);
  auto token = subscribing.get();
}

TEST(HostEndpointManager, UnsubscribeDuringDeliveryAndShutdown)
{
  Registry registry;
  auto manager = HostEndpointManager::get_instance(registry.id);
  std::promise<void> entered, release_callback, finished;
  auto release = release_callback.get_future().share();
  auto target = std::make_shared<int>(7);
  std::weak_ptr<int> weak = target;
  auto token = manager->subscribe_endpoint_updates([&, weak](const auto &, const auto &) {
        auto keep_alive = weak.lock();
        entered.set_value();
        release.wait();
        EXPECT_EQ(*keep_alive, 7);
        finished.set_value();
  });
  ASSERT_TRUE(manager->register_publisher(gid(1), "test"));
  const auto ready = entered.get_future().wait_for(std::chrono::seconds(5));
  EXPECT_EQ(ready, std::future_status::ready);
  token.reset();
  target.reset();
  auto shutdown = std::async(std::launch::async, [&]() {manager.reset();});
  release_callback.set_value();
  EXPECT_EQ(shutdown.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  shutdown.get();
  EXPECT_TRUE(weak.expired());
  // Repeated idle teardown exercises shutdown on both sides of FUTEX_WAIT.
  for (int i = 0; i < 20; ++i) {
    auto idle = HostEndpointManager::get_instance(registry.id);
  }
}

TEST(HostEndpointManager, FailedProcessReleasesRegistryLock)
{
  Registry registry;
  auto manager = HostEndpointManager::get_instance(registry.id);
  {
    Worker owner(registry.id, "lock");
    ASSERT_TRUE(owner.expect('L'));
  }  // SIGKILL while owning the registry lock.
  EXPECT_TRUE(manager->register_publisher(gid(1), "after crash"));
}

TEST(HostEndpointManager, RejectsIncompatibleSizeAndVersion)
{
  Registry registry;
  const auto name = registry_name(registry.id);
  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, 1), 0);
  EXPECT_THROW(HostEndpointManager::get_instance(registry.id), std::runtime_error);
  struct stat status;
  ASSERT_EQ(fstat(fd, &status), 0);
  EXPECT_EQ(status.st_size, 1);
  close(fd);
  shm_unlink(name.c_str());
  HostEndpointManager::get_instance(registry.id).reset();
  fd = shm_open(name.c_str(), O_RDWR, 0600);
  ASSERT_GE(fd, 0);
  uint32_t version = 99;
  ASSERT_EQ(pwrite(fd, &version, sizeof(version), 0), static_cast<ssize_t>(sizeof(version)));
  EXPECT_THROW(HostEndpointManager::get_instance(registry.id), std::runtime_error);
  version = 0;  // Interrupted initialization is recoverable under the file lock.
  ASSERT_EQ(pwrite(fd, &version, sizeof(version), 0), static_cast<ssize_t>(sizeof(version)));
  EXPECT_NO_THROW(HostEndpointManager::get_instance(registry.id).reset());
  close(fd);
}

int main(int argc, char ** argv)
{
  if (argc == 6 && std::string(argv[1]) == "--hem-worker") {
    return worker_main(argv);
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
