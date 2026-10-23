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
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "cuda_buffer/host_endpoint_manager.hpp"
#include "cuda_buffer_backend/cuda_buffer_backend.hpp"
#include "cuda_buffer_backend_msgs/msg/cuda_buffer_descriptor.hpp"

using cuda_buffer_backend::CudaBufferBackend;
using cuda_buffer_backend::CudaBufferImpl;
using host_endpoint_manager::HostEndpointManager;

namespace
{

// When set, the test binary runs as a peer process instead of the tests:
// the value is "<ready_fd>,<done_fd>".
constexpr const char * kPeerEnv = "CUDA_BUFFER_BACKEND_TEST_PEER";
constexpr uint8_t kPeerGidByte = 0x6b;

// Register an IPC-capable subscription as a separate process on this host,
// tell the parent, and stay registered until the parent closes done_fd.
int run_peer(const char * fds)
{
  int ready_fd = -1;
  int done_fd = -1;
  if (std::sscanf(fds, "%d,%d", &ready_fd, &done_fd) != 2) {
    return 1;
  }

  auto manager = HostEndpointManager::get_instance(0);
  manager->set_ipc_capable(true);

  rmw_gid_t gid;
  gid.implementation_identifier = "";
  std::memset(gid.data, kPeerGidByte, RMW_GID_STORAGE_SIZE);
  if (!manager->register_subscription(gid, "test_type")) {
    return 1;
  }

  char c = 1;
  if (write(ready_fd, &c, 1) != 1) {
    return 1;
  }
  ssize_t n = read(done_fd, &c, 1);
  (void)n;
  return 0;
}

}  // namespace

class CudaBufferBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    backend_ = std::make_unique<CudaBufferBackend>();
  }
  void TearDown() override {}

  std::unique_ptr<CudaBufferBackend> backend_;
};

TEST_F(CudaBufferBackendTest, BackendTypeName)
{
  EXPECT_EQ(backend_->get_backend_type(), "cuda");
}

TEST_F(CudaBufferBackendTest, OnDiscoveringEndpointWithoutCudaSupport)
{
  rmw_topic_endpoint_info_t endpoint_info;
  std::memset(&endpoint_info, 0, sizeof(endpoint_info));
  endpoint_info.topic_type = "test_type";

  std::vector<rmw_topic_endpoint_info_t> existing_endpoints;
  std::unordered_map<std::string, std::string> supported_backends;
  supported_backends["demo"] = "version=1.0";

  std::pair<bool, std::vector<std::set<uint32_t>>> result =
    backend_->on_discovering_endpoint(
      endpoint_info, existing_endpoints, supported_backends);
  EXPECT_FALSE(result.first);
}

// These tests need no CUDA device: when none is usable, the endpoint hooks
// must report "not IPC capable" instead of throwing out of the RMW.
TEST_F(CudaBufferBackendTest, OnCreatingEndpointDoesNotThrow)
{
  rmw_topic_endpoint_info_t endpoint_info;
  std::memset(&endpoint_info, 0, sizeof(endpoint_info));
  endpoint_info.topic_type = "test_type";
  endpoint_info.endpoint_type = RMW_ENDPOINT_PUBLISHER;

  EXPECT_NO_THROW(backend_->on_creating_endpoint(endpoint_info));
}

TEST_F(CudaBufferBackendTest, OnDiscoveringEndpointDoesNotThrow)
{
  rmw_topic_endpoint_info_t endpoint_info;
  std::memset(&endpoint_info, 0, sizeof(endpoint_info));
  endpoint_info.topic_type = "test_type";

  std::vector<rmw_topic_endpoint_info_t> existing_endpoints;
  std::unordered_map<std::string, std::string> supported_backends;
  supported_backends["cuda"] = "version=1.0";

  EXPECT_NO_THROW(
    backend_->on_discovering_endpoint(
      endpoint_info, existing_endpoints, supported_backends));
}

// Needs a CUDA device. A subscriber missing from the local endpoint registry
// is on another host or has not registered yet (a peer with no usable device
// is found, with ipc_capable=false). Either way it must not be sent a CUDA
// descriptor yet: the publisher falls back to the host path, which
// create_descriptor_with_endpoint signals with nullptr. The miss is not
// cached, so a same-host peer gets a CUDA descriptor once it registers.
TEST_F(CudaBufferBackendTest, NoDescriptorForUnknownSubscriber)
{
  rmw_topic_endpoint_info_t endpoint_info;
  std::memset(&endpoint_info, 0, sizeof(endpoint_info));
  endpoint_info.topic_type = "test_type";
  endpoint_info.endpoint_type = RMW_ENDPOINT_SUBSCRIPTION;
  std::memset(endpoint_info.endpoint_gid, 0x5a, sizeof(endpoint_info.endpoint_gid));

  std::vector<rmw_topic_endpoint_info_t> existing_endpoints;
  std::unordered_map<std::string, std::string> supported_backends;
  supported_backends["cuda"] = "";
  auto result = backend_->on_discovering_endpoint(
    endpoint_info, existing_endpoints, supported_backends);
  EXPECT_FALSE(result.first);

  CudaBufferImpl<uint8_t> impl(1024);
  EXPECT_EQ(nullptr, backend_->create_descriptor_with_endpoint(&impl, endpoint_info));
}

// Needs a CUDA device. A subscriber on this host under another uid cannot
// import a CUDA descriptor. It must get the host path even when
// on_discovering_endpoint has not classified it yet.
TEST_F(CudaBufferBackendTest, NoDescriptorForSameHostPeerUnderAnotherUid)
{
  if (!CudaBufferImpl<uint8_t>::is_pool_ipc_capable()) {
    GTEST_SKIP() << "needs an IPC-capable CUDA device";
  }

  auto manager = HostEndpointManager::get_instance(0);

  int ready[2];
  int done[2];
  ASSERT_EQ(0, pipe(ready));
  ASSERT_EQ(0, pipe(done));

  // Build the peer's environment before fork: only exec runs in the child.
  std::vector<std::string> extra_env = {
    std::string(kPeerEnv) + "=" + std::to_string(ready[1]) + "," + std::to_string(done[0]),
    "CUDA_BUFFER_UID_OVERRIDE=" + std::to_string(manager->get_uid() + 1),
  };
  std::vector<char *> envp;
  for (auto & e : extra_env) {
    envp.push_back(e.data());
  }
  for (char ** e = environ; *e != nullptr; ++e) {
    envp.push_back(*e);
  }
  envp.push_back(nullptr);
  char exe[] = "/proc/self/exe";
  char * argv[] = {exe, nullptr};

  pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    close(ready[0]);
    close(done[1]);
    execve(exe, argv, envp.data());
    _exit(127);
  }
  close(ready[1]);
  close(done[0]);

  char c = 0;
  const bool registered = read(ready[0], &c, 1) == 1;

  host_endpoint_manager::LocalityInfo peer{};
  std::shared_ptr<void> descriptor;
  if (registered) {
    rmw_gid_t gid;
    gid.implementation_identifier = "";
    std::memset(gid.data, kPeerGidByte, RMW_GID_STORAGE_SIZE);
    manager->refresh_from_remote();
    peer = manager->query_endpoint_locality(gid);

    rmw_topic_endpoint_info_t endpoint_info;
    std::memset(&endpoint_info, 0, sizeof(endpoint_info));
    endpoint_info.topic_type = "test_type";
    endpoint_info.endpoint_type = RMW_ENDPOINT_SUBSCRIPTION;
    std::memset(endpoint_info.endpoint_gid, kPeerGidByte, sizeof(endpoint_info.endpoint_gid));

    CudaBufferImpl<uint8_t> impl(1024);
    descriptor = backend_->create_descriptor_with_endpoint(&impl, endpoint_info);
  }

  close(ready[0]);
  close(done[1]);
  int status = 0;
  waitpid(pid, &status, 0);

  ASSERT_TRUE(registered);
  ASSERT_TRUE(peer.found);
  EXPECT_TRUE(peer.ipc_capable);
  EXPECT_EQ(host_endpoint_manager::EndpointLocality::INTER_PROCESS_SAME_HOST, peer.locality);
  EXPECT_NE(manager->get_uid(), peer.uid);
  EXPECT_EQ(nullptr, descriptor);
}

int main(int argc, char ** argv)
{
  if (const char * fds = std::getenv(kPeerEnv)) {
    return run_peer(fds);
  }
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
