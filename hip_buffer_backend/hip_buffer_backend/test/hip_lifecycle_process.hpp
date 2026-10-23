// Copyright 2026 Open Source Robotics Foundation, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HIP_LIFECYCLE_PROCESS_HPP_
#define HIP_LIFECYCLE_PROCESS_HPP_

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "hip_buffer/hip_buffer_impl.hpp"
#include "hip_buffer_backend/hip_buffer_backend.hpp"
#include "hip_buffer_backend_msgs/msg/hip_buffer_descriptor.hpp"

extern char ** environ;

namespace hip_lifecycle_test
{
using Descriptor = hip_buffer_backend_msgs::msg::HipBufferDescriptor;
using Impl = hip_buffer_backend::HipBufferImpl<uint8_t>;
constexpr size_t kPayloadSize = 4096;
// Each IPC phase has its own deadline, including HIP initialization/import.
// This is a watchdog, NOT the pool's publish grace interval.
constexpr auto kPhaseTimeout = std::chrono::seconds(15);

inline void require(bool condition, const std::string & message)
{
  if (!condition) {throw std::runtime_error(message);}
}

inline void check_hip(hipError_t status, const char * operation)
{
  require(status == hipSuccess, std::string(operation) + ": " + hipGetErrorName(status));
}

// A fixed-size packet avoids ABI-dependent serialization of ROS strings/vectors.
// These buffers deliberately have no write event: every HIP copy is explicitly
// completed before the peer is allowed to import or the pool is allowed to reuse.
struct WireDescriptor
{
  uint64_t size{};
  uint64_t block_size{};
  uint64_t uid{};
  int32_t device{};
  int32_t pid{};
  uint32_t block{};
  char socket[108]{};
};

inline WireDescriptor to_wire(const Descriptor & descriptor)
{
  WireDescriptor wire{};
  wire.size = descriptor.size;
  wire.block_size = descriptor.vmm_block_size;
  wire.uid = descriptor.ipc_uid;
  wire.device = descriptor.device_id;
  wire.pid = descriptor.vmm_pid;
  wire.block = descriptor.vmm_block_id;
  require(descriptor.use_ipc, "descriptor did not select IPC");
  require(descriptor.vmm_socket_path.size() < sizeof(wire.socket), "socket path too long");
  require(std::all_of(
      descriptor.ipc_event_handle.begin(), descriptor.ipc_event_handle.end(),
      [](uint8_t byte) {return byte == 0;}), "unexpected event on synchronized test buffer");
  std::memcpy(wire.socket, descriptor.vmm_socket_path.c_str(),
    descriptor.vmm_socket_path.size() + 1);
  return wire;
}

inline Descriptor from_wire(const WireDescriptor & wire)
{
  Descriptor descriptor;
  descriptor.size = wire.size;
  descriptor.element_type_name = typeid(uint8_t).name();
  descriptor.use_ipc = true;
  descriptor.device_id = wire.device;
  descriptor.vmm_pid = wire.pid;
  descriptor.vmm_block_id = wire.block;
  descriptor.vmm_block_size = wire.block_size;
  descriptor.ipc_uid = wire.uid;
  descriptor.vmm_socket_path = wire.socket;
  return descriptor;
}

enum class Command {Import, ReadHeld, Release, Quit};
struct Request
{
  Command command{};
  WireDescriptor descriptor{};
  uint8_t expected_byte{};
};
struct Reply
{
  uint64_t size{};
  bool payload_matches{};
  uint8_t first_byte{};
  char error[256]{};
};

inline void transfer(int socket, void * packet, size_t size, bool sending)
{
  const auto deadline = std::chrono::steady_clock::now() + kPhaseTimeout;
  for (;;) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now()).count();
    require(remaining > 0, "importer protocol phase timed out");
    pollfd descriptor{socket, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
    const int ready = poll(&descriptor, 1, static_cast<int>(remaining));
    if (ready < 0 && errno == EINTR) {continue;}
    require(ready > 0, "importer socket poll failed or timed out");
    const ssize_t count = sending ?
      send(socket, packet, size, MSG_NOSIGNAL | MSG_DONTWAIT) :
      recv(socket, packet, size, MSG_DONTWAIT | MSG_TRUNC);
    if (count < 0 && (errno == EINTR || errno == EAGAIN)) {continue;}
    require(count == static_cast<ssize_t>(size), "importer socket closed or invalid packet");
    return;
  }
}

inline int importer_main(int socket)
{
  try {
    hip_buffer_backend::HipBufferBackend backend;
    rmw_topic_endpoint_info_t endpoint{};
    std::unique_ptr<void, void (*)(void *)> imported(nullptr, [](void *) {});
    for (;;) {
      Request request;
      transfer(socket, &request, sizeof(request), false);
      Reply reply;
      try {
        if (request.command == Command::Import) {
          require(!imported, "release the previous import before importing again");
          auto descriptor = from_wire(request.descriptor);
          check_hip(hipSetDevice(descriptor.device_id), "hipSetDevice(importer)");
          imported = backend.from_descriptor_with_endpoint(&descriptor, endpoint);
        }
        if (request.command == Command::Import || request.command == Command::ReadHeld) {
          auto * impl = static_cast<rosidl::BufferImplBase<uint8_t> *>(imported.get());
          require(impl != nullptr, "no imported buffer to read");
          reply.size = impl->size();
          if (reply.size != 0) {
            auto * hip_impl = dynamic_cast<Impl *>(impl);
            require(hip_impl && reply.size == kPayloadSize, "unexpected imported buffer");
            std::array<uint8_t, kPayloadSize> bytes{};
            check_hip(hipMemcpy(bytes.data(), hip_impl->get_hip_buffer().get_device_ptr(),
              bytes.size(), hipMemcpyDeviceToHost), "hipMemcpy(import readback)");
            check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(import readback)");
            reply.first_byte = bytes.front();
            reply.payload_matches = std::all_of(bytes.begin(), bytes.end(),
              [&request](uint8_t byte) {return byte == request.expected_byte;});
          }
        } else {
          imported.reset();
        }
      } catch (const std::exception & error) {
        std::snprintf(reply.error, sizeof(reply.error), "%s", error.what());
      }
      transfer(socket, &reply, sizeof(reply), true);
      if (request.command == Command::Quit) {return 0;}
    }
  } catch (const std::exception & error) {
    std::fprintf(stderr, "lifecycle importer: %s\n", error.what());
    return 2;
  }
}

class Importer
{
public:
  Importer()
  {
    int sockets[2];
    require(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) == 0,
      "socketpair failed");
    posix_spawn_file_actions_t actions;
    int result = posix_spawn_file_actions_init(&actions);
    if (result == 0) {
      result = posix_spawn_file_actions_addclose(&actions, sockets[0]);
      if (result == 0) {result = posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);}
      if (result == 0 && sockets[1] != 3) {
        result = posix_spawn_file_actions_addclose(&actions, sockets[1]);
      }
      char executable[] = "/proc/self/exe";
      char mode[] = "--lifecycle-importer";
      char * arguments[] = {executable, mode, nullptr};
      if (result == 0) {
        // exec starts an independent HIP runtime; never fork a HIP-initialized process.
        result = posix_spawn(&pid_, executable, &actions, nullptr, arguments, environ);
      }
      posix_spawn_file_actions_destroy(&actions);
    }
    close(sockets[1]);
    if (result != 0) {
      close(sockets[0]);
      throw std::runtime_error(std::string("posix_spawn: ") + std::strerror(result));
    }
    socket_ = sockets[0];
  }

  Importer(const Importer &) = delete;
  Importer & operator=(const Importer &) = delete;

  ~Importer()
  {
    if (pid_ > 0) {
      try {finish();} catch (...) {
        if (pid_ > 0) {
          kill(pid_, SIGKILL);
          while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
        }
      }
    }
    if (socket_ >= 0) {close(socket_);}
  }

  Reply request(Command command, WireDescriptor descriptor = {}, uint8_t byte = 0)
  {
    Request request{command, descriptor, byte};
    transfer(socket_, &request, sizeof(request), true);
    Reply reply;
    transfer(socket_, &reply, sizeof(reply), false);
    require(reply.error[0] == '\0', std::string("importer API failure: ") + reply.error);
    return reply;
  }

  void finish()
  {
    request(Command::Quit);
    const auto deadline = std::chrono::steady_clock::now() + kPhaseTimeout;
    for (;;) {
      int status = 0;
      const pid_t result = waitpid(pid_, &status, WNOHANG);
      if (result == pid_) {
        pid_ = -1;
        require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "importer exited abnormally");
        return;
      }
      if (result < 0 && errno == EINTR) {continue;}
      require(result == 0, "waitpid failed");
      require(std::chrono::steady_clock::now() < deadline, "importer exit timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

private:
  int socket_{-1};
  pid_t pid_{-1};
};
}  // namespace hip_lifecycle_test

#endif  // HIP_LIFECYCLE_PROCESS_HPP_
