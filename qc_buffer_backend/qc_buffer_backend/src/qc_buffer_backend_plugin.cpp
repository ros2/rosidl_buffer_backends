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

#include "qc_buffer_backend/qc_buffer_backend.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>

#include "pluginlib/class_list_macros.hpp"
#include "qc_buffer/fd_broker.hpp"
#include "qc_buffer/qc_buffer.hpp"
#include "qc_buffer/qc_buffer_impl.hpp"
#include "qc_buffer/rpcmem_loader.hpp"
#include "qc_buffer_backend_msgs/msg/qc_buffer_descriptor.hpp"
#include "rcutils/logging_macros.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"

namespace qc_buffer_backend
{

const rosidl_message_type_support_t *
QcBufferBackend::get_descriptor_type_support() const
{
  return rosidl_typesupport_cpp::get_message_type_support_handle<
    qc_buffer_backend_msgs::msg::QcBufferDescriptor>();
}

std::shared_ptr<void>
QcBufferBackend::create_empty_descriptor() const
{
  return std::make_shared<qc_buffer_backend_msgs::msg::QcBufferDescriptor>();
}

std::shared_ptr<void> QcBufferBackend::create_descriptor_with_endpoint(
  const void * impl,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  (void)endpoint_info;

  auto * qc_impl = dynamic_cast<QcBufferImpl<uint8_t> *>(
    const_cast<rosidl::BufferImplBase<uint8_t> *>(
      static_cast<const rosidl::BufferImplBase<uint8_t> *>(impl)));
  if (!qc_impl) {
    return nullptr;
  }

  if (!RpcMemLoader::instance().available()) {
    return nullptr;
  }

  const auto & qc = qc_impl->get_qc_buffer();
  if (!qc || qc->dmabuf_fd() < 0) {
    return nullptr;
  }

  auto & broker = FdBroker::instance();
  if (!broker.running()) {
    return nullptr;
  }

  // Broker holds a dup(fd) to keep dma-buf alive until all subscribers have taken it.
  broker.register_buffer(qc->uid(), qc->dmabuf_fd(), qc->size());

  auto descriptor = std::make_shared<qc_buffer_backend_msgs::msg::QcBufferDescriptor>();
  descriptor->size = qc_impl->size();
  descriptor->element_type_name = typeid(uint8_t).name();
  descriptor->pid = static_cast<int32_t>(getpid());
  descriptor->vaddr = reinterpret_cast<uint64_t>(qc->data());
  descriptor->uid = qc->uid();
  descriptor->dmabuf_fd = qc->dmabuf_fd();
  descriptor->dmabuf_size = qc->size();
  descriptor->use_ipc = true;
  descriptor->ipc_socket_path = broker.socket_path();

  return descriptor;
}

std::unique_ptr<void, void (*)(void *)> QcBufferBackend::from_descriptor_with_endpoint(
  const void * descriptor_ptr,
  const rmw_topic_endpoint_info_t & endpoint_info) const
{
  (void)endpoint_info;

  auto make_empty = []() {
      auto empty = std::make_unique<QcBufferImpl<uint8_t>>();
      return std::unique_ptr<void, void (*)(void *)>(
        empty.release(), [](void * p) {
          delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
        });
    };

  const auto * descriptor =
    static_cast<const qc_buffer_backend_msgs::msg::QcBufferDescriptor *>(descriptor_ptr);

  if (descriptor->element_type_name != typeid(uint8_t).name()) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "QcBufferDescriptor element type mismatch: expected %s, got %s; dropping",
      typeid(uint8_t).name(), descriptor->element_type_name.c_str());
    return make_empty();
  }

  if (!descriptor->use_ipc || descriptor->ipc_socket_path.empty()) {
    RCUTILS_LOG_WARN_ONCE_NAMED(
      "qc_buffer_backend",
      "qc descriptor missing IPC socket path; falling back to CPU");
    return make_empty();
  }

  // Connect to the publisher's FdBroker socket.
  int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "FdBroker client: socket() failed: %s", strerror(errno));
    return make_empty();
  }

  struct timeval tv{0, 500'000};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(
    addr.sun_path, descriptor->ipc_socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (::connect(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "FdBroker client: connect(%s) failed: %s",
      descriptor->ipc_socket_path.c_str(), strerror(errno));
    ::close(sock);
    return make_empty();
  }

  uint64_t uid = descriptor->uid;
  if (::send(sock, &uid, sizeof(uid), 0) != static_cast<ssize_t>(sizeof(uid))) {
    ::close(sock);
    return make_empty();
  }

  // Receive status byte + fd via SCM_RIGHTS.
  uint8_t status = 0;
  struct iovec iov{&status, 1};
  union {
    char buf[CMSG_SPACE(sizeof(int))];
    struct cmsghdr align;
  } cmsg_buf{};
  struct msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf.buf;
  msg.msg_controllen = sizeof(cmsg_buf.buf);

  ssize_t r = ::recvmsg(sock, &msg, 0);
  ::close(sock);

  if (r <= 0 || status != 0x01) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "FdBroker client: uid %" PRIu64 " not available (status=%u); falling back to CPU",
      uid, static_cast<unsigned>(status));
    return make_empty();
  }

  // If the cmsg buffer was too small, the kernel drops the fd silently.
  if (msg.msg_flags & MSG_CTRUNC) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "FdBroker client: SCM_RIGHTS control message truncated (MSG_CTRUNC) "
      "for uid %" PRIu64 "; fd dropped by kernel, falling back to CPU", uid);
    return make_empty();
  }

  int received_fd = -1;
  struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
    std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
  }
  if (received_fd < 0) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "FdBroker client: SCM_RIGHTS fd missing for uid %" PRIu64,
      uid);
    return make_empty();
  }

  // mmap the received fd into this process's address space.
  try {
    auto result = std::make_unique<QcBufferImpl<uint8_t>>(
      received_fd, descriptor->dmabuf_size, descriptor->size);
    return std::unique_ptr<void, void (*)(void *)>(
      result.release(), [](void * p) {
        delete static_cast<rosidl::BufferImplBase<uint8_t> *>(p);
      });
  } catch (const std::exception & e) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "FdBroker client: QcBufferImpl construction failed: %s", e.what());
    return make_empty();
  }
}

}  // namespace qc_buffer_backend

PLUGINLIB_EXPORT_CLASS(
  qc_buffer_backend::QcBufferBackend,
  rosidl::BufferBackend)
