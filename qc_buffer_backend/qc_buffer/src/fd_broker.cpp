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

#include "qc_buffer/fd_broker.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <string>

#include "rcutils/logging_macros.h"

namespace qc_buffer_backend
{

FdBroker & FdBroker::instance()
{
  static FdBroker broker;
  return broker;
}

FdBroker::FdBroker()
{
  start();
}

FdBroker::~FdBroker()
{
  running_ = false;
  if (server_fd_ >= 0) {
    ::shutdown(server_fd_, SHUT_RDWR);
    ::close(server_fd_);
    server_fd_ = -1;
  }
  if (!socket_path_.empty()) {
    ::unlink(socket_path_.c_str());
  }
  if (worker_.joinable()) {
    worker_.join();
  }
  // Wait for detached handle_connection threads to finish before teardown.
  while (active_connections_.load(std::memory_order_acquire) > 0) {
    std::this_thread::yield();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto & [uid, entry] : table_) {
    ::close(entry.dup_fd);
  }
  table_.clear();
}

void FdBroker::start()
{
  server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    RCUTILS_LOG_ERROR_NAMED(
      "qc_buffer_backend", "FdBroker: socket() failed: %s", strerror(errno));
    return;
  }

  socket_path_ = "/tmp/qc_broker_" + std::to_string(getpid()) + ".sock";
  ::unlink(socket_path_.c_str());

  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  if (::bind(server_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
    RCUTILS_LOG_ERROR_NAMED(
      "qc_buffer_backend", "FdBroker: bind() failed: %s", strerror(errno));
    ::close(server_fd_);
    server_fd_ = -1;
    return;
  }

  if (::listen(server_fd_, 16) < 0) {
    RCUTILS_LOG_ERROR_NAMED(
      "qc_buffer_backend", "FdBroker: listen() failed: %s", strerror(errno));
    ::close(server_fd_);
    server_fd_ = -1;
    return;
  }

  running_ = true;
  worker_ = std::thread(&FdBroker::worker_loop, this);
}

void FdBroker::worker_loop()
{
  while (running_) {
    int conn_fd = ::accept(server_fd_, nullptr, nullptr);
    if (conn_fd < 0) {
      if (!running_) {
        break;
      }
      RCUTILS_LOG_WARN_NAMED(
        "qc_buffer_backend", "FdBroker: accept() failed: %s", strerror(errno));
      continue;
    }
    active_connections_.fetch_add(1, std::memory_order_relaxed);
    std::thread([this, conn_fd]() {handle_connection(conn_fd);}).detach();
  }
}

void FdBroker::handle_connection(int conn_fd)
{
  // 500 ms timeout on send/recv to avoid blocking indefinitely.
  struct timeval tv{0, 500'000};
  ::setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(conn_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  // Read request: [uint64_t uid]
  uint64_t uid = 0;
  ssize_t n = ::recv(conn_fd, &uid, sizeof(uid), MSG_WAITALL);
  if (n != static_cast<ssize_t>(sizeof(uid))) {
    ::close(conn_fd);
    active_connections_.fetch_sub(1, std::memory_order_release);
    return;
  }

  // Dup the stored fd while holding the lock; release lock before sendmsg.
  int tmp_fd = -1;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = table_.find(uid);
    if (it == table_.end()) {
      uint8_t miss = kStatusMiss;
      ::send(conn_fd, &miss, 1, MSG_NOSIGNAL);
      ::close(conn_fd);
      active_connections_.fetch_sub(1, std::memory_order_release);
      return;
    }
    tmp_fd = ::dup(it->second.dup_fd);
  }

  if (tmp_fd < 0) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "FdBroker: dup() failed for uid %" PRIu64 ": %s",
      uid, strerror(errno));
    uint8_t miss = kStatusMiss;
    ::send(conn_fd, &miss, 1, MSG_NOSIGNAL);
    ::close(conn_fd);
    active_connections_.fetch_sub(1, std::memory_order_release);
    return;
  }

  // Send fd via SCM_RIGHTS.
  uint8_t status = kStatusOk;
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

  struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &tmp_fd, sizeof(int));

  ::sendmsg(conn_fd, &msg, MSG_NOSIGNAL);
  ::close(tmp_fd);
  ::close(conn_fd);
  active_connections_.fetch_sub(1, std::memory_order_release);
}

void FdBroker::register_buffer(uint64_t uid, int fd, uint64_t dmabuf_size)
{
  if (!running_ || uid == 0 || fd < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (table_.count(uid)) {
    return;  // Idempotent: same uid from a second endpoint call.
  }
  int dup_fd = ::dup(fd);
  if (dup_fd < 0) {
    RCUTILS_LOG_WARN_NAMED(
      "qc_buffer_backend",
      "FdBroker: dup() failed during register uid %" PRIu64 ": %s",
      uid, strerror(errno));
    return;
  }
  table_[uid] = {dup_fd, dmabuf_size};
  order_.push_back(uid);
  while (order_.size() > kCapacity) {
    uint64_t old = order_.front();
    order_.pop_front();
    auto it = table_.find(old);
    if (it != table_.end()) {
      ::close(it->second.dup_fd);
      table_.erase(it);
    }
  }
}

}  // namespace qc_buffer_backend
