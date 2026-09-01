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

#ifndef QC_BUFFER__FD_BROKER_HPP_
#define QC_BUFFER__FD_BROKER_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "qc_buffer/visibility_control.h"

namespace qc_buffer_backend
{

/// \brief Per-publisher-process fd broker for cross-process dma-buf sharing.
///
/// FdBroker is a process-level singleton. The publisher calls register_buffer()
/// for each newly published buffer; the broker holds a dup(fd) so the dma-buf
/// kernel object stays alive even after QcBuffer destructs. Any subscriber
/// (same process or different process on the same device) connects to the Unix
/// socket, sends the uid, and receives a new fd via SCM_RIGHTS.
///
/// The rolling window (kCapacity entries) bounds retained memory. Once a uid
/// is evicted, the broker closes its dup_fd; subscribers that have already
/// taken a copy are unaffected (they hold their own kernel reference).
class QC_BUFFER_PUBLIC FdBroker
{
public:
  static FdBroker & instance();

  /// Register a newly published buffer. Dups fd and stores it.
  /// Safe to call multiple times with the same uid (idempotent).
  void register_buffer(uint64_t uid, int fd, uint64_t dmabuf_size);

  /// Socket path subscribers should connect to.
  const std::string & socket_path() const {return socket_path_;}

  /// True if the broker is running (socket bound and worker thread live).
  bool running() const {return running_;}

  /// Number of handle_connection threads currently executing.
  int active_connections() const
  {return active_connections_.load(std::memory_order_acquire);}

  ~FdBroker();

private:
  FdBroker();

  void start();
  void worker_loop();
  void handle_connection(int conn_fd);

  struct Entry
  {
    int dup_fd;
    uint64_t dmabuf_size;
  };

  /// Shared FIFO window across all publishers in this process.
  static constexpr size_t kCapacity = 64;
  static constexpr uint8_t kStatusOk = 0x01;
  static constexpr uint8_t kStatusMiss = 0x02;

  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, Entry> table_;
  std::deque<uint64_t> order_;

  int server_fd_{-1};
  std::string socket_path_;
  std::thread worker_;
  std::atomic<bool> running_{false};  // atomic: written by destructor, read by worker
  std::atomic<int> active_connections_{0};  // destructor spins until 0 before teardown
};

}  // namespace qc_buffer_backend

#endif  // QC_BUFFER__FD_BROKER_HPP_
