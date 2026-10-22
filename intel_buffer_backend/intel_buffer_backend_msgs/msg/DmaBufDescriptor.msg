# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DMA-BUF IPC descriptor (dma-heap fd shared via AF_UNIX + SCM_RIGHTS).
#
# Carried in IntelMemoryBufferImageFrame when transport_mode == TRANSPORT_DMA_BUF. A dma_buf fd is
# process-local and cannot travel on the DDS wire, so this descriptor carries a
# rendezvous socket path instead: the subscriber connects to it and receives the
# real fd out-of-band via SCM_RIGHTS, then mmaps the same physical pages (zero
# copy). Works for both dma-heap allocations and camera V4L2 DMA-BUFs.

# Allocation size in bytes. Read on import as the mmap length for the received
# fd. Must match the publisher's buffer size.
uint64 size

# PID of the process that owns the dma_buf fd (the exporter). Read on import to
# rebuild the shared-memory metadata name "/intel_buffer_<dmabuf_pid>_<block_id>",
# which backs the cross-process refcount. Usually equals publisher_pid.
int32 dmabuf_pid

# Pool block id of the exported buffer. Read on import as the second half of the
# metadata segment name above. Also the key under which the export socket is
# registered, so distinct blocks get distinct sockets/mappings.
uint32 dmabuf_pool_block_id

# Size of the underlying pool block in bytes. Carried for completeness/validation
# alongside `size`; the import path uses `size` for the mapping length. Normally
# equal to `size` after alignment.
uint64 dmabuf_block_size

# Path to the AF_UNIX socket the publisher listens on to hand out the dma_buf fd
# via SCM_RIGHTS (e.g. /tmp/intel_buffer_<pid>_blk<id>.sock). THE key field of this
# transport: the subscriber connect()s here to receive the fd, then mmaps it.
# Without it there is no way to obtain the fd in another process.
string dmabuf_socket_path

# Per-publish unique id assigned by the pool (assign_uid). Intended for staleness
# detection on recycled blocks. Passed to import as expected_uid but CURRENTLY
# IGNORED ((void)-cast) — reserved for a future validated-import check.
uint64 ipc_uid

# Identifier of the publisher's memory pool instance. Written by the publisher;
# not yet consumed on import — reserved for multi-pool disambiguation.
uint32 pool_id

# Pool generation counter, bumped on pool grow/shrink. Written by the publisher;
# not yet consumed on import — reserved for staleness detection.
uint32 pool_generation

# The publisher's virtual address of the buffer. Used ONLY for the intra-process
# fast path: if publisher_pid == getpid() and local_ptr != 0, the subscriber
# dereferences it directly and skips the socket/SCM_RIGHTS import. For a real
# camera the publisher sets this to 0 to force the cross-process path.
uint64 local_ptr

# PID of the publishing process. Read on import to decide intra- vs inter-process:
# equals getpid() -> use local_ptr; otherwise -> connect to dmabuf_socket_path
# and import the fd.
int32 publisher_pid
