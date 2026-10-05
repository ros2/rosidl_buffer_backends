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

# Level Zero Host USM IPC descriptor.
#

# Allocation size in bytes. Read on import: passed to import_level_zero_block()
# and used as the length for the imported mapping. Must match the publisher's
# allocation exactly.
uint64 size

# Number of valid bytes in ze_ipc_handle below (the L0 IPC handle is an opaque,
# fixed-capacity blob but only the first ze_ipc_handle_size bytes are meaningful).
# Written by L0UsmIpc::export_handle() on the publisher; read on import so the
# handle is reconstructed with the correct length before zeMemOpenIpcHandle().
uint64 ze_ipc_handle_size

# The Level Zero IPC memory handle (ze_ipc_mem_handle_t) produced by
# zeMemGetIpcHandle() on the publisher. This is the actual token the subscriber
# feeds to zeMemOpenIpcHandle() to map the shared allocation. 64 bytes matches
# the L0 handle storage size. Read on import (with ze_ipc_handle_size).
uint8[64] ze_ipc_handle

# Index of the Level Zero device the allocation lives on, within the PUBLISHER's
# driver enumeration. Only meaningful in the publishing process: a subscriber
# with a different GPU count or driver/device ordering would index a different
# device. Kept as a fallback for import when ze_device_uuid is unset (all-zero)
# or matches no local device. -1 means "unset". On a single-iGPU system this is 0.
int32 ze_device_ordinal

# Stable UUID of the device the allocation lives on (ze_device_properties_t.uuid,
# 16 bytes). Read on import: the subscriber matches this against the UUIDs of its
# own enumerated devices and opens the handle on the matching one, so the correct
# GPU is chosen regardless of enumeration order across processes. All-zero means
# "unset" and the subscriber falls back to ze_device_ordinal.
uint8[16] ze_device_uuid

# Opaque id of the publisher's L0 context (the context pointer, stringified to a
# u64). Informational/diagnostic only — the subscriber creates its OWN context
# to import into, so this is NOT consumed on the import path. Populated to aid
# debugging of the cross-context IPC limitation documented in the README.
uint64 ze_context_id

# Per-publish unique id assigned by the pool (assign_uid). Intended for staleness
# detection so a subscriber can tell whether a recycled pool block still holds
# the frame it expects. Passed to import as expected_uid but CURRENTLY IGNORED
# ((void)-cast) — reserved for a future validated-import check.
uint64 ipc_uid

# Pool block id of the exported buffer. Read on import together with
# publisher_pid to locate the shared-memory refcount segment
# "/intel_buffer_<publisher_pid>_<block_id>", so the L0 consumer attaches
# to the same cross-process refcount the DMA-BUF path uses.
uint32 block_id

# Identifier of the publisher's memory pool instance (random per process). Lets a
# subscriber distinguish blocks from different pools. Written by the publisher;
# not yet consumed on import — reserved for multi-pool disambiguation.
uint32 pool_id

# Pool generation counter, bumped when the pool grows/shrinks. Intended to detect
# that the pool layout changed since a descriptor was published. Written by the
# publisher; not yet consumed on import — reserved for staleness detection.
uint32 pool_generation

# The publisher's virtual address of the buffer (block->ptr as a u64). Used ONLY
# for the intra-process fast path: if publisher_pid == getpid() and local_ptr != 0,
# the subscriber dereferences it directly and skips IPC entirely. Meaningless in
# another process (different address space) — ignored there.
uint64 local_ptr

# PID of the publishing process. Read on import to decide intra- vs inter-process:
# if it equals getpid() the local_ptr shortcut is taken, otherwise the L0 IPC
# handle is opened. Also disambiguates which process owns local_ptr.
int32 publisher_pid

# AF_UNIX socket the publisher serves the allocation's underlying dma_buf fd on,
# via SCM_RIGHTS (same mechanism as the DMA-BUF path). On the xe kernel driver a
# Level Zero IPC handle embeds a PROCESS-LOCAL dma_buf fd, so byte-copying
# ze_ipc_handle across processes fails in zeMemOpenIpcHandle (0x78000004). When
# this path is non-empty the subscriber instead receives the real fd over the
# socket and rebuilds a local IPC handle (zeMemGetIpcHandleFromFileDescriptorExp)
# before opening it. Empty => fall back to the byte-copied ze_ipc_handle (valid
# intra-node and on drivers whose handle is not fd-backed).
string l0_socket_path
