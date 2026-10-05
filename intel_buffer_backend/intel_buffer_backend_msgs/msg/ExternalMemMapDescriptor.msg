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

# Level Zero external-memory-mapping descriptor (producer-owned POSIX SHM pages).
#
# Generation counter of the ring slot backing this frame, captured at publish
# time (ExternalMemMapRing::advance() bumps the counter in the slot itself —
# the last 4 bytes of the region, see external_memmap_generation_ptr() in
# l0_external_memmap.hpp — before handing the slot back out for writing). A
# subscriber that re-reads the slot's counter after finishing its read and
# finds it no longer matches this value knows the producer reclaimed the slot
# mid-read: the frame it just read may be torn. See
# ExternalMemMapImporter::generation_unchanged(). Always 0 for a region not
# backed by ExternalMemMapRing.
uint32 slot_generation

# Payload size in bytes — the number of valid bytes the producer wrote. Read on
# import and validated against the SHM region size; the L0 mapping itself always
# covers the whole (page-aligned) region, which may be larger.
uint64 size

# POSIX SHM name of the producer-owned region, with the leading '/' (e.g.
# "/imb_extmap_4711_2"). THE key field of this transport: it is the only handle a
# subscriber needs — shm_open() it, then map it with the external-memmap extension
# to obtain a local USM pointer over the producer's pages. Names are stable across
# frames (the producer rotates a fixed ring), which lets subscribers map each
# region exactly once and cache it; the extension prohibits mapping the same
# physical pages twice into one context.
string shm_name

# The producer's USM/virtual address of the buffer. Used ONLY for the
# intra-process fast path: if publisher_pid == getpid() and local_ptr != 0 the
# subscriber dereferences it directly and skips shm_open/map entirely. Meaningless
# in another process (different address space) — ignored there.
uint64 local_ptr

# PID of the publishing process. Read on import to decide intra- vs inter-process:
# equals getpid() -> use local_ptr; otherwise -> open and map shm_name.
int32 publisher_pid
