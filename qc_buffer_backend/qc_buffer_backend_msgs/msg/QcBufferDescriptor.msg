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

uint64 size                 # Number of elements (Phase 1: equals byte count since uint8_t)
string element_type_name    # Type name for validation (e.g. typeid(uint8_t).name())

# Intra-process zero-copy core fields
int32  pid                  # Publisher process ID; subscriber compares getpid() for same-process
uint64 vaddr                # Publisher CPU virtual address (valid within the same process)
uint64 uid                  # Unique per allocation, used for registry lookup and staleness detection

# dma-buf metadata (same-process HTP registration; cross-process transfer in Phase 2)
int32  dmabuf_fd            # dma-buf file descriptor (directly valid within the same process)
uint64 dmabuf_size         # Byte size for mapping

# --- Reserved for Phase 2 (cross-process SCM_RIGHTS); not populated in Phase 1 ---
bool   use_ipc             # Whether cross-process IPC is used (always false in Phase 1)
string ipc_socket_path     # Phase 2: Unix socket path for passing the fd
