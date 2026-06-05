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

uint64 size                    # Allocation size in bytes
string element_type_name       # Type name for validation (e.g., "h" for uint8_t)
int32 device_id                # CUDA device ID of the source GPU

# IPC metadata for VMM-based zero-copy sharing
bool use_ipc                   # True if IPC handles are valid
int32 vmm_pid                  # Publisher process ID
uint32 vmm_block_id            # Block ID in the publisher's pool
uint64 vmm_block_size          # Block size (for import mapping)
string vmm_socket_path         # Unix socket path to receive block FD
uint64 ipc_uid                 # Unique ID per publish for staleness detection

# CUDA IPC event for inter-process GPU synchronization
uint8[64] ipc_event_handle     # cudaIpcEventHandle_t (64 bytes)
