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

# Zero-copy buffer descriptor.
#
# Serialized by the intel_buffer backend in place of the raw bytes of any
# uint8[] field backed by an IntelBuffer (e.g. sensor_msgs/Image.data). It
# carries ONLY the backend-specific information needed to import the shared
# USM / DMA-BUF memory — never the data itself. Metadata that describes how to
# interpret the bytes (width, height, step, encoding, header, ...) belongs to
# the enclosing standard message, which remains the source of truth.

# Transport selection — tells subscribers which descriptor below is valid and
# which import path to take. Set by the backend from the buffer's backing
# memory (see INTEL_MEMORY_BUFFER_TYPE / the launch transport_mode param).
uint8 TRANSPORT_LEVEL_ZERO_USM = 0
uint8 TRANSPORT_DMA_BUF = 1
uint8 TRANSPORT_EXTERNAL_MEMMAP = 2
uint8 transport_mode

# L0 USM descriptor. Valid ONLY when transport_mode == TRANSPORT_LEVEL_ZERO_USM;
# ignored otherwise. See L0UsmDescriptor.msg for per-field details.
intel_buffer_backend_msgs/L0UsmDescriptor l0_desc

# DMA-BUF descriptor. Valid ONLY when transport_mode == TRANSPORT_DMA_BUF;
# ignored otherwise. See DmaBufDescriptor.msg for per-field details.
intel_buffer_backend_msgs/DmaBufDescriptor dmabuf_desc

# External-memmap descriptor (producer-owned SHM pages mapped to the device via
# the Level Zero external-memmap extension; no buffer pool involved). Valid ONLY
# when transport_mode == TRANSPORT_EXTERNAL_MEMMAP; ignored otherwise. See
# ExternalMemMapDescriptor.msg for per-field details and lifetime caveats.
intel_buffer_backend_msgs/ExternalMemMapDescriptor extmap_desc
