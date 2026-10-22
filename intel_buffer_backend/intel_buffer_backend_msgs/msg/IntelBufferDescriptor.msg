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

# Zero-copy frame descriptor.
#
# This is the message published on the frame topic. It carries ONLY image
# geometry plus one transport descriptor (~200 B) — never the pixels. The pixel
# data stays in shared USM / DMA-BUF memory and is reached via the descriptor,
# so publisher, inference and visualizer nodes touch the same physical pages
# with zero copies.

# Standard ROS header. header.stamp is the capture time; header.frame_id names
# the source ("camera"). Propagated verbatim onto the annotated sensor_msgs/Image
# so downstream tools (RViz2, tf) keep correct timing/frame association.
std_msgs/Header header

# Image width in pixels. Read by every subscriber to wrap the shared buffer as a
# cv::Mat / ov::Tensor of the right shape. Set from the video/camera geometry.
uint32 width

# Image height in pixels. Same role as width. For NV12 the buffer holds
# height*3/2 rows (Y plane + interleaved UV); subscribers derive that from
# height + encoding.
uint32 height

# Row stride in bytes (bytes per row of the primary/Y plane), which may exceed
# width*channels due to hardware alignment. Read by the visualizer to build the
# cv::Mat with the correct step so padded camera rows are not misread. A value
# smaller than the minimum is treated as tightly packed.
uint32 stride

# Pixel format of the shared buffer: "bgr8" (simulated / BGR source) or "nv12"
# (real camera). Read by the OpenVINO node to pick the GPU preprocessor color
# path and by the visualizer to decide whether to convert NV12->BGR for display.
string encoding

# Monotonic frame counter set by the publisher. Used for logging/throughput
# accounting (e.g. "published N frames") and to help spot dropped frames. Not
# required for buffer import.
uint32 frame_seq

# Nanosecond timestamp (node clock) taken by the publisher when the shared
# buffer is ready to publish. A subscriber subtracts it from its own clock right
# after a successful zero-copy import to measure the buffer-ready -> buffer-read
# delay. 0 means the publisher did not stamp it. Not required for buffer import.
uint64 buffer_ready_stamp_ns

# Transport selection — tells subscribers which descriptor below is valid and
# which import path to take. Set by the publisher from the pool's active
# transport (see INTEL_BUFFER_TYPE / the launch transport_mode param).
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
