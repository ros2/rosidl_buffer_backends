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

"""Intel USM zero-copy pipeline: camera → iGPU inference → RViz2 visualization.

Pipeline:
  1) video_dmabuf_sim      — Publishes IntelMemoryBufferImageFrame (descriptor + fallback data)
  2) openvino_pipeline     — Imports buffer (zero-copy), runs GPU inference,
                             publishes Detection2DArray
  3) detection_visualizer  — Imports buffer, draws bounding boxes, publishes
                             annotated sensor_msgs/Image
  4) rviz2                 — Displays the annotated image

Source modes (selected via `source_mode`):
  "simulated" — video file → allocate_buffer + write_to_buffer (1 memcpy).
                Backing store chosen via `transport_mode` (L0_USM, DMA_BUF or
                EXTERNAL_MEMMAP).
  "camera"    — V4L2 camera → wrap_dmabuf (zero copy end-to-end). Always
                DMA-BUF; `transport_mode` does not apply.

transport_mode=EXTERNAL_MEMMAP is the pool-free path: the publisher allocates its
own ring of shared regions and maps them to the device with the Level Zero
external-memmap extension; subscribers map the same physical pages from the region
name in the descriptor and feed them to the GPU as a USM RemoteTensor. Requires a
driver that advertises the extension.

Usage:
    ros2 launch intel_buffer_backend usm_openvino_video.launch.py \\
        video_path:=/path/to/sample.mp4 transport_mode:=DMA_BUF

    ros2 launch intel_buffer_backend usm_openvino_video.launch.py \\
        source_mode:=camera camera_device:=/dev/video0
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("test_intel_buffer_backend")
    rviz_config = os.path.join(pkg_share, "config", "usm_pipeline.rviz")
    fastdds_profile = os.path.join(pkg_share, "config", "fastdds_shm.xml")

    default_model = os.path.join(
        "/opt/openvino_toolkit/models/convert/public",
        "ssdlite_mobilenet_v2/FP16/ssdlite_mobilenet_v2.xml",
    )

    source_mode = LaunchConfiguration("source_mode")
    transport_mode = LaunchConfiguration("transport_mode")
    video_path = LaunchConfiguration("video_path")
    camera_device = LaunchConfiguration("camera_device")
    framerate = LaunchConfiguration("framerate")
    loop = LaunchConfiguration("loop")
    raw_format = LaunchConfiguration("raw_format")
    model = LaunchConfiguration("model")
    device = LaunchConfiguration("device")
    frame_topic = LaunchConfiguration("frame_topic")
    detections_topic = LaunchConfiguration("detections_topic")
    annotated_topic = LaunchConfiguration("annotated_topic")
    score_thresh = LaunchConfiguration("score_thresh")
    show_rviz = LaunchConfiguration("show_rviz")

    input_format = PythonExpression(
        ["'nv12' if '", source_mode, "' == 'camera' else 'bgr8'"]
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("FASTDDS_DEFAULT_PROFILES_FILE", fastdds_profile),

            # --- Source selection ---
            DeclareLaunchArgument(
                "source_mode",
                default_value="simulated",
                description="'simulated' (video file) or 'camera' (real V4L2 camera).",
            ),
            DeclareLaunchArgument(
                "transport_mode",
                default_value="L0_USM",
                description="Simulated-mode backing store: 'L0_USM' (Level Zero "
                            "host USM), 'DMA_BUF' (kernel dma-heap) or "
                            "'EXTERNAL_MEMMAP' (publisher-owned shared regions "
                            "mapped to the device via the Level Zero "
                            "external-memmap extension; no buffer pool). Ignored "
                            "for source_mode=camera (always DMA-BUF).",
            ),
            DeclareLaunchArgument(
                "video_path",
                default_value="",
                description="Video file path. Pass ONLY for source_mode=simulated.",
            ),
            DeclareLaunchArgument(
                "camera_device",
                default_value="/dev/video0",
                description="V4L2 device (used when source_mode=camera).",
            ),

            # --- Common parameters ---
            DeclareLaunchArgument(
                "framerate",
                default_value="0.0",
                description="FPS; 0 uses video's native rate.",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Loop video playback (simulated mode only).",
            ),
            DeclareLaunchArgument(
                "raw_format",
                default_value="YUYV",
                description="Intermediate raw format for simulation (YUYV or NV12).",
            ),
            DeclareLaunchArgument(
                "model",
                default_value=default_model,
                description="OpenVINO IR model (.xml).",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="GPU",
                description="OpenVINO device (GPU or NPU).",
            ),
            DeclareLaunchArgument(
                "frame_topic",
                default_value="/usm/camera0/frame",
                description="IntelMemoryBufferImageFrame topic (descriptor + data).",
            ),
            DeclareLaunchArgument(
                "detections_topic",
                default_value="/ros2_openvino_toolkit/detected_objects",
                description="Detection output topic.",
            ),
            DeclareLaunchArgument(
                "annotated_topic",
                default_value="/usm/camera0/annotated",
                description="Annotated image topic (sensor_msgs/Image for RViz2).",
            ),
            DeclareLaunchArgument(
                "score_thresh",
                default_value="0.4",
                description="Detection confidence threshold.",
            ),
            DeclareLaunchArgument(
                "show_rviz",
                default_value="true",
                description="Launch RViz2 for visualization.",
            ),

            # --- Node 1: Camera publisher (IntelMemoryBufferImageFrame) ---
            Node(
                package="test_intel_buffer_backend",
                executable="video_publisher_node",
                name="video_dmabuf_sim",
                parameters=[
                    {
                        "source_mode": source_mode,
                        "intel_buffer_type": transport_mode,
                        "video_path": video_path,
                        "camera_device": camera_device,
                        "framerate": framerate,
                        "loop": loop,
                        "raw_format": raw_format,
                        "frame_topic": frame_topic,
                    }
                ],
                output="screen",
            ),

            # --- Node 2: OpenVINO inference (zero-copy import → GPU) ---
            Node(
                package="test_intel_buffer_backend",
                executable="openvino_pipeline_node",
                name="openvino_pipeline",
                parameters=[
                    {
                        "model": model,
                        "device": device,
                        "frame_topic": frame_topic,
                        "detections_topic": detections_topic,
                        "score_thresh": score_thresh,
                        "input_format": input_format,
                    }
                ],
                output="screen",
            ),

            # --- Node 3: Detection visualizer (draws boxes → Image) ---
            Node(
                package="test_intel_buffer_backend",
                executable="detection_visualizer_node",
                name="detection_visualizer",
                parameters=[
                    {
                        "frame_topic": frame_topic,
                        "detections_topic": detections_topic,
                        "annotated_topic": annotated_topic,
                        "throttle_hz": 30.0,
                    }
                ],
                output="screen",
            ),

            # --- Node 4: RViz2 (displays annotated image with bounding boxes) ---
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                condition=IfCondition(show_rviz),
                arguments=["-d", rviz_config],
                output="screen",
            ),
        ]
    )
