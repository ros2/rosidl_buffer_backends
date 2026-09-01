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

# Runs the publisher and subscriber as composable nodes in ONE container
# process, so qc_buffer_backend delivers the payload zero-copy.

import launch
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    pub_node = ComposableNode(
        package='qc_buffer_tutorial',
        plugin='QcImagePublisher',
        name='qc_image_publisher',
    )

    sub_node = ComposableNode(
        package='qc_buffer_tutorial',
        plugin='QcImageSubscriber',
        name='qc_image_subscriber',
    )

    container = ComposableNodeContainer(
        name = "container",
        namespace = "qc_buffer_backend_test",
        package = "rclcpp_components",
        executable='component_container',
        output = "screen",
        composable_node_descriptions=[pub_node, sub_node]
    )

    return launch.LaunchDescription([container])
