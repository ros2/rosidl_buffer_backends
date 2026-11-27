#!/usr/bin/env python3
# Copyright 2026 Open Source Robotics Foundation, Inc.
# Modified for the HIP port by ObeliskGate, 2026.
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

import time
import unittest

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from std_msgs.msg import Bool, UInt32


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    publisher_container = ComposableNodeContainer(
        name='hip_image_publisher_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImagePublisher',
                name='hip_image_publisher',
                parameters=[{
                    'max_publish_count': 20,
                    'publish_rate_ms': 100,
                    'image_width': 64,
                    'image_height': 64,
                }],
            ),
        ],
        output='screen',
    )

    relay1_container = ComposableNodeContainer(
        name='hip_image_gpu_relay_1_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageGpuRelay',
                name='hip_image_gpu_relay_1',
                parameters=[{
                    'input_topic': 'test_hip_image',
                    'output_topic': 'relay_1_out',
                }],
            ),
        ],
        output='screen',
    )

    relay2_container = ComposableNodeContainer(
        name='hip_image_gpu_relay_2_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageGpuRelay',
                name='hip_image_gpu_relay_2',
                parameters=[{
                    'input_topic': 'relay_1_out',
                    'output_topic': 'relay_2_out',
                }],
            ),
        ],
        output='screen',
    )

    relay3_container = ComposableNodeContainer(
        name='hip_image_gpu_relay_3_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageGpuRelay',
                name='hip_image_gpu_relay_3',
                parameters=[{
                    'input_topic': 'relay_2_out',
                    'output_topic': 'relay_3_out',
                }],
            ),
        ],
        output='screen',
    )

    subscriber_container = ComposableNodeContainer(
        name='hip_image_subscriber_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageSubscriber',
                name='hip_image_subscriber',
                parameters=[{
                    'expected_backend': 'hip',
                }],
                remappings=[
                    ('test_hip_image', 'relay_3_out'),
                ],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        publisher_container,
        relay1_container,
        relay2_container,
        relay3_container,
        subscriber_container,
        launch_testing.actions.ReadyToTest(),
    ])


class TestGpuPipeline(unittest.TestCase):
    """Validate GPU-only relay pipeline with no intermediate CPU sync."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_gpu_pipeline')
        self.relay_count = 0
        self.subscriber_count = 0
        self.validation_passed = True

        self.node.create_subscription(
            UInt32, 'relay_count', self._relay_cb, 10)
        self.node.create_subscription(
            UInt32, 'subscriber_count', self._sub_cb, 10)
        self.node.create_subscription(
            Bool, 'validation_result', self._val_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _relay_cb(self, msg):
        self.relay_count = msg.data

    def _sub_cb(self, msg):
        self.subscriber_count = msg.data

    def _val_cb(self, msg):
        self.validation_passed = msg.data

    def test_gpu_pipeline_no_cpu_sync(self):
        """Data flows Publisher->Relay(GPU-only)->Subscriber with correct content."""
        start = time.time()
        while self.subscriber_count < 5 and time.time() - start < 20.0:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertGreaterEqual(
            self.relay_count, 5,
            f'Relay should have forwarded at least 5 messages. Got: {self.relay_count}')
        self.assertGreaterEqual(
            self.subscriber_count, 5,
            f'Subscriber should have received at least 5 messages. Got: {self.subscriber_count}')
        self.assertTrue(
            self.validation_passed,
            'Content validation failed after GPU relay -- event ordering may be broken')


@launch_testing.post_shutdown_test()
class TestGpuPipelineShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
