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
    """Generate launch description for multi-process HIP image pub/sub test over FastRTPS."""
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
                    'max_publish_count': 0,
                    'publish_rate_ms': 100,
                }],
            ),
        ],
        output='screen',
    )

    subscriber1_container = ComposableNodeContainer(
        name='hip_image_subscriber_1_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageSubscriber',
                name='hip_image_subscriber_1',
                parameters=[{}],
                remappings=[
                    ('subscriber_count', 'hip_image_subscriber_1_count'),
                    ('validation_result', 'hip_image_subscriber_1_validation'),
                ],
            ),
        ],
        output='screen',
    )

    subscriber2_container = ComposableNodeContainer(
        name='hip_image_subscriber_2_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageSubscriber',
                name='hip_image_subscriber_2',
                parameters=[{}],
                remappings=[
                    ('subscriber_count', 'hip_image_subscriber_2_count'),
                    ('validation_result', 'hip_image_subscriber_2_validation'),
                ],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        subscriber1_container,
        subscriber2_container,
        publisher_container,
        launch_testing.actions.ReadyToTest(),
    ])


class TestHipImageMultiProcessFastRTPS(unittest.TestCase):
    """Test case for multi-process HIP Buffer-based Image pub/sub over FastRTPS."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_hip_image_multiprocess_fastrtps')
        self.publisher_count = 0
        self.subscriber1_count = 0
        self.subscriber2_count = 0
        self.subscriber1_validation = True
        self.subscriber2_validation = True

        self.node.create_subscription(
            UInt32, 'publisher_count', self._pub_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'hip_image_subscriber_1_count', self._sub1_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'hip_image_subscriber_2_count', self._sub2_count_cb, 10)
        self.node.create_subscription(
            Bool, 'hip_image_subscriber_1_validation', self._sub1_validation_cb, 10)
        self.node.create_subscription(
            Bool, 'hip_image_subscriber_2_validation', self._sub2_validation_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _pub_count_cb(self, msg):
        self.publisher_count = msg.data

    def _sub1_count_cb(self, msg):
        self.subscriber1_count = msg.data

    def _sub2_count_cb(self, msg):
        self.subscriber2_count = msg.data

    def _sub1_validation_cb(self, msg):
        self.subscriber1_validation = msg.data

    def _sub2_validation_cb(self, msg):
        self.subscriber2_validation = msg.data

    def _spin_until(self, target_count=5, timeout_sec=30.0):
        start = time.time()
        while ((self.subscriber1_count < target_count or
                self.subscriber2_count < target_count) and
               time.time() - start < timeout_sec):
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return (self.subscriber1_count >= target_count and
                self.subscriber2_count >= target_count)

    def test_multiprocess_pubsub(self):
        """Test HIP buffer-based multi-process image pub/sub over FastRTPS."""
        success = self._spin_until(target_count=5, timeout_sec=30.0)

        self.assertTrue(
            success,
            f'Failed to receive 5 messages in both subscribers within timeout. '
            f'Sub1: {self.subscriber1_count}, Sub2: {self.subscriber2_count}')
        self.assertGreaterEqual(
            self.publisher_count, 5,
            f'Publisher should have sent at least 5 messages. Sent: {self.publisher_count}')
        self.assertTrue(self.subscriber1_validation, 'Subscriber1 validation failed')
        self.assertTrue(self.subscriber2_validation, 'Subscriber2 validation failed')
        self.assertLessEqual(
            abs(self.subscriber1_count - self.subscriber2_count), 2,
            f'Subscriber1 count ({self.subscriber1_count}) and Subscriber2 count '
            f'({self.subscriber2_count}) differ by more than 2')


@launch_testing.post_shutdown_test()
class TestHipImageMultiProcessFastRTPSShutdown(unittest.TestCase):
    """Test proper shutdown of nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
