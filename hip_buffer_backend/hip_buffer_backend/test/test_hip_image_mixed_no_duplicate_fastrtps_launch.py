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
from launch.actions import SetEnvironmentVariable, TimerAction
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
    """Launch mixed intra/inter pub/sub with slow publish for duplicate detection."""
    container = ComposableNodeContainer(
        name='hip_image_mixed_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImagePublisher',
                name='hip_image_publisher',
                parameters=[{
                    'max_publish_count': 10,
                    'publish_rate_ms': 1000,
                }],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageSubscriber',
                name='hip_image_subscriber_intra',
                remappings=[
                    ('subscriber_count', 'intra_subscriber_count'),
                    ('validation_result', 'intra_validation_result'),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    subscriber_inter_container = ComposableNodeContainer(
        name='hip_image_subscriber_inter_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageSubscriber',
                name='hip_image_subscriber_inter',
                parameters=[{}],
                remappings=[
                    ('subscriber_count', 'inter_subscriber_count'),
                    ('validation_result', 'inter_validation_result'),
                ],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        subscriber_inter_container,
        TimerAction(period=2.0, actions=[
            container,
            launch_testing.actions.ReadyToTest(),
        ]),
    ])


class TestNoDuplicateDelivery(unittest.TestCase):
    """Verify intra-process subscriber receives exactly one message per publish."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_no_duplicate_delivery')
        self.publisher_count = 0
        self.intra_subscriber_count = 0
        self.inter_subscriber_count = 0
        self.intra_validation = True
        self.inter_validation = True

        self.node.create_subscription(
            UInt32, 'publisher_count', self._pub_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'intra_subscriber_count', self._intra_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'inter_subscriber_count', self._inter_count_cb, 10)
        self.node.create_subscription(
            Bool, 'intra_validation_result', self._intra_validation_cb, 10)
        self.node.create_subscription(
            Bool, 'inter_validation_result', self._inter_validation_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _pub_count_cb(self, msg):
        self.publisher_count = msg.data

    def _intra_count_cb(self, msg):
        self.intra_subscriber_count = msg.data

    def _inter_count_cb(self, msg):
        self.inter_subscriber_count = msg.data

    def _intra_validation_cb(self, msg):
        self.intra_validation = msg.data

    def _inter_validation_cb(self, msg):
        self.inter_validation = msg.data

    def _spin_until_done(self, timeout_sec=30.0, drain_sec=3.0):
        start = time.time()
        while self.publisher_count < 10 and time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        drain_start = time.time()
        while time.time() - drain_start < drain_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.publisher_count >= 10

    def test_no_duplicate_messages(self):
        """Verify exact message counts: no duplicates for intra-process subscriber."""
        success = self._spin_until_done(timeout_sec=30.0)

        self.assertTrue(
            success,
            f'Publisher did not reach 10 messages. '
            f'publisher_count={self.publisher_count}')

        self.assertLessEqual(
            self.intra_subscriber_count, self.publisher_count,
            f'DUPLICATE DETECTED: intra subscriber received '
            f'{self.intra_subscriber_count} messages but publisher only sent '
            f'{self.publisher_count}')

        self.assertGreaterEqual(
            self.intra_subscriber_count, self.publisher_count - 1,
            f'Intra subscriber missed messages: received '
            f'{self.intra_subscriber_count}, expected ~{self.publisher_count}')

        self.assertTrue(
            self.intra_validation, 'Intra-process validation failed')
        self.assertTrue(
            self.inter_validation, 'Inter-process validation failed')


@launch_testing.post_shutdown_test()
class TestNoDuplicateDeliveryShutdown(unittest.TestCase):
    """Test proper shutdown of nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
