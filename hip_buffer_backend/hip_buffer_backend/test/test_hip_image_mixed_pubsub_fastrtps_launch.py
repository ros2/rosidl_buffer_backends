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
    """Generate launch description for mixed HIP image pub/sub over FastRTPS."""
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


class TestHipImageMixedPubSubFastRTPS(unittest.TestCase):
    """Test case for mixed intra/inter-process HIP Buffer-based Image pub/sub over FastRTPS."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_hip_image_mixed_pubsub_fastrtps')
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

    def _spin_until(self, target_count=5, timeout_sec=20.0):
        start = time.time()
        while ((self.intra_subscriber_count < target_count or
                self.inter_subscriber_count < target_count) and
               time.time() - start < timeout_sec):
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return (self.intra_subscriber_count >= target_count and
                self.inter_subscriber_count >= target_count)

    def test_mixed_pubsub(self):
        """Test HIP buffer-based mixed intra/inter-process image pub/sub over FastRTPS."""
        success = self._spin_until(target_count=5, timeout_sec=20.0)

        self.assertTrue(
            success,
            f'Failed to receive 5 messages in both subscribers within timeout. '
            f'Intra: {self.intra_subscriber_count}, Inter: {self.inter_subscriber_count}')
        self.assertGreaterEqual(
            self.publisher_count, 5,
            f'Publisher should have sent at least 5 messages. Sent: {self.publisher_count}')
        self.assertTrue(self.intra_validation, 'Intra-process validation failed')
        self.assertTrue(self.inter_validation, 'Inter-process validation failed')
        self.assertLessEqual(
            self.intra_subscriber_count, self.publisher_count,
            f'Intra subscriber received more messages ({self.intra_subscriber_count}) '
            f'than publisher sent ({self.publisher_count}), '
            f'indicating duplicate delivery')
        self.assertLessEqual(
            abs(self.intra_subscriber_count - self.inter_subscriber_count), 2,
            f'Intra count ({self.intra_subscriber_count}) and Inter count '
            f'({self.inter_subscriber_count}) differ by more than 2')


@launch_testing.post_shutdown_test()
class TestHipImageMixedPubSubFastRTPSShutdown(unittest.TestCase):
    """Test proper shutdown of nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
