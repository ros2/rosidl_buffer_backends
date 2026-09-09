#!/usr/bin/env python3
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

import time
import unittest

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node
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
    """One publisher, two subscribers (1-to-N fan-out) over FastRTPS."""
    publisher_node = Node(
        package='torch_conversions_cpu',
        executable='torch_tensor_publisher_node',
        name='torch_tensor_publisher',
        output='screen',
        parameters=[{
            'max_publish_count': 0,
            'publish_rate_ms': 200,
        }],
    )

    subscriber1_node = Node(
        package='torch_conversions_cpu',
        executable='torch_tensor_subscriber_node',
        name='torch_tensor_subscriber_1',
        output='screen',
        remappings=[
            ('subscriber_count', 'subscriber_1_count'),
            ('validation_result', 'subscriber_1_validation'),
        ],
    )

    subscriber2_node = Node(
        package='torch_conversions_cpu',
        executable='torch_tensor_subscriber_node',
        name='torch_tensor_subscriber_2',
        output='screen',
        remappings=[
            ('subscriber_count', 'subscriber_2_count'),
            ('validation_result', 'subscriber_2_validation'),
        ],
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        subscriber1_node,
        subscriber2_node,
        TimerAction(period=2.0, actions=[
            publisher_node,
            launch_testing.actions.ReadyToTest(),
        ]),
    ])


class TestTorchTensorMultiSubFastRTPS(unittest.TestCase):
    """1-to-N: one publisher sending to two subscribers over FastRTPS."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_torch_tensor_multi_sub_fastrtps')
        self.publisher_count = 0
        self.subscriber1_count = 0
        self.subscriber2_count = 0
        self.subscriber1_validation = True
        self.subscriber2_validation = True

        self.node.create_subscription(
            UInt32, 'publisher_count', self._pub_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'subscriber_1_count', self._sub1_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'subscriber_2_count', self._sub2_count_cb, 10)
        self.node.create_subscription(
            Bool, 'subscriber_1_validation', self._sub1_validation_cb, 10)
        self.node.create_subscription(
            Bool, 'subscriber_2_validation', self._sub2_validation_cb, 10)

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
        while ((self.publisher_count < target_count or
                self.subscriber1_count < target_count or
                self.subscriber2_count < target_count) and
               time.time() - start < timeout_sec):
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return (self.publisher_count >= target_count and
                self.subscriber1_count >= target_count and
                self.subscriber2_count >= target_count)

    def test_multi_sub(self):
        """Both subscribers receive messages from the single publisher."""
        success = self._spin_until(target_count=5, timeout_sec=30.0)

        self.assertTrue(
            success,
            f'Failed to receive 5 messages in both subscribers within timeout. '
            f'Sub1: {self.subscriber1_count}, Sub2: {self.subscriber2_count}')
        self.assertGreaterEqual(
            self.publisher_count, 5,
            f'Publisher should have sent at least 5 messages. '
            f'Sent: {self.publisher_count}')
        self.assertTrue(
            self.subscriber1_validation, 'Subscriber1 validation failed')
        self.assertTrue(
            self.subscriber2_validation, 'Subscriber2 validation failed')


@launch_testing.post_shutdown_test()
class TestTorchTensorMultiSubFastRTPSShutdown(unittest.TestCase):
    """Test proper shutdown of nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, 1, -2, -6, -9, -11, -15],
        )
