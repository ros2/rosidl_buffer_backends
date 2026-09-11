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
    """Two publishers, one subscriber (N-to-1 fan-in) over FastRTPS."""
    publisher1_node = Node(
        package='torch_conversions',
        executable='torch_tensor_publisher_node',
        name='torch_tensor_publisher_1',
        output='screen',
        parameters=[{
            'max_publish_count': 0,
            'publish_rate_ms': 200,
        }],
        remappings=[
            ('publisher_count', 'publisher_1_count'),
        ],
    )

    publisher2_node = Node(
        package='torch_conversions',
        executable='torch_tensor_publisher_node',
        name='torch_tensor_publisher_2',
        output='screen',
        parameters=[{
            'max_publish_count': 0,
            'publish_rate_ms': 200,
        }],
        remappings=[
            ('publisher_count', 'publisher_2_count'),
        ],
    )

    subscriber_node = Node(
        package='torch_conversions',
        executable='torch_tensor_subscriber_node',
        name='torch_tensor_subscriber',
        output='screen',
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        subscriber_node,
        TimerAction(period=2.0, actions=[
            publisher1_node,
            publisher2_node,
            launch_testing.actions.ReadyToTest(),
        ]),
    ])


class TestTorchTensorMultiPubFastRTPS(unittest.TestCase):
    """N-to-1: two publishers sending to one subscriber over FastRTPS."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_torch_tensor_multi_pub_fastrtps')
        self.publisher1_count = 0
        self.publisher2_count = 0
        self.subscriber_count = 0
        self.validation_passed = True

        self.node.create_subscription(
            UInt32, 'publisher_1_count', self._pub1_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'publisher_2_count', self._pub2_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'subscriber_count', self._sub_count_cb, 10)
        self.node.create_subscription(
            Bool, 'validation_result', self._validation_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _pub1_count_cb(self, msg):
        self.publisher1_count = msg.data

    def _pub2_count_cb(self, msg):
        self.publisher2_count = msg.data

    def _sub_count_cb(self, msg):
        self.subscriber_count = msg.data

    def _validation_cb(self, msg):
        self.validation_passed = msg.data

    def _spin_until(self, target_count=8, timeout_sec=30.0):
        start = time.time()
        while ((self.subscriber_count < target_count or
                self.publisher1_count < 3 or
                self.publisher2_count < 3) and
               time.time() - start < timeout_sec):
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return (self.subscriber_count >= target_count and
                self.publisher1_count >= 3 and
                self.publisher2_count >= 3)

    def test_multi_pub_single_sub(self):
        """Subscriber receives from both publishers."""
        success = self._spin_until(target_count=8, timeout_sec=30.0)

        self.assertTrue(
            success,
            f'Failed to receive 8 messages within timeout. '
            f'Received: {self.subscriber_count}')
        self.assertGreaterEqual(
            self.publisher1_count, 3,
            f'Publisher1 should have sent at least 3 messages. '
            f'Sent: {self.publisher1_count}')
        self.assertGreaterEqual(
            self.publisher2_count, 3,
            f'Publisher2 should have sent at least 3 messages. '
            f'Sent: {self.publisher2_count}')
        self.assertTrue(self.validation_passed, 'Tensor validation failed')


@launch_testing.post_shutdown_test()
class TestTorchTensorMultiPubFastRTPSShutdown(unittest.TestCase):
    """Test proper shutdown of nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
