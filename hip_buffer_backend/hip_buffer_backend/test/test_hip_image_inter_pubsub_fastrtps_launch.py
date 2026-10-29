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
from std_msgs.msg import Bool, Float64, UInt32


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """Generate launch description for HIP image pub/sub test with FastRTPS RMW."""
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
                    'image_width': 1920,
                    'image_height': 1080,
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
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        publisher_container,
        subscriber_container,
        launch_testing.actions.ReadyToTest(),
    ])


class TestHipImagePubSubFastRTPS(unittest.TestCase):
    """Test case for HIP Buffer-based Image pub/sub over FastRTPS."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_hip_image_pubsub_fastrtps')
        self.publisher_count = 0
        self.subscriber_count = 0
        self.validation_passed = True
        self.latencies = []

        self.node.create_subscription(
            UInt32, 'publisher_count', self._pub_count_cb, 10)
        self.node.create_subscription(
            UInt32, 'subscriber_count', self._sub_count_cb, 10)
        self.node.create_subscription(
            Bool, 'validation_result', self._validation_cb, 10)
        self.node.create_subscription(
            Float64, 'latency_ms', self._latency_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _pub_count_cb(self, msg):
        self.publisher_count = msg.data

    def _sub_count_cb(self, msg):
        self.subscriber_count = msg.data

    def _validation_cb(self, msg):
        self.validation_passed = msg.data

    def _latency_cb(self, msg):
        self.latencies.append(msg.data)

    def _spin_until(self, target_count=5, timeout_sec=15.0):
        start = time.time()
        while self.subscriber_count < target_count and time.time() - start < timeout_sec:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.subscriber_count >= target_count

    def test_inter_process_pubsub(self):
        """Test HIP buffer-based inter-process image pub/sub over FastRTPS."""
        success = self._spin_until(target_count=5, timeout_sec=15.0)

        self.assertTrue(
            success,
            f'Failed to receive 5 messages within timeout. '
            f'Received: {self.subscriber_count}')
        self.assertGreaterEqual(
            self.publisher_count, 5,
            f'Publisher should have sent at least 5 messages. Sent: {self.publisher_count}')
        self.assertTrue(self.validation_passed,
                        'Image validation failed (backend type mismatch indicates CPU fallback)')
        self.assertLessEqual(
            abs(self.publisher_count - self.subscriber_count), 2,
            f'Publisher count ({self.publisher_count}) and subscriber count '
            f'({self.subscriber_count}) differ by more than 2')

        self.assertGreater(len(self.latencies), 0, 'No latency measurements received')
        steady = self.latencies[1:] if len(self.latencies) > 1 else self.latencies
        mean_lat = sum(steady) / len(steady)
        self.assertLess(
            mean_lat, 50.0,
            f'Steady-state mean latency {mean_lat:.3f} ms exceeds 50 ms threshold')


@launch_testing.post_shutdown_test()
class TestHipImagePubSubFastRTPSShutdown(unittest.TestCase):
    """Test proper shutdown of nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
