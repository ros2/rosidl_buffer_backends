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
from std_msgs.msg import Bool, Float64, UInt32


def _make_fallback_subscriber(name, env_name, env_value, remapping_prefix):
    """Create a subscriber component container with env override for fallback testing."""
    return ComposableNodeContainer(
        name=f'{name}_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageSubscriber',
                name=f'{name}',
                parameters=[{
                    'expected_backend': 'cpu',
                }],
                remappings=[
                    ('subscriber_count', f'{remapping_prefix}_count'),
                    ('validation_result', f'{remapping_prefix}_validation'),
                    ('backend_validation', f'{remapping_prefix}_backend_validation'),
                    ('latency_ms', f'{remapping_prefix}_latency'),
                ],
            ),
        ],
        output='screen',
        additional_env={env_name: env_value, 'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp'},
    )


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """Generate launch description for CPU fallback test over FastRTPS."""
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

    ipc_subscriber = ComposableNodeContainer(
        name='ipc_subscriber_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='hip_buffer_backend',
                plugin='HipImageSubscriber',
                name='ipc_subscriber',
                parameters=[{
                    'expected_backend': 'hip',
                }],
                remappings=[
                    ('subscriber_count', 'ipc_count'),
                    ('validation_result', 'ipc_validation'),
                    ('backend_validation', 'ipc_backend_validation'),
                    ('latency_ms', 'ipc_latency'),
                ],
            ),
        ],
        output='screen',
    )

    cross_device_sub = _make_fallback_subscriber(
        'cross_device_sub', 'HIP_BUFFER_DEVICE_ID_OVERRIDE', '999', 'cross_device')
    cross_user_sub = _make_fallback_subscriber(
        'cross_user_sub', 'HIP_BUFFER_UID_OVERRIDE', '99999', 'cross_user')

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        ipc_subscriber,
        cross_device_sub,
        cross_user_sub,
        TimerAction(period=2.0, actions=[
            publisher_container,
            launch_testing.actions.ReadyToTest(),
        ]),
    ])


class TestHipImageCpuFallbackFastRTPS(unittest.TestCase):
    """Test CPU fallback paths for on_discovering_endpoint over FastRTPS."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_hip_cpu_fallback_fastrtps')
        self.ipc_count = 0
        self.cross_device_count = 0
        self.cross_user_count = 0
        self.ipc_validation = True
        self.cross_device_validation = True
        self.cross_user_validation = True
        self.ipc_backend_validation = True
        self.cross_device_backend_validation = True
        self.cross_user_backend_validation = True
        self.ipc_latencies = []
        self.cross_device_latencies = []
        self.cross_user_latencies = []

        self.node.create_subscription(
            UInt32, 'ipc_count', self._ipc_count_cb, 10)
        self.node.create_subscription(
            Bool, 'ipc_validation', self._ipc_validation_cb, 10)
        self.node.create_subscription(
            Float64, 'ipc_latency', self._ipc_latency_cb, 10)
        self.node.create_subscription(
            UInt32, 'cross_device_count', self._cross_device_count_cb, 10)
        self.node.create_subscription(
            Bool, 'cross_device_validation', self._cross_device_validation_cb, 10)
        self.node.create_subscription(
            Float64, 'cross_device_latency', self._cross_device_latency_cb, 10)
        self.node.create_subscription(
            UInt32, 'cross_user_count', self._cross_user_count_cb, 10)
        self.node.create_subscription(
            Bool, 'cross_user_validation', self._cross_user_validation_cb, 10)
        self.node.create_subscription(
            Float64, 'cross_user_latency', self._cross_user_latency_cb, 10)
        self.node.create_subscription(
            Bool, 'ipc_backend_validation', self._ipc_backend_validation_cb, 10)
        self.node.create_subscription(
            Bool, 'cross_device_backend_validation',
            self._cross_device_backend_validation_cb, 10)
        self.node.create_subscription(
            Bool, 'cross_user_backend_validation',
            self._cross_user_backend_validation_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _ipc_count_cb(self, msg):
        self.ipc_count = msg.data

    def _ipc_validation_cb(self, msg):
        self.ipc_validation = msg.data

    def _cross_device_count_cb(self, msg):
        self.cross_device_count = msg.data

    def _cross_device_validation_cb(self, msg):
        self.cross_device_validation = msg.data

    def _cross_user_count_cb(self, msg):
        self.cross_user_count = msg.data

    def _cross_user_validation_cb(self, msg):
        self.cross_user_validation = msg.data

    def _ipc_latency_cb(self, msg):
        self.ipc_latencies.append(msg.data)

    def _cross_device_latency_cb(self, msg):
        self.cross_device_latencies.append(msg.data)

    def _cross_user_latency_cb(self, msg):
        self.cross_user_latencies.append(msg.data)

    def _ipc_backend_validation_cb(self, msg):
        if not msg.data:
            self.ipc_backend_validation = False

    def _cross_device_backend_validation_cb(self, msg):
        if not msg.data:
            self.cross_device_backend_validation = False

    def _cross_user_backend_validation_cb(self, msg):
        if not msg.data:
            self.cross_user_backend_validation = False

    def _spin_until(self, timeout_sec=30.0):
        start = time.time()
        while ((self.ipc_count < 5 or
                self.cross_device_count < 5 or
                self.cross_user_count < 5) and
               time.time() - start < timeout_sec):
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return (self.ipc_count >= 5 and
                self.cross_device_count >= 5 and
                self.cross_user_count >= 5)

    def test_cpu_fallback_paths(self):
        """Test all CPU fallback paths and normal IPC simultaneously over FastRTPS."""
        success = self._spin_until(timeout_sec=30.0)

        self.assertTrue(
            success,
            f'Failed to receive 5 messages from all subscribers. '
            f'IPC: {self.ipc_count}, '
            f'Cross-device: {self.cross_device_count}, '
            f'Cross-user: {self.cross_user_count}')

        self.assertTrue(
            self.ipc_backend_validation,
            'Normal IPC backend check failed (expected backend="hip")')

        self.assertTrue(
            self.ipc_validation,
            'Normal IPC validation failed (content or metadata error)')

        self.assertTrue(
            self.cross_device_backend_validation,
            'Cross-device backend check failed (expected backend="cpu")')

        self.assertTrue(
            self.cross_device_validation,
            'Cross-device validation failed (content or metadata error)')

        self.assertTrue(
            self.cross_user_backend_validation,
            'Cross-user backend check failed (expected backend="cpu")')

        self.assertTrue(
            self.cross_user_validation,
            'Cross-user validation failed (content or metadata error)')


@launch_testing.post_shutdown_test()
class TestHipImageCpuFallbackFastRTPSShutdown(unittest.TestCase):
    """Test proper shutdown of nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
