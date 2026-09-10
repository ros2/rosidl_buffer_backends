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

import os
import time
import unittest

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
from launch_testing_ros.actions import EnableRmwIsolation
import pytest
import rclpy
from std_msgs.msg import Bool, UInt32


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """Generate a deterministic same-process CUDA event regression test."""
    test_domain_id = str(100 + os.getpid() % 100)

    publisher_container = ComposableNodeContainer(
        name='cuda_image_intra_event_publisher_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='cuda_buffer_backend',
                plugin='CudaImageIntraProcessEventPublisher',
                extra_arguments=[{'use_intra_process_comms': False}],
                name='cuda_image_intra_event_publisher',
            ),
        ],
        output='screen',
    )

    subscriber_loader = LoadComposableNodes(
        target_container='/cuda_image_intra_event_publisher_container',
        composable_node_descriptions=[
            ComposableNode(
                package='cuda_buffer_backend',
                plugin='CudaImageSubscriber',
                extra_arguments=[{'use_intra_process_comms': False}],
                name='cuda_image_intra_event_subscriber',
                parameters=[{
                    'expected_backend': 'cuda',
                    'acceptable_buffer_backends': 'cuda',
                    'validate_sequence_pattern': True,
                }],
                remappings=[
                    ('test_cuda_image', 'test_cuda_image_intra_event'),
                    ('subscriber_count', 'cuda_image_intra_event_subscriber_count'),
                    ('validation_result', 'cuda_image_intra_event_validation'),
                    ('backend_validation', 'cuda_image_intra_event_backend_validation'),
                    ('content_validation', 'cuda_image_intra_event_content_validation'),
                    ('metadata_validation', 'cuda_image_intra_event_metadata_validation'),
                ],
            ),
        ],
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', test_domain_id),
        EnableRmwIsolation(),
        publisher_container,
        subscriber_loader,
        launch_testing.actions.ReadyToTest(),
    ])


class TestCudaImageIntraProcessEventFastRTPS(unittest.TestCase):
    """Verify CUDA descriptor transport without publish-side CUDA synchronization."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_cuda_image_intra_event_fastrtps')
        self.received_count = 0
        self.validation_count = 0
        self.backend_validation_count = 0
        self.content_validation_count = 0
        self.metadata_validation_count = 0
        self.nonblocking_validation_count = 0
        self.all_valid = True

        self.node.create_subscription(
            UInt32, 'cuda_image_intra_event_subscriber_count', self._count_cb, 10)
        self.node.create_subscription(
            Bool, 'cuda_image_intra_event_validation', self._validation_cb, 10)
        self.node.create_subscription(
            Bool, 'cuda_image_intra_event_backend_validation', self._backend_cb, 10)
        self.node.create_subscription(
            Bool, 'cuda_image_intra_event_content_validation', self._content_cb, 10)
        self.node.create_subscription(
            Bool, 'cuda_image_intra_event_metadata_validation', self._metadata_cb, 10)
        self.node.create_subscription(
            Bool, 'intra_event_publish_nonblocking', self._nonblocking_cb, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _count_cb(self, msg):
        self.received_count = msg.data

    def _validation_cb(self, msg):
        self.validation_count += 1
        self.all_valid = self.all_valid and msg.data

    def _backend_cb(self, msg):
        self.backend_validation_count += 1
        self.all_valid = self.all_valid and msg.data

    def _content_cb(self, msg):
        self.content_validation_count += 1
        self.all_valid = self.all_valid and msg.data

    def _metadata_cb(self, msg):
        self.metadata_validation_count += 1
        self.all_valid = self.all_valid and msg.data

    def _nonblocking_cb(self, msg):
        self.nonblocking_validation_count += 1
        self.all_valid = self.all_valid and msg.data

    def _complete(self):
        return (
            self.received_count >= 5 and
            self.validation_count >= 5 and
            self.backend_validation_count >= 5 and
            self.content_validation_count >= 5 and
            self.metadata_validation_count >= 5 and
            self.nonblocking_validation_count >= 5
        )

    def test_five_cuda_messages_validate(self):
        """Require an explicit successful validation for every CUDA message."""
        deadline = time.monotonic() + 30.0
        while not self._complete() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.assertTrue(
            self._complete(),
            'Did not observe five complete validations: '
            f'received={self.received_count}, validation={self.validation_count}, '
            f'backend={self.backend_validation_count}, '
            f'content={self.content_validation_count}, '
            f'metadata={self.metadata_validation_count}, '
            f'nonblocking={self.nonblocking_validation_count}',
        )
        self.assertTrue(self.all_valid, 'At least one CUDA same-process event validation failed')


@launch_testing.post_shutdown_test()
class TestCudaImageIntraProcessEventFastRTPSShutdown(unittest.TestCase):
    """Check that the component container shut down cleanly."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
