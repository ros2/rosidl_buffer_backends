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

import re
import unittest

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """Launch an intra-process tensor pub/sub pair over FastRTPS."""
    container = ComposableNodeContainer(
        name='torch_tensor_intraprocess_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='torch_conversions',
                plugin='TorchTensorPublisher',
                parameters=[{'device': 'cpu'}],
                name='torch_tensor_publisher',
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='torch_conversions',
                plugin='TorchTensorSubscriber',
                name='torch_tensor_subscriber',
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([
        container,
        launch_testing.actions.ReadyToTest(),
    ]), {'subscriber': container}


class TestTorchTensorIntraPubSubFastRTPS(unittest.TestCase):

    def test_received_tensors(self, proc_output, subscriber):
        proc_output.assertWaitFor(
            'count=5)', process=subscriber, timeout=15)
        output = b''.join(event.text for event in proc_output[subscriber]).decode()
        samples = re.findall(
            r'Received tensor \(backend=(\w+), min=([^,]+), '
            r'max=([^,]+), count=\d+\)',
            output)
        self.assertGreaterEqual(len(samples), 5)
        for received_backend, minimum, maximum in samples:
            self.assertEqual(received_backend, 'cpu')
            self.assertEqual(float(minimum), float(maximum))
            self.assertGreaterEqual(float(minimum), 0)
            self.assertLessEqual(float(maximum), 255)


@launch_testing.post_shutdown_test()
class TestTorchTensorIntraPubSubFastRTPSShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
