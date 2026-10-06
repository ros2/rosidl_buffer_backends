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
import subprocess
import unittest

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """One publisher, two subscribers (1-to-N fan-out) over FastRTPS."""
    result = subprocess.run(['./check_cuda_available'], timeout=10)
    if result.returncode == 77:
        raise unittest.SkipTest('the CUDA conversion plugin or device is unavailable')
    result.check_returncode()

    publisher_node = Node(
        package='torch_conversions',
        executable='torch_tensor_publisher_node',
        name='torch_tensor_publisher',
        output='screen',
        parameters=[{
            'device': 'cuda',
            'max_publish_count': 0,
            'publish_rate_ms': 200,
        }],
    )

    subscriber1_node = Node(
        package='torch_conversions',
        executable='torch_tensor_subscriber_node',
        name='torch_tensor_subscriber_1',
        output='screen',
    )

    subscriber2_node = Node(
        package='torch_conversions',
        executable='torch_tensor_subscriber_node',
        name='torch_tensor_subscriber_2',
        output='screen',
    )

    return LaunchDescription([
        subscriber1_node,
        subscriber2_node,
        TimerAction(period=2.0, actions=[
            publisher_node,
            launch_testing.actions.ReadyToTest(),
        ]),
    ]), {'subscribers': [subscriber1_node, subscriber2_node]}


class TestTorchTensorMultiSubFastRTPS(unittest.TestCase):

    def test_received_tensors(self, proc_output, subscribers):
        for subscriber in subscribers:
            proc_output.assertWaitFor(
                'count=5)', process=subscriber, timeout=25)
            output = b''.join(event.text for event in proc_output[subscriber]).decode()
            samples = re.findall(
                r'Received tensor \(backend=(\w+), min=([^,]+), '
                r'max=([^,]+), count=\d+\)',
                output)
            self.assertGreaterEqual(len(samples), 5)
            for received_backend, minimum, maximum in samples:
                self.assertEqual(received_backend, 'cuda')
                self.assertEqual(float(minimum), float(maximum))
                self.assertGreaterEqual(float(minimum), 0)
                self.assertLessEqual(float(maximum), 255)


@launch_testing.post_shutdown_test()
class TestTorchTensorMultiSubFastRTPSShutdown(unittest.TestCase):
    """Test proper shutdown of nodes."""

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
