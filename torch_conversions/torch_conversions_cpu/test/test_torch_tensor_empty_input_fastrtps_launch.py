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

from array import array
import time
import unittest

from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from tensor_msgs.msg import ExperimentalTensor


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """Exercise empty-message recovery in the actual C++ subscriber example."""
    subscriber = Node(
        package='torch_conversions',
        executable='torch_tensor_subscriber_node',
        output='screen',
    )
    return LaunchDescription([
        subscriber,
        launch_testing.actions.ReadyToTest(),
    ]), {'subscriber': subscriber}


class TestEmptyTensorRecovery(unittest.TestCase):

    def test_empty_then_valid(self, proc_output, subscriber):
        rclpy.init()
        node = rclpy.create_node('test_empty_tensor_recovery')
        try:
            publisher = node.create_publisher(ExperimentalTensor, 'test_torch_tensor', 10)
            deadline = time.monotonic() + 10
            while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            self.assertGreater(publisher.get_subscription_count(), 0)

            # No storage, as returned by a failed CUDA IPC import.
            publisher.publish(ExperimentalTensor())
            proc_output.assertWaitFor(
                'Skipping tensor with empty or unavailable storage',
                process=subscriber, timeout=10)

            # A valid tensor view can also have zero elements despite nonempty storage.
            publisher.publish(ExperimentalTensor(
                dtype_code=1, dtype_bits=8, dtype_lanes=1,
                shape=[0], data=array('B', [7])))
            deadline = time.monotonic() + 5
            warning = 'Skipping tensor with empty or unavailable storage'
            while time.monotonic() < deadline:
                output = b''.join(event.text for event in proc_output[subscriber]).decode()
                if output.count(warning) >= 2:
                    break
                time.sleep(0.05)
            self.assertGreaterEqual(output.count(warning), 2)

            publisher.publish(ExperimentalTensor(
                dtype_code=1, dtype_bits=8, dtype_lanes=1,
                shape=[32], data=array('B', [7] * 32)))
            proc_output.assertWaitFor(
                'Received tensor (backend=cpu, min=7, max=7, count=1)',
                process=subscriber, timeout=10)
        finally:
            node.destroy_node()
            rclpy.shutdown()


@launch_testing.post_shutdown_test()
class TestEmptyTensorRecoveryShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0])
