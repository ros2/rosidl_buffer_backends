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

import ctypes
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


def _cuda_available():
    try:
        runtime = ctypes.CDLL('libcudart.so')
    except OSError:
        return False
    count = ctypes.c_int()
    return runtime.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0


CUDA_AVAILABLE = _cuda_available()


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    if not CUDA_AVAILABLE:
        return LaunchDescription([launch_testing.actions.ReadyToTest()])

    subscriber = Node(
        package='onnxruntime_conversions_cuda_plugin',
        executable='onnxruntime_cuda_tensor_subscriber_node',
        output='screen',
    )
    publisher = Node(
        package='onnxruntime_conversions_cuda_plugin',
        executable='onnxruntime_cuda_tensor_publisher_node',
        output='screen',
    )
    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        subscriber,
        TimerAction(period=2.0, actions=[
            publisher,
            launch_testing.actions.ReadyToTest(),
        ]),
    ])


@unittest.skipUnless(CUDA_AVAILABLE, 'CUDA device is unavailable')
class TestCudaTensorInterProcessInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_cuda_tensor_inter_process_inference')
        self.publisher_count = 0
        self.subscriber_count = 0
        self.validation_passed = True
        self.node.create_subscription(
            UInt32, 'publisher_count', self._publisher_count, 10)
        self.node.create_subscription(
            UInt32, 'subscriber_count', self._subscriber_count, 10)
        self.node.create_subscription(
            Bool, 'validation_result', self._validation_result, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _publisher_count(self, message):
        self.publisher_count = message.data

    def _subscriber_count(self, message):
        self.subscriber_count = message.data

    def _validation_result(self, message):
        self.validation_passed = message.data

    def test_received_cuda_inference_output(self):
        deadline = time.time() + 20.0
        while self.subscriber_count < 5 and time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertGreaterEqual(self.publisher_count, 5)
        self.assertGreaterEqual(self.subscriber_count, 5)
        self.assertTrue(self.validation_passed)


@launch_testing.post_shutdown_test()
class TestCudaTensorInterProcessShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, 1, -2, -6, -15])
