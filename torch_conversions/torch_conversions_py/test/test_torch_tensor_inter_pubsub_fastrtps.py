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
from launch.actions import SetEnvironmentVariable
from launch.actions import TimerAction
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from rosidl_buffer import Buffer
from tensor_msgs.msg import ExperimentalTensor
import torch
import torch_conversions
from torch_conversions import from_input_tensor_msg
from torch_conversions import set_stream


CUDA_AVAILABLE = torch_conversions._adapter_available('cuda')
TENSOR_SHAPE = (2, 3, 4)
TENSOR_VALUES = (
    3, 17, 29, 43, 59, 71, 89, 101, 113, 127, 139, 149,
    163, 173, 181, 193, 199, 211, 223, 227, 233, 239, 241, 251,
)


def _expected_tensor(device):
    return torch.tensor(
        TENSOR_VALUES,
        dtype=torch.uint8,
        device=device,
    ).reshape(TENSOR_SHAPE)


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    if not CUDA_AVAILABLE:
        return LaunchDescription([
            launch_testing.actions.ReadyToTest(),
        ])

    publisher = Node(
        package='torch_conversions_py',
        executable='torch_tensor_publisher_node',
        output='screen',
        parameters=[{
            'tensor_shape': list(TENSOR_SHAPE),
            'tensor_values': list(TENSOR_VALUES),
        }],
    )

    return LaunchDescription([
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        TimerAction(period=2.0, actions=[
            publisher,
            launch_testing.actions.ReadyToTest(),
        ]),
    ])


@unittest.skipUnless(CUDA_AVAILABLE, 'CUDA support is unavailable')
class TestTorchTensorInterProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self._node = rclpy.create_node(
            'test_torch_tensor_py_inter_pubsub_fastrtps')
        self._received_count = 0
        self._validation_passed = True
        self._node.create_subscription(
            ExperimentalTensor,
            'test_torch_tensor',
            self._tensor_callback,
            10,
            acceptable_buffer_backends='any',
        )

    def tearDown(self):
        self._node.destroy_node()

    def _tensor_callback(self, msg):
        self._received_count += 1
        valid = (
            tuple(msg.shape) == TENSOR_SHAPE
            and msg.dtype_code == 1
            and msg.dtype_bits == 8
            and msg.dtype_lanes == 1
            and isinstance(msg.data, Buffer)
            and msg.data.backend_type == 'cuda'
            and len(msg.data) > 0
        )

        if valid:
            with set_stream():
                tensor = from_input_tensor_msg(msg, clone=False)
                valid = (
                    tensor is not None
                    and tensor.is_cuda
                    and tensor.dtype == torch.uint8
                    and torch.equal(
                        tensor,
                        _expected_tensor(tensor.device),
                    )
                )

        self._validation_passed = self._validation_passed and valid

    def test_inter_process_pubsub(self):
        deadline = time.monotonic() + 15.0
        while self._received_count < 5 and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)

        self.assertGreaterEqual(self._received_count, 5)
        self.assertTrue(self._validation_passed)


@launch_testing.post_shutdown_test()
@unittest.skipUnless(CUDA_AVAILABLE, 'CUDA support is unavailable')
class TestTorchTensorInterProcessShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, 1, -2, -6, -15],
        )
