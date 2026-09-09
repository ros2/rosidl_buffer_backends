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

from math import prod

import rclpy
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor
import torch
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import set_stream


class TorchTensorPublisher(Node):

    def __init__(self):
        super().__init__('torch_tensor_publisher')
        self.declare_parameter('tensor_shape', [1])
        self.declare_parameter('tensor_values', [0])
        self._tensor_shape = tuple(
            self.get_parameter('tensor_shape').value)
        self._tensor_values = self.get_parameter('tensor_values').value
        if prod(self._tensor_shape) != len(self._tensor_values):
            raise ValueError(
                'tensor_values length must match the tensor shape')
        self._publisher = self.create_publisher(
            ExperimentalTensor, 'test_torch_tensor', 10)
        self._timer = self.create_timer(0.1, self._timer_callback)

    def _timer_callback(self):
        with set_stream():
            msg = allocate_tensor_msg(
                self._tensor_shape, torch.uint8, 'cuda')
            output = from_output_tensor_msg(msg)
            tensor = torch.tensor(
                self._tensor_values,
                dtype=torch.uint8,
                device=output.device,
            ).reshape_as(output)
            output.copy_(tensor)
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TorchTensorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
