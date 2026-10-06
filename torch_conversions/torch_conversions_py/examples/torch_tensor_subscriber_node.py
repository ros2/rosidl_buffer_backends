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

"""Read tensor messages as PyTorch views and log their value range."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor
from torch_conversions import from_input_tensor_msg
from torch_conversions import set_stream


class TorchTensorSubscriber(Node):
    """Example subscriber for CPU and accelerator-backed tensor messages."""

    def __init__(self):
        """Subscribe to the example publisher's tensor topic."""
        super().__init__('torch_tensor_subscriber')
        self._received_count = 0
        self._subscription = self.create_subscription(
            ExperimentalTensor, 'test_torch_tensor', self._receive_tensor, 10,
            acceptable_buffer_backends='any')

    def _receive_tensor(self, received):
        # Select a stream for this callback (a no-op on CPU).
        with set_stream():
            # View the message storage directly; treat it as read-only.
            input_tensor = from_input_tensor_msg(received, clone=False)

            # Replace these reductions with your own processing of the tensor.
            minimum = input_tensor.min().item()
            maximum = input_tensor.max().item()
            backend = getattr(received.data, 'backend_type', 'cpu')
            self._received_count += 1
            self.get_logger().info(
                f'Received tensor (backend={backend}, min={minimum:g}, '
                f'max={maximum:g}, count={self._received_count})')


def main(args=None):
    """Run the subscriber until ROS shuts down."""
    rclpy.init(args=args)
    node = TorchTensorSubscriber()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
