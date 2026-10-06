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

"""
Run an ONNX model on received tensor views and log its output range.

Set the model_path ROS parameter to a model with an input named 'input' and an
output named 'output'. Its input shape and dtype must match the incoming tensors.
"""

import onnxruntime as ort
from onnxruntime_conversions import create_stream
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import session_providers
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tensor_msgs.msg import ExperimentalTensor


class OrtTensorSubscriber(Node):
    """Example subscriber that binds received tensor storage to model input."""

    def __init__(self):
        """Load the model and subscribe using the selected conversion backend."""
        super().__init__('onnxruntime_tensor_subscriber')
        model_path = self.declare_parameter('model_path', '').value
        if not model_path:
            raise ValueError('Set the model_path ROS parameter to an ONNX model file')
        backend = self.declare_parameter('backend', '').value or None
        # Keep the stream alive for the session and all message views.
        self._stream = create_stream(backend)
        self._session = ort.InferenceSession(
            model_path, providers=session_providers(stream=self._stream))
        self._received_count = 0
        self._subscription = self.create_subscription(
            ExperimentalTensor, 'test_onnxruntime_tensor', self._receive_tensor, 10,
            acceptable_buffer_backends='any')

    def _receive_tensor(self, received):
        # Scope the read-only message view until inference finishes.
        with from_input_tensor_msg(received, self._stream) as input_value:
            binding = self._session.io_binding()
            try:
                binding.bind_ortvalue_input('input', input_value)
                # Return the inference output on CPU so it can be read below.
                binding.bind_output('output', 'cpu')
                self._session.run_with_iobinding(binding)
                output = binding.copy_outputs_to_cpu()[0]
            finally:
                binding.clear_binding_inputs()

        # Replace this summary with your own processing of the model output.
        backend = getattr(received.data, 'backend_type', 'cpu')
        self._received_count += 1
        self.get_logger().info(
            f'Received tensor (backend={backend}, min={output.min():g}, '
            f'max={output.max():g}, count={self._received_count})')


def main(args=None):
    """Run the subscriber until ROS shuts down."""
    rclpy.init(args=args)
    node = OrtTensorSubscriber()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
