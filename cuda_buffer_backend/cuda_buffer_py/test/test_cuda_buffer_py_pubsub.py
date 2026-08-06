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
import subprocess
import sys
import textwrap
import time
import uuid


def test_multiprocess_rclpy_cuda_buffer_transport():
    """Validate five Python-created CUDA buffers between rclpy processes."""
    topic = f'cuda_buffer_py_{uuid.uuid4().hex}'
    subscriber_source = textwrap.dedent(f"""
        import time

        from cuda_buffer import CudaBuffer
        import rclpy
        from rclpy.node import Node
        from rosidl_buffer import Buffer
        from sensor_msgs.msg import Image

        EXPECTED_PAYLOAD = bytes((index * 13 + 7) % 256 for index in range(64))

        rclpy.init()
        node = Node('cuda_buffer_py_multiprocess_sub')
        received = []

        def callback(msg):
            if not isinstance(msg.data, Buffer):
                received.append(False)
                print(f'VALIDATION_{{len(received)}}_FAIL_NOT_BUFFER')
                return
            if msg.data.backend_type != 'cuda':
                received.append(False)
                print(f'VALIDATION_{{len(received)}}_FAIL_BACKEND_{{msg.data.backend_type}}')
                return

            with CudaBuffer.from_input_buffer(msg.data) as handle:
                assert handle.device_ptr != 0

            valid = msg.data.to_bytes() == EXPECTED_PAYLOAD
            received.append(valid)
            result = 'PASS' if valid else 'FAIL_PAYLOAD'
            print(f'VALIDATION_{{len(received)}}_{{result}}')

        subscription = node.create_subscription(
            Image,
            '{topic}',
            callback,
            10,
            acceptable_buffer_backends='cuda',
        )
        deadline = time.monotonic() + 12.0
        while len(received) < 5 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert len(received) >= 5, (
            f'subscriber validated only {{len(received)}} of 5 messages'
        )
        assert all(received), 'one or more CUDA payload validations failed'
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
        print('SUBSCRIBER_CUDA_OK')
    """)
    publisher_source = textwrap.dedent(f"""
        import time

        from cuda_buffer import CudaBuffer
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        EXPECTED_PAYLOAD = bytes((index * 13 + 7) % 256 for index in range(64))

        rclpy.init()
        node = Node('cuda_buffer_py_multiprocess_pub')
        publisher = node.create_publisher(Image, '{topic}', 10)
        deadline = time.monotonic() + 8.0
        while publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert publisher.get_subscription_count() == 1, 'publisher discovery timed out'
        time.sleep(1.0)

        msg = Image()
        msg.height = 1
        msg.width = 64
        msg.encoding = '8UC1'
        msg.step = 64
        msg.data = CudaBuffer.from_cpu(EXPECTED_PAYLOAD)

        for _ in range(5):
            publisher.publish(msg)
            time.sleep(0.1)

        node.destroy_publisher(publisher)
        node.destroy_node()
        rclpy.shutdown()
        print('PUBLISHER_CUDA_OK')
    """)

    environment = os.environ.copy()
    environment['ROS_LOCALHOST_ONLY'] = '1'
    subscriber = subprocess.Popen(
        [sys.executable, '-c', subscriber_source],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(0.5)
    publisher = subprocess.Popen(
        [sys.executable, '-c', publisher_source],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        publisher_output, _ = publisher.communicate(timeout=15)
        subscriber_output, _ = subscriber.communicate(timeout=15)
    finally:
        if publisher.poll() is None:
            publisher.terminate()
            publisher.wait(timeout=5)
        if subscriber.poll() is None:
            subscriber.terminate()
            subscriber.wait(timeout=5)

    assert publisher.returncode == 0, publisher_output
    assert subscriber.returncode == 0, subscriber_output
    assert 'PUBLISHER_CUDA_OK' in publisher_output
    assert 'SUBSCRIBER_CUDA_OK' in subscriber_output
    for validation_index in range(1, 6):
        assert f'VALIDATION_{validation_index}_PASS' in subscriber_output
    assert 'cudaEventSynchronize on the publish path' not in publisher_output
