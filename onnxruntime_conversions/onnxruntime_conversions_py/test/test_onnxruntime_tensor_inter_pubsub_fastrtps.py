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

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from identity_model import identity_model

import numpy as np
import onnxruntime as ort

from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import available_backends
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg
from onnxruntime_conversions import session_providers

import pytest

import rclpy
from rclpy.node import Node
from rosidl_buffer import Buffer
from tensor_msgs.msg import ExperimentalTensor


@contextmanager
def _cuda_stream():
    runtime = ctypes.CDLL('libcudart.so')
    runtime.cudaStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    runtime.cudaStreamCreateWithFlags.restype = ctypes.c_int
    runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamDestroy.restype = ctypes.c_int
    stream = ctypes.c_void_p()
    result = runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1)
    if result != 0:
        raise RuntimeError(
            f'cudaStreamCreateWithFlags failed with error {result}')
    try:
        yield stream.value
    finally:
        if runtime.cudaStreamDestroy(stream) != 0:
            raise RuntimeError('cudaStreamDestroy failed')


def _cuda_unavailable_reason():
    if 'cuda' not in available_backends():
        return 'the CUDA conversion plugin is unavailable'
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        return 'this ONNX Runtime build has no CUDA execution provider'
    from cuda_buffer import CudaBuffer
    try:
        with _cuda_stream():
            probe = CudaBuffer.from_cpu(b'\x00')
            probe.to_bytes()
    except (OSError, RuntimeError) as error:
        return f'CUDA device is unavailable: {error}'
    return None


def _subscriber(topic, stream):
    session = ort.InferenceSession(
        identity_model((2, 3)),
        providers=session_providers('cuda', 0, stream),
    )
    node = Node('onnxruntime_cuda_tensor_subscriber')
    received = []

    def callback(msg):
        message_index = len(received)
        if not isinstance(msg.data, Buffer) or msg.data.backend_type != 'cuda':
            received.append(False)
            return

        output_msg = allocate_tensor_msg((2, 3), np.float32, 'cuda')
        input_view = from_input_tensor_msg(msg, stream)
        output_view = from_output_tensor_msg(output_msg, stream)
        binding = session.io_binding()
        try:
            binding.bind_ortvalue_input('input', input_view.value)
            binding.bind_ortvalue_output('output', output_view.value)
            session.run_with_iobinding(binding)
        finally:
            binding.clear_binding_inputs()
            binding.clear_binding_outputs()
            input_view.close()
            output_view.close()

        actual = np.frombuffer(
            output_msg.data.to_bytes(), dtype=np.float32).reshape(2, 3)
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        expected += message_index * 10
        received.append(np.array_equal(actual, expected))

    node.create_subscription(
        ExperimentalTensor, topic, callback, 10,
        acceptable_buffer_backends='cuda',
    )
    try:
        deadline = time.monotonic() + 12.0
        while len(received) < 5 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert len(received) >= 5, (
            f'subscriber validated only {len(received)} of 5 messages')
        assert all(received), 'one or more inference outputs were invalid'
    finally:
        node.destroy_node()
    print('SUBSCRIBER_CUDA_ONNX_OK')


def _publisher(topic, stream):
    node = Node('onnxruntime_cuda_tensor_publisher')
    publisher = node.create_publisher(ExperimentalTensor, topic, 10)
    try:
        deadline = time.monotonic() + 8.0
        while publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert publisher.get_subscription_count() == 1, (
            'publisher discovery timed out')
        time.sleep(1.0)

        for message_index in range(5):
            msg = allocate_tensor_msg((2, 3), np.float32, 'cuda')
            values = np.arange(6, dtype=np.float32).reshape(2, 3)
            values += message_index * 10
            output_view = from_output_tensor_msg(msg, stream)
            output_view.value.update_inplace(values)
            output_view.close()
            assert output_view.closed
            publisher.publish(msg)
            time.sleep(0.1)
    finally:
        node.destroy_node()
    print('PUBLISHER_CUDA_ONNX_OK')


def test_cuda_onnx_inference_crosses_fastrtps_process_boundary():
    unavailable_reason = _cuda_unavailable_reason()
    if unavailable_reason is not None:
        pytest.skip(unavailable_reason)
    topic = f'onnxruntime_tensor_{uuid.uuid4().hex}'
    environment = os.environ.copy()
    environment['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    environment['ROS_LOCALHOST_ONLY'] = '1'

    def start(role):
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), role, topic],
            env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)

    processes = []
    try:
        subscriber = start('subscriber')
        processes.append(subscriber)
        time.sleep(0.5)
        publisher = start('publisher')
        processes.append(publisher)
        publisher_output, _ = publisher.communicate(timeout=15)
        subscriber_output, _ = subscriber.communicate(timeout=15)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    assert publisher.returncode == 0, publisher_output
    assert subscriber.returncode == 0, subscriber_output
    assert 'PUBLISHER_CUDA_ONNX_OK' in publisher_output
    assert 'SUBSCRIBER_CUDA_ONNX_OK' in subscriber_output
    assert 'cudaEventSynchronize on the publish path' not in publisher_output


if __name__ == '__main__':
    worker = {'publisher': _publisher, 'subscriber': _subscriber}[sys.argv[1]]
    with _cuda_stream() as stream:
        rclpy.init(args=[])
        try:
            worker(sys.argv[2], stream)
        finally:
            rclpy.shutdown()
