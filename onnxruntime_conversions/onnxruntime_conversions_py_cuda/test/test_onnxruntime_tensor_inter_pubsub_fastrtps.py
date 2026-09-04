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
import subprocess
import sys
import textwrap
import time
import uuid

import onnxruntime as ort
import pytest


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


def _with_cuda_stream(source):
    setup = """
import ctypes

runtime = ctypes.CDLL('libcudart.so')
runtime.cudaStreamCreateWithFlags.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
runtime.cudaStreamCreateWithFlags.restype = ctypes.c_int
runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
runtime.cudaStreamDestroy.restype = ctypes.c_int
cuda_stream = ctypes.c_void_p()
result = runtime.cudaStreamCreateWithFlags(ctypes.byref(cuda_stream), 1)
if result != 0:
    raise RuntimeError(
        f'cudaStreamCreateWithFlags failed with error {result}')
stream = cuda_stream.value
"""
    cleanup = """
finally:
    if runtime.cudaStreamDestroy(cuda_stream) != 0:
        raise RuntimeError('cudaStreamDestroy failed')
"""
    return textwrap.dedent(setup) + '\ntry:\n' + textwrap.indent(
        source, '    ') + textwrap.dedent(cleanup)


def _cuda_unavailable_reason():
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        return 'ONNX Runtime CUDAExecutionProvider is unavailable'
    try:
        from cuda_buffer import CudaBuffer
    except (ImportError, OSError) as error:
        return f'cuda_buffer is unavailable: {error}'
    try:
        with _cuda_stream():
            probe = CudaBuffer.from_cpu(b'\x00')
            probe.to_bytes()
    except (OSError, RuntimeError) as error:
        return f'CUDA device is unavailable: {error}'
    return None


def test_cuda_onnx_inference_crosses_fastrtps_process_boundary():
    unavailable_reason = _cuda_unavailable_reason()
    if unavailable_reason is not None:
        pytest.skip(unavailable_reason)
    topic = f'onnxruntime_tensor_{uuid.uuid4().hex}'
    subscriber_source = _with_cuda_stream(textwrap.dedent(f"""
        import time

        import numpy as np
        import onnx
        from onnx import helper
        from onnx import TensorProto
        import onnxruntime as ort
        import rclpy
        from rclpy.node import Node
        from rosidl_buffer import Buffer
        from onnxruntime_conversions import allocate_tensor_msg
        from onnxruntime_conversions import from_input_tensor_msg
        from onnxruntime_conversions import from_output_tensor_msg
        from tensor_msgs.msg import ExperimentalTensor

        graph = helper.make_graph(
            [helper.make_node('Identity', ['input'], ['output'])],
            'identity',
            [helper.make_tensor_value_info(
                'input', TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info(
                'output', TensorProto.FLOAT, [2, 3])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid('', 18)],
            ir_version=onnx.IR_VERSION,
        )
        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=[
                (
                    'CUDAExecutionProvider',
                    {{'user_compute_stream': str(stream)}},
                ),
                'CPUExecutionProvider',
            ],
        )
        rclpy.init()
        node = Node('onnxruntime_cuda_tensor_subscriber')
        received = []

        def callback(msg):
            validation_index = len(received) + 1
            if not isinstance(msg.data, Buffer):
                received.append(False)
                print(f'VALIDATION_{{validation_index}}_FAIL_NOT_BUFFER')
                return
            if msg.data.backend_type != 'cuda':
                received.append(False)
                print(
                    f'VALIDATION_{{validation_index}}_FAIL_BACKEND_'
                    f'{{msg.data.backend_type}}')
                return

            output_msg = allocate_tensor_msg(
                (2, 3), np.float32, stream=stream)
            input_view = from_input_tensor_msg(msg, stream=stream)
            output_view = from_output_tensor_msg(output_msg, stream=stream)
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
            expected += (validation_index - 1) * 10
            valid = np.array_equal(actual, expected)
            received.append(valid)
            result = 'PASS' if valid else 'FAIL_PAYLOAD'
            print(f'VALIDATION_{{validation_index}}_{{result}}')

        subscription = node.create_subscription(
            ExperimentalTensor,
            '{topic}',
            callback,
            10,
            acceptable_buffer_backends='cuda',
        )
        deadline = time.monotonic() + 12.0
        while len(received) < 5 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert len(received) >= 5, (
            f'subscriber validated only {{len(received)}} of 5 messages')
        assert all(received), 'one or more inference outputs were invalid'
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
        del session
        print('SUBSCRIBER_CUDA_ONNX_OK')
    """))
    publisher_source = _with_cuda_stream(textwrap.dedent(f"""
        import time

        import numpy as np
        import rclpy
        from rclpy.node import Node
        from onnxruntime_conversions import allocate_tensor_msg
        from onnxruntime_conversions import from_output_tensor_msg
        from tensor_msgs.msg import ExperimentalTensor

        rclpy.init()
        node = Node('onnxruntime_cuda_tensor_publisher')
        publisher = node.create_publisher(ExperimentalTensor, '{topic}', 10)
        deadline = time.monotonic() + 8.0
        while publisher.get_subscription_count() < 1 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert publisher.get_subscription_count() == 1, (
            'publisher discovery timed out')
        time.sleep(1.0)

        for message_index in range(5):
            msg = allocate_tensor_msg(
                (2, 3), np.float32, stream=stream)
            values = np.arange(6, dtype=np.float32).reshape(2, 3)
            values += message_index * 10
            output_view = from_output_tensor_msg(msg, stream=stream)
            output_view.value.update_inplace(values)
            output_view.close()
            assert output_view.closed
            publisher.publish(msg)
            print(f'PUBLISHED_{{message_index + 1}}_CLOSED')
            time.sleep(0.1)

        node.destroy_publisher(publisher)
        node.destroy_node()
        rclpy.shutdown()
        print('PUBLISHER_CUDA_ONNX_OK')
    """))
    environment = os.environ.copy()
    environment['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
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
    assert 'PUBLISHER_CUDA_ONNX_OK' in publisher_output
    assert 'SUBSCRIBER_CUDA_ONNX_OK' in subscriber_output
    for validation_index in range(1, 6):
        assert f'PUBLISHED_{validation_index}_CLOSED' in publisher_output
        assert f'VALIDATION_{validation_index}_PASS' in subscriber_output
    assert 'cudaEventSynchronize on the publish path' not in publisher_output
