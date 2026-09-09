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

import numpy as np

import onnx
from onnx import helper
from onnx import TensorProto

import onnxruntime as ort

from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import available_backends
from onnxruntime_conversions import default_backend
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg
from onnxruntime_conversions import session_providers

import pytest


pytestmark = pytest.mark.skipif(
    'cuda' not in available_backends(),
    reason='the CUDA storage plugin is unavailable',
)


@pytest.fixture
def cuda_buffer():
    return pytest.importorskip('cuda_buffer').CudaBuffer


@pytest.fixture
def cuda_stream():
    runtime = ctypes.CDLL('libcudart.so')
    runtime.cudaStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    runtime.cudaStreamCreateWithFlags.restype = ctypes.c_int
    runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamDestroy.restype = ctypes.c_int
    stream = ctypes.c_void_p()
    result = runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1)
    if result != 0:
        pytest.skip(f'cudaStreamCreateWithFlags failed with error {result}')
    try:
        yield stream.value
    finally:
        assert runtime.cudaStreamDestroy(stream) == 0


def test_cuda_is_preferred_over_host_memory():
    assert default_backend() == 'cuda'
    assert allocate_tensor_msg((4,), np.float32).data.backend_type == 'cuda'


def test_host_storage_stays_available_alongside_cuda():
    msg = allocate_tensor_msg((4,), np.float32, 'cpu')

    assert not hasattr(msg.data, 'backend_type')


def test_views_report_cuda_device_memory(cuda_stream):
    msg = allocate_tensor_msg((2, 3), np.float32, 'cuda')

    with from_output_tensor_msg(msg, cuda_stream) as value:
        assert value.device_name().lower() == 'cuda'
        assert value.shape() == [2, 3]
        assert value.element_type() == TensorProto.FLOAT
        assert value.data_ptr() != 0


def test_views_alias_the_device_pointer(cuda_buffer, cuda_stream):
    msg = allocate_tensor_msg((8,), TensorProto.BOOL, 'cuda')
    msg.data = cuda_buffer.from_cpu(bytes([0, 1] * 4))

    with cuda_buffer.from_input_buffer(msg.data, cuda_stream) as handle:
        expected_pointer = handle.device_ptr

    with from_input_tensor_msg(msg, cuda_stream) as value:
        assert value.data_ptr() == expected_pointer
        assert value.element_type() == TensorProto.BOOL
        ortvalue = getattr(value, '_ortvalue', value)
        assert ortvalue.__dlpack_device__() == (2, 0)


def test_input_and_output_views_alias_the_same_storage(cuda_stream):
    msg = allocate_tensor_msg((6,), np.float32, 'cuda')

    with from_output_tensor_msg(msg, cuda_stream) as output:
        with from_input_tensor_msg(msg, cuda_stream) as source:
            assert source.data_ptr() == output.data_ptr()


def test_views_describe_device_data_without_copying_it(
    cuda_buffer, cuda_stream,
):
    source = np.arange(4, dtype=np.float32)
    msg = allocate_tensor_msg((4,), np.float32, 'cuda')
    msg.data = cuda_buffer.from_cpu(source.tobytes())

    with from_input_tensor_msg(msg, cuda_stream) as value:
        assert value.device_name().lower() == 'cuda'
        # Reading device memory needs a session to stage the copy, so the
        # data itself is checked through the buffer.
        with pytest.raises(RuntimeError, match='non-CPU tensor'):
            value.numpy()

    assert np.array_equal(
        np.frombuffer(msg.data.to_bytes(), dtype=np.float32), source)


def test_the_default_stream_is_accepted(cuda_stream):
    del cuda_stream
    msg = allocate_tensor_msg((1,), np.float32, 'cuda')

    for conversion in (from_input_tensor_msg, from_output_tensor_msg):
        with conversion(msg, 0) as value:
            assert value.device_name().lower() == 'cuda'


def test_session_providers_binds_the_compute_stream(cuda_stream):
    providers = session_providers('cuda', 0, cuda_stream)

    assert providers[0][0] == 'CUDAExecutionProvider'
    assert providers[0][1]['user_compute_stream'] == str(cuda_stream)
    assert providers[0][1]['device_id'] == '0'
    assert providers[-1] == 'CPUExecutionProvider'


def test_identity_inference_on_device_storage(cuda_buffer, cuda_stream):
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        pytest.skip('this ONNX Runtime build has no CUDA execution provider')

    graph = helper.make_graph(
        [helper.make_node('Identity', ['input'], ['output'])],
        'identity',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 18)],
        ir_version=onnx.IR_VERSION,
    )
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=session_providers('cuda', 0, cuda_stream),
    )
    input_msg = allocate_tensor_msg((4,), np.float32, 'cuda')
    input_msg.data = cuda_buffer.from_cpu(
        np.arange(4, dtype=np.float32).tobytes())
    output_msg = allocate_tensor_msg((4,), np.float32, 'cuda')

    input_view = from_input_tensor_msg(input_msg, cuda_stream)
    output_view = from_output_tensor_msg(output_msg, cuda_stream)
    binding = session.io_binding()
    binding.bind_ortvalue_input('input', input_view.value)
    binding.bind_ortvalue_output('output', output_view.value)
    session.run_with_iobinding(binding)
    binding.clear_binding_inputs()
    binding.clear_binding_outputs()
    input_view.close()
    output_view.close()

    assert np.frombuffer(
        output_msg.data.to_bytes(), dtype=np.float32).tolist() == [
        0.0, 1.0, 2.0, 3.0,
    ]
