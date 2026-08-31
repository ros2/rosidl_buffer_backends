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

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort
from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg
import pytest


pytestmark = pytest.mark.skipif(
    'CUDAExecutionProvider' not in ort.get_available_providers(),
    reason='ONNX Runtime CUDAExecutionProvider is unavailable',
)


@pytest.fixture
def cuda_buffer():
    return pytest.importorskip('cuda_buffer').CudaBuffer


def test_cuda_allocation_and_dlpack_pointer(cuda_buffer):
    msg = allocate_tensor_msg((2, 3), np.float32, 'cuda')
    assert msg.data.backend_type == 'cuda'

    stream = cuda_buffer.get_internal_stream()
    view = from_output_tensor_msg(msg, stream)
    assert view.value.device_name().lower() == 'cuda'
    assert view.value.shape() == [2, 3]
    assert view.value.element_type() == TensorProto.FLOAT
    assert view.value.data_ptr() != 0
    view.close()
    assert view.closed


def test_cuda_input_dlpack_alias_and_bool_compatibility(cuda_buffer):
    msg = allocate_tensor_msg((8,), TensorProto.BOOL, 'cuda')
    msg.data = cuda_buffer.from_cpu(bytes([0, 1] * 4))
    stream = cuda_buffer.get_internal_stream()

    with cuda_buffer.from_input_buffer(msg.data, stream) as handle:
        expected_pointer = handle.device_ptr
        expected_device = handle.device_id

    view = from_input_tensor_msg(msg, stream)
    assert view.value.data_ptr() == expected_pointer
    assert view.value.__dlpack_device__() == (2, expected_device)
    assert view.value.element_type() == TensorProto.BOOL
    view.close()


def test_cuda_identity_inference(cuda_buffer):
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
    stream = cuda_buffer.get_internal_stream()
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=[
            (
                'CUDAExecutionProvider',
                {'user_compute_stream': str(stream)},
            ),
            'CPUExecutionProvider',
        ],
    )
    input_msg = allocate_tensor_msg((4,), np.float32, 'cuda')
    input_msg.data = cuda_buffer.from_cpu(
        np.arange(4, dtype=np.float32).tobytes())
    output_msg = allocate_tensor_msg((4,), np.float32, 'cuda')

    input_view = from_input_tensor_msg(input_msg, stream)
    output_view = from_output_tensor_msg(output_msg, stream)
    binding = session.io_binding()
    binding.bind_ortvalue_input('input', input_view.value)
    binding.bind_ortvalue_output('output', output_view.value)
    session.run_with_iobinding(binding)
    binding.clear_binding_inputs()
    binding.clear_binding_outputs()
    input_view.close()
    output_view.close()

    assert np.frombuffer(output_msg.data.to_bytes(), dtype=np.float32).tolist() == [
        0.0, 1.0, 2.0, 3.0,
    ]
