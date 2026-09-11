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

from array import array
import gc

from identity_model import ElementType as TensorProto
from identity_model import identity_model

import numpy as np

import onnxruntime as ort

from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import available_backends
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg
from onnxruntime_conversions import OrtTensorView
from onnxruntime_conversions import session_providers
from onnxruntime_conversions import to_tensor_msg

import pytest

from tensor_msgs.msg import ExperimentalTensor


SUPPORTED_TYPES = [
    (TensorProto.FLOAT, np.float32),
    (TensorProto.UINT8, np.uint8),
    (TensorProto.INT8, np.int8),
    (TensorProto.UINT16, np.uint16),
    (TensorProto.INT16, np.int16),
    (TensorProto.INT32, np.int32),
    (TensorProto.INT64, np.int64),
    (TensorProto.BOOL, np.bool_),
    (TensorProto.FLOAT16, np.float16),
    (TensorProto.DOUBLE, np.float64),
    (TensorProto.UINT32, np.uint32),
    (TensorProto.UINT64, np.uint64),
]


def identity_session():
    return ort.InferenceSession(
        identity_model((2, 3)), providers=session_providers('cpu'))


@pytest.mark.parametrize('element_type,numpy_type', SUPPORTED_TYPES)
def test_host_allocation_and_aliasing(element_type, numpy_type):
    msg = allocate_tensor_msg((2, 3), element_type, 'cpu')

    assert msg.shape == array('q', [2, 3])
    assert msg.strides == array('q', [3, 1])
    assert msg.byte_offset == 0
    assert len(msg.data) == 6 * np.dtype(numpy_type).itemsize

    view = from_output_tensor_msg(msg)
    assert isinstance(view, OrtTensorView)
    assert view.value.element_type() == element_type
    assert view.value.shape() == [2, 3]
    values = np.arange(6).astype(numpy_type).reshape(2, 3)
    view.value.update_inplace(values)
    view.close()

    assert np.array_equal(
        np.frombuffer(msg.data, dtype=numpy_type).reshape(2, 3), values)


@pytest.mark.parametrize('element_type,numpy_type', SUPPORTED_TYPES)
def test_numpy_dtypes_name_the_same_element_types(element_type, numpy_type):
    msg = allocate_tensor_msg((4,), numpy_type, 'cpu')

    with from_input_tensor_msg(msg) as value:
        assert value.element_type() == element_type


def test_bfloat16_survives_the_round_trip():
    msg = allocate_tensor_msg((4,), TensorProto.BFLOAT16, 'cpu')
    np.frombuffer(msg.data, dtype=np.uint16)[:] = np.arange(4)

    with from_input_tensor_msg(msg) as value:
        assert value.element_type() == TensorProto.BFLOAT16
        assert value.shape() == [4]


def test_input_and_output_views_alias_the_same_storage():
    msg = allocate_tensor_msg((6,), np.float32, 'cpu')
    base = np.frombuffer(msg.data, dtype=np.float32)

    with from_output_tensor_msg(msg) as value:
        value.update_inplace(np.arange(6, dtype=np.float32))
    assert np.array_equal(base, np.arange(6, dtype=np.float32))

    with from_input_tensor_msg(msg) as value:
        assert np.array_equal(value.numpy(), base)


def test_scalar_and_zero_element_shapes():
    scalar = allocate_tensor_msg((), np.float32, 'cpu')
    empty = allocate_tensor_msg((2, 0, 3), np.float32, 'cpu')

    assert len(scalar.data) == 4
    assert from_input_tensor_msg(scalar).value.shape() == []
    assert len(empty.data) == 0
    assert from_input_tensor_msg(empty) is None
    assert from_output_tensor_msg(empty) is None


def test_byte_offset_aliases_a_subview():
    msg = allocate_tensor_msg((4,), np.float32, 'cpu')
    msg.shape = [2]
    msg.strides = [1]
    msg.byte_offset = 4

    with from_output_tensor_msg(msg) as value:
        value.update_inplace(np.array([3.0, 7.0], dtype=np.float32))

    assert np.array_equal(
        np.frombuffer(msg.data, dtype=np.float32), [0.0, 3.0, 7.0, 0.0])


def test_context_manager_and_close_are_deterministic():
    msg = allocate_tensor_msg((2,), np.float32, 'cpu')
    view = from_input_tensor_msg(msg)

    with view as value:
        assert value is view.value
        assert not view.closed

    assert view.closed
    view.close()
    with pytest.raises(RuntimeError, match='closed'):
        view.value


def test_view_keeps_message_storage_alive():
    msg = allocate_tensor_msg((3,), np.int32, 'cpu')
    np.frombuffer(msg.data, dtype=np.int32)[:] = [4, 5, 6]
    view = from_input_tensor_msg(msg)
    del msg
    gc.collect()

    assert np.array_equal(view.value.numpy(), [4, 5, 6])


@pytest.mark.parametrize(
    'mutate,error,match',
    [
        (lambda msg: setattr(msg, 'dtype_lanes', 2), ValueError,
         'dtype_lanes'),
        (lambda msg: setattr(msg, 'dtype_code', 5), ValueError,
         'unsupported by ONNX Runtime'),
        (lambda msg: setattr(msg, 'shape', [-1]), (ValueError, OverflowError),
         'nonnegative'),
        (lambda msg: setattr(msg, 'strides', [1, 2]), ValueError,
         'contiguous'),
        (lambda msg: setattr(msg, 'data', array('B', bytes(3))), ValueError,
         'buffer has 3'),
    ],
)
def test_invalid_metadata_is_rejected(mutate, error, match):
    # Wide enough that the mutations below, and not the core bounds check,
    # are what each case reports.
    msg = allocate_tensor_msg((2,), np.float32, 'cpu')
    msg.shape = [1]
    msg.strides = [1]
    mutate(msg)

    with pytest.raises(error, match=match):
        from_input_tensor_msg(msg)


@pytest.mark.parametrize('element_type', [8, np.complex64])
def test_unsupported_element_types_are_rejected(element_type):
    with pytest.raises(ValueError, match='Unsupported ONNX tensor element'):
        allocate_tensor_msg((1,), element_type, 'cpu')


def test_host_storage_is_always_available():
    assert 'cpu' in available_backends()


def test_to_tensor_msg_new_and_existing_destination():
    source_array = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = ort.OrtValue.ortvalue_from_numpy(source_array)

    created = to_tensor_msg(source)
    existing = allocate_tensor_msg((24,), np.uint8, 'cpu')
    returned = to_tensor_msg(existing, source)

    assert returned is existing
    assert created.shape == existing.shape == array('q', [2, 3])
    for msg in (created, existing):
        assert np.array_equal(
            np.frombuffer(msg.data, dtype=np.float32).reshape(2, 3),
            source_array,
        )


def test_to_tensor_msg_rejects_a_small_destination():
    source = ort.OrtValue.ortvalue_from_numpy(np.arange(4, dtype=np.float32))
    destination = allocate_tensor_msg((1,), np.float32, 'cpu')

    with pytest.raises(ValueError, match='destination buffer'):
        to_tensor_msg(destination, source)


def test_to_tensor_msg_rejects_non_tensor_arguments():
    with pytest.raises(TypeError, match='tensor OrtValue'):
        to_tensor_msg(np.arange(4, dtype=np.float32))
    with pytest.raises(TypeError, match='ExperimentalTensor'):
        to_tensor_msg(
            'not a message',
            ort.OrtValue.ortvalue_from_numpy(np.zeros(1, dtype=np.float32)),
        )


def test_shape_overflow_is_rejected():
    msg = ExperimentalTensor(
        dtype_code=2,
        dtype_bits=32,
        dtype_lanes=1,
        shape=[2 ** 62, 4],
        strides=[],
        byte_offset=0,
        data=array('B', bytes(16)),
    )

    with pytest.raises(ValueError, match='buffer has 16'):
        from_input_tensor_msg(msg)


def test_session_providers_for_host_memory():
    assert session_providers('cpu') == ['CPUExecutionProvider']
    with pytest.raises(ValueError, match='no execution stream'):
        session_providers('cpu', stream=1)
    with pytest.raises(ValueError, match='requires an explicit execution'):
        session_providers('cuda')
    with pytest.raises(ValueError, match='No ONNX Runtime execution provider'):
        session_providers('vulkan', stream=1)


def test_identity_inference_writes_directly_to_the_message():
    session = identity_session()
    source = ort.OrtValue.ortvalue_from_numpy(
        np.arange(6, dtype=np.float32).reshape(2, 3))
    msg = allocate_tensor_msg((2, 3), np.float32, 'cpu')

    binding = session.io_binding()
    binding.bind_ortvalue_input('input', source)
    view = from_output_tensor_msg(msg)
    binding.bind_ortvalue_output('output', view.value)
    session.run_with_iobinding(binding)
    binding.clear_binding_outputs()
    view.close()

    assert np.array_equal(
        np.frombuffer(msg.data, dtype=np.float32).reshape(2, 3),
        source.numpy(),
    )


def test_identity_inference_reads_directly_from_the_message():
    session = identity_session()
    msg = allocate_tensor_msg((2, 3), np.float32, 'cpu')
    np.frombuffer(msg.data, dtype=np.float32)[:] = np.arange(6)

    with from_input_tensor_msg(msg) as value:
        binding = session.io_binding()
        binding.bind_ortvalue_input('input', value)
        binding.bind_output('output', 'cpu')
        session.run_with_iobinding(binding)
        result = binding.get_outputs()[0].numpy()

    assert np.array_equal(result, np.arange(6, dtype=np.float32).reshape(2, 3))
