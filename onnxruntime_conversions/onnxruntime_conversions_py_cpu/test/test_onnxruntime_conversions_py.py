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
import weakref

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort
from onnxruntime_conversions import _core
from onnxruntime_conversions import _dlpack_bridge
from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg
from onnxruntime_conversions import OrtTensorView
from onnxruntime_conversions import to_tensor_msg
from onnxruntime_conversions._adapter import OrtConversionRegistry
from onnxruntime_conversions._cpu_adapter import CpuOrtConversionAdapter
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


@pytest.mark.parametrize('element_type,numpy_type', SUPPORTED_TYPES)
def test_cpu_allocation_and_aliasing(element_type, numpy_type):
    msg = allocate_tensor_msg((2, 3), element_type)

    assert msg.shape == array('q', [2, 3])
    assert msg.strides == array('q', [3, 1])
    assert msg.byte_offset == 0
    assert len(msg.data) == 6 * np.dtype(numpy_type).itemsize

    view = from_output_tensor_msg(msg)
    assert isinstance(view, OrtTensorView)
    assert view.value.data_ptr() == np.frombuffer(
        msg.data, dtype=np.uint8).ctypes.data
    values = np.arange(6).astype(numpy_type).reshape(2, 3)
    view.value.update_inplace(values)
    view.close()

    assert np.array_equal(
        np.frombuffer(msg.data, dtype=numpy_type).reshape(2, 3), values)


def test_bfloat16_uses_onnx_type_factory_without_copy():
    msg = allocate_tensor_msg((4,), TensorProto.BFLOAT16)
    msg.data = array('B', np.arange(4, dtype=np.uint16).tobytes())

    with from_input_tensor_msg(msg) as value:
        assert value.element_type() == TensorProto.BFLOAT16
        assert value.data_ptr() == np.frombuffer(msg.data).ctypes.data


def test_scalar_empty_shape_and_zero_dimension():
    scalar = allocate_tensor_msg((), np.float32)
    empty = allocate_tensor_msg((2, 0, 3), np.float32)

    assert scalar.strides == array('q')
    assert len(scalar.data) == 4
    assert empty.strides == array('q', [0, 3, 1])
    assert len(empty.data) == 0
    assert from_input_tensor_msg(scalar).value.shape() == []
    assert from_input_tensor_msg(empty).value.shape() == [2, 0, 3]


def test_aligned_byte_offset_aliases_subview():
    msg = allocate_tensor_msg((2,), np.float32)
    msg.data = array('B', bytes(16))
    msg.byte_offset = 4

    with from_output_tensor_msg(msg) as value:
        value.update_inplace(np.array([3.0, 7.0], dtype=np.float32))
        assert value.data_ptr() == np.frombuffer(msg.data).ctypes.data + 4

    raw = np.frombuffer(msg.data, dtype=np.float32)
    assert np.array_equal(raw, [0.0, 3.0, 7.0, 0.0])


def test_context_manager_and_close_are_deterministic():
    msg = allocate_tensor_msg((2,), np.float32)
    view = from_input_tensor_msg(msg)

    with view as value:
        assert value is view.value
        assert not view.closed

    assert view.closed
    view.close()
    with pytest.raises(RuntimeError, match='closed'):
        view.value
    with pytest.raises(RuntimeError, match='closed'):
        view.__enter__()


def test_view_keeps_message_storage_alive():
    msg = allocate_tensor_msg((3,), np.int32)
    np.frombuffer(msg.data, dtype=np.int32)[:] = [4, 5, 6]
    view = from_input_tensor_msg(msg)
    del msg
    gc.collect()

    assert np.array_equal(view.value.numpy(), [4, 5, 6])


def test_platform_neutral_dlpack_bridge_retains_owner():
    class Producer:

        def __init__(self, capsule):
            self.capsule = capsule

        def __dlpack__(self, stream=None):
            capsule = self.capsule
            self.capsule = None
            return capsule

        def __dlpack_device__(self):
            return (1, 0)

    source = np.arange(4, dtype=np.float32)
    source_ref = weakref.ref(source)
    capsule = _dlpack_bridge.make_dlpack_capsule(
        source.ctypes.data, 1, 0, 2, 32, 1, source.shape, 0, source)
    value = ort.OrtValue.from_dlpack(Producer(capsule))
    del source
    gc.collect()

    assert source_ref() is not None
    assert np.array_equal(value.numpy(), [0.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    'mutate,match',
    [
        (lambda msg: setattr(msg, 'dtype_lanes', 2), 'dtype_lanes'),
        (lambda msg: setattr(msg, 'dtype_code', 5), 'dtype'),
        (lambda msg: setattr(msg, 'shape', [-1]), 'nonnegative'),
        (lambda msg: setattr(msg, 'strides', [1, 2]), 'contiguous'),
        (lambda msg: setattr(msg, 'byte_offset', 1), 'element-aligned'),
        (lambda msg: setattr(msg, 'byte_offset', 16), 'backing buffer'),
        (lambda msg: setattr(msg, 'data', array('B', bytes(3))), 'exceeds'),
    ],
)
def test_invalid_metadata_is_rejected(mutate, match):
    msg = allocate_tensor_msg((1,), np.float32)
    mutate(msg)

    with pytest.raises((ValueError, OverflowError), match=match):
        from_input_tensor_msg(msg)


def test_shape_overflow_is_rejected():
    msg = ExperimentalTensor(
        dtype_code=2,
        dtype_bits=32,
        dtype_lanes=1,
        shape=[2 ** 62, 4],
        strides=[],
        byte_offset=0,
        data=[],
    )

    with pytest.raises(OverflowError, match='element count'):
        from_input_tensor_msg(msg)


def test_cpu_rejects_stream_and_unknown_device():
    msg = allocate_tensor_msg((1,), np.float32)
    with pytest.raises(ValueError, match='does not accept a stream'):
        from_input_tensor_msg(msg, stream=1)
    with pytest.raises(ValueError, match='Unsupported tensor device'):
        allocate_tensor_msg((1,), np.float32, 'rocm')
    with pytest.raises(ValueError, match='device_id 0'):
        allocate_tensor_msg((1,), np.float32, 'cpu', 1)


def test_cpu_package_selects_cpu_by_default():
    msg = allocate_tensor_msg((1,), np.float32)
    assert _core._backend_type(msg.data) == 'cpu'


def test_registry_rejects_ambiguous_default():
    class OtherCpuAdapter(CpuOrtConversionAdapter):
        device_type = 'other'
        buffer_backend = 'other'

    registry = OrtConversionRegistry()
    registry.register(CpuOrtConversionAdapter())
    registry.register(OtherCpuAdapter())
    with pytest.raises(RuntimeError, match='Ambiguous default'):
        registry.default()


def test_explicit_cpu_overrides_platform_default(monkeypatch):
    monkeypatch.setattr(
        _core._registry,
        'default',
        lambda: pytest.fail('default adapter must not be selected'),
    )
    msg = allocate_tensor_msg(
        (1,), np.float32, device_type='cpu', stream=123)
    assert _core._backend_type(msg.data) == 'cpu'


def test_unregistered_device_raises():
    with pytest.raises(ValueError, match='Unsupported tensor device'):
        allocate_tensor_msg(
            (1,), np.float32, device_type='cuda', stream=123)


def test_cuda_default_does_not_fallback_after_allocation_error(monkeypatch):
    class FailingAdapter:

        def allocate(self, metadata, device_id, stream):
            raise RuntimeError('injected CUDA allocation failure')

    monkeypatch.setattr(
        _core._registry, 'default', lambda: FailingAdapter())
    with pytest.raises(RuntimeError, match='injected CUDA allocation failure'):
        allocate_tensor_msg((1,), np.float32, stream=123)


def test_to_tensor_msg_new_and_existing_destination():
    source_array = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = ort.OrtValue.ortvalue_from_numpy(source_array)

    created = to_tensor_msg(source)
    existing = allocate_tensor_msg((24,), np.uint8)
    returned = to_tensor_msg(existing, source)

    assert returned is existing
    assert created.shape == existing.shape == array('q', [2, 3])
    assert np.array_equal(
        np.frombuffer(created.data, dtype=np.float32).reshape(2, 3),
        source_array,
    )
    assert np.array_equal(
        np.frombuffer(existing.data, dtype=np.float32).reshape(2, 3),
        source_array,
    )


def test_to_tensor_msg_rejects_small_destination():
    source = ort.OrtValue.ortvalue_from_numpy(
        np.arange(4, dtype=np.float32))
    destination = allocate_tensor_msg((1,), np.float32)

    with pytest.raises(ValueError, match='destination buffer'):
        to_tensor_msg(destination, source)


def test_identity_inference_writes_directly_to_message():
    graph = helper.make_graph(
        [helper.make_node('Identity', ['input'], ['output'])],
        'identity',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 18)],
        ir_version=onnx.IR_VERSION,
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=['CPUExecutionProvider'])
    source = ort.OrtValue.ortvalue_from_numpy(
        np.arange(6, dtype=np.float32).reshape(2, 3))
    msg = allocate_tensor_msg((2, 3), np.float32)

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
