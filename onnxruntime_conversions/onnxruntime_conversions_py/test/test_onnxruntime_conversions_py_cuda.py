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
import gc
import time

from identity_model import ElementType
from identity_model import identity_model
from identity_model import matmul_model

import numpy as np

import onnxruntime as ort

from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import available_backends
from onnxruntime_conversions import borrow_stream
from onnxruntime_conversions import create_stream
from onnxruntime_conversions import default_backend
from onnxruntime_conversions import from_input_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg
from onnxruntime_conversions import session_providers
from onnxruntime_conversions import to_tensor_msg

import pytest


pytestmark = [
    pytest.mark.skipif(
        'cuda' not in available_backends(),
        reason='the CUDA conversion plugin is unavailable',
    ),
    pytest.mark.skipif(
        'CUDAExecutionProvider' not in ort.get_available_providers(),
        reason='this onnxruntime build has no CUDA execution provider',
    ),
]


@pytest.fixture
def cuda_buffer():
    return pytest.importorskip('cuda_buffer').CudaBuffer


@pytest.fixture
def cuda_runtime():
    runtime = ctypes.CDLL('libcudart.so')
    runtime.cudaStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    runtime.cudaStreamCreateWithFlags.restype = ctypes.c_int
    runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamDestroy.restype = ctypes.c_int
    runtime.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamSynchronize.restype = ctypes.c_int
    runtime.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_void_p]
    runtime.cudaMemcpyAsync.restype = ctypes.c_int
    runtime.cudaLaunchHostFunc.argtypes = [
        ctypes.c_void_p, ctypes.CFUNCTYPE(None, ctypes.c_void_p), ctypes.c_void_p]
    runtime.cudaLaunchHostFunc.restype = ctypes.c_int
    runtime.cudaStreamQuery.argtypes = [ctypes.c_void_p]
    runtime.cudaStreamQuery.restype = ctypes.c_int
    runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaGetDeviceCount.restype = ctypes.c_int
    return runtime


@pytest.fixture
def cuda_stream(cuda_runtime):
    runtime = cuda_runtime
    stream = ctypes.c_void_p()
    result = runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1)
    if result != 0:
        pytest.skip(f'cudaStreamCreateWithFlags failed with error {result}')
    try:
        yield stream.value
    finally:
        assert runtime.cudaStreamDestroy(stream) == 0


def cuda_message(values, cuda_buffer):
    msg = allocate_tensor_msg(values.shape, values.dtype, 'cuda')
    msg.data = cuda_buffer.from_cpu(values.tobytes())
    return msg


def test_cuda_is_preferred_over_host_memory():
    assert default_backend() == 'cuda'
    assert allocate_tensor_msg((4,), np.float32).data.backend_type == 'cuda'


def test_stream_ownership_and_validation(cuda_runtime, cuda_stream):
    owner = create_stream()
    assert owner.backend == 'cuda'
    assert owner.device_id == 0
    assert owner.owns_stream
    assert create_stream('cuda').handle != owner.handle
    retained = owner
    del owner
    gc.collect()
    assert cuda_runtime.cudaStreamSynchronize(retained.handle) == 0
    borrowed = borrow_stream(cuda_stream, 'cuda')
    assert not borrowed.owns_stream
    assert borrowed.handle == cuda_stream
    del borrowed
    gc.collect()
    assert cuda_runtime.cudaStreamSynchronize(cuda_stream) == 0
    assert borrow_stream(0, 'cuda').handle == 0
    with pytest.raises(ValueError, match='integer native handle'):
        borrow_stream(None, 'cuda')
    count = ctypes.c_int()
    assert cuda_runtime.cudaGetDeviceCount(ctypes.byref(count)) == 0
    with pytest.raises(RuntimeError, match='cudaSetDevice'):
        create_stream('cuda', count.value)
    with pytest.raises(RuntimeError, match='cudaSetDevice'):
        borrow_stream(cuda_stream, 'cuda', count.value)


@pytest.mark.parametrize('owned', [True, False])
def test_stream_orders_matmul_inference(owned, cuda_runtime, cuda_stream):
    stream = create_stream() if owned else borrow_stream(cuda_stream, 'cuda')
    session = ort.InferenceSession(matmul_model(), providers=session_providers(stream=stream))
    provider_options = session.get_provider_options()['CUDAExecutionProvider']
    assert int(provider_options['user_compute_stream']) == stream.handle
    input_msg = allocate_tensor_msg((2, 2), np.float32, stream=stream)
    output_msg = allocate_tensor_msg((2, 2), np.float32, stream=stream)
    zeros = np.zeros((2, 2), dtype=np.float32)
    values = np.full((2, 2), 7, dtype=np.float32)
    delayed = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: time.sleep(0.2))
    with from_output_tensor_msg(input_msg, stream) as value:
        assert cuda_runtime.cudaMemcpyAsync(
            value.data_ptr(), zeros.ctypes.data, zeros.nbytes, 1, stream.handle) == 0
        assert cuda_runtime.cudaStreamSynchronize(stream.handle) == 0
        assert cuda_runtime.cudaLaunchHostFunc(stream.handle, delayed, None) == 0
        assert cuda_runtime.cudaMemcpyAsync(
            value.data_ptr(), values.ctypes.data, values.nbytes, 1, stream.handle) == 0
    del value
    with from_input_tensor_msg(input_msg, stream) as source:
        with from_output_tensor_msg(output_msg, stream) as output:
            binding = session.io_binding()
            binding.bind_ortvalue_input('input', source)
            binding.bind_ortvalue_output('output', output)
            session.run_with_iobinding(binding)
            assert cuda_runtime.cudaStreamSynchronize(stream.handle) == 0
            assert np.array_equal(output.numpy(), np.full((2, 2), 98, dtype=np.float32))
            del binding, output
    del source


def test_host_storage_stays_available_alongside_cuda():
    msg = allocate_tensor_msg((4,), np.float32, 'cpu')

    assert not hasattr(msg.data, 'backend_type')


def test_views_report_cuda_device_memory(cuda_stream):
    msg = allocate_tensor_msg((2, 3), np.float32, 'cuda')

    with from_output_tensor_msg(msg, cuda_stream) as value:
        assert value.device_name().lower() == 'cuda'
        assert value.shape() == [2, 3]
        assert value.element_type() == ElementType.FLOAT
        assert value.data_ptr() != 0


@pytest.mark.parametrize('conversion', [
    from_input_tensor_msg, from_output_tensor_msg])
def test_views_keep_storage_alive_after_message_destruction(
    conversion, cuda_stream,
):
    msg = allocate_tensor_msg((4,), np.float32, 'cuda')
    view = conversion(msg, cuda_stream)
    pointer = view.value.data_ptr()
    del msg
    gc.collect()
    assert view.value.data_ptr() == pointer
    assert view.value.numpy().shape == (4,)
    view.close()


def test_view_orders_the_producer_before_consumer_work(cuda_runtime, cuda_stream):
    producer = ctypes.c_void_p()
    assert cuda_runtime.cudaStreamCreateWithFlags(ctypes.byref(producer), 1) == 0
    delayed = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: time.sleep(0.2))
    expected = np.arange(4, dtype=np.float32)
    msg = allocate_tensor_msg((4,), np.float32, 'cuda')
    try:
        with from_output_tensor_msg(msg, producer.value) as value:
            assert cuda_runtime.cudaLaunchHostFunc(producer, delayed, None) == 0
            assert cuda_runtime.cudaMemcpyAsync(
                value.data_ptr(), expected.ctypes.data, expected.nbytes,
                1, producer) == 0
        with from_input_tensor_msg(msg, cuda_stream) as value:
            actual = np.empty_like(expected)
            assert cuda_runtime.cudaMemcpyAsync(
                actual.ctypes.data, value.data_ptr(), actual.nbytes,
                2, cuda_stream) == 0
            assert cuda_runtime.cudaStreamSynchronize(cuda_stream) == 0
            assert np.array_equal(actual, expected)
    finally:
        assert cuda_runtime.cudaStreamDestroy(producer) == 0


def test_copy_finishes_before_releasing_the_source(cuda_runtime, cuda_stream):
    delayed = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: time.sleep(0.2))
    expected = np.arange(4, dtype=np.float32)
    value = ort.OrtValue.ortvalue_from_numpy(expected)
    msg = allocate_tensor_msg((4,), np.float32, 'cuda')
    assert cuda_runtime.cudaLaunchHostFunc(cuda_stream, delayed, None) == 0
    to_tensor_msg(msg, value, cuda_stream)
    assert cuda_runtime.cudaStreamQuery(cuda_stream) == 0
    del value
    with from_input_tensor_msg(msg, cuda_stream) as result:
        assert np.array_equal(result.numpy(), expected)


def test_preserves_and_validates_device_indices(cuda_runtime):
    count = ctypes.c_int()
    assert cuda_runtime.cudaGetDeviceCount(ctypes.byref(count)) == 0
    current = ctypes.c_int()
    assert cuda_runtime.cudaGetDevice(ctypes.byref(current)) == 0
    allocate_tensor_msg((4,), np.float32, 'cuda')
    with pytest.raises(RuntimeError, match='cudaSetDevice'):
        allocate_tensor_msg((4,), np.float32, 'cuda', count.value)
    for device in range(count.value):
        if device != current.value:
            with pytest.raises(RuntimeError, match='CUDA buffer is on device'):
                allocate_tensor_msg((4,), np.float32, 'cuda', device)
            continue
        msg = allocate_tensor_msg((4,), np.float32, 'cuda', device)
        with from_input_tensor_msg(msg, 0) as value:
            assert value.__dlpack_device__() == (2, device)
            copied = to_tensor_msg(value, stream=0)
        with from_input_tensor_msg(copied, 0) as value:
            assert value.__dlpack_device__() == (2, device)


def test_cuda_views_require_an_explicit_stream():
    msg = allocate_tensor_msg((4,), np.float32, 'cuda')
    with pytest.raises(ValueError, match='explicit execution stream'):
        from_input_tensor_msg(msg)


def test_views_alias_the_device_pointer(cuda_buffer, cuda_stream):
    msg = allocate_tensor_msg((8,), ElementType.BOOL, 'cuda')
    msg.data = cuda_buffer.from_cpu(bytes([0, 1] * 4))

    with cuda_buffer.from_input_buffer(msg.data, cuda_stream) as handle:
        expected_pointer = handle.device_ptr

    with from_input_tensor_msg(msg, cuda_stream) as value:
        assert value.data_ptr() == expected_pointer
        assert value.element_type() == ElementType.BOOL
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
        assert value.data_ptr() != 0
        assert np.array_equal(value.numpy(), source)

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
    session = ort.InferenceSession(
        identity_model((4,)),
        providers=session_providers('cuda', 0, cuda_stream),
    )
    input_msg = cuda_message(np.arange(4, dtype=np.float32), cuda_buffer)
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


def test_to_tensor_msg_copies_host_values_onto_the_device(cuda_stream):
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    value = ort.OrtValue.ortvalue_from_numpy(source)
    destination = allocate_tensor_msg((2, 3), np.float32, 'cuda')

    to_tensor_msg(destination, value, cuda_stream)

    assert destination.data.backend_type == 'cuda'
    assert list(destination.shape) == [2, 3]
    assert np.array_equal(
        np.frombuffer(destination.data.to_bytes(), dtype=np.float32).reshape(
            2, 3),
        source)


def test_to_tensor_msg_copies_device_values_without_staging_on_the_host(
    cuda_buffer, cuda_stream,
):
    source = np.arange(4, dtype=np.float32)
    staged = cuda_message(source, cuda_buffer)
    destination = allocate_tensor_msg((4,), np.float32, 'cuda')

    with from_input_tensor_msg(staged, cuda_stream) as value:
        to_tensor_msg(destination, value, cuda_stream)

    assert destination.data.backend_type == 'cuda'
    assert np.array_equal(
        np.frombuffer(destination.data.to_bytes(), dtype=np.float32), source)


def test_to_tensor_msg_allocates_on_the_device_that_owns_the_value(
    cuda_buffer, cuda_stream,
):
    source = np.arange(4, dtype=np.float32)
    staged = cuda_message(source, cuda_buffer)

    with from_input_tensor_msg(staged, cuda_stream) as value:
        allocated = to_tensor_msg(value, stream=cuda_stream)

    assert allocated.data.backend_type == 'cuda'
    assert np.array_equal(
        np.frombuffer(allocated.data.to_bytes(), dtype=np.float32), source)


def test_to_tensor_msg_copies_device_values_into_host_storage(
    cuda_buffer, cuda_stream,
):
    source = np.arange(4, dtype=np.float32)
    staged = cuda_message(source, cuda_buffer)
    destination = allocate_tensor_msg((4,), np.float32, 'cpu')

    with from_input_tensor_msg(staged, cuda_stream) as value:
        to_tensor_msg(destination, value, cuda_stream)

    assert np.array_equal(
        np.frombuffer(destination.data, dtype=np.float32), source)
