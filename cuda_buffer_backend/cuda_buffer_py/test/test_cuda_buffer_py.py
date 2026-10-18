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
import ctypes.util
import gc
import subprocess
import sys
import textwrap

from cuda_buffer import CudaBuffer
from cuda_buffer import CudaReadHandle
from cuda_buffer import CudaWriteHandle
import pytest
from rosidl_buffer import Buffer


@pytest.fixture
def cudart():
    runtime = ctypes.CDLL(ctypes.util.find_library('cudart') or 'libcudart.so')
    signatures = {
        'cudaStreamCreate': [ctypes.POINTER(ctypes.c_void_p)],
        'cudaStreamDestroy': [ctypes.c_void_p],
        'cudaStreamQuery': [ctypes.c_void_p],
        'cudaStreamSynchronize': [ctypes.c_void_p],
        'cudaMemsetAsync': [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p],
        'cudaMemcpyAsync': [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
    }
    for name, arguments in signatures.items():
        function = getattr(runtime, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    return runtime


@pytest.fixture
def stream(cudart):
    pointer = ctypes.c_void_p()
    assert cudart.cudaStreamCreate(ctypes.byref(pointer)) == 0
    try:
        yield pointer.value
    finally:
        assert cudart.cudaStreamSynchronize(pointer) == 0
        assert cudart.cudaStreamDestroy(pointer) == 0


def test_core_operations():
    assert {name for name in vars(CudaBuffer) if not name.startswith('_')} == {
        'allocate_buffer', 'from_output_buffer', 'from_input_buffer', 'to_buffer'}


@pytest.mark.parametrize('source', [
    bytes(range(32)), bytearray(range(32)), memoryview(bytes(range(32))), range(32),
])
def test_to_buffer_host_data(source):
    buffer = CudaBuffer.allocate_buffer(32)

    with CudaBuffer.from_output_buffer(buffer) as handle:
        assert CudaBuffer.to_buffer(source, 32, handle) is None
        assert not handle.closed
        assert handle.buffer is buffer

    assert isinstance(buffer, Buffer)
    assert buffer.backend_type == 'cuda'
    assert len(buffer) == 32
    assert buffer.to_bytes() == bytes(range(32))


@pytest.mark.parametrize('explicit_stream', [False, True])
def test_to_buffer_keeps_producer_open(cudart, stream, explicit_stream):
    buffer = CudaBuffer.allocate_buffer(32)
    copy_stream = stream if explicit_stream else None
    with CudaBuffer.from_output_buffer(buffer, stream) as handle:
        pointer = handle.device_ptr
        CudaBuffer.to_buffer(bytes(32), 32, handle, copy_stream)
        CudaBuffer.to_buffer(b'\x01\x02\x03\x04', 4, handle, copy_stream)
        assert not handle.closed
        assert handle.device_ptr == pointer
        # Additional device writes belong to the same producer scope.
        assert cudart.cudaMemsetAsync(pointer + 4, 9, 28, stream) == 0

    assert buffer.to_bytes() == b'\x01\x02\x03\x04' + bytes([9]) * 28
    with pytest.raises(RuntimeError, match='write.*finalized'):
        CudaBuffer.from_output_buffer(buffer, stream)


@pytest.mark.parametrize('explicit_stream', [False, True])
@pytest.mark.parametrize('byte_count', [16, 32])
def test_to_buffer_device_data(cudart, stream, explicit_stream, byte_count):
    source = CudaBuffer.allocate_buffer(32)
    # The source producer uses a different stream. The read handle below must
    # wait for that producer before the D2D copy on the destination stream.
    with CudaBuffer.from_output_buffer(source) as writer:
        CudaBuffer.to_buffer(bytes(range(32)), 32, writer)

    destination = CudaBuffer.allocate_buffer(32)
    with CudaBuffer.from_output_buffer(destination, stream) as writer:
        CudaBuffer.to_buffer(bytes(32), 32, writer)
        with CudaBuffer.from_input_buffer(source, stream) as reader:
            CudaBuffer.to_buffer(
                reader.device_ptr, byte_count, writer, stream if explicit_stream else None)
            assert not writer.closed
            assert not reader.closed
        # The read event keeps the source allocation alive for the queued copy.
        del source
        gc.collect()
        assert cudart.cudaMemsetAsync(writer.device_ptr + 31, 99, 1, stream) == 0

    expected = bytes(range(byte_count)) + bytes(32 - byte_count)
    assert destination.to_bytes() == expected[:31] + b'c'


def test_to_buffer_host_source_lifetime(cudart, stream):
    size = 8 * 1024 * 1024
    source = bytearray([23]) * size
    buffer = CudaBuffer.allocate_buffer(size)
    with CudaBuffer.from_output_buffer(buffer, stream) as handle:
        CudaBuffer.to_buffer(source, size, handle, stream)
        # Completion is part of the host-copy contract, including the temporary
        # bytes conversion above; enqueueing alone cannot protect that storage.
        assert cudart.cudaStreamQuery(stream) == 0
        source[:] = bytes(size)
        del source
        gc.collect()
        assert not handle.closed

    assert buffer.to_bytes() == bytes([23]) * size


@pytest.mark.parametrize('source, byte_count, message', [
    (b'abcd', -1, 'nonnegative'),
    (b'abcd', 5, 'destination size'),
    (b'ab', 3, 'host source size'),
    (0, 1, 'nonzero'),
    (1, 5, 'destination size'),
    (1, -1, 'nonnegative'),
])
def test_to_buffer_validates_sizes(source, byte_count, message):
    buffer = CudaBuffer.allocate_buffer(4)
    with CudaBuffer.from_output_buffer(buffer) as handle:
        with pytest.raises(ValueError, match=message):
            CudaBuffer.to_buffer(source, byte_count, handle)
        assert not handle.closed
        CudaBuffer.to_buffer(b'abcd', 4, handle)
    assert buffer.to_bytes() == b'abcd'


@pytest.mark.parametrize('byte_count', [None, 1.5, '1'])
def test_to_buffer_requires_integer_size(byte_count):
    buffer = CudaBuffer.allocate_buffer(4)
    with CudaBuffer.from_output_buffer(buffer) as handle:
        with pytest.raises(TypeError):
            CudaBuffer.to_buffer(b'abcd', byte_count, handle)


def test_to_buffer_zero_bytes():
    buffer = CudaBuffer.allocate_buffer(4)
    with CudaBuffer.from_output_buffer(buffer) as handle:
        CudaBuffer.to_buffer(b'abcd', 4, handle)
        CudaBuffer.to_buffer(b'', 0, handle)
        CudaBuffer.to_buffer(0, 0, handle)
    assert buffer.to_bytes() == b'abcd'


def test_to_buffer_host_prefix():
    buffer = CudaBuffer.allocate_buffer(4)
    with CudaBuffer.from_output_buffer(buffer) as handle:
        CudaBuffer.to_buffer(b'abcdefgh', 4, handle)
    assert buffer.to_bytes() == b'abcd'


@pytest.mark.parametrize('source, byte_count', [(b'a', 1), (b'', 0), (0, 0), (1, 1)])
def test_to_buffer_rejects_closed_handle(source, byte_count):
    buffer = CudaBuffer.allocate_buffer(4)
    handle = CudaBuffer.from_output_buffer(buffer)
    handle.close()
    with pytest.raises(RuntimeError, match='closed'):
        CudaBuffer.to_buffer(source, byte_count, handle)


def test_to_buffer_requires_write_handle():
    with CudaBuffer.from_input_buffer(b'abcd') as handle:
        with pytest.raises(TypeError):
            CudaBuffer.to_buffer(b'abcd', 4, handle)


def test_to_buffer_rejects_different_stream(stream):
    buffer = CudaBuffer.allocate_buffer(4)
    with CudaBuffer.from_output_buffer(buffer) as handle:
        with pytest.raises(ValueError, match='stream must match'):
            CudaBuffer.to_buffer(b'abcd', 4, handle, stream)
        CudaBuffer.to_buffer(b'abcd', 4, handle)
    assert buffer.to_bytes() == b'abcd'


def test_standalone_without_torch():
    # Use a fresh interpreter so an already-imported Torch cannot hide a dependency.
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent("""
            import sys
            sys.modules['torch'] = None
            from cuda_buffer import CudaBuffer
            buffer = CudaBuffer.allocate_buffer(4)
            with CudaBuffer.from_output_buffer(buffer) as handle:
                CudaBuffer.to_buffer(b'abcd', 4, handle)
            assert buffer.to_bytes() == b'abcd'
        """)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_empty_buffer():
    buffer = CudaBuffer.allocate_buffer(0)

    assert buffer.backend_type == 'cuda'
    assert len(buffer) == 0
    assert buffer.to_bytes() == b''


def test_allocate_buffer_is_uninitialized_cuda_storage():
    buffer = CudaBuffer.allocate_buffer(64)

    assert isinstance(buffer, Buffer)
    assert buffer.backend_type == 'cuda'
    assert len(buffer) == 64


def test_from_output_cuda_buffer():
    buffer = CudaBuffer.allocate_buffer(64)

    with CudaBuffer.from_output_buffer(buffer) as handle:
        assert isinstance(handle, CudaWriteHandle)
        assert handle.buffer is buffer
        assert handle.device_ptr != 0
        assert handle.get_ptr() == handle.device_ptr
        assert not handle.closed

    assert handle.closed
    assert handle.buffer is buffer
    with pytest.raises(RuntimeError, match='closed'):
        handle.get_ptr()


def test_from_output_cpu_data_promotes_to_cuda():
    handle = CudaBuffer.from_output_buffer(bytes(32))

    assert isinstance(handle, CudaWriteHandle)
    assert isinstance(handle.buffer, Buffer)
    assert handle.buffer.backend_type == 'cuda'
    assert len(handle.buffer) == 32
    assert handle.device_ptr != 0
    CudaBuffer.to_buffer(bytes(range(32)), 32, handle)
    handle.close()
    handle.close()
    assert handle.closed
    assert handle.buffer.backend_type == 'cuda'
    assert handle.buffer.to_bytes() == bytes(range(32))


def test_from_output_can_only_be_acquired_once():
    buffer = CudaBuffer.allocate_buffer(32)

    with CudaBuffer.from_output_buffer(buffer):
        with pytest.raises(RuntimeError, match='already in use'):
            CudaBuffer.from_output_buffer(buffer)

    with pytest.raises(RuntimeError, match='write.*finalized'):
        CudaBuffer.from_output_buffer(buffer)


def test_from_output_empty_buffer_raises():
    with pytest.raises(RuntimeError, match='empty buffer'):
        CudaBuffer.from_output_buffer(b'')


def test_from_input_cuda_buffer():
    buffer = CudaBuffer.allocate_buffer(32)
    with CudaBuffer.from_output_buffer(buffer) as handle:
        CudaBuffer.to_buffer(bytes(range(32)), 32, handle)

    with CudaBuffer.from_input_buffer(buffer) as handle:
        assert isinstance(handle, CudaReadHandle)
        assert handle.device_ptr != 0
        assert handle.get_ptr() == handle.device_ptr
        assert not handle.closed

    assert handle.closed
    with pytest.raises(RuntimeError, match='closed'):
        handle.get_ptr()


def test_from_input_cpu_data_promotes_to_cuda():
    handle = CudaBuffer.from_input_buffer(bytes(range(32)))

    assert isinstance(handle, CudaReadHandle)
    assert handle.device_ptr != 0
    handle.close()
    handle.close()
    assert handle.closed


@pytest.mark.parametrize('buffer_type', [bytes, bytearray, memoryview])
def test_cpu_input_promotion_lifetime(cudart, stream, buffer_type):
    expected = bytes(range(256)) * 4096
    source = buffer_type(expected)
    with CudaBuffer.from_input_buffer(source, stream) as handle:
        assert cudart.cudaStreamQuery(stream) == 0
        del source
        gc.collect()
        host = ctypes.create_string_buffer(len(expected))
        assert cudart.cudaMemcpyAsync(host, handle.device_ptr, len(expected), 2, stream) == 0
        assert cudart.cudaStreamSynchronize(stream) == 0
        assert host.raw == expected


def test_from_input_empty_buffer_raises():
    with pytest.raises(RuntimeError, match='empty buffer'):
        CudaBuffer.from_input_buffer(b'')
