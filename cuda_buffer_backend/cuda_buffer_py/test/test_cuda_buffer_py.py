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

from cuda_buffer import CudaBuffer
from cuda_buffer import CudaReadHandle
from cuda_buffer import CudaWriteHandle
import pytest
from rosidl_buffer import Buffer


def test_from_cpu_bytes():
    data = bytes(range(32))

    buffer = CudaBuffer.from_cpu(data)

    assert isinstance(buffer, Buffer)
    assert buffer.backend_type == 'cuda'
    assert len(buffer) == len(data)
    assert buffer.to_bytes() == data


def test_from_cpu_iterable():
    buffer = CudaBuffer.from_cpu(range(16))

    assert buffer.backend_type == 'cuda'
    assert buffer.to_bytes() == bytes(range(16))


def test_from_size_is_zero_initialized():
    buffer = CudaBuffer.from_size(64)

    assert buffer.backend_type == 'cuda'
    assert len(buffer) == 64
    assert buffer.to_bytes() == bytes(64)


def test_empty_buffer():
    buffer = CudaBuffer.from_cpu(b'')

    assert buffer.backend_type == 'cuda'
    assert len(buffer) == 0
    assert buffer.to_bytes() == b''


def test_allocate_buffer_is_uninitialized_cuda_storage():
    buffer = CudaBuffer.allocate_buffer(64)

    assert isinstance(buffer, Buffer)
    assert buffer.backend_type == 'cuda'
    assert len(buffer) == 64


def test_internal_stream_is_exposed():
    assert isinstance(CudaBuffer.get_internal_stream(), int)
    assert CudaBuffer.get_internal_stream() == CudaBuffer.get_internal_stream()


def test_from_output_cuda_buffer():
    buffer = CudaBuffer.allocate_buffer(64)

    with CudaBuffer.from_output_buffer(buffer) as handle:
        assert isinstance(handle, CudaWriteHandle)
        assert handle.buffer is buffer
        assert handle.device_ptr != 0
        assert handle.device_id >= 0
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
    handle.close()
    handle.close()
    assert handle.closed
    assert handle.buffer.backend_type == 'cuda'


def test_from_output_can_only_be_acquired_once():
    buffer = CudaBuffer.allocate_buffer(32)

    with CudaBuffer.from_output_buffer(buffer):
        pass

    with pytest.raises(RuntimeError, match='write.*finalized'):
        CudaBuffer.from_output_buffer(buffer)


def test_from_output_empty_buffer_raises():
    with pytest.raises(RuntimeError, match='empty buffer'):
        CudaBuffer.from_output_buffer(b'')


def test_from_input_cuda_buffer():
    buffer = CudaBuffer.from_cpu(bytes(range(32)))

    with CudaBuffer.from_input_buffer(buffer) as handle:
        assert isinstance(handle, CudaReadHandle)
        assert handle.device_ptr != 0
        assert handle.device_id >= 0
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


def test_from_input_empty_buffer_raises():
    with pytest.raises(RuntimeError, match='empty buffer'):
        CudaBuffer.from_input_buffer(b'')
