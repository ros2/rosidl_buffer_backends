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

import dlpack_conversions
from dlpack_conversions import allocate_tensor_msg
from dlpack_conversions import from_input_tensor_msg
from dlpack_conversions import from_output_tensor_msg

import numpy

import pytest

from tensor_msgs.msg import ExperimentalTensor


FLOAT32 = (2, 32, 1)
INT32 = (0, 32, 1)


class Consumer:
    """Present a bare DLPack capsule through the array API protocol."""

    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, stream=None, **kwargs):
        del stream, kwargs
        return self._capsule

    def __dlpack_device__(self):
        return (dlpack_conversions.CPU, 0)


def through_dlpack(capsule) -> numpy.ndarray:
    """Read a capsule. numpy hands back a read-only view of DLPack v0."""
    return numpy.from_dlpack(Consumer(capsule))


def fill(msg: ExperimentalTensor, values) -> None:
    """Write through the message storage rather than the DLPack view."""
    numpy.frombuffer(msg.data, dtype=numpy.int32)[:] = values


def test_plugin_is_discovered_through_the_ament_index():
    assert 'cpu' in dlpack_conversions.available_backends()
    assert dlpack_conversions.backend_available('cpu')
    assert dlpack_conversions.backend_for_device(
        dlpack_conversions.CPU) == 'cpu'


def test_allocate_fills_dlpack_metadata():
    msg = allocate_tensor_msg((2, 3, 4), FLOAT32, 'cpu')

    assert isinstance(msg.data, array)
    assert list(msg.shape) == [2, 3, 4]
    assert list(msg.strides) == [12, 4, 1]
    assert (msg.dtype_code, msg.dtype_bits, msg.dtype_lanes) == FLOAT32
    assert msg.byte_offset == 0
    assert len(msg.data) == 96


def test_input_and_output_views_alias_message_storage():
    msg = allocate_tensor_msg((4,), INT32, 'cpu')
    fill(msg, [10, 20, 30, 40])

    assert through_dlpack(from_input_tensor_msg(msg)).tolist() == \
        [10, 20, 30, 40]
    assert through_dlpack(from_output_tensor_msg(msg)).tolist() == \
        [10, 20, 30, 40]


def test_byte_offset_selects_a_storage_subregion():
    msg = allocate_tensor_msg((16,), INT32, 'cpu')
    fill(msg, numpy.arange(16) * 100)
    msg.shape = [4]
    msg.strides = [1]
    msg.byte_offset = 16

    assert through_dlpack(from_input_tensor_msg(msg)).tolist() == \
        [400, 500, 600, 700]


def test_non_contiguous_strides_are_honored():
    msg = allocate_tensor_msg((8,), INT32, 'cpu')
    fill(msg, numpy.arange(8))
    msg.shape = [4]
    msg.strides = [2]

    assert through_dlpack(from_input_tensor_msg(msg)).tolist() == [0, 2, 4, 6]


def test_empty_buffer_returns_none():
    msg = ExperimentalTensor()

    assert from_input_tensor_msg(msg) is None
    assert from_output_tensor_msg(msg) is None


def test_invalid_shape_and_strides_are_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        allocate_tensor_msg((-1,), FLOAT32, 'cpu')

    msg = allocate_tensor_msg((2, 2), FLOAT32, 'cpu')
    msg.strides = [1]
    with pytest.raises(ValueError, match='match rank and be nonnegative'):
        from_input_tensor_msg(msg)


def test_undersized_storage_is_rejected():
    msg = allocate_tensor_msg((4,), INT32, 'cpu')
    msg.shape = [64]
    msg.strides = [1]

    with pytest.raises(ValueError, match='buffer has 16'):
        from_input_tensor_msg(msg)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match='No storage plugin provides'):
        allocate_tensor_msg((4,), FLOAT32, 'nonexistent')


def test_the_host_plugin_refuses_to_read_accelerator_memory():
    from dlpack_conversions_cpu._plugin import CpuStoragePlugin

    with pytest.raises(ValueError, match="'cuda' plugin owns that copy"):
        CpuStoragePlugin().copy_to(array('B', bytes(16)), 0, 16, 'cuda', None)


def test_the_host_plugin_copies_host_memory_into_storage():
    from dlpack_conversions_cpu._plugin import CpuStoragePlugin

    source = numpy.arange(4, dtype=numpy.int32)
    destination = array('B', bytes(source.nbytes))

    CpuStoragePlugin().copy_to(
        destination, source.ctypes.data, source.nbytes, 'cpu', None)

    assert numpy.array_equal(
        numpy.frombuffer(destination, dtype=numpy.int32), source)
