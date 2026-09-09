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
from pathlib import Path

from cuda_buffer import CudaBuffer

import dlpack_conversions
from dlpack_conversions import allocate_tensor_msg
from dlpack_conversions import from_input_tensor_msg
from dlpack_conversions import from_output_tensor_msg

import pytest


FLOAT32 = (2, 32, 1)

CUDA_AVAILABLE = dlpack_conversions.backend_available('cuda')


class DLDevice(ctypes.Structure):
    _fields_ = [
        ('device_type', ctypes.c_int32),
        ('device_id', ctypes.c_int32),
    ]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ('code', ctypes.c_uint8),
        ('bits', ctypes.c_uint8),
        ('lanes', ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.c_void_p),
        ('device', DLDevice),
        ('ndim', ctypes.c_int32),
        ('dtype', DLDataType),
        ('shape', ctypes.POINTER(ctypes.c_int64)),
        ('strides', ctypes.POINTER(ctypes.c_int64)),
        ('byte_offset', ctypes.c_uint64),
    ]


def dl_tensor(capsule) -> DLTensor:
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.POINTER(DLTensor)
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return get_pointer(capsule, b'dltensor').contents


def test_plugin_has_no_framework_dependency():
    root = Path(__file__).parents[1]
    source = (root / 'dlpack_conversions_cuda/_plugin.py').read_text()
    manifest = (root / 'package.xml').read_text()

    assert 'import torch' not in source
    assert 'onnxruntime' not in source
    assert 'torch' not in manifest
    assert 'onnxruntime' not in manifest


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA support is unavailable')
def test_plugin_is_discovered_through_the_ament_index():
    assert dlpack_conversions.backend_for_device(
        dlpack_conversions.CUDA) == 'cuda'
    assert dlpack_conversions.default_backend() == 'cuda'


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA support is unavailable')
def test_allocate_uses_cuda_storage():
    msg = allocate_tensor_msg((2, 3), FLOAT32, 'cuda')

    assert msg.data.backend_type == 'cuda'
    assert len(msg.data) == 24
    assert list(msg.strides) == [3, 1]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA support is unavailable')
def test_capsule_aliases_device_storage():
    msg = allocate_tensor_msg((8,), FLOAT32, 'cuda')
    with CudaBuffer.from_input_buffer(msg.data, 0) as handle:
        expected_pointer = handle.device_ptr
        expected_device = handle.device_id

    # The ctypes view borrows the capsule's memory, so keep the capsule.
    capsule = from_output_tensor_msg(msg, 0)
    tensor = dl_tensor(capsule)

    assert tensor.data == expected_pointer
    assert tensor.device.device_type == dlpack_conversions.CUDA
    assert tensor.device.device_id == expected_device
    assert tensor.byte_offset == 0
    assert [tensor.shape[0]] == [8]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA support is unavailable')
def test_byte_offset_is_folded_into_the_device_pointer():
    msg = allocate_tensor_msg((8,), FLOAT32, 'cuda')
    base = dl_tensor(from_input_tensor_msg(msg, 0)).data
    msg.shape = [4]
    msg.strides = [1]
    msg.byte_offset = 16

    tensor = dl_tensor(from_input_tensor_msg(msg, 0))

    assert tensor.data == base + 16
    assert tensor.byte_offset == 0
