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
import ctypes
import gc
from pathlib import Path
import weakref

import dlpack_conversions
from dlpack_conversions import StorageRegistry
from dlpack_conversions._dlpack_bridge import buffer_address
from dlpack_conversions._dlpack_bridge import capsule_device
from dlpack_conversions._dlpack_bridge import make_dlpack_capsule

import numpy

import pytest


class Consumer:
    """Present a bare DLPack capsule through the array API protocol."""

    def __init__(self, capsule, device_type=dlpack_conversions.CPU):
        self._capsule = capsule
        self._device_type = device_type

    def __dlpack__(self, stream=None, **kwargs):
        del stream, kwargs
        return self._capsule

    def __dlpack_device__(self):
        return (self._device_type, 0)


class HostPlugin:
    """Minimal host storage plugin standing in for an installed one."""

    backends = ('host',)
    dl_device_types = (dlpack_conversions.CPU,)
    priority = 0
    is_fallback = True

    def is_available(self) -> bool:
        return True

    def matches(self, data: object) -> bool:
        return isinstance(data, array)

    def allocate(self, byte_count: int, backend: str) -> array:
        del backend
        return array('B', bytes(byte_count))

    def acquire_input(self, data, metadata, stream) -> object:
        del stream
        return make_dlpack_capsule(
            buffer_address(data) + metadata.byte_offset,
            dlpack_conversions.CPU, 0,
            metadata.dtype_code, metadata.dtype_bits, metadata.dtype_lanes,
            list(metadata.shape), list(metadata.strides), 0, data,
        )

    def acquire_output(self, data, metadata, stream) -> object:
        return self.acquire_input(data, metadata, stream)

    def unavailable_error(self) -> RuntimeError:
        return RuntimeError('unavailable')


class Accelerator(HostPlugin):
    backends = ('fake',)
    dl_device_types = (dlpack_conversions.ROCM,)
    priority = 50
    is_fallback = False


def test_bridge_retains_owner_and_honors_strides():
    class Owner:

        def __init__(self):
            self.data = (ctypes.c_float * 8)(*range(8))

    owner = Owner()
    owner_ref = weakref.ref(owner)
    capsule = make_dlpack_capsule(
        ctypes.addressof(owner.data),
        dlpack_conversions.CPU,
        0,
        2,
        32,
        1,
        [2, 2],
        [4, 2],
        0,
        owner,
    )
    view = numpy.from_dlpack(Consumer(capsule))
    del capsule
    del owner
    gc.collect()

    assert owner_ref() is not None
    assert view.tolist() == [[0.0, 2.0], [4.0, 6.0]]
    owner_ref().data[0] = 42
    assert view[0, 0] == 42

    del view
    gc.collect()
    assert owner_ref() is None


def test_buffer_address_matches_host_storage():
    storage = array('B', bytes(8))

    assert buffer_address(storage) == storage.buffer_info()[0]


def test_capsule_device_reports_the_producing_device():
    storage = array('B', bytes(16))
    plugin = HostPlugin()
    metadata = dlpack_conversions.TensorMetadata(
        [4], [1], 0, 32, 1, 4, 0, 4)

    capsule = plugin.acquire_input(storage, metadata, None)

    assert capsule_device(capsule) == (dlpack_conversions.CPU, 0)


def test_core_has_no_framework_dependency():
    root = Path(__file__).parents[1]
    source = (root / 'src/dlpack_bridge.cpp').read_text()
    cmake = (root / 'CMakeLists.txt').read_text()
    manifest = (root / 'package.xml').read_text()

    assert '#include <ATen/' not in source
    assert '#include <torch/' not in source
    assert '#include <cuda' not in source
    assert 'onnxruntime' not in source
    assert 'find_package(cuda_buffer' not in cmake
    assert 'find_package(libtorch_vendor' not in cmake
    assert 'torch' not in manifest
    assert 'onnxruntime' not in manifest
    assert 'cuda' not in manifest


def test_registry_rejects_duplicate_backend():
    registry = StorageRegistry()
    registry.register(HostPlugin())

    with pytest.raises(ValueError, match='already registered'):
        registry.register(HostPlugin())


def test_registry_reports_backends_and_devices():
    registry = StorageRegistry()
    registry.register(HostPlugin())

    assert registry.backends() == ['host']
    assert registry.backend_for_device(dlpack_conversions.CPU) == 'host'
    assert registry.backend_for_device(dlpack_conversions.CUDA) is None
    assert registry.device_for_backend('host') == dlpack_conversions.CPU
    assert registry.default_backend() == 'host'


def test_registry_prefers_the_highest_priority_backend():
    registry = StorageRegistry()
    registry.register(HostPlugin())
    registry.register(Accelerator())

    assert registry.backends() == ['fake', 'host']
    assert registry.default_backend() == 'fake'


def test_registry_skips_unavailable_plugins():
    class Missing(Accelerator):

        def is_available(self) -> bool:
            return False

    registry = StorageRegistry()
    registry.register(HostPlugin())
    registry.register(Missing())

    assert registry.backends() == ['host']
    assert registry.default_backend() == 'host'


def test_registry_rejects_unknown_backend_and_storage():
    registry = StorageRegistry()
    registry.register(HostPlugin())

    with pytest.raises(ValueError, match='No storage plugin provides'):
        registry.for_backend('cuda')
    with pytest.raises(ValueError, match='Unsupported tensor storage'):
        registry.for_data(object())


def test_environment_override_selects_a_backend(monkeypatch):
    registry = StorageRegistry()
    registry.register(HostPlugin())
    registry.register(Accelerator())

    monkeypatch.setenv('ROSIDL_TENSOR_BACKEND', 'host')
    assert registry.default_backend() == 'host'

    monkeypatch.setenv('ROSIDL_TENSOR_BACKEND', 'rocm')
    with pytest.raises(RuntimeError, match='which is not available'):
        registry.default_backend()


def test_contiguous_strides_are_row_major():
    assert dlpack_conversions.contiguous_strides([2, 3, 4]) == [12, 4, 1]
    assert dlpack_conversions.contiguous_strides([]) == []


def test_to_tensor_msg_allocates_and_copies_from_a_capsule():
    source = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)

    msg = dlpack_conversions.to_tensor_msg(source.__dlpack__())

    assert list(msg.shape) == [2, 3]
    assert list(msg.strides) == [3, 1]
    assert msg.byte_offset == 0
    assert (msg.dtype_code, msg.dtype_bits, msg.dtype_lanes) == (2, 32, 1)
    copied = numpy.from_dlpack(
        Consumer(dlpack_conversions.from_input_tensor_msg(msg)))
    assert numpy.array_equal(copied, source)


def test_to_tensor_msg_copies_into_an_existing_message():
    source = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    destination = dlpack_conversions.allocate_tensor_msg(
        (2, 3), (2, 32, 1), 'cpu')

    returned = dlpack_conversions.to_tensor_msg(
        destination, source.__dlpack__())

    assert returned is destination
    copied = numpy.from_dlpack(
        Consumer(dlpack_conversions.from_input_tensor_msg(destination)))
    assert numpy.array_equal(copied, source)


def test_to_tensor_msg_rejects_a_source_that_does_not_fit():
    source = numpy.arange(6, dtype=numpy.float32)
    destination = dlpack_conversions.allocate_tensor_msg(
        (2,), (2, 32, 1), 'cpu')

    with pytest.raises(ValueError, match='Source tensor needs 24 bytes'):
        dlpack_conversions.to_tensor_msg(destination, source.__dlpack__())


def test_to_tensor_msg_rejects_a_non_contiguous_source():
    source = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)[:, ::2]

    with pytest.raises(ValueError, match='non-contiguous'):
        dlpack_conversions.to_tensor_msg(source.__dlpack__())


def test_to_tensor_msg_rejects_a_destination_that_is_not_a_message():
    source = numpy.arange(2, dtype=numpy.float32)

    with pytest.raises(TypeError, match='must be an ExperimentalTensor'):
        dlpack_conversions.to_tensor_msg(object(), source.__dlpack__())
