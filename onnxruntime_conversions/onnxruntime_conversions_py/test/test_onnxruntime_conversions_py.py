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

import numpy as np

import onnxruntime as ort

from onnxruntime_conversions import _core
from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import to_tensor_msg
from onnxruntime_conversions._plugin import ConversionRegistry

import pytest


def test_registry_distinguishes_accelerators_and_preserves_device_index(monkeypatch):
    class Plugin:
        priority = 100

        def __init__(self, backend, device_type):
            self.backends = (backend,)
            self.device_types = (device_type,)
            self.allocated_device = None

        def is_available(self):
            return True

        def allocate(self, byte_count, backend, device_id):
            self.allocated_device = device_id
            return array('B', [0]) * byte_count

    registry = ConversionRegistry()
    cuda = Plugin('cuda', 2)
    rocm = Plugin('rocm', 10)
    registry.register(cuda)
    registry.register(rocm)
    assert registry.for_device(2) is cuda
    assert registry.for_device(10) is rocm
    monkeypatch.setattr(_core, '_REGISTRY', registry)

    allocate_tensor_msg((4,), np.float32, 'rocm', device_id=3)
    assert rocm.allocated_device == 3
    assert cuda.allocated_device is None


@pytest.mark.parametrize('element_type', [8, np.complex64])
def test_unsupported_element_types_are_rejected(element_type):
    with pytest.raises(ValueError, match='Unsupported ONNX tensor element'):
        allocate_tensor_msg((1,), element_type, 'cpu')


def test_to_tensor_msg_rejects_non_tensor_arguments():
    with pytest.raises(TypeError, match='tensor OrtValue'):
        to_tensor_msg(np.arange(4, dtype=np.float32))
    with pytest.raises(TypeError, match='ExperimentalTensor'):
        to_tensor_msg(
            'not a message',
            ort.OrtValue.ortvalue_from_numpy(np.zeros(1, dtype=np.float32)),
        )
