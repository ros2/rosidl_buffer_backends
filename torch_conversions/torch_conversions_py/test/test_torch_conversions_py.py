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

import pytest

from tensor_msgs.msg import ExperimentalTensor

import torch

from torch_conversions import allocate_tensor_msg
from torch_conversions import from_input_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import to_tensor_msg


def test_empty_buffer_returns_none():
    msg = ExperimentalTensor()

    assert from_input_tensor_msg(msg) is None
    assert from_output_tensor_msg(msg) is None


def test_unsupported_torch_dtype_is_rejected():
    with pytest.raises(TypeError, match='Unsupported torch dtype'):
        to_tensor_msg(torch.zeros(4, dtype=torch.complex64))
    with pytest.raises(TypeError, match='Unsupported torch dtype'):
        allocate_tensor_msg((4,), torch.complex64, 'cpu')


def test_unsupported_device_is_rejected():
    with pytest.raises(RuntimeError, match='No Torch conversion plugin serves device'):
        allocate_tensor_msg((4,), torch.float32, 'meta')


def test_new_device_plugin_receives_the_complete_device(monkeypatch):
    from torch_conversions import _core
    from torch_conversions._plugin import ConversionRegistry

    class Plugin:
        backends = ('test_xpu',)
        device_types = ('xpu',)
        priority = 0

        def is_available(self):
            return True

        def allocate(self, byte_count, backend, device):
            assert backend == 'test_xpu'
            assert device == torch.device('xpu:3')
            return array('B', bytes(byte_count))

    registry = ConversionRegistry()
    registry.register(Plugin())
    monkeypatch.setattr(_core, '_REGISTRY', registry)
    msg = allocate_tensor_msg((2,), torch.float32, 'xpu:3')
    assert len(msg.data) == 8
