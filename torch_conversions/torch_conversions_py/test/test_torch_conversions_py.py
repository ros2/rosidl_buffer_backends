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
import subprocess
import sys


import pytest

from tensor_msgs.msg import ExperimentalTensor

import torch

import torch_conversions
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_input_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import to_tensor_msg


CUDA_AVAILABLE = torch_conversions.backend_available('cuda')


def test_cpu_conversion_works_in_a_fresh_interpreter():
    subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import torch, torch_conversions; '
                'msg = torch_conversions.allocate_tensor_msg('
                "(1,), torch.uint8, 'cpu'); "
                'assert len(msg.data) == 1'
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    'dtype,expected',
    [
        (torch.uint8, (1, 8, 1)),
        (torch.int8, (0, 8, 1)),
        (torch.int16, (0, 16, 1)),
        (torch.int32, (0, 32, 1)),
        (torch.int64, (0, 64, 1)),
        (torch.float16, (2, 16, 1)),
        (torch.bfloat16, (4, 16, 1)),
        (torch.float32, (2, 32, 1)),
        (torch.float64, (2, 64, 1)),
        (torch.bool, (6, 8, 1)),
    ],
)
def test_allocate_cpu_metadata(dtype, expected):
    msg = allocate_tensor_msg((2, 3, 4), dtype, 'cpu')

    assert list(msg.shape) == [2, 3, 4]
    assert list(msg.strides) == [12, 4, 1]
    assert (msg.dtype_code, msg.dtype_bits, msg.dtype_lanes) == expected
    assert msg.byte_offset == 0
    assert len(msg.data) == 24 * torch.empty((), dtype=dtype).element_size()


def test_cpu_write_read_round_trip():
    msg = allocate_tensor_msg((4,), torch.int32, 'cpu')
    output = from_output_tensor_msg(msg)
    output.copy_(torch.tensor([10, 20, 30, 40], dtype=torch.int32))

    result = from_input_tensor_msg(msg, clone=False)

    expected = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    assert torch.equal(result, expected)


def test_input_clone_is_independent_and_zero_copy_view_is_shared():
    source = torch.arange(6, dtype=torch.float32)
    msg = to_tensor_msg(source)

    clone = from_input_tensor_msg(msg)
    view = from_input_tensor_msg(msg, clone=False)
    view[0] = 99

    assert clone[0].item() == 0
    assert from_input_tensor_msg(msg, clone=False)[0].item() == 99


def test_copy_into_existing_message_updates_metadata():
    msg = allocate_tensor_msg((16,), torch.float32, 'cpu')
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    returned = to_tensor_msg(msg, source)

    assert returned is msg
    assert list(msg.shape) == [2, 3]
    assert list(msg.strides) == [3, 1]
    assert msg.byte_offset == 0
    assert torch.equal(from_input_tensor_msg(msg), source)


def test_copy_allocates_new_message():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    msg = to_tensor_msg(source)

    assert list(msg.shape) == [2, 3]
    assert torch.equal(from_input_tensor_msg(msg), source)


def test_byte_offset_selects_storage_subregion():
    msg = allocate_tensor_msg((16,), torch.int32, 'cpu')
    output = from_output_tensor_msg(msg)
    output.copy_(torch.arange(16, dtype=torch.int32) * 100)
    msg.shape = [4]
    msg.strides = [1]
    msg.byte_offset = 4 * torch.empty((), dtype=torch.int32).element_size()

    view = from_input_tensor_msg(msg, clone=False)

    expected = torch.tensor([400, 500, 600, 700], dtype=torch.int32)
    assert torch.equal(view, expected)


def test_empty_buffer_returns_none():
    msg = ExperimentalTensor()

    assert from_input_tensor_msg(msg) is None
    assert from_output_tensor_msg(msg) is None


def test_oversized_tensor_is_rejected():
    msg = allocate_tensor_msg((4,), torch.uint8, 'cpu')

    with pytest.raises(ValueError, match='exceeds allocated message storage'):
        to_tensor_msg(msg, torch.zeros(128, dtype=torch.uint8))


def test_invalid_shape_and_strides_are_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        allocate_tensor_msg((-1,), torch.float32, 'cpu')

    msg = allocate_tensor_msg((2, 2), torch.float32, 'cpu')
    msg.strides = [1]
    with pytest.raises(ValueError, match='match shape rank'):
        from_input_tensor_msg(msg)

    msg.strides = [1, -1]
    with pytest.raises(ValueError, match='Negative shape dimensions and strides'):
        from_input_tensor_msg(msg)


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


def test_default_allocation_is_usable_by_this_torch_build():
    msg = allocate_tensor_msg((4,), torch.float32)

    tensor = from_output_tensor_msg(msg)

    assert tensor is not None
    tensor.fill_(1.0)


@pytest.mark.skipif(CUDA_AVAILABLE, reason='CUDA support is available')
def test_cpu_only_configuration_defaults_to_cpu_and_rejects_cuda():
    msg = allocate_tensor_msg((4,), torch.float32)

    assert isinstance(msg.data, array)
    assert torch_conversions.default_backend() == 'cpu'
    with pytest.raises(RuntimeError, match='No Torch conversion plugin serves device'):
        allocate_tensor_msg((4,), torch.float32, 'cuda')
