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
import torch_conversions
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_input_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import to_tensor_msg


CUDA_AVAILABLE = torch_conversions._cuda_available()


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

    assert torch.equal(result, torch.tensor([10, 20, 30, 40], dtype=torch.int32))


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

    assert torch.equal(view, torch.tensor([400, 500, 600, 700], dtype=torch.int32))


def test_empty_buffer_returns_none():
    msg = ExperimentalTensor()

    assert from_input_tensor_msg(msg) is None
    assert from_output_tensor_msg(msg) is None


def test_oversized_tensor_is_rejected():
    msg = allocate_tensor_msg((4,), torch.uint8, 'cpu')

    with pytest.raises(ValueError, match='buffer has 4'):
        to_tensor_msg(msg, torch.zeros(128, dtype=torch.uint8))


def test_invalid_metadata_is_rejected():
    msg = ExperimentalTensor()
    msg.dtype_code = 2
    msg.dtype_bits = 128
    msg.dtype_lanes = 1
    msg.shape = [1]
    msg.strides = [1]
    msg.data = array('B', bytes(16))

    with pytest.raises(TypeError, match='Unsupported DLPack dtype'):
        from_input_tensor_msg(msg)


def test_invalid_shape_and_strides_are_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        allocate_tensor_msg((-1,), torch.float32, 'cpu')

    msg = allocate_tensor_msg((2, 2), torch.float32, 'cpu')
    msg.strides = [1]
    with pytest.raises(ValueError, match='match the shape rank'):
        from_input_tensor_msg(msg)


def test_unsupported_torch_dtype_is_rejected():
    with pytest.raises(TypeError, match='Unsupported torch dtype'):
        to_tensor_msg(torch.zeros(4, dtype=torch.complex64))


@pytest.mark.skipif(CUDA_AVAILABLE, reason='CUDA support is available')
def test_cpu_only_configuration_defaults_to_cpu_and_rejects_cuda():
    msg = allocate_tensor_msg((4,), torch.float32)

    assert isinstance(msg.data, array)
    with pytest.raises(RuntimeError, match='CUDA'):
        allocate_tensor_msg((4,), torch.float32, 'cuda')


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA support is unavailable')
def test_cuda_write_read_round_trip():
    msg = allocate_tensor_msg((6,), torch.float32, 'cuda')
    output = from_output_tensor_msg(msg)
    output.copy_(torch.arange(6, dtype=torch.float32, device='cuda'))

    result = from_input_tensor_msg(msg)

    assert result.is_cuda
    assert torch.equal(result.cpu(), torch.arange(6, dtype=torch.float32))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA support is unavailable')
def test_cuda_tensor_to_message_round_trip():
    source = torch.arange(12, dtype=torch.int32, device='cuda').reshape(3, 4)

    msg = to_tensor_msg(source)
    result = from_input_tensor_msg(msg, clone=False)

    assert msg.data.backend_type == 'cuda'
    assert result.is_cuda
    assert torch.equal(result.cpu(), source.cpu())
