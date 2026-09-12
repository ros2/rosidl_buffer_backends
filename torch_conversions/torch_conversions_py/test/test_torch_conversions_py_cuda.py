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

import gc

import pytest

import torch

import torch_conversions
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_input_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import set_stream
from torch_conversions import to_tensor_msg


CUDA_AVAILABLE = (
    torch_conversions.backend_available('cuda') and torch.cuda.is_available()
)

pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason='the CUDA conversion plugin or CUDA device is unavailable',
)


def test_cuda_write_read_round_trip():
    msg = allocate_tensor_msg((6,), torch.float32, 'cuda')
    output = from_output_tensor_msg(msg)
    output.copy_(torch.arange(6, dtype=torch.float32, device='cuda'))

    result = from_input_tensor_msg(msg)

    assert result.is_cuda
    assert torch.equal(result.cpu(), torch.arange(6, dtype=torch.float32))


def test_cuda_tensor_to_message_round_trip():
    source = torch.arange(12, dtype=torch.int32, device='cuda').reshape(3, 4)

    msg = to_tensor_msg(source)
    result = from_input_tensor_msg(msg, clone=False)

    assert msg.data.backend_type == 'cuda'
    assert result.is_cuda
    assert torch.equal(result.cpu(), source.cpu())


def test_cuda_zero_copy_view_aliases_message_storage():
    cuda_buffer = pytest.importorskip('cuda_buffer').CudaBuffer
    msg = allocate_tensor_msg((8,), torch.float32, 'cuda')
    output = from_output_tensor_msg(msg)
    output.copy_(torch.arange(8, dtype=torch.float32, device='cuda'))
    output_pointer = output.data_ptr()
    del output

    stream = torch.cuda.current_stream().cuda_stream
    with cuda_buffer.from_input_buffer(msg.data, stream) as handle:
        expected_pointer = handle.device_ptr
        expected_device = handle.device_id

    view = from_input_tensor_msg(msg, clone=False)

    assert view.data_ptr() == output_pointer
    assert view.data_ptr() == expected_pointer
    assert view.device.index == expected_device
    assert torch.equal(view.cpu(), torch.arange(8, dtype=torch.float32))


def test_cuda_byte_offset_selects_storage_subregion():
    msg = allocate_tensor_msg((16,), torch.int32, 'cuda')
    output = from_output_tensor_msg(msg)
    output.copy_(torch.arange(16, dtype=torch.int32, device='cuda') * 100)
    msg.shape = [4]
    msg.strides = [1]
    msg.byte_offset = 16

    view = from_input_tensor_msg(msg, clone=False)

    expected = torch.tensor([400, 500, 600, 700], dtype=torch.int32)
    assert torch.equal(view.cpu(), expected)


def test_cuda_is_the_default_backend_when_installed():
    msg = allocate_tensor_msg((4,), torch.float32)

    assert msg.data.backend_type == 'cuda'


def test_set_stream_selects_non_default_cuda_stream():
    default_stream = torch.cuda.current_stream().cuda_stream

    with set_stream('cuda'):
        selected_stream = torch.cuda.current_stream().cuda_stream
        msg = allocate_tensor_msg((1,), torch.float32, 'cuda')
        output = from_output_tensor_msg(msg)
        output.fill_(1)

    assert selected_stream != default_stream


def test_conversions_use_the_active_torch_stream():
    with set_stream('cuda'):
        msg = allocate_tensor_msg((4,), torch.float32, 'cuda')
        from_output_tensor_msg(msg).fill_(2)
        torch.cuda.current_stream().synchronize()

    assert torch.equal(
        from_input_tensor_msg(msg).cpu(), torch.full((4,), 2.0)
    )


def test_clone_waits_on_the_supplied_consumer_stream():
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    msg = allocate_tensor_msg((4,), torch.float32, 'cuda')
    with torch.cuda.stream(producer):
        output = from_output_tensor_msg(msg, stream=producer.cuda_stream)
        output.zero_()
        producer.synchronize()
        torch.cuda._sleep(200_000_000)
        output.fill_(7)
        del output

    result = from_input_tensor_msg(msg, stream=consumer.cuda_stream)
    consumer.synchronize()
    assert torch.equal(result.cpu(), torch.full((4,), 7.0))


def test_non_contiguous_copy_uses_the_supplied_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.zeros((2, 3), device='cuda')
        stream.synchronize()
        torch.cuda._sleep(200_000_000)
        source.fill_(5)
    msg = to_tensor_msg(source.T, stream=stream.cuda_stream)
    del source
    result = from_input_tensor_msg(msg, stream=stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(result.cpu(), torch.full((3, 2), 5.0))


@pytest.mark.parametrize('output_view', [False, True])
def test_cuda_view_retains_storage_after_message_destruction(output_view):
    with set_stream('cuda'):
        msg = allocate_tensor_msg((4,), torch.float32, 'cuda')
        value = from_output_tensor_msg(msg)
        value.fill_(9)
        pointer = value.data_ptr()
        if not output_view:
            del value
            value = from_input_tensor_msg(msg, clone=False)
        del msg
        gc.collect()
        assert value.data_ptr() == pointer
        assert torch.equal(value.cpu(), torch.full((4,), 9.0))


def test_device_index_is_preserved_and_validated():
    count = torch.cuda.device_count()
    current = torch.cuda.current_device()
    assert len(allocate_tensor_msg((0,), torch.float32, 'cuda').data) == 0
    allocate_tensor_msg((1,), torch.float32, 'cuda')
    with pytest.raises(RuntimeError):
        allocate_tensor_msg((1,), torch.float32, f'cuda:{count}')
    for device in range(count):
        if device != current:
            with pytest.raises(RuntimeError, match='CUDA buffer is on device'):
                allocate_tensor_msg((1,), torch.float32, f'cuda:{device}')
            continue
        msg = allocate_tensor_msg((1,), torch.float32, f'cuda:{device}')
        with torch.cuda.device(device):
            value = from_output_tensor_msg(msg)
            assert value.device.index == device
