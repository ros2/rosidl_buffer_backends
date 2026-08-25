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

from importlib.metadata import entry_points
import subprocess
import sys

import pytest

import torch

import torch_conversions
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_input_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import set_stream
from torch_conversions import to_tensor_msg


CUDA_AVAILABLE = torch_conversions._adapter_available('cuda')


def test_cuda_adapter_entry_point_is_discoverable():
    adapters = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group='torch_conversions.adapters')
    }

    assert adapters['cuda'] == 'torch_conversions_cuda_plugin:register'


def test_plugin_can_be_imported_before_core():
    subprocess.run(
        [
            sys.executable,
            '-c',
            'import torch_conversions_cuda_plugin; import torch_conversions',
        ],
        check=True,
    )


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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA support is unavailable')
def test_cuda_zero_copy_view_aliases_message_storage():
    msg = allocate_tensor_msg((8,), torch.float32, 'cuda')
    output = from_output_tensor_msg(msg)
    output.copy_(torch.arange(8, dtype=torch.float32, device='cuda'))
    output_pointer = output.data_ptr()
    del output

    view = from_input_tensor_msg(msg, clone=False)

    assert view.data_ptr() == output_pointer
    assert torch.equal(view.cpu(), torch.arange(8, dtype=torch.float32))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA support is unavailable')
def test_set_stream_selects_non_default_cuda_stream():
    default_stream = torch.cuda.current_stream().cuda_stream

    with set_stream('cuda'):
        selected_stream = torch.cuda.current_stream().cuda_stream
        msg = allocate_tensor_msg((1,), torch.float32, 'cuda')
        output = from_output_tensor_msg(msg)
        output.fill_(1)

    assert selected_stream != default_stream
