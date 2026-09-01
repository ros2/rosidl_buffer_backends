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
import gc
from pathlib import Path
import weakref

from cuda_buffer import CudaBuffer
import pytest

import torch

import torch_conversions
from torch_conversions import _dlpack_bridge
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_input_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import set_stream
from torch_conversions import to_tensor_msg


CUDA_AVAILABLE = torch_conversions._adapter_available('cuda')


def test_platform_neutral_bridge_retains_owner_and_strides():
    class Owner:

        def __init__(self):
            self.data = (ctypes.c_float * 8)(*range(8))

    owner = Owner()
    owner_ref = weakref.ref(owner)
    capsule = _dlpack_bridge.make_dlpack_capsule(
        ctypes.addressof(owner.data),
        1,
        0,
        2,
        32,
        1,
        [2, 2],
        [4, 2],
        0,
        owner,
    )
    tensor = torch.utils.dlpack.from_dlpack(capsule)
    del capsule
    del owner
    gc.collect()

    assert owner_ref() is not None
    assert torch.equal(tensor, torch.tensor([[0.0, 2.0], [4.0, 6.0]]))
    tensor[0, 0] = 42
    assert owner_ref().data[0] == 42

    del tensor
    gc.collect()
    assert owner_ref() is None


def test_bridge_build_has_no_cuda_or_native_buffer_dependency():
    package_root = Path(__file__).parents[1]
    source = (package_root / 'src/dlpack_bridge.cpp').read_text()
    cmake = (package_root / 'CMakeLists.txt').read_text()
    manifest = (package_root / 'package.xml').read_text()

    assert '#include <ATen/' not in source
    assert '#include <cuda' not in source
    assert '#include "cuda_buffer/' not in source
    assert '#include "rosidl_buffer/' not in source
    assert 'find_package(cuda_buffer' not in cmake
    assert 'find_package(libtorch_vendor' not in cmake
    assert 'find_package(rosidl_buffer' not in cmake
    assert 'cuda_buffer::' not in cmake
    assert 'rosidl_buffer::' not in cmake
    assert '<build_depend>libtorch_vendor</build_depend>' not in manifest
    assert '<exec_depend>cuda_buffer_py</exec_depend>' not in manifest


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

    stream = torch.cuda.current_stream().cuda_stream
    with CudaBuffer.from_input_buffer(msg.data, stream) as handle:
        expected_pointer = handle.device_ptr
        expected_device = handle.device_id

    view = from_input_tensor_msg(msg, clone=False)

    assert view.data_ptr() == output_pointer
    assert view.data_ptr() == expected_pointer
    assert view.device.index == expected_device
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
