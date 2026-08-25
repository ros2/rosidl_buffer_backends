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

import importlib.util
import os
import subprocess
import sys


def test_bridge_has_no_direct_cuda_buffer_dependency():
    spec = importlib.util.find_spec(
        'torch_conversions._cuda_dlpack_bridge'
    )
    assert spec is not None
    assert spec.origin is not None

    result = subprocess.run(
        ['readelf', '-d', spec.origin],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'libcuda_buffer.so' not in result.stdout


def test_cpu_conversion_works_without_cuda_buffer_library():
    script = """
from array import array

import torch

import torch_conversions

assert not torch_conversions._cuda_available()
msg = torch_conversions.allocate_tensor_msg((4,), torch.float32)
assert isinstance(msg.data, array)
output = torch_conversions.from_output_tensor_msg(msg)
output.fill_(3)
assert torch.equal(
    torch_conversions.from_input_tensor_msg(msg),
    torch.full((4,), 3, dtype=torch.float32),
)
"""
    environment = os.environ.copy()
    environment['CUDA_BUFFER_LIBRARY_PATH'] = (
        '/path/that/does/not/exist/libcuda_buffer.so'
    )

    subprocess.run(
        [sys.executable, '-c', script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
