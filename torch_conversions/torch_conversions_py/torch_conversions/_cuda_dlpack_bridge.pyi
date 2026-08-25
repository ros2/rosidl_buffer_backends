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

from typing import Sequence

from rosidl_buffer import Buffer


def _cuda_buffer_available() -> bool: ...

def _from_input_dlpack(
    buffer: Buffer,
    shape: Sequence[int],
    strides: Sequence[int],
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    byte_offset: int,
    stream: int,
) -> object: ...

def _from_output_dlpack(
    buffer: Buffer,
    shape: Sequence[int],
    strides: Sequence[int],
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    byte_offset: int,
    stream: int,
) -> object: ...
