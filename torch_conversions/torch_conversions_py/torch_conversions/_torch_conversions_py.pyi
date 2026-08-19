from typing import Sequence

from rosidl_buffer import Buffer


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
