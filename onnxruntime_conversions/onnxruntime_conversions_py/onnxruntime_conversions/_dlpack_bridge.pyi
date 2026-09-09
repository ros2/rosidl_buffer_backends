from typing import Sequence


def make_dlpack_capsule(
    data: int,
    device_type: int,
    device_id: int,
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    shape: Sequence[int],
    byte_offset: int,
    owner: object,
) -> object: ...
