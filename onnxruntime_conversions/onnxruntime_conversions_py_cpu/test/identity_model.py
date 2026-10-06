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

"""Encode test models without loading ONNX's duplicate protobuf descriptors."""

from typing import Sequence


class ElementType:
    """ONNX tensor element types, as numbered by the ONNX specification."""

    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    BFLOAT16 = 16


_VARINT = 0
_LENGTH_DELIMITED = 2


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        encoded.append(chunk | 0x80 if value else chunk)
        if not value:
            return bytes(encoded)


def _tag(field: int, wire_type: int) -> bytes:
    return _varint(field << 3 | wire_type)


def _number(field: int, value: int) -> bytes:
    return _tag(field, _VARINT) + _varint(value)


def _bytes(field: int, payload: bytes) -> bytes:
    return _tag(field, _LENGTH_DELIMITED) + _varint(len(payload)) + payload


def _text(field: int, value: str) -> bytes:
    return _bytes(field, value.encode())


def _value_info(name: str, element_type: int, shape: Sequence[int]) -> bytes:
    dims = b''.join(_bytes(1, _number(1, int(dim))) for dim in shape)
    tensor_type = _number(1, element_type) + _bytes(2, dims)
    return _text(1, name) + _bytes(2, _bytes(1, tensor_type))


def identity_model(
    shape: Sequence[int],
    element_type: int = ElementType.FLOAT,
) -> bytes:
    """Serialize a ModelProto holding one Identity node over ``shape``."""
    node = _text(1, 'input') + _text(2, 'output') + _text(4, 'Identity')
    return _model(node, shape, element_type)


def matmul_model() -> bytes:
    """Serialize MatMul(input, input) for a float32 [2, 2] tensor."""
    node = _text(1, 'input') * 2 + _text(2, 'output') + _text(4, 'MatMul')
    return _model(node, (2, 2), ElementType.FLOAT)


def _model(node: bytes, shape: Sequence[int], element_type: int) -> bytes:
    graph = (
        _bytes(1, node)
        + _text(2, 'identity')
        + _bytes(11, _value_info('input', element_type, shape))
        + _bytes(12, _value_info('output', element_type, shape))
    )
    opset_import = _text(1, '') + _number(2, 18)
    return _number(1, 10) + _bytes(7, graph) + _bytes(8, opset_import)
