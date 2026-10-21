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

"""Discover endpoint buffer metadata through an optional Fast DDS ABI."""

from collections.abc import Mapping
import ctypes
from ctypes.util import find_library
import sys
from typing import Optional
from typing import Protocol
from typing import TypeAlias

from rclpy.utilities import get_rmw_implementation_identifier


BackendMetadata: TypeAlias = Optional[Mapping[str, str]]

_RMW_FASTRTPS_CPP = 'rmw_fastrtps_cpp'
_QUERY_SYMBOL = 'rmw_fastrtps_cpp_get_endpoint_buffer_backend_metadata'
_GID_SIZE = 16

_STATUS_SUCCESS = 0
_STATUS_INVALID_ARGUMENT = 1
_STATUS_INCORRECT_RMW_IMPLEMENTATION = 2
_STATUS_ENDPOINT_NOT_FOUND = 3
_STATUS_BUFFER_TOO_SMALL = 4
_STATUS_NOT_AVAILABLE = 5


class BackendMetadataUnavailable(NotImplementedError):
    """Indicate that backend metadata cannot be queried safely."""


class TopicEndpoint(Protocol):
    """The TopicEndpointInfo fields used to identify an endpoint."""

    endpoint_gid: list[int]
    endpoint_type: int


def _library_candidates(name):
    discovered = find_library(name)
    candidates = [discovered] if discovered else []
    if sys.platform == 'win32':
        candidates.append(f'{name}.dll')
    elif sys.platform == 'darwin':
        candidates.append(f'lib{name}.dylib')
    else:
        candidates.append(f'lib{name}.so')
    return tuple(dict.fromkeys(candidates))


def _load_library(name):
    errors = []
    for candidate in _library_candidates(name):
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(str(error))
    raise OSError('; '.join(errors))


def _load_native_functions(library_loader=_load_library):
    try:
        rcl_library = library_loader('rcl')
    except OSError as error:
        raise BackendMetadataUnavailable(
            'the native rcl library could not be loaded'
        ) from error

    try:
        get_rmw_handle = rcl_library.rcl_node_get_rmw_handle
    except AttributeError as error:
        raise BackendMetadataUnavailable(
            'the installed rcl library does not expose rcl_node_get_rmw_handle'
        ) from error
    get_rmw_handle.argtypes = [ctypes.c_void_p]
    get_rmw_handle.restype = ctypes.c_void_p

    try:
        rmw_library = library_loader(_RMW_FASTRTPS_CPP)
    except OSError as error:
        raise BackendMetadataUnavailable(
            'incompatible rmw_fastrtps_cpp: its native library could not be loaded'
        ) from error

    try:
        query_metadata = getattr(rmw_library, _QUERY_SYMBOL)
    except AttributeError as error:
        raise BackendMetadataUnavailable(
            'incompatible rmw_fastrtps_cpp: the installed version does not '
            f'expose {_QUERY_SYMBOL}'
        ) from error
    query_metadata.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    query_metadata.restype = ctypes.c_int
    return get_rmw_handle, query_metadata


def _raise_for_status(status):
    if status == _STATUS_SUCCESS:
        return
    if status == _STATUS_ENDPOINT_NOT_FOUND:
        reason = 'this endpoint did not advertise Fast DDS buffer metadata'
    elif status == _STATUS_INCORRECT_RMW_IMPLEMENTATION:
        reason = 'the query rejected the active RMW implementation'
    elif status == _STATUS_INVALID_ARGUMENT:
        reason = 'the Fast DDS query rejected the endpoint information'
    elif status == _STATUS_BUFFER_TOO_SMALL:
        reason = 'the Fast DDS metadata changed while it was being queried'
    elif status == _STATUS_NOT_AVAILABLE:
        reason = 'the Fast DDS endpoint metadata registry is unavailable'
    else:
        reason = f'the Fast DDS query returned unknown status {status}'
    raise BackendMetadataUnavailable(reason)


def _decode_metadata(output, entry_count):
    parts = bytes(output).split(b'\0')
    if len(parts) != 2 * entry_count + 1 or parts[-1]:
        raise BackendMetadataUnavailable(
            'the Fast DDS query returned malformed endpoint metadata'
        )

    try:
        decoded = [part.decode('utf-8') for part in parts[:-1]]
    except UnicodeDecodeError as error:
        raise BackendMetadataUnavailable(
            'the Fast DDS query returned non-UTF-8 endpoint metadata'
        ) from error
    return dict(zip(decoded[::2], decoded[1::2]))


def _query_metadata(rmw_node, endpoint, query_metadata):
    gid_values = tuple(endpoint.endpoint_gid)
    if len(gid_values) != _GID_SIZE or any(
        value < 0 or value > 255 for value in gid_values
    ):
        raise BackendMetadataUnavailable(
            f'the endpoint GID is not {_GID_SIZE} bytes'
        )

    endpoint_type = int(endpoint.endpoint_type)
    if endpoint_type not in (1, 2):
        raise BackendMetadataUnavailable(
            'the graph entry is not a publisher or subscription'
        )

    gid = (ctypes.c_uint8 * _GID_SIZE)(*gid_values)
    required_size = ctypes.c_size_t()
    entry_count = ctypes.c_size_t()
    status = query_metadata(
        rmw_node,
        gid,
        endpoint_type,
        None,
        0,
        ctypes.byref(required_size),
        ctypes.byref(entry_count),
    )
    _raise_for_status(status)
    if required_size.value == 0:
        if entry_count.value != 0:
            raise BackendMetadataUnavailable(
                'the Fast DDS query returned inconsistent endpoint metadata'
            )
        return {}

    output = (ctypes.c_uint8 * required_size.value)()
    status = query_metadata(
        rmw_node,
        gid,
        endpoint_type,
        output,
        len(output),
        ctypes.byref(required_size),
        ctypes.byref(entry_count),
    )
    _raise_for_status(status)
    if required_size.value != len(output):
        raise BackendMetadataUnavailable(
            'the Fast DDS metadata size changed while it was being queried'
        )
    return _decode_metadata(output, entry_count.value)


def discover_endpoint_backend_metadata(
    node: object,
    endpoint: TopicEndpoint,
) -> BackendMetadata:
    """Return the backend metadata advertised by one Fast DDS endpoint."""
    rmw_implementation = get_rmw_implementation_identifier()
    if rmw_implementation != _RMW_FASTRTPS_CPP:
        raise BackendMetadataUnavailable(
            f"RMW implementation '{rmw_implementation}' is not supported by "
            'this ros2buffer version'
        )

    get_rmw_handle, query_metadata = _load_native_functions()
    try:
        handle = node.handle
        with handle:
            rmw_node = get_rmw_handle(ctypes.c_void_p(handle.pointer))
            if not rmw_node:
                raise BackendMetadataUnavailable(
                    'the CLI node does not have a valid RMW handle'
                )
            return _query_metadata(rmw_node, endpoint, query_metadata)
    except BackendMetadataUnavailable:
        raise
    except (AttributeError, TypeError) as error:
        raise BackendMetadataUnavailable(
            'the CLI node does not expose a compatible native handle'
        ) from error
