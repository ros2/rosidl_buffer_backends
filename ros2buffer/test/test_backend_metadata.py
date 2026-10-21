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
from types import SimpleNamespace

import pytest
from ros2buffer.api import BackendMetadataUnavailable
from ros2buffer.api import discover_endpoint_backend_metadata
import ros2buffer.api.backend_metadata as backend_metadata


class FakeHandle:

    pointer = 0x1234

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class FakeSymbol:

    def __call__(self, *args):
        del args


def make_endpoint(endpoint_type=1):
    return SimpleNamespace(
        endpoint_gid=list(range(16)),
        endpoint_type=endpoint_type,
    )


def make_native_functions(metadata, *, status=0):
    payload = b''.join(
        name.encode() + b'\0' + value.encode() + b'\0'
        for name, value in sorted(metadata.items())
    )

    def get_rmw_handle(rcl_node):
        assert rcl_node.value == FakeHandle.pointer
        return 0x5678

    def query_metadata(
        rmw_node,
        gid,
        endpoint_type,
        output,
        output_capacity,
        required_size,
        entry_count,
    ):
        assert rmw_node == 0x5678
        assert list(gid) == list(range(16))
        assert endpoint_type in (1, 2)
        required_size._obj.value = len(payload)
        entry_count._obj.value = len(metadata)
        if status or output is None:
            return status
        assert output_capacity == len(payload)
        ctypes.memmove(output, payload, len(payload))
        return 0

    return get_rmw_handle, query_metadata


def test_discovers_fastdds_endpoint_metadata(monkeypatch):
    monkeypatch.setattr(
        backend_metadata,
        'get_rmw_implementation_identifier',
        lambda: 'rmw_fastrtps_cpp',
    )
    monkeypatch.setattr(
        backend_metadata,
        '_load_native_functions',
        lambda: make_native_functions({'cuda': 'device=0', 'cpu': ''}),
    )
    node = SimpleNamespace(handle=FakeHandle())

    assert discover_endpoint_backend_metadata(node, make_endpoint()) == {
        'cpu': '',
        'cuda': 'device=0',
    }


def test_reports_legacy_fastdds_as_incompatible(monkeypatch):
    rcl_library = SimpleNamespace(rcl_node_get_rmw_handle=FakeSymbol())
    old_rmw_library = SimpleNamespace()

    def load_library(name):
        return {
            'rcl': rcl_library,
            'rmw_fastrtps_cpp': old_rmw_library,
        }[name]

    with pytest.raises(
        BackendMetadataUnavailable,
        match='incompatible rmw_fastrtps_cpp.*does not expose',
    ):
        backend_metadata._load_native_functions(load_library)


def test_reports_other_rmw_as_unsupported(monkeypatch):
    monkeypatch.setattr(
        backend_metadata,
        'get_rmw_implementation_identifier',
        lambda: 'rmw_zenoh_cpp',
    )

    with pytest.raises(
        BackendMetadataUnavailable,
        match="RMW implementation 'rmw_zenoh_cpp' is not supported",
    ):
        discover_endpoint_backend_metadata(
            SimpleNamespace(handle=FakeHandle()),
            make_endpoint(),
        )


def test_reports_endpoint_without_advertised_metadata(monkeypatch):
    monkeypatch.setattr(
        backend_metadata,
        'get_rmw_implementation_identifier',
        lambda: 'rmw_fastrtps_cpp',
    )
    monkeypatch.setattr(
        backend_metadata,
        '_load_native_functions',
        lambda: make_native_functions({}, status=3),
    )

    with pytest.raises(
        BackendMetadataUnavailable,
        match='did not advertise Fast DDS buffer metadata',
    ):
        discover_endpoint_backend_metadata(
            SimpleNamespace(handle=FakeHandle()),
            make_endpoint(endpoint_type=2),
        )
