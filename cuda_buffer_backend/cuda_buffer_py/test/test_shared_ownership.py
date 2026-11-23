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
import sys

import _cuda_buffer_test_utils as native
import pytest
from rosidl_buffer import _get_buffer_ptr
import rosidl_buffer._rosidl_buffer_py as bindings


def test_successful_handoff():
    count = native.DestructionCount()
    buffer = native.make_tracked_buffer(count)
    assert count.value == 0
    assert buffer.backend_type == 'cpu'
    assert buffer.to_bytes() == b'\x01\x02\x03\x04'
    del buffer
    gc.collect()
    assert count.value == 1


def test_import_failure(monkeypatch):
    count = native.DestructionCount()
    monkeypatch.setitem(sys.modules, 'rosidl_buffer._rosidl_buffer_py', None)
    with pytest.raises(ImportError):
        native.make_tracked_buffer(count)
    gc.collect()
    assert count.value == 1


def test_missing_helper_has_dependency_error(monkeypatch):
    count = native.DestructionCount()
    monkeypatch.delattr(bindings, '_share_buffer_from_capsule')
    message = 'requires rosidl_buffer_py with _share_buffer_from_capsule'
    with pytest.raises(ImportError, match=message):
        native.make_tracked_buffer(count)
    gc.collect()
    assert count.value == 1


def test_failure_during_call_setup(monkeypatch):
    count = native.DestructionCount()
    # Python call dispatch fails before any receiver can run.
    monkeypatch.setattr(bindings, '_share_buffer_from_capsule', None)
    with pytest.raises(TypeError):
        native.make_tracked_buffer(count)
    gc.collect()
    assert count.value == 1


@pytest.mark.parametrize('share_first', [False, True])
@pytest.mark.parametrize('retain_wrapper', [False, True])
def test_receiver_exception(monkeypatch, share_first, retain_wrapper):
    count = native.DestructionCount()
    original = bindings._share_buffer_from_capsule
    retained = []

    def failing_receiver(capsule):
        if share_first:
            wrapper = original(capsule)
            assert wrapper.to_bytes() == b'\x01\x02\x03\x04'
            if retain_wrapper:
                retained.append(wrapper)
            del wrapper
        raise RuntimeError('receiver failed')

    monkeypatch.setattr(bindings, '_share_buffer_from_capsule', failing_receiver)
    with pytest.raises(RuntimeError, match='receiver failed'):
        native.make_tracked_buffer(count)
    gc.collect()
    assert count.value == (0 if retained else 1)
    if retained:
        assert retained[0].to_bytes() == b'\x01\x02\x03\x04'
    retained.clear()
    gc.collect()
    assert count.value == 1


def test_receiver_can_retain_capsule_and_share_repeatedly(monkeypatch):
    count = native.DestructionCount()
    original = bindings._share_buffer_from_capsule
    retained = []

    def receiver(capsule):
        retained.append(capsule)
        return original(capsule)

    monkeypatch.setattr(bindings, '_share_buffer_from_capsule', receiver)
    first = native.make_tracked_buffer(count)
    second = original(retained[0])
    assert _get_buffer_ptr(first) == _get_buffer_ptr(second)
    del first, second
    gc.collect()
    assert count.value == 0
    retained.clear()
    gc.collect()
    assert count.value == 1


def test_result_discarded_by_receiver(monkeypatch):
    count = native.DestructionCount()
    original = bindings._share_buffer_from_capsule

    def receiver(capsule):
        wrapper = original(capsule)
        del wrapper
        return None

    monkeypatch.setattr(bindings, '_share_buffer_from_capsule', receiver)
    assert native.make_tracked_buffer(count) is None
    gc.collect()
    assert count.value == 1
