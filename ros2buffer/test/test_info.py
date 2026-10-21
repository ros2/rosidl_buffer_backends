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

from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace

import ros2buffer.verb.info as info_module
from ros2buffer.verb.info import InfoVerb


class FakeNode:

    def __init__(self, *, topics=None, publishers=None, subscribers=None):
        self.topics = topics or []
        self.publishers = publishers or []
        self.subscribers = subscribers or []

    def get_topic_names_and_types(self):
        return self.topics

    def get_publishers_info_by_topic(self, topic_name):
        return self.publishers

    def get_subscriptions_info_by_topic(self, topic_name):
        return self.subscribers


def make_endpoint(name, namespace, topic_type, gid):
    return SimpleNamespace(
        node_name=name,
        node_namespace=namespace,
        topic_type=topic_type,
        endpoint_gid=gid,
    )


def run_info(monkeypatch, node, *, discovery=None, topic_name='/image'):
    monkeypatch.setattr(
        info_module,
        'NodeStrategy',
        lambda args: nullcontext(node),
    )
    verb = InfoVerb(backend_discovery=discovery)
    return verb.main(args=Namespace(topic_name=topic_name))


def test_unknown_topic(monkeypatch, capsys):
    result = run_info(
        monkeypatch,
        FakeNode(topics=[('/other', ['example_interfaces/msg/String'])]),
    )

    assert result == "Unknown topic '/image'"
    assert capsys.readouterr().out == ''


def test_lists_publishers_and_subscribers_with_backend_metadata(
    monkeypatch,
    capsys,
):
    publisher = make_endpoint(
        'camera',
        '/',
        'sensor_msgs/msg/Image',
        [0x01, 0xA2],
    )
    subscriber = make_endpoint(
        'processor',
        '/vision',
        'sensor_msgs/msg/Image',
        [0x0B, 0x0C],
    )
    node = FakeNode(
        topics=[('/image', ['sensor_msgs/msg/Image'])],
        publishers=[publisher],
        subscribers=[subscriber],
    )

    metadata = {
        id(publisher): {'cuda': 'device=0', 'cpu': ''},
        id(subscriber): {'cpu': ''},
    }
    result = run_info(
        monkeypatch,
        node,
        discovery=lambda _, endpoint: metadata[id(endpoint)],
    )

    assert result is None
    assert capsys.readouterr().out == (
        'Topic: /image\n'
        'Publishers (1):\n'
        '  - Node: /camera\n'
        '    GID: 01.a2\n'
        '    Topic type: sensor_msgs/msg/Image\n'
        '    Buffer backends: cpu, cuda\n'
        'Subscribers (1):\n'
        '  - Node: /vision/processor\n'
        '    GID: 0b.0c\n'
        '    Topic type: sensor_msgs/msg/Image\n'
        '    Buffer backends: cpu\n'
    )


def test_reports_unavailable_metadata_per_endpoint(monkeypatch, capsys):
    publisher = make_endpoint(
        'camera',
        '/',
        'sensor_msgs/msg/Image',
        [0x01],
    )
    node = FakeNode(
        topics=[('/image', ['sensor_msgs/msg/Image'])],
        publishers=[publisher],
    )

    assert run_info(monkeypatch, node) is None
    output = capsys.readouterr().out
    assert (
        'Buffer backends: unavailable '
        '(the CLI node does not expose a compatible native handle)'
    ) in output
    assert 'Subscribers (0):\n  None\n' in output


def test_distinguishes_known_empty_metadata(monkeypatch, capsys):
    subscriber = make_endpoint(
        'listener',
        '',
        'example_interfaces/msg/String',
        [0x02],
    )
    node = FakeNode(
        topics=[('/image', ['example_interfaces/msg/String'])],
        subscribers=[subscriber],
    )

    assert run_info(
        monkeypatch,
        node,
        discovery=lambda _node, _endpoint: {},
    ) is None
    assert 'Buffer backends: none advertised\n' in capsys.readouterr().out


def test_backend_provider_not_implemented_is_unavailable(monkeypatch, capsys):
    publisher = make_endpoint(
        'talker',
        '/',
        'example_interfaces/msg/String',
        [0x03],
    )
    node = FakeNode(
        topics=[('/image', ['example_interfaces/msg/String'])],
        publishers=[publisher],
    )

    def unavailable(_node, _endpoint):
        raise NotImplementedError

    assert run_info(monkeypatch, node, discovery=unavailable) is None
    assert 'Buffer backends: unavailable' in capsys.readouterr().out


def test_backend_provider_explains_incompatible_rmw(monkeypatch, capsys):
    publisher = make_endpoint(
        'talker',
        '/',
        'example_interfaces/msg/String',
        [0x04],
    )
    node = FakeNode(
        topics=[('/image', ['example_interfaces/msg/String'])],
        publishers=[publisher],
    )

    def incompatible(_node, _endpoint):
        raise NotImplementedError(
            'incompatible rmw_fastrtps_cpp: metadata API is missing'
        )

    assert run_info(monkeypatch, node, discovery=incompatible) is None
    assert (
        'Buffer backends: unavailable '
        '(incompatible rmw_fastrtps_cpp: metadata API is missing)\n'
    ) in capsys.readouterr().out


def test_graph_endpoint_query_not_implemented(monkeypatch, capsys):

    class UnsupportedNode(FakeNode):

        def get_publishers_info_by_topic(self, topic_name):
            raise NotImplementedError('Endpoint information is unsupported')

    node = UnsupportedNode(
        topics=[('/image', ['example_interfaces/msg/String'])],
    )

    assert run_info(monkeypatch, node) == 'Endpoint information is unsupported'
    assert capsys.readouterr().out == ''
