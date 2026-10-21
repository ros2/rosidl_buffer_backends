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

"""Implementation of ``ros2 buffer info``."""

from collections.abc import Callable
from collections.abc import Mapping
from typing import Optional

from ros2buffer.api import discover_endpoint_backend_metadata
from ros2buffer.verb import VerbExtension
from ros2cli.node.strategy import add_arguments as add_strategy_node_arguments
from ros2cli.node.strategy import NodeStrategy


BackendDiscovery = Callable[[object, object], Optional[Mapping[str, str]]]

_UNAVAILABLE = 'unavailable (not exposed by the current ROS graph API)'


def _format_gid(gid):
    return '.'.join(format(value, '02x') for value in gid)


def _format_node_name(endpoint):
    namespace = endpoint.node_namespace.rstrip('/')
    name = endpoint.node_name.lstrip('/')
    if not name:
        return '<unknown>'
    if not namespace:
        return '/' + name
    return namespace + '/' + name


def _format_backends(metadata, unavailable_reason=None):
    if metadata is None:
        if unavailable_reason:
            return f'unavailable ({unavailable_reason})'
        return _UNAVAILABLE
    if not metadata:
        return 'none advertised'
    return ', '.join(sorted(metadata))


def _print_endpoints(
    heading,
    endpoints,
    node,
    backend_discovery,
):
    print(f'{heading} ({len(endpoints)}):')
    if not endpoints:
        print('  None')
        return

    for endpoint in endpoints:
        unavailable_reason = None
        try:
            metadata = backend_discovery(node, endpoint)
        except NotImplementedError as error:
            metadata = None
            unavailable_reason = str(error)

        print(f'  - Node: {_format_node_name(endpoint)}')
        print(f'    GID: {_format_gid(endpoint.endpoint_gid)}')
        print(f'    Topic type: {endpoint.topic_type}')
        print(
            '    Buffer backends: '
            f'{_format_backends(metadata, unavailable_reason)}'
        )


class InfoVerb(VerbExtension):
    """Print buffer backend information for a topic."""

    def __init__(self, *, backend_discovery=None):
        super().__init__()
        self._backend_discovery = (
            backend_discovery or discover_endpoint_backend_metadata
        )

    def add_arguments(self, parser, cli_name):
        add_strategy_node_arguments(parser)
        parser.add_argument(
            'topic_name',
            help="Name of the ROS topic to inspect (for example, '/camera/image').",
        )

    def main(self, *, args):
        with NodeStrategy(args) as node:
            topics = node.get_topic_names_and_types()
            if not any(name == args.topic_name for name, _ in topics):
                return f"Unknown topic '{args.topic_name}'"

            try:
                publishers = node.get_publishers_info_by_topic(args.topic_name)
                subscribers = node.get_subscriptions_info_by_topic(args.topic_name)
            except NotImplementedError as error:
                return str(error)

            print(f'Topic: {args.topic_name}')
            _print_endpoints(
                'Publishers',
                publishers,
                node,
                self._backend_discovery,
            )
            _print_endpoints(
                'Subscribers',
                subscribers,
                node,
                self._backend_discovery,
            )
