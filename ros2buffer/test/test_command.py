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
from unittest.mock import Mock

from ros2buffer.command.buffer import BufferCommand


def test_command_without_verb_prints_help():
    parser = Mock()
    command = BufferCommand()
    command._subparser = parser

    assert command.main(parser=parser, args=Namespace()) == 0
    parser.print_help.assert_called_once_with()


def test_command_dispatches_to_verb():
    extension = Mock()
    extension.main.return_value = 7
    args = Namespace(_verb=extension)

    assert BufferCommand().main(parser=Mock(), args=args) == 7
    extension.main.assert_called_once_with(args=args)
