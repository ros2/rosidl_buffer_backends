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

from pathlib import Path
import unittest

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
from launch_testing_ros.actions import EnableRmwIsolation
import pytest


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    """One Rust publisher and two Rust GPU subscribers in separate processes."""
    runner = LaunchConfiguration('test_runner')
    publisher = Node(
        executable=runner,
        arguments=['publisher', '2'], output='screen')
    subscribers = [
        Node(executable=runner,
             arguments=['subscriber', str(index), 'cuda'], output='screen')
        for index in range(2)
    ]
    return LaunchDescription([
        DeclareLaunchArgument(
            'test_runner',
            default_value=str(Path(__file__).resolve().parents[1]
                              / 'target/debug/examples/cuda_buffer_rs_test_runner')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        EnableRmwIsolation(),
        publisher, *subscribers, launch_testing.actions.ReadyToTest(),
    ]), {'publisher': publisher, 'subscribers': subscribers}


class TestCudaImageInterPubSubFastRTPS(unittest.TestCase):

    def test_delivery(self, proc_output, proc_info, publisher, subscribers):
        for index, subscriber in enumerate(subscribers):
            proc_output.assertWaitFor(
                f'GPU_SUBSCRIBER_PASS id={index} samples=10',
                process=subscriber, timeout=60, stream='stdout')
        proc_output.assertWaitFor(
            'PUBLISHER_PASS samples=10 subscribers=2',
            process=publisher, timeout=60, stream='stdout')
        for process in [publisher, *subscribers]:
            proc_info.assertWaitForShutdown(process=process, timeout=10)


@launch_testing.post_shutdown_test()
class TestExitCodes(unittest.TestCase):

    def test_clean_exit(self, proc_info, publisher, subscribers):
        for process in [publisher, *subscribers]:
            launch_testing.asserts.assertExitCodes(
                proc_info, process=process, allowable_exit_codes=[0])
