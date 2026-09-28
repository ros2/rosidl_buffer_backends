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
    """Rust CUDA publisher with GPU and ordinary CPU Image subscribers."""
    runner = LaunchConfiguration('test_runner')
    publisher = Node(
        executable=runner,
        arguments=['publisher', '2'], output='screen')
    gpu = Node(
        executable=runner,
        arguments=['subscriber', '0', 'cuda'], output='screen')
    cpu = Node(
        executable=runner,
        arguments=['subscriber', '1', 'cpu'], output='screen')
    return LaunchDescription([
        DeclareLaunchArgument(
            'test_runner',
            default_value=str(Path(__file__).resolve().parents[1]
                              / 'target/debug/examples/cuda_buffer_rs_test_runner')),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        EnableRmwIsolation(),
        publisher, gpu, cpu, launch_testing.actions.ReadyToTest(),
    ]), {'publisher': publisher, 'gpu': gpu, 'cpu': cpu}


class TestCudaImageCpuFallbackFastRTPS(unittest.TestCase):

    def test_delivery(self, proc_output, proc_info, publisher, gpu, cpu):
        proc_output.assertWaitFor(
            'GPU_SUBSCRIBER_PASS id=0 samples=10', process=gpu, timeout=60, stream='stdout')
        proc_output.assertWaitFor(
            'CPU_SUBSCRIBER_PASS id=1 samples=10', process=cpu, timeout=60, stream='stdout')
        proc_output.assertWaitFor(
            'PUBLISHER_PASS samples=10 subscribers=2',
            process=publisher, timeout=60, stream='stdout')
        for process in [publisher, gpu, cpu]:
            proc_info.assertWaitForShutdown(process=process, timeout=10)


@launch_testing.post_shutdown_test()
class TestExitCodes(unittest.TestCase):

    def test_clean_exit(self, proc_info, publisher, gpu, cpu):
        for process in [publisher, gpu, cpu]:
            launch_testing.asserts.assertExitCodes(
                proc_info, process=process, allowable_exit_codes=[0])
