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

from setuptools import find_packages
from setuptools import setup


package_name = 'ros2buffer'


setup(
    name=package_name,
    version='0.1.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
    ],
    package_data={'': ['py.typed']},
    install_requires=['ros2cli'],
    zip_safe=True,
    maintainer='Yuankun Zhu',
    maintainer_email='yuankunz@nvidia.com',
    url='https://github.com/ros2/rosidl_buffer_backends',
    keywords=['ROS 2', 'ros2cli', 'buffer'],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
    ],
    description='Inspect ROSIDL buffer backend support from the ROS 2 command line.',
    long_description=(
        'The package provides the ros2 buffer command for inspecting ROSIDL '
        'buffer backend support on topic endpoints.'
    ),
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'ros2cli.command': [
            'buffer = ros2buffer.command.buffer:BufferCommand',
        ],
        'ros2cli.extension_point': [
            'ros2buffer.verb = ros2buffer.verb:VerbExtension',
        ],
        'ros2buffer.verb': [
            'info = ros2buffer.verb.info:InfoVerb',
        ],
    },
)
