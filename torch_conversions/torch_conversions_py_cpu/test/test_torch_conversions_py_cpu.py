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

import torch

from torch_conversions import from_input_tensor_msg
from torch_conversions import to_tensor_msg


def test_cpu_round_trip():
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    msg = to_tensor_msg(source)
    assert torch.equal(source, from_input_tensor_msg(msg))
