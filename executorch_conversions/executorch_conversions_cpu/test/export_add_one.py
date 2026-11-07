#!/usr/bin/env python3
# Copyright 2026 Meta Platforms, Inc. and affiliates.
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

"""Export add_one.pte, the program the tests run on message storage."""

import argparse
import os

from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
import torch


class AddOne(torch.nn.Module):

    def forward(self, x):
        return x + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', default='add_one.pte')
    parser.add_argument(
        '--cuda', action='store_true',
        help='export for the CUDA backend, and write its data file next to the program')
    args = parser.parse_args()

    # The runtime then reads and writes the caller's storage in place.
    unplanned = MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False)
    if args.cuda:
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

        with torch.no_grad():
            exported = torch.export.export(AddOne().cuda(), (torch.zeros(4, device='cuda'),))
            spec = [CudaBackend.generate_method_name_compile_spec('forward')]
            program = to_edge_transform_and_lower(
                exported, partitioner=[CudaPartitioner(spec)]).to_executorch(
                    ExecutorchBackendConfig(
                        memory_planning_pass=unplanned,
                        # Inputs and outputs stay in device memory.
                        propagate_device_config=PropagateDeviceConfig(
                            skip_h2d_for_method_inputs=True,
                            skip_d2h_for_method_outputs=True),
                        enable_non_cpu_memory_planning=True))
    else:
        program = to_edge_transform_and_lower(
            torch.export.export(AddOne(), (torch.zeros(4),))).to_executorch(
                ExecutorchBackendConfig(memory_planning_pass=unplanned))
    with open(args.output, 'wb') as file:
        program.write_to_file(file)
    if args.cuda:
        program.write_tensor_data_to_file(os.path.dirname(os.path.abspath(args.output)))


if __name__ == '__main__':
    main()
