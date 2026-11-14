# ExecuTorch conversions

A C++ library with a device-neutral API for converting between
`tensor_msgs/msg/ExperimentalTensor` messages and
[ExecuTorch](https://github.com/pytorch/executorch) tensors. CPU and accelerator
support comes from separately installed plugins, so applications use the same
conversion API on every device. The plugin interface and package layout follow
`torch_conversions`.

ExecuTorch runs exported PyTorch programs (`.pte` files) without PyTorch. These
packages link only the ExecuTorch runtime libraries, so a node that runs a policy
does not load LibTorch.

The API writes directly into message storage, reads received messages as tensor
views, and copies existing tensors into messages. With the CUDA plugin, message
storage is a `cuda_buffer_backend` buffer, so a program reads its inputs from the
publisher's GPU memory and writes its outputs into the memory the next node
receives, with no copy through the CPU. That holds when both nodes run on one
host as the same user and use the same GPU; otherwise `cuda_buffer` delivers a
host message instead. CUDA messages need a driver with CUDA virtual memory
management: without it, allocating one throws. The
ExecuTorch CUDA backend may still copy on the GPU, for example when it replays
CUDA graphs.

## Running a program on messages

This subscriber runs an ExecuTorch program on every received tensor and
publishes the result. The output is written straight into the outgoing
message's storage. Its subscription sets
`options.acceptable_buffer_backends = "any"`, as in the `torch_conversions`
example; with the default, every message arrives in host memory:

```cpp
#include <executorch/extension/module/module.h>

#include "executorch_conversions/executorch_conversions.hpp"

using namespace executorch::aten;
using namespace executorch::runtime;

void PolicyNode::receive(const tensor_msgs::msg::ExperimentalTensor & received)
{
  std::unique_ptr<executorch_conversions::TensorMsg> reply;
  try {
    // Allocate the reply on the device the program writes to.
    reply = executorch_conversions::allocate_tensor_msg(
      {1, 100, 14}, ScalarType::Float, Device(DeviceType::CUDA));
    // Views borrow message storage; no bytes are copied.
    auto input = executorch_conversions::from_input_tensor_msg(received);
    if (!input) {
      return;  // the message has no data
    }
    // A CPU program does not check where its input lives and crashes on a GPU
    // tensor. This CUDA program would return an error for a host message.
    if (input->device().type() != DeviceType::CUDA) {
      RCLCPP_WARN_ONCE(get_logger(), "Skipping a message in host memory");
      return;
    }
    auto output = executorch_conversions::from_output_tensor_msg(*reply);

    // The program writes into the reply. A Method keeps the pointer it was
    // given, so set the output again for every message.
    if (module_->set_output(*output) != Error::Ok || !module_->forward(*input).ok()) {
      return;
    }
  } catch (const std::exception & error) {
    // Any publisher can send a message the library cannot view, and the
    // allocation needs a usable CUDA plugin.
    RCLCPP_ERROR(get_logger(), "Skipping a message: %s", error.what());
    return;
  }  // The views are released here, before the message goes to the publisher.

  publisher_->publish(std::move(reply));
}
```

To run a program in place on message storage, export it so the runtime does
not plan its own buffers for inputs and outputs:

```python
import torch
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass

edge = to_edge_transform_and_lower(torch.export.export(model, example_inputs))
program = edge.to_executorch(ExecutorchBackendConfig(
    memory_planning_pass=MemoryPlanningPass(
        alloc_graph_input=False, alloc_graph_output=False)))
```

For the CUDA backend, also pass
`partitioner=[CudaPartitioner([CudaBackend.generate_method_name_compile_spec('forward')])]`
(from `executorch.backends.cuda.cuda_partitioner` and
`executorch.backends.cuda.cuda_backend`) to `to_edge_transform_and_lower`, and keep inputs
and outputs in device memory with `propagate_device_config=PropagateDeviceConfig(skip_h2d_for_method_inputs=True,
skip_d2h_for_method_outputs=True)` (from `executorch.exir.passes.propagate_device_config`)
and `enable_non_cpu_memory_planning=True`, as
`executorch_conversions_cpu/test/export_add_one.py --cuda` does. The CUDA backend
keeps its compiled kernels in a separate `.ptd` data file: write it with
`program.write_tensor_data_to_file(directory)`, and pass its path to the `Module`
with the program.

A program run in place must not change its inputs. Its input is the received
message, which other subscriptions in the same process may share. A CUDA message
is the publisher's GPU memory, so on the CUDA path subscribers in other processes
share it too. Copy the input with `to_tensor_msg` first for a program that changes it.

## Example nodes

The core package installs a publisher and a subscriber:

```bash
ros2 run executorch_conversions et_tensor_publisher_node --ros-args -p device:=cuda
ros2 run executorch_conversions et_tensor_subscriber_node
```

The publisher copies a host frame into a newly allocated message with
`to_tensor_msg`. The subscriber views the received storage with
`from_input_tensor_msg` and copies it to a host message to print it.

## Streams

The CUDA plugin follows the stream ExecuTorch's CUDA backend runs on. When an
`executorch::extension::cuda::CallerStreamGuard` is active, both use that
stream. Otherwise both use the per-thread default stream. So the wait on the
publisher's write and the program's kernels are ordered without extra
synchronization.

The per-thread default stream is a different stream on each thread. So make, use
and release a view on the per-thread stream on the thread that runs the program.
When views may be released on another thread, use a `CallerStreamGuard` with a
stream you created instead. A view on the per-thread stream that is released on
another thread anyway first waits for all work on its GPU. While any thread in the
process records a CUDA graph, CUDA refuses that wait and the recording fails, so
the release waits until the recording has stopped. A stream passed to a conversion call
orders the message's reads and writes, and `to_tensor_msg` also copies on it. So
it must be the stream the program runs on, and it must outlive the view. A view
waits for the message's producer only on its own stream, so copy it with
`to_tensor_msg` on that stream, or synchronize with it first.

The stream is chosen when the view is made, and the CUDA backend chooses its own
again when the program runs. So a `CallerStreamGuard` must already be active when
the views are made, not only around `forward()`. A stream on another GPU than the
message is refused.

```cpp
bool ran = false;
{
  CallerStreamGuard guard(stream);
  auto input = executorch_conversions::from_input_tensor_msg(*request);
  auto output = executorch_conversions::from_output_tensor_msg(*reply);
  ran = module_->set_output(*output) == Error::Ok && module_->forward(*input).ok();
}  // The views are released before the reply is published.
if (ran) {
  publisher_->publish(std::move(reply));
}
```

ExecuTorch's CUDA backend refuses to run a program that uses CUDA graphs under a
`CallerStreamGuard`, so leave the stream unset for such a program.

## Limits

- A message view must have a contiguous (row-major) layout, because an
  ExecuTorch program reads its inputs in that layout whatever their strides
  say. Other strides are rejected with an exception.
- A view is on the device of the message storage. A node whose program runs on
  one device should check the view's device before running the program, because
  a CPU program does not. Subscribing only to the CUDA buffer backend is not enough:
  `cuda_buffer` sends CPU messages when it cannot share GPU memory.
  `allocate_tensor_msg` without a device
  uses the highest-priority available plugin, which is CUDA when it is
  installed and a GPU is usable.
- A program with dynamic output shapes can write fewer elements than the reply
  holds. Set the reply's `shape` from the program's output after running it,
  and clear its `strides`.
- `to_tensor_msg` throws for a tensor that is not contiguous, for example a
  channels-last one, because the views could not read the message.
- ExecuTorch does not compare a tensor's layout with the one the program was
  exported for, so a program exported with channels-last inputs reads a
  contiguous view wrongly. Export programs with contiguous inputs and outputs.
- Sizes, and the contiguous strides computed from them, must fit in 32 bits, and
  a tensor has at most 16 dimensions. These are ExecuTorch limits.
- `set_output` checks only that the message holds at least the output's bytes,
  not its dtype or shape.
- A process that also loads LibTorch must not use the ExecuTorch CUDA backend:
  both export the same `aoti_torch_*` functions, and the program can bind to
  LibTorch's copies and crash.
- The installed libraries find the ExecuTorch runtime through an absolute RUNPATH,
  so moving an install tree breaks them. Build it where it will run.
- `cuda_buffer` serves one GPU per process: the current one when the process
  creates its first CUDA buffer, publisher or subscription.
- The supported dtypes are the ones `torch_conversions` supports: uint8, int8,
  int16, int32, int64, float16, bfloat16, float32, float64 and bool.

## Build

From this repository's root after sourcing a ROS 2 Lyrical or Rolling
environment:

```bash
colcon build --merge-install --packages-up-to executorch_conversions_cuda executorch_conversions_cpu
source install/setup.bash
```

`executorch_vendor` and `executorch_cuda_vendor` reuse an installed
ExecuTorch 1.6 (for example from `pip install executorch`). A nightly build
reports itself as 1.6 too, but only one from 2026-09-30 or later has every API
these packages use, so an older nightly is not reused. Without a usable
ExecuTorch 1.6, they
download a pinned wheel and install only its C++ runtime: the CPU build for
`executorch_vendor`, and the CUDA 13 build for `executorch_cuda_vendor`, which
also installs a copy of the toolkit's CUDA runtime (`libcudart`, under the
NVIDIA CUDA EULA). The CUDA plugin, and any package that runs a CUDA program,
needs `executorch_cuda_vendor` and a matching CUDA toolkit: CUDA 13 for the
downloaded wheel, or the CUDA version a reused install was built for.
With a toolkit of another CUDA version, or when `executorch_vendor` uses another
ExecuTorch release, both CUDA packages print a warning and build nothing. A reused
source build does not name its CUDA version, so it is not
checked. The CUDA plugin also needs CUDA 12.8 or newer, and skips itself the same
way on an older toolkit. Without any CUDA toolkit, `cuda_buffer` cannot build, so build the CPU
packages alone:

```bash
colcon build --merge-install --packages-up-to executorch_conversions_cpu
```

A package that uses both vendors must find `executorch_cuda_vendor` before
`find_package(executorch)` runs, directly or through `executorch_conversions`.
Otherwise configure stops with an error instead of mixing the two runtimes. With
`ament_auto_find_build_dependencies()`, which finds packages in `package.xml`
order, list `executorch_cuda_vendor` before `executorch_conversions`. When
`executorch_cuda_vendor` installed nothing, `EXECUTORCH_CUDA_VENDOR_SKIPPED` says
why, and a CUDA package should skip itself as `executorch_conversions_cuda` does:

```cmake
find_package(executorch_cuda_vendor REQUIRED)
if(EXECUTORCH_CUDA_VENDOR_SKIPPED)
  message(WARNING "${PROJECT_NAME}: skipping: ${EXECUTORCH_CUDA_VENDOR_SKIPPED}")
  ament_package()
  return()
endif()
```

A node that runs a program also links the operators and backends the program
uses, because the ExecuTorch runtime carries only primitive operators. For
example, link `executorch::kernels_optimized` for CPU operators and
`executorch::backend_cuda` for the CUDA backend. Without them, `forward()` fails
with a missing operator or an unregistered backend. A node that uses
`CallerStreamGuard` also links `executorch::extension_cuda`, which defines it.

CMake 3.28 or newer is required.

## Package structure

| Role | Package |
| --- | --- |
| ExecuTorch provider | `executorch_vendor`, `executorch_cuda_vendor` |
| Public API and plugin registry | `executorch_conversions` |
| CPU plugin | `executorch_conversions_cpu` |
| CUDA plugin | `executorch_conversions_cuda` |

## License

Apache-2.0
