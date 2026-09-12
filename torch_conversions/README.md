# PyTorch conversions

Conversions between `tensor_msgs/msg/ExperimentalTensor` and PyTorch tensors.
The public packages are framework-specific and device-neutral:

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `torch_conversions` | `torch_conversions_py` |
| CPU implementation | `torch_conversions_cpu` | `torch_conversions_py_cpu` |
| CUDA implementation | `torch_conversions_cuda` | `torch_conversions_py_cuda` |

Install CPU and accelerator plugins as separate Debian packages without
rebuilding the conversion core or a device-neutral application. C++ discovers
plugins on first registry use; Python discovers them when the module is imported.
Restart the application after installing a plugin or provider and source the
ROS environment before starting it.

Default allocation uses the available plugin with the highest priority: CPU
is 0 and CUDA is 100. Set `ROSIDL_TENSOR_BACKEND=cpu` or `cuda` to override that
default, or pass a device to `allocate_tensor_msg`. Views use the message's
storage backend; copies into a new message use the source tensor's device.
An explicit device such as `cuda:1` preserves its device index. Additional
accelerators can provide plugins without adding device-specific logic to the
core; their framework providers must satisfy the core's ABI contract.

## Torch provider boundary

The core and C++ plugins share LibTorch's C++ ABI. A CPU installation uses
`libtorch_vendor`, which installs the official CPU LibTorch 2.9.1 distribution.
The CUDA plugin depends on `libtorch_cuda_vendor`, which installs a matching
CUDA LibTorch 2.9.1 tree and places it ahead of the system libraries for newly
started ROS processes.
Only one set of LibTorch SONAMEs is loaded in a process.

Python follows the same layout: `python3_torch_vendor` installs the official
CPU Torch wheel, while the CUDA plugin installs `python3_torch_cuda_vendor` as a
higher-priority ROS-prefix overlay. After the CUDA overlay is installed, both
CPU and CUDA conversion plugins use that CUDA-capable Torch distribution.

The dependency boundary is:

| Package | Required framework package | CUDA dependency |
| --- | --- | --- |
| `torch_conversions` | `libtorch_vendor` | No |
| `torch_conversions_cpu` | inherited from `torch_conversions` | No |
| `torch_conversions_cuda` | `libtorch_cuda_vendor` | Yes |
| `torch_conversions_py` | `python3_torch_vendor` | No |
| `torch_conversions_py_cpu` | inherited from `torch_conversions_py` | No |
| `torch_conversions_py_cuda` | `python3_torch_cuda_vendor` | Yes |

The CUDA providers depend on their CPU counterparts; both package sets can
coexist. ROS environment hooks select the CUDA overlay for newly started
processes. CPU-only installations do not require CUDA runtime dependencies.

## Storage lifetime

C++ views borrow message storage on both CPU and CUDA: keep the message alive
and do not replace or resize its buffer while a view exists. A CUDA access
handle does not own the buffer. The default input conversion clones the tensor;
`clone=false` returns a borrowed view. Python views retain their backing buffer,
but it must not be resized while a view exists. Release every output view before
publishing so its write handle records completion.

## Execution streams

Conversions use Torch's current stream on the buffer or source tensor's device
unless an execution stream is passed explicitly. The same conversion calls
work on CPU and CUDA without passing a CUDA stream or including CUDA headers.

Use `auto guard = torch_conversions::set_stream()` in C++, or `with set_stream():`
in Python, to scope tensor operations to a plugin-selected stream. CPU is a
no-op; CUDA selects a pooled stream. The previous stream and device are restored
when the scope exits, including on exceptions. Selection follows the allocation
policy: an explicit device first, then `ROSIDL_TENSOR_BACKEND`, then plugin
priority. Pass the same explicit device to `set_stream(device)` and allocation
when overriding the default. Both helpers affect the calling thread only.

The guard does not synchronize on exit. Keep conversions and subsequent tensor
operations inside its scope, and release views before leaving it. If the
application already manages a Torch stream, conversions can use that stream
without `set_stream()`. Work on other streams requires explicit synchronization.
In C++, an explicit `nullptr` also selects Torch's current stream;
`cudaStreamLegacy` selects the legacy default stream.
Copies into messages complete before returning, including any contiguous
temporary; zero-copy views and input clones can still have queued GPU work.

Caller-provided streams must remain alive until all views are released and
their work completes. Application operations and model tensors must support
the selected device; stream selection does not move them between devices.

Select the buffer's CUDA device before conversion and keep it selected when
releasing views. The existing CUDA buffer pool supports one allocation device
per process; choose that device before the first allocation. A request for a
different device fails instead of returning storage on the wrong GPU.

Copies between different GPU indices are rejected. Allocate and copy on the
source device, or perform an explicit device transfer in Torch first.

## C++

CPU installation:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions \
  ros-$ROS_DISTRO-torch-conversions-cpu
```

Add CUDA later:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-cuda
```

Source build:

```bash
rosdep install --from-paths . --ignore-src -y
colcon build --merge-install --packages-up-to \
  torch_conversions_cpu torch_conversions_cuda
```

For a CPU-only source build, restrict rosdep to the CPU source packages as
well as omitting the CUDA target:

```bash
rosdep install --from-paths \
  tensor_msgs \
  torch_vendor/libtorch_vendor \
  torch_conversions/torch_conversions \
  torch_conversions/torch_conversions_cpu \
  --ignore-src -y
colcon build --merge-install --packages-up-to torch_conversions_cpu
```

### Publisher

```cpp
#include "torch_conversions/torch_conversions.hpp"

auto guard = torch_conversions::set_stream();
auto msg = torch_conversions::allocate_tensor_msg(
  {480, 640, 3}, torch::kUInt8);
{
  at::Tensor output = torch_conversions::from_output_tensor_msg(*msg);
  my_pipeline(output);
}
publisher->publish(std::move(msg));
```

### Subscriber and existing tensors

```cpp
auto guard = torch_conversions::set_stream();
auto tensor = torch_conversions::from_input_tensor_msg(
  *received, /*clone=*/true);
auto view = torch_conversions::from_input_tensor_msg(
  *received, /*clone=*/false);
auto outgoing = torch_conversions::to_tensor_msg(tensor);
torch_conversions::to_tensor_msg(*preallocated, tensor);
```

Every conversion accepts an optional execution stream. When omitted, the CUDA
plugin uses Torch's current stream on the buffer or source tensor's device.

## Python

CPU installation:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-py \
  ros-$ROS_DISTRO-torch-conversions-py-cpu
```

Add CUDA later:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-py-cuda
```

Source build:

```bash
rosdep install --from-paths . --ignore-src -y
colcon build --merge-install --packages-up-to \
  torch_conversions_py_cpu torch_conversions_py_cuda
```

For a Python CPU-only source build, restrict both commands to the CPU stack:

```bash
rosdep install --from-paths \
  tensor_msgs \
  torch_vendor/python3_torch_vendor \
  torch_conversions/torch_conversions_py \
  torch_conversions/torch_conversions_py_cpu \
  --ignore-src -y
colcon build --merge-install --packages-up-to torch_conversions_py_cpu
```

```python
import torch
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_input_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import set_stream
from torch_conversions import to_tensor_msg

with set_stream():
    msg = allocate_tensor_msg((480, 640, 3), torch.uint8)
    output = from_output_tensor_msg(msg)
    my_pipeline(output)
    del output

    view = from_input_tensor_msg(msg, clone=False)
    outgoing = to_tensor_msg(view + 1)
    del view
```

Python uses Torch's current CUDA stream when no explicit `stream=` integer is
provided. `to_tensor_msg(msg, tensor)` reuses preallocated message storage. The
CUDA plugin constructs zero-copy views through a private DLPack capsule bridge
that retains the buffer and its access handle.

## License

Apache-2.0
