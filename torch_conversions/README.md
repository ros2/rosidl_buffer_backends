# PyTorch conversions

Conversions between `tensor_msgs/ExperimentalTensor` and PyTorch tensors.
The public packages are framework-specific and device-neutral:

| Role | C++ | Python |
| --- | --- | --- |
| Public API and plugin registry | `torch_conversions` | `torch_conversions_py` |
| CPU implementation | `torch_conversions_cpu` | `torch_conversions_py_cpu` |
| CUDA implementation | `torch_conversions_cuda` | `torch_conversions_py_cuda` |

Install a CPU or CUDA plugin Debian to choose the available implementations
without changing application code or rebuilding the core. C++ discovers plugins
on first registry use; Python discovers them when the module is imported.
Restart the application after installing a plugin or provider and source the
ROS environment before starting it.

Default allocation uses the available plugin with the highest priority: CPU
is 0 and CUDA is 100. Set `ROSIDL_TENSOR_BACKEND=cpu` or `cuda` to override that
default, or pass a device to `allocate_tensor_msg`. Views use the message's
storage backend; copies into a new message use the source tensor's device.

## Torch provider boundary

The core and C++ plugins share LibTorch's C++ ABI. A CPU installation uses
`libtorch_vendor`, which
installs the official CPU LibTorch 2.9.1 distribution. The CUDA plugin depends
on `libtorch_cuda_vendor`, which installs a matching CUDA LibTorch 2.9.1 tree
and places it ahead of the system libraries for newly started ROS processes.
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

The CUDA providers depend on their CPU counterparts so both package sets can
coexist, but they do not link the two Torch distributions into one process.
ROS environment hooks prepend the CUDA provider directories after the CUDA
packages are installed. The exact shared-library ABI and Python package
version are pinned to 2.9.1 on both sides of the overlay.

CPU-only installations do not require CUDA runtime dependencies.

## Storage lifetime

C++ CPU views borrow message storage: keep the message alive and do not resize
its buffer while a view exists. The default input conversion clones the tensor;
`clone=false` returns a view. CUDA views retain a buffer access handle until
the tensor is released. Release output views before publishing so their write
handles can record completion on the execution stream.

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

The explicit core package in the first command is optional because APT also
installs it transitively from `torch_conversions_cpu`.

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
#include <c10/cuda/CUDAStream.h>
#include "torch_conversions/torch_conversions.hpp"

void * stream = c10::cuda::getCurrentCUDAStream().stream();
auto msg = torch_conversions::allocate_tensor_msg(
  {480, 640, 3}, torch::kUInt8, c10::kCUDA);
{
  at::Tensor output = torch_conversions::from_output_tensor_msg(*msg, stream);
  my_pipeline(output);
}
publisher->publish(std::move(msg));
```

### Subscriber and existing tensors

```cpp
auto tensor = torch_conversions::from_input_tensor_msg(
  *received, /*clone=*/true, stream);
auto view = torch_conversions::from_input_tensor_msg(
  *received, /*clone=*/false, stream);
auto outgoing = torch_conversions::to_tensor_msg(tensor, stream);
torch_conversions::to_tensor_msg(*preallocated, tensor, stream);
```

Every conversion accepts an optional execution stream. Pass the stream when
work does not run on the default CUDA stream so buffer access is ordered
against the caller's kernels. Host calls can omit it.

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

As with C++, the explicit Python core package is optional when installing its
CPU plugin.

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

with set_stream('cuda'):
    msg = allocate_tensor_msg((480, 640, 3), torch.uint8, 'cuda')
    output = from_output_tensor_msg(msg)
    my_pipeline(output)

view = from_input_tensor_msg(msg, clone=False)
outgoing = to_tensor_msg(torch.arange(12, device='cuda').reshape(3, 4))
```

Python uses Torch's current CUDA stream when no explicit `stream=` integer is
provided. `to_tensor_msg(msg, tensor)` reuses preallocated message storage. The
CUDA plugin constructs zero-copy views through a private DLPack capsule bridge
that retains the buffer and its access handle.

## License

Apache-2.0
