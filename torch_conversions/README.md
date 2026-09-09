# PyTorch conversions

Conversions between `tensor_msgs/ExperimentalTensor` and PyTorch.

`torch_conversions` translates between PyTorch tensors and DLPack, and
[`dlpack_conversions`](../dlpack_conversions/README.md) owns the message
storage behind it. Which memory a tensor lands in is decided by the storage
plugin installed alongside the adapter, not by the adapter itself, so adding
`dlpack_conversions_cuda` to an existing install is enough to move the same
code onto the GPU. Pass a backend name to choose among several, or set
`ROSIDL_TENSOR_BACKEND` to pick one for the whole process.

| Device | C++ plugin | Python plugin |
| --- | --- | --- |
| Host | `dlpack_conversions_cpu` | `dlpack_conversions_py_cpu` |
| CUDA | `dlpack_conversions_cuda` | `dlpack_conversions_py_cuda` |

## C++

`torch_conversions` is header-only and compiles against whichever libtorch the
consumer already resolved, so it does not pin a PyTorch version.

Debian:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions \
  ros-$ROS_DISTRO-dlpack-conversions-cuda
```

Source:

```bash
colcon build --merge-install --packages-up-to torch_conversions \
  dlpack_conversions_cuda
```

### Publisher

```cpp
#include "torch_conversions/torch_conversions.hpp"

auto guard = torch_conversions::set_stream();
auto msg = torch_conversions::allocate_tensor_msg(
  {480, 640, 3}, torch::kUInt8, c10::kCUDA);
at::Tensor output = torch_conversions::from_output_tensor_msg(*msg);
my_pipeline(output);
publisher->publish(std::move(msg));
```

### Subscriber

```cpp
auto guard = torch_conversions::set_stream();
at::Tensor tensor = torch_conversions::from_input_tensor_msg(*received);
at::Tensor view = torch_conversions::from_input_tensor_msg(
  *received, /*clone=*/false);
```

### Existing tensor

```cpp
at::Tensor tensor = my_pipeline().contiguous();
auto msg = torch_conversions::to_tensor_msg(tensor);
publisher->publish(std::move(msg));
```

`to_tensor_msg(*msg, tensor)` reuses a pre-sized message.

## Python

Debian:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-py \
  ros-$ROS_DISTRO-dlpack-conversions-py-cuda
```

Source:

```bash
colcon build --merge-install --packages-up-to torch_conversions_py \
  dlpack_conversions_py_cuda
```

### Publisher

```python
import torch
from torch_conversions import allocate_tensor_msg
from torch_conversions import from_output_tensor_msg
from torch_conversions import set_stream

with set_stream():
    msg = allocate_tensor_msg((480, 640, 3), torch.uint8, 'cuda')
    output = from_output_tensor_msg(msg)
    my_pipeline(output)
publisher.publish(msg)
```

### Subscriber

```python
from torch_conversions import from_input_tensor_msg
from torch_conversions import set_stream

def callback(msg):
    with set_stream():
        tensor = from_input_tensor_msg(msg)
        view = from_input_tensor_msg(msg, clone=False)
```

### Existing tensor

```python
from torch_conversions import to_tensor_msg

msg = to_tensor_msg(torch.arange(12, device='cuda').reshape(3, 4))
publisher.publish(msg)
```

`to_tensor_msg(msg, tensor)` reuses a pre-sized message.

## License

Apache-2.0
