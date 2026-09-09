# PyTorch conversions

Conversions between `tensor_msgs/ExperimentalTensor` and PyTorch.

A runtime plugin provides the implementation. Examples below use
`torch_conversions_cuda` and `torch_conversions_py_cuda`. The CPU-only plugins
are `torch_conversions_cpu` and `torch_conversions_py_cpu`.

## C++

Debian:

```bash
sudo apt install ros-$ROS_DISTRO-torch-conversions-cuda
```

Source:

```bash
colcon build --merge-install --packages-up-to torch_conversions_cuda
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
sudo apt install ros-$ROS_DISTRO-torch-conversions-py-cuda
```

Source:

```bash
colcon build --merge-install --packages-up-to torch_conversions_py_cuda
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
