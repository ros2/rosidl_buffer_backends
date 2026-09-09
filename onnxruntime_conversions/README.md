# ONNX Runtime conversions

Zero-copy conversions between `tensor_msgs/msg/ExperimentalTensor` and ONNX
Runtime 1.23.2. CUDA uses CUDA 12 and cuDNN 9.

A runtime plugin provides the implementation. Examples below use
`onnxruntime_conversions_cuda` and `onnxruntime_conversions_py_cuda`. The
CPU-only plugins are `onnxruntime_conversions_cpu` and
`onnxruntime_conversions_py_cpu`.

## C++

Debian:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-cuda
```

Source:

```bash
colcon build --merge-install --packages-up-to onnxruntime_conversions_cuda
```

### Publisher

```cpp
#include <cuda_runtime.h>
#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

std::shared_ptr<onnxruntime_conversions::TensorMsg> msg(
  onnxruntime_conversions::allocate_tensor_msg(
    {480, 640, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cuda"));
{
  auto output = onnxruntime_conversions::from_output_tensor_msg(
    msg, memory_info, stream);
  my_pipeline(output.value());
}
publisher->publish(std::move(*msg));
```

### Subscriber

```cpp
void callback(
  const onnxruntime_conversions::TensorMsg::SharedPtr received)
{
  auto input = onnxruntime_conversions::from_input_tensor_msg(
    received, memory_info, stream);
  my_pipeline(input.value());
}
```

### Existing tensor

```cpp
auto msg = onnxruntime_conversions::to_tensor_msg(
  ort_value, "cuda", stream);
publisher->publish(std::move(msg));
```

### CUDA session

```cpp
Ort::SessionOptions options;
onnxruntime_conversions::ConversionConfiguration config;
config.device_id = 0;
config.execution_stream = stream;
onnxruntime_conversions::configure_session_options(
  options, "cuda", config);
Ort::Session session(env, model, model_size, options);
```

## Python

Debian:

```bash
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cuda
```

Source:

```bash
colcon build --merge-install --packages-up-to onnxruntime_conversions_py_cuda
```

### Publisher

```python
import numpy as np
from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg

msg = allocate_tensor_msg((480, 640, 3), np.float32, 'cuda', stream=stream)
output = from_output_tensor_msg(msg, stream)
my_pipeline(output.value)
output.close()
publisher.publish(msg)
```

### Subscriber

```python
from onnxruntime_conversions import from_input_tensor_msg

def callback(received):
    view = from_input_tensor_msg(received, stream)
    my_pipeline(view.value)
    view.close()
```

### Existing tensor

```python
from onnxruntime_conversions import to_tensor_msg

msg = to_tensor_msg(ort_value, stream=stream, device_type='cuda')
publisher.publish(msg)
```

## Version

Require ONNX Runtime 1.23.x. Remove other copies from `/usr/local`, pip,
Conda, or `PYTHONPATH` if they take precedence.
