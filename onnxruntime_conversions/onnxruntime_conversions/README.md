# ONNX Runtime conversions

Zero-copy C++ and Python views between
`tensor_msgs/msg/ExperimentalTensor` messages and ONNX Runtime tensors.

## Installation

```bash
# C++ CPU
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions

# C++ CUDA; this automatically installs the C++ CPU/core package
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-cuda-plugin

# Python CPU
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cpu

# Python CUDA; install this instead of the Python CPU package
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cuda
```

The CUDA packages pull their C++ or Python core dependencies automatically.
The Python CPU and CUDA packages cannot be installed together because their
ONNX Runtime wheels provide the same `onnxruntime` import.

For source builds:

```bash
# C++ CPU
colcon build --packages-up-to onnxruntime_conversions

# C++ CUDA; use this instead to include the core package
colcon build --packages-up-to onnxruntime_conversions_cuda_plugin

# Python CPU
colcon build --packages-up-to onnxruntime_conversions_py_cpu

# Python CUDA; use this instead of the Python CPU target
colcon build --packages-up-to onnxruntime_conversions_py_cuda
```

Build only one Python variant into an install prefix. CUDA source builds
detect CUDA 12 or CUDA 13 from the local toolkit; set `CUDAToolkit_ROOT` when
multiple toolkits are installed.

## C++

```cpp
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <utility>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

auto allocated = onnxruntime_conversions::allocate_tensor_msg(
  {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "cpu");
std::shared_ptr<onnxruntime_conversions::TensorMsg> msg(
  std::move(allocated));

auto memory_info = Ort::MemoryInfo::CreateCpu(
  OrtArenaAllocator, OrtMemTypeDefault);
auto view = onnxruntime_conversions::from_output_tensor_msg(
  msg, memory_info);
Ort::Value & value = view.value();
```

Keep `OrtTensorView` alive while ONNX Runtime accesses the tensor. The view
protects the message storage. CUDA operations require a non-null explicit
stream supplied through `BackendConfiguration`.

```cpp
onnxruntime_conversions::BackendConfiguration configuration;
configuration.device_id = 0;
configuration.execution_stream = cuda_stream;

auto msg = onnxruntime_conversions::allocate_tensor_msg(
  {2, 3},
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
  "cuda",
  configuration);
```

Use `configure_session_options()` with the same backend configuration before
constructing an ONNX Runtime session.

## Python

The installed Python variant selects the default allocation backend. The CUDA
variant defaults to CUDA but still supports explicit CPU allocation.

```python
import numpy as np

from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg

# CPU package: CPU is the default.
msg = allocate_tensor_msg((2, 3), np.float32)
with from_output_tensor_msg(msg) as value:
    value.update_inplace(np.ones((2, 3), dtype=np.float32))

# CUDA package: CUDA is the default and the application supplies the stream.
cuda_msg = allocate_tensor_msg((2, 3), np.float32, stream=cuda_stream)
with from_output_tensor_msg(cuda_msg, stream=cuda_stream) as value:
    value.update_inplace(np.ones((2, 3), dtype=np.float32))

# CPU remains available with the CUDA package.
cpu_msg = allocate_tensor_msg(
    (2, 3), np.float32, device_type='cpu')
```
