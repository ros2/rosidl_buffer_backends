# ONNX Runtime conversions

Zero-copy C++ and Python views between
`tensor_msgs/msg/ExperimentalTensor` messages and ONNX Runtime tensors.

## ONNX Runtime conversion (C++)

Choose one of the following variants. Use either its Debian command or its
source-build command.

### CPU

```bash
# Debian
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions

# Source
colcon build --merge-install --packages-up-to onnxruntime_conversions
```

### Accelerator (CUDA reference)

Install or build only this top-level target. It pulls the base C++ package
automatically.

```bash
# Debian
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-cuda

# Source
colcon build --merge-install --packages-up-to onnxruntime_conversions_cuda
```

### Usage

The following example uses the CUDA accelerator package. Because that package
is installed and a CUDA stream is supplied, the conversion selects CUDA
automatically; no device type or device ID is required.

```cpp
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <utility>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

onnxruntime_conversions::ConversionConfiguration configuration;
configuration.execution_stream = cuda_stream;

auto allocated = onnxruntime_conversions::allocate_tensor_msg(
  {2, 3}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, configuration);
std::shared_ptr<onnxruntime_conversions::TensorMsg> msg(
  std::move(allocated));

Ort::MemoryInfo memory_info{
  "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
auto view = onnxruntime_conversions::from_output_tensor_msg(
  msg, memory_info, cuda_stream);
Ort::Value & value = view.value();
```

Keep `OrtTensorView` alive while ONNX Runtime accesses the tensor. The view
protects the message storage. CUDA requires the non-null explicit stream shown
above.

Use `configure_session_options()` with the same conversion configuration before
constructing an ONNX Runtime session.

## ONNX Runtime conversion (Python)

Choose exactly one of the following variants. Use either its Debian command or
its source-build command.

### CPU

```bash
# Debian
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py

# Source
colcon build --merge-install --packages-up-to onnxruntime_conversions_py
```

### Accelerator (CUDA reference)

```bash
# Debian
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cuda

# Source
colcon build --merge-install --packages-up-to onnxruntime_conversions_py_cuda
```

This target pulls the shared Python core automatically. Do not install or
build it with the CPU variant in the same prefix because their ONNX Runtime
distributions provide the same `onnxruntime` import.

During a source build, the CUDA reference adapter detects CUDA 12 or CUDA 13
from the local toolkit. Set `CUDAToolkit_ROOT` when multiple toolkits are
installed.

The installed Python variant selects the default conversion adapter. An
accelerator variant uses its accelerator by default while keeping explicit CPU
allocation available.

### Usage

The following example uses the CUDA accelerator package. The installed adapter
makes CUDA the default, so no device type or device ID is required. CUDA still
requires an explicit stream.

```python
import numpy as np

from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg

msg = allocate_tensor_msg((2, 3), np.float32, stream=cuda_stream)
with from_output_tensor_msg(msg, stream=cuda_stream) as value:
    value.update_inplace(np.ones((2, 3), dtype=np.float32))
```
