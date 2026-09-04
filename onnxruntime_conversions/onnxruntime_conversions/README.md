# ONNX Runtime conversions

`onnxruntime_conversions` provides C++ zero-copy views between
`tensor_msgs::msg::ExperimentalTensor` messages and ONNX Runtime
`Ort::Value` tensors. It contains the conversion registry and the CPU storage
backend.

The optional `onnxruntime_conversions_cuda_plugin` package adds CUDA storage
and the ONNX Runtime CUDA execution provider without changing this API.

## Dependencies

The package uses `onnxruntime_core_vendor` for the ONNX Runtime C++ API.
Install the CUDA plugin and its vendor dependencies only when CUDA conversion
is required.

```bash
# CPU
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions

# Optional CUDA backend
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-cuda-plugin
```

For a source workspace:

```bash
colcon build --packages-up-to onnxruntime_conversions
colcon build --packages-up-to onnxruntime_conversions_cuda_plugin
```

## Basic use

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
owns the backend lease that protects the message storage.

`from_input_tensor_msg()` creates a read lease,
`from_output_tensor_msg()` creates a write lease, and `to_tensor_msg()` copies
an existing `Ort::Value` into message storage.

## Backend selection

Pass `"cpu"` or `"cuda"` to request a backend explicitly. Passing `"auto"`
uses the highest-priority backend that supports the supplied
`BackendConfiguration`.

CUDA operations require a non-null explicit stream:

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
constructing an ONNX Runtime session. `available_backends()` returns the
successfully loaded backend IDs.

Python conversions are provided separately by
`onnxruntime_conversions_py_cpu` and `onnxruntime_conversions_py_cuda`.
