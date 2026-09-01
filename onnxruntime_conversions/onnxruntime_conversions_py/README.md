# onnxruntime_conversions_py

Python conversions between `tensor_msgs/msg/ExperimentalTensor` and
`onnxruntime.OrtValue`.

The package supports CPU storage everywhere and CUDA storage when both
`cuda_buffer_py` and the vendor's CUDA runtime are available. CUDA imports are
lazy, so CPU use does not require CUDA.

## Dependency

`python_onnxruntime_vendor` supplies the Python runtime coordinated with the
`onnxruntime_gpu_vendor` C++ SDK; no external `onnxruntime` installation is
required. The CPU, CUDA 12, or CUDA 13 variant is selected when the native
vendor package is built and remains fixed in binary packages.

The vendor's Python runtime is specific to the Python ABI, operating system,
and architecture for which its wheel was built. Use a vendor binary compatible
with the Python interpreter and platform in the ROS environment.

## Usage

```python
import numpy as np
import onnxruntime as ort

from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg

msg = allocate_tensor_msg((1, 4), np.float32)
with from_output_tensor_msg(msg) as output:
    binding.bind_ortvalue_output('output', output)
    session.run_with_iobinding(binding)
    binding.clear_binding_outputs()
```

An output view must be closed after synchronous inference and before the
message is published. When using CUDA, pass the same explicit stream to the
conversion functions and to the ONNX Runtime CUDA execution provider through
its `user_compute_stream` option.
