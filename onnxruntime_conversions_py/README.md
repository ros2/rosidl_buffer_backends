# onnxruntime_conversions_py

Python conversions between `tensor_msgs/msg/ExperimentalTensor` and
`onnxruntime.OrtValue`.

The package supports CPU storage everywhere and CUDA storage when both
`cuda_buffer_py` and ONNX Runtime's `CUDAExecutionProvider` are installed.
CUDA imports are lazy, so CPU use does not require CUDA.

## Dependency

Install Python `onnxruntime>=1.28` in the same environment as ROS. For CUDA,
install the matching `onnxruntime-gpu` distribution instead. ONNX
Runtime's Python package remains an external source-workspace dependency;
this package does not claim that a matching ROS buildfarm binary is available.

The public rosdep keys `python3-onnxruntime-pip` and
`python3-onnxruntime-gpu-pip` can install one selected runtime for source
workspaces. Direct CPU installation can use:

```bash
python -m pip install onnxruntime
```

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
