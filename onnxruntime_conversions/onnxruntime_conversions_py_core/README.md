# onnxruntime_conversions_py_core

Python conversions between `tensor_msgs/msg/ExperimentalTensor` and
`onnxruntime.OrtValue`. The ROS package is
`onnxruntime_conversions_py_core`; the Python import remains
`onnxruntime_conversions`.

The package supports CPU storage everywhere and CUDA storage when both
`cuda_buffer_py` and `python_onnxruntime_cuda_vendor` are installed. CUDA
imports are lazy, so CPU use does not require CUDA.

## Dependency

This package contains vendor-neutral conversion code and does not depend on a
specific Python ONNX Runtime vendor. Install exactly one runtime metapackage:

- `onnxruntime_conversions_py_cpu` installs this package with the CPU-only
  `python_onnxruntime_vendor`.
- `onnxruntime_conversions_py_cuda` installs this package with
  `python_onnxruntime_cuda_vendor`, `cuda_buffer_py`, and
  `cuda_buffer_backend`.

The CPU and CUDA Python vendors are mutually exclusive because both own the
same `onnxruntime` import path. The CUDA wheel includes the CPU execution
provider, so CUDA users do not install the CPU vendor.

The vendor's Python runtime is specific to the Python ABI, operating system,
and architecture for which its wheel was built. Use a vendor binary compatible
with the Python interpreter and platform in the ROS environment.

GPU native vendors and the CUDA Python vendor must use a merged install so
ONNX Runtime can discover the CUDA provider beside the core library.

Install the CPU runtime from apt:

```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py
```

Install the CUDA runtime from apt:

```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-onnxruntime-conversions-py-cuda
```

For a source workspace containing this repository, build one path only:

```bash
# CPU
colcon build --merge-install --packages-up-to onnxruntime_conversions_py_cpu

# CUDA, in a separate workspace/install prefix from the CPU Python vendor
colcon build --merge-install --packages-up-to onnxruntime_conversions_py_cuda
```

## Usage

```python
import numpy as np
import onnxruntime as ort

from onnxruntime_conversions import allocate_tensor_msg
from onnxruntime_conversions import from_output_tensor_msg

stream = create_cuda_stream()
try:
    session = ort.InferenceSession(
        model,
        providers=[
            (
                'CUDAExecutionProvider',
                {'user_compute_stream': str(stream)},
            ),
            'CPUExecutionProvider',
        ],
    )
    msg = allocate_tensor_msg((1, 4), np.float32, stream=stream)
    with from_output_tensor_msg(msg, stream=stream) as output:
        binding.bind_ortvalue_output('output', output.value)
        session.run_with_iobinding(binding)
        binding.clear_binding_outputs()
finally:
    destroy_cuda_stream(stream)
```

An output view must be closed after synchronous inference and before the
message is published. The application owns the CUDA stream and must keep it
alive until all conversions, inference, and queued work are complete. Create
the stream with the CUDA runtime or the application's existing CUDA API, pass
its nonzero integer `cudaStream_t` pointer to every conversion, and pass the
same pointer as a decimal string to the ONNX Runtime CUDA execution provider's
`user_compute_stream` option. CUDA conversions reject an omitted or zero
stream. Views over CPU-backed messages continue to require `stream=None`.

The default `device_type='auto'` selects CUDA only when the stream is positive
and nonzero, the CUDA Execution Provider is present, `cuda_buffer_py` imports,
and the CUDA runtime validates the device and stream. Otherwise allocation
selects CPU. `device_type='cpu'` always selects CPU. `device_type='cuda'` is
strict and raises when any CUDA requirement is unavailable or unusable.
Failures after CUDA has been selected are returned to the caller without a CPU
retry. CUDA imports remain lazy.
