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
    msg = allocate_tensor_msg((1, 4), np.float32, device_type='cuda')
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
stream. CPU conversions continue to require `stream=None`.
