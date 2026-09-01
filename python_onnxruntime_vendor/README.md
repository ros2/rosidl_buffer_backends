# python_onnxruntime_vendor

This package installs the Python ONNX Runtime wheel that matches
`onnxruntime_gpu_vendor`. The native vendor's fixed `cpu`, `cuda12`, or
`cuda13` variant and version select the wheel:

- `cpu`: `onnxruntime` from PyPI
- `cuda12`: `onnxruntime-gpu` from Microsoft's official CUDA 12 feed
- `cuda13`: `onnxruntime-gpu` from PyPI

The wheel is installed without dependencies. Its ONNX Runtime native
libraries are replaced by relative links to the canonical libraries under
`opt/onnxruntime_gpu_vendor`, avoiding duplicate runtime copies. TensorRT is
not included.

Binary packages retain the Python ABI, platform, ONNX Runtime version, and
runtime variant selected when they were built.
