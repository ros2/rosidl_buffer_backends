# python_onnxruntime_vendor

This package installs the CPU-only `onnxruntime` Python wheel coordinated with
`onnxruntime_core_vendor`. It has no CUDA selector or CUDA provider dependency.

The wheel is installed without dependencies. Its native libraries are removed
and replaced by relative links to canonical split-package files:

- `libonnxruntime.so*` and `libonnxruntime_providers_shared.so*` come from
  `onnxruntime_core_vendor`.
- No CUDA or TensorRT provider library is installed.

Binary packages retain the Python ABI, platform, ONNX Runtime version, and
CPU runtime selected when they were built.

This package and `python_onnxruntime_cuda_vendor` are mutually exclusive
because both install the same Python `onnxruntime` import path. CUDA users
install only the CUDA vendor; its wheel also includes the CPU execution
provider.
