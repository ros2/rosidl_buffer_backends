# python_onnxruntime_cuda_vendor

This package installs the `onnxruntime-gpu` Python wheel matching
`onnxruntime_core_vendor` and `onnxruntime_cuda_vendor`. The CUDA 12
or CUDA 13 wheel is selected directly from the provider vendor configuration;
no environment variable or package-discovery condition is required.

The wheel is installed without dependencies. Its bundled native libraries are
removed and replaced by relative links to canonical files:

- `libonnxruntime.so*` and `libonnxruntime_providers_shared.so*` come from
  `onnxruntime_core_vendor`.
- `libonnxruntime_providers_cuda.so` comes from
  `onnxruntime_cuda_vendor`.

The provider and core versions and CUDA archive variants must match. TensorRT
is not included. Install the core and CUDA provider into one merged prefix
before building this package.

This package and `python_onnxruntime_vendor` are mutually exclusive because
both install the same Python `onnxruntime` import path. CUDA users install only
this package, which includes both CUDA and CPU execution providers.
