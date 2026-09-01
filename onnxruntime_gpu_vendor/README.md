# onnxruntime_gpu_vendor

This package installs the official ONNX Runtime C++ SDK and native runtime
selected when the package is configured from source. It uses the CPU archive
when CUDA is absent and the matching CUDA 12 or CUDA 13 GPU archive otherwise.

Set `ONNXRUNTIME_GPU_VENDOR_VARIANT` to `cpu`, `cuda12`, or `cuda13` to override
auto-detection. `ONNXRUNTIME_GPU_VENDOR_VERSION` and
`ONNXRUNTIME_GPU_VENDOR_SHA256` can select another official release.

Binary releases contain the variant selected when the binary was built; they
do not select or change variants when installed.

The archive under `opt/onnxruntime_gpu_vendor` is the canonical native runtime.
`python_onnxruntime_vendor` installs the coordinated Python wheel and links it to
these libraries rather than retaining duplicate native runtime copies.
TensorRT is not included.

Use `onnxruntime_gpu_vendor` directly for C++ consumers and
`python_onnxruntime_vendor` for Python consumers.
