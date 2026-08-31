# onnxruntime_gpu_vendor

This package installs the official ONNX Runtime C++ archive selected when the
package is configured from source. It uses the CPU archive when CUDA is absent,
and the matching CUDA 12 or CUDA 13 GPU archive when that toolkit is found.

Set `ONNXRUNTIME_GPU_VENDOR_VARIANT` to `cpu`, `cuda12`, or `cuda13` to override
auto-detection. `ONNXRUNTIME_GPU_VENDOR_VERSION` and
`ONNXRUNTIME_GPU_VENDOR_SHA256` can select another official release.

Binary releases contain the variant selected when the binary was built; they
do not select a variant when installed.
