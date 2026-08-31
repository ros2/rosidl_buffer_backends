# onnxruntime_cuda_vendor

`onnxruntime_cuda_vendor` installs the official Microsoft ONNX Runtime 1.28.0
GPU archive for CUDA 13 on amd64. It exports the CMake target
`onnxruntime_cuda::onnxruntime`.

The package is intentionally deterministic: it does not inspect the build host
or select another CPU, CUDA, or TensorRT variant. Unsupported architectures
fail during CMake configuration. The installed runtime contains
`libonnxruntime`, the shared provider support library, and the CUDA provider.
The TensorRT provider from the upstream archive is excluded.

CUDA 13 and cuDNN 9 must be installed through the `nvidia-cuda` and
`nvidia-cudnn` rosdep keys. Runtime library discovery is configured through an
ament environment hook and the exported target's rpath.

The vendored ONNX Runtime binaries are distributed under the MIT license in
`LICENSE_ONNXRUNTIME`.
