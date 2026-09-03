# onnxruntime_cuda_vendor

This optional package installs only `libonnxruntime_providers_cuda.so` beside
the runtime owned by `onnxruntime_core_vendor`. It requires CUDAToolkit and
does not install another copy of the ONNX Runtime core. It supplies the native
provider plugin; conversion support is registered separately at runtime by
`onnxruntime_conversions_cuda_plugin`.

The provider version, archive variant, and checksum must exactly match the
core package. They default to the values exported by
`onnxruntime_core_vendor`; mismatched overrides fail configuration. Configure
the core with `ONNXRUNTIME_CORE_VENDOR_VARIANT=cuda13` to use CUDA 13. CUDA 12
is the default.

ONNX Runtime resolves provider libraries from the physical directory that
contains `libonnxruntime.so`. Consequently, the core and provider packages
must be installed into one shared prefix, as they are for `/opt/ros` and
Debian packages. Separate isolated colcon prefixes do not satisfy this
runtime rule even when both prefixes are sourced. Development and
post-install tests must therefore use a merged install prefix.

Build `onnxruntime_core_vendor` first, this package second, and CUDA consumers
afterward. Selecting a GPU archive for the core is only an archive-provenance
choice; CUDA is available only when this provider is installed and a consumer
requests the CUDA execution provider.
