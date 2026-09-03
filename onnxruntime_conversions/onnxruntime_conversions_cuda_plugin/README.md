# onnxruntime_conversions_cuda_plugin

This package provides the optional `cuda` plugin for
`onnxruntime_conversions`. It allocates `cuda_buffer` message storage and
configures the ONNX Runtime CUDA Execution Provider. Pluginlib discovers this
conversion backend at runtime; neither `onnxruntime_core_vendor` nor its source
archive enables CUDA in consumers at compile time.

Install `onnxruntime_core_vendor`, `onnxruntime_cuda_vendor`,
`onnxruntime_conversions`, and this package in that order. The core and
provider vendors must share a merged install prefix because ONNX Runtime
discovers `libonnxruntime_providers_cuda.so` beside `libonnxruntime.so`.
If this plugin cannot load, requesting `cuda` reports
`ONNX Runtime conversion backend 'cuda' is unavailable.` and includes the
discoverable classes and exact plugin load failure.

Every conversion and session configuration requires a non-null,
application-owned CUDA stream. Storage leases retain CUDA read or write
handles, and completion is ordered with CUDA events when the lease is
released. Device-to-device copies remain asynchronous; callers must keep the
source `Ort::Value` alive until work on the supplied stream has completed.

With backend `auto`, this plugin is selected only when CUDA hardware/runtime
and the configured device are usable and the application supplies a valid
non-null stream. Without that stream, `auto` selects CPU. Explicit `cuda`
remains strict, and an error after selection is never retried on CPU.

Zero-byte CUDA tensors are rejected because `cuda_buffer` has no typed device
pointer for empty storage and ONNX Runtime requires a valid device pointer
when constructing an externally backed CUDA tensor.
