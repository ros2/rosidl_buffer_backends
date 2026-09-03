# onnxruntime_core_vendor

This package installs the headers, core runtime, and shared provider support
library from a pinned official ONNX Runtime archive. It does not install the
CUDA execution provider and has no CUDA dependency, so the staged runtime can
be installed and used on CPU-only hosts.

On x86_64, the default source archive is ONNX Runtime 1.28.0 for CUDA 12. Set
`ONNXRUNTIME_CORE_VENDOR_VARIANT` to `cuda13` only to choose the GPU archive
from which the core files are extracted. This setting does not enable CUDA in
this package or in consumers.

On arm64, including JetPack hosts, the package uses the official
`onnxruntime-linux-aarch64` CPU archive. Native C++ CPU conversions are
supported. Python also requires a matching ONNX Runtime wheel and Python 3.11
or newer for ONNX Runtime 1.28; this excludes JetPack 6's default Python 3.10.
The archive has no CUDA execution provider, so JetPack GPU execution is not
supported by this package.

`ONNXRUNTIME_CORE_VENDOR_VERSION` and `ONNXRUNTIME_CORE_VENDOR_SHA256` provide
coordinated release overrides.

Install `onnxruntime_cuda_vendor` built against the same archive to
add the runtime-discovered CUDA execution-provider plugin on x86_64. The core
package remains the only owner of the headers, `libonnxruntime.so*`, and
`libonnxruntime_providers_shared.so*`.

Build and install this package first. For CUDA, install the provider package
second into the same merged prefix; ONNX Runtime searches for provider plugins
beside the physical core library and cannot discover one in an isolated
sibling prefix.
