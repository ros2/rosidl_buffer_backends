# python_torch_vendor

This package installs the Python Torch distribution. It uses the same CUDA
detection, Torch version mapping, and platform reuse policy as
`libtorch_vendor`, without depending on that package.

On x86_64, the package downloads the matching wheel from the corresponding
PyTorch wheel index and installs it into the ROS prefix. On aarch64, it follows
`libtorch_vendor` by validating and reusing the platform-provided PyTorch
installation, such as the JetPack wheel on Jetson.

Set `FORCE_BUILD_VENDOR_PKG=ON` to install the selected wheel instead of
reusing an existing Python Torch installation.
