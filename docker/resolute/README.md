# ROS 2 Rolling on Ubuntu Resolute: PR #19

These Docker environments validate the ONNX Runtime conversion refactor on
top of
[`ros2/rosidl_buffer_backends#19`](https://github.com/ros2/rosidl_buffer_backends/pull/19).

All external build inputs are declared in `package.xml` and resolved through
rosdep. The containers start without CUDA, ONNX Runtime, an NVIDIA driver, or
a GPU device.

## Release-like Debian acceptance test

From the repository root:

```bash
bash docker/resolute/debian/test-pipeline.sh
```

The first container installs the dependency closure and builds the 15 project
Debians without GPU access. A second pristine container installs the C++ and
Python cores plus CPU plugins, compiles the production-Debian test sources in
an external workspace, and runs them manually. It then installs only the CUDA
plugins, proves the core files were not replaced, and runs the CUDA unit,
launch, process-boundary, and in-process backend-selection tests on the GPU.

See [`debian/README.md`](debian/README.md) for the exact package/test flow and
[`../../onnxruntime_conversions/README.md`](../../onnxruntime_conversions/README.md)
for the plugin and exclusive-provider boundary.

The consumer phase requires NVIDIA Container Toolkit and a compatible host
driver.
