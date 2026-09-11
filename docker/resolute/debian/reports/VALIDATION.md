# Torch Debian validation

Date: 2026-09-11 (UTC)

## Result

PASS — all 15 repository packages were generated as binary Debians using
standard Bloom/debhelper rules in a clean ROS 2 Rolling/Ubuntu Resolute Docker
builder. The builder started without a GPU device, NVIDIA driver, CUDA toolkit,
LibTorch, or Python Torch. Build and runtime dependencies were resolved from
the package manifests with rosdep; unreleased repository packages used the
local mappings that their future rosdistro entries will provide.

The generated artifacts were consumed from a local APT repository with the
network repositories disabled. APT was pinned to the exact locally generated
`0.1.2-0resolute` revision so a cached higher-revision public package could not
be substituted. The local repository contains 77 cached dependency and
generated Debian artifacts, with a `SHA256SUMS` manifest alongside them.

## Installed-package tests

Production conversion Debians contain no test or launch assets. After
installing the Debians, the test consumer copied a dedicated source-only test
package into `/tmp`, compiled the original C++ tests and ROS nodes against the
installed packages, generated launch files from the original source, and ran
them manually.

### Stage 1: core and CPU plugins only

- C++ CPU unit tests: 11/11 passed.
- Python CPU unit tests: 24/24 passed.
- C++ ROS launch scenarios: 4/4 passed (intra-process, inter-process,
  multi-publisher, and multi-subscriber).
- Python ROS launch scenario: 1/1 passed.
- The CUDA C++ and Python plugin packages were confirmed absent.
- No `nvidia-cuda*` Debian was installed by the CPU package set.
- `libtorch_conversions.so` resolved LibTorch from
  `/opt/ros/rolling/opt/libtorch_vendor`.
- The runtime probe selected `cpu`.

### Stage 2: install CUDA plugins without replacing the cores

- Installed only `ros-rolling-torch-conversions-cuda` and
  `ros-rolling-torch-conversions-py-cuda` from the same offline repository.
- SHA-256 checks confirmed that the C++ core library, Python core module,
  LibTorch provider, and Python Torch native provider were all byte-identical
  before and after plugin installation.
- The newly sourced process resolved `libtorch_conversions.so` from
  `/opt/ros/rolling/opt/libtorch_cuda_vendor` and imported Python Torch
  `2.9.1+cu126` from `python3_torch_cuda_vendor`.
- GPU used: NVIDIA GeForce RTX 4080, driver 610.57.04.

### Stage 3: installed CUDA plugins

- C++ CUDA unit tests: 4/4 passed.
- Python CUDA unit tests: 7/7 passed.
- C++ ROS launch scenarios on CUDA: 4/4 passed.
- Python ROS launch scenario on CUDA: 1/1 passed.
- A single Python process switched CPU → CUDA → CPU successfully.
- Final marker: `staged-debian-installed-source-tests=pass`.

## Production package-content audit

All six conversion Debians were checked with `dpkg-deb -c`; none ships a
`test`, `tests`, or `launch` directory or a launch Python file:

- `ros-rolling-torch-conversions`
- `ros-rolling-torch-conversions-cpu`
- `ros-rolling-torch-conversions-cuda`
- `ros-rolling-torch-conversions-py`
- `ros-rolling-torch-conversions-py-cpu`
- `ros-rolling-torch-conversions-py-cuda`

The complete command output is in
[`torch-debian-installed-tests.log`](torch-debian-installed-tests.log).

## Final source audit

- All 16 discovered package manifests are valid and use version `0.1.2`.
- All Torch providers select PyTorch/LibTorch `2.9.1`.
- CUDA 12 and 13 variant-policy tests pass for the C++ and Python providers.
- ROS flake8, pep257, cpplint, uncrustify, XML, CMake, and copyright checks
  pass.
- Shell syntax, Python syntax, README links, and `git diff --check` pass.
- ROS package discovery reports the four Torch providers and six conversion
  packages; no removed `dlpack_conversions` package remains.
