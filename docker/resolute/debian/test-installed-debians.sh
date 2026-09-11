#!/usr/bin/env bash
set -euo pipefail

source_root="${1:?source root is required}"
report_file="${2:?report file is required}"
test_source="${source_root}/docker/resolute/debian/installed_tests/torch_conversions_installed_tests"
test_workspace=/tmp/torch-conversions-installed-test-ws
install_prefix="${test_workspace}/install"
test_package=torch_conversions_installed_tests

mkdir -p "$(dirname "${report_file}")"
exec > >(tee "${report_file}") 2>&1

set +u
source /opt/ros/rolling/setup.bash
set -u

core_library=/opt/ros/rolling/lib/libtorch_conversions.so
python_core="$(python3 -c 'import torch_conversions._core; print(torch_conversions._core.__file__)')"
cpu_provider_library=/opt/ros/rolling/opt/libtorch_vendor/lib/libtorch.so
python_cpu_provider="$(python3 -c 'import torch; print(torch._C.__file__)')"

test -f "${core_library}"
test -f "${python_core}"
test -f "${cpu_provider_library}"
test -f "${python_cpu_provider}"
cpu_linkage="$(ldd "${core_library}")"
grep -F '/opt/ros/rolling/opt/libtorch_vendor/' <<<"${cpu_linkage}"
! dpkg-query -W ros-rolling-torch-conversions-cuda >/dev/null 2>&1
! dpkg-query -W ros-rolling-torch-conversions-py-cuda >/dev/null 2>&1
! dpkg-query -W ros-rolling-libtorch-cuda-vendor >/dev/null 2>&1
! dpkg-query -W ros-rolling-python3-torch-cuda-vendor >/dev/null 2>&1

mkdir -p /root/.ros/rosdep
cp -a /var/local/ros-debs/rosdep-sources-cache /root/.ros/rosdep/sources.cache
rosdep install \
  --from-paths "${test_source}" \
  --ignore-src \
  --skip-keys "libtorch_vendor python3_torch_vendor torch_conversions torch_conversions_cpu torch_conversions_py torch_conversions_py_cpu" \
  --rosdistro rolling \
  --as-root apt:true \
  -y

mkdir -p "${test_workspace}/src"
cp -a "${test_source}" "${test_workspace}/src/"
colcon --log-base "${test_workspace}/log" build \
  --base-paths "${test_workspace}/src" \
  --build-base "${test_workspace}/build" \
  --install-base "${install_prefix}" \
  --packages-select "${test_package}" \
  --cmake-args \
    -DBUILD_TESTING=ON \
    -DBUILD_CUDA_TESTS=OFF \
    -DBACKENDS_SOURCE_ROOT="${source_root}"
set +u
source "${install_prefix}/setup.bash"
set -u

test_bin="${install_prefix}/${test_package}/lib/${test_package}"
launch_dir="${install_prefix}/${test_package}/share/${test_package}/test"
python_test_root="${source_root}/torch_conversions/torch_conversions_py/test"

run_cpp_launch_tests() {
  local launch_file
  for launch_file in \
      test_torch_tensor_intra_pubsub_fastrtps_launch.py \
      test_torch_tensor_inter_pubsub_fastrtps_launch.py \
      test_torch_tensor_multi_pub_fastrtps_launch.py \
      test_torch_tensor_multi_sub_fastrtps_launch.py; do
    launch_test "${launch_dir}/${launch_file}"
  done
}

run_python_launch_test() {
  launch_test "${launch_dir}/test_torch_tensor_inter_pubsub_fastrtps.py"
}

echo '=== Stage 1: installed core plus CPU plugins ==='
ROSIDL_TENSOR_BACKEND=cpu "${test_bin}/runtime_probe"
"${test_bin}/test_torch_conversions"
python3 -m pytest -p no:cacheprovider -v "${python_test_root}/test_torch_conversions_py.py"
ROSIDL_TENSOR_BACKEND=cpu run_cpp_launch_tests
ROSIDL_TENSOR_BACKEND=cpu run_python_launch_test

core_checksum_before="$(sha256sum "${core_library}" | cut -d' ' -f1)"
python_checksum_before="$(sha256sum "${python_core}" | cut -d' ' -f1)"
provider_checksum_before="$(sha256sum "${cpu_provider_library}" | cut -d' ' -f1)"
python_provider_checksum_before="$(sha256sum "${python_cpu_provider}" | cut -d' ' -f1)"
echo "cpp-core-sha256-before=${core_checksum_before}"
echo "python-core-sha256-before=${python_checksum_before}"
echo "cpp-provider-sha256-before=${provider_checksum_before}"
echo "python-provider-sha256-before=${python_provider_checksum_before}"

echo '=== Stage 2: install independently built CUDA plugins ==='
apt-get install --no-install-recommends -y \
  ros-rolling-torch-conversions-cuda \
  ros-rolling-torch-conversions-py-cuda

core_checksum_after="$(sha256sum "${core_library}" | cut -d' ' -f1)"
python_checksum_after="$(sha256sum "${python_core}" | cut -d' ' -f1)"
provider_checksum_after="$(sha256sum "${cpu_provider_library}" | cut -d' ' -f1)"
python_provider_checksum_after="$(sha256sum "${python_cpu_provider}" | cut -d' ' -f1)"
test "${core_checksum_before}" = "${core_checksum_after}"
test "${python_checksum_before}" = "${python_checksum_after}"
test "${provider_checksum_before}" = "${provider_checksum_after}"
test "${python_provider_checksum_before}" = "${python_provider_checksum_after}"
echo "cpp-core-sha256-after=${core_checksum_after}"
echo "python-core-sha256-after=${python_checksum_after}"
echo "cpp-provider-sha256-after=${provider_checksum_after}"
echo "python-provider-sha256-after=${python_provider_checksum_after}"
echo 'core-files-unchanged=pass'
echo 'provider-files-unchanged=pass'

cuda_provider_library=/opt/ros/rolling/opt/libtorch_cuda_vendor/lib/libtorch.so
test -f "${cuda_provider_library}"

set +u
source /opt/ros/rolling/setup.bash
set -u
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python3 -c \
  'import torch; assert torch.__version__.split("+")[0] == "2.9.1"; assert torch.version.cuda is not None; print(torch.__version__, torch._C.__file__)'
cuda_linkage="$(ldd "${core_library}")"
grep -F '/opt/ros/rolling/opt/libtorch_cuda_vendor/' <<<"${cuda_linkage}"

# Reconfigure the source-only installed tests against the CUDA provider only
# after the CUDA Debians have been installed. The CPU stage intentionally has
# neither the CUDA provider nor its generated CUDA headers.
colcon --log-base "${test_workspace}/log-cuda" build \
  --base-paths "${test_workspace}/src" \
  --build-base "${test_workspace}/build-cuda" \
  --install-base "${install_prefix}-cuda" \
  --packages-select "${test_package}" \
  --cmake-args \
    -DBUILD_TESTING=ON \
    -DBUILD_CUDA_TESTS=ON \
    -DBACKENDS_SOURCE_ROOT="${source_root}"
set +u
source "${install_prefix}-cuda/setup.bash"
set -u
test_bin="${install_prefix}-cuda/${test_package}/lib/${test_package}"
launch_dir="${install_prefix}-cuda/${test_package}/share/${test_package}/test"

echo '=== Stage 3: CUDA default plus CPU/CUDA runtime selection ==='
ROSIDL_TENSOR_BACKEND=cuda "${test_bin}/runtime_probe" --require-cuda
"${test_bin}/test_torch_conversions_cuda"
python3 -m pytest -p no:cacheprovider -v "${python_test_root}/test_torch_conversions_py_cuda.py"
ROSIDL_TENSOR_BACKEND=cuda run_cpp_launch_tests
ROSIDL_TENSOR_BACKEND=cuda run_python_launch_test

python3 - <<'PY'
import torch
import torch_conversions

assert torch_conversions.available_backends() == ['cpu', 'cuda']
for backend in ('cpu', 'cuda', 'cpu'):
    device = backend
    msg = torch_conversions.allocate_tensor_msg((8,), torch.float32, device)
    tensor = torch_conversions.from_output_tensor_msg(msg)
    tensor.fill_(3)
    assert torch_conversions.from_input_tensor_msg(msg).sum().item() == 24
print('python-runtime-switch=pass')
PY

echo '=== Installed package versions ==='
dpkg-query -W \
  ros-rolling-libtorch-vendor \
  ros-rolling-libtorch-cuda-vendor \
  ros-rolling-python3-torch-vendor \
  ros-rolling-python3-torch-cuda-vendor \
  ros-rolling-torch-conversions \
  ros-rolling-torch-conversions-cpu \
  ros-rolling-torch-conversions-cuda \
  ros-rolling-torch-conversions-py \
  ros-rolling-torch-conversions-py-cpu \
  ros-rolling-torch-conversions-py-cuda

echo 'staged-debian-installed-source-tests=pass'
