#!/usr/bin/env bash
set -euo pipefail

source_root="${1:?source root is required}"
report_file="${2:?report file is required}"
test_source="${source_root}/docker/resolute/debian/installed_tests/onnxruntime_conversions_installed_tests"
test_workspace=/tmp/onnxruntime-conversions-installed-test-ws
install_prefix="${test_workspace}/install"
test_package=onnxruntime_conversions_installed_tests

mkdir -p "$(dirname "${report_file}")"
exec > >(tee "${report_file}") 2>&1

set +u
source /opt/ros/rolling/setup.bash
set -u

core_library=/opt/ros/rolling/lib/libonnxruntime_conversions.so
python_core="$(python3 -c 'import onnxruntime_conversions._core; print(onnxruntime_conversions._core.__file__)')"
provider_library="$(find /opt/ros/rolling/opt/onnxruntime_cuda_vendor/lib \
  -maxdepth 1 -type f -name 'libonnxruntime.so.*' -print -quit)"
python_provider="$(python3 -c 'import onnxruntime.capi.onnxruntime_pybind11_state as module; print(module.__file__)')"
test -f "${core_library}"
test -f "${python_core}"
test -f "${provider_library}"
test -f "${python_provider}"
cpp_provider_version="$(tr -d '[:space:]' \
  < /opt/ros/rolling/opt/onnxruntime_cuda_vendor/VERSION_NUMBER)"
python_provider_version="$(python3 -c 'import onnxruntime; print(onnxruntime.__version__)')"
test "${cpp_provider_version}" = 1.26.0
test "${python_provider_version}" = 1.26.0
python3 - <<'PY'
import onnxruntime as ort

assert hasattr(ort.OrtValue, 'from_dlpack')
assert hasattr(ort.OrtValue, '__dlpack__')
assert hasattr(ort.OrtValue, '__dlpack_device__')
PY
echo "cpp-provider-version=${cpp_provider_version}"
echo "python-provider-version=${python_provider_version}"
echo 'python-dlpack-api=pass'
! dpkg-query -W ros-rolling-onnxruntime-conversions-cuda >/dev/null 2>&1
! dpkg-query -W ros-rolling-onnxruntime-conversions-py-cuda >/dev/null 2>&1

mkdir -p /root/.ros/rosdep
cp -a /var/local/ros-debs/rosdep-sources-cache /root/.ros/rosdep/sources.cache
rosdep install \
  --from-paths "${test_source}" \
  --ignore-src \
  --skip-keys "onnxruntime_conversions onnxruntime_conversions_cpu onnxruntime_conversions_cuda onnxruntime_conversions_py onnxruntime_conversions_py_cpu onnxruntime_conversions_py_cuda onnxruntime_cuda_vendor" \
  --rosdistro rolling --as-root apt:true -y

mkdir -p "${test_workspace}/src"
cp -a "${test_source}" "${test_workspace}/src/"
colcon --log-base "${test_workspace}/log" build \
  --base-paths "${test_workspace}/src" \
  --build-base "${test_workspace}/build" \
  --install-base "${install_prefix}" \
  --packages-select "${test_package}" \
  --cmake-args -DBUILD_TESTING=ON -DBACKENDS_SOURCE_ROOT="${source_root}"
set +u
source "${install_prefix}/setup.bash"
set -u

test_bin="${install_prefix}/${test_package}/lib/${test_package}"
launch_file="${install_prefix}/${test_package}/share/${test_package}/test/test_ort_tensor_inter_pubsub_fastrtps_launch.py"
python_test_root="${source_root}/onnxruntime_conversions/onnxruntime_conversions_py/test"

echo '=== Stage 1: installed core plus CPU plugins ==='
ROSIDL_TENSOR_BACKEND=cpu "${test_bin}/runtime_probe"
"${test_bin}/test_onnxruntime_conversions"
PYTHONPATH="${python_test_root}:${PYTHONPATH:-}" \
  python3 -m pytest -p no:cacheprovider -v \
  "${python_test_root}/test_onnxruntime_conversions_py.py"

core_checksum_before="$(sha256sum "${core_library}" | cut -d' ' -f1)"
python_checksum_before="$(sha256sum "${python_core}" | cut -d' ' -f1)"
provider_checksum_before="$(sha256sum "${provider_library}" | cut -d' ' -f1)"
python_provider_checksum_before="$(sha256sum "${python_provider}" | cut -d' ' -f1)"
echo "cpp-core-sha256-before=${core_checksum_before}"
echo "python-core-sha256-before=${python_checksum_before}"
echo "cpp-provider-sha256-before=${provider_checksum_before}"
echo "python-provider-sha256-before=${python_provider_checksum_before}"

echo '=== Stage 2: install independently built CUDA plugins ==='
apt-get install --no-install-recommends -y \
  ros-rolling-onnxruntime-conversions-cuda \
  ros-rolling-onnxruntime-conversions-py-cuda

core_checksum_after="$(sha256sum "${core_library}" | cut -d' ' -f1)"
python_checksum_after="$(sha256sum "${python_core}" | cut -d' ' -f1)"
provider_checksum_after="$(sha256sum "${provider_library}" | cut -d' ' -f1)"
python_provider_checksum_after="$(sha256sum "${python_provider}" | cut -d' ' -f1)"
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

set +u
source /opt/ros/rolling/setup.bash
source "${install_prefix}/setup.bash"
set -u
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo '=== Stage 3: CUDA default plus CPU/CUDA runtime selection ==='
ROSIDL_TENSOR_BACKEND=cuda "${test_bin}/runtime_probe" --require-cuda
"${test_bin}/test_onnxruntime_conversions_cuda"
PYTHONPATH="${python_test_root}:${PYTHONPATH:-}" \
  python3 -m pytest -p no:cacheprovider -v \
  "${python_test_root}/test_onnxruntime_conversions_py_cuda.py"
launch_test "${launch_file}"
PYTHONPATH="${python_test_root}:${PYTHONPATH:-}" \
  python3 -m pytest -p no:cacheprovider -v \
  "${python_test_root}/test_onnxruntime_tensor_inter_pubsub_fastrtps.py"

python3 - <<'PY'
import ctypes
import numpy as np
import onnxruntime as ort
import onnxruntime_conversions as conversions

assert conversions.available_backends() == ['cpu', 'cuda']
assert 'CUDAExecutionProvider' in ort.get_available_providers()
runtime = ctypes.CDLL('libcudart.so')
runtime.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
runtime.cudaStreamCreateWithFlags.restype = ctypes.c_int
runtime.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
runtime.cudaStreamDestroy.restype = ctypes.c_int
stream = ctypes.c_void_p()
assert runtime.cudaStreamCreateWithFlags(ctypes.byref(stream), 1) == 0
try:
    for backend in ('cpu', 'cuda', 'cpu'):
        msg = conversions.allocate_tensor_msg((8,), np.float32, backend)
        selected_stream = stream.value if backend == 'cuda' else None
        with conversions.from_output_tensor_msg(msg, selected_stream) as value:
            value.update_inplace(np.full((8,), 3, dtype=np.float32))
        payload = msg.data if backend == 'cpu' else msg.data.to_bytes()
        assert np.frombuffer(payload, dtype=np.float32).sum() == 24
finally:
    assert runtime.cudaStreamDestroy(stream) == 0
print('python-runtime-switch=pass')
PY

echo '=== Installed package versions ==='
dpkg-query -W \
  ros-rolling-onnxruntime-cuda-vendor \
  ros-rolling-python-onnxruntime-cuda-vendor \
  ros-rolling-onnxruntime-conversions \
  ros-rolling-onnxruntime-conversions-cpu \
  ros-rolling-onnxruntime-conversions-cuda \
  ros-rolling-onnxruntime-conversions-py \
  ros-rolling-onnxruntime-conversions-py-cpu \
  ros-rolling-onnxruntime-conversions-py-cuda

echo 'staged-debian-installed-source-tests=pass'
