#!/usr/bin/env bash
set -euo pipefail

set +u
source "/opt/ros/${ROS_DISTRO:-rolling}/setup.bash"
source /ws/install/setup.bash
source /ws/test_install/setup.bash
set -u

test_package=onnxruntime_conversions_installed_tests
test_bin=/ws/test_install/${test_package}/lib/${test_package}
launch_file=/ws/test_install/${test_package}/share/${test_package}/test/test_ort_tensor_inter_pubsub_fastrtps_launch.py
python_test_root=/ws/src/rosidl_buffer_backends/onnxruntime_conversions/onnxruntime_conversions_py/test

backend=cpu
cuda_argument=()
if python3 -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" 2>/dev/null; then
  backend=cuda
  cuda_argument=(--require-cuda)
fi

ROSIDL_TENSOR_BACKEND="${backend}" \
  "${test_bin}/runtime_probe" "${cuda_argument[@]}"
"${test_bin}/test_onnxruntime_conversions"
PYTHONPATH="${python_test_root}:${PYTHONPATH:-}" \
  python3 -m pytest -p no:cacheprovider -v \
  "${python_test_root}/test_onnxruntime_conversions_py.py"

if [[ "${backend}" == cuda ]]; then
  "${test_bin}/test_onnxruntime_conversions_cuda"
  PYTHONPATH="${python_test_root}:${PYTHONPATH:-}" \
    python3 -m pytest -p no:cacheprovider -v \
    "${python_test_root}/test_onnxruntime_conversions_py_cuda.py"
  launch_test "${launch_file}"
  PYTHONPATH="${python_test_root}:${PYTHONPATH:-}" \
    python3 -m pytest -p no:cacheprovider -v \
    "${python_test_root}/test_onnxruntime_tensor_inter_pubsub_fastrtps.py"
fi
