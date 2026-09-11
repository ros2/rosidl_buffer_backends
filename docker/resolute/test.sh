#!/usr/bin/env bash
set -euo pipefail

set +u
source "/opt/ros/${ROS_DISTRO:-rolling}/setup.bash"
source /ws/install/setup.bash
source /ws/test_install/setup.bash
set -u

test_package=torch_conversions_installed_tests
test_bin=/ws/test_install/${test_package}/lib/${test_package}
launch_dir=/ws/test_install/${test_package}/share/${test_package}/test
python_test_root=/ws/src/rosidl_buffer_backends/torch_conversions/torch_conversions_py/test

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

backend=cpu
cuda_argument=()
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  backend=cuda
  cuda_argument=(--require-cuda)
fi

ROSIDL_TENSOR_BACKEND="${backend}" "${test_bin}/runtime_probe" "${cuda_argument[@]}"
"${test_bin}/test_torch_conversions"
python3 -m pytest -p no:cacheprovider -v \
  "${python_test_root}/test_torch_conversions_py.py"

if [[ "${backend}" == cuda ]]; then
  "${test_bin}/test_torch_conversions_cuda"
  python3 -m pytest -p no:cacheprovider -v \
    "${python_test_root}/test_torch_conversions_py_cuda.py"
fi

ROSIDL_TENSOR_BACKEND="${backend}" run_cpp_launch_tests
ROSIDL_TENSOR_BACKEND="${backend}" run_python_launch_test
