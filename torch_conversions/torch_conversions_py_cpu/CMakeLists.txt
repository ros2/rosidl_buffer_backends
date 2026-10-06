# Copyright 2026 Open Source Robotics Foundation, Inc.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(torch_conversions_py_cpu)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(torch_conversions_cpu)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION
    share/ament_index/resource_index/torch_conversions__python_plugins
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(ament_cmake_ros REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(test_torch_conversions_py_cpu
    test/test_torch_conversions_py_cpu.py TIMEOUT 90)

  add_launch_test(test/test_torch_tensor_inter_pubsub_fastrtps.py
    TARGET test_torch_tensor_inter_pubsub_fastrtps TIMEOUT 40
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
endif()

ament_package()
