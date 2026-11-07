# Copyright 2026 Open Source Robotics Foundation, Inc.
# Copyright 2026 Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.28)
project(executorch_conversions_cuda)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(executorch_cuda_vendor REQUIRED)
if(EXECUTORCH_CUDA_VENDOR_SKIPPED)
  message(WARNING "${PROJECT_NAME}: skipping, because executorch_cuda_vendor installed no "
    "ExecuTorch: ${EXECUTORCH_CUDA_VENDOR_SKIPPED}.")
  ament_package()
  return()
endif()
find_package(CUDAToolkit REQUIRED)
# The plugin checks a caller's stream with cudaStreamGetDevice, which CUDA 12.8 added.
if(CUDAToolkit_VERSION VERSION_LESS 12.8)
  message(WARNING "${PROJECT_NAME}: skipping, because it needs CUDA 12.8 or newer, and the "
    "toolkit is CUDA ${CUDAToolkit_VERSION}.")
  ament_package()
  return()
endif()
message(STATUS "${PROJECT_NAME}: CUDA ${CUDAToolkit_VERSION} from ${CUDAToolkit_BIN_DIR}")
find_package(cuda_buffer REQUIRED)
find_package(executorch ${EXECUTORCH_VENDOR_VERSION_RANGE} REQUIRED CONFIG COMPONENTS extension_cuda)
find_package(executorch_conversions REQUIRED)
find_package(pluginlib REQUIRED)

add_library(${PROJECT_NAME} SHARED src/cuda_conversion_plugin.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
  CUDA::cudart
  cuda_buffer::cuda_buffer
  executorch::extension_cuda
  executorch_conversions::executorch_conversions
  pluginlib::pluginlib
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

pluginlib_export_plugin_description_file(
  executorch_conversions
  executorch_conversions_cuda.xml
)

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(ament_cmake_ros REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE ament_cmake_cppcheck)
  ament_cppcheck(LIBRARIES googletest)
  ament_lint_auto_find_test_dependencies()

  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/check_cuda_available.cpp"
    CONTENT [=[
#include <cuda_runtime_api.h>

#include "executorch_conversions/executorch_conversions.hpp"

int main()
{
  int count = 0;
  const bool has_device = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  const bool has_plugin = executorch_conversions::backend_available("cuda");
  return has_device ? (has_plugin ? 0 : 1) : 77;
}
]=])
  add_executable(check_cuda_available "${CMAKE_CURRENT_BINARY_DIR}/check_cuda_available.cpp")
  target_link_libraries(check_cuda_available
    executorch_conversions::executorch_conversions CUDA::cudart)

  ament_add_gtest(test_executorch_conversions_cuda test/test_executorch_conversions_cuda.cpp)
  if(TARGET test_executorch_conversions_cuda)
    target_link_libraries(test_executorch_conversions_cuda
      executorch_conversions::executorch_conversions
      executorch::extension_cuda
      executorch::backend_cuda
      CUDA::cudart)
  endif()

  add_launch_test(test/test_et_tensor_intra_pubsub_fastrtps_launch.py
    TARGET test_et_tensor_intra_pubsub_fastrtps_launch TIMEOUT 30
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
  add_launch_test(test/test_et_tensor_inter_pubsub_fastrtps_launch.py
    TARGET test_et_tensor_inter_pubsub_fastrtps_launch TIMEOUT 30
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
  add_launch_test(test/test_et_tensor_multi_pub_fastrtps_launch.py
    TARGET test_et_tensor_multi_pub_fastrtps_launch TIMEOUT 45
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
  add_launch_test(test/test_et_tensor_multi_sub_fastrtps_launch.py
    TARGET test_et_tensor_multi_sub_fastrtps_launch TIMEOUT 45
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
endif()

ament_package()
