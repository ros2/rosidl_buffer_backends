# Copyright 2026 Open Source Robotics Foundation, Inc.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(torch_conversions_cuda)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
message(STATUS "${PROJECT_NAME}: CUDA ${CUDAToolkit_VERSION} from ${CUDAToolkit_BIN_DIR}")
find_package(cuda_buffer REQUIRED)
find_package(libtorch_cuda_vendor REQUIRED)
find_package(Torch REQUIRED)
find_package(pluginlib REQUIRED)
find_package(torch_conversions REQUIRED)

add_library(${PROJECT_NAME} SHARED src/cuda_conversion_plugin.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
  CUDA::cudart
  cuda_buffer::cuda_buffer
  torch
  torch_conversions::torch_conversions
  pluginlib::pluginlib
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

pluginlib_export_plugin_description_file(
  torch_conversions
  torch_conversions_cuda.xml
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
#include "torch_conversions/torch_conversions.hpp"

int main()
{
  return torch::cuda::is_available() && torch_conversions::backend_available("cuda") ? 0 : 77;
}
]=])
  add_executable(check_cuda_available "${CMAKE_CURRENT_BINARY_DIR}/check_cuda_available.cpp")
  target_link_libraries(check_cuda_available
    torch_conversions::torch_conversions torch)

  ament_add_gtest(test_torch_conversions_cuda test/test_torch_conversions_cuda.cpp)
  if(TARGET test_torch_conversions_cuda)
    target_link_libraries(test_torch_conversions_cuda
      torch_conversions::torch_conversions
      torch CUDA::cudart)
  endif()

  add_launch_test(test/test_torch_tensor_intra_pubsub_fastrtps_launch.py
    TARGET test_torch_tensor_intra_pubsub_fastrtps_launch TIMEOUT 30
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
  add_launch_test(test/test_torch_tensor_inter_pubsub_fastrtps_launch.py
    TARGET test_torch_tensor_inter_pubsub_fastrtps_launch TIMEOUT 30
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
  add_launch_test(test/test_torch_tensor_multi_pub_fastrtps_launch.py
    TARGET test_torch_tensor_multi_pub_fastrtps_launch TIMEOUT 45
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
  add_launch_test(test/test_torch_tensor_multi_sub_fastrtps_launch.py
    TARGET test_torch_tensor_multi_sub_fastrtps_launch TIMEOUT 45
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
endif()

ament_package()
