# Copyright 2026 Open Source Robotics Foundation, Inc.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions_cuda)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
message(STATUS "${PROJECT_NAME}: CUDA ${CUDAToolkit_VERSION} from ${CUDAToolkit_BIN_DIR}")
find_package(cuda_buffer REQUIRED)
find_package(onnxruntime_cuda_vendor REQUIRED)
find_package(onnxruntime_conversions REQUIRED)
find_package(pluginlib REQUIRED)

add_library(${PROJECT_NAME} SHARED src/cuda_conversion_plugin.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
  CUDA::cudart
  cuda_buffer::cuda_buffer
  onnxruntime::onnxruntime
  onnxruntime_conversions::onnxruntime_conversions
  pluginlib::pluginlib
)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

pluginlib_export_plugin_description_file(
  onnxruntime_conversions onnxruntime_conversions_cuda.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(ament_cmake_ros REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE ament_cmake_cppcheck)
  ament_cppcheck(LIBRARIES googletest)
  ament_lint_auto_find_test_dependencies()

  add_executable(check_cuda_available test/check_cuda_available.cpp)
  target_link_libraries(check_cuda_available
    onnxruntime_conversions::onnxruntime_conversions)

  ament_add_gtest(test_onnxruntime_conversions_cuda test/test_onnxruntime_conversions_cuda.cpp)
  if(TARGET test_onnxruntime_conversions_cuda)
    target_link_libraries(test_onnxruntime_conversions_cuda
      onnxruntime_conversions::onnxruntime_conversions
      onnxruntime::onnxruntime CUDA::cudart)
  endif()

  add_launch_test(test/test_ort_tensor_inter_pubsub_fastrtps_launch.py
    TARGET test_ort_tensor_inter_pubsub_fastrtps_launch TIMEOUT 30
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
endif()

ament_package()
