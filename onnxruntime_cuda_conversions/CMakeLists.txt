# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(onnxruntime_cuda_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(cuda_buffer REQUIRED)
find_package(onnxruntime_cuda_vendor REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(tensor_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/onnxruntime_cuda_conversions.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    onnxruntime_cuda::onnxruntime
    rosidl_buffer::rosidl_buffer
    ${tensor_msgs_TARGETS}
  PRIVATE
    cuda_buffer::cuda_buffer
    CUDA::cudart
    CUDA::cuda_driver
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  CUDAToolkit
  cuda_buffer
  onnxruntime_cuda_vendor
  rosidl_buffer
  tensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_onnxruntime_cuda_conversions
    test/test_onnxruntime_cuda_conversions.cpp
  )
  if(TARGET test_onnxruntime_cuda_conversions)
    target_link_libraries(test_onnxruntime_cuda_conversions
      ${PROJECT_NAME}
      onnxruntime_cuda::onnxruntime
      rosidl_buffer::rosidl_buffer
      cuda_buffer::cuda_buffer
      CUDA::cudart
      CUDA::cuda_driver
      ${tensor_msgs_TARGETS}
    )
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(std_msgs REQUIRED)

  add_library(onnxruntime_cuda_tensor_publisher_component SHARED
    test/src/onnxruntime_cuda_tensor_publisher_node.cpp
  )
  set_target_properties(onnxruntime_cuda_tensor_publisher_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(onnxruntime_cuda_tensor_publisher_component
    ${PROJECT_NAME}
    onnxruntime_cuda::onnxruntime
    rclcpp::rclcpp
    rclcpp_components::component
    CUDA::cudart
    CUDA::cuda_driver
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
  )
  rclcpp_components_register_node(onnxruntime_cuda_tensor_publisher_component
    PLUGIN "OnnxRuntimeCudaTensorPublisher"
    EXECUTABLE onnxruntime_cuda_tensor_publisher_node
  )

  add_library(onnxruntime_cuda_tensor_subscriber_component SHARED
    test/src/onnxruntime_cuda_tensor_subscriber_node.cpp
  )
  set_target_properties(onnxruntime_cuda_tensor_subscriber_component PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_link_libraries(onnxruntime_cuda_tensor_subscriber_component
    ${PROJECT_NAME}
    onnxruntime_cuda::onnxruntime
    rclcpp::rclcpp
    rclcpp_components::component
    CUDA::cudart
    CUDA::cuda_driver
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS}
  )
  rclcpp_components_register_node(onnxruntime_cuda_tensor_subscriber_component
    PLUGIN "OnnxRuntimeCudaTensorSubscriber"
    EXECUTABLE onnxruntime_cuda_tensor_subscriber_node
  )

  install(
    TARGETS
      onnxruntime_cuda_tensor_publisher_component
      onnxruntime_cuda_tensor_subscriber_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )

  add_launch_test(
    test/test_onnxruntime_cuda_tensor_inter_pubsub_fastrtps_launch.py
    TARGET test_onnxruntime_cuda_tensor_inter_pubsub_fastrtps
    TIMEOUT 40
  )
endif()

ament_package()
