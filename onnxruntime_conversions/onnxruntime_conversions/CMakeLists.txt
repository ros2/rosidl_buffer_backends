cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(onnxruntime_core_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tensor_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED src/onnxruntime_conversions.cpp)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME}
  PRIVATE ONNXRUNTIME_CONVERSIONS_BUILDING_DLL)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    onnxruntime::onnxruntime
    tensor_msgs::tensor_msgs__rosidl_generator_cpp
  PRIVATE
    pluginlib::pluginlib
    ${CMAKE_DL_LIBS}
)
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} NAMESPACE ${PROJECT_NAME}::)
ament_export_dependencies(onnxruntime_core_vendor pluginlib tensor_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(CUDAToolkit QUIET)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(ament_cmake_ros REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(rclcpp_components REQUIRED)
  find_package(rosidl_runtime_cpp REQUIRED)
  find_package(std_msgs REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_onnxruntime_conversions
    test/test_onnxruntime_conversions.cpp)
  if(TARGET test_onnxruntime_conversions)
    target_link_libraries(test_onnxruntime_conversions ${PROJECT_NAME})
  endif()

  if(CUDAToolkit_FOUND)
    ament_add_gtest(test_onnxruntime_conversions_cuda
      test/test_onnxruntime_conversions_cuda.cpp)
    if(TARGET test_onnxruntime_conversions_cuda)
      target_link_libraries(test_onnxruntime_conversions_cuda
        ${PROJECT_NAME} CUDA::cudart)
    endif()

  endif()

  add_library(ort_tensor_publisher_component SHARED
    test/src/ort_tensor_publisher_node.cpp)
  target_link_libraries(ort_tensor_publisher_component
    ${PROJECT_NAME}
    rclcpp::rclcpp
    rclcpp_components::component
    rosidl_runtime_cpp::rosidl_runtime_cpp
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS})
  rclcpp_components_register_node(ort_tensor_publisher_component
    PLUGIN "OrtTensorPublisher"
    EXECUTABLE ort_tensor_publisher_node)

  add_library(ort_tensor_subscriber_component SHARED
    test/src/ort_tensor_subscriber_node.cpp)
  target_link_libraries(ort_tensor_subscriber_component
    ${PROJECT_NAME}
    rclcpp::rclcpp
    rclcpp_components::component
    rosidl_runtime_cpp::rosidl_runtime_cpp
    ${std_msgs_TARGETS}
    ${tensor_msgs_TARGETS})
  rclcpp_components_register_node(ort_tensor_subscriber_component
    PLUGIN "OrtTensorSubscriber"
    EXECUTABLE ort_tensor_subscriber_node)

  install(
    TARGETS
      ort_tensor_publisher_component
      ort_tensor_subscriber_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

  add_launch_test(test/test_ort_tensor_inter_pubsub_fastrtps_launch.py
    TARGET test_ort_tensor_inter_pubsub_fastrtps_launch TIMEOUT 30
    RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
endif()

ament_package(CONFIG_EXTRAS_POST cmake/onnxruntime_conversions-extras.cmake)
