# The ExecuTorch CMake package defines its imported targets only on CMake 3.28+.
cmake_minimum_required(VERSION 3.28)
project(executorch_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(executorch_vendor REQUIRED)
find_package(executorch ${EXECUTORCH_VENDOR_VERSION_RANGE} REQUIRED CONFIG)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(tensor_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED src/executorch_conversions.cpp)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME}
  PRIVATE EXECUTORCH_CONVERSIONS_BUILDING_DLL)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    executorch::runtime
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
ament_export_dependencies(executorch_vendor pluginlib tensor_msgs)

add_library(et_tensor_publisher_component SHARED
  examples/et_tensor_publisher_node.cpp)
target_link_libraries(et_tensor_publisher_component
  ${PROJECT_NAME}
  rclcpp::rclcpp
  rclcpp_components::component
  rosidl_runtime_cpp::rosidl_runtime_cpp
  ${tensor_msgs_TARGETS})
rclcpp_components_register_node(et_tensor_publisher_component
  PLUGIN "ExecuTorchTensorPublisher"
  EXECUTABLE et_tensor_publisher_node)

add_library(et_tensor_subscriber_component SHARED
  examples/et_tensor_subscriber_node.cpp)
target_link_libraries(et_tensor_subscriber_component
  ${PROJECT_NAME}
  rclcpp::rclcpp
  rclcpp_components::component
  rosidl_runtime_cpp::rosidl_runtime_cpp
  ${tensor_msgs_TARGETS})
rclcpp_components_register_node(et_tensor_subscriber_component
  PLUGIN "ExecuTorchTensorSubscriber"
  EXECUTABLE et_tensor_subscriber_node)

install(
  TARGETS
    et_tensor_publisher_component
    et_tensor_subscriber_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_executorch_conversions test/test_executorch_conversions.cpp)
  if(TARGET test_executorch_conversions)
    target_link_libraries(test_executorch_conversions ${PROJECT_NAME})
  endif()
endif()

ament_package(CONFIG_EXTRAS_POST cmake/executorch_conversions-extras.cmake.in)
