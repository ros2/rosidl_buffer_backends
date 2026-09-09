cmake_minimum_required(VERSION 3.20)
project(torch_conversions)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rcutils REQUIRED)
find_package(rmw REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(tensor_msgs REQUIRED)

add_library(interface INTERFACE)
add_library(${PROJECT_NAME}::interface ALIAS interface)
target_compile_features(interface INTERFACE cxx_std_17)
target_include_directories(interface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(interface INTERFACE
  pluginlib::pluginlib
  rcutils::rcutils
  rmw::rmw
  rosidl_buffer::rosidl_buffer
  tensor_msgs::tensor_msgs__rosidl_generator_cpp
)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} INTERFACE interface)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(
  TARGETS interface ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} NAMESPACE ${PROJECT_NAME}::)
ament_export_dependencies(
  ament_cmake_core pluginlib rcutils rmw rosidl_buffer tensor_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(
  CONFIG_EXTRAS_POST
    cmake/torch_conversions_runtime_discovery.cmake.in
)
