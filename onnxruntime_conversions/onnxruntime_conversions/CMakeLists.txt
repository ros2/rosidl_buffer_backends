cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_core REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rosidl_buffer REQUIRED)
find_package(tensor_msgs REQUIRED)

add_library(${PROJECT_NAME}_interface INTERFACE)
set_target_properties(${PROJECT_NAME}_interface PROPERTIES EXPORT_NAME interface)
add_library(${PROJECT_NAME}::interface ALIAS ${PROJECT_NAME}_interface)
target_compile_features(${PROJECT_NAME}_interface INTERFACE cxx_std_17)
target_include_directories(${PROJECT_NAME}_interface INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME}_interface INTERFACE
  pluginlib::pluginlib
  rosidl_buffer::rosidl_buffer
  tensor_msgs::tensor_msgs__rosidl_generator_cpp
)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} INTERFACE ${PROJECT_NAME}_interface)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)
install(
  TARGETS ${PROJECT_NAME}_interface ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  INCLUDES DESTINATION include/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
)
ament_export_dependencies(ament_cmake_core pluginlib rosidl_buffer tensor_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST
  cmake/onnxruntime_conversions_runtime_discovery.cmake
)
