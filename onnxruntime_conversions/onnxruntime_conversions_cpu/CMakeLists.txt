# Copyright 2026 Open Source Robotics Foundation, Inc.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(onnxruntime_conversions_cpu)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(onnxruntime_conversions REQUIRED)
find_package(onnxruntime_core_vendor REQUIRED)
find_package(pluginlib REQUIRED)

add_library(${PROJECT_NAME} SHARED src/cpu_conversion_plugin.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
  onnxruntime::onnxruntime
  onnxruntime_conversions::onnxruntime_conversions
  pluginlib::pluginlib
)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

pluginlib_export_plugin_description_file(
  onnxruntime_conversions onnxruntime_conversions_cpu.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
