cmake_minimum_required(VERSION 3.20)
project(dlpack_conversions_cpu)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(dlpack_conversions REQUIRED)
find_package(pluginlib REQUIRED)

add_library(${PROJECT_NAME} SHARED src/cpu_storage_plugin.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
  dlpack_conversions::interface
  pluginlib::pluginlib
)

pluginlib_export_plugin_description_file(
  dlpack_conversions
  dlpack_conversions_cpu.xml
)

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
