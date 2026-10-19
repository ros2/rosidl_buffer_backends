cmake_minimum_required(VERSION 3.20)
project(qc_buffer_backend)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/qc_buffer_backend_plugin.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME}
  ${qc_buffer_backend_msgs_TARGETS}
  rosidl_buffer_backend::rosidl_buffer_backend
  rosidl_runtime_cpp::rosidl_runtime_cpp
  rosidl_buffer_backend_registry::rosidl_buffer_backend_registry
  rosidl_typesupport_cpp::rosidl_typesupport_cpp
  pluginlib::pluginlib
  qc_buffer::qc_buffer
  rcutils::rcutils
)

pluginlib_export_plugin_description_file(rosidl_buffer_backend qc_buffer_plugin.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_qc_buffer_backend
    test/test_qc_buffer_backend.cpp
  )
  if(TARGET test_qc_buffer_backend)
    target_link_libraries(test_qc_buffer_backend
      ${PROJECT_NAME}
      qc_buffer::qc_buffer
      ${qc_buffer_backend_msgs_TARGETS}
      rosidl_buffer::rosidl_buffer
      rosidl_buffer_backend::rosidl_buffer_backend
    )
  endif()
endif()

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
