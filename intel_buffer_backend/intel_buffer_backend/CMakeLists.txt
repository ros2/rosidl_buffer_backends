cmake_minimum_required(VERSION 3.20)
project(intel_buffer_backend)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wformat -Wformat-security
    -fcf-protection=full
    -D_FORTIFY_SOURCE=3
    -D_GLIBCXX_ASSERTIONS
  )
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fstack-protector)
  else()
    add_compile_options(
      -Werror -Wconversion -Wimplicit-fallthrough
      -Werror=format-security
      -fstack-protector-strong -fstack-clash-protection
    )
    add_link_options(
      -Wl,-z,noexecstack
      -Wl,-z,relro,-z,now
    )
  endif()
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Level Zero is used only by intel_buffer library; this package
# accesses L0 functionality via L0UsmIpc/DmaBufIpc abstractions.

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/intel_buffer_backend_plugin.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME}
  ${intel_buffer_backend_msgs_TARGETS}
  rosidl_buffer_backend::rosidl_buffer_backend
  rosidl_runtime_cpp::rosidl_runtime_cpp
  rosidl_buffer_backend_registry::rosidl_buffer_backend_registry
  rosidl_typesupport_cpp::rosidl_typesupport_cpp
  pluginlib::pluginlib
  intel_buffer::intel_buffer
  rcutils::rcutils
)

pluginlib_export_plugin_description_file(rosidl_buffer_backend intel_buffer_plugin.xml)

# The sample pipeline nodes (camera publisher, OpenVINO inference, visualizer)
# and their launch/config now live in the separate test_intel_buffer_backend
# package. This package builds only the buffer-backend plugin.

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(rmw REQUIRED)

  ament_auto_add_gtest(test_intel_buffer_backend_plugin
    test/test_intel_buffer_backend_plugin.cpp
  )
  target_link_libraries(test_intel_buffer_backend_plugin
    ${PROJECT_NAME}
    rmw::rmw
  )
endif()

ament_auto_package(USE_SCOPED_HEADER_INSTALL_DIR)
