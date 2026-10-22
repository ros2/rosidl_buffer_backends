cmake_minimum_required(VERSION 3.20)
project(intel_buffer)

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
      -Wl,-z,nodlopen
    )
  endif()
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_buffer REQUIRED)

# The generic, ROS-free core (pool, IPC, sync). Standalone CMake package found
# via CMAKE_PREFIX_PATH / the install tree. This adapter is a thin, header-only
# ROS 2 layer over it.
find_package(intel_memory_core REQUIRED)

# Header-only adapter: IntelBufferImpl (rosidl::BufferImplBase) and the
# rosidl::Buffer-returning helper API. All non-ROS logic lives in the core lib.
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME} INTERFACE
  rosidl_buffer::rosidl_buffer
  intel_memory_core::intel_memory_core
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rosidl_buffer intel_memory_core)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_intel_buffer_api test/test_intel_buffer_api.cpp)
  target_include_directories(test_intel_buffer_api PRIVATE include)
  target_link_libraries(test_intel_buffer_api
    rosidl_buffer::rosidl_buffer
    intel_memory_core::intel_memory_core
  )

  ament_add_gtest(test_intel_buffer_impl test/test_intel_buffer_impl.cpp)
  target_include_directories(test_intel_buffer_impl PRIVATE include)
  target_link_libraries(test_intel_buffer_impl
    rosidl_buffer::rosidl_buffer
    intel_memory_core::intel_memory_core
  )
endif()

ament_package()
