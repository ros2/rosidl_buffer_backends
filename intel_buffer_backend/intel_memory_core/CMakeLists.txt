cmake_minimum_required(VERSION 3.20)
project(intel_memory_core VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Standalone, ROS-free library: buffer allocation, memory-pool management,
# cross-process IPC sharing (Level Zero USM + DMA-BUF), and shared-memory
# refcount synchronization. Usable by any C++ application; the ROS 2 buffer
# backend (intel_buffer) is just one consumer.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

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

# --- Intel Level Zero (optional; DMA-BUF path works without it) ---
find_library(LEVEL_ZERO_LIB NAMES ze_loader
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h
  PATHS /usr/include /usr/local/include)
if(NOT LEVEL_ZERO_LIB OR NOT LEVEL_ZERO_INCLUDE_DIR)
  message(WARNING "Level Zero not found — intel_memory_core builds without L0 USM support")
  set(LEVEL_ZERO_LIB "")
  set(LEVEL_ZERO_INCLUDE_DIR "")
endif()

# --- OpenCL (optional; enables GPU zero-copy dma_buf import) ---
# Used only by src/gpu_cl_dmabuf_import.cpp to wrap a dma_buf fd as a cl_mem via
# cl_khr_external_memory_dma_buf. When absent, that source compiles a stub and
# import_dmabuf_as_cl_mem() returns nullptr (the consumer then falls back to a
# host-staged ov::Tensor). oneAPI ships CL headers + libOpenCL under its prefix.
find_library(OPENCL_LIB NAMES OpenCL
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
        /opt/intel/oneapi/2026.1/lib)
find_path(OPENCL_INCLUDE_DIR NAMES CL/cl.h
  PATHS /usr/include /usr/local/include /opt/intel/oneapi/2026.1/include)
if(NOT OPENCL_LIB OR NOT OPENCL_INCLUDE_DIR)
  message(WARNING "OpenCL not found — intel_memory_core builds without GPU dma_buf import")
  set(OPENCL_LIB "")
  set(OPENCL_INCLUDE_DIR "")
endif()

add_library(intel_memory_core SHARED
  src/intel_memory_buffer_pool.cpp
  src/intel_memory_ipc_manager.cpp
  src/l0_usm_ipc.cpp
  src/l0_external_memmap.cpp
  src/dmabuf_ipc.cpp
  src/gpu_cl_dmabuf_import.cpp
)
if(OPENCL_LIB AND OPENCL_INCLUDE_DIR)
  target_compile_definitions(intel_memory_core PRIVATE INTEL_MEMORY_CORE_HAVE_OPENCL)
endif()
add_library(intel_memory_core::intel_memory_core ALIAS intel_memory_core)

target_include_directories(intel_memory_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${LEVEL_ZERO_INCLUDE_DIR}
)
# OpenCL headers are a private build detail (only gpu_cl_dmabuf_import.cpp uses
# them); the public header stays OpenCL-free via opaque void* handles.
target_include_directories(intel_memory_core PRIVATE ${OPENCL_INCLUDE_DIR})

# rt: shm_open/shm_unlink. pthread: std::thread in the IPC accept loops.
find_package(Threads REQUIRED)
target_link_libraries(intel_memory_core PUBLIC
  ${LEVEL_ZERO_LIB}
  rt
  Threads::Threads
)
target_link_libraries(intel_memory_core PRIVATE ${OPENCL_LIB})

# --- Standalone example (built with -DINTEL_MEMORY_CORE_BUILD_EXAMPLE=ON) ---
option(INTEL_MEMORY_CORE_BUILD_EXAMPLE "Build the non-ROS usage example" OFF)
if(INTEL_MEMORY_CORE_BUILD_EXAMPLE)
  add_executable(intel_memory_core_example example/standalone_example.cpp)
  target_link_libraries(intel_memory_core_example PRIVATE intel_memory_core)

  add_executable(intel_memory_core_memmap_example
    example/external_memmap_example.cpp)
  target_link_libraries(intel_memory_core_memmap_example
    PRIVATE intel_memory_core)
endif()

# --- Install + export a CMake package config so non-ROS apps can find_package ---
install(TARGETS intel_memory_core
  EXPORT intel_memory_coreTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT intel_memory_coreTargets
  FILE intel_memory_coreTargets.cmake
  NAMESPACE intel_memory_core::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/intel_memory_core
)

include(CMakePackageConfigHelpers)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/intel_memory_coreConfig.cmake.in"
"@PACKAGE_INIT@\n\
include(CMakeFindDependencyMacro)\n\
find_dependency(Threads)\n\
include(\"\${CMAKE_CURRENT_LIST_DIR}/intel_memory_coreTargets.cmake\")\n")
configure_package_config_file(
  "${CMAKE_CURRENT_BINARY_DIR}/intel_memory_coreConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/intel_memory_coreConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/intel_memory_core
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/intel_memory_coreConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/intel_memory_coreConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/intel_memory_coreConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/intel_memory_core
)

# --- Unit tests (built with -DBUILD_TESTING=ON, colcon's default) ---
option(INTEL_MEMORY_CORE_BUILD_TESTS "Build unit tests" ${BUILD_TESTING})
if(INTEL_MEMORY_CORE_BUILD_TESTS)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  enable_testing()

  add_executable(intel_memory_core_test
    test/test_dmabuf_ipc_free_functions.cpp
    test/test_shared_sysmem.cpp
    test/test_l0_external_memmap.cpp
    test/test_gpu_cl_dmabuf_import.cpp
    test/test_intel_memory_buffer_pool.cpp
    test/test_intel_memory_ipc_manager.cpp
  )
  target_link_libraries(intel_memory_core_test PRIVATE
    intel_memory_core
    GTest::gtest
    GTest::gtest_main
  )
  gtest_discover_tests(intel_memory_core_test)
endif()
