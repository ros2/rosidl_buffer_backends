# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(libtorch_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

set(LIBTORCH_VENDOR_VERSION "2.9.1")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
if(NOT _system_arch MATCHES "x86_64|amd64")
  message(FATAL_ERROR
    "${PROJECT_NAME}: the official CPU LibTorch archive is supported on amd64 only")
endif()

set(_archive_prefix "libtorch-shared-with-deps")
set(_url
  "https://download.pytorch.org/libtorch/cpu/${_archive_prefix}-${LIBTORCH_VENDOR_VERSION}%2Bcpu.zip")
set(_archive "${CMAKE_CURRENT_BINARY_DIR}/libtorch_archive")
set(_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/libtorch_extracted")
set(_torch_root "${_extract_dir}/libtorch")

if(NOT EXISTS "${_torch_root}/share/cmake/Torch/TorchConfig.cmake")
  message(STATUS
    "${PROJECT_NAME}: downloading LibTorch ${LIBTORCH_VENDOR_VERSION}+cpu")
  file(DOWNLOAD "${_url}" "${_archive}" SHOW_PROGRESS STATUS _dl_status)
  list(GET _dl_status 0 _dl_code)
  if(NOT _dl_code EQUAL 0)
    list(GET _dl_status 1 _dl_msg)
    message(FATAL_ERROR "${PROJECT_NAME}: download failed: ${_dl_msg}")
  endif()
  file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_extract_dir}")
  file(REMOVE "${_archive}")
endif()

# Exclude the Python binding, upstream test fixtures, and the NNAPI plugin that
# links the Python binding. Python bindings are supplied by the Python vendors.
set(_non_cpp_provider_libraries
  libtorch_python.so
  libc10d_cuda_test.so
  libtorchbind_test.so
  libjitbackend_test.so
  libbackend_with_compiler.so
  libaoti_custom_ops.so
  libnnapi_backend.so)
foreach(_library IN LISTS _non_cpp_provider_libraries)
  file(REMOVE "${_torch_root}/lib/${_library}")
endforeach()

install(DIRECTORY "${_torch_root}/"
  DESTINATION "opt/${PROJECT_NAME}"
  USE_SOURCE_PERMISSIONS)

install(CODE "
  file(GLOB_RECURSE _elfs
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/opt/${PROJECT_NAME}/*.so\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/opt/${PROJECT_NAME}/*.so.*\")
  foreach(_elf IN LISTS _elfs)
    if(EXISTS \"\${_elf}\" AND NOT IS_SYMLINK \"\${_elf}\")
      execute_process(
        COMMAND \"${CMAKE_STRIP}\" --strip-unneeded \"\${_elf}\"
        OUTPUT_QUIET ERROR_QUIET)
    endif()
  endforeach()
")

ament_index_register_resource("vendor_packages" CONTENT "opt/${PROJECT_NAME}")
if(APPLE)
  set(_LIBDIR_VAR_NAME "DYLD_LIBRARY_PATH")
elseif(WIN32)
  set(_LIBDIR_VAR_NAME "PATH")
else()
  set(_LIBDIR_VAR_NAME "LD_LIBRARY_PATH")
endif()
ament_environment_hooks(
  "${ament_cmake_vendor_package_DIR}/templates/vendor_package.sh.in"
  "${ament_cmake_vendor_package_DIR}/templates/vendor_package.dsv.in")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST "libtorch_vendor-extras.cmake.in")
