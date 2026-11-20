# Copyright 2026 Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

# The ExecuTorch CMake package defines its imported targets only on CMake 3.28+.
cmake_minimum_required(VERSION 3.28)
project(executorch_vendor)

find_package(ament_cmake REQUIRED)
include(cmake/executorch_vendor_helpers.cmake)

set(EXECUTORCH_VENDOR_VERSION "1.6.0")
# ExecuTorch promises no C++ ABI compatibility across minor releases.
set(EXECUTORCH_VENDOR_VERSION_RANGE "${EXECUTORCH_VENDOR_VERSION}...<1.7")

option(FORCE_BUILD_VENDOR_PKG "Stage the pinned wheel instead of reusing an installed ExecuTorch" OFF)

executorch_vendor_find_installed()
if(executorch_FOUND)
  get_filename_component(EXECUTORCH_VENDOR_DIR "${executorch_DIR}" REALPATH)
  # executorch_VERSION keeps only the numbers, so a nightly reads as its release.
  string(REGEX REPLACE "\\+.*$" "" EXECUTORCH_VENDOR_SELECTED_VERSION "${EXECUTORCH_BUILD_VERSION}")
  message(STATUS "${PROJECT_NAME}: reusing ExecuTorch ${EXECUTORCH_BUILD_VERSION} from ${EXECUTORCH_VENDOR_DIR}")
else()
  # The CPU build, so packages that only run on the CPU do not need CUDA. The
  # C++ libraries in the wheel do not depend on the Python version, so one
  # CPython tag per architecture is enough.
  set(_index "https://download.pytorch.org/whl/cpu")
  set(_wheel_prefix "executorch-${EXECUTORCH_VENDOR_VERSION}%2Bcpu-cp312-cp312-manylinux_2_28")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch)
  if(_arch MATCHES "^(aarch64|arm64)$")
    executorch_vendor_install_wheel("${_index}/${_wheel_prefix}_aarch64.whl"
      "48c1703bfb6abc14efe3162186f1cd93b6bbf114044f0adcdd2f9c442d265685")
  elseif(_arch MATCHES "^(x86_64|amd64)$")
    executorch_vendor_install_wheel("${_index}/${_wheel_prefix}_x86_64.whl"
      "533a8269b1c406d59b74c590e0ddc8a7ffacdfeb5d162a0117aa42d12c8a1f1c")
  else()
    message(FATAL_ERROR "${PROJECT_NAME}: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'")
  endif()
  set(EXECUTORCH_VENDOR_DIR "")
  set(EXECUTORCH_VENDOR_SELECTED_VERSION "${EXECUTORCH_VENDOR_VERSION}")
endif()

install(DIRECTORY cmake/ DESTINATION share/${PROJECT_NAME}/cmake)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST "executorch_vendor-extras.cmake.in")
