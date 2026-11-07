# Copyright 2026 Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.28)
project(executorch_cuda_vendor)

find_package(ament_cmake REQUIRED)
# Keeps executorch_vendor from pointing executorch_DIR at its CPU copy.
set(EXECUTORCH_CUDA_VENDOR_ACTIVE TRUE)
find_package(executorch_vendor REQUIRED)

option(FORCE_BUILD_VENDOR_PKG "Stage the pinned wheel instead of reusing an installed ExecuTorch" OFF)

executorch_vendor_find_installed(extension_cuda backend_cuda)
if(executorch_FOUND)
  # executorch_VERSION keeps only the numbers, so a nightly reads as its release.
  string(REGEX REPLACE "\\+.*$" "" _version "${EXECUTORCH_BUILD_VERSION}")
  set(_cuda_major "")
  if(EXECUTORCH_BUILD_VERSION MATCHES "\\+cu([0-9]+)[0-9]$")
    set(_cuda_major "${CMAKE_MATCH_1}")
  endif()
else()
  # The release of the wheels pinned below, so a pin that moves without them is caught by
  # the release check instead of a failed download.
  set(_version "1.6.0")
  # The PyPI wheel is the CUDA 13 build.
  set(_cuda_major 13)
endif()
# ExecuTorch's CUDA libraries need the CUDA runtime of the same major version.
# Without it this package installs nothing, so a workspace on another CUDA
# version still builds, and dependent CUDA packages can skip themselves too.
find_package(CUDAToolkit QUIET)
set(EXECUTORCH_CUDA_VENDOR_SKIPPED "")
set(_hint "Select a toolkit with CUDAToolkit_ROOT in a clean build folder (or with colcon's --cmake-clean-cache) to build it.")
if(NOT CUDAToolkit_FOUND)
  set(EXECUTORCH_CUDA_VENDOR_SKIPPED "no usable CUDA toolkit was found")
elseif(NOT _cuda_major)
  # A source build reports a git hash instead of +cuNNN.
  message(STATUS "${PROJECT_NAME}: ExecuTorch ${EXECUTORCH_BUILD_VERSION} does not name its "
    "CUDA version, so it cannot be checked against CUDA ${CUDAToolkit_VERSION}")
elseif(NOT CUDAToolkit_VERSION_MAJOR STREQUAL _cuda_major)
  set(EXECUTORCH_CUDA_VENDOR_SKIPPED "it needs CUDA ${_cuda_major}, and the toolkit is CUDA ${CUDAToolkit_VERSION}")
endif()
# Both copies install libexecutorch.so, and a process loads whichever comes
# first. They are the same code only for the same release.
if(NOT EXECUTORCH_CUDA_VENDOR_SKIPPED AND NOT _version STREQUAL EXECUTORCH_VENDOR_SELECTED_VERSION)
  set(EXECUTORCH_CUDA_VENDOR_SKIPPED
    "executorch_vendor uses ExecuTorch ${EXECUTORCH_VENDOR_SELECTED_VERSION}")
  set(_hint "Install the matching release, or rebuild executorch_vendor and ${PROJECT_NAME} with -DFORCE_BUILD_VENDOR_PKG=ON to build it.")
endif()
if(EXECUTORCH_CUDA_VENDOR_SKIPPED)
  message(WARNING "${PROJECT_NAME}: skipping ExecuTorch ${_version} with CUDA, because "
    "${EXECUTORCH_CUDA_VENDOR_SKIPPED}. ${_hint}")
  ament_package(CONFIG_EXTRAS_POST "executorch_cuda_vendor-extras.cmake.in")
  return()
endif()

if(executorch_FOUND)
  get_filename_component(EXECUTORCH_CUDA_VENDOR_DIR "${executorch_DIR}" REALPATH)
  message(STATUS "${PROJECT_NAME}: reusing ExecuTorch ${EXECUTORCH_BUILD_VERSION} from ${EXECUTORCH_CUDA_VENDOR_DIR}")
else()
  set(_pypi "https://files.pythonhosted.org/packages")
  set(_wheel_prefix "executorch-${_version}-cp312-cp312-manylinux_2_28")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch)
  if(_arch MATCHES "^(aarch64|arm64)$")
    executorch_vendor_install_wheel(
      "${_pypi}/d5/7f/9028996ec1dead9c57ed2c6384037e3567c2f514e4ed0c54480d2d3eeb48/${_wheel_prefix}_aarch64.whl"
      "ff595d236f4052a196a03a867ca0f53d2e3b003d00ab871433660360e194e78e"
      "executorch/backends/cuda/*.so")
  elseif(_arch MATCHES "^(x86_64|amd64)$")
    executorch_vendor_install_wheel(
      "${_pypi}/ef/24/f8fde978e1e5fd629d68c74a3f4acee30acb05f4f90c433cc1a8e29d8eed/${_wheel_prefix}_x86_64.whl"
      "d05efe9afe00495112ecf304ca548fcd3239d39e96ac25381b528cdbad948b54"
      "executorch/backends/cuda/*.so")
  else()
    message(FATAL_ERROR "${PROJECT_NAME}: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'")
  endif()
  set(EXECUTORCH_CUDA_VENDOR_DIR "")
  # The wheel's CUDA libraries look for the CUDA runtime in a pip package that
  # is not installed here, and also in backends/cuda. A copy there of the runtime
  # of the toolkit checked above lets them load without the toolkit on the search
  # path. It stays out of lib, which is on the RUNPATH of everything that links
  # ExecuTorch, so code built with a newer toolkit still gets that toolkit's runtime.
  get_target_property(_cudart CUDA::cudart IMPORTED_LOCATION)
  get_filename_component(_cudart_dir "${_cudart}" DIRECTORY)
  if(NOT EXISTS "${_cudart_dir}/libcudart.so.${_cuda_major}")
    message(FATAL_ERROR "${PROJECT_NAME}: no libcudart.so.${_cuda_major} in ${_cudart_dir}")
  endif()
  file(REAL_PATH "${_cudart_dir}/libcudart.so.${_cuda_major}" _cudart_file)
  install(FILES "${_cudart_file}"
    DESTINATION "opt/${PROJECT_NAME}/backends/cuda" RENAME "libcudart.so.${_cuda_major}")
  install(FILES NOTICE.cudart DESTINATION "share/${PROJECT_NAME}")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST "executorch_cuda_vendor-extras.cmake.in")
