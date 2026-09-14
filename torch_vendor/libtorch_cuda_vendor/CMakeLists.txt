# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(libtorch_cuda_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

include(cmake/libtorch_cuda_vendor_policy.cmake)

set(LIBTORCH_CUDA_VENDOR_VARIANT "" CACHE STRING
  "Fallback CUDA variant override (cu130, cu132).")

option(FORCE_BUILD_VENDOR_PKG
  "Disable reuse of a platform-provided LibTorch distribution"
  OFF)

if(LIBTORCH_CUDA_VENDOR_VARIANT)
  libtorch_cuda_vendor_validate_variant(
    "${LIBTORCH_CUDA_VENDOR_VARIANT}")
endif()

set(_requested_variant "${LIBTORCH_CUDA_VENDOR_VARIANT}")
find_package(CUDAToolkit QUIET)

# Probe the caller's SDK paths before applying the CPU vendor defaults.
set(_requested_torch_dir "${Torch_DIR}")
find_package(libtorch_vendor REQUIRED)
set(Torch_DIR "${_requested_torch_dir}")
libtorch_vendor_find_compatible(TRUE LIBTORCH_CUDA_VENDOR_TORCH_DIR _system_variant
  LIBTORCH_CUDA_VENDOR_SELECTED_VERSION "${LIBTORCH_VENDOR_SELECTED_VERSION}")
if(LIBTORCH_CUDA_VENDOR_TORCH_DIR AND _requested_variant AND NOT _system_variant STREQUAL _requested_variant)
  message(STATUS "${PROJECT_NAME}: installed Torch build ${_system_variant} differs from "
    "explicitly requested ${_requested_variant}; staging pinned fallback")
  set(LIBTORCH_CUDA_VENDOR_TORCH_DIR "")
endif()
if(LIBTORCH_CUDA_VENDOR_TORCH_DIR)
  set(LIBTORCH_CUDA_VENDOR_VARIANT "${_system_variant}")
endif()
if(NOT LIBTORCH_CUDA_VENDOR_TORCH_DIR)
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "${PROJECT_NAME}: the pinned CUDA fallback requires a CUDA 13.1 toolkit. "
      "Install it through rosdep/APT and select it with -DCUDAToolkit_ROOT=/usr/local/cuda-13.1.")
  endif()
  libtorch_cuda_vendor_variant_for_cuda("${CUDAToolkit_VERSION}" _fallback_variant)
  if(_requested_variant)
    libtorch_cuda_vendor_validate_variant_for_cuda("${_requested_variant}" "${CUDAToolkit_VERSION}")
    set(LIBTORCH_CUDA_VENDOR_VARIANT "${_requested_variant}")
  else()
    set(LIBTORCH_CUDA_VENDOR_VARIANT "${_fallback_variant}")
  endif()
  set(LIBTORCH_CUDA_VENDOR_SELECTED_VERSION "${LIBTORCH_CUDA_VENDOR_VERSION}")
  if(NOT LIBTORCH_VENDOR_SELECTED_VERSION STREQUAL LIBTORCH_CUDA_VENDOR_SELECTED_VERSION)
    message(FATAL_ERROR "${PROJECT_NAME}: no compatible CUDA provider for the selected CPU/core LibTorch "
      "${LIBTORCH_VENDOR_SELECTED_VERSION}; the pinned CUDA fallback is ${LIBTORCH_CUDA_VENDOR_SELECTED_VERSION}. "
      "Select a matching CUDA SDK with Torch_DIR, or rebuild all Torch vendors and conversions "
      "with -DFORCE_BUILD_VENDOR_PKG=ON.")
  endif()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
  if(_system_arch MATCHES "^(aarch64|arm64)$")
    include("${libtorch_vendor_DIR}/libtorch_vendor_wheel.cmake")
    libtorch_vendor_stage_wheel("${LIBTORCH_CUDA_VENDOR_VERSION}"
      "${LIBTORCH_CUDA_VENDOR_VARIANT}" _vendor_source_root)
    set(_torch_root "${_vendor_source_root}/torch")
    set(LIBTORCH_CUDA_VENDOR_SDK_SUBDIR "/torch")
  elseif(_system_arch MATCHES "^(x86_64|amd64)$")
    set(_archive_prefix "libtorch-shared-with-deps")
    string(CONCAT _url
      "https://download.pytorch.org/libtorch/${LIBTORCH_CUDA_VENDOR_VARIANT}/"
      "${_archive_prefix}-${LIBTORCH_CUDA_VENDOR_VERSION}%2B"
      "${LIBTORCH_CUDA_VENDOR_VARIANT}.zip")

    set(_archive "${CMAKE_CURRENT_BINARY_DIR}/libtorch_archive")
    set(_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/libtorch-${LIBTORCH_CUDA_VENDOR_VERSION}-${LIBTORCH_CUDA_VENDOR_VARIANT}")
    set(_torch_root "${_extract_dir}/libtorch")

    if(NOT EXISTS "${_torch_root}/share/cmake/Torch/TorchConfig.cmake")
      message(STATUS
        "${PROJECT_NAME}: downloading LibTorch ${LIBTORCH_CUDA_VENDOR_VERSION}+"
        "${LIBTORCH_CUDA_VENDOR_VARIANT}")
      message(STATUS "${PROJECT_NAME}: URL: ${_url}")
      file(DOWNLOAD "${_url}" "${_archive}" SHOW_PROGRESS STATUS _dl_status)
      list(GET _dl_status 0 _dl_code)
      if(NOT _dl_code EQUAL 0)
        list(GET _dl_status 1 _dl_msg)
        message(FATAL_ERROR "${PROJECT_NAME}: download failed: ${_dl_msg}")
      endif()
      file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_extract_dir}")
      file(REMOVE "${_archive}")
    else()
      message(STATUS "${PROJECT_NAME}: using cached LibTorch at ${_torch_root}")
    endif()

    set(_vendor_source_root "${_torch_root}")
  else()
    message(FATAL_ERROR "${PROJECT_NAME}: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'")
  endif()

  file(GLOB _unused_cufile_rdma "${_torch_root}/lib/libcufile_rdma*")
  if(_unused_cufile_rdma)
    file(REMOVE ${_unused_cufile_rdma})
  endif()

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

  install(DIRECTORY "${_vendor_source_root}/"
    DESTINATION "opt/${PROJECT_NAME}"
    USE_SOURCE_PERMISSIONS
    PATTERN ".complete" EXCLUDE
  )

  ament_index_register_resource("vendor_packages"
    CONTENT "opt/${PROJECT_NAME}${LIBTORCH_CUDA_VENDOR_SDK_SUBDIR}")
  if(LIBTORCH_CUDA_VENDOR_SDK_SUBDIR)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libtorch.dsv"
      "prepend-non-duplicate;LD_LIBRARY_PATH;opt/${PROJECT_NAME}${LIBTORCH_CUDA_VENDOR_SDK_SUBDIR}/lib\n")
    ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/libtorch.dsv")
  else()
    if(APPLE)
      set(_LIBDIR_VAR_NAME "DYLD_LIBRARY_PATH")
    elseif(WIN32)
      set(_LIBDIR_VAR_NAME "PATH")
    else()
      set(_LIBDIR_VAR_NAME "LD_LIBRARY_PATH")
    endif()
    ament_environment_hooks(
      "${ament_cmake_vendor_package_DIR}/templates/vendor_package.sh.in"
      "${ament_cmake_vendor_package_DIR}/templates/vendor_package.dsv.in"
    )
  endif()

endif()

if(LIBTORCH_CUDA_VENDOR_TORCH_DIR)
  get_filename_component(_reused_root "${LIBTORCH_CUDA_VENDOR_TORCH_DIR}/../../.." ABSOLUTE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reused_libtorch.dsv"
    "prepend-non-duplicate;LD_LIBRARY_PATH;${_reused_root}/lib\n")
  ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/reused_libtorch.dsv")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  set(_cuda_root_argument)
  if(CUDAToolkit_ROOT)
    set(_cuda_root_argument "-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
  endif()
  add_test(
    NAME libtorch_cuda_vendor_selection
    COMMAND "${CMAKE_COMMAND}"
      -S "${CMAKE_CURRENT_SOURCE_DIR}/test/consumer"
      -B "${CMAKE_CURRENT_BINARY_DIR}/selection_test"
      "-DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX};${CMAKE_PREFIX_PATH}"
      ${_cuda_root_argument}
  )
  add_test(
    NAME libtorch_cuda_vendor_policy
    COMMAND "${CMAKE_COMMAND}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/test/test_version_policy.cmake"
  )
endif()

ament_package(
  CONFIG_EXTRAS_POST "libtorch_cuda_vendor-extras.cmake.in"
)
