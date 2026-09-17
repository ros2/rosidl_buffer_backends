# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(libtorch_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

include(cmake/libtorch_vendor_common.cmake)
set(LIBTORCH_VENDOR_VERSION "${LIBTORCH_VENDOR_FALLBACK_VERSION}")

option(FORCE_BUILD_VENDOR_PKG "Stage the pinned distribution instead of reusing LibTorch" OFF)
include(cmake/libtorch_vendor_reuse.cmake)
libtorch_vendor_find_compatible(FALSE LIBTORCH_VENDOR_TORCH_DIR LIBTORCH_VENDOR_VARIANT
  LIBTORCH_VENDOR_SELECTED_VERSION)
if(NOT LIBTORCH_VENDOR_TORCH_DIR)
  set(LIBTORCH_VENDOR_SELECTED_VERSION "${LIBTORCH_VENDOR_VERSION}")
  set(LIBTORCH_VENDOR_VARIANT cpu)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
  if(_system_arch MATCHES "^(aarch64|arm64)$")
    include(cmake/libtorch_vendor_wheel.cmake)
    libtorch_vendor_stage_wheel("${LIBTORCH_VENDOR_VERSION}" "cpu" _vendor_source_root)
    set(_torch_root "${_vendor_source_root}/torch")
    set(LIBTORCH_VENDOR_SDK_SUBDIR "/torch")
  elseif(_system_arch MATCHES "^(x86_64|amd64)$")
    set(_archive_prefix "libtorch-shared-with-deps")
    set(_url
      "https://download.pytorch.org/libtorch/cpu/${_archive_prefix}-${LIBTORCH_VENDOR_VERSION}%2Bcpu.zip")
    set(_archive "${CMAKE_CURRENT_BINARY_DIR}/libtorch_archive")
    set(_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/libtorch-${LIBTORCH_VENDOR_VERSION}-cpu")
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

    set(_vendor_source_root "${_torch_root}")
  else()
    message(FATAL_ERROR "${PROJECT_NAME}: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'")
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
    PATTERN ".complete" EXCLUDE)

  ament_index_register_resource("vendor_packages"
    CONTENT "opt/${PROJECT_NAME}${LIBTORCH_VENDOR_SDK_SUBDIR}")
  if(LIBTORCH_VENDOR_SDK_SUBDIR)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/libtorch.dsv"
      "prepend-non-duplicate;LD_LIBRARY_PATH;opt/${PROJECT_NAME}${LIBTORCH_VENDOR_SDK_SUBDIR}/lib\n")
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
      "${ament_cmake_vendor_package_DIR}/templates/vendor_package.dsv.in")
  endif()

endif()
install(DIRECTORY cmake/ DESTINATION share/${PROJECT_NAME}/cmake)

if(LIBTORCH_VENDOR_TORCH_DIR)
  get_filename_component(_reused_root "${LIBTORCH_VENDOR_TORCH_DIR}/../../.." ABSOLUTE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/reused_libtorch.dsv"
    "prepend-non-duplicate;LD_LIBRARY_PATH;${_reused_root}/lib\n")
  ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/reused_libtorch.dsv")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST "libtorch_vendor-extras.cmake.in")
