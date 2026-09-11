# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(libtorch_cuda_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

include(cmake/libtorch_cuda_vendor_policy.cmake)

set(LIBTORCH_CUDA_VENDOR_VARIANT "" CACHE STRING
  "CUDA variant override (cu126, cu128, cu130). Auto-detected if empty.")

option(FORCE_BUILD_VENDOR_PKG
  "Disable reuse of a platform-provided LibTorch distribution"
  OFF)

if(LIBTORCH_CUDA_VENDOR_VARIANT)
  libtorch_cuda_vendor_validate_variant(
    "${LIBTORCH_CUDA_VENDOR_VARIANT}")
endif()

find_package(CUDAToolkit REQUIRED)
libtorch_cuda_vendor_variant_for_cuda(
  "${CUDAToolkit_VERSION}" _detected_variant)
if(LIBTORCH_CUDA_VENDOR_VARIANT AND
    NOT LIBTORCH_CUDA_VENDOR_VARIANT STREQUAL _detected_variant)
  message(FATAL_ERROR
    "${PROJECT_NAME}: requested ${LIBTORCH_CUDA_VENDOR_VARIANT}, but CUDA "
    "${CUDAToolkit_VERSION} requires ${_detected_variant}")
endif()
set(LIBTORCH_CUDA_VENDOR_VARIANT "${_detected_variant}")

# On aarch64, reuse JetPack's LibTorch and identify Tegra from the device tree.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
if(_system_arch MATCHES "aarch64|arm64")
  set(_is_aarch64 TRUE)
  set(_is_tegra FALSE)
  if(EXISTS "/proc/device-tree/compatible")
    file(READ "/proc/device-tree/compatible" _dt_compat_hex HEX)
    string(HEX "nvidia,tegra" _tegra_needle_hex)
    if(_dt_compat_hex MATCHES "${_tegra_needle_hex}")
      set(_is_tegra TRUE)
    endif()
  endif()
else()
  set(_is_aarch64 FALSE)
  set(_is_tegra FALSE)
endif()

set(LIBTORCH_CUDA_VENDOR_TORCH_DIR "")
if(_is_aarch64)
  if(NOT _is_tegra)
    message(FATAL_ERROR
      "${PROJECT_NAME}: CUDA aarch64 support is only available on NVIDIA Tegra")
  endif()
  if(FORCE_BUILD_VENDOR_PKG)
    message(FATAL_ERROR
      "${PROJECT_NAME}: no CUDA LibTorch archive is published for aarch64; "
      "the JetPack PyTorch installation must be reused")
  endif()

  execute_process(
    COMMAND python3 -c
      "import os, torch; cuda=torch.version.cuda or ''; print(torch.__version__.split('+')[0] + ';cu' + cuda.replace('.', '') + ';' + os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))"
    OUTPUT_VARIABLE _system_torch
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _system_torch_rc
    ERROR_QUIET
  )
  if(NOT _system_torch_rc EQUAL 0)
    message(FATAL_ERROR
      "${PROJECT_NAME}: JetPack CUDA-enabled PyTorch ${LIBTORCH_CUDA_VENDOR_VERSION} is required")
  endif()
  list(GET _system_torch 0 _system_torch_version)
  list(GET _system_torch 1 _system_torch_variant)
  list(GET _system_torch 2 _system_torch_dir)
  if(NOT _system_torch_version STREQUAL LIBTORCH_CUDA_VENDOR_VERSION OR
      NOT _system_torch_variant STREQUAL LIBTORCH_CUDA_VENDOR_VARIANT OR
      NOT EXISTS "${_system_torch_dir}/TorchConfig.cmake")
    message(FATAL_ERROR
      "${PROJECT_NAME}: JetPack Torch ${_system_torch_version}+"
      "${_system_torch_variant} does not match required "
      "${LIBTORCH_CUDA_VENDOR_VERSION}+${LIBTORCH_CUDA_VENDOR_VARIANT}")
  endif()
  set(LIBTORCH_CUDA_VENDOR_TORCH_DIR "${_system_torch_dir}")
else()
  set(_archive_prefix "libtorch-shared-with-deps")
  string(CONCAT _url
    "https://download.pytorch.org/libtorch/${LIBTORCH_CUDA_VENDOR_VARIANT}/"
    "${_archive_prefix}-${LIBTORCH_CUDA_VENDOR_VERSION}%2B"
    "${LIBTORCH_CUDA_VENDOR_VARIANT}.zip")

  set(_archive "${CMAKE_CURRENT_BINARY_DIR}/libtorch_archive")
  set(_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/libtorch_extracted")
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

  # Omit the optional GPUDirect Storage RDMA transport and its RDMA dependencies.
  file(GLOB _unused_cufile_rdma "${_torch_root}/lib/libcufile_rdma*")
  if(_unused_cufile_rdma)
    file(REMOVE ${_unused_cufile_rdma})
  endif()

  # Exclude the Python binding, upstream test fixtures, and the NNAPI plugin
  # that links the Python binding. Python bindings are supplied by the Python
  # vendors.
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
    USE_SOURCE_PERMISSIONS
  )

  # Strip upstream debug sections before debhelper generates dbgsym packages.
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
    "${ament_cmake_vendor_package_DIR}/templates/vendor_package.dsv.in"
  )
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  add_test(
    NAME libtorch_cuda_vendor_policy
    COMMAND "${CMAKE_COMMAND}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/test/test_version_policy.cmake"
  )
endif()

ament_package(
  CONFIG_EXTRAS_POST "libtorch_cuda_vendor-extras.cmake.in"
)
