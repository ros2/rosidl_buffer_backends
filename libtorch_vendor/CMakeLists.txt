# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(libtorch_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

set(LIBTORCH_VERSION "" CACHE STRING
  "LibTorch version to download. Auto-selected per CUDA variant if empty.")
set(LIBTORCH_CUDA_VERSION "" CACHE STRING
  "CUDA variant override (cu118, cu121, cu124, cu126, cu128, cu129, cpu). Auto-detected if empty.")

option(FORCE_BUILD_VENDOR_PKG
  "Build vendor packages from source, even if system-installed packages are available"
  OFF)

if(LIBTORCH_CUDA_VERSION STREQUAL "")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
      set(LIBTORCH_CUDA_VERSION "cu129")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
      set(LIBTORCH_CUDA_VERSION "cu128")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.6")
      set(LIBTORCH_CUDA_VERSION "cu126")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.4")
      set(LIBTORCH_CUDA_VERSION "cu124")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.1")
      set(LIBTORCH_CUDA_VERSION "cu121")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
      set(LIBTORCH_CUDA_VERSION "cu118")
    else()
      set(LIBTORCH_CUDA_VERSION "cpu")
    endif()
  else()
    set(LIBTORCH_CUDA_VERSION "cpu")
  endif()
endif()

if(LIBTORCH_VERSION STREQUAL "")
  if(LIBTORCH_CUDA_VERSION STREQUAL "cu118")
    set(LIBTORCH_VERSION "2.7.1")
  elseif(LIBTORCH_CUDA_VERSION STREQUAL "cu121")
    set(LIBTORCH_VERSION "2.5.1")
  elseif(LIBTORCH_CUDA_VERSION STREQUAL "cu124")
    set(LIBTORCH_VERSION "2.6.0")
  else()
    set(LIBTORCH_VERSION "2.11.0")
  endif()
endif()

find_package(Torch QUIET)

if(NOT Torch_FOUND OR FORCE_BUILD_VENDOR_PKG)
  if(Torch_FOUND)
    message(STATUS "${PROJECT_NAME}: forcing vendor build (FORCE_BUILD_VENDOR_PKG)")
  endif()

  # Starting with LibTorch 2.8.0, PyTorch dropped the "-cxx11-abi-" infix from
  # the Linux archive name (they no longer ship the pre-cxx11-abi variant).
  if(LIBTORCH_VERSION VERSION_GREATER_EQUAL "2.8.0")
    set(_archive_prefix "libtorch-shared-with-deps")
  else()
    set(_archive_prefix "libtorch-cxx11-abi-shared-with-deps")
  endif()
  string(CONCAT _url
    "https://download.pytorch.org/libtorch/${LIBTORCH_CUDA_VERSION}/"
    "${_archive_prefix}-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA_VERSION}.zip")
  set(_archive "${CMAKE_CURRENT_BINARY_DIR}/libtorch.zip")
  set(_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/libtorch_extracted")

  if(NOT EXISTS "${_extract_dir}/libtorch/share/cmake/Torch/TorchConfig.cmake")
    message(STATUS "${PROJECT_NAME}: downloading LibTorch ${LIBTORCH_VERSION}+${LIBTORCH_CUDA_VERSION}")
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
    message(STATUS "${PROJECT_NAME}: using cached LibTorch at ${_extract_dir}")
  endif()

  install(DIRECTORY "${_extract_dir}/libtorch/"
    DESTINATION "opt/${PROJECT_NAME}"
    USE_SOURCE_PERMISSIONS
  )

  # Pre-strip libtorch_cpu.so so dh_strip's objcopy doesn't fail on its
  # non-standard ELF section table on the buildfarm.
  install(CODE "
    set(_lt \"\${CMAKE_INSTALL_PREFIX}/opt/${PROJECT_NAME}/lib/libtorch_cpu.so\")
    if(EXISTS \"\${_lt}\" AND NOT IS_SYMLINK \"\${_lt}\")
      execute_process(
        COMMAND strip --strip-unneeded \"\${_lt}\"
        OUTPUT_QUIET ERROR_QUIET)
    endif()
  ")

  ament_index_register_resource("vendor_packages" CONTENT "opt/${PROJECT_NAME}")

  # Install environment hooks so downstream binaries can locate the vendored
  # shared libraries at runtime (LD_LIBRARY_PATH / PATH).
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
else()
  message(STATUS "${PROJECT_NAME}: system LibTorch ${Torch_VERSION} found, skipping download")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(
  CONFIG_EXTRAS_POST "libtorch_vendor-extras.cmake.in"
)
