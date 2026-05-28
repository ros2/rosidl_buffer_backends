# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(libtorch_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_vendor_package REQUIRED)

set(LIBTORCH_VERSION "" CACHE STRING
  "LibTorch version to download. Auto-selected per CUDA variant if empty.")
set(LIBTORCH_CUDA_VERSION "" CACHE STRING
  "CUDA variant override (cu118, cu121, cu124, cu126, cu128, cu129, cu130, cpu). Auto-detected if empty.")
set(LIBTORCH_URL "" CACHE STRING
  "Explicit LibTorch or PyTorch wheel archive URL.")
set(LIBTORCH_ARCHIVE_LAYOUT "" CACHE STRING
  "Archive layout override: libtorch or torch_wheel. Auto-detected if empty.")

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

# Provide a non-empty CMAKE_CUDA_ARCHITECTURES (required by CMake 3.28+
# when find_package(Torch) calls enable_language(CUDA)).
if(NOT CMAKE_CUDA_ARCHITECTURES)
  execute_process(
    COMMAND python3 -c "import torch; print(';'.join(a.split('_',1)[-1] for a in torch.cuda.get_arch_list()))"
    OUTPUT_VARIABLE _torch_arch_list
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_arch_rc
    ERROR_QUIET
  )
  if(_torch_arch_rc EQUAL 0 AND _torch_arch_list)
    set(CMAKE_CUDA_ARCHITECTURES "${_torch_arch_list}" CACHE STRING
      "CUDA archs (auto-detected by libtorch_vendor)" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING
      "CUDA archs (libtorch_vendor fallback, satisfies CMake validation)" FORCE)
  endif()
endif()

# PyTorch only publishes pre-built LibTorch archives for x86_64
# On aarch64 (e.g. NVIDIA Jetson) LibTorch ships within Jetpack,
# so the platform-provided installation must be reused.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
if(_system_arch MATCHES "aarch64|arm64")
  set(_is_aarch64 TRUE)
else()
  set(_is_aarch64 FALSE)
endif()

# Help find_package(Torch) locate a pip/system-installed LibTorch. This enables
# "reuse if present" on x86_64 and is the primary discovery path on aarch64.
if(NOT Torch_DIR)
  execute_process(
    COMMAND python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))"
    OUTPUT_VARIABLE _py_torch_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _py_torch_rc
    ERROR_QUIET
  )
  if(_py_torch_rc EQUAL 0 AND EXISTS "${_py_torch_dir}/TorchConfig.cmake")
    set(Torch_DIR "${_py_torch_dir}")
    message(STATUS "${PROJECT_NAME}: discovered LibTorch via python3 at ${Torch_DIR}")
  endif()
endif()

find_package(Torch QUIET)

if(Torch_FOUND AND NOT FORCE_BUILD_VENDOR_PKG)
  message(STATUS "${PROJECT_NAME}: system LibTorch ${Torch_VERSION} found, skipping download")
  set(LIBTORCH_VENDOR_TORCH_DIR "${Torch_DIR}" CACHE PATH
    "Torch_DIR found by libtorch_vendor")
elseif(_is_aarch64)
  message(FATAL_ERROR
    "${PROJECT_NAME}: no pre-built LibTorch archive is published for aarch64. "
    "On NVIDIA Jetson, LibTorch ships within JetPack, so the platform-provided "
    "installation must be reused. Ensure the JetPack PyTorch is installed so that "
    "'python3 -c \"import torch\"' or find_package(Torch) can locate it, then "
    "rebuild. Set Torch_DIR explicitly if it lives in a non-standard location.")
else()
  if(Torch_FOUND)
    message(STATUS "${PROJECT_NAME}: forcing vendor build (FORCE_BUILD_VENDOR_PKG)")
  endif()

  if(LIBTORCH_URL STREQUAL "")
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
  else()
    set(_url "${LIBTORCH_URL}")
  endif()

  if(LIBTORCH_ARCHIVE_LAYOUT STREQUAL "")
    if(_url MATCHES "\\.whl($|[?#])")
      set(LIBTORCH_ARCHIVE_LAYOUT "torch_wheel")
    else()
      set(LIBTORCH_ARCHIVE_LAYOUT "libtorch")
    endif()
  endif()

  set(_archive "${CMAKE_CURRENT_BINARY_DIR}/libtorch_archive")
  set(_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/libtorch_extracted")

  if(LIBTORCH_ARCHIVE_LAYOUT STREQUAL "torch_wheel")
    set(_torch_root "${_extract_dir}/torch")
  elseif(LIBTORCH_ARCHIVE_LAYOUT STREQUAL "libtorch")
    set(_torch_root "${_extract_dir}/libtorch")
  else()
    message(FATAL_ERROR
      "${PROJECT_NAME}: unsupported LIBTORCH_ARCHIVE_LAYOUT '${LIBTORCH_ARCHIVE_LAYOUT}'")
  endif()

  if(NOT EXISTS "${_torch_root}/share/cmake/Torch/TorchConfig.cmake")
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
    message(STATUS "${PROJECT_NAME}: using cached LibTorch at ${_torch_root}")
  endif()

  install(DIRECTORY "${_torch_root}/"
    DESTINATION "opt/${PROJECT_NAME}"
    USE_SOURCE_PERMISSIONS
  )

  # Pre-strip libtorch_cpu.so so dh_strip's objcopy doesn't fail on its
  # non-standard ELF section table on the buildfarm.
  install(CODE "
    set(_lt \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/opt/${PROJECT_NAME}/lib/libtorch_cpu.so\")
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
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(
  CONFIG_EXTRAS_POST "libtorch_vendor-extras.cmake.in"
)

