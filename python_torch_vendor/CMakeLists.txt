# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(python_torch_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(PYTHON_TORCH_VERSION "" CACHE STRING
  "Python Torch version to install. Auto-selected per CUDA variant if empty.")
set(PYTHON_TORCH_CUDA_VERSION "" CACHE STRING
  "CUDA variant override (cu118, cu121, cu124, cu126, cu128, cu129, cpu). Auto-detected if empty.")

option(FORCE_BUILD_VENDOR_PKG
  "Install the selected wheel even if Python Torch is already available"
  OFF)

if(PYTHON_TORCH_CUDA_VERSION STREQUAL "")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
      set(PYTHON_TORCH_CUDA_VERSION "cu129")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
      set(PYTHON_TORCH_CUDA_VERSION "cu128")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.6")
      set(PYTHON_TORCH_CUDA_VERSION "cu126")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.4")
      set(PYTHON_TORCH_CUDA_VERSION "cu124")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.1")
      set(PYTHON_TORCH_CUDA_VERSION "cu121")
    elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
      set(PYTHON_TORCH_CUDA_VERSION "cu118")
    else()
      set(PYTHON_TORCH_CUDA_VERSION "cpu")
    endif()
  else()
    set(PYTHON_TORCH_CUDA_VERSION "cpu")
  endif()
endif()

if(PYTHON_TORCH_VERSION STREQUAL "")
  if(PYTHON_TORCH_CUDA_VERSION STREQUAL "cu118")
    set(PYTHON_TORCH_VERSION "2.7.1")
  elseif(PYTHON_TORCH_CUDA_VERSION STREQUAL "cu121")
    set(PYTHON_TORCH_VERSION "2.5.1")
  elseif(PYTHON_TORCH_CUDA_VERSION STREQUAL "cu124")
    set(PYTHON_TORCH_VERSION "2.6.0")
  else()
    set(PYTHON_TORCH_VERSION "2.11.0")
  endif()
endif()

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

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.__version__)"
  OUTPUT_VARIABLE _system_torch_version
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _system_torch_rc
  ERROR_QUIET
)

if(_system_torch_rc EQUAL 0 AND NOT FORCE_BUILD_VENDOR_PKG)
  message(STATUS
    "${PROJECT_NAME}: system Python Torch ${_system_torch_version} found, skipping wheel installation")
  if(_is_aarch64 AND NOT _is_tegra)
    message(WARNING
      "${PROJECT_NAME}: reusing Python Torch on a non-Tegra aarch64 platform. "
      "This configuration has only been verified on NVIDIA Tegra.")
  endif()
elseif(_is_tegra)
  message(FATAL_ERROR
    "${PROJECT_NAME}: Python Torch must be provided by JetPack on NVIDIA Tegra")
elseif(_is_aarch64)
  message(FATAL_ERROR
    "${PROJECT_NAME}: no Python Torch wheel is provided for non-Tegra aarch64; "
    "install a compatible platform package and rebuild")
else()
  set(_index_url
    "https://download.pytorch.org/whl/${PYTHON_TORCH_CUDA_VERSION}")
  set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheel")
  set(_stage_dir "${CMAKE_CURRENT_BINARY_DIR}/stage")
  set(_marker
    "${_stage_dir}/.torch-${PYTHON_TORCH_VERSION}-${PYTHON_TORCH_CUDA_VERSION}")

  if(NOT EXISTS "${_marker}")
    file(REMOVE_RECURSE "${_wheel_dir}" "${_stage_dir}")
    file(MAKE_DIRECTORY "${_wheel_dir}" "${_stage_dir}")

    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip download
        --disable-pip-version-check
        --no-cache-dir
        --no-deps
        --only-binary=:all:
        --dest "${_wheel_dir}"
        --index-url "${_index_url}"
        "torch==${PYTHON_TORCH_VERSION}"
      RESULT_VARIABLE _download_rc
      COMMAND_ERROR_IS_FATAL ANY
    )

    file(GLOB _torch_wheels "${_wheel_dir}/torch-*.whl")
    list(LENGTH _torch_wheels _wheel_count)
    if(NOT _wheel_count EQUAL 1)
      message(FATAL_ERROR
        "${PROJECT_NAME}: expected one Torch wheel, found ${_wheel_count}")
    endif()

    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip install
        --disable-pip-version-check
        --no-cache-dir
        --only-binary=:all:
        --target "${_stage_dir}"
        --extra-index-url "${_index_url}"
        "${_torch_wheels}"
      RESULT_VARIABLE _install_rc
      COMMAND_ERROR_IS_FATAL ANY
    )

    set(_torch_query
      "import torch
cuda = torch.version.cuda
variant = 'cpu'
if cuda:
  variant = 'cu' + cuda.replace('.', '')
print(torch.__version__.split('+')[0] + ';' + variant)")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${_stage_dir}"
        "PYTHONNOUSERSITE=1"
        "${Python3_EXECUTABLE}" -c "${_torch_query}"
      OUTPUT_VARIABLE _vendored_torch
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _vendored_query_rc
      COMMAND_ERROR_IS_FATAL ANY
    )
    list(GET _vendored_torch 0 _vendored_version)
    list(GET _vendored_torch 1 _vendored_variant)
    if(NOT _vendored_version VERSION_EQUAL PYTHON_TORCH_VERSION OR
        NOT _vendored_variant STREQUAL PYTHON_TORCH_CUDA_VERSION)
      message(FATAL_ERROR
        "${PROJECT_NAME}: downloaded torch ${_vendored_version}+${_vendored_variant} "
        "does not match ${PYTHON_TORCH_VERSION}+${PYTHON_TORCH_CUDA_VERSION}")
    endif()
    file(WRITE "${_marker}" "")
  endif()

  ament_get_python_install_dir(PYTHON_INSTALL_DIR)
  _ament_cmake_python_register_environment_hook()
  install(DIRECTORY "${_stage_dir}/"
    DESTINATION "${PYTHON_INSTALL_DIR}"
    PATTERN ".torch-*" EXCLUDE
  )
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(
  CONFIG_EXTRAS_POST "python_torch_vendor-extras.cmake.in"
)
