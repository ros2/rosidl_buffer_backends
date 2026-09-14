# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(python3_torch_cuda_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/python3_torch_cuda_vendor_policy.cmake)

set(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT "" CACHE STRING
  "CUDA wheel variant override (cu130, cu132).")
option(FORCE_BUILD_PYTHON3_TORCH_CUDA_VENDOR_PKG
  "Disable reuse of the installed Python Torch distribution"
  OFF)

if(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT STREQUAL "cpu")
  message(FATAL_ERROR
    "${PROJECT_NAME}: 'cpu' is not a CUDA variant; use python3_torch_vendor")
elseif(NOT PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT STREQUAL "")
  python3_torch_cuda_vendor_validate_variant(
    "${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")
endif()

set(_requested_variant "${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")
find_package(CUDAToolkit QUIET)

option(FORCE_BUILD_VENDOR_PKG "Stage the pinned Python provider instead of reusing it" OFF)
if(FORCE_BUILD_PYTHON3_TORCH_CUDA_VENDOR_PKG)
  set(FORCE_BUILD_VENDOR_PKG ON)
endif()
find_package(python3_torch_vendor REQUIRED)
include("${python3_torch_vendor_DIR}/python_provider_reuse.cmake")
python_provider_find_compatible(torch "${PYTHON3_TORCH_VENDOR_SUPPORTED_VERSION_RANGE}" TRUE
  _reuse_root _reuse_variant PYTHON3_TORCH_CUDA_VENDOR_SELECTED_VERSION)
if(_reuse_root AND _requested_variant AND NOT _reuse_variant STREQUAL _requested_variant)
  message(STATUS "${PROJECT_NAME}: installed Torch build ${_reuse_variant} differs from "
    "explicitly requested ${_requested_variant}; staging pinned fallback")
  set(_reuse_root "")
endif()
if(_reuse_root)
  set(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT "${_reuse_variant}")
  python_provider_reuse_hooks("${_reuse_root}")
else()
  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "${PROJECT_NAME}: the pinned CUDA fallback requires a CUDA 13.1 toolkit. "
      "Install it through rosdep/APT and select it with -DCUDAToolkit_ROOT=/usr/local/cuda-13.1.")
  endif()
  python3_torch_cuda_vendor_variant_for_cuda("${CUDAToolkit_VERSION}" _fallback_variant)
  if(_requested_variant)
    python3_torch_cuda_vendor_validate_variant_for_cuda("${_requested_variant}" "${CUDAToolkit_VERSION}")
    set(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT "${_requested_variant}")
  else()
    set(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT "${_fallback_variant}")
  endif()
  set(PYTHON3_TORCH_CUDA_VENDOR_SELECTED_VERSION "${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
  if(NOT _system_arch MATCHES "^(x86_64|amd64|aarch64|arm64)$")
    message(FATAL_ERROR "${PROJECT_NAME}: unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}'")
  endif()
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "${PROJECT_NAME}: wheel staging requires a native build for the target architecture")
  endif()
  set(_index_url
    "https://download.pytorch.org/whl/${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")
  set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheel-${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}-${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")
  set(_stage_dir "${CMAKE_CURRENT_BINARY_DIR}/stage")
  string(CONCAT _marker
    "${_stage_dir}/.torch-${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}-${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}"
    "-py${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}-${_system_arch}")

  if(NOT EXISTS "${_marker}")
    file(REMOVE_RECURSE "${_wheel_dir}" "${_stage_dir}")
    file(MAKE_DIRECTORY "${_wheel_dir}" "${_stage_dir}")

    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip download
        --disable-pip-version-check
        --no-cache-dir
        --only-binary=:all:
        --retries 10
        --timeout 600
        --dest "${_wheel_dir}"
        --index-url "${_index_url}"
        --extra-index-url "https://pypi.org/simple"
        "torch==${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}+${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}"
      COMMAND_ERROR_IS_FATAL ANY
    )

    file(GLOB _all_wheels "${_wheel_dir}/*.whl")
    # setuptools is supplied by the system package.
    list(FILTER _all_wheels EXCLUDE REGEX "/setuptools-[^/]*[.]whl$")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip install
        --disable-pip-version-check
        --no-cache-dir
        --no-deps
        --only-binary=:all:
        --retries 10
        --timeout 600
        --target "${_stage_dir}"
        ${_all_wheels}
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
      COMMAND_ERROR_IS_FATAL ANY
    )
    list(GET _vendored_torch 0 _vendored_version)
    list(GET _vendored_torch 1 _vendored_variant)
    if(NOT _vendored_version STREQUAL PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION OR
        NOT _vendored_variant STREQUAL PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT)
      message(FATAL_ERROR
        "${PROJECT_NAME}: downloaded torch ${_vendored_version}+${_vendored_variant} "
        "does not match ${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}+"
        "${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")
    endif()
    file(WRITE "${_marker}" "")
  endif()

  file(GLOB _unused_transport_plugins
    "${_stage_dir}/nvidia/cufile/lib/libcufile_rdma*"
    "${_stage_dir}/nvidia/cu13/lib/libcufile_rdma*"
    "${_stage_dir}/nvidia/nvshmem/lib/nvshmem_transport_ib*"
    "${_stage_dir}/nvidia/nvshmem/lib/nvshmem_transport_libfabric*"
    "${_stage_dir}/nvidia/nvshmem/lib/nvshmem_transport_ucx*"
    "${_stage_dir}/nvidia/nvshmem/lib/nvshmem_bootstrap_mpi*"
    "${_stage_dir}/nvidia/nvshmem/lib/nvshmem_bootstrap_pmix*"
    "${_stage_dir}/nvidia/nvshmem/lib/nvshmem_bootstrap_shmem*"
  )
  if(_unused_transport_plugins)
    file(REMOVE ${_unused_transport_plugins})
  endif()

  ament_get_python_install_dir(PYTHON_INSTALL_DIR)
  set(_vendor_python_dir "opt/${PROJECT_NAME}/${PYTHON_INSTALL_DIR}")

  install(DIRECTORY "${_stage_dir}/"
    DESTINATION "${_vendor_python_dir}"
    PATTERN ".torch-*" EXCLUDE
    PATTERN "setuptools" EXCLUDE
    PATTERN "setuptools-*.dist-info" EXCLUDE
    PATTERN "_distutils_hack" EXCLUDE
    PATTERN "pkg_resources" EXCLUDE
    PATTERN "distutils-precedence.pth" EXCLUDE
  )

  configure_file(environment/pythonpath.dsv.in pythonpath.dsv @ONLY)
  ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/pythonpath.dsv")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  add_test(
    NAME python3_torch_cuda_vendor_policy
    COMMAND "${CMAKE_COMMAND}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/test/test_version_policy.cmake"
  )
endif()

ament_package(
  CONFIG_EXTRAS_POST "python3_torch_cuda_vendor-extras.cmake.in"
)
