# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(python3_torch_cuda_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/python3_torch_cuda_vendor_policy.cmake)

set(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT "" CACHE STRING
  "CUDA wheel variant override (cu126, cu128, cu130).")
option(FORCE_BUILD_PYTHON3_TORCH_CUDA_VENDOR_PKG
  "Disable reuse of the JetPack-provided Python Torch distribution"
  OFF)

if(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT STREQUAL "cpu")
  message(FATAL_ERROR
    "${PROJECT_NAME}: 'cpu' is not a CUDA variant; use python3_torch_vendor")
elseif(NOT PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT STREQUAL "")
  python3_torch_cuda_vendor_validate_variant(
    "${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")
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

set(_reuse_system_torch FALSE)
if(_is_tegra)
  set(_system_torch_query
    "import torch
cuda = torch.version.cuda
variant = 'cpu'
if cuda is not None:
  variant = 'cu' + cuda.replace('.', '')
print(torch.__version__.split('+')[0] + ';' + variant)")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "${_system_torch_query}"
    OUTPUT_VARIABLE _system_torch
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _system_torch_rc
    ERROR_QUIET
  )
  set(_system_torch_compatible FALSE)
  if(_system_torch_rc EQUAL 0)
    list(GET _system_torch 0 _system_torch_version)
    list(GET _system_torch 1 _system_torch_variant)
    python3_torch_cuda_vendor_can_reuse(
      "${_system_torch_version}"
      "${_system_torch_variant}"
      _system_torch_compatible)
  endif()

  if(FORCE_BUILD_PYTHON3_TORCH_CUDA_VENDOR_PKG)
    message(FATAL_ERROR
      "${PROJECT_NAME}: CUDA PyTorch wheels are not staged on Tegra; "
      "the JetPack-provided installation must be reused")
  elseif(NOT _system_torch_rc EQUAL 0 OR _system_torch_variant STREQUAL "cpu")
    message(FATAL_ERROR
      "${PROJECT_NAME}: JetPack CUDA-enabled Python Torch is required on "
      "NVIDIA Tegra but was not found")
  elseif(NOT _system_torch_version STREQUAL
      PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION)
    message(FATAL_ERROR
      "${PROJECT_NAME}: JetPack provides PyTorch ${_system_torch_version}; "
      "exactly ${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION} is required")
  elseif(NOT _system_torch_compatible)
    message(FATAL_ERROR
      "${PROJECT_NAME}: JetPack provides unsupported CUDA variant "
      "${_system_torch_variant}")
  elseif(NOT PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT STREQUAL "" AND
      NOT PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT STREQUAL _system_torch_variant)
    message(FATAL_ERROR
      "${PROJECT_NAME}: JetPack provides ${_system_torch_variant}, which does "
      "not match requested variant ${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")
  else()
    set(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT "${_system_torch_variant}")
    set(_reuse_system_torch TRUE)
  endif()
endif()

if(_reuse_system_torch)
  message(STATUS
    "${PROJECT_NAME}: system CUDA Python Torch "
    "${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}+${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT} "
    "found, skipping wheel installation")
else()
  if(PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT STREQUAL "")
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR
        "${PROJECT_NAME}: no CUDA Toolkit was found. Install a supported "
        "CUDA Toolkit or set PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT explicitly.")
    else()
      python3_torch_cuda_vendor_variant_for_cuda(
        "${CUDAToolkit_VERSION}"
        PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT)
    endif()
  endif()

  if(_is_aarch64)
    message(FATAL_ERROR
      "${PROJECT_NAME}: no CUDA Python Torch wheel is provided for non-Tegra "
      "aarch64; install a compatible platform package and rebuild")
  endif()

  set(_index_url
    "https://download.pytorch.org/whl/${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")
  set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheel")
  set(_stage_dir "${CMAKE_CURRENT_BINARY_DIR}/stage")
  set(_marker
    "${_stage_dir}/.torch-${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}-${PYTHON3_TORCH_CUDA_VENDOR_CUDA_VARIANT}")

  if(NOT EXISTS "${_marker}")
    file(REMOVE_RECURSE "${_stage_dir}")
    file(MAKE_DIRECTORY "${_wheel_dir}" "${_stage_dir}")

    file(GLOB _torch_wheels "${_wheel_dir}/torch-*.whl")
    list(LENGTH _torch_wheels _wheel_count)
    if(NOT _wheel_count EQUAL 1)
      file(REMOVE_RECURSE "${_wheel_dir}")
      file(MAKE_DIRECTORY "${_wheel_dir}")
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pip download
          --disable-pip-version-check
          --no-cache-dir
          --no-deps
          --only-binary=:all:
          --retries 10
          --timeout 600
          --dest "${_wheel_dir}"
          --index-url "${_index_url}"
          "torch==${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}"
        COMMAND_ERROR_IS_FATAL ANY
      )
      file(GLOB _torch_wheels "${_wheel_dir}/torch-*.whl")
      list(LENGTH _torch_wheels _wheel_count)
      if(NOT _wheel_count EQUAL 1)
        message(FATAL_ERROR
          "${PROJECT_NAME}: expected one Torch wheel, found ${_wheel_count}")
      endif()
    endif()

    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip download
        --disable-pip-version-check
        --no-cache-dir
        --only-binary=:all:
        --retries 10
        --timeout 600
        --dest "${_wheel_dir}"
        --find-links "${_wheel_dir}"
        --index-url "${_index_url}"
        --extra-index-url "https://pypi.org/simple"
        "torch==${PYTHON3_TORCH_CUDA_VENDOR_TORCH_VERSION}"
      COMMAND_ERROR_IS_FATAL ANY
    )

    file(GLOB _all_wheels "${_wheel_dir}/*.whl")
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

  # Omit optional HPC transports; retain NVSHMEM core and CUDA libraries.
  file(GLOB _unused_transport_plugins
    "${_stage_dir}/nvidia/cufile/lib/libcufile_rdma*"
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
  )

  # Strip upstream debug sections before debhelper generates dbgsym packages.
  install(CODE "
    file(GLOB_RECURSE _elfs
      \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_vendor_python_dir}/*.so\"
      \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_vendor_python_dir}/*.so.*\")
    foreach(_elf IN LISTS _elfs)
      if(EXISTS \"\${_elf}\" AND NOT IS_SYMLINK \"\${_elf}\")
        execute_process(
          COMMAND \"${CMAKE_STRIP}\" --strip-unneeded \"\${_elf}\"
          COMMAND_ERROR_IS_FATAL ANY)
      endif()
    endforeach()
  ")

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
