# Copyright 2026 NVIDIA Corporation
# Licensed under the Apache License, Version 2.0

cmake_minimum_required(VERSION 3.20)
project(python3_torch_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(PYTHON3_TORCH_VENDOR_TORCH_VERSION "2.9.1")
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
if(NOT _system_arch MATCHES "x86_64|amd64")
  message(FATAL_ERROR
    "${PROJECT_NAME}: the official CPU Torch wheel is supported on amd64 only")
endif()

set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheel")
set(_stage_dir "${CMAKE_CURRENT_BINARY_DIR}/stage")
set(_marker "${_stage_dir}/.torch-${PYTHON3_TORCH_VENDOR_TORCH_VERSION}-cpu")
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
      --index-url "https://download.pytorch.org/whl/cpu"
      --extra-index-url "https://pypi.org/simple"
      "torch==${PYTHON3_TORCH_VENDOR_TORCH_VERSION}"
    COMMAND_ERROR_IS_FATAL ANY)
  file(GLOB _all_wheels "${_wheel_dir}/*.whl")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip install
      --disable-pip-version-check
      --no-cache-dir
      --no-deps
      --only-binary=:all:
      --target "${_stage_dir}"
      ${_all_wheels}
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${_stage_dir}"
      "PYTHONNOUSERSITE=1"
      "${Python3_EXECUTABLE}" -c
      "import torch; assert torch.__version__.split('+')[0] == '${PYTHON3_TORCH_VENDOR_TORCH_VERSION}'; assert torch.version.cuda is None"
    COMMAND_ERROR_IS_FATAL ANY)
  file(WRITE "${_marker}" "")
endif()

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
set(_vendor_python_dir "opt/${PROJECT_NAME}/${PYTHON_INSTALL_DIR}")
install(DIRECTORY "${_stage_dir}/"
  DESTINATION "${_vendor_python_dir}"
  PATTERN ".torch-*" EXCLUDE)

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

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST "python3_torch_vendor-extras.cmake.in")
