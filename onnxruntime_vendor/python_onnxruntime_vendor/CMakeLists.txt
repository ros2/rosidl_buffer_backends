cmake_minimum_required(VERSION 3.20)
project(python_onnxruntime_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(PYTHON_ONNXRUNTIME_VENDOR_VERSION "1.29.0")
set(PYTHON_ONNXRUNTIME_VENDOR_SUPPORTED_VERSION_RANGE ">=1.27.0")

option(FORCE_BUILD_VENDOR_PKG "Stage the pinned Python provider instead of reusing it" OFF)
include(cmake/python_provider_reuse.cmake)
python_provider_find_compatible(onnxruntime "${PYTHON_ONNXRUNTIME_VENDOR_SUPPORTED_VERSION_RANGE}" FALSE
  _reuse_root _reuse_variant PYTHON_ONNXRUNTIME_VENDOR_SELECTED_VERSION)
if(_reuse_root)
  python_provider_reuse_hooks("${_reuse_root}")
else()
  set(PYTHON_ONNXRUNTIME_VENDOR_SELECTED_VERSION "${PYTHON_ONNXRUNTIME_VENDOR_VERSION}")
  set(_python_package "onnxruntime")
  set(_index_url "https://pypi.org/simple")

  ament_get_python_install_dir(PYTHON_INSTALL_DIR)
  set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheel")
  set(_stage_root "${CMAKE_CURRENT_BINARY_DIR}/python_stage")
  set(_python_stage "${_stage_root}/${PYTHON_INSTALL_DIR}")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
  string(CONCAT _stage_marker
    "${_stage_root}/.${_python_package}-"
    "${PYTHON_ONNXRUNTIME_VENDOR_VERSION}-"
    "py${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}-${_system_arch}")

  if(NOT EXISTS "${_stage_marker}")
    file(REMOVE_RECURSE "${_wheel_dir}" "${_stage_root}")
    file(MAKE_DIRECTORY "${_wheel_dir}" "${_python_stage}")

    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip download
        --disable-pip-version-check
        --no-cache-dir
        --no-deps
        --only-binary=:all:
        --dest "${_wheel_dir}"
        --index-url "${_index_url}"
        "${_python_package}==${PYTHON_ONNXRUNTIME_VENDOR_VERSION}"
      COMMAND_ERROR_IS_FATAL ANY)

    file(GLOB _onnxruntime_wheels "${_wheel_dir}/*.whl")
    list(LENGTH _onnxruntime_wheels _wheel_count)
    if(NOT _wheel_count EQUAL 1)
      message(FATAL_ERROR
        "${PROJECT_NAME}: expected one Python wheel, found ${_wheel_count}")
    endif()

    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pip install
        --disable-pip-version-check
        --no-cache-dir
        --no-deps
        --only-binary=:all:
        --target "${_python_stage}"
        "${_onnxruntime_wheels}"
      COMMAND_ERROR_IS_FATAL ANY)

    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${_python_stage}" "PYTHONNOUSERSITE=1"
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe_python.py"
      onnxruntime "${PYTHON_ONNXRUNTIME_VENDOR_VERSION}" FALSE "${CUDAToolkit_VERSION_MAJOR}"
      COMMAND_ERROR_IS_FATAL ANY)
    file(WRITE "${_stage_marker}" "")
  endif()

  ament_environment_hooks(environment/pythonpath.dsv.in)
  install(DIRECTORY "${_python_stage}/"
    DESTINATION "opt/${PROJECT_NAME}/${PYTHON_INSTALL_DIR}"
    USE_SOURCE_PERMISSIONS)

endif()
install(DIRECTORY cmake/ DESTINATION share/${PROJECT_NAME}/cmake)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package(CONFIG_EXTRAS_POST "python_onnxruntime_vendor-extras.cmake.in")
