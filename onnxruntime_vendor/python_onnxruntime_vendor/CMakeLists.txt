cmake_minimum_required(VERSION 3.20)
project(python_onnxruntime_vendor)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(PYTHON_ONNXRUNTIME_VENDOR_VERSION "1.28.0" CACHE STRING
  "ONNX Runtime Python wheel version")
set(_python_package "onnxruntime")
set(_index_url "https://pypi.org/simple")

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
set(_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/wheel")
set(_stage_root "${CMAKE_CURRENT_BINARY_DIR}/python_stage")
set(_python_stage "${_stage_root}/${PYTHON_INSTALL_DIR}")
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _system_arch)
string(CONCAT _stage_marker
  "${_stage_root}/.${_python_package}-${PYTHON_ONNXRUNTIME_VENDOR_VERSION}-"
  "cpu-bundled-"
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

  file(WRITE "${_stage_marker}" "")
endif()

_ament_cmake_python_register_environment_hook()
install(DIRECTORY "${_python_stage}/"
  DESTINATION "${PYTHON_INSTALL_DIR}"
  USE_SOURCE_PERMISSIONS)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
