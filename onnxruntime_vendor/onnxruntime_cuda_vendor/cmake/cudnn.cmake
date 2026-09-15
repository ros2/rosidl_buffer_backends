# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(CUDAToolkit_VERSION_MAJOR EQUAL 12)
  set(ONNXRUNTIME_CUDNN_VERSION "9.10.2.21")
  set(_cudnn_min_version 91002)
else()
  set(ONNXRUNTIME_CUDNN_VERSION "9.24.0.43")
  set(_cudnn_min_version 92400)
endif()
set(_cudnn_package "nvidia-cudnn-cu${CUDAToolkit_VERSION_MAJOR}")
find_library(_cudnn_library NAMES cudnn libcudnn.so.9
  HINTS "${CUDNN_ROOT}" ENV CUDNN_ROOT PATH_SUFFIXES lib lib64 NO_CACHE)
set(_reuse_cudnn FALSE)
if(_cudnn_library AND NOT FORCE_BUILD_VENDOR_PKG)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import ctypes,sys; c=ctypes.CDLL(sys.argv[1]); c.cudnnGetVersion.restype=ctypes.c_size_t; c.cudnnGetCudartVersion.restype=ctypes.c_size_t; assert ${_cudnn_min_version} <= c.cudnnGetVersion() < 100000; assert c.cudnnGetCudartVersion() // 1000 == ${CUDAToolkit_VERSION_MAJOR}"
    "${_cudnn_library}" RESULT_VARIABLE _cudnn_result ERROR_QUIET)
  if(_cudnn_result EQUAL 0)
    set(_reuse_cudnn TRUE)
    get_filename_component(_cudnn_dir "${_cudnn_library}" DIRECTORY)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cudnn.dsv"
      "prepend-non-duplicate;LD_LIBRARY_PATH;${_cudnn_dir}\n")
    ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/cudnn.dsv")
    message(STATUS "${PROJECT_NAME}: reusing CUDA ${CUDAToolkit_VERSION_MAJOR} cuDNN at ${_cudnn_library}")
  endif()
endif()
if(NOT _reuse_cudnn)
  set(_cudnn_stage "${CMAKE_CURRENT_BINARY_DIR}/cudnn-${ONNXRUNTIME_CUDNN_VERSION}-cuda${CUDAToolkit_VERSION_MAJOR}")
  if(NOT EXISTS "${_cudnn_stage}/nvidia/cudnn/lib/libcudnn.so.9")
    set(_cudnn_wheels "${CMAKE_CURRENT_BINARY_DIR}/cudnn_wheel")
    file(REMOVE_RECURSE "${_cudnn_wheels}" "${_cudnn_stage}")
    file(MAKE_DIRECTORY "${_cudnn_wheels}")
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m pip download
      --disable-pip-version-check --no-cache-dir --no-deps --only-binary=:all:
      --index-url https://pypi.org/simple --dest "${_cudnn_wheels}"
      "${_cudnn_package}==${ONNXRUNTIME_CUDNN_VERSION}"
      COMMAND_ERROR_IS_FATAL ANY)
    file(GLOB _wheels "${_cudnn_wheels}/*.whl")
    list(LENGTH _wheels _count)
    if(NOT _count EQUAL 1)
      message(FATAL_ERROR "Expected one NVIDIA cuDNN wheel")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_wheels}" DESTINATION "${_cudnn_stage}")
  endif()
  install(DIRECTORY "${_cudnn_stage}/nvidia/cudnn/lib/"
    DESTINATION "opt/${PROJECT_NAME}/lib" USE_SOURCE_PERMISSIONS
    FILES_MATCHING PATTERN "libcudnn*.so*")
  file(GLOB_RECURSE _cudnn_licenses "${_cudnn_stage}/*LICENSE*")
  install(FILES ${_cudnn_licenses} DESTINATION "share/${PROJECT_NAME}/cudnn_licenses")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cudnn.dsv"
    "prepend-non-duplicate;LD_LIBRARY_PATH;opt/${PROJECT_NAME}/lib\n")
  ament_environment_hooks("${CMAKE_CURRENT_BINARY_DIR}/cudnn.dsv")
endif()
