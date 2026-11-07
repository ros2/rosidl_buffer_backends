# Copyright 2026 Open Source Robotics Foundation, Inc.
# Copyright 2026 NVIDIA Corporation
# Copyright 2026 Meta Platforms, Inc. and affiliates.
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

include_guard(GLOBAL)

# Looks for an installed ExecuTorch in range, with the given components. A
# `pip install executorch` has no __init__.py, so the package directory comes
# from its search locations rather than from a module file. The build host's
# Python does not describe the target when cross-compiling.
macro(executorch_vendor_find_installed)
  if(NOT FORCE_BUILD_VENDOR_PKG AND NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_FOUND)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
          "import importlib.util as u; s = u.find_spec('executorch'); s and print(next(iter(s.submodule_search_locations)))"
        OUTPUT_VARIABLE _python_executorch_root
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    endif()
    find_package(executorch ${EXECUTORCH_VENDOR_VERSION_RANGE} CONFIG QUIET
      HINTS "${_python_executorch_root}" COMPONENTS ${ARGN})
    # A nightly reports the release it leads to, and nightlies before this date lack
    # APIs that executorch_conversions uses.
    if(executorch_FOUND AND EXECUTORCH_BUILD_VERSION MATCHES "\\.dev([0-9]+)"
      AND CMAKE_MATCH_1 LESS 20260930)
      message(STATUS "${PROJECT_NAME}: not reusing ExecuTorch ${EXECUTORCH_BUILD_VERSION}, "
        "a nightly from before 2026-09-30")
      set(executorch_FOUND FALSE)
    endif()
  endif()
endmacro()

# A dependency that found another ExecuTorch first leaves its imported targets,
# and find_package(executorch) then reuses them silently.
macro(executorch_vendor_check_existing_targets)
  if(TARGET executorch::runtime)
    get_target_property(_existing_runtime executorch::runtime IMPORTED_LOCATION)
    get_filename_component(_existing_root "${_existing_runtime}" DIRECTORY)
    get_filename_component(_existing_root "${_existing_root}/.." REALPATH)
    # Find the package root the way ExecuTorch's config does: it holds share/cmake.
    set(_selected_root "")
    foreach(_candidate
        "${executorch_DIR}/.." "${executorch_DIR}/../.." "${executorch_DIR}/../../..")
      if(_selected_root STREQUAL "" AND EXISTS "${_candidate}/share/cmake/executorch-config.cmake")
        get_filename_component(_selected_root "${_candidate}" REALPATH)
      endif()
    endforeach()
    if(NOT _existing_root STREQUAL _selected_root)
      message(FATAL_ERROR "ExecuTorch in ${_existing_root} already defined imported targets. "
        "Find executorch_cuda_vendor, or else executorch_vendor, before anything that "
        "finds ExecuTorch.")
    endif()
  endif()
endmacro()

# Rewriting the ELF layout keeps debugedit from corrupting section headers that
# wheel processing placed before later sections. Symbols are kept, so Debian's
# dh_strip can still make debug packages. Same as libtorch_vendor.
function(executorch_vendor_prepare_libraries root)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR EXISTS "${root}/.elf-normalized")
    return()
  endif()
  file(GLOB_RECURSE libraries LIST_DIRECTORIES FALSE "${root}/*.so" "${root}/*.so.*")
  foreach(library IN LISTS libraries)
    if(IS_SYMLINK "${library}")
      continue()
    endif()
    file(READ "${library}" magic LIMIT 4 HEX)
    if(magic STREQUAL "7f454c46")
      execute_process(COMMAND "${CMAKE_OBJCOPY}" --preserve-dates "${library}"
        COMMAND_ERROR_IS_FATAL ANY)
    endif()
  endforeach()
  file(WRITE "${root}/.elf-normalized" "")
endfunction()

# Installs the C++ runtime of a pinned ExecuTorch wheel under opt/<project>.
# Extra arguments are more archive patterns to extract. There is no library path
# hook: it would put this runtime ahead of any other ExecuTorch in every process,
# and the libraries that link it find it through their RUNPATH.
function(executorch_vendor_install_wheel url sha256)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "${PROJECT_NAME}: the pinned ExecuTorch wheels are Linux builds, "
      "so they cannot be installed on ${CMAKE_SYSTEM_NAME}")
  endif()
  set(_wheel "${CMAKE_CURRENT_BINARY_DIR}/executorch.whl")
  # Keyed on the hash, so a new pin downloads and checks its wheel in an existing build tree.
  set(_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/executorch-wheel-${sha256}")
  # Written after a complete extraction, so an interrupted one starts over.
  if(NOT EXISTS "${_extract_dir}/.complete")
    file(REMOVE_RECURSE "${_extract_dir}")
    message(STATUS "${PROJECT_NAME}: downloading ${url}")
    file(DOWNLOAD "${url}" "${_wheel}" EXPECTED_HASH SHA256=${sha256} STATUS _status)
    list(GET _status 0 _code)
    if(NOT _code EQUAL 0)
      list(GET _status 1 _message)
      message(FATAL_ERROR "${PROJECT_NAME}: download failed: ${_message}")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_wheel}" DESTINATION "${_extract_dir}"
      PATTERNS "executorch/lib/*" "executorch/include/*" "executorch/share/*"
        "*.dist-info/licenses/LICENSE" ${ARGN})
    file(REMOVE "${_wheel}")
    file(WRITE "${_extract_dir}/.complete" "")
  endif()
  executorch_vendor_prepare_libraries("${_extract_dir}/executorch")
  file(GLOB _license "${_extract_dir}/*.dist-info/licenses/LICENSE")

  # The package config locates its libraries relative to share/cmake, so the
  # wheel layout is kept as it is. The QNN adapter is left out, since nothing here
  # uses it, and the package config defines its target only when it is present.
  install(DIRECTORY "${_extract_dir}/executorch/"
    DESTINATION "opt/${PROJECT_NAME}"
    USE_SOURCE_PERMISSIONS
    PATTERN ".elf-normalized" EXCLUDE
    PATTERN "libexecutorch_backend_qnn.so*" EXCLUDE)
  install(FILES "${_license}" DESTINATION "share/${PROJECT_NAME}" RENAME LICENSE.executorch)
  # The wheel carries only ExecuTorch's own license, while its libraries also
  # contain the third-party code this file lists.
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES" DESTINATION "share/${PROJECT_NAME}")

  ament_index_register_resource("vendor_packages" CONTENT "opt/${PROJECT_NAME}")
endfunction()
