// Copyright 2026 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DLPACK_CONVERSIONS__VISIBILITY_CONTROL_HPP_
#define DLPACK_CONVERSIONS__VISIBILITY_CONTROL_HPP_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define DLPACK_CONVERSIONS_EXPORT __attribute__ ((dllexport))
    #define DLPACK_CONVERSIONS_IMPORT __attribute__ ((dllimport))
  #else
    #define DLPACK_CONVERSIONS_EXPORT __declspec(dllexport)
    #define DLPACK_CONVERSIONS_IMPORT __declspec(dllimport)
  #endif
  #ifdef DLPACK_CONVERSIONS_BUILDING_DLL
    #define DLPACK_CONVERSIONS_PUBLIC DLPACK_CONVERSIONS_EXPORT
  #else
    #define DLPACK_CONVERSIONS_PUBLIC DLPACK_CONVERSIONS_IMPORT
  #endif
#else
  #define DLPACK_CONVERSIONS_EXPORT __attribute__ ((visibility("default")))
  #define DLPACK_CONVERSIONS_IMPORT
  #define DLPACK_CONVERSIONS_PUBLIC DLPACK_CONVERSIONS_EXPORT
#endif

#endif  // DLPACK_CONVERSIONS__VISIBILITY_CONTROL_HPP_
