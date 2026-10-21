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

#ifndef ONNXRUNTIME_CONVERSIONS__VISIBILITY_CONTROL_HPP_
#define ONNXRUNTIME_CONVERSIONS__VISIBILITY_CONTROL_HPP_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define ONNXRUNTIME_CONVERSIONS_EXPORT __attribute__((dllexport))
    #define ONNXRUNTIME_CONVERSIONS_IMPORT __attribute__((dllimport))
  #else
    #define ONNXRUNTIME_CONVERSIONS_EXPORT __declspec(dllexport)
    #define ONNXRUNTIME_CONVERSIONS_IMPORT __declspec(dllimport)
  #endif
  #ifdef ONNXRUNTIME_CONVERSIONS_BUILDING_DLL
    #define ONNXRUNTIME_CONVERSIONS_PUBLIC ONNXRUNTIME_CONVERSIONS_EXPORT
  #else
    #define ONNXRUNTIME_CONVERSIONS_PUBLIC ONNXRUNTIME_CONVERSIONS_IMPORT
  #endif
#else
  #define ONNXRUNTIME_CONVERSIONS_PUBLIC __attribute__((visibility("default")))
#endif

#endif  // ONNXRUNTIME_CONVERSIONS__VISIBILITY_CONTROL_HPP_
