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

#ifndef TORCH_CONVERSIONS__VISIBILITY_CONTROL_HPP_
#define TORCH_CONVERSIONS__VISIBILITY_CONTROL_HPP_

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define TORCH_CONVERSIONS_EXPORT __attribute__((dllexport))
    #define TORCH_CONVERSIONS_IMPORT __attribute__((dllimport))
  #else
    #define TORCH_CONVERSIONS_EXPORT __declspec(dllexport)
    #define TORCH_CONVERSIONS_IMPORT __declspec(dllimport)
  #endif
  #ifdef TORCH_CONVERSIONS_BUILDING_DLL
    #define TORCH_CONVERSIONS_PUBLIC TORCH_CONVERSIONS_EXPORT
  #else
    #define TORCH_CONVERSIONS_PUBLIC TORCH_CONVERSIONS_IMPORT
  #endif
#else
  #define TORCH_CONVERSIONS_PUBLIC __attribute__((visibility("default")))
#endif

#endif  // TORCH_CONVERSIONS__VISIBILITY_CONTROL_HPP_
