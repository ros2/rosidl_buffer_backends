// Copyright 2026 Open Source Robotics Foundation, Inc.
// Modified for the HIP port by ObeliskGate, 2026.
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

#ifndef HIP_BUFFER__VISIBILITY_CONTROL_H_
#define HIP_BUFFER__VISIBILITY_CONTROL_H_

#ifdef __cplusplus
extern "C"
{
#endif

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define HIP_BUFFER_EXPORT __attribute__ ((dllexport))
    #define HIP_BUFFER_IMPORT __attribute__ ((dllimport))
  #else
    #define HIP_BUFFER_EXPORT __declspec(dllexport)
    #define HIP_BUFFER_IMPORT __declspec(dllimport)
  #endif
  #ifdef HIP_BUFFER_BUILDING_DLL
    #define HIP_BUFFER_PUBLIC HIP_BUFFER_EXPORT
  #else
    #define HIP_BUFFER_PUBLIC HIP_BUFFER_IMPORT
  #endif
  #define HIP_BUFFER_LOCAL
#else
  #define HIP_BUFFER_EXPORT __attribute__ ((visibility("default")))
  #define HIP_BUFFER_IMPORT
  #if __GNUC__ >= 4
    #define HIP_BUFFER_PUBLIC __attribute__ ((visibility("default")))
    #define HIP_BUFFER_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define HIP_BUFFER_PUBLIC
    #define HIP_BUFFER_LOCAL
  #endif
#endif

#ifdef __cplusplus
}
#endif

#endif  // HIP_BUFFER__VISIBILITY_CONTROL_H_
