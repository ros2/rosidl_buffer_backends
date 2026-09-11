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

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "onnxruntime_conversions/onnxruntime_conversions.hpp"

int main(int argc, char ** argv)
{
  const bool require_cuda = argc > 1 && std::string(argv[1]) == "--require-cuda";
  const auto backends = onnxruntime_conversions::available_backends();
  std::cout << "available=";
  for (const auto & backend : backends) {
    std::cout << backend << ',';
  }
  std::cout << " default=" << onnxruntime_conversions::default_backend()
            << std::endl;
  if (require_cuda &&
    std::find(backends.begin(), backends.end(), "cuda") == backends.end())
  {
    return 2;
  }
  return 0;
}
