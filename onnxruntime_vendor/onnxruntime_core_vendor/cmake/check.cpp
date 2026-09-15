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

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <iostream>
#if ORT_API_VERSION < 27
#error "Require ONNX Runtime headers with C API >=27"
#endif
int main()
{
  const char * version = OrtGetApiBase()->GetVersionString();
  int major = 0;
  int minor = 0;
  int patch = 0;
  char extra = 0;
  if (std::sscanf(version, "%d.%d.%d%c", &major, &minor, &patch, &extra) != 3 ||
    major < 1 || (major == 1 && minor < 27) || minor < 0 || patch < 0)
  {
    std::cerr << "Rejected ONNX Runtime " << version << ": require >=1.27.0\n";
    return 1;
  }
  // ONNX Runtime 1.x uses the release minor as its C API version.
  if (major == 1 && minor != ORT_API_VERSION) {
    std::cerr << "Rejected ONNX Runtime " << version << ": headers use C API " <<
      ORT_API_VERSION << " and must match the runtime minor\n";
    return 1;
  }
  const OrtApi * api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!api) {
    std::cerr << "ONNX Runtime does not provide the headers' C API\n";
    return 1;
  }
  Ort::InitApi(api);
  Ort::Env env;
  const int64_t shape[] = {2, 3};
  float storage[6] = {1, 2, 3, 4, 5, 6};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(memory, storage, 6, shape, 2);
  if (tensor.GetTensorTypeAndShapeInfo().GetElementCount() != 6 ||
    tensor.GetTensorMutableData<float>() != storage)
  {
    return 1;
  }
  std::cout << "ONNXRUNTIME_VERSION=" << version << "\n";
  return 0;
}
