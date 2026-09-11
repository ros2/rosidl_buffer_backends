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

#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "torch_conversions/torch_conversions.hpp"

int main(int argc, char ** argv)
{
  const bool require_cuda =
    argc == 2 && std::string(argv[1]) == "--require-cuda";
  const auto backends = torch_conversions::available_backends();

  std::cout << "plugins=";
  for (size_t index = 0; index < backends.size(); ++index) {
    std::cout << (index == 0 ? "" : ",") << backends[index];
  }
  std::cout << std::endl;

  if (!torch_conversions::backend_available("cpu")) {
    throw std::runtime_error("CPU plugin was not discovered");
  }

  auto default_msg = torch_conversions::allocate_tensor_msg(
    {1}, at::kFloat);
  auto default_output =
    torch_conversions::from_output_tensor_msg(*default_msg);
  default_output.fill_(5.0);
  const auto default_result =
    torch_conversions::from_input_tensor_msg(*default_msg);
  if (default_result.cpu().item<float>() != 5.0f) {
    throw std::runtime_error("Default plugin round-trip failed");
  }
  std::cout << "default=" << default_msg->data.get_backend_type() << std::endl;

  const auto cpu_source = torch::arange(12, torch::kFloat).reshape({3, 4});
  auto cpu_msg = torch_conversions::to_tensor_msg(cpu_source);
  const auto cpu_result =
    torch_conversions::from_input_tensor_msg(*cpu_msg);
  if (!torch::equal(cpu_source, cpu_result)) {
    throw std::runtime_error("CPU plugin round-trip failed");
  }
  std::cout << "cpu=pass" << std::endl;

  const bool has_cuda =
    torch_conversions::backend_available("cuda");
  if (require_cuda && !has_cuda) {
    throw std::runtime_error("CUDA plugin was required but not discovered");
  }
  if (has_cuda) {
    const auto cuda_source = cpu_source.to(torch::kCUDA);
    auto cuda_msg = torch_conversions::to_tensor_msg(cuda_source);
    const auto cuda_result =
      torch_conversions::from_input_tensor_msg(*cuda_msg);
    if (!torch::equal(cpu_source, cuda_result.cpu())) {
      throw std::runtime_error("CUDA plugin round-trip failed");
    }
    std::cout << "cuda=pass" << std::endl;

    if (setenv("ROSIDL_TENSOR_BACKEND", "cpu", 1) != 0) {
      throw std::runtime_error("Failed to select the CPU plugin");
    }
    auto switched_cpu = torch_conversions::allocate_tensor_msg(
      {1}, at::kFloat);
    if (setenv("ROSIDL_TENSOR_BACKEND", "cuda", 1) != 0) {
      throw std::runtime_error("Failed to select the CUDA plugin");
    }
    auto switched_cuda = torch_conversions::allocate_tensor_msg(
      {1}, at::kFloat);
    unsetenv("ROSIDL_TENSOR_BACKEND");
    if (switched_cpu->data.get_backend_type() != "cpu" ||
      switched_cuda->data.get_backend_type() != "cuda")
    {
      throw std::runtime_error("Runtime plugin switching failed");
    }
    std::cout << "runtime-switch=pass" << std::endl;
  } else {
    std::cout << "cuda=not-installed" << std::endl;
  }
  return 0;
}
