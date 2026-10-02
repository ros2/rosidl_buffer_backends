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

#ifndef TEST_MODELS_HPP_
#define TEST_MODELS_HPP_

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <vector>

namespace test_models
{

inline Ort::Model single_node_model(Ort::Node node, const std::vector<int64_t> & shape)
{
  Ort::TensorTypeAndShapeInfo tensor_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape);
  auto type_info = Ort::TypeInfo::CreateTensorInfo(tensor_info.GetConst());
  std::vector<Ort::ValueInfo> inputs;
  inputs.emplace_back("input", type_info.GetConst());
  std::vector<Ort::ValueInfo> outputs;
  outputs.emplace_back("output", type_info.GetConst());

  Ort::Graph graph;
  graph.SetInputs(inputs);
  graph.SetOutputs(outputs);
  graph.AddNode(node);

  Ort::Model model({{"", 18}});
  model.AddGraph(graph);
  return model;
}

inline Ort::Model identity_model(const std::vector<int64_t> & shape)
{
  return single_node_model(
    Ort::Node("Identity", "", "identity", {"input"}, {"output"}), shape);
}

inline Ort::Model matmul_model()
{
  return single_node_model(
    Ort::Node("MatMul", "", "matmul", {"input", "input"}, {"output"}), {2, 2});
}

}  // namespace test_models

#endif  // TEST_MODELS_HPP_
