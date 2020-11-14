#include "module_test.h"

TEST_P(ModuleTests, CompiledModuleIsClose) {
  std::vector<torch::jit::IValue> jit_inputs_ivalues;
  std::vector<torch::jit::IValue> trt_inputs_ivalues;
  for (auto in_shape : input_shapes) {
    auto in = at::randint(5, in_shape, {at::kCUDA}).to(torch::kF16);
    jit_inputs_ivalues.push_back(in.clone());
    trt_inputs_ivalues.push_back(in.clone());
  }

  auto compile_spec = trtorch::CompileSpec(input_shapes);
  compile_spec.op_precision = torch::kF16;
  compile_spec.device.device_type = trtorch::CompileSpec::DeviceType::kDLA;
  compile_spec.device.dla_core = 1;
  compile_spec.device.allow_gpu_fallback = true;
  compile_spec.workspace_size = 1 << 28;

  auto trt_mod = trtorch::CompileGraph(mod, compile_spec);
  torch::jit::IValue trt_results_ivalues = trtorch::tests::util::RunModuleForward(trt_mod, trt_inputs_ivalues);
  std::vector<at::Tensor> trt_results;
  trt_results.push_back(trt_results_ivalues.toTensor());
 
  mod.to(torch::kHalf);
  for (auto layer : mod.named_modules()) {
    if (layer.name.find(".bn") != std::string::npos) {
      layer.value.to(torch::kFloat);
    }
  }

  torch::jit::IValue jit_results_ivalues = trtorch::tests::util::RunModuleForward(mod, jit_inputs_ivalues);
  std::vector<at::Tensor> jit_results;
  jit_results.push_back(jit_results_ivalues.toTensor());


  for (size_t i = 0; i < trt_results.size(); i++) {
    ASSERT_TRUE(trtorch::tests::util::almostEqual(jit_results[i], trt_results[i].reshape_as(jit_results[i]), 2e-5));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CompiledModuleForwardIsCloseSuite,
    ModuleTests,
    testing::Values(
        PathAndInSize({"tests/modules/resnet18_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet50_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/mobilenet_v2_traced.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet18_scripted.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/resnet50_scripted.jit.pt", {{1, 3, 224, 224}}}),
        PathAndInSize({"tests/modules/mobilenet_v2_scripted.jit.pt", {{1, 3, 224, 224}}})));