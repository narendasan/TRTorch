#include "torch/csrc/jit/passes/subgraph_rewrite.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void SwitchToDynamicShapeAdaptiveAvgPool2D(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string aap_pattern = R"IR(
    graph(%0, %1):
      %2: Tensor = aten::adaptive_avg_pool2d(%0, %1)
      return (%out))IR";
  std::string dyn_aap_pattern = R"IR(
    graph(%0, %1):
      %2: Tensor = trt::adaptive_avg_pool2d_rt(%0, %1)
      return (%out))IR";;


  torch::jit::SubgraphRewriter switch_to_dyn_aap;
  switch_to_dyn_aap.RegisterRewritePattern(aap_pattern, dyn_aap_pattern);
  switch_to_dyn_aap.runOnGraph(graph);
}

void SwitchToDynamicOps(std::shared_ptr<torch::jit::Graph>& graph) {
    SwitchToDynamicShapeAdaptiveAvgPool2D(graph);
}


} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
