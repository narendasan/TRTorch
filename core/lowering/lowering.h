#pragma once
#include <memory>
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace lowering {

struct LoweringInfo {
    bool input_is_dynamic;
};

void LowerBlock(torch::jit::Block* b, const LoweringInfo& info);
void LowerGraph(std::shared_ptr<torch::jit::Graph>& g, const LoweringInfo& info);
torch::jit::Module LowerModule(const torch::jit::script::Module& mod, const LoweringInfo& info);
std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<at::Tensor>> Lower(const torch::jit::script::Module& mod,
                                                                             std::string method_name,
                                                                             const LoweringInfo& info);

} // namespace lowering
} // namespace core
} // namespace trtorch
