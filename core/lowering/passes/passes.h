#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void FuseFlattenLinear(std::shared_ptr<torch::jit::Graph>& graph);
void ExpandLogSoftmax(std::shared_ptr<torch::jit::Graph>& graph);
void RemoveDropout(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackBatchNorm(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackAddMM(std::shared_ptr<torch::jit::Graph>& graph);
void SwitchToDynamicOps(std::shared_ptr<torch::jit::Graph>& graph);
void EliminateExceptionOrPassPattern(std::shared_ptr<torch::jit::Graph> graph);

} // namespace irfusers
} // namespace lowering
} // namespace core
} // trtorch
