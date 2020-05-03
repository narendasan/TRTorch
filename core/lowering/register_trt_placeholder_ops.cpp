#include "torch/csrc/jit/runtime/custom_operator.h"

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

RegisterOperators trt_const_op_reg({
  /// Op marks a Tensor to be conveted from an Torch Tensor
  /// to a TRT constant Tensor
  Operator(
    "trt::const(Tensor val) -> Tensor",
    [](Stack& stack) {
      return 0; //noop
    },
    aliasAnalysisFromSchema()),
  /// Op indicates to use the dynamic shape calculation variant of
  /// adaptive_avg_pool2d
  Operator(
    "trt::adaptive_avg_pool2d_rt(Tensor self, int[2] output_size) -> (Tensor)",
    [](Stack& stack) {
      return 0; //noop
    },
    aliasAnalysisFromSchema()),
  });


} // namespace jit
} // namespace torch