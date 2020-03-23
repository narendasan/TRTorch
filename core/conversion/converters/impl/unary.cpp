#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

#define convert(unary, trt_type)                                               \
  auto unary##_registrations TRTORCH_UNUSED =                                  \
      RegisterNodeConversionPatterns().pattern(                                \
          {"aten::" #unary "(Tensor self) -> Tensor",                          \
           [](ConversionCtx *ctx, const torch::jit::Node *n,                   \
              args &args) -> bool {                                            \
             auto in = args[0].ITensor();                                      \
             auto unary =                                                      \
                 ctx->net->addUnary(*in, nvinfer1::UnaryOperation::trt_type);  \
                                                                               \
             TRTORCH_CHECK(                                                    \
                 unary,                                                        \
                 "Unable to create " #unary " layer from node: " << *n);       \
                                                                               \
             unary->setName(util::node_info(n).c_str());                       \
             auto out_value = n->outputs()[0];                                 \
             auto out_tensor = unary->getOutput(0);                            \
             out_tensor->setName(out_value->debugName().c_str());              \
             ctx->value_tensor_map[out_value] = out_tensor;                    \
             LOG_DEBUG(                                                        \
                 "Output tensor shape: " << out_tensor->getDimensions());      \
                                                                               \
             return true;                                                      \
           }});

convert(cos, kCOS);
convert(acos, kACOS);
convert(cosh, kCOSH);
convert(sin, kSIN);
convert(asin, kASIN);
convert(sinh, kSINH);
convert(tan, kTAN);
convert(atan, kATAN);
convert(abs, kABS);
convert(floor, kFLOOR);
convert(reciprocal, kRECIP);
convert(log, kLOG);
convert(ceil, kCEIL);
convert(sqrt, kSQRT);
convert(exp, kEXP);
convert(neg, kNEG);

#undef convert

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
