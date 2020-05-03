#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto pooling_registrations = RegisterNodeConversionPatterns()
  .pattern({
    "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], int[2] dilation=[1, 1], bool ceil_mode=False) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
      auto shape = util::toVec(in->getDimensions());

      // Max Pool needs at least 4D input
      if (shape.size() < 4) {
        auto new_shape = util::toDimsPad(shape, 4);
        LOG_DEBUG("Input shape is less than 4D got: " << util::toDims(shape) << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
        auto shuffle = ctx->net->addShuffle(*in);
        shuffle->setReshapeDimensions(new_shape);
        shuffle->setName((util::node_info(n) + " [Reshape to " + util::toStr(new_shape) + ']').c_str());
        in = shuffle->getOutput(0);
      }


      auto kernel_size = util::toDimsHW(args[1].unwrapToIntList());
      LOG_DEBUG("kernel_size: " << kernel_size);
      auto padding = util::toDimsHW(args[3].unwrapToIntList());
      LOG_DEBUG("padding: " << padding);
      auto dilation = util::toDims(args[4].unwrapToIntList());

      TRTORCH_ASSERT(dilation == util::toDims(std::vector<int64_t>({1,1})), "Pooling dilation is not supported in TensorRT");

      LOG_DEBUG("dilation: " << dilation);
      LOG_WARNING("Dilation not used in max pooling converter");
      bool ceil_mode = args[5].IValue()->to<bool>();

      auto new_layer = ctx->net->addPoolingNd(*in, nvinfer1::PoolingType::kMAX, kernel_size);
      TRTORCH_CHECK(new_layer, "Unable to create Max Pool 2D layer from node: " << *n);

      new_layer->setName(util::node_info(n).c_str());
      new_layer->setPaddingNd(padding);
      if (args[2].unwrapToIntList().size() == 2) {
        auto stride = util::toDims(args[2].unwrapToIntList());
        new_layer->setStrideNd(stride);
      }

      auto padding_mode = ceil_mode ? nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP :  nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
      new_layer->setPaddingMode(padding_mode);

      new_layer->setName(util::node_info(n).c_str());
      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
      return true;
    }
  }).pattern({
    "trt::adaptive_avg_pool2d_rt(Tensor self, int[2] output_size) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
      auto ct_in_shape = util::toVec(in->getDimensions());
      auto get_in_shape = ctx->net->addShape(*in);
      auto rt_in_shape = get_in_shape->getOutput(0);

      auto out_shape_arr = args[1].IValue()->toIntList();

      if (ct_in_shape.size() < 4) {
        auto new_shape = util::toDimsPad(ct_in_shape, 4);
        LOG_DEBUG("Input shape is less than 4D got: " << util::toDims(ct_in_shape) << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
        auto shuffle = ctx->net->addShuffle(*in);
        shuffle->setReshapeDimensions(new_shape);
        shuffle->setName((util::node_info(n) + " [Reshape to " + util::toStr(new_shape) + ']').c_str());
        in = shuffle->getOutput(0);
        ct_in_shape = util::toVec(in->getDimensions());
      }

      auto feat_shape = ctx->net->addSlice(*rt_in_shape,
                                           util::toDims(std::vector<int64_t>{2}),
                                           util::toDims(std::vector<int64_t>{2}),
                                           util::toDims(std::vector<int64_t>{1}))->getOutput(0);
      auto out_shape = list_to_const(ctx, out_shape_arr);

      auto stride = ctx->net->addElementWise(*feat_shape, *out_shape, nvinfer1::ElementWiseOperation::kFLOOR_DIV)->getOutput(0);

      auto minus_one = Weights(ctx, int64_t(-1));
      auto one = Weights(ctx, int64_t(1));

      auto window_se1 = ctx->net->addScale(*out_shape, nvinfer1::ScaleMode::kUNIFORM, minus_one.data, one.data, one.data)->getOutput(0);
      auto window_se2 = ctx->net->addElementWise(*window_se1, *stride, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
      auto window = ctx->net->addElementWise(*feat_shape, *window_se2, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);

      auto new_layer = ctx->net->addPoolingNd(*in, nvinfer1::PoolingType::kAVERAGE, *window);
      TRTORCH_CHECK(new_layer, "Unable to create average pooling layer from node: " << *n);

      new_layer->setStrideNd(stride);

      new_layer->setName(util::node_info(n).c_str());
      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
      return true;
    }
  }).pattern({
    "aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
      auto in_shape = util::toVec(in->getDimensions());

      if (in_shape.size() < 4) {
        auto new_shape = util::toDimsPad(in_shape, 4);
        LOG_DEBUG("Input shape is less than 4D got: " << util::toDims(in_shape) << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
        auto shuffle = ctx->net->addShuffle(*in);
        shuffle->setReshapeDimensions(new_shape);
        shuffle->setName((util::node_info(n) + " [Reshape to " + util::toStr(new_shape) + ']').c_str());
        in = shuffle->getOutput(0);
        in_shape = util::toVec(in->getDimensions());
      }

      auto out_shape = args[1].IValue()->toIntList();

      std::vector<int64_t> stride(out_shape.size());
      for (size_t i = 0; i < out_shape.size(); i++) {
        stride[(stride.size() - 1) - i] = in_shape[(in_shape.size() - 1) - i] / out_shape[(out_shape.size() - 1) - i];
      }
      LOG_DEBUG("Stride: " << util::toDims(stride));

      std::vector<int64_t> window(out_shape.size());
      for (size_t i = 0; i < out_shape.size(); i++) {
        window[window.size() - 1 - i] = in_shape[in_shape.size() - 1 - i] - (out_shape[out_shape.size() - 1 - i] - 1) * stride[stride.size() - 1 - i];
      }

      LOG_DEBUG("Window: " << util::toDims(window));

      auto new_layer = ctx->net->addPoolingNd(*in, nvinfer1::PoolingType::kAVERAGE, util::toDims(window));
      TRTORCH_CHECK(new_layer, "Unable to create average pooling layer from node: " << *n);

      new_layer->setStrideNd(util::toDims(stride));

      new_layer->setName(util::node_info(n).c_str());
      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
      return true;
    }
  });










            auto in = args[0].ITensor();
            auto in_shape = util::toVec(in->getDimensions());



            auto out_shape = args[1].IValue()->toIntList();

            std::vector<int64_t> stride(out_shape.size());
            for (size_t i = 0; i < out_shape.size(); i++) {
                stride[(stride.size() - 1) - i] = in_shape[(in_shape.size() - 1) - i] / out_shape[(out_shape.size() - 1) - i];
            }
            LOG_DEBUG("Stride: " << util::toDims(stride));

            std::vector<int64_t> window(out_shape.size());
            for (size_t i = 0; i < out_shape.size(); i++) {
                window[window.size() - 1 - i] = in_shape[in_shape.size() - 1 - i] - (out_shape[out_shape.size() - 1 - i] - 1) * stride[stride.size() - 1 - i];
            }

            LOG_DEBUG("Window: " << util::toDims(window));

            auto new_layer = ctx->net->addPoolingNd(*in, nvinfer1::PoolingType::kAVERAGE, util::toDims(window));
            TRTORCH_CHECK(new_layer, "Unable to create average pooling layer from node: " << *n);

            new_layer->setStrideNd(util::toDims(stride));

            new_layer->setName(util::node_info(n).c_str());
            auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

            LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            return true;
        }
    });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // trtorch
