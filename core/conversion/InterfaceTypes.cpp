#include <set>

#include "core/util/prelude.h"
#include "core/conversion/conversion.h"

namespace trtorch {
namespace core {
namespace conversion {

GraphParams get_named_params(c10::ArrayRef<torch::jit::Value*> inputs,
                             std::vector<at::Tensor> params) {
    GraphParams named_params;
    auto type_lut = torch::jit::script::string_to_type_lut();
    auto param_it = params.begin();
    for (auto in : inputs) {
        if (in->type() != type_lut["Tensor"]                            \
            && in->isCompleteTensor() && param_it != params.end()) {
            named_params[in] = *param_it;
            ++param_it;
        }
    }
    //ASSERT(named_params.size() == params.size);
    if (named_params.size() != params.size()) {
        LOG_ERROR("Graph parameter parsing failed");
    }
    return std::move(named_params);
}

InputRange::InputRange(std::vector<int64_t> d) {
    if (d.size() > 5) {
        LOG_WARNING("Verify that this dim size is accepted");
    }

    opt = util::toDims(d);
    min = util::toDims(d);
    max = util::toDims(d);
    input_shape = util::toDims(d);

}


InputRange::InputRange(std::vector<int64_t> min_shape, std::vector<int64_t> opt_shape, std::vector<int64_t> max_shape) {
    if (min_shape.size() > 5 || opt_shape.size() > 5 || max_shape.size() > 5) {
        LOG_WARNING("Verify that this dim size is accepted");
    }

    std::set<size_t> sizes;
    sizes.insert(min_shape.size());
    sizes.insert(opt_shape.size());
    sizes.insert(max_shape.size());

    if (sizes.size() != 1) {
        LOG_ERROR("Expected all input sizes have the same dimensions, but found dimensions: min(" \
                  << min_shape.size() << "), opt("
                  << opt_shape.size() << "), max("
                  << max_shape.size() << ")");
    }

    min = util::toDimsPad(min_shape, 4);
    opt = util::toDimsPad(opt_shape, 4);
    max = util::toDimsPad(max_shape, 4);

    std::vector<int64_t> dyn_shape;
    for (size_t i = 0; i < opt_shape.size(); i++) {
        std::set<uint64_t> dim;
        dim.insert(min_shape[i]);
        dim.insert(opt_shape[i]);
        dim.insert(max_shape[i]);
        if (dim.size() != 1) {
            dyn_shape.push_back(-1);
            input_is_dynamic = true;
        } else {
            dyn_shape.push_back(opt_shape[i]);
        }
    }

    input_shape = util::toDimsPad(dyn_shape, 4);

}

} // namespace conversion
} // namespace core
} // namespace trtorch

