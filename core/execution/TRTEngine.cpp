#include <algorithm>

#include "NvInfer.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

#include "core/util/prelude.h"
#include "core/execution/execution.h"

namespace trtorch {
namespace core {
namespace execution {

std::string slugify(std::string s) {
    std::replace(s.begin(), s.end(), '.', '_');
    return s;
}

TRTEngine::TRTEngine(std::string serialized_engine)
    : logger(std::string("[] - "),
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on()) {
    std::string _name = "deserialized_trt";
    new (this) TRTEngine(_name, serialized_engine);
}

TRTEngine::TRTEngine(std::string mod_name, std::string serialized_engine)
    : logger(std::string("[") + mod_name + std::string("_engine] - "),
        util::logging::get_logger().get_reportable_severity(),
        util::logging::get_logger().get_is_colored_output_on()) {

    rt = nvinfer1::createInferRuntime(logger);

    name = slugify(mod_name) + "_engine";

    cuda_engine = rt->deserializeCudaEngine(serialized_engine.c_str(), serialized_engine.size());
    // Easy way to get a unique name for each engine, maybe there is a more descriptive way (using something associated with the graph maybe)
    id = reinterpret_cast<EngineID>(cuda_engine);

    exec_ctx = cuda_engine->createExecutionContext();

    uint64_t inputs = 0;
    uint64_t outputs = 0;

    for (int64_t x = 0; x < cuda_engine->getNbBindings(); x++) {
        if(cuda_engine->bindingIsInput(x)) {
            inputs++;
        } else {
            outputs++;
        }
    }
    num_io = std::make_pair(inputs, outputs);
}

TRTEngine& TRTEngine::operator=(const TRTEngine& other) {
    id = other.id;
    rt = other.rt;
    cuda_engine = other.cuda_engine;
    exec_ctx = other.exec_ctx;
    num_io = other.num_io;
    return (*this);
}

// TODO: Implement a call method
// c10::List<at::Tensor> TRTEngine::Run(c10::List<at::Tensor> inputs) {
//     auto input_vec = inputs.vec();
//    auto output_vec = RunCudaEngine(exec_ctx, num_io, input_vec);
//
//     return c10::List<at::Tensor>(output_vec);
// }

static auto TRTORCH_UNUSED TRTEngineTSRegistrtion = torch::class_<TRTEngine>("tensorrt", "Engine")
    .def(torch::init<std::string>())
    // TODO: .def("__call__", &TRTEngine::Run)
    // TODO: .def("run", &TRTEngine::Run)
    .def_pickle(
        [](const c10::intrusive_ptr<TRTEngine>& self) -> std::string {
            auto serialized_engine = self->cuda_engine->serialize();
            return std::string((const char*)serialized_engine->data(), serialized_engine->size());
        },
        [](std::string seralized_engine) -> c10::intrusive_ptr<TRTEngine> {
            return c10::make_intrusive<TRTEngine>(std::move(seralized_engine));
        }
    );

} // namespace execution
} // namespace core
} // namespace trtorch
