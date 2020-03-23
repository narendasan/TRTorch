#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "trtorch/trtorch.h"
#include "torch/script.h"
#include "torch/csrc/jit/pybind_utils.h"
#include "Python.h"

namespace py = pybind11;

namespace trtorch {
namespace pyapi {

torch::jit::script::Module test(const torch::jit::script::Module mod) {
    std::cout << "hello from C++" << std::endl;
    return mod;
}

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& mod, std::vector<std::vector<int64_t>> dims) {
    py::gil_scoped_acquire gil;
    std::cout << "hello from C++" << std::endl;
    mod.dump(true, true, true);
    auto trt_mod = trtorch::CompileGraph(mod, dims);
    return std::move(trt_mod);
}

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod, std::string method_name, std::vector<std::vector<int64_t>> dims) {
    py::gil_scoped_acquire gil;
    mod.dump(true, true, true);
    auto trt_engine = trtorch::ConvertGraphToTRTEngine(mod, method_name, dims);
    return trt_engine;
}

PYBIND11_MODULE(_trtorch_internal, m) {
    py::module::import("torch");
    m.doc() = "TRTorch: Ahead of Time compilation for PyTorch JIT. A tool to convert PyTorch JIT to TensorRT";
    m.def("_compile_graph", &trtorch::pyapi::CompileGraph, "Ingest a PyTorch JIT module and convert supported subgraphs to TensorRT engines, returns a JIT module with the engines embedded");
    m.def("_convert_graph_to_trt_engine", &trtorch::pyapi::ConvertGraphToTRTEngine, "Given a PyTorch JIT Module, convert forward into a TensorRT engine and return a serialized engine");
    m.def("_dump_build_info", &trtorch::dump_build_info, "Print build info about the compiler to console");
    m.def("_get_build_info", &trtorch::get_build_info, "Returns build info about the compiler as a string");
    m.def("_test", &trtorch::pyapi::test);
}

// namespace logging {
// PYBIND11_MODULE(logging, m) {
//     m.attr("__name__") = "trtorch.logging";
//     m.def("get_logging_prefix", &trtorch::logging::get_logging_prefix, "Get the current prefix for the logging output");
//     m.def("set_logging_prefix", &trtorch::logging::set_logging_prefix, "Set the logging prefix for logging output");
//     m.def("get_reportable_log_level", &trtorch::logging::get_reportable_log_level, "Get the current log level");
//     m.def("set_reportable_log_level", &trtorch::logging::set_reportable_log_level, "Set the level required to be met for a log message to be printed");
//     m.def("get_is_colored_output_on", &trtorch::logging::get_is_colored_output_on, "Get if the logging output will be colored");
//     m.def("set_is_colored_output_on", &trtorch::logging::set_is_colored_output_on, "Set if the logging output should be colored");
//     m.def("log", &trtorch::logging::log, "Add a message to the logger");
//     py::enum_<trtorch::logging::Level>(m, "Level", py::arithmetic())
//         .value("INTERNAL_ERROR", trtorch::logging::Level::kINTERNAL_ERROR)
//         .value("ERROR", trtorch::logging::Level::kERROR)
//         .value("WARNING", trtorch::logging::Level::kWARNING)
//         .value("INFO", trtorch::logging::Level::kINFO)
//         .value("DEBUG", trtorch::logging::Level::kDEBUG)
//         .export_values();
// }
//} // namespace logging 
} // namespace py
} // namespace trtorch
