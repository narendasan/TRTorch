import os, sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import torch
import trtorch._trtorch_internal as _trtorch_internal

def spec_to_list(spec):
    print(spec)
    if len(list(spec)) == 3:
        return [spec["min"], spec["opt"], spec["max"]]
    elif len(list(spec)) == 1:
        return [spec["opt"]]

def compile_graph(module, input_sizes):
    input_size_vec = [spec_to_list(in_shape_range) for in_shape_range in input_sizes]
    print(dir(module))
    print(module._c)
    return _trtorch_internal._compile_graph(module._c, input_size_vec)

def convert_graph_to_trt_engine(module, method, input_sizes):
    input_size_vec = [spec_to_list(in_shape_range) for in_shape_range in input_sizes]
    return _trtorch_internal._convert_graph_to_trt_engine(module, method, input_size_vec)

def dump_build_info():
    _trtorch_internal._dump_build_info()

def get_build_info():
    return _trtorch_internal._get_build_info()

def test(mod):
    return _trtorch_internal._test(mod._c)
