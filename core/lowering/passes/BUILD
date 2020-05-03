package(default_visibility = ["//visibility:public"])

cc_library(
    name = "passes",
    hdrs = [
        "passes.h",
    ],
    srcs = [
        "exception_elimination.cpp",
        "expand_log_softmax.cpp",
        "fuse_flatten_linear.cpp",
        "switch_to_dynamic_shape_calculation_ops.cpp",
        "remove_dropout.cpp",
        "unpack_batch_norm.cpp",
        "unpack_addmm.cpp"
    ],
    deps = [
        "//core/util:prelude",
        "@libtorch//:libtorch",
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/lowering/irfusers/",
    srcs = ["irfusers.h"],
)

