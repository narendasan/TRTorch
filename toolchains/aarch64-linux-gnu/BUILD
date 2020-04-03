package(default_visibility = ["//visibility:public"])
load("//toolchains/aarch64-linux-gnu:aarch64_linux_gnu_cc_toolchain_config.bzl", "aarch64_linux_gnu_cc_toolchain_config")

config_setting(
    name = "gcc_7.5,
    values = {"define": "platforms=jetpach_4.3+linux"}
)

aarch64_linux_gnu_cc_toolchain_config(
    name = "aarch64_linux_gnu_toolchain_config",
    cpu = "aarch64",
    gcc_toolchain_version = select({
        "//conditions:default": "gcc_7.5",
    })
)

cc_toolchain_suite(
    name = "aarch64-unknown-linux-gnu",
    toolchains = {
        "aarch64": ":gcc-linux-aarch64",
    },
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain(
    name = "gcc-linux-aarch64",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = "empty",
    strip_files = "empty",
    supports_param_files = 1,
    toolchain_identifier = "aarch64-unknown-linux-gnu",
    toolchain_config = ":aarch64_linux_gnu_toolchain_config"
)

filegroup(
    name = "gcc",
    srcs = [
        "usr/bin/aarch64-linux-gnu-gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "usr/bin/aarch64-linux-gnu-ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "usr/bin/aarch64-linux-gnu-ld.gold",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "usr/bin/aarch64-linux-gnu-nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "usr/bin/aarch64-linux-gnu-objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "usr/bin/aarch64-linux-gnu-objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "usr/bin/aarch64-linux-gnu-strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "usr/bin/aarch64-linux-gnu-as",
    ],
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "usr/lib/gcc-cross/aarch64-linux-gnu/7/**",
        "usr/aarch64-linux-gnu/**",
    ]),
)

filegroup(
    name = "compiler_components",
    srcs = [
        ":ar",
        ":as",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)