workspace(name = "TRTorch")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()
# Only needed if using the packaging rules.
load("@rules_python//python:pip.bzl", "pip_repositories", "pip_import")
pip_repositories()

http_archive(
    name = "libtorch",
    build_file = "@//third_party/libtorch:BUILD",
    strip_prefix = "libtorch",
    urls = ["https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.4.0.zip"],
    sha256 = "f214bfde532877aa5d4e0803e51a28fa8edd97b6a44b6615f75a70352b6b542e"
)

http_archive(
    name = "rules_pkg",
    url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz",
    sha256 = "352c090cc3d3f9a6b4e676cf42a6047c16824959b438895a76c2989c6d7c246a",
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

new_local_repository(
    name = "cuda",
    path = "/usr/local/cuda-10.1/targets/x86_64-linux/",
    build_file = "@//third_party/cuda:BUILD",
)



# #new_local_repository(
# #    name = "cudnn",
# #    path = "/usr/",
# #    build_file = "@//third_party/cudnn:BUILD"
# #)

http_archive(
    name = "cudnn",
    urls = ["https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-linux-x64-v7.6.5.32.tgz"],
    build_file = "@//third_party/cudnn:BUILD",
    sha256 = "7eaec8039a2c30ab0bc758d303588767693def6bf49b22485a2c00bf2e136cb3",
    strip_prefix = "cuda"
)
    

# #new_local_repository(
# #   name = "tensorrt",
# #   path = "/usr/",
# #   build_file = "@//third_party/tensorrt:BUILD"
# #)

http_archive(
    name = "tensorrt",
    urls = ["https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/6.0/GA_6.0.1.5/tars/TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz"],
    build_file = "@//third_party/tensorrt:BUILD",
    sha256 = "e20b7bd051cdd448c5690a30ba01e83b0a0855edc4012107c0af01fde5b4037a",
    strip_prefix = "TensorRT-6.0.1.5"
)

git_repository(
    name = "googletest",
    remote = "https://github.com/google/googletest",
    commit = "703bd9caab50b139428cea1aaff9974ebee5742e",
    shallow_since = "1570114335 -0400"
)
