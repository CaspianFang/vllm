import contextlib
import itertools
import io
import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import List, Set

from packaging.version import parse, Version
import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

MAIN_CUDA_VERSION = "12.1"

# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = ["-O2", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")


def glob(pattern: str):
    root = Path(__name__).parent
    return [str(p) for p in root.glob(pattern)]


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def generate_flashinfer_cu() -> list[str]:
    root = Path(__name__).parent

    page_sizes = os.environ.get("PUNICA_PAGE_SIZES", "16").split(",")
    group_sizes = os.environ.get("PUNICA_GROUP_SIZES", "1,2,4,8").split(",")
    head_dims = os.environ.get("PUNICA_HEAD_DIMS", "128").split(",")
    page_sizes = [int(x) for x in page_sizes]
    group_sizes = [int(x) for x in group_sizes]
    head_dims = [int(x) for x in head_dims]
    dtypes = {"fp16": "nv_half", "bf16": "nv_bfloat16"}
    funcs = ["prefill", "decode"]
    prefix = "csrc/punica/flashinfer_adapter/generated"
    (root / prefix).mkdir(parents=True, exist_ok=True)
    files = []

    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    if not path.exists():
        with open(root / prefix / "dispatch.inc", "w") as f:
            f.write("#define _DISPATCH_CASES_page_size(...)       \\\n")
            for x in page_sizes:
                f.write(f"  _DISPATCH_CASE({x}, PAGE_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_group_size(...)      \\\n")
            for x in group_sizes:
                f.write(f"  _DISPATCH_CASE({x}, GROUP_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_head_dim(...)        \\\n")
            for x in head_dims:
                f.write(f"  _DISPATCH_CASE({x}, HEAD_DIM, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("\n")

    # impl
    for func, page_size, group_size, head_dim, dtype in itertools.product(
        funcs, page_sizes, group_sizes, head_dims, dtypes
    ):
        fname = f"batch_{func}_p{page_size}_g{group_size}_h{head_dim}_{dtype}.cu"
        files.append(prefix + "/" + fname)
        if (root / prefix / fname).exists():
            continue
        with open(root / prefix / fname, "w") as f:
            f.write('#include "../flashinfer_decl.h"\n\n')
            f.write(f'#include "flashinfer/{func}.cuh"\n\n')
            f.write(
                f"INST_Batch{func.capitalize()}({dtypes[dtype]}, {page_size}, {group_size}, {head_dim})\n"
            )

    return files


def get_torch_arch_list() -> Set[str]:
    # TORCH_CUDA_ARCH_LIST can have one or more architectures,
    # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
    # compiler to additionally include PTX code that can be runtime-compiled
    # and executed on the 8.6 or newer architectures. While the PTX code will
    # not give the best performance on the newer architectures, it provides
    # forward compatibility.
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # List are separated by ; or space.
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}.")
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}.",
            stacklevel=2)
    return arch_list


# First, check the TORCH_CUDA_ARCH_LIST environment variable.
compute_capabilities = get_torch_arch_list()
if not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability below 7.0 are not supported.")
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    # If no GPU is specified nor available, add all supported architectures
    # based on the NVCC CUDA version.
    compute_capabilities = SUPPORTED_ARCHS.copy()
    if nvcc_cuda_version < Version("11.1"):
        compute_capabilities.remove("8.6")
    if nvcc_cuda_version < Version("11.8"):
        compute_capabilities.remove("8.9")
        compute_capabilities.remove("9.0")

# Validate the NVCC CUDA version.
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
if (nvcc_cuda_version < Version("11.1")
        and any(cc.startswith("8.6") for cc in compute_capabilities)):
    raise RuntimeError(
        "CUDA 11.1 or higher is required for compute capability 8.6.")
if nvcc_cuda_version < Version("11.8"):
    if any(cc.startswith("8.9") for cc in compute_capabilities):
        # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
        # However, GPUs with compute capability 8.9 can also run the code generated by
        # the previous versions of CUDA 11 and targeting compute capability 8.0.
        # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
        # instead of 8.9.
        warnings.warn(
            "CUDA 11.8 or higher is required for compute capability 8.9. "
            "Targeting compute capability 8.0 instead.",
            stacklevel=2)
        compute_capabilities = set(cc for cc in compute_capabilities
                                   if not cc.startswith("8.9"))
        compute_capabilities.add("8.0+PTX")
    if any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 11.8 or higher is required for compute capability 9.0.")

# Use NVCC threads to parallelize the build.
if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 8)
    NVCC_FLAGS += ["--threads", str(num_threads)]

NVCC_FLAGS_PUNICA = NVCC_FLAGS.copy()

# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    num = capability[0] + capability[2]
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]
    if int(capability[0]) >= 8:
        NVCC_FLAGS_PUNICA += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS_PUNICA += [
                "-gencode", f"arch=compute_{num},code=compute_{num}"
            ]

# changes for punica kernels
NVCC_FLAGS += torch_cpp_ext.COMMON_NVCC_FLAGS
REMOVE_NVCC_FLAGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]
for flag in REMOVE_NVCC_FLAGS:
    with contextlib.suppress(ValueError):
        torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)

ext_modules = []

install_punica = bool(int(os.getenv("VLLM_INSTALL_PUNICA_KERNELS", "1")))
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 8:
        install_punica = False
        break
if install_punica:
    ext_modules.append(
        CUDAExtension(
            name="vllm._punica_C",
            sources=["csrc/punica/punica_ops.cc",
                     "csrc/punica/flashinfer_adapter/flashinfer_all.cu",
                     "csrc/punica/rms_norm/rms_norm_cutlass.cu",
                     "csrc/punica/sgmv/sgmv_cutlass.cu",
                     "csrc/punica/sgmv_flashinfer/sgmv_all.cu", 
            ] 
            + glob("csrc/punica/bgmv/*.cu") + generate_flashinfer_cu(),
            include_dirs=[
                str(Path(__name__).parent.resolve() / "third_party/cutlass/include"),
                str(Path(__name__).parent.resolve() / "third_party/flashinfer/include"),
                # str(Path(__name__).parent.resolve() / "third_party/nvbench/include"),
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS_PUNICA,
            },
        ))

vllm_extension = CUDAExtension(
    name="vllm._C",
    sources=[
        "csrc/cache_kernels.cu",
        "csrc/attention/attention_kernels.cu",
        "csrc/pos_encoding_kernels.cu",
        "csrc/activation_kernels.cu",
        "csrc/layernorm_kernels.cu",
        "csrc/quantization/awq/gemm_kernels.cu",
        "csrc/quantization/squeezellm/quant_cuda_kernel.cu",
        "csrc/cuda_utils_kernels.cu",
        "csrc/pybind.cpp",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(vllm_extension)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_vllm_version() -> str:
    version = find_version(get_path("vllm", "__init__.py"))
    cuda_version = str(nvcc_cuda_version)
    if cuda_version != MAIN_CUDA_VERSION:
        cuda_version_str = cuda_version.replace(".", "")[:3]
        version += f"+cu{cuda_version_str}"
    return version


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="vllm",
    version=get_vllm_version(),
    author="vLLM Team",
    license="Apache 2.0",
    description=("A high-throughput and memory-efficient inference and "
                 "serving engine for LLMs"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm",
        "Documentation": "https://vllm.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=("benchmarks", "csrc", "docs",
                                               "examples", "tests")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    package_data={"vllm": ["py.typed"]},
)
