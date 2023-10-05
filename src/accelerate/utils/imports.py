# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.metadata
import os
import warnings
from functools import lru_cache

import torch
from packaging import version
from packaging.version import parse

from .environment import parse_flag_from_env, str_to_bool
from .versions import compare_versions, is_torch_version


try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    _tpu_available = True
except ImportError:
    _tpu_available = False


# Cache this result has it's a C FFI call which can be pretty time-consuming
_torch_distributed_available = torch.distributed.is_available()


def _is_package_available(pkg_name):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            _ = importlib.metadata.metadata(pkg_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False


def is_torch_distributed_available() -> bool:
    return _torch_distributed_available


def is_ccl_available():
    try:
        pass
    except ImportError:
        print(
            "Intel(R) oneCCL Bindings for PyTorch* is required to run DDP on Intel(R) GPUs, but it is not"
            " detected. If you see \"ValueError: Invalid backend: 'ccl'\" error, please install Intel(R) oneCCL"
            " Bindings for PyTorch*."
        )
    return (
        importlib.util.find_spec("torch_ccl") is not None
        or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
    )


def get_ccl_version():
    return importlib.metadata.version("oneccl_bind_pt")


def is_fp8_available():
    return _is_package_available("transformer_engine")


def is_cuda_available():
    """
    Checks if `cuda` is available via an `nvml-based` check which won't trigger the drivers and leave cuda
    uninitialized.
    """
    try:
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = str(1)
        available = torch.cuda.is_available()
    finally:
        os.environ.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    return available


@lru_cache
def is_tpu_available(check_device=True):
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    # Due to bugs on the amp series GPUs, we disable torch-xla on them
    if is_cuda_available():
        return False
    if check_device:
        if _tpu_available:
            try:
                # Will raise a RuntimeError if no XLA configuration is found
                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
    return _tpu_available


def is_deepspeed_available():
    return _is_package_available("deepspeed")


def is_bf16_available(ignore_tpu=False):
    "Checks if bf16 is supported, optionally ignoring the TPU"
    if is_tpu_available():
        return not ignore_tpu
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return True


def is_4bit_bnb_available():
    package_exists = _is_package_available("bitsandbytes")
    if package_exists:
        bnb_version = version.parse(importlib.metadata.version("bitsandbytes"))
        return compare_versions(bnb_version, ">=", "0.39.0")
    return False


def is_8bit_bnb_available():
    package_exists = _is_package_available("bitsandbytes")
    if package_exists:
        bnb_version = version.parse(importlib.metadata.version("bitsandbytes"))
        return compare_versions(bnb_version, ">=", "0.37.2")
    return False


def is_bnb_available():
    return _is_package_available("bitsandbytes")


def is_megatron_lm_available():
    if str_to_bool(os.environ.get("ACCELERATE_USE_MEGATRON_LM", "False")) == 1:
        package_exists = importlib.util.find_spec("megatron") is not None
        if package_exists:
            try:
                megatron_version = parse(importlib.metadata.version("megatron-lm"))
                return compare_versions(megatron_version, ">=", "2.2.0")
            except Exception as e:
                warnings.warn(f"Parse Megatron version failed. Exception:{e}")
                return False


def is_safetensors_available():
    return _is_package_available("safetensors")


def is_transformers_available():
    return _is_package_available("transformers")


def is_datasets_available():
    return _is_package_available("datasets")


def is_timm_available():
    return _is_package_available("timm")


def is_aim_available():
    package_exists = _is_package_available("aim")
    if package_exists:
        aim_version = version.parse(importlib.metadata.version("aim"))
        return compare_versions(aim_version, "<", "4.0.0")
    return False


def is_tensorboard_available():
    return _is_package_available("tensorboard") or _is_package_available("tensorboardX")


def is_wandb_available():
    return _is_package_available("wandb")


def is_comet_ml_available():
    return _is_package_available("comet_ml")


def is_boto3_available():
    return _is_package_available("boto3")


def is_rich_available():
    if _is_package_available("rich"):
        if "ACCELERATE_DISABLE_RICH" in os.environ:
            warnings.warn(
                "`ACCELERATE_DISABLE_RICH` is deprecated and will be removed in v0.22.0 and deactivated by default. Please use `ACCELERATE_ENABLE_RICH` if you wish to use `rich`."
            )
            return not parse_flag_from_env("ACCELERATE_DISABLE_RICH", False)
        return parse_flag_from_env("ACCELERATE_ENABLE_RICH", False)
    return False


def is_sagemaker_available():
    return _is_package_available("sagemaker")


def is_tqdm_available():
    return _is_package_available("tqdm")


def is_clearml_available():
    return _is_package_available("clearml")


def is_mlflow_available():
    if _is_package_available("mlflow"):
        return True

    if importlib.util.find_spec("mlflow") is not None:
        try:
            _ = importlib.metadata.metadata("mlflow-skinny")
            return True
        except importlib.metadata.PackageNotFoundError:
            return False
    return False


def is_mps_available():
    return is_torch_version(">=", "1.12") and torch.backends.mps.is_available() and torch.backends.mps.is_built()


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


@lru_cache
def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
def is_xpu_available(check_device=False):
    "check if user disables it explicitly"
    if not parse_flag_from_env("ACCELERATE_USE_XPU", default=True):
        return False
    "Checks if `intel_extension_for_pytorch` is installed and potentially if a XPU is in the environment"
    if is_ipex_available():
        import torch

        if is_torch_version("<=", "1.12"):
            return False
    else:
        return False

    import intel_extension_for_pytorch  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU  is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()
