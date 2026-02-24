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
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.machinery
import importlib.metadata
import importlib.util
import json
import operator
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any

import packaging.version
from packaging import version

from . import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()


def _is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str]:
    """Check if `pkg_name` exist, and optionally try to get its version"""
    spec = importlib.util.find_spec(pkg_name)
    package_exists = spec is not None
    package_version = "N/A"
    if package_exists and return_version:
        try:
            # importlib.metadata works with the distribution package, which may be different from the import
            # name (e.g. `PIL` is the import name, but `pillow` is the distribution name)
            distributions = PACKAGE_DISTRIBUTION_MAPPING[pkg_name]
            # Per PEP 503, underscores and hyphens are equivalent in package names.
            # Prefer the distribution that matches the (normalized) package name.
            normalized_pkg_name = pkg_name.replace("_", "-")
            if normalized_pkg_name in distributions:
                distribution_name = normalized_pkg_name
            elif pkg_name in distributions:
                distribution_name = pkg_name
            else:
                distribution_name = distributions[0]
            package_version = importlib.metadata.version(distribution_name)
        except (importlib.metadata.PackageNotFoundError, KeyError):
            # If we cannot find the metadata (because of editable install for example), try to import directly.
            # Note that this branch will almost never be run, so we do not import packages for nothing here
            package = importlib.import_module(pkg_name)
            package_version = getattr(package, "__version__", "N/A")
        logger.debug(f"Detected {pkg_name} version: {package_version}")

    if return_version:
        return package_exists, package_version
    else:
        return package_exists, None


def resolve_internal_import(module: ModuleType | None, chained_path: str) -> Callable | ModuleType | None:
    """
    Check if a given `module` has an internal import path as defined by the `chained_path`.
    This can either be the full path (not exposed in `__init__`) OR the last part of the chain (exposed in `__init__`).

    This is an important helper function for kernels based modules to apply the import from the module
    itself, i.e. stay compatible with original libraries in certain cases.

    Example:
        Module: `mamba_ssm`
        Chained Path: `ops.triton.selective_state_update.selective_state_update`
        Resulting import attempt at:
            - `mamba_ssm.selective_state_update`
            - `mamba_ssm.ops.triton.selective_state_update.selective_state_update`
    """
    if not module:
        return None

    if final_module := getattr(module, chained_path.split(".")[-1], None):
        return final_module

    final_module = module
    for path in chained_path.split("."):
        final_module = getattr(final_module, path, None)
        if not final_module:
            return None

    return final_module


def is_env_variable_true(env_variable: str) -> bool:
    """Detect whether `env_variable` has been set to a true value in the environment"""
    return os.getenv(env_variable, "false").lower() in ("true", "1", "y", "yes", "on")


def is_env_variable_false(env_variable: str) -> bool:
    """Detect whether `env_variable` has been set to a false value in the environment"""
    return os.getenv(env_variable, "true").lower() in ("false", "0", "n", "no", "off")


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

# Try to run a native pytorch job in an environment with TorchXLA installed by setting this value to 0.
USE_TORCH_XLA = os.environ.get("USE_TORCH_XLA", "1").upper()

ACCELERATE_MIN_VERSION = "1.1.0"
BITSANDBYTES_MIN_VERSION = "0.46.1"
SCHEDULEFREE_MIN_VERSION = "1.2.6"
FSDP_MIN_VERSION = "1.12.0"
GGUF_MIN_VERSION = "0.10.0"
XLA_FSDPV2_MIN_VERSION = "2.2.0"
HQQ_MIN_VERSION = "0.2.1"
VPTQ_MIN_VERSION = "0.0.4"
TORCHAO_MIN_VERSION = "0.4.0"
AUTOROUND_MIN_VERSION = "0.5.0"
TRITON_MIN_VERSION = "1.0.0"
KERNELS_MIN_VERSION = "0.10.2"


@lru_cache
def is_torch_available() -> bool:
    try:
        is_available, torch_version = _is_package_available("torch", return_version=True)
        parsed_version = version.parse(torch_version)
        if is_available and parsed_version < version.parse("2.4.0"):
            logger.warning_once(f"Disabling PyTorch because PyTorch >= 2.4 is required but found {torch_version}")
        return is_available and version.parse(torch_version) >= version.parse("2.4.0")
    except packaging.version.InvalidVersion:
        return False


@lru_cache
def get_torch_version() -> str:
    _, torch_version = _is_package_available("torch", return_version=True)
    return torch_version


@lru_cache
def is_torch_greater_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is greater than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not is_torch_available():
        return False

    if accept_dev:
        return version.parse(version.parse(get_torch_version()).base_version) >= version.parse(library_version)
    else:
        return version.parse(get_torch_version()) >= version.parse(library_version)


@lru_cache
def is_torch_less_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is less than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not is_torch_available():
        return False

    if accept_dev:
        return version.parse(version.parse(get_torch_version()).base_version) <= version.parse(library_version)
    else:
        return version.parse(get_torch_version()) <= version.parse(library_version)


@lru_cache
def is_torch_accelerator_available() -> bool:
    if is_torch_available():
        import torch

        return hasattr(torch, "accelerator")

    return False


@lru_cache
def is_torch_cuda_available() -> bool:
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    return False


@lru_cache
def is_cuda_platform() -> bool:
    if is_torch_available():
        import torch

        return torch.version.cuda is not None
    return False


@lru_cache
def is_rocm_platform() -> bool:
    if is_torch_available():
        import torch

        return torch.version.hip is not None
    return False


@lru_cache
def is_habana_gaudi1() -> bool:
    if not is_torch_hpu_available():
        return False

    import habana_frameworks.torch.utils.experimental as htexp

    # Check if the device is Gaudi1 (vs Gaudi2, Gaudi3)
    return htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi


@lru_cache
def is_torch_mps_available(min_version: str | None = None) -> bool:
    if is_torch_available():
        import torch

        backend_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        if min_version is not None:
            flag = version.parse(get_torch_version()) >= version.parse(min_version)
            backend_available = backend_available and flag
        return backend_available
    return False


@lru_cache
def is_torch_npu_available(check_device=False) -> bool:
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if not is_torch_available() or not _is_package_available("torch_npu")[0]:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            if hasattr(torch, "npu"):
                _ = torch.npu.device_count()
                return torch.npu.is_available()
            return False
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
def is_torch_xpu_available(check_device: bool = False) -> bool:
    """
    Checks if XPU acceleration is available via stock PyTorch (>=2.6) and
    potentially if a XPU is in the environment.
    """
    if not is_torch_available():
        return False

    torch_version = version.parse(get_torch_version())
    if torch_version.major == 2 and torch_version.minor < 6:
        return False

    import torch

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache
def is_torch_mlu_available() -> bool:
    """
    Checks if `mlu` is available via an `cndev-based` check which won't trigger the drivers and leave mlu
    uninitialized.
    """
    if not is_torch_available() or not _is_package_available("torch_mlu")[0]:
        return False

    import torch
    import torch_mlu  # noqa: F401

    pytorch_cndev_based_mlu_check_previous_value = os.environ.get("PYTORCH_CNDEV_BASED_MLU_CHECK")
    try:
        os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = str(1)
        available = torch.mlu.is_available() if hasattr(torch, "mlu") else False
    finally:
        if pytorch_cndev_based_mlu_check_previous_value:
            os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = pytorch_cndev_based_mlu_check_previous_value
        else:
            os.environ.pop("PYTORCH_CNDEV_BASED_MLU_CHECK", None)

    return available


@lru_cache
def is_torch_musa_available(check_device=False) -> bool:
    "Checks if `torch_musa` is installed and potentially if a MUSA is in the environment"
    if not is_torch_available() or not _is_package_available("torch_musa")[0]:
        return False

    import torch
    import torch_musa  # noqa: F401

    torch_musa_min_version = "0.33.0"
    accelerate_available, accelerate_version = _is_package_available("accelerate", return_version=True)
    if accelerate_available and version.parse(accelerate_version) < version.parse(torch_musa_min_version):
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no MUSA is found
            if hasattr(torch, "musa"):
                _ = torch.musa.device_count()
                return torch.musa.is_available()
            return False
        except RuntimeError:
            return False
    return hasattr(torch, "musa") and torch.musa.is_available()


@lru_cache
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False) -> bool:
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."

    torch_xla_available = USE_TORCH_XLA in ENV_VARS_TRUE_VALUES and _is_package_available("torch_xla")[0]
    if not torch_xla_available:
        return False

    import torch_xla

    if check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == "TPU"

    return True


@lru_cache
def is_torch_hpu_available() -> bool:
    "Checks if `torch.hpu` is available and potentially if a HPU is in the environment"
    if (
        not is_torch_available()
        or not _is_package_available("habana_frameworks")[0]
        or not _is_package_available("habana_frameworks.torch")[0]
    ):
        return False

    torch_hpu_min_accelerate_version = "1.5.0"
    accelerate_available, accelerate_version = _is_package_available("accelerate", return_version=True)
    if accelerate_available and version.parse(accelerate_version) < version.parse(torch_hpu_min_accelerate_version):
        return False

    import torch

    if os.environ.get("PT_HPU_LAZY_MODE", "1") == "1":
        # import habana_frameworks.torch in case of lazy mode to patch torch with torch.hpu
        import habana_frameworks.torch  # noqa: F401

    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        return False

    # We patch torch.gather for int64 tensors to avoid a bug on Gaudi
    # Graph compile failed with synStatus 26 [Generic failure]
    # This can be removed once bug is fixed but for now we need it.
    original_gather = torch.gather

    def patched_gather(input: torch.Tensor, dim: int, index: torch.LongTensor) -> torch.Tensor:
        if input.dtype == torch.int64 and input.device.type == "hpu":
            return original_gather(input.to(torch.int32), dim, index).to(torch.int64)
        else:
            return original_gather(input, dim, index)

    torch.gather = patched_gather
    torch.Tensor.gather = patched_gather

    original_take_along_dim = torch.take_along_dim

    def patched_take_along_dim(input: torch.Tensor, indices: torch.LongTensor, dim: int | None = None) -> torch.Tensor:
        if input.dtype == torch.int64 and input.device.type == "hpu":
            return original_take_along_dim(input.to(torch.int32), indices, dim).to(torch.int64)
        else:
            return original_take_along_dim(input, indices, dim)

    torch.take_along_dim = patched_take_along_dim

    original_cholesky = torch.linalg.cholesky

    def safe_cholesky(A, *args, **kwargs):
        output = original_cholesky(A, *args, **kwargs)

        if torch.isnan(output).any():
            jitter_value = 1e-9
            diag_jitter = torch.eye(A.size(-1), dtype=A.dtype, device=A.device) * jitter_value
            output = original_cholesky(A + diag_jitter, *args, **kwargs)

        return output

    torch.linalg.cholesky = safe_cholesky

    original_scatter = torch.scatter

    def patched_scatter(
        input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if input.device.type == "hpu" and input is src:
            return original_scatter(input, dim, index, src.clone(), *args, **kwargs)
        else:
            return original_scatter(input, dim, index, src, *args, **kwargs)

    torch.scatter = patched_scatter
    torch.Tensor.scatter = patched_scatter

    # IlyasMoutawwakil: we patch torch.compile to use the HPU backend by default
    # https://github.com/huggingface/transformers/pull/38790#discussion_r2157043944
    # This is necessary for cases where torch.compile is used as a decorator (defaulting to inductor)
    # https://github.com/huggingface/transformers/blob/af6120b3eb2470b994c21421bb6eaa76576128b0/src/transformers/models/modernbert/modeling_modernbert.py#L204
    original_compile = torch.compile

    def hpu_backend_compile(*args, **kwargs):
        if kwargs.get("backend") not in ["hpu_backend", "eager"]:
            logger.warning(
                f"Calling torch.compile with backend={kwargs.get('backend')} on a Gaudi device is not supported. "
                "We will override the backend with 'hpu_backend' to avoid errors."
            )
            kwargs["backend"] = "hpu_backend"

        return original_compile(*args, **kwargs)

    torch.compile = hpu_backend_compile

    return True


@lru_cache
def is_torch_bf16_gpu_available() -> bool:
    if not is_torch_available():
        return False

    import torch

    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    if is_torch_xpu_available():
        return torch.xpu.is_bf16_supported()
    if is_torch_hpu_available():
        return True
    if is_torch_npu_available():
        return torch.npu.is_bf16_supported() if hasattr(torch, "npu") else False
    if is_torch_mps_available():
        # Note: Emulated in software by Metal using fp32 for hardware without native support (like M1/M2)
        return torch.backends.mps.is_macos_or_newer(14, 0)
    if is_torch_musa_available():
        return torch.musa.is_bf16_supported() if hasattr(torch, "musa") else False
    return False


@lru_cache
def is_torch_fp16_available_on_device(device: str) -> bool:
    if not is_torch_available():
        return False

    if is_torch_hpu_available():
        if is_habana_gaudi1():
            return False
        else:
            return True

    import torch

    try:
        x = torch.zeros(2, 2, dtype=torch.float16, device=device)
        _ = x @ x
        # At this moment, let's be strict of the check: check if `LayerNorm` is also supported on device, because many
        # models use this layer.
        batch, sentence_length, embedding_dim = 3, 4, 5
        embedding = torch.randn(batch, sentence_length, embedding_dim, dtype=torch.float16, device=device)
        layer_norm = torch.nn.LayerNorm(embedding_dim, dtype=torch.float16, device=device)
        _ = layer_norm(embedding)
        return True
    except Exception:
        return False


@lru_cache
def is_torch_bf16_available_on_device(device: str) -> bool:
    if not is_torch_available():
        return False

    import torch

    if device == "cuda":
        return is_torch_bf16_gpu_available()

    if device == "hpu":
        return True

    try:
        x = torch.zeros(2, 2, dtype=torch.bfloat16, device=device)
        _ = x @ x
        return True
    except Exception:
        return False


@lru_cache
def is_torch_tf32_available() -> bool:
    if not is_torch_available():
        return False

    import torch

    if is_torch_musa_available():
        if hasattr(torch, "musa"):
            device_info = torch.musa.get_device_properties(torch.musa.current_device())
            if f"{device_info.major}{device_info.minor}" >= "22":
                return True
        return False
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    return True


@lru_cache
def enable_tf32(enable: bool) -> None:
    """
    Set TF32 mode using the appropriate PyTorch API.
    For PyTorch 2.9+, uses the new fp32_precision API.
    For older versions, uses the legacy allow_tf32 flags.
    Args:
        enable: Whether to enable TF32 mode
    """
    import torch

    pytorch_version = version.parse(get_torch_version())
    if pytorch_version >= version.parse("2.9.0"):
        precision_mode = "tf32" if enable else "ieee"
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = precision_mode
    else:
        if is_torch_musa_available():
            if hasattr(torch.backends, "mudnn"):
                torch.backends.mudnn.allow_tf32 = enable
        else:
            torch.backends.cuda.matmul.allow_tf32 = enable
            torch.backends.cudnn.allow_tf32 = enable


@lru_cache
def is_torch_flex_attn_available() -> bool:
    return is_torch_available() and version.parse(get_torch_version()) >= version.parse("2.5.0")


@lru_cache
def is_grouped_mm_available() -> bool:
    return is_torch_available() and version.parse(get_torch_version()) >= version.parse("2.9.0")


@lru_cache
def is_kenlm_available() -> bool:
    return _is_package_available("kenlm")[0]


@lru_cache
def is_kernels_available(MIN_VERSION: str = KERNELS_MIN_VERSION) -> bool:
    is_available, kernels_version = _is_package_available("kernels", return_version=True)
    return is_available and version.parse(kernels_version) >= version.parse(MIN_VERSION)


@lru_cache
def is_cv2_available() -> bool:
    return _is_package_available("cv2")[0]


@lru_cache
def is_yt_dlp_available() -> bool:
    return _is_package_available("yt_dlp")[0]


@lru_cache
def is_libcst_available() -> bool:
    return _is_package_available("libcst")[0]


@lru_cache
def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION) -> bool:
    is_available, accelerate_version = _is_package_available("accelerate", return_version=True)
    return is_available and version.parse(accelerate_version) >= version.parse(min_version)


@lru_cache
def is_triton_available(min_version: str = TRITON_MIN_VERSION) -> bool:
    is_available, triton_version = _is_package_available("triton", return_version=True)
    return is_available and version.parse(triton_version) >= version.parse(min_version)


@lru_cache
def is_hadamard_available() -> bool:
    return _is_package_available("fast_hadamard_transform")[0]


@lru_cache
def is_hqq_available(min_version: str = HQQ_MIN_VERSION) -> bool:
    is_available, hqq_version = _is_package_available("hqq", return_version=True)
    return is_available and version.parse(hqq_version) >= version.parse(min_version)


@lru_cache
def is_pygments_available() -> bool:
    return _is_package_available("pygments")[0]


@lru_cache
def is_torchvision_available() -> bool:
    return _is_package_available("torchvision")[0]


@lru_cache
def is_torchvision_v2_available() -> bool:
    return is_torchvision_available()


@lru_cache
def is_galore_torch_available() -> bool:
    return _is_package_available("galore_torch")[0]


@lru_cache
def is_apollo_torch_available() -> bool:
    return _is_package_available("apollo_torch")[0]


@lru_cache
def is_torch_optimi_available() -> bool:
    return _is_package_available("optimi")[0]


@lru_cache
def is_lomo_available() -> bool:
    return _is_package_available("lomo_optim")[0]


@lru_cache
def is_grokadamw_available() -> bool:
    return _is_package_available("grokadamw")[0]


@lru_cache
def is_schedulefree_available(min_version: str = SCHEDULEFREE_MIN_VERSION) -> bool:
    is_available, schedulefree_version = _is_package_available("schedulefree", return_version=True)
    return is_available and version.parse(schedulefree_version) >= version.parse(min_version)


@lru_cache
def is_pyctcdecode_available() -> bool:
    return _is_package_available("pyctcdecode")[0]


@lru_cache
def is_librosa_available() -> bool:
    return _is_package_available("librosa")[0]


@lru_cache
def is_essentia_available() -> bool:
    return _is_package_available("essentia")[0]


@lru_cache
def is_pydantic_available() -> bool:
    return _is_package_available("pydantic")[0]


@lru_cache
def is_fastapi_available() -> bool:
    return _is_package_available("fastapi")[0]


@lru_cache
def is_uvicorn_available() -> bool:
    return _is_package_available("uvicorn")[0]


@lru_cache
def is_openai_available() -> bool:
    return _is_package_available("openai")[0]


@lru_cache
def is_pretty_midi_available() -> bool:
    return _is_package_available("pretty_midi")[0]


@lru_cache
def is_mamba_ssm_available() -> bool:
    return is_torch_cuda_available() and _is_package_available("mamba_ssm")[0]


@lru_cache
def is_mamba_2_ssm_available() -> bool:
    is_available, mamba_ssm_version = _is_package_available("mamba_ssm", return_version=True)
    return is_torch_cuda_available() and is_available and version.parse(mamba_ssm_version) >= version.parse("2.0.4")


@lru_cache
def is_flash_linear_attention_available():
    is_available, fla_version = _is_package_available("fla", return_version=True)
    return is_torch_cuda_available() and is_available and version.parse(fla_version) >= version.parse("0.2.2")


@lru_cache
def is_causal_conv1d_available() -> bool:
    return is_torch_cuda_available() and _is_package_available("causal_conv1d")[0]


@lru_cache
def is_xlstm_available() -> bool:
    return is_torch_available() and _is_package_available("xlstm")[0]


@lru_cache
def is_mambapy_available() -> bool:
    return is_torch_available() and _is_package_available("mambapy")[0]


@lru_cache
def is_peft_available() -> bool:
    return _is_package_available("peft")[0]


@lru_cache
def is_bs4_available() -> bool:
    return _is_package_available("bs4")[0]


@lru_cache
def is_coloredlogs_available() -> bool:
    return _is_package_available("coloredlogs")[0]


@lru_cache
def is_onnx_available() -> bool:
    return _is_package_available("onnx")[0]


@lru_cache
def is_flute_available() -> bool:
    is_available, flute_version = _is_package_available("flute", return_version=True)
    return is_available and version.parse(flute_version) >= version.parse("0.4.1")


@lru_cache
def is_g2p_en_available() -> bool:
    return _is_package_available("g2p_en")[0]


@lru_cache
def is_torch_neuroncore_available(check_device=True) -> bool:
    return is_torch_xla_available() and _is_package_available("torch_neuronx")[0]


@lru_cache
def is_torch_tensorrt_fx_available() -> bool:
    return _is_package_available("torch_tensorrt")[0] and _is_package_available("torch_tensorrt.fx")[0]


@lru_cache
def is_datasets_available() -> bool:
    return _is_package_available("datasets")[0]


@lru_cache
def is_detectron2_available() -> bool:
    # We need this try/except block because otherwise after uninstalling the library, it stays available for some reason
    # i.e. `import detectron2` and `import detectron2.modeling` still work, even though the library is uninstalled
    # (the package exists but the objects are not reachable) - so here we explicitly try to import an object from it
    try:
        from detectron2.modeling import META_ARCH_REGISTRY  # noqa

        return True
    except Exception:
        return False


@lru_cache
def is_rjieba_available() -> bool:
    return _is_package_available("rjieba")[0]


@lru_cache
def is_psutil_available() -> bool:
    return _is_package_available("psutil")[0]


@lru_cache
def is_py3nvml_available() -> bool:
    return _is_package_available("py3nvml")[0]


@lru_cache
def is_sacremoses_available() -> bool:
    return _is_package_available("sacremoses")[0]


@lru_cache
def is_apex_available() -> bool:
    return _is_package_available("apex")[0]


@lru_cache
def is_aqlm_available() -> bool:
    return _is_package_available("aqlm")[0]


@lru_cache
def is_vptq_available(min_version: str = VPTQ_MIN_VERSION) -> bool:
    is_available, vptq_version = _is_package_available("vptq", return_version=True)
    return is_available and version.parse(vptq_version) >= version.parse(min_version)


@lru_cache
def is_av_available() -> bool:
    return _is_package_available("av")[0]


@lru_cache
def is_decord_available() -> bool:
    return _is_package_available("decord")[0]


@lru_cache
def is_torchcodec_available() -> bool:
    return _is_package_available("torchcodec")[0]


@lru_cache
def is_ninja_available() -> bool:
    r"""
    Code comes from *torch.utils.cpp_extension.is_ninja_available()*. Returns `True` if the
    [ninja](https://ninja-build.org/) build system is available on the system, `False` otherwise.
    """
    try:
        subprocess.check_output(["ninja", "--version"])
    except Exception:
        return False
    else:
        return True


@lru_cache
def is_bitsandbytes_available(min_version: str = BITSANDBYTES_MIN_VERSION) -> bool:
    is_available, bitsandbytes_version = _is_package_available("bitsandbytes", return_version=True)
    return is_available and version.parse(bitsandbytes_version) >= version.parse(min_version)


@lru_cache
def is_flash_attn_2_available() -> bool:
    is_available, flash_attn_version = _is_package_available("flash_attn", return_version=True)
    if not is_available or not (is_torch_cuda_available() or is_torch_mlu_available()):
        return False

    import torch

    try:
        if torch.version.cuda:
            return version.parse(flash_attn_version) >= version.parse("2.1.0")
        elif torch.version.hip:
            # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
            return version.parse(flash_attn_version) >= version.parse("2.0.4")
        elif is_torch_mlu_available():
            return version.parse(flash_attn_version) >= version.parse("2.3.3")
        else:
            return False
    except packaging.version.InvalidVersion:
        return False


@lru_cache
def is_flash_attn_3_available() -> bool:
    return is_torch_cuda_available() and _is_package_available("flash_attn_3")[0]


@lru_cache
def is_flash_attn_greater_or_equal_2_10() -> bool:
    _, flash_attn_version = _is_package_available("flash_attn", return_version=True)
    return is_flash_attn_2_available() and version.parse(flash_attn_version) >= version.parse("2.1.0")


@lru_cache
def is_flash_attn_greater_or_equal(library_version: str) -> bool:
    is_available, flash_attn_version = _is_package_available("flash_attn", return_version=True)
    if not is_available:
        return False
    try:
        return version.parse(flash_attn_version) >= version.parse(library_version)
    except packaging.version.InvalidVersion:
        return False


@lru_cache
def is_huggingface_hub_greater_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    is_available, hub_version = _is_package_available("huggingface_hub", return_version=True)
    if not is_available:
        return False

    if accept_dev:
        return version.parse(version.parse(hub_version).base_version) >= version.parse(library_version)
    else:
        return version.parse(hub_version) >= version.parse(library_version)


@lru_cache
def is_quanto_greater(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is greater than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not is_optimum_quanto_available():
        return False

    _, quanto_version = _is_package_available("optimum.quanto", return_version=True)
    if accept_dev:
        return version.parse(version.parse(quanto_version).base_version) > version.parse(library_version)
    else:
        return version.parse(quanto_version) > version.parse(library_version)


@lru_cache
def is_torchdistx_available():
    return _is_package_available("torchdistx")[0]


@lru_cache
def is_faiss_available() -> bool:
    return _is_package_available("faiss")[0]


@lru_cache
def is_scipy_available() -> bool:
    return _is_package_available("scipy")[0]


@lru_cache
def is_sklearn_available() -> bool:
    return _is_package_available("sklearn")[0]


@lru_cache
def is_sentencepiece_available() -> bool:
    return _is_package_available("sentencepiece")[0]


@lru_cache
def is_seqio_available() -> bool:
    return _is_package_available("seqio")[0]


@lru_cache
def is_gguf_available(min_version: str = GGUF_MIN_VERSION) -> bool:
    is_available, gguf_version = _is_package_available("gguf", return_version=True)
    return is_available and version.parse(gguf_version) >= version.parse(min_version)


@lru_cache
def is_protobuf_available() -> bool:
    return _is_package_available("google")[0] and _is_package_available("google.protobuf")[0]


@lru_cache
def is_fsdp_available(min_version: str = FSDP_MIN_VERSION) -> bool:
    return is_torch_available() and version.parse(get_torch_version()) >= version.parse(min_version)


@lru_cache
def is_optimum_available() -> bool:
    return _is_package_available("optimum")[0]


@lru_cache
def is_llm_awq_available() -> bool:
    return _is_package_available("awq")[0]


@lru_cache
def is_auto_round_available(min_version: str = AUTOROUND_MIN_VERSION) -> bool:
    is_available, auto_round_version = _is_package_available("auto_round", return_version=True)
    return is_available and version.parse(auto_round_version) >= version.parse(min_version)


@lru_cache
def is_optimum_quanto_available():
    return is_optimum_available() and _is_package_available("optimum.quanto")[0]


@lru_cache
def is_quark_available() -> bool:
    return _is_package_available("quark")[0]


@lru_cache
def is_fp_quant_available():
    is_available, fp_quant_version = _is_package_available("fp_quant", return_version=True)
    return is_available and version.parse(fp_quant_version) >= version.parse("0.3.2")


@lru_cache
def is_qutlass_available():
    is_available, qutlass_version = _is_package_available("qutlass", return_version=True)
    return is_available and version.parse(qutlass_version) >= version.parse("0.2.0")


@lru_cache
def is_compressed_tensors_available() -> bool:
    return _is_package_available("compressed_tensors")[0]


@lru_cache
def is_sinq_available() -> bool:
    return _is_package_available("sinq")


@lru_cache
def is_gptqmodel_available() -> bool:
    return _is_package_available("gptqmodel")[0]


@lru_cache
def is_fbgemm_gpu_available() -> bool:
    return _is_package_available("fbgemm_gpu")[0]


@lru_cache
def is_levenshtein_available() -> bool:
    return _is_package_available("Levenshtein")[0]


@lru_cache
def is_optimum_neuron_available() -> bool:
    return is_optimum_available() and _is_package_available("optimum.neuron")[0]


@lru_cache
def is_tokenizers_available() -> bool:
    return _is_package_available("tokenizers")[0]


@lru_cache
def is_vision_available() -> bool:
    return _is_package_available("PIL")[0]


@lru_cache
def is_pytesseract_available() -> bool:
    return _is_package_available("pytesseract")[0]


@lru_cache
def is_pytest_available() -> bool:
    return _is_package_available("pytest")[0]


@lru_cache
def is_pytest_order_available() -> bool:
    return is_pytest_available() and _is_package_available("pytest_order")[0]


@lru_cache
def is_spacy_available() -> bool:
    return _is_package_available("spacy")[0]


@lru_cache
def is_pytorch_quantization_available() -> bool:
    return _is_package_available("pytorch_quantization")[0]


@lru_cache
def is_pandas_available() -> bool:
    return _is_package_available("pandas")[0]


@lru_cache
def is_soundfile_available() -> bool:
    return _is_package_available("soundfile")[0]


@lru_cache
def is_timm_available() -> bool:
    return _is_package_available("timm")[0]


@lru_cache
def is_natten_available() -> bool:
    return _is_package_available("natten")[0]


@lru_cache
def is_nltk_available() -> bool:
    return _is_package_available("nltk")[0]


@lru_cache
def is_numba_available() -> bool:
    is_available = _is_package_available("numba")[0]
    if not is_available:
        return False

    numpy_available, numpy_version = _is_package_available("numpy", return_version=True)
    return not numpy_available or version.parse(numpy_version) < version.parse("2.2.0")


@lru_cache
def is_torchaudio_available() -> bool:
    return _is_package_available("torchaudio")[0]


@lru_cache
def is_torchao_available(min_version: str = TORCHAO_MIN_VERSION) -> bool:
    is_available, torchao_version = _is_package_available("torchao", return_version=True)
    return is_available and version.parse(torchao_version) >= version.parse(min_version)


@lru_cache
def is_speech_available() -> bool:
    # For now this depends on torchaudio but the exact dependency might evolve in the future.
    return is_torchaudio_available()


@lru_cache
def is_spqr_available() -> bool:
    return _is_package_available("spqr_quant")[0]


@lru_cache
def is_phonemizer_available() -> bool:
    return _is_package_available("phonemizer")[0]


@lru_cache
def is_uroman_available() -> bool:
    return _is_package_available("uroman")[0]


@lru_cache
def is_sudachi_available() -> bool:
    return _is_package_available("sudachipy")[0]


@lru_cache
def is_sudachi_projection_available() -> bool:
    is_available, sudachipy_version = _is_package_available("sudachipy", return_version=True)
    return is_available and version.parse(sudachipy_version) >= version.parse("0.6.8")


@lru_cache
def is_jumanpp_available() -> bool:
    return _is_package_available("rhoknp")[0] and shutil.which("jumanpp") is not None


@lru_cache
def is_cython_available() -> bool:
    return _is_package_available("pyximport")[0]


@lru_cache
def is_jinja_available() -> bool:
    return _is_package_available("jinja2")[0]


@lru_cache
def is_jmespath_available() -> bool:
    return _is_package_available("jmespath")[0]


@lru_cache
def is_mlx_available() -> bool:
    return _is_package_available("mlx")[0]


@lru_cache
def is_num2words_available() -> bool:
    return _is_package_available("num2words")[0]


@lru_cache
def is_tiktoken_available(with_blobfile: bool = True) -> bool:
    if not _is_package_available("tiktoken")[0]:
        return False
    return with_blobfile and _is_package_available("blobfile")[0] or True


@lru_cache
def is_liger_kernel_available() -> bool:
    is_available, liger_kernel_version = _is_package_available("liger_kernel", return_version=True)
    return is_available and version.parse(liger_kernel_version) >= version.parse("0.3.0")


@lru_cache
def is_rich_available() -> bool:
    return _is_package_available("rich")[0]


@lru_cache
def is_matplotlib_available() -> bool:
    return _is_package_available("matplotlib")[0]


@lru_cache
def is_mistral_common_available() -> bool:
    return _is_package_available("mistral_common")[0]


@lru_cache
def is_opentelemetry_available() -> bool:
    try:
        return _is_package_available("opentelemetry")[0] and version.parse(
            importlib.metadata.version("opentelemetry-api")
        ) >= version.parse("1.30.0")
    except Exception as _:
        return False


@lru_cache
def is_pynvml_available() -> bool:
    return _is_package_available("pynvml")[0]


def check_torch_load_is_safe() -> None:
    if not is_torch_greater_or_equal("2.6"):
        raise ValueError(
            "Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users "
            "to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply "
            "when loading files with safetensors."
            "\nSee the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434"
        )


def torch_only_method(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if not is_torch_available():
            raise ImportError("You need to install pytorch to use this method or class")
        else:
            return fn(*args, **kwargs)

    return wrapper


def is_torch_deterministic() -> bool:
    """
    Check whether pytorch uses deterministic algorithms by looking if torch.set_deterministic_debug_mode() is set to 1 or 2"
    """
    if is_torch_available():
        import torch

        if torch.get_deterministic_debug_mode() == 0:
            return False
        else:
            return True

    return False


@lru_cache
def get_torch_major_and_minor_version() -> str:
    torch_version = get_torch_version()
    if torch_version == "N/A":
        return "N/A"
    parsed_version = version.parse(torch_version)
    return str(parsed_version.major) + "." + str(parsed_version.minor)


def is_torchdynamo_compiling() -> bool:
    # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622)
    # hence rather relying on `torch.compiler.is_compiling()` when possible (torch>=2.3)
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        return False


def is_torchdynamo_exporting() -> bool:
    try:
        import torch

        return torch.compiler.is_exporting()
    except Exception:
        return False


def is_torch_fx_proxy(x) -> bool:
    try:
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    except Exception:
        return False


def is_fake_tensor(x) -> bool:
    try:
        import torch

        return isinstance(x, torch._subclasses.FakeTensor)
    except Exception:
        return False


def is_jax_jitting(x):
    """returns True if we are inside of `jax.jit` context, False otherwise.

    When a torch model is being compiled with `jax.jit` using torchax,
    the tensor that goes through the model would be an instance of
    `torchax.tensor.Tensor`, which is a tensor subclass. This tensor has
    a `jax` method to return the inner Jax array
    (https://github.com/google/torchax/blob/13ce870a1d9adb2430333c27bb623469e3aea34e/torchax/tensor.py#L134).
    Here we use ducktyping to detect if the inner jax array is a jax Tracer
    then we are in tracing context. (See more at: https://github.com/jax-ml/jax/discussions/9241)

    Args:
      x: torch.Tensor

    Returns:
      bool: whether we are inside of jax jit tracing.
    """

    if not hasattr(x, "jax"):
        return False
    try:
        import jax

        return isinstance(x.jax(), jax.core.Tracer)
    except Exception:
        return False


def is_jit_tracing() -> bool:
    try:
        import torch

        return torch.jit.is_tracing()
    except Exception:
        return False


def is_cuda_stream_capturing() -> bool:
    try:
        import torch

        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def is_tracing(tensor=None) -> bool:
    """Checks whether we are tracing a graph with dynamo (compile or export), torch.jit, torch.fx, jax.jit (with torchax) or
    CUDA stream capturing or FakeTensor"""

    # Note that `is_torchdynamo_compiling` checks both compiling and exporting (the export check is stricter and
    # only checks export)
    _is_tracing = is_torchdynamo_compiling() or is_jit_tracing() or is_cuda_stream_capturing()
    if tensor is not None:
        _is_tracing |= is_torch_fx_proxy(tensor)
        _is_tracing |= is_fake_tensor(tensor)
        _is_tracing |= is_jax_jitting(tensor)

    return _is_tracing


def torch_compilable_check(cond: Any, msg: str | Callable[[], str], error_type: type[Exception] = ValueError) -> None:
    """
    Combines the functionalities of `torch._check`, `torch._check_with` and `torch._check_tensor_all_with` to provide a
    unified way to perform checks that are compatible with TorchDynamo (torch.compile & torch.export).

    The advantage of using `torch._check(cond, msg, error_type)` over `if cond: raise error_type(msg)` is that the former
    works as a truthfulness hint for TorchDynamo, instead of failing with a data-dependent control flow error during compilation.

    All checks using this method can be disabled in production environments by setting `TRANSFORMERS_DISABLE_TORCH_CHECK=1`.

    Args:
        cond (`bool`, `torch.Tensor` or `Callable[[], bool | torch.Tensor]`): The condition to check.
        msg (`str` or `Callable[[], str]`): The error message to display if the condition is not met.
        error_type (`type[Exception]`, *optional*, defaults to `ValueError`): The type of error to raise if the condition is not met.

    Raises:
        error_type: If the condition is not met.
    """
    if os.getenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "0") == "1":
        return

    import torch

    if not callable(msg):
        # torch._check requires msg to be a callable but we want to keep the API simple for users
        def msg_callable():
            return msg
    else:
        msg_callable = msg

    if callable(cond):
        cond = cond()

    if isinstance(cond, torch.Tensor):
        torch._check_tensor_all_with(error_type, cond, msg_callable)
    else:
        torch._check_with(error_type, cond, msg_callable)


@lru_cache
def is_in_notebook() -> bool:
    try:
        # Check if we are running inside Marimo
        if "marimo" in sys.modules:
            return True
        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        # Removed the lines to include VSCode
        if "DATABRICKS_RUNTIME_VERSION" in os.environ and os.environ["DATABRICKS_RUNTIME_VERSION"] < "11.0":
            # Databricks Runtime 11.0 and above uses IPython kernel by default so it should be compatible with Jupyter notebook
            # https://docs.microsoft.com/en-us/azure/databricks/notebooks/ipython-kernel
            raise ImportError("databricks")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


def is_sagemaker_dp_enabled() -> bool:
    # Get the sagemaker specific env variable.
    sagemaker_params = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        sagemaker_params = json.loads(sagemaker_params)
        if not sagemaker_params.get("sagemaker_distributed_dataparallel_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return _is_package_available("smdistributed")[0]


def is_sagemaker_mp_enabled() -> bool:
    # Get the sagemaker specific mp parameters from smp_options variable.
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # Parse it and check the field "partitions" is included, it is required for model parallel.
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    # Get the sagemaker specific framework parameters from mpi_options variable.
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return _is_package_available("smdistributed")[0]


def is_training_run_on_sagemaker() -> bool:
    return "SAGEMAKER_JOB_NAME" in os.environ


# docstyle-ignore
AV_IMPORT_ERROR = """
{0} requires the PyAv library but it was not found in your environment. You can install it with:
```
pip install av
```
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
YT_DLP_IMPORT_ERROR = """
{0} requires the YT-DLP library but it was not found in your environment. You can install it with:
```
pip install yt-dlp
```
Please note that you may need to restart your runtime after installation.
"""

DECORD_IMPORT_ERROR = """
{0} requires the PyAv library but it was not found in your environment. You can install it with:
```
pip install decord
```
Please note that you may need to restart your runtime after installation.
"""

TORCHCODEC_IMPORT_ERROR = """
{0} requires the TorchCodec (https://github.com/pytorch/torchcodec) library, but it was not found in your environment. You can install it with:
```
pip install torchcodec
```
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
CV2_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with:
```
pip install opencv-python
```
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
DATASETS_IMPORT_ERROR = """
{0} requires the 🤗 Datasets library but it was not found in your environment. You can install it with:
```
pip install datasets
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install datasets
```
then restarting your kernel.

Note that if you have a local folder named `datasets` or a local python file named `datasets.py` in your current
working directory, python may try to import this instead of the 🤗 Datasets library. You should rename this folder or
that python file if that's the case. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
TOKENIZERS_IMPORT_ERROR = """
{0} requires the 🤗 Tokenizers library but it was not found in your environment. You can install it with:
```
pip install tokenizers
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install tokenizers
```
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Check out the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Check out the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
FAISS_IMPORT_ERROR = """
{0} requires the faiss library but it was not found in your environment. Check out the instructions on the
installation page of its repo: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Check out the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
TORCHVISION_IMPORT_ERROR = """
{0} requires the Torchvision library but it was not found in your environment. Check out the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SKLEARN_IMPORT_ERROR = """
{0} requires the scikit-learn library but it was not found in your environment. You can install it with:
```
pip install -U scikit-learn
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install -U scikit-learn
```
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Check out the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


LEVENSHTEIN_IMPORT_ERROR = """
{0} requires the python-Levenshtein library but it was not found in your environment. You can install it with pip: `pip
install python-Levenshtein`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
G2P_EN_IMPORT_ERROR = """
{0} requires the g2p-en library but it was not found in your environment. You can install it with pip:
`pip install g2p-en`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYTORCH_QUANTIZATION_IMPORT_ERROR = """
{0} requires the pytorch-quantization library but it was not found in your environment. You can install it with pip:
`pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TORCHAUDIO_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. Please install it and restart your
runtime.
"""

# docstyle-ignore
PANDAS_IMPORT_ERROR = """
{0} requires the pandas library but it was not found in your environment. You can install it with pip as
explained here: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PHONEMIZER_IMPORT_ERROR = """
{0} requires the phonemizer library but it was not found in your environment. You can install it with pip:
`pip install phonemizer`. Please note that you may need to restart your runtime after installation.
"""
# docstyle-ignore
UROMAN_IMPORT_ERROR = """
{0} requires the uroman library but it was not found in your environment. You can install it with pip:
`pip install uroman`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SACREMOSES_IMPORT_ERROR = """
{0} requires the sacremoses library but it was not found in your environment. You can install it with pip:
`pip install sacremoses`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
SPEECH_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TIMM_IMPORT_ERROR = """
{0} requires the timm library but it was not found in your environment. You can install it with pip:
`pip install timm`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
NATTEN_IMPORT_ERROR = """
{0} requires the natten library but it was not found in your environment. You can install it by referring to:
shi-labs.com/natten . You can also install it with pip (may take longer to build):
`pip install natten`. Please note that you may need to restart your runtime after installation.
"""

NUMEXPR_IMPORT_ERROR = """
{0} requires the numexpr library but it was not found in your environment. You can install it by referring to:
https://numexpr.readthedocs.io/en/latest/index.html.
"""


# docstyle-ignore
NLTK_IMPORT_ERROR = """
{0} requires the NLTK library but it was not found in your environment. You can install it by referring to:
https://www.nltk.org/install.html. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYDANTIC_IMPORT_ERROR = """
{0} requires the pydantic library but it was not found in your environment. You can install it with pip:
`pip install pydantic`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
FASTAPI_IMPORT_ERROR = """
{0} requires the fastapi library but it was not found in your environment. You can install it with pip:
`pip install fastapi`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
UVICORN_IMPORT_ERROR = """
{0} requires the uvicorn library but it was not found in your environment. You can install it with pip:
`pip install uvicorn`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
OPENAI_IMPORT_ERROR = """
{0} requires the openai library but it was not found in your environment. You can install it with pip:
`pip install openai`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYCTCDECODE_IMPORT_ERROR = """
{0} requires the pyctcdecode library but it was not found in your environment. You can install it with pip:
`pip install pyctcdecode`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library >= {ACCELERATE_MIN_VERSION} it was not found in your environment.
You can install or update it with pip: `pip install --upgrade accelerate`. Please note that you may need to restart your
runtime after installation.
"""

# docstyle-ignore
ESSENTIA_IMPORT_ERROR = """
{0} requires essentia library. But that was not found in your environment. You can install them with pip:
`pip install essentia==2.1b6.dev1034`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
LIBROSA_IMPORT_ERROR = """
{0} requires the librosa library. But that was not found in your environment. You can install them with pip:
`pip install librosa`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PRETTY_MIDI_IMPORT_ERROR = """
{0} requires the pretty_midi library. But that was not found in your environment. You can install them with pip:
`pip install pretty_midi`
Please note that you may need to restart your runtime after installation.
"""


CYTHON_IMPORT_ERROR = """
{0} requires the Cython library but it was not found in your environment. You can install it with pip: `pip install
Cython`. Please note that you may need to restart your runtime after installation.
"""

RJIEBA_IMPORT_ERROR = """
{0} requires the rjieba library but it was not found in your environment. You can install it with pip: `pip install
rjieba`. Please note that you may need to restart your runtime after installation.
"""

PEFT_IMPORT_ERROR = """
{0} requires the peft library but it was not found in your environment. You can install it with pip: `pip install
peft`. Please note that you may need to restart your runtime after installation.
"""

JINJA_IMPORT_ERROR = """
{0} requires the jinja library but it was not found in your environment. You can install it with pip: `pip install
jinja2`. Please note that you may need to restart your runtime after installation.
"""

RICH_IMPORT_ERROR = """
{0} requires the rich library but it was not found in your environment. You can install it with pip: `pip install
rich`. Please note that you may need to restart your runtime after installation.
"""

MISTRAL_COMMON_IMPORT_ERROR = """
{0} requires the mistral-common library but it was not found in your environment. You can install it with pip: `pip install mistral-common`. Please note that you may need to restart your runtime after installation.
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("av", (is_av_available, AV_IMPORT_ERROR)),
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("decord", (is_decord_available, DECORD_IMPORT_ERROR)),
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        ("essentia", (is_essentia_available, ESSENTIA_IMPORT_ERROR)),
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        ("g2p_en", (is_g2p_en_available, G2P_EN_IMPORT_ERROR)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("phonemizer", (is_phonemizer_available, PHONEMIZER_IMPORT_ERROR)),
        ("uroman", (is_uroman_available, UROMAN_IMPORT_ERROR)),
        ("pretty_midi", (is_pretty_midi_available, PRETTY_MIDI_IMPORT_ERROR)),
        ("levenshtein", (is_levenshtein_available, LEVENSHTEIN_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("sacremoses", (is_sacremoses_available, SACREMOSES_IMPORT_ERROR)),
        ("pytorch_quantization", (is_pytorch_quantization_available, PYTORCH_QUANTIZATION_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
        ("natten", (is_natten_available, NATTEN_IMPORT_ERROR)),
        ("nltk", (is_nltk_available, NLTK_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("torchvision", (is_torchvision_available, TORCHVISION_IMPORT_ERROR)),
        ("torchcodec", (is_torchcodec_available, TORCHCODEC_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        ("rjieba", (is_rjieba_available, RJIEBA_IMPORT_ERROR)),
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        ("jinja", (is_jinja_available, JINJA_IMPORT_ERROR)),
        ("yt_dlp", (is_yt_dlp_available, YT_DLP_IMPORT_ERROR)),
        ("rich", (is_rich_available, RICH_IMPORT_ERROR)),
        ("pydantic", (is_pydantic_available, PYDANTIC_IMPORT_ERROR)),
        ("fastapi", (is_fastapi_available, FASTAPI_IMPORT_ERROR)),
        ("uvicorn", (is_uvicorn_available, UVICORN_IMPORT_ERROR)),
        ("openai", (is_openai_available, OPENAI_IMPORT_ERROR)),
        ("mistral-common", (is_mistral_common_available, MISTRAL_COMMON_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    """
    Method that automatically raises in case the specified backends are not available. It is often used during class
    initialization to ensure the required dependencies are installed:

    ```py
    requires_backends(self, ["torch"])
    ```

    The backends should be defined in the `BACKEND_MAPPING` defined in `transformers.utils.import_utils`.

    Args:
        obj: object to be checked
        backends: list or tuple of backends to check.
    """
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    failed = []
    for backend in backends:
        if isinstance(backend, Backend):
            available, msg = backend.is_satisfied, backend.error_message
        else:
            available, msg = BACKENDS_MAPPING[backend]

        if not available():
            failed.append(msg.format(name))

    if failed:
        raise ImportError("".join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    is_dummy = True

    def __getattribute__(cls, key):
        if (key.startswith("_") and key != "_from_config") or key == "is_dummy" or key == "mro" or key == "call":
            return super().__getattribute__(key)
        requires_backends(cls, cls._backends)


BACKENDS_T = frozenset[str]
IMPORT_STRUCTURE_T = dict[BACKENDS_T, dict[str, set[str]]]


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: IMPORT_STRUCTURE_T,
        module_spec: importlib.machinery.ModuleSpec | None = None,
        extra_objects: dict[str, object] | None = None,
        explicit_import_shortcut: dict[str, list[str]] | None = None,
    ):
        super().__init__(name)

        self._object_missing_backend = {}
        self._explicit_import_shortcut = explicit_import_shortcut if explicit_import_shortcut else {}

        if any(isinstance(key, frozenset) for key in import_structure):
            self._modules = set()
            self._class_to_module = {}
            self.__all__ = []

            _import_structure = {}

            for backends, module in import_structure.items():
                missing_backends = []

                # This ensures that if a module is importable, then all other keys of the module are importable.
                # As an example, in module.keys() we might have the following:
                #
                # dict_keys(['models.nllb_moe.configuration_nllb_moe', 'models.sew_d.configuration_sew_d'])
                #
                # with this, we don't only want to be able to import these explicitly, we want to be able to import
                # every intermediate module as well. Therefore, this is what is returned:
                #
                # {
                #     'models.nllb_moe.configuration_nllb_moe',
                #     'models.sew_d.configuration_sew_d',
                #     'models',
                #     'models.sew_d', 'models.nllb_moe'
                # }

                module_keys = set(
                    chain(*[[k.rsplit(".", i)[0] for i in range(k.count(".") + 1)] for k in list(module.keys())])
                )

                for backend in backends:
                    if backend in BACKENDS_MAPPING:
                        callable, _ = BACKENDS_MAPPING[backend]
                    else:
                        if any(key in backend for key in ["=", "<", ">"]):
                            backend = Backend(backend)
                            callable = backend.is_satisfied
                        else:
                            raise ValueError(
                                f"Backend should be defined in the BACKENDS_MAPPING. Offending backend: {backend}"
                            )

                    try:
                        if not callable():
                            missing_backends.append(backend)
                    except (ModuleNotFoundError, RuntimeError):
                        missing_backends.append(backend)

                self._modules = self._modules.union(module_keys)

                for key, values in module.items():
                    if missing_backends:
                        self._object_missing_backend[key] = missing_backends

                    for value in values:
                        self._class_to_module[value] = key
                        if missing_backends:
                            self._object_missing_backend[value] = missing_backends
                    _import_structure.setdefault(key, []).extend(values)

                # Needed for autocompletion in an IDE
                self.__all__.extend(module_keys | set(chain(*module.values())))

            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = _import_structure

        # This can be removed once every exportable object has a `require()` require.
        else:
            self._modules = set(import_structure.keys())
            self._class_to_module = {}
            for key, values in import_structure.items():
                for value in values:
                    self._class_to_module[value] = key
            # Needed for autocompletion in an IDE
            self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = list(super().__dir__())
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._object_missing_backend:
            missing_backends = self._object_missing_backend[name]

            class Placeholder(metaclass=DummyObject):
                _backends = missing_backends

                def __init__(self, *args, **kwargs):
                    requires_backends(self, missing_backends)

                def call(self, *args, **kwargs):
                    pass

            Placeholder.__name__ = name

            if name not in self._class_to_module:
                module_name = f"transformers.{name}"
            else:
                module_name = self._class_to_module[name]
                if not module_name.startswith("transformers."):
                    module_name = f"transformers.{module_name}"

            Placeholder.__module__ = module_name

            value = Placeholder
        elif name in self._class_to_module:
            try:
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            except (ModuleNotFoundError, RuntimeError, AttributeError) as e:
                # V5: If trying to import a *TokenizerFast symbol, transparently fall back to the
                # non-Fast symbol from the same module when available. This lets us keep only one
                # backend tokenizer class while preserving legacy public names.
                if name.endswith("TokenizerFast"):
                    fallback_name = name[:-4]
                    # Prefer importing the module that declares the fallback symbol if known
                    try:
                        if fallback_name in self._class_to_module:
                            fb_module = self._get_module(self._class_to_module[fallback_name])
                            fallback_value = getattr(fb_module, fallback_name)
                        else:
                            module = self._get_module(self._class_to_module[name])
                            fallback_value = getattr(module, fallback_name)
                        setattr(self, fallback_name, fallback_value)
                        value = fallback_value
                    except Exception:
                        # If we can't find the fallback here, try converter logic as a last resort
                        # before giving up
                        value = None
                        # Try converter mapping for Fast tokenizers that don't exist
                        if value is None and name.endswith("TokenizerFast"):
                            lookup_name = name[:-4]
                            try:
                                from ..convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS

                                if lookup_name in SLOW_TO_FAST_CONVERTERS:
                                    converter_class = SLOW_TO_FAST_CONVERTERS[lookup_name]
                                    converter_base_name = converter_class.__name__.replace("Converter", "")
                                    preferred_tokenizer_name = f"{converter_base_name}Tokenizer"

                                    candidate_names = [preferred_tokenizer_name]
                                    for tokenizer_name, tokenizer_converter in SLOW_TO_FAST_CONVERTERS.items():
                                        if tokenizer_converter is converter_class and tokenizer_name != lookup_name:
                                            if tokenizer_name not in candidate_names:
                                                candidate_names.append(tokenizer_name)

                                    # Try to import the preferred candidate directly
                                    import importlib

                                    for candidate_name in candidate_names:
                                        base_tokenizer_class = None

                                        # Try to derive module path from tokenizer name (e.g., "AlbertTokenizer" -> "albert")
                                        # Remove "Tokenizer" suffix and convert to lowercase
                                        if candidate_name.endswith("Tokenizer"):
                                            model_name = candidate_name[:-10].lower()  # Remove "Tokenizer"
                                            module_path = f"transformers.models.{model_name}.tokenization_{model_name}"
                                            try:
                                                module = importlib.import_module(module_path)
                                                base_tokenizer_class = getattr(module, candidate_name)
                                            except Exception:
                                                logger.debug(f"{module_path} does not have {candidate_name} defined.")

                                        # Fallback: try via _class_to_module
                                        if base_tokenizer_class is None and candidate_name in self._class_to_module:
                                            try:
                                                alias_module_name = self._class_to_module[candidate_name]
                                                alias_module = self._get_module(alias_module_name)
                                                base_tokenizer_class = getattr(alias_module, candidate_name)
                                            except Exception:
                                                logger.debug(
                                                    f"{alias_module_name} does not have {candidate_name} defined"
                                                )

                                        # If we still don't have base_tokenizer_class, skip this candidate
                                        if base_tokenizer_class is None:
                                            logger.debug(f"skipping candidate {candidate_name}")
                                            continue

                                        # If we got here, we have base_tokenizer_class
                                        value = base_tokenizer_class

                                        setattr(self, candidate_name, base_tokenizer_class)
                                        if lookup_name != candidate_name:
                                            setattr(self, lookup_name, value)
                                        setattr(self, name, value)
                                        break
                            except Exception as e:
                                logger.debug(f"Could not create tokenizer alias: {e}")

                        if value is None:
                            raise ModuleNotFoundError(
                                f"Could not import module '{name}'. Are this object's requirements defined correctly?"
                            ) from e
                else:
                    raise ModuleNotFoundError(
                        f"Could not import module '{name}'. Are this object's requirements defined correctly?"
                    ) from e

        elif name in self._modules:
            try:
                value = self._get_module(name)
            except (ModuleNotFoundError, RuntimeError) as e:
                raise ModuleNotFoundError(
                    f"Could not import module '{name}'. Are this object's requirements defined correctly?"
                ) from e
        else:
            # V5: If a *TokenizerFast symbol is requested but not present in the import structure,
            # try to resolve to the corresponding non-Fast symbol's module if available.
            if name.endswith("TokenizerFast"):
                fallback_name = name[:-4]
                if fallback_name in self._class_to_module:
                    try:
                        fb_module = self._get_module(self._class_to_module[fallback_name])
                        value = getattr(fb_module, fallback_name)
                        setattr(self, fallback_name, value)
                        setattr(self, name, value)
                        return value
                    except Exception as e:
                        logger.debug(f"Could not load fallback {fallback_name}: {e}")
            # V5: If a tokenizer class doesn't exist, check if it should alias to another tokenizer
            # via the converter mapping (e.g., FNetTokenizer -> AlbertTokenizer via AlbertConverter)
            value = None
            if name.endswith("Tokenizer") or name.endswith("TokenizerFast"):
                # Strip "Fast" suffix for converter lookup if present
                lookup_name = name[:-4] if name.endswith("TokenizerFast") else name

                try:
                    # Lazy import to avoid circular dependencies
                    from ..convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS

                    # Check if this tokenizer has a converter mapping
                    if lookup_name in SLOW_TO_FAST_CONVERTERS:
                        converter_class = SLOW_TO_FAST_CONVERTERS[lookup_name]

                        # Find which tokenizer class uses the same converter (reverse lookup)
                        # Prefer the tokenizer that matches the converter name pattern
                        # (e.g., AlbertConverter -> AlbertTokenizer)
                        converter_base_name = converter_class.__name__.replace("Converter", "")
                        preferred_tokenizer_name = f"{converter_base_name}Tokenizer"

                        # Try preferred tokenizer first
                        candidate_names = [preferred_tokenizer_name]
                        # Then try all other tokenizers with the same converter
                        for tokenizer_name, tokenizer_converter in SLOW_TO_FAST_CONVERTERS.items():
                            if tokenizer_converter is converter_class and tokenizer_name != lookup_name:
                                if tokenizer_name not in candidate_names:
                                    candidate_names.append(tokenizer_name)

                        # Try to import one of the candidate tokenizers
                        for candidate_name in candidate_names:
                            if candidate_name in self._class_to_module:
                                try:
                                    alias_module = self._get_module(self._class_to_module[candidate_name])
                                    base_tokenizer_class = getattr(alias_module, candidate_name)
                                    value = base_tokenizer_class

                                    # Cache both names for future imports
                                    setattr(self, candidate_name, base_tokenizer_class)
                                    if lookup_name != candidate_name:
                                        setattr(self, lookup_name, value)
                                    setattr(self, name, value)
                                    break
                                except Exception:
                                    # If this candidate fails, try the next one
                                    continue
                            else:
                                # Candidate not in _class_to_module - might need recursive resolution
                                # Try importing it directly to trigger lazy loading
                                try:
                                    # Try to get it from transformers module to trigger lazy loading
                                    transformers_module = sys.modules.get("transformers")
                                    if transformers_module and hasattr(transformers_module, candidate_name):
                                        base_tokenizer_class = getattr(transformers_module, candidate_name)
                                        value = base_tokenizer_class

                                        if lookup_name != candidate_name:
                                            setattr(self, lookup_name, value)
                                        setattr(self, name, value)
                                        break
                                except Exception:
                                    continue
                except (ImportError, AttributeError):
                    pass

            if value is None:
                for key, values in self._explicit_import_shortcut.items():
                    if name in values:
                        value = self._get_module(key)
                        break

            if value is None:
                raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""


def direct_transformers_import(path: str, file="__init__.py") -> ModuleType:
    """Imports transformers directly

    Args:
        path (`str`): The path to the source file
        file (`str`, *optional*): The file to join with the path. Defaults to "__init__.py".

    Returns:
        `ModuleType`: The resulting imported module
    """
    name = "transformers"
    location = os.path.join(path, file)
    spec = importlib.util.spec_from_file_location(name, location, submodule_search_locations=[path])
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module = sys.modules[name]
        return module
    raise ImportError(f"Could not load module {name} from {location}")


class VersionComparison(Enum):
    EQUAL = operator.eq
    NOT_EQUAL = operator.ne
    GREATER_THAN = operator.gt
    LESS_THAN = operator.lt
    GREATER_THAN_OR_EQUAL = operator.ge
    LESS_THAN_OR_EQUAL = operator.le

    @staticmethod
    def from_string(version_string: str) -> "VersionComparison":
        string_to_operator = {
            "=": VersionComparison.EQUAL,
            "==": VersionComparison.EQUAL,
            "!=": VersionComparison.NOT_EQUAL,
            ">": VersionComparison.GREATER_THAN,
            "<": VersionComparison.LESS_THAN,
            ">=": VersionComparison.GREATER_THAN_OR_EQUAL,
            "<=": VersionComparison.LESS_THAN_OR_EQUAL,
        }

        return string_to_operator[version_string]


@lru_cache
def split_package_version(package_version_str) -> tuple[str, str, str]:
    pattern = r"([a-zA-Z0-9_-]+)([!<>=~]+)([0-9.]+)"
    match = re.match(pattern, package_version_str)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    else:
        raise ValueError(f"Invalid package version string: {package_version_str}")


class Backend:
    def __init__(self, backend_requirement: str):
        self.package_name, self.version_comparison, self.version = split_package_version(backend_requirement)

        if self.package_name not in BACKENDS_MAPPING:
            raise ValueError(
                f"Backends should be defined in the BACKENDS_MAPPING. Offending backend: {self.package_name}"
            )

    def get_installed_version(self) -> str:
        """Return the currently installed version of the backend"""
        is_available, current_version = _is_package_available(self.package_name, return_version=True)
        if not is_available:
            raise RuntimeError(f"Backend {self.package_name} is not available.")
        return current_version

    def is_satisfied(self) -> bool:
        return VersionComparison.from_string(self.version_comparison).value(
            version.parse(self.get_installed_version()), version.parse(self.version)
        )

    def __repr__(self) -> str:
        return f'Backend("{self.package_name}", {VersionComparison[self.version_comparison]}, "{self.version}")'

    @property
    def error_message(self):
        return (
            f"{{0}} requires the {self.package_name} library version {self.version_comparison}{self.version}. That"
            f" library was not found with this version in your environment."
        )


def requires(*, backends=()):
    """
    This decorator enables two things:
    - Attaching a `__backends` tuple to an object to see what are the necessary backends for it
      to execute correctly without instantiating it
    - The '@requires' string is used to dynamically import objects
    """

    if not isinstance(backends, tuple):
        raise TypeError("Backends should be a tuple.")

    applied_backends = []
    for backend in backends:
        if backend in BACKENDS_MAPPING:
            applied_backends.append(backend)
        else:
            if any(key in backend for key in ["=", "<", ">"]):
                applied_backends.append(Backend(backend))
            else:
                raise ValueError(f"Backend should be defined in the BACKENDS_MAPPING. Offending backend: {backend}")

    def inner_fn(fun):
        fun.__backends = applied_backends
        return fun

    return inner_fn


BASE_FILE_REQUIREMENTS = {
    lambda e: "modeling_" in e: ("torch",),
    lambda e: e.startswith("tokenization_") and e.endswith("_fast"): ("tokenizers",),
    lambda e: e.startswith("image_processing_") and e.endswith("_fast"): ("vision", "torch", "torchvision"),
    lambda e: e.startswith("image_processing_"): ("vision",),
    lambda e: e.startswith("video_processing_"): ("vision", "torch", "torchvision"),
}


def fetch__all__(file_content) -> list[str]:
    """
    Returns the content of the __all__ variable in the file content.
    Returns None if not defined, otherwise returns a list of strings.
    """

    if "__all__" not in file_content:
        return []

    start_index = None
    lines = file_content.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("__all__"):
            start_index = index

    # There is no line starting with `__all__`
    if start_index is None:
        return []

    lines = lines[start_index:]

    if not lines[0].startswith("__all__"):
        raise ValueError(
            "fetch__all__ accepts a list of lines, with the first line being the __all__ variable declaration"
        )

    # __all__ is defined on a single line
    if lines[0].endswith("]"):
        return [obj.strip("\"' ") for obj in lines[0].split("=")[1].strip(" []").split(",")]

    # __all__ is defined on multiple lines
    else:
        _all: list[str] = []
        for __all__line_index in range(1, len(lines)):
            if lines[__all__line_index].strip() == "]":
                return _all
            else:
                _all.append(lines[__all__line_index].strip("\"', "))

        return _all


@lru_cache
def create_import_structure_from_path(module_path):
    """
    This method takes the path to a file/a folder and returns the import structure.
    If a file is given, it will return the import structure of the parent folder.

    Import structures are designed to be digestible by `_LazyModule` objects. They are
    created from the __all__ definitions in each files as well as the `@require` decorators
    above methods and objects.

    The import structure allows explicit display of the required backends for a given object.
    These backends are specified in two ways:

    1. Through their `@require`, if they are exported with that decorator. This `@require` decorator
       accepts a `backend` tuple kwarg mentioning which backends are required to run this object.

    2. If an object is defined in a file with "default" backends, it will have, at a minimum, this
       backend specified. The default backends are defined according to the filename:

       - If a file is named like `modeling_*.py`, it will have a `torch` backend
       - If a file is named like `tokenization_*_fast.py`, it will have a `tokenizers` backend
       - If a file is named like `image_processing*_fast.py`, it will have a `torchvision` + `torch` backend

    Backends serve the purpose of displaying a clear error message to the user in case the backends are not installed.
    Should an object be imported without its required backends being in the environment, any attempt to use the
    object will raise an error mentioning which backend(s) should be added to the environment in order to use
    that object.

    Here's an example of an input import structure at the src.transformers.models level:

    {
        'albert': {
            frozenset(): {
                'configuration_albert': {'AlbertConfig'}
            },
            frozenset({'tokenizers'}): {
                'tokenization_albert_fast': {'AlbertTokenizer'}
            },
        },
        'align': {
            frozenset(): {
                'configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
                'processing_align': {'AlignProcessor'}
            },
        },
        'altclip': {
            frozenset(): {
                'configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
                'processing_altclip': {'AltCLIPProcessor'},
            }
        }
    }
    """
    import_structure = {}

    if os.path.isfile(module_path):
        module_path = os.path.dirname(module_path)

    directory = module_path
    adjacent_modules = []

    for f in os.listdir(module_path):
        if f != "__pycache__" and os.path.isdir(os.path.join(module_path, f)):
            import_structure[f] = create_import_structure_from_path(os.path.join(module_path, f))

        elif not os.path.isdir(os.path.join(directory, f)):
            adjacent_modules.append(f)

    # We're only taking a look at files different from __init__.py
    # We could theoretically require things directly from the __init__.py
    # files, but this is not supported at this time.
    if "__init__.py" in adjacent_modules:
        adjacent_modules.remove("__init__.py")

    # Modular files should not be imported
    def find_substring(substring, list_):
        return any(substring in x for x in list_)

    if find_substring("modular_", adjacent_modules) and find_substring("modeling_", adjacent_modules):
        adjacent_modules = [module for module in adjacent_modules if "modular_" not in module]

    module_requirements = {}
    for module_name in adjacent_modules:
        # Only modules ending in `.py` are accepted here.
        if not module_name.endswith(".py"):
            continue

        with open(os.path.join(directory, module_name), encoding="utf-8") as f:
            file_content = f.read()

        # Remove the .py suffix
        module_name = module_name[:-3]

        previous_line = ""
        previous_index = 0

        # Some files have some requirements by default.
        # For example, any file named `modeling_xxx.py`
        # should have torch as a required backend.
        base_requirements = ()
        for string_check, requirements in BASE_FILE_REQUIREMENTS.items():
            if string_check(module_name):
                base_requirements = requirements
                break

        # Objects that have a `@require` assigned to them will get exported
        # with the backends specified in the decorator as well as the file backends.
        exported_objects = set()
        if "@requires" in file_content:
            lines = file_content.split("\n")
            for index, line in enumerate(lines):
                # This allows exporting items with other decorators. We'll take a look
                # at the line that follows at the same indentation level.
                if line.startswith((" ", "\t", "@", ")")) and not line.startswith("@requires"):
                    continue

                # Skipping line enables putting whatever we want between the
                # export() call and the actual class/method definition.
                # This is what enables having # Copied from statements, docs, etc.
                skip_line = False

                if "@requires" in previous_line:
                    skip_line = False

                    # Backends are defined on the same line as export
                    if "backends" in previous_line:
                        backends_string = previous_line.split("backends=")[1].split("(")[1].split(")")[0]
                        backends = tuple(sorted([b.strip("'\",") for b in backends_string.split(", ") if b]))

                    # Backends are defined in the lines following export, for example such as:
                    # @export(
                    #     backends=(
                    #             "sentencepiece",
                    #             "torch",
                    #     )
                    # )
                    #
                    # or
                    #
                    # @export(
                    #     backends=(
                    #             "sentencepiece",
                    #     )
                    # )
                    elif "backends" in lines[previous_index + 1]:
                        backends = []
                        for backend_line in lines[previous_index:index]:
                            if "backends" in backend_line:
                                backend_line = backend_line.split("=")[1]
                            if '"' in backend_line or "'" in backend_line:
                                if ", " in backend_line:
                                    backends.extend(backend.strip("()\"', ") for backend in backend_line.split(", "))
                                else:
                                    backends.append(backend_line.strip("()\"', "))

                            # If the line is only a ')', then we reached the end of the backends and we break.
                            if backend_line.strip() == ")":
                                break
                        backends = tuple(backends)

                    # No backends are registered for export
                    else:
                        backends = ()

                    backends = frozenset(backends + base_requirements)
                    if backends not in module_requirements:
                        module_requirements[backends] = {}
                    if module_name not in module_requirements[backends]:
                        module_requirements[backends][module_name] = set()

                    if not line.startswith("class") and not line.startswith("def"):
                        skip_line = True
                    else:
                        start_index = 6 if line.startswith("class") else 4
                        object_name = line[start_index:].split("(")[0].strip(":")
                        module_requirements[backends][module_name].add(object_name)
                        exported_objects.add(object_name)

                if not skip_line:
                    previous_line = line
                    previous_index = index

        # All objects that are in __all__ should be exported by default.
        # These objects are exported with the file backends.
        if "__all__" in file_content:
            for _all_object in fetch__all__(file_content):
                if _all_object not in exported_objects:
                    backends = frozenset(base_requirements)
                    if backends not in module_requirements:
                        module_requirements[backends] = {}
                    if module_name not in module_requirements[backends]:
                        module_requirements[backends][module_name] = set()

                    module_requirements[backends][module_name].add(_all_object)

    import_structure = {**module_requirements, **import_structure}
    return import_structure


def spread_import_structure(nested_import_structure):
    """
    This method takes as input an unordered import structure and brings the required backends at the top-level,
    aggregating modules and objects under their required backends.

    Here's an example of an input import structure at the src.transformers.models level:

    {
        'albert': {
            frozenset(): {
                'configuration_albert': {'AlbertConfig'}
            },
            frozenset({'tokenizers'}): {
                'tokenization_albert_fast': {'AlbertTokenizer'}
            },
        },
        'align': {
            frozenset(): {
                'configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
                'processing_align': {'AlignProcessor'}
            },
        },
        'altclip': {
            frozenset(): {
                'configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
                'processing_altclip': {'AltCLIPProcessor'},
            }
        }
    }

    Here's an example of an output import structure at the src.transformers.models level:

    {
        frozenset({'tokenizers'}): {
            'albert.tokenization_albert_fast': {'AlbertTokenizer'}
        },
        frozenset(): {
            'albert.configuration_albert': {'AlbertConfig'},
            'align.processing_align': {'AlignProcessor'},
            'align.configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
            'altclip.configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
            'altclip.processing_altclip': {'AltCLIPProcessor'}
        }
    }

    """

    def propagate_frozenset(unordered_import_structure):
        frozenset_first_import_structure = {}
        for _key, _value in unordered_import_structure.items():
            # If the value is not a dict but a string, no need for custom manipulation
            if not isinstance(_value, dict):
                frozenset_first_import_structure[_key] = _value

            elif any(isinstance(v, frozenset) for v in _value):
                for k, v in _value.items():
                    if isinstance(k, frozenset):
                        # Here we want to switch around _key and k to propagate k upstream if it is a frozenset
                        if k not in frozenset_first_import_structure:
                            frozenset_first_import_structure[k] = {}
                        if _key not in frozenset_first_import_structure[k]:
                            frozenset_first_import_structure[k][_key] = {}

                        frozenset_first_import_structure[k][_key].update(v)

                    else:
                        # If k is not a frozenset, it means that the dictionary is not "level": some keys (top-level)
                        # are frozensets, whereas some are not -> frozenset keys are at an unknown depth-level of the
                        # dictionary.
                        #
                        # We recursively propagate the frozenset for this specific dictionary so that the frozensets
                        # are at the top-level when we handle them.
                        propagated_frozenset = propagate_frozenset({k: v})
                        for r_k, r_v in propagated_frozenset.items():
                            if isinstance(_key, frozenset):
                                if r_k not in frozenset_first_import_structure:
                                    frozenset_first_import_structure[r_k] = {}
                                if _key not in frozenset_first_import_structure[r_k]:
                                    frozenset_first_import_structure[r_k][_key] = {}

                                # _key is a frozenset -> we switch around the r_k and _key
                                frozenset_first_import_structure[r_k][_key].update(r_v)
                            else:
                                if _key not in frozenset_first_import_structure:
                                    frozenset_first_import_structure[_key] = {}
                                if r_k not in frozenset_first_import_structure[_key]:
                                    frozenset_first_import_structure[_key][r_k] = {}

                                # _key is not a frozenset -> we keep the order of r_k and _key
                                frozenset_first_import_structure[_key][r_k].update(r_v)

            else:
                frozenset_first_import_structure[_key] = propagate_frozenset(_value)

        return frozenset_first_import_structure

    def flatten_dict(_dict, previous_key=None):
        items = []
        for _key, _value in _dict.items():
            _key = f"{previous_key}.{_key}" if previous_key is not None else _key
            if isinstance(_value, dict):
                items.extend(flatten_dict(_value, _key).items())
            else:
                items.append((_key, _value))
        return dict(items)

    # The tuples contain the necessary backends. We want these first, so we propagate them up the
    # import structure.
    ordered_import_structure = nested_import_structure

    # 6 is a number that gives us sufficient depth to go through all files and foreseeable folder depths
    # while not taking too long to parse.
    for i in range(6):
        ordered_import_structure = propagate_frozenset(ordered_import_structure)

    # We then flatten the dict so that it references a module path.
    flattened_import_structure = {}
    for key, value in ordered_import_structure.copy().items():
        if isinstance(key, str):
            del ordered_import_structure[key]
        else:
            flattened_import_structure[key] = flatten_dict(value)

    return flattened_import_structure


@lru_cache
def define_import_structure(module_path: str, prefix: str | None = None) -> IMPORT_STRUCTURE_T:
    """
    This method takes a module_path as input and creates an import structure digestible by a _LazyModule.

    Here's an example of an output import structure at the src.transformers.models level:

    {
        frozenset({'tokenizers'}): {
            'albert.tokenization_albert_fast': {'AlbertTokenizer'}
        },
        frozenset(): {
            'albert.configuration_albert': {'AlbertConfig'},
            'align.processing_align': {'AlignProcessor'},
            'align.configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
            'altclip.configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
            'altclip.processing_altclip': {'AltCLIPProcessor'}
        }
    }

    The import structure is a dict defined with frozensets as keys, and dicts of strings to sets of objects.

    If `prefix` is not None, it will add that prefix to all keys in the returned dict.
    """
    import_structure = create_import_structure_from_path(module_path)
    spread_dict = spread_import_structure(import_structure)

    if prefix is None:
        return spread_dict
    else:
        spread_dict = {k: {f"{prefix}.{kk}": vv for kk, vv in v.items()} for k, v in spread_dict.items()}
        return spread_dict


def clear_import_cache() -> None:
    """
    Clear cached Transformers modules to allow reloading modified code.

    This is useful when actively developing/modifying Transformers code.
    """
    # Get all transformers modules
    transformers_modules = [mod_name for mod_name in sys.modules if mod_name.startswith("transformers.")]

    # Remove them from sys.modules
    for mod_name in transformers_modules:
        module = sys.modules[mod_name]
        # Clear _LazyModule caches if applicable
        if isinstance(module, _LazyModule):
            module._objects = {}  # Clear cached objects
        del sys.modules[mod_name]

    # Force reload main transformers module
    if "transformers" in sys.modules:
        main_module = sys.modules["transformers"]
        if isinstance(main_module, _LazyModule):
            main_module._objects = {}  # Clear cached objects
        importlib.reload(main_module)
