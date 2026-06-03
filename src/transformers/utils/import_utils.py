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
import warnings
from collections import OrderedDict
from enum import Enum
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Callable, Optional, Union

from packaging import version

from . import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # TODO: Once python 3.9 support is dropped, `importlib.metadata.packages_distributions()`
            # should be used here to map from package name to distribution names
            # e.g. PIL -> Pillow, Pillow-SIMD; quark -> amd-quark; onnxruntime -> onnxruntime-gpu.
            # `importlib.metadata.packages_distributions()` is not available in Python 3.9.

            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            elif pkg_name == "quark":
                # TODO: remove once `importlib.metadata.packages_distributions()` is supported.
                try:
                    package_version = importlib.metadata.version("amd-quark")
                except Exception:
                    package_exists = False
            elif pkg_name == "triton":
                try:
                    # import triton works for both linux and windows
                    package = importlib.import_module(pkg_name)
                    package_version = getattr(package, "__version__", "N/A")
                except Exception:
                    try:
                        package_version = importlib.metadata.version("pytorch-triton")  # pytorch-triton
                    except Exception:
                        package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

# Try to run a native pytorch job in an environment with TorchXLA installed by setting this value to 0.
USE_TORCH_XLA = os.environ.get("USE_TORCH_XLA", "1").upper()

FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()

# `transformers` requires `torch>=1.11` but this variable is exposed publicly, and we can't simply remove it.
# This is the version of torch required to run torch.fx features and torch.onnx with dictionary inputs.
TORCH_FX_REQUIRED_VERSION = version.parse("1.10")

ACCELERATE_MIN_VERSION = "0.26.0"
SCHEDULEFREE_MIN_VERSION = "1.2.6"
FSDP_MIN_VERSION = "1.12.0"
GGUF_MIN_VERSION = "0.10.0"
XLA_FSDPV2_MIN_VERSION = "2.2.0"
HQQ_MIN_VERSION = "0.2.1"
VPTQ_MIN_VERSION = "0.0.4"
TORCHAO_MIN_VERSION = "0.4.0"
AUTOROUND_MIN_VERSION = "0.5.0"
TRITON_MIN_VERSION = "1.0.0"

_accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)
_apex_available = _is_package_available("apex")
_apollo_torch_available = _is_package_available("apollo_torch")
_aqlm_available = _is_package_available("aqlm")
_vptq_available, _vptq_version = _is_package_available("vptq", return_version=True)
_av_available = importlib.util.find_spec("av") is not None
_decord_available = importlib.util.find_spec("decord") is not None
_torchcodec_available = importlib.util.find_spec("torchcodec") is not None
_libcst_available = _is_package_available("libcst")
_bitsandbytes_available = _is_package_available("bitsandbytes")
_eetq_available = _is_package_available("eetq")
_fbgemm_gpu_available = _is_package_available("fbgemm_gpu")
_galore_torch_available = _is_package_available("galore_torch")
_lomo_available = _is_package_available("lomo_optim")
_grokadamw_available = _is_package_available("grokadamw")
_schedulefree_available, _schedulefree_version = _is_package_available("schedulefree", return_version=True)
_torch_optimi_available = importlib.util.find_spec("optimi") is not None
# `importlib.metadata.version` doesn't work with `bs4` but `beautifulsoup4`. For `importlib.util.find_spec`, reversed.
_bs4_available = importlib.util.find_spec("bs4") is not None
_coloredlogs_available = _is_package_available("coloredlogs")
# `importlib.metadata.util` doesn't work with `opencv-python-headless`.
_cv2_available = importlib.util.find_spec("cv2") is not None
_yt_dlp_available = importlib.util.find_spec("yt_dlp") is not None
_datasets_available = _is_package_available("datasets")
_detectron2_available = _is_package_available("detectron2")
# We need to check `faiss`, `faiss-cpu` and `faiss-gpu`.
_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    _faiss_version = importlib.metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib.metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib.metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib.metadata.PackageNotFoundError:
        try:
            _faiss_version = importlib.metadata.version("faiss-gpu")
            logger.debug(f"Successfully imported faiss version {_faiss_version}")
        except importlib.metadata.PackageNotFoundError:
            _faiss_available = False
_ftfy_available = _is_package_available("ftfy")
_g2p_en_available = _is_package_available("g2p_en")
_hadamard_available = _is_package_available("fast_hadamard_transform")
_ipex_available, _ipex_version = _is_package_available("intel_extension_for_pytorch", return_version=True)
_jinja_available = _is_package_available("jinja2")
_kenlm_available = _is_package_available("kenlm")
_keras_nlp_available = _is_package_available("keras_nlp")
_levenshtein_available = _is_package_available("Levenshtein")
_librosa_available = _is_package_available("librosa")
_natten_available = _is_package_available("natten")
_nltk_available = _is_package_available("nltk")
_onnx_available = _is_package_available("onnx")
_openai_available = _is_package_available("openai")
_optimum_available = _is_package_available("optimum")
_auto_gptq_available = _is_package_available("auto_gptq")
_gptqmodel_available = _is_package_available("gptqmodel")
_auto_round_available, _auto_round_version = _is_package_available("auto_round", return_version=True)
# `importlib.metadata.version` doesn't work with `awq`
_auto_awq_available = importlib.util.find_spec("awq") is not None
_quark_available = _is_package_available("quark")
_fp_quant_available, _fp_quant_version = _is_package_available("fp_quant", return_version=True)
_qutlass_available, _qutlass_version = _is_package_available("qutlass", return_version=True)
_is_optimum_quanto_available = False
try:
    importlib.metadata.version("optimum_quanto")
    _is_optimum_quanto_available = True
except importlib.metadata.PackageNotFoundError:
    _is_optimum_quanto_available = False
# For compressed_tensors, only check spec to allow compressed_tensors-nightly package
_compressed_tensors_available = importlib.util.find_spec("compressed_tensors") is not None
_pandas_available = _is_package_available("pandas")
_peft_available = _is_package_available("peft")
_phonemizer_available = _is_package_available("phonemizer")
_uroman_available = _is_package_available("uroman")
_psutil_available = _is_package_available("psutil")
_py3nvml_available = _is_package_available("py3nvml")
_pyctcdecode_available = _is_package_available("pyctcdecode")
_pygments_available = _is_package_available("pygments")
_pytesseract_available = _is_package_available("pytesseract")
_pytest_available = _is_package_available("pytest")
_pytorch_quantization_available = _is_package_available("pytorch_quantization")
_rjieba_available = _is_package_available("rjieba")
_sacremoses_available = _is_package_available("sacremoses")
_safetensors_available = _is_package_available("safetensors")
_scipy_available = _is_package_available("scipy")
_sentencepiece_available = _is_package_available("sentencepiece")
_is_seqio_available = _is_package_available("seqio")
_is_gguf_available, _gguf_version = _is_package_available("gguf", return_version=True)
_sklearn_available = importlib.util.find_spec("sklearn") is not None
if _sklearn_available:
    try:
        importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        _sklearn_available = False
_smdistributed_available = importlib.util.find_spec("smdistributed") is not None
_soundfile_available = _is_package_available("soundfile")
_spacy_available = _is_package_available("spacy")
_sudachipy_available, _sudachipy_version = _is_package_available("sudachipy", return_version=True)
_tensorflow_probability_available = _is_package_available("tensorflow_probability")
_tensorflow_text_available = _is_package_available("tensorflow_text")
_tf2onnx_available = _is_package_available("tf2onnx")
_timm_available = _is_package_available("timm")
_tokenizers_available = _is_package_available("tokenizers")
_torchaudio_available = _is_package_available("torchaudio")
_torchao_available, _torchao_version = _is_package_available("torchao", return_version=True)
_torchdistx_available = _is_package_available("torchdistx")
_torchvision_available, _torchvision_version = _is_package_available("torchvision", return_version=True)
_mlx_available = _is_package_available("mlx")
_num2words_available = _is_package_available("num2words")
_hqq_available, _hqq_version = _is_package_available("hqq", return_version=True)
_tiktoken_available = _is_package_available("tiktoken")
_blobfile_available = _is_package_available("blobfile")
_liger_kernel_available = _is_package_available("liger_kernel")
_spqr_available = _is_package_available("spqr_quant")
_rich_available = _is_package_available("rich")
_kernels_available = _is_package_available("kernels")
_matplotlib_available = _is_package_available("matplotlib")
_mistral_common_available = _is_package_available("mistral_common")
_triton_available, _triton_version = _is_package_available("triton", return_version=True)

_torch_version = "N/A"
_torch_available = False
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available, _torch_version = _is_package_available("torch", return_version=True)
    if _torch_available:
        _torch_available = version.parse(_torch_version) >= version.parse("2.1.0")
        if not _torch_available:
            logger.warning(f"Disabling PyTorch because PyTorch >= 2.1 is required but found {_torch_version}")
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


_tf_version = "N/A"
_tf_available = False
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = True
else:
    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        # Note: _is_package_available("tensorflow") fails for tensorflow-cpu. Please test any changes to the line below
        # with tensorflow-cpu to make sure it still works!
        _tf_available = importlib.util.find_spec("tensorflow") is not None
        if _tf_available:
            candidates = (
                "tensorflow",
                "tensorflow-cpu",
                "tensorflow-gpu",
                "tf-nightly",
                "tf-nightly-cpu",
                "tf-nightly-gpu",
                "tf-nightly-rocm",
                "intel-tensorflow",
                "intel-tensorflow-avx512",
                "tensorflow-rocm",
                "tensorflow-macos",
                "tensorflow-aarch64",
            )
            _tf_version = None
            # For the metadata, we have to look for both tensorflow and tensorflow-cpu
            for pkg in candidates:
                try:
                    _tf_version = importlib.metadata.version(pkg)
                    break
                except importlib.metadata.PackageNotFoundError:
                    pass
            _tf_available = _tf_version is not None
        if _tf_available:
            if version.parse(_tf_version) < version.parse("2"):
                logger.info(
                    f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum."
                )
                _tf_available = False
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")


_essentia_available = importlib.util.find_spec("essentia") is not None
try:
    _essentia_version = importlib.metadata.version("essentia")
    logger.debug(f"Successfully imported essentia version {_essentia_version}")
except importlib.metadata.PackageNotFoundError:
    _essentia_version = False


_pydantic_available = importlib.util.find_spec("pydantic") is not None
try:
    _pydantic_version = importlib.metadata.version("pydantic")
    logger.debug(f"Successfully imported pydantic version {_pydantic_version}")
except importlib.metadata.PackageNotFoundError:
    _pydantic_available = False


_fastapi_available = importlib.util.find_spec("fastapi") is not None
try:
    _fastapi_version = importlib.metadata.version("fastapi")
    logger.debug(f"Successfully imported pydantic version {_fastapi_version}")
except importlib.metadata.PackageNotFoundError:
    _fastapi_available = False


_uvicorn_available = importlib.util.find_spec("uvicorn") is not None
try:
    _uvicorn_version = importlib.metadata.version("uvicorn")
    logger.debug(f"Successfully imported pydantic version {_uvicorn_version}")
except importlib.metadata.PackageNotFoundError:
    _uvicorn_available = False


_pretty_midi_available = importlib.util.find_spec("pretty_midi") is not None
try:
    _pretty_midi_version = importlib.metadata.version("pretty_midi")
    logger.debug(f"Successfully imported pretty_midi version {_pretty_midi_version}")
except importlib.metadata.PackageNotFoundError:
    _pretty_midi_available = False


ccl_version = "N/A"
_is_ccl_available = (
    importlib.util.find_spec("torch_ccl") is not None
    or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
)
try:
    ccl_version = importlib.metadata.version("oneccl_bind_pt")
    logger.debug(f"Detected oneccl_bind_pt version {ccl_version}")
except importlib.metadata.PackageNotFoundError:
    _is_ccl_available = False


_flax_available = False
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available, _flax_version = _is_package_available("flax", return_version=True)
    if _flax_available:
        _jax_available, _jax_version = _is_package_available("jax", return_version=True)
        if _jax_available:
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        else:
            _flax_available = _jax_available = False
            _jax_version = _flax_version = "N/A"


_torch_xla_available = False
if USE_TORCH_XLA in ENV_VARS_TRUE_VALUES:
    _torch_xla_available, _torch_xla_version = _is_package_available("torch_xla", return_version=True)
    if _torch_xla_available:
        logger.info(f"Torch XLA version {_torch_xla_version} available.")


def is_kenlm_available() -> Union[tuple[bool, str], bool]:
    return _kenlm_available


def is_kernels_available() -> Union[tuple[bool, str], bool]:
    return _kernels_available


def is_cv2_available() -> Union[tuple[bool, str], bool]:
    return _cv2_available


def is_yt_dlp_available() -> Union[tuple[bool, str], bool]:
    return _yt_dlp_available


def is_torch_available() -> Union[tuple[bool, str], bool]:
    return _torch_available


def is_libcst_available() -> Union[tuple[bool, str], bool]:
    return _libcst_available


def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION) -> bool:
    return _accelerate_available and version.parse(_accelerate_version) >= version.parse(min_version)


def is_torch_accelerator_available() -> bool:
    if is_torch_available():
        import torch

        return hasattr(torch, "accelerator")

    return False


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


def is_triton_available(min_version: str = TRITON_MIN_VERSION) -> bool:
    return _triton_available and version.parse(_triton_version) >= version.parse(min_version)


def is_hadamard_available() -> Union[tuple[bool, str], bool]:
    return _hadamard_available


def is_hqq_available(min_version: str = HQQ_MIN_VERSION) -> bool:
    return _hqq_available and version.parse(_hqq_version) >= version.parse(min_version)


def is_pygments_available() -> Union[tuple[bool, str], bool]:
    return _pygments_available


def get_torch_version() -> str:
    return _torch_version


def get_torch_major_and_minor_version() -> str:
    if _torch_version == "N/A":
        return "N/A"
    parsed_version = version.parse(_torch_version)
    return str(parsed_version.major) + "." + str(parsed_version.minor)


def is_torch_sdpa_available():
    # Mostly retained for backward compatibility in remote code, since sdpa works correctly on all torch versions >= 2.2
    if not is_torch_available() or _torch_version == "N/A":
        return False
    return True


def is_torch_flex_attn_available() -> bool:
    if not is_torch_available() or _torch_version == "N/A":
        return False

    # TODO check if some bugs cause push backs on the exact version
    # NOTE: We require torch>=2.5.0 as it is the first release
    return version.parse(_torch_version) >= version.parse("2.5.0")


def is_torchvision_available() -> bool:
    return _torchvision_available


def is_torchvision_v2_available() -> bool:
    return is_torchvision_available()


def is_galore_torch_available() -> Union[tuple[bool, str], bool]:
    return _galore_torch_available


def is_apollo_torch_available() -> Union[tuple[bool, str], bool]:
    return _apollo_torch_available


def is_torch_optimi_available() -> Union[tuple[bool, str], bool]:
    return _torch_optimi_available


def is_lomo_available() -> Union[tuple[bool, str], bool]:
    return _lomo_available


def is_grokadamw_available() -> Union[tuple[bool, str], bool]:
    return _grokadamw_available


def is_schedulefree_available(min_version: str = SCHEDULEFREE_MIN_VERSION) -> bool:
    return _schedulefree_available and version.parse(_schedulefree_version) >= version.parse(min_version)


def is_pyctcdecode_available() -> Union[tuple[bool, str], bool]:
    return _pyctcdecode_available


def is_librosa_available() -> Union[tuple[bool, str], bool]:
    return _librosa_available


def is_essentia_available() -> Union[tuple[bool, str], bool]:
    return _essentia_available


def is_pydantic_available() -> Union[tuple[bool, str], bool]:
    return _pydantic_available


def is_fastapi_available() -> Union[tuple[bool, str], bool]:
    return _fastapi_available


def is_uvicorn_available() -> Union[tuple[bool, str], bool]:
    return _uvicorn_available


def is_openai_available() -> Union[tuple[bool, str], bool]:
    return _openai_available


def is_pretty_midi_available() -> Union[tuple[bool, str], bool]:
    return _pretty_midi_available


def is_torch_cuda_available() -> bool:
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def is_cuda_platform() -> bool:
    if is_torch_available():
        import torch

        return torch.version.cuda is not None
    else:
        return False


def is_rocm_platform() -> bool:
    if is_torch_available():
        import torch

        return torch.version.hip is not None
    else:
        return False


def is_mamba_ssm_available() -> Union[tuple[bool, str], bool]:
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False
        else:
            return _is_package_available("mamba_ssm")
    return False


def is_mamba_2_ssm_available() -> bool:
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False
        else:
            if _is_package_available("mamba_ssm"):
                import mamba_ssm

                if version.parse(mamba_ssm.__version__) >= version.parse("2.0.4"):
                    return True
    return False


def is_flash_linear_attention_available():
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False

        try:
            import fla

            if version.parse(fla.__version__) >= version.parse("0.2.2"):
                return True
        except Exception:
            pass
    return False


def is_causal_conv1d_available() -> Union[tuple[bool, str], bool]:
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False
        return _is_package_available("causal_conv1d")
    return False


def is_xlstm_available() -> Union[tuple[bool, str], bool]:
    if is_torch_available():
        return _is_package_available("xlstm")
    return False


def is_mambapy_available() -> Union[tuple[bool, str], bool]:
    if is_torch_available():
        return _is_package_available("mambapy")
    return False


def is_torch_mps_available(min_version: Optional[str] = None) -> bool:
    if is_torch_available():
        import torch

        if hasattr(torch.backends, "mps"):
            backend_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            if min_version is not None:
                flag = version.parse(_torch_version) >= version.parse(min_version)
                backend_available = backend_available and flag
            return backend_available
    return False


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
        return torch.npu.is_bf16_supported()
    if is_torch_mps_available():
        # Note: Emulated in software by Metal using fp32 for hardware without native support (like M1/M2)
        return torch.backends.mps.is_macos_or_newer(14, 0)
    if is_torch_musa_available():
        return torch.musa.is_bf16_supported()
    return False


def is_torch_bf16_cpu_available() -> Union[tuple[bool, str], bool]:
    return is_torch_available()


def is_torch_bf16_available() -> bool:
    # the original bf16 check was for gpu only, but later a cpu/bf16 combo has emerged so this util
    # has become ambiguous and therefore deprecated
    warnings.warn(
        "The util is_torch_bf16_available is deprecated, please use is_torch_bf16_gpu_available "
        "or is_torch_bf16_cpu_available instead according to whether it's used with cpu or gpu",
        FutureWarning,
    )
    return is_torch_bf16_gpu_available()


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

    except:  # noqa: E722
        # TODO: more precise exception matching, if possible.
        # most backends should return `RuntimeError` however this is not guaranteed.
        return False

    return True


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
    except:  # noqa: E722
        # TODO: more precise exception matching, if possible.
        # most backends should return `RuntimeError` however this is not guaranteed.
        return False

    return True


def is_torch_tf32_available() -> bool:
    if not is_torch_available():
        return False

    import torch

    if is_torch_musa_available():
        device_info = torch.musa.get_device_properties(torch.musa.current_device())
        if f"{device_info.major}{device_info.minor}" >= "22":
            return True
        return False
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    return True


def is_torch_fx_available() -> Union[tuple[bool, str], bool]:
    return is_torch_available()


def is_peft_available() -> Union[tuple[bool, str], bool]:
    return _peft_available


def is_bs4_available() -> Union[tuple[bool, str], bool]:
    return _bs4_available


def is_tf_available() -> bool:
    return _tf_available


def is_coloredlogs_available() -> Union[tuple[bool, str], bool]:
    return _coloredlogs_available


def is_tf2onnx_available() -> Union[tuple[bool, str], bool]:
    return _tf2onnx_available


def is_onnx_available() -> Union[tuple[bool, str], bool]:
    return _onnx_available


def is_flax_available() -> bool:
    return _flax_available


def is_flute_available() -> bool:
    try:
        return importlib.util.find_spec("flute") is not None and importlib.metadata.version("flute-kernel") >= "0.4.1"
    except importlib.metadata.PackageNotFoundError:
        return False


def is_ftfy_available() -> Union[tuple[bool, str], bool]:
    return _ftfy_available


def is_g2p_en_available() -> Union[tuple[bool, str], bool]:
    return _g2p_en_available


@lru_cache
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False) -> bool:
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."

    if not _torch_xla_available:
        return False

    import torch_xla

    if check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == "TPU"

    return True


@lru_cache
def is_torch_neuroncore_available(check_device=True) -> bool:
    if importlib.util.find_spec("torch_neuronx") is not None:
        return is_torch_xla_available()
    return False


@lru_cache
def is_torch_npu_available(check_device=False) -> bool:
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if not _torch_available or importlib.util.find_spec("torch_npu") is None:
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
def is_torch_mlu_available() -> bool:
    """
    Checks if `mlu` is available via an `cndev-based` check which won't trigger the drivers and leave mlu
    uninitialized.
    """
    if not _torch_available or importlib.util.find_spec("torch_mlu") is None:
        return False

    import torch
    import torch_mlu  # noqa: F401

    pytorch_cndev_based_mlu_check_previous_value = os.environ.get("PYTORCH_CNDEV_BASED_MLU_CHECK")
    try:
        os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = str(1)
        available = torch.mlu.is_available()
    finally:
        if pytorch_cndev_based_mlu_check_previous_value:
            os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = pytorch_cndev_based_mlu_check_previous_value
        else:
            os.environ.pop("PYTORCH_CNDEV_BASED_MLU_CHECK", None)

    return available


@lru_cache
def is_torch_musa_available(check_device=False) -> bool:
    "Checks if `torch_musa` is installed and potentially if a MUSA is in the environment"
    if not _torch_available or importlib.util.find_spec("torch_musa") is None:
        return False

    import torch
    import torch_musa  # noqa: F401

    torch_musa_min_version = "0.33.0"
    if _accelerate_available and version.parse(_accelerate_version) < version.parse(torch_musa_min_version):
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no MUSA is found
            _ = torch.musa.device_count()
            return torch.musa.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "musa") and torch.musa.is_available()


@lru_cache
def is_torch_hpu_available() -> bool:
    "Checks if `torch.hpu` is available and potentially if a HPU is in the environment"
    if (
        not _torch_available
        or importlib.util.find_spec("habana_frameworks") is None
        or importlib.util.find_spec("habana_frameworks.torch") is None
    ):
        return False

    torch_hpu_min_accelerate_version = "1.5.0"
    if _accelerate_available and version.parse(_accelerate_version) < version.parse(torch_hpu_min_accelerate_version):
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

    def patched_take_along_dim(
        input: torch.Tensor, indices: torch.LongTensor, dim: Optional[int] = None
    ) -> torch.Tensor:
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
def is_habana_gaudi1() -> bool:
    if not is_torch_hpu_available():
        return False

    import habana_frameworks.torch.utils.experimental as htexp

    # Check if the device is Gaudi1 (vs Gaudi2, Gaudi3)
    return htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi


def is_torchdynamo_available() -> Union[tuple[bool, str], bool]:
    return is_torch_available()


def is_torch_compile_available() -> Union[tuple[bool, str], bool]:
    return is_torch_available()


def is_torchdynamo_compiling() -> Union[tuple[bool, str], bool]:
    if not is_torch_available():
        return False

    # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622)
    # hence rather relying on `torch.compiler.is_compiling()` when possible (torch>=2.3)
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        try:
            import torch._dynamo as dynamo

            return dynamo.is_compiling()
        except Exception:
            return False


def is_torchdynamo_exporting() -> bool:
    if not is_torch_available():
        return False

    try:
        import torch

        return torch.compiler.is_exporting()
    except Exception:
        try:
            import torch._dynamo as dynamo

            return dynamo.is_exporting()
        except Exception:
            return False


def is_torch_tensorrt_fx_available() -> bool:
    if importlib.util.find_spec("torch_tensorrt") is None:
        return False
    return importlib.util.find_spec("torch_tensorrt.fx") is not None


def is_datasets_available() -> Union[tuple[bool, str], bool]:
    return _datasets_available


def is_detectron2_available() -> Union[tuple[bool, str], bool]:
    return _detectron2_available


def is_rjieba_available() -> Union[tuple[bool, str], bool]:
    return _rjieba_available


def is_psutil_available() -> Union[tuple[bool, str], bool]:
    return _psutil_available


def is_py3nvml_available() -> Union[tuple[bool, str], bool]:
    return _py3nvml_available


def is_sacremoses_available() -> Union[tuple[bool, str], bool]:
    return _sacremoses_available


def is_apex_available() -> Union[tuple[bool, str], bool]:
    return _apex_available


def is_aqlm_available() -> Union[tuple[bool, str], bool]:
    return _aqlm_available


def is_vptq_available(min_version: str = VPTQ_MIN_VERSION) -> bool:
    return _vptq_available and version.parse(_vptq_version) >= version.parse(min_version)


def is_av_available() -> bool:
    return _av_available


def is_decord_available() -> bool:
    return _decord_available


def is_torchcodec_available() -> bool:
    return _torchcodec_available


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


def is_ipex_available(min_version: str = "") -> bool:
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    if not is_torch_available() or not _ipex_available:
        return False

    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        logger.warning(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    if min_version:
        return version.parse(_ipex_version) >= version.parse(min_version)
    return True


@lru_cache
def is_torch_xpu_available(check_device: bool = False) -> bool:
    """
    Checks if XPU acceleration is available either via native PyTorch (>=2.6),
    `intel_extension_for_pytorch` or via stock PyTorch (>=2.4) and potentially
    if a XPU is in the environment.
    """
    if not is_torch_available():
        return False

    torch_version = version.parse(_torch_version)
    if torch_version.major == 2 and torch_version.minor < 6:
        if is_ipex_available():
            import intel_extension_for_pytorch  # noqa: F401
        elif torch_version.major == 2 and torch_version.minor < 4:
            return False

    import torch

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU  is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache
def is_bitsandbytes_available(check_library_only: bool = False) -> bool:
    if not _bitsandbytes_available:
        return False

    if check_library_only:
        return True

    if not is_torch_available():
        return False

    import torch

    # `bitsandbytes` versions older than 0.43.1 eagerly require CUDA at import time,
    # so those versions of the library are practically only available when CUDA is too.
    if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.1"):
        return torch.cuda.is_available()

    # Newer versions of `bitsandbytes` can be imported on systems without CUDA.
    return True


def is_bitsandbytes_multi_backend_available() -> bool:
    if not is_bitsandbytes_available():
        return False

    import bitsandbytes as bnb

    return "multi_backend" in getattr(bnb, "features", set())


def is_flash_attn_2_available() -> bool:
    if not is_torch_available():
        return False

    if not _is_package_available("flash_attn"):
        return False

    # Let's add an extra check to see if cuda is available
    import torch

    if not (torch.cuda.is_available() or is_torch_mlu_available()):
        return False

    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    elif torch.version.hip:
        # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    elif is_torch_mlu_available():
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.3.3")
    else:
        return False


@lru_cache
def is_flash_attn_3_available() -> bool:
    if not is_torch_available():
        return False

    if not _is_package_available("flash_attn_3"):
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    # TODO: Check for a minimum version when FA3 is stable
    # return version.parse(importlib.metadata.version("flash_attn_3")) >= version.parse("3.0.0")

    return True


@lru_cache
def is_flash_attn_greater_or_equal_2_10() -> bool:
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")


@lru_cache
def is_flash_attn_greater_or_equal(library_version: str) -> bool:
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(library_version)


@lru_cache
def is_torch_greater_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is greater than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not _is_package_available("torch"):
        return False

    if accept_dev:
        return version.parse(version.parse(importlib.metadata.version("torch")).base_version) >= version.parse(
            library_version
        )
    else:
        return version.parse(importlib.metadata.version("torch")) >= version.parse(library_version)


@lru_cache
def is_torch_less_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is less than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not _is_package_available("torch"):
        return False

    if accept_dev:
        return version.parse(version.parse(importlib.metadata.version("torch")).base_version) <= version.parse(
            library_version
        )
    else:
        return version.parse(importlib.metadata.version("torch")) <= version.parse(library_version)


@lru_cache
def is_huggingface_hub_greater_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    if not _is_package_available("huggingface_hub"):
        return False

    if accept_dev:
        return version.parse(
            version.parse(importlib.metadata.version("huggingface_hub")).base_version
        ) >= version.parse(library_version)
    else:
        return version.parse(importlib.metadata.version("huggingface_hub")) >= version.parse(library_version)


@lru_cache
def is_quanto_greater(library_version: str, accept_dev: bool = False) -> bool:
    """
    Accepts a library version and returns True if the current version of the library is greater than or equal to the
    given version. If `accept_dev` is True, it will also accept development versions (e.g. 2.7.0.dev20250320 matches
    2.7.0).
    """
    if not _is_package_available("optimum.quanto"):
        return False

    if accept_dev:
        return version.parse(version.parse(importlib.metadata.version("optimum-quanto")).base_version) > version.parse(
            library_version
        )
    else:
        return version.parse(importlib.metadata.version("optimum-quanto")) > version.parse(library_version)


def is_torchdistx_available():
    return _torchdistx_available


def is_faiss_available() -> bool:
    return _faiss_available


def is_scipy_available() -> Union[tuple[bool, str], bool]:
    return _scipy_available


def is_sklearn_available() -> Union[tuple[bool, str], bool]:
    return _sklearn_available


def is_sentencepiece_available() -> Union[tuple[bool, str], bool]:
    return _sentencepiece_available


def is_seqio_available() -> Union[tuple[bool, str], bool]:
    return _is_seqio_available


def is_gguf_available(min_version: str = GGUF_MIN_VERSION) -> bool:
    return _is_gguf_available and version.parse(_gguf_version) >= version.parse(min_version)


def is_protobuf_available() -> bool:
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_fsdp_available(min_version: str = FSDP_MIN_VERSION) -> bool:
    return is_torch_available() and version.parse(_torch_version) >= version.parse(min_version)


def is_optimum_available() -> Union[tuple[bool, str], bool]:
    return _optimum_available


def is_auto_awq_available() -> bool:
    return _auto_awq_available


def is_auto_round_available(min_version: str = AUTOROUND_MIN_VERSION) -> bool:
    return _auto_round_available and version.parse(_auto_round_version) >= version.parse(min_version)


def is_optimum_quanto_available():
    # `importlib.metadata.version` doesn't work with `optimum.quanto`, need to put `optimum_quanto`
    return _is_optimum_quanto_available


def is_quark_available() -> Union[tuple[bool, str], bool]:
    return _quark_available


def is_fp_quant_available():
    return _fp_quant_available and version.parse(_fp_quant_version) >= version.parse("0.2.0")


def is_qutlass_available():
    return _qutlass_available and version.parse(_qutlass_version) >= version.parse("0.1.0")


def is_compressed_tensors_available() -> bool:
    return _compressed_tensors_available


def is_auto_gptq_available() -> Union[tuple[bool, str], bool]:
    return _auto_gptq_available


def is_gptqmodel_available() -> Union[tuple[bool, str], bool]:
    return _gptqmodel_available


def is_eetq_available() -> Union[tuple[bool, str], bool]:
    return _eetq_available


def is_fbgemm_gpu_available() -> Union[tuple[bool, str], bool]:
    return _fbgemm_gpu_available


def is_levenshtein_available() -> Union[tuple[bool, str], bool]:
    return _levenshtein_available


def is_optimum_neuron_available() -> Union[tuple[bool, str], bool]:
    return _optimum_available and _is_package_available("optimum.neuron")


def is_safetensors_available() -> Union[tuple[bool, str], bool]:
    return _safetensors_available


def is_tokenizers_available() -> Union[tuple[bool, str], bool]:
    return _tokenizers_available


@lru_cache
def is_vision_available() -> bool:
    _pil_available = importlib.util.find_spec("PIL") is not None
    if _pil_available:
        try:
            package_version = importlib.metadata.version("Pillow")
        except importlib.metadata.PackageNotFoundError:
            try:
                package_version = importlib.metadata.version("Pillow-SIMD")
            except importlib.metadata.PackageNotFoundError:
                return False
        logger.debug(f"Detected PIL version {package_version}")
    return _pil_available


def is_pytesseract_available() -> Union[tuple[bool, str], bool]:
    return _pytesseract_available


def is_pytest_available() -> Union[tuple[bool, str], bool]:
    return _pytest_available


def is_spacy_available() -> Union[tuple[bool, str], bool]:
    return _spacy_available


def is_tensorflow_text_available() -> Union[tuple[bool, str], bool]:
    return is_tf_available() and _tensorflow_text_available


def is_keras_nlp_available() -> Union[tuple[bool, str], bool]:
    return is_tensorflow_text_available() and _keras_nlp_available


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


def is_pytorch_quantization_available() -> Union[tuple[bool, str], bool]:
    return _pytorch_quantization_available


def is_tensorflow_probability_available() -> Union[tuple[bool, str], bool]:
    return _tensorflow_probability_available


def is_pandas_available() -> Union[tuple[bool, str], bool]:
    return _pandas_available


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
    return _smdistributed_available


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
    return _smdistributed_available


def is_training_run_on_sagemaker() -> bool:
    return "SAGEMAKER_JOB_NAME" in os.environ


def is_soundfile_available() -> Union[tuple[bool, str], bool]:
    return _soundfile_available


def is_timm_available() -> Union[tuple[bool, str], bool]:
    return _timm_available


def is_natten_available() -> Union[tuple[bool, str], bool]:
    return _natten_available


def is_nltk_available() -> Union[tuple[bool, str], bool]:
    return _nltk_available


def is_torchaudio_available() -> Union[tuple[bool, str], bool]:
    return _torchaudio_available


def is_torchao_available(min_version: str = TORCHAO_MIN_VERSION) -> bool:
    return _torchao_available and version.parse(_torchao_version) >= version.parse(min_version)


def is_speech_available() -> Union[tuple[bool, str], bool]:
    # For now this depends on torchaudio but the exact dependency might evolve in the future.
    return _torchaudio_available


def is_spqr_available() -> Union[tuple[bool, str], bool]:
    return _spqr_available


def is_phonemizer_available() -> Union[tuple[bool, str], bool]:
    return _phonemizer_available


def is_uroman_available() -> Union[tuple[bool, str], bool]:
    return _uroman_available


def torch_only_method(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if not _torch_available:
            raise ImportError(
                "You need to install pytorch to use this method or class, "
                "or activate it with environment variables USE_TORCH=1 and USE_TF=0."
            )
        else:
            return fn(*args, **kwargs)

    return wrapper


def is_ccl_available() -> bool:
    return _is_ccl_available


def is_sudachi_available() -> bool:
    return _sudachipy_available


def get_sudachi_version() -> bool:
    return _sudachipy_version


def is_sudachi_projection_available() -> bool:
    if not is_sudachi_available():
        return False

    # NOTE: We require sudachipy>=0.6.8 to use projection option in sudachi_kwargs for the constructor of BertJapaneseTokenizer.
    # - `projection` option is not supported in sudachipy<0.6.8, see https://github.com/WorksApplications/sudachi.rs/issues/230
    return version.parse(_sudachipy_version) >= version.parse("0.6.8")


def is_jumanpp_available() -> bool:
    return (importlib.util.find_spec("rhoknp") is not None) and (shutil.which("jumanpp") is not None)


def is_cython_available() -> bool:
    return importlib.util.find_spec("pyximport") is not None


def is_jinja_available() -> Union[tuple[bool, str], bool]:
    return _jinja_available


def is_mlx_available() -> Union[tuple[bool, str], bool]:
    return _mlx_available


def is_num2words_available() -> Union[tuple[bool, str], bool]:
    return _num2words_available


def is_tiktoken_available() -> Union[tuple[bool, str], bool]:
    return _tiktoken_available and _blobfile_available


def is_liger_kernel_available() -> bool:
    if not _liger_kernel_available:
        return False

    return version.parse(importlib.metadata.version("liger_kernel")) >= version.parse("0.3.0")


def is_rich_available() -> Union[tuple[bool, str], bool]:
    return _rich_available


def is_matplotlib_available() -> Union[tuple[bool, str], bool]:
    return _matplotlib_available


def is_mistral_common_available() -> Union[tuple[bool, str], bool]:
    return _mistral_common_available


def check_torch_load_is_safe() -> None:
    if not is_torch_greater_or_equal("2.6"):
        raise ValueError(
            "Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users "
            "to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply "
            "when loading files with safetensors."
            "\nSee the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434"
        )


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
PYTORCH_IMPORT_ERROR_WITH_TF = """
{0} requires the PyTorch library but it was not found in your environment.
However, we were able to find a TensorFlow installation. TensorFlow classes begin
with "TF", but are otherwise identically named to our PyTorch classes. This
means that the TF equivalent of the class you tried to import would be "TF{0}".
If you want to use TensorFlow, please use TF classes instead!

If you really do want to use PyTorch please go to
https://pytorch.org/get-started/locally/ and follow the instructions that
match your environment.
"""

# docstyle-ignore
TF_IMPORT_ERROR_WITH_PYTORCH = """
{0} requires the TensorFlow library but it was not found in your environment.
However, we were able to find a PyTorch installation. PyTorch classes do not begin
with "TF", but are otherwise identically named to our TF classes.
If you want to use PyTorch, please use those classes instead!

If you really do want to use TensorFlow, please follow the instructions on the
installation page https://www.tensorflow.org/install that match your environment.
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
TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Check out the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Check out the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Check out the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Check out the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
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
TENSORFLOW_PROBABILITY_IMPORT_ERROR = """
{0} requires the tensorflow_probability library but it was not found in your environment. You can install it with pip as
explained here: https://github.com/tensorflow/probability. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TENSORFLOW_TEXT_IMPORT_ERROR = """
{0} requires the tensorflow_text library but it was not found in your environment. You can install it with pip as
explained here: https://www.tensorflow.org/text/guide/tf_text_intro.
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
KERAS_NLP_IMPORT_ERROR = """
{0} requires the keras_nlp library but it was not found in your environment. You can install it with pip.
Please note that you may need to restart your runtime after installation.
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
CCL_IMPORT_ERROR = """
{0} requires the torch ccl library but it was not found in your environment. You can install it with pip:
`pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable`
Please note that you may need to restart your runtime after installation.
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
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
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
        ("tensorflow_probability", (is_tensorflow_probability_available, TENSORFLOW_PROBABILITY_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("tensorflow_text", (is_tensorflow_text_available, TENSORFLOW_TEXT_IMPORT_ERROR)),
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
        ("oneccl_bind_pt", (is_ccl_available, CCL_IMPORT_ERROR)),
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        ("rjieba", (is_rjieba_available, RJIEBA_IMPORT_ERROR)),
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        ("jinja", (is_jinja_available, JINJA_IMPORT_ERROR)),
        ("yt_dlp", (is_yt_dlp_available, YT_DLP_IMPORT_ERROR)),
        ("rich", (is_rich_available, RICH_IMPORT_ERROR)),
        ("keras_nlp", (is_keras_nlp_available, KERAS_NLP_IMPORT_ERROR)),
        ("pydantic", (is_pydantic_available, PYDANTIC_IMPORT_ERROR)),
        ("fastapi", (is_fastapi_available, FASTAPI_IMPORT_ERROR)),
        ("uvicorn", (is_uvicorn_available, UVICORN_IMPORT_ERROR)),
        ("openai", (is_openai_available, OPENAI_IMPORT_ERROR)),
        ("mistral-common", (is_mistral_common_available, MISTRAL_COMMON_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    # Raise an error for users who might not realize that classes without "TF" are torch-only
    if "torch" in backends and "tf" not in backends and not is_torch_available() and is_tf_available():
        raise ImportError(PYTORCH_IMPORT_ERROR_WITH_TF.format(name))

    # Raise the inverse error for PyTorch users trying to load TF classes
    if "tf" in backends and "torch" not in backends and is_torch_available() and not is_tf_available():
        raise ImportError(TF_IMPORT_ERROR_WITH_PYTORCH.format(name))

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


def is_torch_fx_proxy(x):
    if is_torch_fx_available():
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    return False


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
        module_spec: Optional[importlib.machinery.ModuleSpec] = None,
        extra_objects: Optional[dict[str, object]] = None,
        explicit_import_shortcut: Optional[dict[str, list[str]]] = None,
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
                    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError, RuntimeError):
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
        result = super().__dir__()
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
            except (ModuleNotFoundError, RuntimeError) as e:
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
            value = None
            for key, values in self._explicit_import_shortcut.items():
                if name in values:
                    value = self._get_module(key)

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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module = sys.modules[name]
    return module


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
            "=": VersionComparison.EQUAL.value,
            "==": VersionComparison.EQUAL.value,
            "!=": VersionComparison.NOT_EQUAL.value,
            ">": VersionComparison.GREATER_THAN.value,
            "<": VersionComparison.LESS_THAN.value,
            ">=": VersionComparison.GREATER_THAN_OR_EQUAL.value,
            "<=": VersionComparison.LESS_THAN_OR_EQUAL.value,
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

    def is_satisfied(self) -> bool:
        return VersionComparison.from_string(self.version_comparison)(
            version.parse(importlib.metadata.version(self.package_name)), version.parse(self.version)
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
    lambda e: "modeling_tf_" in e: ("tf",),
    lambda e: "modeling_flax_" in e: ("flax",),
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
       - If a file is named like `modeling_tf_*.py`, it will have a `tf` backend
       - If a file is named like `modeling_flax_*.py`, it will have a `flax` backend
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
                'configuration_albert': {'AlbertConfig', 'AlbertOnnxConfig'}
            },
            frozenset({'tokenizers'}): {
                'tokenization_albert_fast': {'AlbertTokenizerFast'}
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
        # For example, any file named `modeling_tf_xxx.py`
        # should have TensorFlow as a required backend.
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
                    #             "tf",
                    #     )
                    # )
                    #
                    # or
                    #
                    # @export(
                    #     backends=(
                    #             "sentencepiece", "tf"
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
                'configuration_albert': {'AlbertConfig', 'AlbertOnnxConfig'}
            },
            frozenset({'tokenizers'}): {
                'tokenization_albert_fast': {'AlbertTokenizerFast'}
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
            'albert.tokenization_albert_fast': {'AlbertTokenizerFast'}
        },
        frozenset(): {
            'albert.configuration_albert': {'AlbertConfig', 'AlbertOnnxConfig'},
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
def define_import_structure(module_path: str, prefix: Optional[str] = None) -> IMPORT_STRUCTURE_T:
    """
    This method takes a module_path as input and creates an import structure digestible by a _LazyModule.

    Here's an example of an output import structure at the src.transformers.models level:

    {
        frozenset({'tokenizers'}): {
            'albert.tokenization_albert_fast': {'AlbertTokenizerFast'}
        },
        frozenset(): {
            'albert.configuration_albert': {'AlbertConfig', 'AlbertOnnxConfig'},
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
