# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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
Utilities for working with the local dataset cache. Parts of this file is adapted from the AllenNLP library at
https://github.com/allenai/allennlp.
"""
import copy
import fnmatch
import functools
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
from collections import OrderedDict, UserDict
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from enum import Enum
from functools import partial, wraps
from hashlib import sha256
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, ContextManager, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZipFile, is_zipfile

import numpy as np
from packaging import version

import requests
from filelock import FileLock
from huggingface_hub import HfFolder, Repository, create_repo, list_repo_files, whoami
from requests.exceptions import HTTPError
from transformers.utils.logging import tqdm
from transformers.utils.versions import importlib_metadata

from . import __version__
from .utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "intel-tensorflow-avx512",
            "tensorflow-rocm",
            "tensorflow-macos",
        )
        _tf_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_available = _tf_version is not None
    if _tf_available:
        if version.parse(_tf_version) < version.parse("2"):
            logger.info(f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum.")
            _tf_available = False
        else:
            logger.info(f"TensorFlow version {_tf_version} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")
    _tf_available = False


if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None
    if _flax_available:
        try:
            _jax_version = importlib_metadata.version("jax")
            _flax_version = importlib_metadata.version("flax")
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _flax_available = False
else:
    _flax_available = False


_datasets_available = importlib.util.find_spec("datasets") is not None
try:
    # Check we're not importing a "datasets" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    _ = importlib_metadata.version("datasets")
    _datasets_metadata = importlib_metadata.metadata("datasets")
    if _datasets_metadata.get("author", "") != "HuggingFace Inc.":
        _datasets_available = False
except importlib_metadata.PackageNotFoundError:
    _datasets_available = False


_detectron2_available = importlib.util.find_spec("detectron2") is not None
try:
    _detectron2_version = importlib_metadata.version("detectron2")
    logger.debug(f"Successfully imported detectron2 version {_detectron2_version}")
except importlib_metadata.PackageNotFoundError:
    _detectron2_available = False


_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    _faiss_version = importlib_metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib_metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib_metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib_metadata.PackageNotFoundError:
        _faiss_available = False


coloredlogs = importlib.util.find_spec("coloredlogs") is not None
try:
    _coloredlogs_available = importlib_metadata.version("coloredlogs")
    logger.debug(f"Successfully imported sympy version {_coloredlogs_available}")
except importlib_metadata.PackageNotFoundError:
    _coloredlogs_available = False


sympy_available = importlib.util.find_spec("sympy") is not None
try:
    _sympy_available = importlib_metadata.version("sympy")
    logger.debug(f"Successfully imported sympy version {_sympy_available}")
except importlib_metadata.PackageNotFoundError:
    _sympy_available = False


_tf2onnx_available = importlib.util.find_spec("tf2onnx") is not None
try:
    _tf2onnx_version = importlib_metadata.version("tf2onnx")
    logger.debug(f"Successfully imported tf2onnx version {_tf2onnx_version}")
except importlib_metadata.PackageNotFoundError:
    _tf2onnx_available = False

_onnx_available = importlib.util.find_spec("onnxruntime") is not None
try:
    _onxx_version = importlib_metadata.version("onnx")
    logger.debug(f"Successfully imported onnx version {_onxx_version}")
except importlib_metadata.PackageNotFoundError:
    _onnx_available = False


_scatter_available = importlib.util.find_spec("torch_scatter") is not None
try:
    _scatter_version = importlib_metadata.version("torch_scatter")
    logger.debug(f"Successfully imported torch-scatter version {_scatter_version}")
except importlib_metadata.PackageNotFoundError:
    _scatter_available = False


_pytorch_quantization_available = importlib.util.find_spec("pytorch_quantization") is not None
try:
    _pytorch_quantization_version = importlib_metadata.version("pytorch_quantization")
    logger.debug(f"Successfully imported pytorch-quantization version {_pytorch_quantization_version}")
except importlib_metadata.PackageNotFoundError:
    _pytorch_quantization_available = False


_soundfile_available = importlib.util.find_spec("soundfile") is not None
try:
    _soundfile_version = importlib_metadata.version("soundfile")
    logger.debug(f"Successfully imported soundfile version {_soundfile_version}")
except importlib_metadata.PackageNotFoundError:
    _soundfile_available = False


_tensorflow_probability_available = importlib.util.find_spec("tensorflow_probability") is not None
try:
    _tensorflow_probability_version = importlib_metadata.version("tensorflow_probability")
    logger.debug(f"Successfully imported tensorflow-probability version {_tensorflow_probability_version}")
except importlib_metadata.PackageNotFoundError:
    _tensorflow_probability_available = False


_timm_available = importlib.util.find_spec("timm") is not None
try:
    _timm_version = importlib_metadata.version("timm")
    logger.debug(f"Successfully imported timm version {_timm_version}")
except importlib_metadata.PackageNotFoundError:
    _timm_available = False


_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
try:
    _torchaudio_version = importlib_metadata.version("torchaudio")
    logger.debug(f"Successfully imported torchaudio version {_torchaudio_version}")
except importlib_metadata.PackageNotFoundError:
    _torchaudio_available = False


_phonemizer_available = importlib.util.find_spec("phonemizer") is not None
try:
    _phonemizer_version = importlib_metadata.version("phonemizer")
    logger.debug(f"Successfully imported phonemizer version {_phonemizer_version}")
except importlib_metadata.PackageNotFoundError:
    _phonemizer_available = False


_pyctcdecode_available = importlib.util.find_spec("pyctcdecode") is not None
try:
    _pyctcdecode_version = importlib_metadata.version("pyctcdecode")
    logger.debug(f"Successfully imported pyctcdecode version {_pyctcdecode_version}")
except importlib_metadata.PackageNotFoundError:
    _pyctcdecode_available = False


_librosa_available = importlib.util.find_spec("librosa") is not None
try:
    _librosa_version = importlib_metadata.version("librosa")
    logger.debug(f"Successfully imported librosa version {_librosa_version}")
except importlib_metadata.PackageNotFoundError:
    _librosa_available = False


torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
old_default_cache_path = os.path.join(torch_cache_home, "transformers")
# New default cache, shared with the Datasets library
hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "transformers")

# Onetime move from the old location to the new one if no ENV variable has been set.
if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(default_cache_path)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    logger.warning(
        "In Transformers v4.0.0, the default path to cache downloaded models changed from "
        "'~/.cache/torch/transformers' to '~/.cache/huggingface/transformers'. Since you don't seem to have overridden "
        "and '~/.cache/torch/transformers' is a directory that exists, we're moving it to "
        "'~/.cache/huggingface/transformers' to avoid redownloading models you have already in the cache. You should "
        "only see this message once."
    )
    shutil.move(old_default_cache_path, default_cache_path)

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"
SESSION_ID = uuid4().hex
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", False) in ENV_VARS_TRUE_VALUES

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
MODEL_CARD_NAME = "modelcard.json"

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"

_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://moon-staging.huggingface.co" if _staging_mode else "https://huggingface.co"

HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", _default_endpoint)
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"

# This is the version of torch required to run torch.fx features and torch.onnx with dictionary inputs.
TORCH_FX_REQUIRED_VERSION = version.parse("1.9")
TORCH_ONNX_DICT_INPUTS_MINIMUM_VERSION = version.parse("1.8")

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def is_offline_mode():
    return _is_offline_mode


def is_torch_available():
    return _torch_available


def is_pyctcdecode_available():
    return _pyctcdecode_available


def is_librosa_available():
    return _librosa_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def is_torch_bf16_available():
    if not is_torch_available():
        return False

    import torch

    # since currently no utility function is available we build our own.
    # some bits come from https://github.com/pytorch/pytorch/blob/2289a12f21c54da93bf5d696e3f9aea83dd9c10d/torch/testing/_internal/common_cuda.py#L51
    # with additional check for torch version
    # to succeed:
    # 1. the hardware needs to support bf16 (arch >= Ampere)
    # 2. torch >= 1.10 (1.9 should be enough for AMP API has changed in 1.10, so using 1.10 as minimal)
    # 3. CUDA >= 11
    # 4. torch.autocast exists
    # XXX: one problem here is that it may give invalid results on mixed gpus setup, so it's
    # really only correct for the 0th gpu (or currently set default device if different from 0)

    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    if version.parse(torch.__version__) < version.parse("1.10"):
        return False
    if not hasattr(torch, "autocast"):
        return False

    return True


def is_torch_tf32_available():
    if not is_torch_available():
        return False

    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    if version.parse(torch.__version__) < version.parse("1.7"):
        return False

    return True


_torch_fx_available = _torch_onnx_dict_inputs_support_available = False
if _torch_available:
    torch_version = version.parse(importlib_metadata.version("torch"))
    _torch_fx_available = (torch_version.major, torch_version.minor) == (
        TORCH_FX_REQUIRED_VERSION.major,
        TORCH_FX_REQUIRED_VERSION.minor,
    )

    _torch_onnx_dict_inputs_support_available = torch_version >= TORCH_ONNX_DICT_INPUTS_MINIMUM_VERSION


def is_torch_fx_available():
    return _torch_fx_available


def is_torch_onnx_dict_inputs_support_available():
    return _torch_onnx_dict_inputs_support_available


def is_tf_available():
    return _tf_available


def is_coloredlogs_available():
    return _coloredlogs_available


def is_tf2onnx_available():
    return _tf2onnx_available


def is_onnx_available():
    return _onnx_available


def is_flax_available():
    return _flax_available


def is_torch_tpu_available():
    if not _torch_available:
        return False
    # This test is probably enough, but just in case, we unpack a bit.
    if importlib.util.find_spec("torch_xla") is None:
        return False
    if importlib.util.find_spec("torch_xla.core") is None:
        return False
    return importlib.util.find_spec("torch_xla.core.xla_model") is not None


def is_datasets_available():
    return _datasets_available


def is_detectron2_available():
    return _detectron2_available


def is_rjieba_available():
    return importlib.util.find_spec("rjieba") is not None


def is_psutil_available():
    return importlib.util.find_spec("psutil") is not None


def is_py3nvml_available():
    return importlib.util.find_spec("py3nvml") is not None


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


def is_faiss_available():
    return _faiss_available


def is_scipy_available():
    return importlib.util.find_spec("scipy") is not None


def is_sklearn_available():
    if importlib.util.find_spec("sklearn") is None:
        return False
    return is_scipy_available() and importlib.util.find_spec("sklearn.metrics")


def is_sentencepiece_available():
    return importlib.util.find_spec("sentencepiece") is not None


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_tokenizers_available():
    return importlib.util.find_spec("tokenizers") is not None


def is_vision_available():
    return importlib.util.find_spec("PIL") is not None


def is_pytesseract_available():
    return importlib.util.find_spec("pytesseract") is not None


def is_spacy_available():
    return importlib.util.find_spec("spacy") is not None


def is_ftfy_available():
    return importlib.util.find_spec("ftfy") is not None


def is_in_notebook():
    try:
        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


def is_scatter_available():
    return _scatter_available


def is_pytorch_quantization_available():
    return _pytorch_quantization_available


def is_tensorflow_probability_available():
    return _tensorflow_probability_available


def is_pandas_available():
    return importlib.util.find_spec("pandas") is not None


def is_sagemaker_dp_enabled():
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
    return importlib.util.find_spec("smdistributed") is not None


def is_sagemaker_mp_enabled():
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
    return importlib.util.find_spec("smdistributed") is not None


def is_training_run_on_sagemaker():
    return "SAGEMAKER_JOB_NAME" in os.environ


def is_soundfile_availble():
    return _soundfile_available


def is_timm_available():
    return _timm_available


def is_torchaudio_available():
    return _torchaudio_available


def is_speech_available():
    # For now this depends on torchaudio but the exact dependency might evolve in the future.
    return _torchaudio_available


def is_phonemizer_available():
    return _phonemizer_available


def torch_only_method(fn):
    def wrapper(*args, **kwargs):
        if not _torch_available:
            raise ImportError(
                "You need to install pytorch to use this method or class, "
                "or activate it with environment variables USE_TORCH=1 and USE_TF=0."
            )
        else:
            return fn(*args, **kwargs)

    return wrapper


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
that python file if that's the case.
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
"""


# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment.
"""


# docstyle-ignore
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment.
"""


# docstyle-ignore
FAISS_IMPORT_ERROR = """
{0} requires the faiss library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md and follow the ones
that match your environment.
"""


# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
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
"""


# docstyle-ignore
TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
"""


# docstyle-ignore
DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment.
"""


# docstyle-ignore
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
"""


# docstyle-ignore
SCATTER_IMPORT_ERROR = """
{0} requires the torch-scatter library but it was not found in your environment. You can install it with pip as
explained here: https://github.com/rusty1s/pytorch_scatter.
"""

# docstyle-ignore
PYTORCH_QUANTIZATION_IMPORT_ERROR = """
{0} requires the pytorch-quantization library but it was not found in your environment. You can install it with pip:
`pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`
"""

# docstyle-ignore
TENSORFLOW_PROBABILITY_IMPORT_ERROR = """
{0} requires the tensorflow_probability library but it was not found in your environment. You can install it with pip as
explained here: https://github.com/tensorflow/probability.
"""


# docstyle-ignore
PANDAS_IMPORT_ERROR = """
{0} requires the pandas library but it was not found in your environment. You can install it with pip as
explained here: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html.
"""


# docstyle-ignore
PHONEMIZER_IMPORT_ERROR = """
{0} requires the phonemizer library but it was not found in your environment. You can install it with pip:
`pip install phonemizer`
"""


# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`
"""


# docstyle-ignore
SPEECH_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`
"""

# docstyle-ignore
TIMM_IMPORT_ERROR = """
{0} requires the timm library but it was not found in your environment. You can install it with pip:
`pip install timm`
"""

# docstyle-ignore
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`
"""


# docstyle-ignore
PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`
"""

This module should not be update anymore and is only left for backward compatibility.
"""

from . import __version__

# Backward compatibility imports, to make sure all those objects can be found in file_utils
from .utils import (
    CLOUDFRONT_DISTRIB_PREFIX,
    CONFIG_NAME,
    DISABLE_TELEMETRY,
    DUMMY_INPUTS,
    DUMMY_MASK,
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    FEATURE_EXTRACTOR_NAME,
    FLAX_WEIGHTS_NAME,
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_PREFIX,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    MODEL_CARD_NAME,
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    S3_BUCKET_PREFIX,
    SENTENCEPIECE_UNDERLINE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TORCH_FX_REQUIRED_VERSION,
    TRANSFORMERS_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    DummyObject,
    EntryNotFoundError,
    ExplicitEnum,
    ModelOutput,
    PaddingStrategy,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    TensorType,
    _LazyModule,
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    cached_path,
    cached_property,
    copy_func,
    default_cache_path,
    define_sagemaker_information,
    filename_to_url,
    get_cached_models,
    get_file_from_repo,
    get_from_cache,
    get_full_repo_name,
    get_list_of_files,
    has_file,
    hf_bucket_url,
    http_get,
    http_user_agent,
    is_apex_available,
    is_coloredlogs_available,
    is_datasets_available,
    is_detectron2_available,
    is_faiss_available,
    is_flax_available,
    is_ftfy_available,
    is_in_notebook,
    is_librosa_available,
    is_local_clone,
    is_offline_mode,
    is_onnx_available,
    is_pandas_available,
    is_phonemizer_available,
    is_protobuf_available,
    is_psutil_available,
    is_py3nvml_available,
    is_pyctcdecode_available,
    is_pytesseract_available,
    is_pytorch_quantization_available,
    is_remote_url,
    is_rjieba_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_scatter_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_soundfile_availble,
    is_spacy_available,
    is_speech_available,
    is_tensor,
    is_tensorflow_probability_available,
    is_tf2onnx_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_bf16_available,
    is_torch_cuda_available,
    is_torch_fx_available,
    is_torch_fx_proxy,
    is_torch_onnx_dict_inputs_support_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    is_torchaudio_available,
    is_training_run_on_sagemaker,
    is_vision_available,
    replace_return_docstrings,
    requires_backends,
    tf_required,
    to_numpy,
    to_py_obj,
    torch_only_method,
    torch_required,
    torch_version,
    url_to_filename,
)
