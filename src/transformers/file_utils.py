# Copyright 2020 The HuggingFace Team, the AllenNLP library authors. All rights reserved.
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
from contextlib import contextmanager
from dataclasses import fields
from enum import Enum
from functools import partial, wraps
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZipFile, is_zipfile

import numpy as np
from packaging import version
from tqdm.auto import tqdm

import requests
from filelock import FileLock
from huggingface_hub import HfApi, HfFolder, Repository
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


_onnx_available = (
    importlib.util.find_spec("keras2onnx") is not None and importlib.util.find_spec("onnxruntime") is not None
)
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


_soundfile_available = importlib.util.find_spec("soundfile") is not None
try:
    _soundfile_version = importlib_metadata.version("soundfile")
    logger.debug(f"Successfully imported soundfile version {_soundfile_version}")
except importlib_metadata.PackageNotFoundError:
    _soundfile_available = False


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

PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",
}

# This is the version of torch required to run torch.fx features.
TORCH_FX_REQUIRED_VERSION = version.parse("1.8")

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def is_offline_mode():
    return _is_offline_mode


def is_torch_available():
    return _torch_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


_torch_fx_available = False
if _torch_available:
    torch_version = version.parse(importlib_metadata.version("torch"))
    _torch_fx_available = (torch_version.major, torch_version.minor) == (
        TORCH_FX_REQUIRED_VERSION.major,
        TORCH_FX_REQUIRED_VERSION.minor,
    )


def is_torch_fx_available():
    return _torch_fx_available


def is_tf_available():
    return _tf_available


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


def is_psutil_available():
    return importlib.util.find_spec("psutil") is not None


def is_py3nvml_available():
    return importlib.util.find_spec("py3nvml") is not None


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


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
PANDAS_IMPORT_ERROR = """
{0} requires the pandas library but it was not found in your environment. You can install it with pip as
explained here: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html.
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


BACKENDS_MAPPING = OrderedDict(
    [
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("scatter", (is_scatter_available, SCATTER_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not all(BACKENDS_MAPPING[backend][0]() for backend in backends):
        raise ImportError("".join([BACKENDS_MAPPING[backend][1].format(name) for backend in backends]))


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        class_name = f":class:`~transformers.{fn.__qualname__.split('.')[0]}`"
        intro = f"   The {class_name} forward method, overrides the :func:`__call__` special method."
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within this function, one should call the
        :class:`Module` instance afterwards instead of this since the former takes care of running the pre and post
        processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + "".join(docstr)
        return fn

    return docstring_decorator


PT_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(torch.FloatTensor)`: A :class:`~{full_output_type}` or a tuple of
        :obj:`torch.FloatTensor` (if ``return_dict=False`` is passed or when ``config.return_dict=False``) comprising
        various elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


TF_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(tf.Tensor)`: A :class:`~{full_output_type}` or a tuple of
        :obj:`tf.Tensor` (if ``return_dict=False`` is passed or when ``config.return_dict=False``) comprising various
        elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    docstrings = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = docstrings.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        docstrings = "\n".join(lines[(i + 1) :])
        docstrings = _convert_output_args_doc(docstrings)

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    return intro + docstrings


PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> inputs = tokenizer(question, text, return_tensors='pt')
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])

        >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        >>> loss = outputs.loss
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")
        >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."
        >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
        >>> outputs = model(**{{k: v.unsqueeze(0) for k,v in encoding.items()}}, labels=labels)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> import torch
        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs, labels=inputs["input_ids"])
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

PT_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": PT_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": PT_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": PT_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": PT_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": PT_MASKED_LM_SAMPLE,
    "LMHead": PT_CAUSAL_LM_SAMPLE,
    "BaseModel": PT_BASE_MODEL_SAMPLE,
}


TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> input_ids = inputs["input_ids"]
        >>> inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> input_dict = tokenizer(question, text, return_tensors='tf')
        >>> outputs = model(input_dict)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits

        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        >>> answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
        >>> inputs["labels"] = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]

        >>> outputs = model(inputs)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='tf', padding=True)
        >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
        >>> outputs = model(inputs)  # batch size is 1

        >>> # the linear classifier still needs to be trained
        >>> logits = outputs.logits
"""

TF_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)
        >>> logits = outputs.logits
"""

TF_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": TF_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": TF_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": TF_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": TF_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": TF_MASKED_LM_SAMPLE,
    "LMHead": TF_CAUSAL_LM_SAMPLE,
    "BaseModel": TF_BASE_MODEL_SAMPLE,
}


FLAX_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
"""

FLAX_QUESTION_ANSWERING_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> inputs = tokenizer(question, text, return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> start_scores = outputs.start_logits
        >>> end_scores = outputs.end_logits
"""

FLAX_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')

        >>> outputs = model(**inputs, labels=labels)
        >>> logits = outputs.logits
"""

FLAX_MASKED_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors='jax')

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
"""

FLAX_BASE_MODEL_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

FLAX_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> choice0 = "It is eaten with a fork and a knife."
        >>> choice1 = "It is eaten while held in the hand."

        >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='jax', padding=True)
        >>> outputs = model(**{{k: v[None, :] for k,v in encoding.items()}})

        >>> logits = outputs.logits
"""

FLAX_CAUSAL_LM_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
        >>> outputs = model(**inputs, labels=inputs["input_ids"])

        >>> logits = outputs.logits
"""

FLAX_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": FLAX_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": FLAX_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": FLAX_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": FLAX_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": FLAX_MASKED_LM_SAMPLE,
    "BaseModel": FLAX_BASE_MODEL_SAMPLE,
    "LMHead": FLAX_CAUSAL_LM_SAMPLE,
}


def add_code_sample_docstrings(
    *docstr, tokenizer_class=None, checkpoint=None, output_type=None, config_class=None, mask=None, model_cls=None
):
    def docstring_decorator(fn):
        # model_class defaults to function's class if not specified otherwise
        model_class = fn.__qualname__.split(".")[0] if model_cls is None else model_cls

        if model_class[:2] == "TF":
            sample_docstrings = TF_SAMPLE_DOCSTRINGS
        elif model_class[:4] == "Flax":
            sample_docstrings = FLAX_SAMPLE_DOCSTRINGS
        else:
            sample_docstrings = PT_SAMPLE_DOCSTRINGS

        doc_kwargs = dict(model_class=model_class, tokenizer_class=tokenizer_class, checkpoint=checkpoint)

        if "SequenceClassification" in model_class:
            code_sample = sample_docstrings["SequenceClassification"]
        elif "QuestionAnswering" in model_class:
            code_sample = sample_docstrings["QuestionAnswering"]
        elif "TokenClassification" in model_class:
            code_sample = sample_docstrings["TokenClassification"]
        elif "MultipleChoice" in model_class:
            code_sample = sample_docstrings["MultipleChoice"]
        elif "MaskedLM" in model_class or model_class in ["FlaubertWithLMHeadModel", "XLMWithLMHeadModel"]:
            doc_kwargs["mask"] = "[MASK]" if mask is None else mask
            code_sample = sample_docstrings["MaskedLM"]
        elif "LMHead" in model_class or "CausalLM" in model_class:
            code_sample = sample_docstrings["LMHead"]
        elif "Model" in model_class or "Encoder" in model_class:
            code_sample = sample_docstrings["BaseModel"]
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")

        output_doc = _prepare_output_docstrings(output_type, config_class) if output_type is not None else ""
        built_doc = code_sample.format(**doc_kwargs)
        fn.__doc__ = (fn.__doc__ or "") + "".join(docstr) + output_doc + built_doc
        return fn

    return docstring_decorator


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            lines[i] = _prepare_output_docstrings(output_type, config_class)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def hf_bucket_url(
    model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None, mirror=None
) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if mirror:
        endpoint = PRESET_MIRROR_DICT.get(mirror, mirror)
        legacy_format = "/" not in model_id
        if legacy_format:
            return f"{endpoint}/{model_id}-{filename}"
        else:
            return f"{endpoint}/{model_id}/{filename}"

    if revision is None:
        revision = "main"
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`. Raise ``EnvironmentError`` if `filename` or
    its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError(f"file {cache_path} not found")

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError(f"file {meta_path} not found")

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def get_cached_models(cache_dir: Union[str, Path] = None) -> List[Tuple]:
    """
    Returns a list of tuples representing model binaries that are cached locally. Each tuple has shape
    :obj:`(model_url, etag, size_MB)`. Filenames in :obj:`cache_dir` are use to get the metadata for each model, only
    urls ending with `.bin` are added.

    Args:
        cache_dir (:obj:`Union[str, Path]`, `optional`):
            The cache directory to search for models within. Will default to the transformers cache if unset.

    Returns:
        List[Tuple]: List of tuples each with shape :obj:`(model_url, etag, size_MB)`
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    elif isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cached_models = []
    for file in os.listdir(cache_dir):
        if file.endswith(".json"):
            meta_path = os.path.join(cache_dir, file)
            with open(meta_path, encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                url = metadata["url"]
                etag = metadata["etag"]
                if url.endswith(".bin"):
                    size_MB = os.path.getsize(meta_path.strip(".json")) / 1e6
                    cached_models.append((url, etag, size_MB))

    return cached_models


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        use_auth_token: Optional string or boolean to use as Bearer token for remote files. If True,
            will get token from ~/.huggingface.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError(f"Archive format of {output_path} could not be identified")

        return output_path_extracted

    return output_path


def define_sagemaker_information():
    try:
        instance_data = requests.get(os.environ["ECS_CONTAINER_METADATA_URI"]).json()
        dlc_container_used = instance_data["Image"]
        dlc_tag = instance_data["Image"].split(":")[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None

    sagemaker_params = json.loads(os.getenv("SM_FRAMEWORK_PARAMS", "{}"))
    runs_distributed_training = True if "sagemaker_distributed_dataparallel_enabled" in sagemaker_params else False
    account_id = os.getenv("TRAINING_JOB_ARN").split(":")[4] if "TRAINING_JOB_ARN" in os.environ else None

    sagemaker_object = {
        "sm_framework": os.getenv("SM_FRAMEWORK_MODULE", None),
        "sm_region": os.getenv("AWS_REGION", None),
        "sm_number_gpu": os.getenv("SM_NUM_GPUS", 0),
        "sm_number_cpu": os.getenv("SM_NUM_CPUS", 0),
        "sm_distributed_training": runs_distributed_training,
        "sm_deep_learning_container": dlc_container_used,
        "sm_deep_learning_container_tag": dlc_tag,
        "sm_account_id": account_id,
    }
    return sagemaker_object


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    if is_training_run_on_sagemaker():
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def http_get(url: str, temp_file: BinaryIO, proxies=None, resume_size=0, headers: Optional[Dict[str, str]] = None):
    """
    Download remote file. Do not gobble up errors.
    """
    headers = copy.deepcopy(headers)
    if resume_size > 0:
        headers["Range"] = f"bytes={resume_size}-"
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    r.raise_for_status()
    content_length = r.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
        disable=bool(logging.get_verbosity() == logging.NOTSET),
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
        headers["authorization"] = f"Bearer {token}"

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout)
            r.raise_for_status()
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Otherwise, our Internet connection is down.
            # etag is None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise FileNotFoundError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}")

            http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, headers=headers)

        logger.info(f"storing {url} in cache at {cache_path}")
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


def torch_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_torch_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires PyTorch.")

    return wrapper


def tf_required(func):
    # Chose a different decorator name than in tests so it's clear they are not the same.
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_tf_available():
            return func(*args, **kwargs)
        else:
            raise ImportError(f"Method `{func.__name__}` requires TF.")

    return wrapper


def is_torch_fx_proxy(x):
    if is_torch_fx_available():
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    return False


def is_tensor(x):
    """
    Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor`, obj:`jaxlib.xla_extension.DeviceArray` or
    :obj:`np.ndarray`.
    """
    if is_torch_fx_proxy(x):
        return True
    if is_torch_available():
        import torch

        if isinstance(x, torch.Tensor):
            return True
    if is_tf_available():
        import tensorflow as tf

        if isinstance(x, tf.Tensor):
            return True

    if is_flax_available():
        import jaxlib.xla_extension as jax_xla
        from jax.core import Tracer

        if isinstance(x, (jax_xla.DeviceArray, Tracer)):
            return True

    return isinstance(x, np.ndarray)


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


def _is_torch_device(x):
    import torch

    return isinstance(x, torch.device)


def _is_tensorflow(x):
    import tensorflow as tf

    return isinstance(x, tf.Tensor)


def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811

    return isinstance(x, jnp.ndarray)


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif is_tf_available() and _is_tensorflow(obj):
        return obj.numpy().tolist()
    elif is_torch_available() and _is_torch(obj):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the ``return_tensors`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class _BaseLazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, import_structure):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + sum(import_structure.values(), [])

    # Needed for autocompletion in an IDE
    def __dir__(self):
        return super().__dir__() + self.__all__

    def __getattr__(self, name: str) -> Any:
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        raise NotImplementedError


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def is_local_clone(repo_path, repo_url):
    """
    Checks if the folder in `repo_path` is a local clone of `repo_url`.
    """
    # First double-check that `repo_path` is a git repo
    if not os.path.exists(os.path.join(repo_path, ".git")):
        return False
    test_git = subprocess.run("git branch".split(), cwd=repo_path)
    if test_git.returncode != 0:
        return False

    # Then look at its remotes
    remotes = subprocess.run(
        "git remote -v".split(),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
        cwd=repo_path,
    ).stdout

    return repo_url in remotes.split()


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def push_to_hub(
        self,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        use_temp_dir: bool = False,
        commit_message: Optional[str] = None,
        organization: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        """
        Upload model checkpoint or tokenizer files to the 🤗 Model Hub while synchronizing a local clone of the repo in
        :obj:`repo_path_or_name`.

        Parameters:
            repo_path_or_name (:obj:`str`, `optional`):
                Can either be a repository name for your model or tokenizer in the Hub or a path to a local folder (in
                which case the repository will have the name of that local folder). If not specified, will default to
                the name given by :obj:`repo_url` and a local directory with that name will be created.
            repo_url (:obj:`str`, `optional`):
                Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
                repository will be created in your namespace (unless you specify an :obj:`organization`) with
                :obj:`repo_name`.
            use_temp_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clone the distant repo in a temporary directory or in :obj:`repo_path_or_name` inside
                the current working directory. This will slow things down if you are making changes in an existing repo
                since you will need to clone the repo before every push.
            commit_message (:obj:`str`, `optional`):
                Message to commit while pushing. Will default to :obj:`"add config"`, :obj:`"add tokenizer"` or
                :obj:`"add model"` depending on the type of the class.
            organization (:obj:`str`, `optional`):
                Organization in which you want to push your model or tokenizer (you must be a member of this
                organization).
            private (:obj:`bool`, `optional`):
                Whether or not the repository created should be private (requires a paying subscription).
            use_auth_token (:obj:`bool` or :obj:`str`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`). Will default to
                :obj:`True` if :obj:`repo_url` is not specified.


        Returns:
            The url of the commit of your model in the given repository.

        Examples::

            # Upload a model to the Hub:
            from transformers import AutoModel

            model = BertModel.from_pretrained("bert-base-cased")
            # Fine-tuning code

            # Push the model to your namespace with the name "my-finetuned-bert" and have a local clone in the
            # `my-finetuned-bert` folder.
            model.push_to_hub("my-finetuned-bert")

            # Push the model to your namespace with the name "my-finetuned-bert" with no local clone.
            model.push_to_hub("my-finetuned-bert", use_temp_dir=True)

            # Push the model to an organization with the name "my-finetuned-bert" and have a local clone in the
            # `my-finetuned-bert` folder.
            model.push_to_hub("my-finetuned-bert", organization="huggingface")

            # Make a change to an existing repo that has been cloned locally in `my-finetuned-bert`.
            model.push_to_hub("my-finetuned-bert", repo_url="https://huggingface.co/sgugger/my-finetuned-bert")
        """
        if use_temp_dir:
            # Make sure we use the right `repo_name` for the `repo_url` before replacing it.
            if repo_url is None:
                if use_auth_token is None:
                    use_auth_token = True
                repo_name = Path(repo_path_or_name).name
                repo_url = self._get_repo_url_from_name(
                    repo_name, organization=organization, private=private, use_auth_token=use_auth_token
                )
            repo_path_or_name = tempfile.mkdtemp()

        # Create or clone the repo. If the repo is already cloned, this just retrieves the path to the repo.
        repo = self._create_or_get_repo(
            repo_path_or_name=repo_path_or_name,
            repo_url=repo_url,
            organization=organization,
            private=private,
            use_auth_token=use_auth_token,
        )
        # Save the files in the cloned repo
        self.save_pretrained(repo_path_or_name)
        # Commit and push!
        url = self._push_to_hub(repo, commit_message=commit_message)

        # Clean up! Clean up! Everybody everywhere!
        if use_temp_dir:
            shutil.rmtree(repo_path_or_name)

        return url

    @staticmethod
    def _get_repo_url_from_name(
        repo_name: str,
        organization: Optional[str] = None,
        private: bool = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        if isinstance(use_auth_token, str):
            token = use_auth_token
        elif use_auth_token:
            token = HfFolder.get_token()
            if token is None:
                raise ValueError(
                    "You must login to the Hugging Face hub on this computer by typing `transformers-cli login` and "
                    "entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own "
                    "token as the `use_auth_token` argument."
                )
        else:
            token = None

        # Special provision for the test endpoint (CI)
        return HfApi(endpoint=HUGGINGFACE_CO_RESOLVE_ENDPOINT).create_repo(
            token,
            repo_name,
            organization=organization,
            private=private,
            repo_type=None,
            exist_ok=True,
        )

    @classmethod
    def _create_or_get_repo(
        cls,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
        private: bool = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> Repository:
        if repo_path_or_name is None and repo_url is None:
            raise ValueError("You need to specify a `repo_path_or_name` or a `repo_url`.")

        if use_auth_token is None and repo_url is None:
            use_auth_token = True

        if repo_path_or_name is None:
            repo_path_or_name = repo_url.split("/")[-1]

        if repo_url is None and not os.path.exists(repo_path_or_name):
            repo_name = Path(repo_path_or_name).name
            repo_url = cls._get_repo_url_from_name(
                repo_name, organization=organization, private=private, use_auth_token=use_auth_token
            )

        # Create a working directory if it does not exist.
        if not os.path.exists(repo_path_or_name):
            os.makedirs(repo_path_or_name)

        repo = Repository(repo_path_or_name, clone_from=repo_url, use_auth_token=use_auth_token)
        repo.git_pull()
        return repo

    @classmethod
    def _push_to_hub(cls, repo: Repository, commit_message: Optional[str] = None) -> str:
        if commit_message is None:
            if "Tokenizer" in cls.__name__:
                commit_message = "add tokenizer"
            elif "Config" in cls.__name__:
                commit_message = "add config"
            else:
                commit_message = "add model"

        return repo.push_to_hub(commit_message=commit_message)
