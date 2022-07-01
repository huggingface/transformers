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

import importlib.util
import json
import os
import sys
import warnings
from collections import OrderedDict
from functools import wraps
from itertools import chain
from types import ModuleType
from typing import Any

from packaging import version

from transformers.utils.versions import importlib_metadata

from . import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

_torch_version = "N/A"
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


_tf_version = "N/A"
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

_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    _ftfy_version = importlib_metadata.version("ftfy")
    logger.debug(f"Successfully imported ftfy version {_ftfy_version}")
except importlib_metadata.PackageNotFoundError:
    _ftfy_available = False


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


# This is the version of torch required to run torch.fx features and torch.onnx with dictionary inputs.
TORCH_FX_REQUIRED_VERSION = version.parse("1.10")
TORCH_ONNX_DICT_INPUTS_MINIMUM_VERSION = version.parse("1.8")


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


def is_torch_bf16_gpu_available():
    if not is_torch_available():
        return False

    import torch

    # since currently no utility function is available we build our own.
    # some bits come from https://github.com/pytorch/pytorch/blob/2289a12f21c54da93bf5d696e3f9aea83dd9c10d/torch/testing/_internal/common_cuda.py#L51
    # with additional check for torch version
    # to succeed:
    # 1. torch >= 1.10 (1.9 should be enough for AMP API has changed in 1.10, so using 1.10 as minimal)
    # 2. the hardware needs to support bf16 (GPU arch >= Ampere, or CPU)
    # 3. if using gpu, CUDA >= 11
    # 4. torch.autocast exists
    # XXX: one problem here is that it may give invalid results on mixed gpus setup, so it's
    # really only correct for the 0th gpu (or currently set default device if different from 0)
    if version.parse(torch.__version__) < version.parse("1.10"):
        return False

    if torch.cuda.is_available() and torch.version.cuda is not None:
        if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
            return False
        if int(torch.version.cuda.split(".")[0]) < 11:
            return False
        if not hasattr(torch.cuda.amp, "autocast"):
            return False
    else:
        return False

    return True


def is_torch_bf16_cpu_available():
    if not is_torch_available():
        return False

    import torch

    if version.parse(torch.__version__) < version.parse("1.10"):
        return False

    try:
        # multiple levels of AttributeError depending on the pytorch version so do them all in one check
        _ = torch.cpu.amp.autocast
    except AttributeError:
        return False

    return True


def is_torch_bf16_available():
    # the original bf16 check was for gpu only, but later a cpu/bf16 combo has emerged so this util
    # has become ambiguous and therefore deprecated
    warnings.warn(
        "The util is_torch_bf16_available is deprecated, please use is_torch_bf16_gpu_available "
        "or is_torch_bf16_cpu_available instead according to whether it's used with cpu or gpu",
        FutureWarning,
    )
    return is_torch_bf16_gpu_available()


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


torch_version = None
_torch_fx_available = _torch_onnx_dict_inputs_support_available = False
if _torch_available:
    torch_version = version.parse(importlib_metadata.version("torch"))
    _torch_fx_available = (torch_version.major, torch_version.minor) >= (
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


def is_ftfy_available():
    return _ftfy_available


def is_torch_tpu_available(check_device=True):
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    if not _torch_available:
        return False
    if importlib.util.find_spec("torch_xla") is not None:
        if check_device:
            # We need to check if `xla_device` can be found, will raise a RuntimeError if not
            try:
                import torch_xla.core.xla_model as xm

                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False


def is_torchdynamo_available():
    return importlib.util.find_spec("torchdynamo") is not None


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


def is_ipex_available():
    return importlib.util.find_spec("intel_extension_for_pytorch") is not None


def is_bitsandbytes_available():
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


def is_accelerate_available():
    return importlib.util.find_spec("accelerate") is not None


def is_tokenizers_available():
    return importlib.util.find_spec("tokenizers") is not None


def is_vision_available():
    return importlib.util.find_spec("PIL") is not None


def is_pytesseract_available():
    return importlib.util.find_spec("pytesseract") is not None


def is_spacy_available():
    return importlib.util.find_spec("spacy") is not None


def is_tensorflow_text_available():
    return importlib.util.find_spec("tensorflow_text") is not None


def is_in_notebook():
    try:
        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:
            raise ImportError("vscode")
        if "DATABRICKS_RUNTIME_VERSION" in os.environ and os.environ["DATABRICKS_RUNTIME_VERSION"] < "11.0":
            # Databricks Runtime 11.0 and above uses IPython kernel by default so it should be compatible with Jupyter notebook
            # https://docs.microsoft.com/en-us/azure/databricks/notebooks/ipython-kernel
            raise ImportError("databricks")

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
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Checkout the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment.
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
TENSORFLOW_TEXT_IMPORT_ERROR = """
{0} requires the tensorflow_text library but it was not found in your environment. You can install it with pip as
explained here: https://www.tensorflow.org/text/guide/tf_text_intro.
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

# docstyle-ignore
PYCTCDECODE_IMPORT_ERROR = """
{0} requires the pyctcdecode library but it was not found in your environment. You can install it with pip:
`pip install pyctcdecode`
"""

# docstyle-ignore
ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library but it was not found in your environment. You can install it with pip:
`pip install accelerate`
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("phonemizer", (is_phonemizer_available, PHONEMIZER_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("scatter", (is_scatter_available, SCATTER_IMPORT_ERROR)),
        ("pytorch_quantization", (is_pytorch_quantization_available, PYTORCH_QUANTIZATION_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        ("tensorflow_probability", (is_tensorflow_probability_available, TENSORFLOW_PROBABILITY_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("tensorflow_text", (is_tensorflow_text_available, TENSORFLOW_TEXT_IMPORT_ERROR)),
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)


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


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
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
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""
