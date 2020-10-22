"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import fnmatch
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
from functools import partial, wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import numpy as np
from tqdm.auto import tqdm

import requests
from filelock import FileLock

from . import __version__
from .utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
    if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
        import torch

        _torch_available = True  # pylint: disable=invalid-name
        logger.info("PyTorch version {} available.".format(torch.__version__))
    else:
        logger.info("Disabling PyTorch because USE_TF is set")
        _torch_available = False
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name

try:
    USE_TF = os.environ.get("USE_TF", "AUTO").upper()
    USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        import tensorflow as tf

        assert hasattr(tf, "__version__") and int(tf.__version__[0]) >= 2
        _tf_available = True  # pylint: disable=invalid-name
        logger.info("TensorFlow version {} available.".format(tf.__version__))
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")
        _tf_available = False
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name


try:
    USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

    if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
        import flax
        import jax

        logger.info("JAX version {}, Flax: available".format(jax.__version__))
        logger.info("Flax available: {}".format(flax))
        _flax_available = True
    else:
        _flax_available = False
except ImportError:
    _flax_available = False  # pylint: disable=invalid-name


try:
    import datasets  # noqa: F401

    # Check we're not importing a "datasets" directory somewhere
    _datasets_available = hasattr(datasets, "__version__") and hasattr(datasets, "load_dataset")
    if _datasets_available:
        logger.debug(f"Succesfully imported datasets version {datasets.__version__}")
    else:
        logger.debug("Imported a datasets object but this doesn't seem to be the 🤗 datasets library.")

except ImportError:
    _datasets_available = False

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )


try:
    import torch_xla.core.xla_model as xm  # noqa: F401

    if _torch_available:
        _torch_tpu_available = True  # pylint: disable=
    else:
        _torch_tpu_available = False
except ImportError:
    _torch_tpu_available = False


try:
    import psutil  # noqa: F401

    _psutil_available = True

except ImportError:
    _psutil_available = False


try:
    import py3nvml  # noqa: F401

    _py3nvml_available = True

except ImportError:
    _py3nvml_available = False


try:
    from apex import amp  # noqa: F401

    _has_apex = True
except ImportError:
    _has_apex = False


try:
    import faiss  # noqa: F401

    _faiss_available = True
    logger.debug(f"Succesfully imported faiss version {faiss.__version__}")
except ImportError:
    _faiss_available = False

try:
    import sklearn.metrics  # noqa: F401

    import scipy.stats  # noqa: F401

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False

try:
    # Test copied from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
    get_ipython = sys.modules["IPython"].get_ipython
    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
    if "VSCODE_PID" in os.environ:
        raise ImportError("vscode")

    import IPython  # noqa: F401

    _in_notebook = True
except (AttributeError, ImportError, KeyError):
    _in_notebook = False


try:
    import sentencepiece  # noqa: F401

    _sentencepiece_available = True

except ImportError:
    _sentencepiece_available = False


try:
    import tokenizers  # noqa: F401

    _tokenizers_available = True

except ImportError:
    _tokenizers_available = False


default_cache_path = os.path.join(torch_cache_home, "transformers")


PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
CONFIG_NAME = "config.json"
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
PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",
}


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available


def is_flax_available():
    return _flax_available


def is_torch_tpu_available():
    return _torch_tpu_available


def is_datasets_available():
    return _datasets_available


def is_psutil_available():
    return _psutil_available


def is_py3nvml_available():
    return _py3nvml_available


def is_apex_available():
    return _has_apex


def is_faiss_available():
    return _faiss_available


def is_sklearn_available():
    return _has_sklearn


def is_sentencepiece_available():
    return _sentencepiece_available


def is_tokenizers_available():
    return _tokenizers_available


def is_in_notebook():
    return _in_notebook


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


DATASETS_IMPORT_ERROR = """
{0} requires the 🤗 Datasets library but it was not found in your enviromnent. You can install it with:
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


TOKENIZERS_IMPORT_ERROR = """
{0} requires the 🤗 Tokenizers library but it was not found in your enviromnent. You can install it with:
```
pip install tokenizers
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install tokenizers
```
"""


SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your enviromnent. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your enviromnent.
"""


FAISS_IMPORT_ERROR = """
{0} requires the faiss library but it was not found in your enviromnent. Checkout the instructions on the
installation page of its repo: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md and follow the ones
that match your enviromnent.
"""


PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your enviromnent. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your enviromnent.
"""


SKLEARN_IMPORT_ERROR = """
{0} requires the scikit-learn library but it was not found in your enviromnent. You can install it with:
```
pip install -U scikit-learn
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install -U scikit-learn
```
"""


TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your enviromnent. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your enviromnent.
"""


FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your enviromnent. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your enviromnent.
"""


def requires_datasets(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_datasets_available():
        raise ImportError(DATASETS_IMPORT_ERROR.format(name))


def requires_faiss(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_faiss_available():
        raise ImportError(FAISS_IMPORT_ERROR.format(name))


def requires_pytorch(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_torch_available():
        raise ImportError(PYTORCH_IMPORT_ERROR.format(name))


def requires_sklearn(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_sklearn_available():
        raise ImportError(SKLEARN_IMPORT_ERROR.format(name))


def requires_tf(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_tf_available():
        raise ImportError(TENSORFLOW_IMPORT_ERROR.format(name))


def requires_flax(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_flax_available():
        raise ImportError(FLAX_IMPORT_ERROR.format(name))


def requires_tokenizers(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_tokenizers_available():
        raise ImportError(TOKENIZERS_IMPORT_ERROR.format(name))


def requires_sentencepiece(obj):
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not is_sentencepiece_available():
        raise ImportError(SENTENCEPIECE_IMPORT_ERROR.format(name))


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
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
        :class:`~{full_output_type}` or :obj:`tuple(torch.FloatTensor)`:
        A :class:`~{full_output_type}` (if ``return_dict=True`` is passed or when ``config.return_dict=True``) or a
        tuple of :obj:`torch.FloatTensor` comprising various elements depending on the configuration
        (:class:`~transformers.{config_class}`) and inputs.

"""


TF_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(tf.Tensor)`:
        A :class:`~{full_output_type}` (if ``return_dict=True`` is passed or when ``config.return_dict=True``) or a
        tuple of :obj:`tf.Tensor` comprising various elements depending on the configuration
        (:class:`~transformers.{config_class}`) and inputs.

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import torch

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs, labels=inputs["input_ids"])
        >>> loss = outputs.loss
        >>> logits = outputs.logits
"""

TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True))

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True))

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True))

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True))

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True))

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)

        >>> last_hidden_states = outputs.last_hidden_states
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example::

        >>> from transformers import {tokenizer_class}, {model_class}
        >>> import tensorflow as tf

        >>> tokenizer = {tokenizer_class}.from_pretrained('{checkpoint}')
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True))

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
        >>> model = {model_class}.from_pretrained('{checkpoint}', return_dict=True))

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
        >>> outputs = model(inputs)
        >>> logits = outputs.logits
"""


def add_code_sample_docstrings(
    *docstr, tokenizer_class=None, checkpoint=None, output_type=None, config_class=None, mask=None
):
    def docstring_decorator(fn):
        model_class = fn.__qualname__.split(".")[0]
        is_tf_class = model_class[:2] == "TF"
        doc_kwargs = dict(model_class=model_class, tokenizer_class=tokenizer_class, checkpoint=checkpoint)

        if "SequenceClassification" in model_class:
            code_sample = TF_SEQUENCE_CLASSIFICATION_SAMPLE if is_tf_class else PT_SEQUENCE_CLASSIFICATION_SAMPLE
        elif "QuestionAnswering" in model_class:
            code_sample = TF_QUESTION_ANSWERING_SAMPLE if is_tf_class else PT_QUESTION_ANSWERING_SAMPLE
        elif "TokenClassification" in model_class:
            code_sample = TF_TOKEN_CLASSIFICATION_SAMPLE if is_tf_class else PT_TOKEN_CLASSIFICATION_SAMPLE
        elif "MultipleChoice" in model_class:
            code_sample = TF_MULTIPLE_CHOICE_SAMPLE if is_tf_class else PT_MULTIPLE_CHOICE_SAMPLE
        elif "MaskedLM" in model_class or model_class in ["FlaubertWithLMHeadModel", "XLMWithLMHeadModel"]:
            doc_kwargs["mask"] = "[MASK]" if mask is None else mask
            code_sample = TF_MASKED_LM_SAMPLE if is_tf_class else PT_MASKED_LM_SAMPLE
        elif "LMHead" in model_class:
            code_sample = TF_CAUSAL_LM_SAMPLE if is_tf_class else PT_CAUSAL_LM_SAMPLE
        elif "Model" in model_class or "Encoder" in model_class:
            code_sample = TF_BASE_MODEL_SAMPLE if is_tf_class else PT_BASE_MODEL_SAMPLE
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


def hf_bucket_url(model_id: str, filename: str, use_cdn=True, mirror=None) -> str:
    """
    Resolve a model identifier, and a file name, to a HF-hosted url
    on either S3 or Cloudfront (a Content Delivery Network, or CDN).

    Cloudfront is replicated over the globe so downloads are way faster
    for the end user (and it also lowers our bandwidth costs). However, it
    is more aggressively cached by default, so may not always reflect the
    latest changes to the underlying file (default TTL is 24 hours).

    In terms of client-side caching from this library, even though
    Cloudfront relays the ETags from S3, using one or the other
    (or switching from one to the other) will affect caching: cached files
    are not shared between the two because the cached file's name contains
    a hash of the url.
    """
    endpoint = (
        PRESET_MIRROR_DICT.get(mirror, mirror)
        if mirror
        else CLOUDFRONT_DISTRIB_PREFIX
        if use_cdn
        else S3_BUCKET_PREFIX
    )
    legacy_format = "/" not in model_id
    if legacy_format:
        return f"{endpoint}/{model_id}-{filename}"
    else:
        return f"{endpoint}/{model_id}/{filename}"


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletly recieved file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and overide the folder where it was extracted.

    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

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
                raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

        return output_path_extracted

    return output_path


def http_get(url, temp_file, proxies=None, resume_size=0, user_agent: Union[Dict, str, None] = None):
    ua = "transformers/{}; python/{}".format(__version__, sys.version.split()[0])
    if is_torch_available():
        ua += "; torch/{}".format(torch.__version__)
    if is_tf_available():
        ua += "; tensorflow/{}".format(tf.__version__)
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    headers = {"user-agent": ua}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
        disable=bool(logging.get_verbosity() == logging.NOTSET),
    )
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(
    url,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.

    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    etag = None
    if not local_files_only:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            if response.status_code == 200:
                etag = response.headers.get("ETag")
        except (EnvironmentError, requests.exceptions.Timeout):
            # etag is already None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
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
                    raise ValueError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                return None

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
            def _resumable_file_manager():
                with open(incomplete_path, "a+b") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

            http_get(url, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)

        logger.info("storing %s in cache at %s", url, cache_path)
        os.replace(temp_file.name, cache_path)

        logger.info("creating metadata file for %s", cache_path)
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


def is_tensor(x):
    """ Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor` or :obj:`np.ndarray`. """
    if is_torch_available():
        import torch

        if isinstance(x, torch.Tensor):
            return True
    if is_tf_available():
        import tensorflow as tf

        if isinstance(x, tf.Tensor):
            return True
    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a
    regular python dictionary.

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
