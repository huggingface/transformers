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

import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Union
from unittest import mock

import huggingface_hub
import requests

from transformers import logging as transformers_logging

from .integrations import (
    is_clearml_available,
    is_fairscale_available,
    is_optuna_available,
    is_ray_available,
    is_sigopt_available,
    is_wandb_available,
)
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
    is_accelerate_available,
    is_apex_available,
    is_auto_gptq_available,
    is_bitsandbytes_available,
    is_bs4_available,
    is_cython_available,
    is_decord_available,
    is_detectron2_available,
    is_essentia_available,
    is_faiss_available,
    is_flax_available,
    is_ftfy_available,
    is_ipex_available,
    is_jieba_available,
    is_jinja_available,
    is_jumanpp_available,
    is_keras_nlp_available,
    is_librosa_available,
    is_natten_available,
    is_onnx_available,
    is_optimum_available,
    is_pandas_available,
    is_peft_available,
    is_phonemizer_available,
    is_pretty_midi_available,
    is_pyctcdecode_available,
    is_pytesseract_available,
    is_pytest_available,
    is_pytorch_quantization_available,
    is_rjieba_available,
    is_safetensors_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_seqio_available,
    is_soundfile_availble,
    is_spacy_available,
    is_sudachi_available,
    is_tensorflow_probability_available,
    is_tensorflow_text_available,
    is_tf2onnx_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tensorrt_fx_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    is_torch_xpu_available,
    is_torchaudio_available,
    is_torchdynamo_available,
    is_torchvision_available,
    is_vision_available,
    strtobool,
)


if is_accelerate_available():
    from accelerate.state import AcceleratorState, PartialState


if is_pytest_available():
    from _pytest.doctest import (
        Module,
        _get_checker,
        _get_continue_on_failure,
        _get_runner,
        _is_mocked,
        _patch_unwrap_mock_aware,
        get_optionflags,
        import_path,
    )
    from _pytest.outcomes import skip
    from pytest import DoctestItem
else:
    Module = object
    DoctestItem = object


SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
DUMMY_UNKNOWN_IDENTIFIER = "julien-c/dummy-unknown"
DUMMY_DIFF_TOKENIZER_IDENTIFIER = "julien-c/dummy-diff-tokenizer"
# Used to test Auto{Config, Model, Tokenizer} model_type detection.

# Used to test the hub
USER = "__DUMMY_TRANSFORMERS_USER__"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"

# Not critical, only usable on the sandboxed CI instance.
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


def parse_int_from_env(key, default=None):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError:
            raise ValueError(f"If set, {key} must be a int.")
    return _value


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
_run_pt_tf_cross_tests = parse_flag_from_env("RUN_PT_TF_CROSS_TESTS", default=True)
_run_pt_flax_cross_tests = parse_flag_from_env("RUN_PT_FLAX_CROSS_TESTS", default=True)
_run_custom_tokenizers = parse_flag_from_env("RUN_CUSTOM_TOKENIZERS", default=False)
_run_staging = parse_flag_from_env("HUGGINGFACE_CO_STAGING", default=False)
_tf_gpu_memory_limit = parse_int_from_env("TF_GPU_MEMORY_LIMIT", default=None)
_run_pipeline_tests = parse_flag_from_env("RUN_PIPELINE_TESTS", default=True)
_run_tool_tests = parse_flag_from_env("RUN_TOOL_TESTS", default=False)
_run_third_party_device_tests = parse_flag_from_env("RUN_THIRD_PARTY_DEVICE_TESTS", default=False)


def is_pt_tf_cross_test(test_case):
    """
    Decorator marking a test as a test that control interactions between PyTorch and TensorFlow.

    PT+TF tests are skipped by default and we can run only them by setting RUN_PT_TF_CROSS_TESTS environment variable
    to a truthy value and selecting the is_pt_tf_cross_test pytest mark.

    """
    if not _run_pt_tf_cross_tests or not is_torch_available() or not is_tf_available():
        return unittest.skip("test is PT+TF test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pt_tf_cross_test()(test_case)


def is_pt_flax_cross_test(test_case):
    """
    Decorator marking a test as a test that control interactions between PyTorch and Flax

    PT+FLAX tests are skipped by default and we can run only them by setting RUN_PT_FLAX_CROSS_TESTS environment
    variable to a truthy value and selecting the is_pt_flax_cross_test pytest mark.

    """
    if not _run_pt_flax_cross_tests or not is_torch_available() or not is_flax_available():
        return unittest.skip("test is PT+FLAX test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pt_flax_cross_test()(test_case)


def is_staging_test(test_case):
    """
    Decorator marking a test as a staging test.

    Those tests will run using the staging environment of huggingface.co instead of the real model hub.
    """
    if not _run_staging:
        return unittest.skip("test is staging test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_staging_test()(test_case)


def is_pipeline_test(test_case):
    """
    Decorator marking a test as a pipeline test. If RUN_PIPELINE_TESTS is set to a falsy value, those tests will be
    skipped.
    """
    if not _run_pipeline_tests:
        return unittest.skip("test is pipeline test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pipeline_test()(test_case)


def is_tool_test(test_case):
    """
    Decorator marking a test as a tool test. If RUN_TOOL_TESTS is set to a falsy value, those tests will be skipped.
    """
    if not _run_tool_tests:
        return unittest.skip("test is a tool test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_tool_test()(test_case)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


def tooslow(test_case):
    """
    Decorator marking a test as too slow.

    Slow tests are skipped while they're in the process of being fixed. No test should stay tagged as "tooslow" as
    these will not be tested by the CI.

    """
    return unittest.skip("test is too slow")(test_case)


def custom_tokenizers(test_case):
    """
    Decorator marking a test for a custom tokenizer.

    Custom tokenizers require additional dependencies, and are skipped by default. Set the RUN_CUSTOM_TOKENIZERS
    environment variable to a truthy value to run them.
    """
    return unittest.skipUnless(_run_custom_tokenizers, "test of custom tokenizers")(test_case)


def require_bs4(test_case):
    """
    Decorator marking a test that requires BeautifulSoup4. These tests are skipped when BeautifulSoup4 isn't installed.
    """
    return unittest.skipUnless(is_bs4_available(), "test requires BeautifulSoup4")(test_case)


def require_accelerate(test_case):
    """
    Decorator marking a test that requires accelerate. These tests are skipped when accelerate isn't installed.
    """
    return unittest.skipUnless(is_accelerate_available(), "test requires accelerate")(test_case)


def require_safetensors(test_case):
    """
    Decorator marking a test that requires safetensors. These tests are skipped when safetensors isn't installed.
    """
    return unittest.skipUnless(is_safetensors_available(), "test requires safetensors")(test_case)


def require_rjieba(test_case):
    """
    Decorator marking a test that requires rjieba. These tests are skipped when rjieba isn't installed.
    """
    return unittest.skipUnless(is_rjieba_available(), "test requires rjieba")(test_case)


def require_jieba(test_case):
    """
    Decorator marking a test that requires jieba. These tests are skipped when jieba isn't installed.
    """
    return unittest.skipUnless(is_jieba_available(), "test requires jieba")(test_case)


def require_jinja(test_case):
    """
    Decorator marking a test that requires jinja. These tests are skipped when jinja isn't installed.
    """
    return unittest.skipUnless(is_jinja_available(), "test requires jinja")(test_case)


def require_tf2onnx(test_case):
    return unittest.skipUnless(is_tf2onnx_available(), "test requires tf2onnx")(test_case)


def require_onnx(test_case):
    return unittest.skipUnless(is_onnx_available(), "test requires ONNX")(test_case)


def require_timm(test_case):
    """
    Decorator marking a test that requires Timm.

    These tests are skipped when Timm isn't installed.

    """
    return unittest.skipUnless(is_timm_available(), "test requires Timm")(test_case)


def require_natten(test_case):
    """
    Decorator marking a test that requires NATTEN.

    These tests are skipped when NATTEN isn't installed.

    """
    return unittest.skipUnless(is_natten_available(), "test requires natten")(test_case)


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


def require_peft(test_case):
    """
    Decorator marking a test that requires PEFT.

    These tests are skipped when PEFT isn't installed.

    """
    return unittest.skipUnless(is_peft_available(), "test requires PEFT")(test_case)


def require_torchvision(test_case):
    """
    Decorator marking a test that requires Torchvision.

    These tests are skipped when Torchvision isn't installed.

    """
    return unittest.skipUnless(is_torchvision_available(), "test requires Torchvision")(test_case)


def require_torch_or_tf(test_case):
    """
    Decorator marking a test that requires PyTorch or TensorFlow.

    These tests are skipped when neither PyTorch not TensorFlow is installed.

    """
    return unittest.skipUnless(is_torch_available() or is_tf_available(), "test requires PyTorch or TensorFlow")(
        test_case
    )


def require_intel_extension_for_pytorch(test_case):
    """
    Decorator marking a test that requires Intel Extension for PyTorch.

    These tests are skipped when Intel Extension for PyTorch isn't installed or it does not match current PyTorch
    version.

    """
    return unittest.skipUnless(
        is_ipex_available(),
        "test requires Intel Extension for PyTorch to be installed and match current PyTorch version, see"
        " https://github.com/intel/intel-extension-for-pytorch",
    )(test_case)


def require_tensorflow_probability(test_case):
    """
    Decorator marking a test that requires TensorFlow probability.

    These tests are skipped when TensorFlow probability isn't installed.

    """
    return unittest.skipUnless(is_tensorflow_probability_available(), "test requires TensorFlow probability")(
        test_case
    )


def require_torchaudio(test_case):
    """
    Decorator marking a test that requires torchaudio. These tests are skipped when torchaudio isn't installed.
    """
    return unittest.skipUnless(is_torchaudio_available(), "test requires torchaudio")(test_case)


def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow. These tests are skipped when TensorFlow isn't installed.
    """
    return unittest.skipUnless(is_tf_available(), "test requires TensorFlow")(test_case)


def require_flax(test_case):
    """
    Decorator marking a test that requires JAX & Flax. These tests are skipped when one / both are not installed
    """
    return unittest.skipUnless(is_flax_available(), "test requires JAX & Flax")(test_case)


def require_sentencepiece(test_case):
    """
    Decorator marking a test that requires SentencePiece. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_sentencepiece_available(), "test requires SentencePiece")(test_case)


def require_seqio(test_case):
    """
    Decorator marking a test that requires SentencePiece. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_seqio_available(), "test requires Seqio")(test_case)


def require_scipy(test_case):
    """
    Decorator marking a test that requires Scipy. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_scipy_available(), "test requires Scipy")(test_case)


def require_tokenizers(test_case):
    """
    Decorator marking a test that requires 🤗 Tokenizers. These tests are skipped when 🤗 Tokenizers isn't installed.
    """
    return unittest.skipUnless(is_tokenizers_available(), "test requires tokenizers")(test_case)


def require_tensorflow_text(test_case):
    """
    Decorator marking a test that requires tensorflow_text. These tests are skipped when tensroflow_text isn't
    installed.
    """
    return unittest.skipUnless(is_tensorflow_text_available(), "test requires tensorflow_text")(test_case)


def require_keras_nlp(test_case):
    """
    Decorator marking a test that requires keras_nlp. These tests are skipped when keras_nlp isn't installed.
    """
    return unittest.skipUnless(is_keras_nlp_available(), "test requires keras_nlp")(test_case)


def require_pandas(test_case):
    """
    Decorator marking a test that requires pandas. These tests are skipped when pandas isn't installed.
    """
    return unittest.skipUnless(is_pandas_available(), "test requires pandas")(test_case)


def require_pytesseract(test_case):
    """
    Decorator marking a test that requires PyTesseract. These tests are skipped when PyTesseract isn't installed.
    """
    return unittest.skipUnless(is_pytesseract_available(), "test requires PyTesseract")(test_case)


def require_pytorch_quantization(test_case):
    """
    Decorator marking a test that requires PyTorch Quantization Toolkit. These tests are skipped when PyTorch
    Quantization Toolkit isn't installed.
    """
    return unittest.skipUnless(is_pytorch_quantization_available(), "test requires PyTorch Quantization Toolkit")(
        test_case
    )


def require_vision(test_case):
    """
    Decorator marking a test that requires the vision dependencies. These tests are skipped when torchaudio isn't
    installed.
    """
    return unittest.skipUnless(is_vision_available(), "test requires vision")(test_case)


def require_ftfy(test_case):
    """
    Decorator marking a test that requires ftfy. These tests are skipped when ftfy isn't installed.
    """
    return unittest.skipUnless(is_ftfy_available(), "test requires ftfy")(test_case)


def require_spacy(test_case):
    """
    Decorator marking a test that requires SpaCy. These tests are skipped when SpaCy isn't installed.
    """
    return unittest.skipUnless(is_spacy_available(), "test requires spacy")(test_case)


def require_decord(test_case):
    """
    Decorator marking a test that requires decord. These tests are skipped when decord isn't installed.
    """
    return unittest.skipUnless(is_decord_available(), "test requires decord")(test_case)


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch). These tests are skipped on a machine without
    multiple GPUs.

    To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests -k "multi_gpu"
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)


def require_torch_non_multi_gpu(test_case):
    """
    Decorator marking a test that requires 0 or 1 GPU setup (in PyTorch).
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    return unittest.skipUnless(torch.cuda.device_count() < 2, "test requires 0 or 1 GPU")(test_case)


def require_torch_up_to_2_gpus(test_case):
    """
    Decorator marking a test that requires 0 or 1 or 2 GPU setup (in PyTorch).
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    return unittest.skipUnless(torch.cuda.device_count() < 3, "test requires 0 or 1 or 2 GPUs")(test_case)


def require_torch_tpu(test_case):
    """
    Decorator marking a test that requires a TPU (in PyTorch).
    """
    return unittest.skipUnless(is_torch_tpu_available(check_device=False), "test requires PyTorch TPU")(test_case)


def require_torch_neuroncore(test_case):
    """
    Decorator marking a test that requires NeuronCore (in PyTorch).
    """
    return unittest.skipUnless(is_torch_neuroncore_available(check_device=False), "test requires PyTorch NeuronCore")(
        test_case
    )


def require_torch_npu(test_case):
    """
    Decorator marking a test that requires NPU (in PyTorch).
    """
    return unittest.skipUnless(is_torch_npu_available(), "test requires PyTorch NPU")(test_case)


def require_torch_multi_npu(test_case):
    """
    Decorator marking a test that requires a multi-NPU setup (in PyTorch). These tests are skipped on a machine without
    multiple NPUs.

    To run *only* the multi_npu tests, assuming all test names contain multi_npu: $ pytest -sv ./tests -k "multi_npu"
    """
    if not is_torch_npu_available():
        return unittest.skip("test requires PyTorch NPU")(test_case)

    return unittest.skipUnless(torch.npu.device_count() > 1, "test requires multiple NPUs")(test_case)


def require_torch_xpu(test_case):
    """
    Decorator marking a test that requires XPU and IPEX.

    These tests are skipped when Intel Extension for PyTorch isn't installed or it does not match current PyTorch
    version.
    """
    return unittest.skipUnless(is_torch_xpu_available(), "test requires IPEX and an XPU device")(test_case)


def require_torch_multi_xpu(test_case):
    """
    Decorator marking a test that requires a multi-XPU setup with IPEX and atleast one XPU device. These tests are
    skipped on a machine without IPEX or multiple XPUs.

    To run *only* the multi_xpu tests, assuming all test names contain multi_xpu: $ pytest -sv ./tests -k "multi_xpu"
    """
    if not is_torch_xpu_available():
        return unittest.skip("test requires IPEX and atleast one XPU device")(test_case)

    return unittest.skipUnless(torch.xpu.device_count() > 1, "test requires multiple XPUs")(test_case)


if is_torch_available():
    # Set env var CUDA_VISIBLE_DEVICES="" to force cpu-mode
    import torch

    if "TRANSFORMERS_TEST_DEVICE" in os.environ:
        torch_device = os.environ["TRANSFORMERS_TEST_DEVICE"]
        try:
            # try creating device to see if provided device is valid
            _ = torch.device(torch_device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Unknown testing device specified by environment variable `TRANSFORMERS_TEST_DEVICE`: {torch_device}"
            ) from e
    elif torch.cuda.is_available():
        torch_device = "cuda"
    elif _run_third_party_device_tests and is_torch_npu_available():
        torch_device = "npu"
    elif _run_third_party_device_tests and is_torch_xpu_available():
        torch_device = "xpu"
    else:
        torch_device = "cpu"

    if "TRANSFORMERS_TEST_BACKEND" in os.environ:
        backend = os.environ["TRANSFORMERS_TEST_BACKEND"]
        try:
            _ = importlib.import_module(backend)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Failed to import `TRANSFORMERS_TEST_BACKEND` '{backend}'! This should be the name of an installed module. The original error (look up to see its"
                f" traceback):\n{e}"
            ) from e

else:
    torch_device = None

if is_tf_available():
    import tensorflow as tf

if is_flax_available():
    import jax

    jax_device = jax.default_backend()
else:
    jax_device = None


def require_torchdynamo(test_case):
    """Decorator marking a test that requires TorchDynamo"""
    return unittest.skipUnless(is_torchdynamo_available(), "test requires TorchDynamo")(test_case)


def require_torch_tensorrt_fx(test_case):
    """Decorator marking a test that requires Torch-TensorRT FX"""
    return unittest.skipUnless(is_torch_tensorrt_fx_available(), "test requires Torch-TensorRT FX")(test_case)


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(torch_device == "cuda", "test requires CUDA")(test_case)


def require_torch_bf16_gpu(test_case):
    """Decorator marking a test that requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0"""
    return unittest.skipUnless(
        is_torch_bf16_gpu_available(),
        "test requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0",
    )(test_case)


def require_torch_bf16_cpu(test_case):
    """Decorator marking a test that requires torch>=1.10, using CPU."""
    return unittest.skipUnless(
        is_torch_bf16_cpu_available(),
        "test requires torch>=1.10, using CPU",
    )(test_case)


def require_torch_tf32(test_case):
    """Decorator marking a test that requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7."""
    return unittest.skipUnless(
        is_torch_tf32_available(), "test requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7"
    )(test_case)


def require_detectron2(test_case):
    """Decorator marking a test that requires detectron2."""
    return unittest.skipUnless(is_detectron2_available(), "test requires `detectron2`")(test_case)


def require_faiss(test_case):
    """Decorator marking a test that requires faiss."""
    return unittest.skipUnless(is_faiss_available(), "test requires `faiss`")(test_case)


def require_optuna(test_case):
    """
    Decorator marking a test that requires optuna.

    These tests are skipped when optuna isn't installed.

    """
    return unittest.skipUnless(is_optuna_available(), "test requires optuna")(test_case)


def require_ray(test_case):
    """
    Decorator marking a test that requires Ray/tune.

    These tests are skipped when Ray/tune isn't installed.

    """
    return unittest.skipUnless(is_ray_available(), "test requires Ray/tune")(test_case)


def require_sigopt(test_case):
    """
    Decorator marking a test that requires SigOpt.

    These tests are skipped when SigOpt isn't installed.

    """
    return unittest.skipUnless(is_sigopt_available(), "test requires SigOpt")(test_case)


def require_wandb(test_case):
    """
    Decorator marking a test that requires wandb.

    These tests are skipped when wandb isn't installed.

    """
    return unittest.skipUnless(is_wandb_available(), "test requires wandb")(test_case)


def require_clearml(test_case):
    """
    Decorator marking a test requires clearml.

    These tests are skipped when clearml isn't installed.

    """
    return unittest.skipUnless(is_clearml_available(), "test requires clearml")(test_case)


def require_soundfile(test_case):
    """
    Decorator marking a test that requires soundfile

    These tests are skipped when soundfile isn't installed.

    """
    return unittest.skipUnless(is_soundfile_availble(), "test requires soundfile")(test_case)


def require_deepspeed(test_case):
    """
    Decorator marking a test that requires deepspeed
    """
    return unittest.skipUnless(is_deepspeed_available(), "test requires deepspeed")(test_case)


def require_fairscale(test_case):
    """
    Decorator marking a test that requires fairscale
    """
    return unittest.skipUnless(is_fairscale_available(), "test requires fairscale")(test_case)


def require_apex(test_case):
    """
    Decorator marking a test that requires apex
    """
    return unittest.skipUnless(is_apex_available(), "test requires apex")(test_case)


def require_bitsandbytes(test_case):
    """
    Decorator for bits and bytes (bnb) dependency
    """
    return unittest.skipUnless(is_bitsandbytes_available(), "test requires bnb")(test_case)


def require_optimum(test_case):
    """
    Decorator for optimum dependency
    """
    return unittest.skipUnless(is_optimum_available(), "test requires optimum")(test_case)


def require_auto_gptq(test_case):
    """
    Decorator for auto_gptq dependency
    """
    return unittest.skipUnless(is_auto_gptq_available(), "test requires auto-gptq")(test_case)


def require_phonemizer(test_case):
    """
    Decorator marking a test that requires phonemizer
    """
    return unittest.skipUnless(is_phonemizer_available(), "test requires phonemizer")(test_case)


def require_pyctcdecode(test_case):
    """
    Decorator marking a test that requires pyctcdecode
    """
    return unittest.skipUnless(is_pyctcdecode_available(), "test requires pyctcdecode")(test_case)


def require_librosa(test_case):
    """
    Decorator marking a test that requires librosa
    """
    return unittest.skipUnless(is_librosa_available(), "test requires librosa")(test_case)


def require_essentia(test_case):
    """
    Decorator marking a test that requires essentia
    """
    return unittest.skipUnless(is_essentia_available(), "test requires essentia")(test_case)


def require_pretty_midi(test_case):
    """
    Decorator marking a test that requires pretty_midi
    """
    return unittest.skipUnless(is_pretty_midi_available(), "test requires pretty_midi")(test_case)


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def require_usr_bin_time(test_case):
    """
    Decorator marking a test that requires `/usr/bin/time`
    """
    return unittest.skipUnless(cmd_exists("/usr/bin/time"), "test requires /usr/bin/time")(test_case)


def require_sudachi(test_case):
    """
    Decorator marking a test that requires sudachi
    """
    return unittest.skipUnless(is_sudachi_available(), "test requires sudachi")(test_case)


def require_jumanpp(test_case):
    """
    Decorator marking a test that requires jumanpp
    """
    return unittest.skipUnless(is_jumanpp_available(), "test requires jumanpp")(test_case)


def require_cython(test_case):
    """
    Decorator marking a test that requires jumanpp
    """
    return unittest.skipUnless(is_cython_available(), "test requires cython")(test_case)


def get_gpu_count():
    """
    Return the number of available gpus (regardless of whether torch, tf or jax is used)
    """
    if is_torch_available():
        import torch

        return torch.cuda.device_count()
    elif is_tf_available():
        import tensorflow as tf

        return len(tf.config.list_physical_devices("GPU"))
    elif is_flax_available():
        import jax

        return jax.device_count()
    else:
        return 0


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


#
# Helper functions for dealing with testing text outputs
# The original code came from:
# https://github.com/fastai/fastai/blob/master/tests/utils/text.py


# When any function contains print() calls that get overwritten, like progress bars,
# a special care needs to be applied, since under pytest -s captured output (capsys
# or contextlib.redirect_stdout) contains any temporary printed strings, followed by
# \r's. This helper function ensures that the buffer will contain the same output
# with and without -s in pytest, by turning:
# foo bar\r tar mar\r final message
# into:
# final message
# it can handle a single string or a multiline buffer
def apply_print_resets(buf):
    return re.sub(r"^.*\r", "", buf, 0, re.M)


def assert_screenout(out, what):
    out_pr = apply_print_resets(out).lower()
    match_str = out_pr.find(what.lower())
    assert match_str != -1, f"expecting to find {what} in output: f{out_pr}"


class CaptureStd:
    """
    Context manager to capture:

        - stdout: replay it, clean it up and make it available via `obj.out`
        - stderr: replay it and make it available via `obj.err`

    Args:
        out (`bool`, *optional*, defaults to `True`): Whether to capture stdout or not.
        err (`bool`, *optional*, defaults to `True`): Whether to capture stderr or not.
        replay (`bool`, *optional*, defaults to `True`): Whether to replay or not.
            By default each captured stream gets replayed back on context's exit, so that one can see what the test was
            doing. If this is a not wanted behavior and the captured data shouldn't be replayed, pass `replay=False` to
            disable this feature.

    Examples:

    ```python
    # to capture stdout only with auto-replay
    with CaptureStdout() as cs:
        print("Secret message")
    assert "message" in cs.out

    # to capture stderr only with auto-replay
    import sys

    with CaptureStderr() as cs:
        print("Warning: ", file=sys.stderr)
    assert "Warning" in cs.err

    # to capture both streams with auto-replay
    with CaptureStd() as cs:
        print("Secret message")
        print("Warning: ", file=sys.stderr)
    assert "message" in cs.out
    assert "Warning" in cs.err

    # to capture just one of the streams, and not the other, with auto-replay
    with CaptureStd(err=False) as cs:
        print("Secret message")
    assert "message" in cs.out
    # but best use the stream-specific subclasses

    # to capture without auto-replay
    with CaptureStd(replay=False) as cs:
        print("Secret message")
    assert "message" in cs.out
    ```"""

    def __init__(self, out=True, err=True, replay=True):
        self.replay = replay

        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

    def __enter__(self):
        if self.out_buf:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        if self.err_buf:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        return self

    def __exit__(self, *exc):
        if self.out_buf:
            sys.stdout = self.out_old
            captured = self.out_buf.getvalue()
            if self.replay:
                sys.stdout.write(captured)
            self.out = apply_print_resets(captured)

        if self.err_buf:
            sys.stderr = self.err_old
            captured = self.err_buf.getvalue()
            if self.replay:
                sys.stderr.write(captured)
            self.err = captured

    def __repr__(self):
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg


# in tests it's the best to capture only the stream that's wanted, otherwise
# it's easy to miss things, so unless you need to capture both streams, use the
# subclasses below (less typing). Or alternatively, configure `CaptureStd` to
# disable the stream you don't need to test.


class CaptureStdout(CaptureStd):
    """Same as CaptureStd but captures only stdout"""

    def __init__(self, replay=True):
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):
    """Same as CaptureStd but captures only stderr"""

    def __init__(self, replay=True):
        super().__init__(out=False, replay=replay)


class CaptureLogger:
    """
    Context manager to capture `logging` streams

    Args:
        logger: 'logging` logger object

    Returns:
        The captured output is available via `self.out`

    Example:

    ```python
    >>> from transformers import logging
    >>> from transformers.testing_utils import CaptureLogger

    >>> msg = "Testing 1, 2, 3"
    >>> logging.set_verbosity_info()
    >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart")
    >>> with CaptureLogger(logger) as cl:
    ...     logger.info(msg)
    >>> assert cl.out, msg + "\n"
    ```
    """

    def __init__(self, logger):
        self.logger = logger
        self.io = StringIO()
        self.sh = logging.StreamHandler(self.io)
        self.out = ""

    def __enter__(self):
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.sh)
        self.out = self.io.getvalue()

    def __repr__(self):
        return f"captured: {self.out}\n"


@contextlib.contextmanager
def LoggingLevel(level):
    """
    This is a context manager to temporarily change transformers modules logging level to the desired value and have it
    restored to the original setting at the end of the scope.

    Example:

    ```python
    with LoggingLevel(logging.INFO):
        AutoModel.from_pretrained("gpt2")  # calls logger.info() several times
    ```
    """
    orig_level = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity(level)
        yield
    finally:
        transformers_logging.set_verbosity(orig_level)


@contextlib.contextmanager
# adapted from https://stackoverflow.com/a/64789046/9201239
def ExtendSysPath(path: Union[str, os.PathLike]) -> Iterator[None]:
    """
    Temporary add given path to `sys.path`.

    Usage :

    ```python
    with ExtendSysPath("/path/to/dir"):
        mymodule = importlib.import_module("mymodule")
    ```
    """

    path = os.fspath(path)
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)


class TestCasePlus(unittest.TestCase):
    """
    This class extends *unittest.TestCase* with additional features.

    Feature 1: A set of fully resolved important file and dir path accessors.

    In tests often we need to know where things are relative to the current test file, and it's not trivial since the
    test could be invoked from more than one directory or could reside in sub-directories with different depths. This
    class solves this problem by sorting out all the basic paths and provides easy accessors to them:

    - `pathlib` objects (all fully resolved):

       - `test_file_path` - the current test file path (=`__file__`)
       - `test_file_dir` - the directory containing the current test file
       - `tests_dir` - the directory of the `tests` test suite
       - `examples_dir` - the directory of the `examples` test suite
       - `repo_root_dir` - the directory of the repository
       - `src_dir` - the directory of `src` (i.e. where the `transformers` sub-dir resides)

    - stringified paths---same as above but these return paths as strings, rather than `pathlib` objects:

       - `test_file_path_str`
       - `test_file_dir_str`
       - `tests_dir_str`
       - `examples_dir_str`
       - `repo_root_dir_str`
       - `src_dir_str`

    Feature 2: Flexible auto-removable temporary dirs which are guaranteed to get removed at the end of test.

    1. Create a unique temporary dir:

    ```python
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
    ```

    `tmp_dir` will contain the path to the created temporary dir. It will be automatically removed at the end of the
    test.


    2. Create a temporary dir of my choice, ensure it's empty before the test starts and don't
    empty it after the test.

    ```python
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
    ```

    This is useful for debug when you want to monitor a specific directory and want to make sure the previous tests
    didn't leave any data in there.

    3. You can override the first two options by directly overriding the `before` and `after` args, leading to the
        following behavior:

    `before=True`: the temporary dir will always be cleared at the beginning of the test.

    `before=False`: if the temporary dir already existed, any existing files will remain there.

    `after=True`: the temporary dir will always be deleted at the end of the test.

    `after=False`: the temporary dir will always be left intact at the end of the test.

    Note 1: In order to run the equivalent of `rm -r` safely, only subdirs of the project repository checkout are
    allowed if an explicit `tmp_dir` is used, so that by mistake no `/tmp` or similar important part of the filesystem
    will get nuked. i.e. please always pass paths that start with `./`

    Note 2: Each test can register multiple temporary dirs and they all will get auto-removed, unless requested
    otherwise.

    Feature 3: Get a copy of the `os.environ` object that sets up `PYTHONPATH` specific to the current test suite. This
    is useful for invoking external programs from the test suite - e.g. distributed training.


    ```python
    def test_whatever(self):
        env = self.get_env()
    ```"""

    def setUp(self):
        # get_auto_remove_tmp_dir feature:
        self.teardown_tmp_dirs = []

        # figure out the resolved paths for repo_root, tests, examples, etc.
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self._test_file_dir = path.parents[0]
        for up in [1, 2, 3]:
            tmp_dir = path.parents[up]
            if (tmp_dir / "src").is_dir() and (tmp_dir / "tests").is_dir():
                break
        if tmp_dir:
            self._repo_root_dir = tmp_dir
        else:
            raise ValueError(f"can't figure out the root of the repo from {self._test_file_path}")
        self._tests_dir = self._repo_root_dir / "tests"
        self._examples_dir = self._repo_root_dir / "examples"
        self._src_dir = self._repo_root_dir / "src"

    @property
    def test_file_path(self):
        return self._test_file_path

    @property
    def test_file_path_str(self):
        return str(self._test_file_path)

    @property
    def test_file_dir(self):
        return self._test_file_dir

    @property
    def test_file_dir_str(self):
        return str(self._test_file_dir)

    @property
    def tests_dir(self):
        return self._tests_dir

    @property
    def tests_dir_str(self):
        return str(self._tests_dir)

    @property
    def examples_dir(self):
        return self._examples_dir

    @property
    def examples_dir_str(self):
        return str(self._examples_dir)

    @property
    def repo_root_dir(self):
        return self._repo_root_dir

    @property
    def repo_root_dir_str(self):
        return str(self._repo_root_dir)

    @property
    def src_dir(self):
        return self._src_dir

    @property
    def src_dir_str(self):
        return str(self._src_dir)

    def get_env(self):
        """
        Return a copy of the `os.environ` object that sets up `PYTHONPATH` correctly, depending on the test suite it's
        invoked from. This is useful for invoking external programs from the test suite - e.g. distributed training.

        It always inserts `./src` first, then `./tests` or `./examples` depending on the test suite type and finally
        the preset `PYTHONPATH` if any (all full resolved paths).

        """
        env = os.environ.copy()
        paths = [self.src_dir_str]
        if "/examples" in self.test_file_dir_str:
            paths.append(self.examples_dir_str)
        else:
            paths.append(self.tests_dir_str)
        paths.append(env.get("PYTHONPATH", ""))

        env["PYTHONPATH"] = ":".join(paths)
        return env

    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
        """
        Args:
            tmp_dir (`string`, *optional*):
                if `None`:

                   - a unique temporary path will be created
                   - sets `before=True` if `before` is `None`
                   - sets `after=True` if `after` is `None`
                else:

                   - `tmp_dir` will be created
                   - sets `before=True` if `before` is `None`
                   - sets `after=False` if `after` is `None`
            before (`bool`, *optional*):
                If `True` and the `tmp_dir` already exists, make sure to empty it right away if `False` and the
                `tmp_dir` already exists, any existing files will remain there.
            after (`bool`, *optional*):
                If `True`, delete the `tmp_dir` at the end of the test if `False`, leave the `tmp_dir` and its contents
                intact at the end of the test.

        Returns:
            tmp_dir(`string`): either the same value as passed via *tmp_dir* or the path to the auto-selected tmp dir
        """
        if tmp_dir is not None:
            # defining the most likely desired behavior for when a custom path is provided.
            # this most likely indicates the debug mode where we want an easily locatable dir that:
            # 1. gets cleared out before the test (if it already exists)
            # 2. is left intact after the test
            if before is None:
                before = True
            if after is None:
                after = False

            # using provided path
            path = Path(tmp_dir).resolve()

            # to avoid nuking parts of the filesystem, only relative paths are allowed
            if not tmp_dir.startswith("./"):
                raise ValueError(
                    f"`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`"
                )

            # ensure the dir is empty to start with
            if before is True and path.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            path.mkdir(parents=True, exist_ok=True)

        else:
            # defining the most likely desired behavior for when a unique tmp path is auto generated
            # (not a debug mode), here we require a unique tmp dir that:
            # 1. is empty before the test (it will be empty in this situation anyway)
            # 2. gets fully removed after the test
            if before is None:
                before = True
            if after is None:
                after = True

            # using unique tmp dir (always empty, regardless of `before`)
            tmp_dir = tempfile.mkdtemp()

        if after is True:
            # register for deletion
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir

    def python_one_liner_max_rss(self, one_liner_str):
        """
        Runs the passed python one liner (just the code) and returns how much max cpu memory was used to run the
        program.

        Args:
            one_liner_str (`string`):
                a python one liner code that gets passed to `python -c`

        Returns:
            max cpu memory bytes used to run the program. This value is likely to vary slightly from run to run.

        Requirements:
            this helper needs `/usr/bin/time` to be installed (`apt install time`)

        Example:

        ```
        one_liner_str = 'from transformers import AutoModel; AutoModel.from_pretrained("t5-large")'
        max_rss = self.python_one_liner_max_rss(one_liner_str)
        ```
        """

        if not cmd_exists("/usr/bin/time"):
            raise ValueError("/usr/bin/time is required, install with `apt install time`")

        cmd = shlex.split(f"/usr/bin/time -f %M python -c '{one_liner_str}'")
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        # returned data is in KB so convert to bytes
        max_rss = int(cs.err.split("\n")[-2].replace("stderr: ", "")) * 1024
        return max_rss

    def tearDown(self):
        # get_auto_remove_tmp_dir feature: remove registered temp dirs
        for path in self.teardown_tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.teardown_tmp_dirs = []
        if is_accelerate_available():
            AcceleratorState._reset_state()
            PartialState._reset_state()

            # delete all the env variables having `ACCELERATE` in them
            for k in list(os.environ.keys()):
                if "ACCELERATE" in k:
                    del os.environ[k]


def mockenv(**kwargs):
    """
    this is a convenience wrapper, that allows this ::

    @mockenv(RUN_SLOW=True, USE_TF=False) def test_something():
        run_slow = os.getenv("RUN_SLOW", False) use_tf = os.getenv("USE_TF", False)

    """
    return mock.patch.dict(os.environ, kwargs)


# from https://stackoverflow.com/a/34333710/9201239
@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    Temporarily updates the `os.environ` dictionary in-place. Similar to mockenv

    The `os.environ` dictionary is updated in-place so that the modification is sure to work in all situations.

    Args:
      remove: Environment variables to remove.
      update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


# --- pytest conf functions --- #

# to avoid multiple invocation from tests/conftest.py and examples/conftest.py - make sure it's called only once
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store",
            default=False,
            help="generate report files. The value of this option is used as a prefix to report names",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:
    - tr: `terminalreporter` passed from `conftest.py`
    - id: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.

    NB: this functions taps into a private _pytest API and while unlikely, it could break should pytest do internal
    changes - also it calls default internal methods of terminalreporter which can be hijacked by various `pytest-`
    plugins and interfere.

    """
    from _pytest.config import create_terminal_writer

    if not len(id):
        id = "tests"

    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars

    dir = f"reports/{id}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    report_files = {
        k: f"{dir}/{k}.txt"
        for k in [
            "durations",
            "errors",
            "failures_long",
            "failures_short",
            "failures_line",
            "passes",
            "stats",
            "summary_short",
            "warnings",
        ]
    }

    # custom durations report
    # note: there is no need to call pytest --durations=XX to get this separate report
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # sec
            f.write("slowest durations\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    def summary_failures_short(tr):
        # expecting that the reports were --tb=long (default) so we chop them off here to the last frame
        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "FAILURES SHORT STACK")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # chop off the optional leading extra frames, leaving only the last one
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # note: not printing out any rep.sections to keep the report short

    # use ready-made report funcs, we are just hijacking the filehandle to log to a dedicated file each
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # note: some pytest plugins may interfere by hijacking the default `terminalreporter` (e.g.
    # pytest-instafail does that)

    # report failures with line/short/long styles
    config.option.tbstyle = "auto"  # full tb
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    # config.option.tbstyle = "short" # short tb
    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # one line per error
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # normal warnings
        tr.summary_warnings()  # final warnings

    tr.reportchars = "wPpsxXEf"  # emulate -rA (used in summary_passes() and short_test_summary())

    # Skip the `passes` report, as it starts to take more than 5 minutes, and sometimes it timeouts on CircleCI if it
    # takes > 10 minutes (as this part doesn't generate any output on the terminal).
    # (also, it seems there is no useful information in this report, and we rarely need to read it)
    # with open(report_files["passes"], "w") as f:
    #     tr._tw = create_terminal_writer(config, f)
    #     tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # restore:
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle


# --- distributed testing functions --- #

# adapted from https://stackoverflow.com/a/59041913/9201239
import asyncio  # noqa


class _RunOutput:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    if echo:
        print("\nRunning: ", " ".join(cmd))

    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # note: there is a warning for a possible deadlock when using `wait` with huge amounts of data in the pipe
    # https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process.wait
    #
    # If it starts hanging, will need to switch to the following code. The problem is that no data
    # will be seen until it's done and if it hangs for example there will be no debug info.
    # out, err = await p.communicate()
    # return _RunOutput(p.returncode, out, err)

    out = []
    err = []

    def tee(line, sink, pipe, label=""):
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    # XXX: the timeout doesn't seem to make any difference here
    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda l: tee(l, out, sys.stdout, label="stdout:")),
            _read_stream(p.stderr, lambda l: tee(l, err, sys.stderr, label="stderr:")),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)


def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    # check that the subprocess actually did run and produced some output, should the test rely on
    # the remote side to do the testing
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result


def pytest_xdist_worker_id():
    """
    Returns an int value of worker's numerical id under `pytest-xdist`'s concurrent workers `pytest -n N` regime, or 0
    if `-n 1` or `pytest-xdist` isn't being used.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker = re.sub(r"^gw", "", worker, 0, re.M)
    return int(worker)


def get_torch_dist_unique_port():
    """
    Returns a port number that can be fed to `torch.distributed.launch`'s `--master_port` argument.

    Under `pytest-xdist` it adds a delta number based on a worker id so that concurrent tests don't try to use the same
    port at once.
    """
    port = 29500
    uniq_delta = pytest_xdist_worker_id()
    return port + uniq_delta


def nested_simplify(obj, decimals=3):
    """
    Simplifies an object by rounding float numbers, and downcasting tensors/numpy arrays to get simple equality test
    within tests.
    """
    import numpy as np

    if isinstance(obj, list):
        return [nested_simplify(item, decimals) for item in obj]
    if isinstance(obj, tuple):
        return tuple([nested_simplify(item, decimals) for item in obj])
    elif isinstance(obj, np.ndarray):
        return nested_simplify(obj.tolist())
    elif isinstance(obj, Mapping):
        return {nested_simplify(k, decimals): nested_simplify(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (str, int, np.int64)):
        return obj
    elif obj is None:
        return obj
    elif is_torch_available() and isinstance(obj, torch.Tensor):
        return nested_simplify(obj.tolist(), decimals)
    elif is_tf_available() and tf.is_tensor(obj):
        return nested_simplify(obj.numpy().tolist())
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, (np.int32, np.float32)):
        return nested_simplify(obj.item(), decimals)
    else:
        raise Exception(f"Not supported: {type(obj)}")


def check_json_file_has_correct_format(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        if len(lines) == 1:
            # length can only be 1 if dict is empty
            assert lines[0] == "{}"
        else:
            # otherwise make sure json has correct format (at least 3 lines)
            assert len(lines) >= 3
            # each key one line, ident should be 2, min length is 3
            assert lines[0].strip() == "{"
            for line in lines[1:-1]:
                left_indent = len(lines[1]) - len(lines[1].lstrip())
                assert left_indent == 2
            assert lines[-1].strip() == "}"


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# These utils relate to ensuring the right error message is received when running scripts
class SubprocessCallException(Exception):
    pass


def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occured while running `command`
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


class RequestCounter:
    """
    Helper class that will count all requests made online.
    """

    def __enter__(self):
        self.head_request_count = 0
        self.get_request_count = 0
        self.other_request_count = 0

        # Mock `get_session` to count HTTP calls.
        self.old_get_session = huggingface_hub.utils._http.get_session
        self.session = requests.Session()
        self.session.request = self.new_request
        huggingface_hub.utils._http.get_session = lambda: self.session
        return self

    def __exit__(self, *args, **kwargs):
        huggingface_hub.utils._http.get_session = self.old_get_session

    def new_request(self, method, **kwargs):
        if method == "GET":
            self.get_request_count += 1
        elif method == "HEAD":
            self.head_request_count += 1
        else:
            self.other_request_count += 1

        return requests.request(method=method, **kwargs)


def is_flaky(max_attempts: int = 5, wait_before_retry: Optional[float] = None, description: Optional[str] = None):
    """
    To decorate flaky tests. They will be retried on failures.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """

    def decorator(test_func_ref):
        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    return test_func_ref(*args, **kwargs)

                except Exception as err:
                    print(f"Test failed with {err} at try {retry_count}/{max_attempts}.", file=sys.stderr)
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1

            return test_func_ref(*args, **kwargs)

        return wrapper

    return decorator


def run_test_in_subprocess(test_case, target_func, inputs=None, timeout=None):
    """
    To run a test in a subprocess. In particular, this can avoid (GPU) memory issue.

    Args:
        test_case (`unittest.TestCase`):
            The test that will run `target_func`.
        target_func (`Callable`):
            The function implementing the actual testing logic.
        inputs (`dict`, *optional*, defaults to `None`):
            The inputs that will be passed to `target_func` through an (input) queue.
        timeout (`int`, *optional*, defaults to `None`):
            The timeout (in seconds) that will be passed to the input and output queues. If not specified, the env.
            variable `PYTEST_TIMEOUT` will be checked. If still `None`, its value will be set to `600`.
    """
    if timeout is None:
        timeout = int(os.environ.get("PYTEST_TIMEOUT", 600))

    start_methohd = "spawn"
    ctx = multiprocessing.get_context(start_methohd)

    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)

    # We can't send `unittest.TestCase` to the child, otherwise we get issues regarding pickle.
    input_queue.put(inputs, timeout=timeout)

    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()
    # Kill the child process if we can't get outputs from it in time: otherwise, the hanging subprocess prevents
    # the test to exit properly.
    try:
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    except Exception as e:
        process.terminate()
        test_case.fail(e)
    process.join(timeout=timeout)

    if results["error"] is not None:
        test_case.fail(f'{results["error"]}')


"""
The following contains utils to run the documentation tests without having to overwrite any files.

The `preprocess_string` function adds `# doctest: +IGNORE_RESULT` markers on the fly anywhere a `load_dataset` call is
made as a print would otherwise fail the corresonding line.

To skip cuda tests, make sure to call `SKIP_CUDA_DOCTEST=1 pytest --doctest-modules <path_to_files_to_test>
"""


def preprocess_string(string, skip_cuda_tests):
    """Prepare a docstring or a `.md` file to be run by doctest.

    The argument `string` would be the whole file content if it is a `.md` file. For a python file, it would be one of
    its docstring. In each case, it may contain multiple python code examples. If `skip_cuda_tests` is `True` and a
    cuda stuff is detective (with a heuristic), this method will return an empty string so no doctest will be run for
    `string`.
    """
    codeblock_pattern = r"(```(?:python|py)\s*\n\s*>>> )((?:.*?\n)*?.*?```)"
    codeblocks = re.split(re.compile(codeblock_pattern, flags=re.MULTILINE | re.DOTALL), string)
    is_cuda_found = False
    for i, codeblock in enumerate(codeblocks):
        if "load_dataset(" in codeblock and "# doctest: +IGNORE_RESULT" not in codeblock:
            codeblocks[i] = re.sub(r"(>>> .*load_dataset\(.*)", r"\1 # doctest: +IGNORE_RESULT", codeblock)
        if (
            (">>>" in codeblock or "..." in codeblock)
            and re.search(r"cuda|to\(0\)|device=0", codeblock)
            and skip_cuda_tests
        ):
            is_cuda_found = True
            break

    modified_string = ""
    if not is_cuda_found:
        modified_string = "".join(codeblocks)

    return modified_string


class HfDocTestParser(doctest.DocTestParser):
    """
    Overwrites the DocTestParser from doctest to properly parse the codeblocks that are formatted with black. This
    means that there are no extra lines at the end of our snippets. The `# doctest: +IGNORE_RESULT` marker is also
    added anywhere a `load_dataset` call is made as a print would otherwise fail the corresponding line.

    Tests involving cuda are skipped base on a naive pattern that should be updated if it is not enough.
    """

    # This regular expression is used to find doctest examples in a
    # string.  It defines three groups: `source` is the source code
    # (including leading indentation and prompts); `indent` is the
    # indentation of the first (PS1) line of the source code; and
    # `want` is the expected output (including leading indentation).
    # fmt: off
    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
             (?![ ]*>>>)          # Not a line starting with PS1
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:(?!```).)*        # Match any character except '`' until a '```' is found (this is specific to HF because black removes the last line)
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:\n|$)  # Match a new line or end of string
          )*)
        ''', re.MULTILINE | re.VERBOSE
    )
    # fmt: on

    # !!!!!!!!!!! HF Specific !!!!!!!!!!!
    skip_cuda_tests: bool = bool(os.environ.get("SKIP_CUDA_DOCTEST", False))
    # !!!!!!!!!!! HF Specific !!!!!!!!!!!

    def parse(self, string, name="<string>"):
        """
        Overwrites the `parse` method to incorporate a skip for CUDA tests, and remove logs and dataset prints before
        calling `super().parse`
        """
        string = preprocess_string(string, self.skip_cuda_tests)
        return super().parse(string, name)


class HfDoctestModule(Module):
    """
    Overwrites the `DoctestModule` of the pytest package to make sure the HFDocTestParser is used when discovering
    tests.
    """

    def collect(self) -> Iterable[DoctestItem]:
        class MockAwareDocTestFinder(doctest.DocTestFinder):
            """A hackish doctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456 https://bugs.python.org/issue25532
            """

            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct line number is returned. This will be
                reported upstream. #8796
                """
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)

                if hasattr(obj, "__wrapped__"):
                    # Get the main obj in case of it being wrapped
                    obj = inspect.unwrap(obj)

                # Type ignored because this is a private function.
                return super()._find_lineno(  # type:ignore[misc]
                    obj,
                    source_lines,
                )

            def _find(self, tests, obj, name, module, source_lines, globs, seen) -> None:
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    # Type ignored because this is a private function.
                    super()._find(  # type:ignore[misc]
                        tests, obj, name, module, source_lines, globs, seen
                    )

        if self.path.name == "conftest.py":
            module = self.config.pluginmanager._importconftest(
                self.path,
                self.config.getoption("importmode"),
                rootpath=self.config.rootpath,
            )
        else:
            try:
                module = import_path(
                    self.path,
                    root=self.config.rootpath,
                    mode=self.config.getoption("importmode"),
                )
            except ImportError:
                if self.config.getvalue("doctest_ignore_import_errors"):
                    skip("unable to import module %r" % self.path)
                else:
                    raise

        # !!!!!!!!!!! HF Specific !!!!!!!!!!!
        finder = MockAwareDocTestFinder(parser=HfDocTestParser())
        # !!!!!!!!!!! HF Specific !!!!!!!!!!!
        optionflags = get_optionflags(self)
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )
        for test in finder.find(module, module.__name__):
            if test.examples:  # skip empty doctests and cuda
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)
