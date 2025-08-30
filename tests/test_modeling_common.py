# Copyright 2019 HuggingFace Inc.
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
import copy
import gc
import inspect
import math
import os
import os.path
import random
import re
import tempfile
import unittest
import warnings
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pytest
from packaging import version
from parameterized import parameterized
from pytest import mark

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorWithFlattening,
    PretrainedConfig,
    PreTrainedModel,
    is_torch_available,
    logging,
    set_seed,
)
from transformers.integrations import HfDeepSpeedConfig
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)
from transformers.modeling_utils import _get_tied_weight_keys
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.testing_utils import (
    CaptureLogger,
    backend_device_count,
    backend_empty_cache,
    backend_memory_allocated,
    backend_torch_accelerator_module,
    get_device_properties,
    hub_retry,
    is_flaky,
    require_accelerate,
    require_bitsandbytes,
    require_deepspeed,
    require_flash_attn,
    require_flash_attn_3,
    require_kernels,
    require_non_hpu,
    require_safetensors,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_greater_or_equal,
    require_torch_mps,
    require_torch_multi_accelerator,
    require_torch_multi_gpu,
    run_first,
    run_test_using_subprocess,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import (
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    is_accelerate_available,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
)
from transformers.utils.generic import ContextManagers

from .generation.test_utils import GenerationTesterMixin


if is_accelerate_available():
    from accelerate.utils import compute_module_sizes


if is_torch_available():
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
    from torch import nn

    from transformers import MODEL_MAPPING
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.modeling_utils import load_state_dict, no_init_weights
    from transformers.pytorch_utils import id_tensor_storage

from transformers.utils.fx import _FX_SUPPORTED_MODELS_WITH_KV_CACHE, symbolic_trace


if is_deepspeed_available():
    import deepspeed


# used in other test files e.g. when overwriting the test
TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION = [
    (
        # test name for the test runner
        f"{dtype}_pad_{padding_side}{'' if use_attention_mask else '_no_attn_mask'}"
        f"{'_sdpa_kernels' if enable_kernels else ''}",
        # parameterization
        *(dtype, padding_side, use_attention_mask, False, enable_kernels),
    )
    for dtype in ("fp16", "fp32", "bf16")
    for padding_side in ("left", "right")
    for use_attention_mask in (True, False)
    for enable_kernels in (True, False)
    # Extra test case: `output_attentions=True` has special attention mask handling and sdpa reverts to eager
] + [("fp32_pad_left_output_attentions", "fp32", "left", True, True, False)]


def _test_eager_matches_sdpa_inference(
    self,
    name,
    dtype,
    padding_side,
    use_attention_mask,
    output_attentions,
    enable_kernels,
    atols=None,
    rtols=None,
):
    """
    This test is written as a regular function to be able to overload it easily with different tolerances.
    Otherwise, `paramterezie.expand` prevents it as it removes the original function from the namespace.
    """
    # TODO: we shouldn't need to do this skip, i.e. the test would be composable from the model tester. CLIP-like
    # models have a custom mixin, which we detect to skip this test.
    if any(".CLIPModelTesterMixin" in str(base) for base in self.__class__.__bases__):
        self.skipTest(reason="CLIP-like models have a different `test_eager_matches_sdpa_inference`")

    if not self.has_attentions:
        self.skipTest(reason="Model architecture does not support attentions")

    if not self.all_model_classes[0]._supports_sdpa:
        self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

    # convert shorthand name to torch.dtype
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32

    if not is_torch_fp16_available_on_device(torch_device) and dtype == torch.float16:
        self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

    if not is_torch_bf16_available_on_device(torch_device) and dtype == torch.bfloat16:
        self.skipTest(
            f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
        )

    # Dictionary of tolerances for eager <> sdpa tests. Key = (device, sdpa_kernels_enabled, dtype)
    if atols is None:
        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
    if rtols is None:
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,  # (different from others)
            ("cuda", True, torch.float16): 5e-3,
        }

    set_model_tester_for_less_flaky_test(self)

    for model_class in self.all_model_classes:
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)
        model = model_class(config)
        # TODO: standardize the interfaces for musicgen models, see other todo in this test
        if model.__class__.__name__ == "MusicgenMelodyForConditionalGeneration":
            is_encoder_decoder = True
        else:
            is_encoder_decoder = model.config.is_encoder_decoder

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model_from_pretrained_kwargs = {
                "pretrained_model_name_or_path": tmpdirname,
                "dtype": dtype,
            }

            if hasattr(config, "use_mask_token") or "use_mask_token" in inspect.signature(model.__init__).parameters:
                model_from_pretrained_kwargs["use_mask_token"] = True

            # TODO: remove this try/except, models should have a shared API
            try:
                model_sdpa = model_class.from_pretrained(**model_from_pretrained_kwargs, attn_implementation="sdpa")
            except ValueError:
                model_sdpa = model_class.from_pretrained(**model_from_pretrained_kwargs)
            model_sdpa = model_sdpa.eval().to(torch_device)

            model_eager = model_class.from_pretrained(**model_from_pretrained_kwargs, attn_implementation="eager")
            model_eager = model_eager.eval().to(torch_device)

        set_model_for_less_flaky_test(model_eager)
        set_model_for_less_flaky_test(model_sdpa)

        can_output_attn = "output_attentions" in inspect.signature(model_sdpa.forward).parameters
        if not (self.has_attentions and can_output_attn) and output_attentions:
            self.skipTest(reason="Model does not support output_attentions")

        # TODO: if we can also check with `batch_size=1` without being flaky?
        for batch_size in [7]:
            # musicgen decoder models; TODO: find better abstraction
            if (
                model.__class__.__name__.startswith("Musicgen")
                and hasattr(self.model_tester, "num_codebooks")
                and not hasattr(model_eager, "text_encoder")
            ):
                input_data_batch_size = batch_size * self.model_tester.num_codebooks
            else:
                input_data_batch_size = batch_size

            processed_inputs = {}
            processed_inputs[model.main_input_name] = inputs_dict[model.main_input_name]

            for key in getattr(self, "additional_model_inputs", []):
                # Some models don't have all `additional_model_inputs`, especially when we
                # craft cases to test model in different settings
                if key in inputs_dict:
                    processed_inputs[key] = inputs_dict[key]

            for key, value in processed_inputs.items():
                if torch.is_floating_point(value):
                    value = value.to(dtype)

                # extend value to have at least `input_data_batch_size` elements
                if value.shape[0] < input_data_batch_size:
                    size = (input_data_batch_size - value.shape[0], *value.shape[1:])
                    if torch.is_floating_point(value):
                        extension = torch.rand(size=size, dtype=value.dtype, device=torch_device)
                    else:
                        extension = torch.randint(high=5, size=size, dtype=value.dtype, device=torch_device)
                    value = torch.cat((value, extension), dim=0).to(torch_device)

                processed_inputs[key] = value[:input_data_batch_size]

            if not use_attention_mask:
                dummy_attention_mask = None
            else:
                dummy_attention_mask = inputs_dict.get("attention_mask", None)
                if dummy_attention_mask is None:
                    if is_encoder_decoder:
                        seqlen = inputs_dict.get("decoder_input_ids", processed_inputs[model.main_input_name]).shape[
                            -1
                        ]
                    else:
                        seqlen = processed_inputs[model.main_input_name].shape[-1]
                    dummy_attention_mask = torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)

                # extend dummy_attention_mask to have at least `batch_size` elements
                if dummy_attention_mask.shape[0] < batch_size:
                    size = (batch_size - dummy_attention_mask.shape[0], *dummy_attention_mask.shape[1:])
                    extension = torch.ones(size=size, dtype=dummy_attention_mask.dtype, device=torch_device)
                    dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)

                dummy_attention_mask = dummy_attention_mask[:batch_size].to(torch_device)

                dummy_attention_mask[:] = 1
                if padding_side == "left":
                    dummy_attention_mask[-1, :2] = 0
                    dummy_attention_mask[-1, 2:] = 1
                elif padding_side == "right":
                    dummy_attention_mask[-1, -2:] = 0
                    dummy_attention_mask[-1, :-2] = 1

            if is_encoder_decoder:
                # musicgen encoder-decoder models; TODO: find better abstraction
                if model.__class__.__name__.startswith("Musicgen") and hasattr(self.model_tester, "num_codebooks"):
                    input_data_batch_size = batch_size * self.model_tester.num_codebooks
                else:
                    input_data_batch_size = batch_size

                decoder_input_ids = inputs_dict.get("decoder_input_ids", processed_inputs[model.main_input_name])
                decoder_input_ids = decoder_input_ids[:input_data_batch_size]
                if decoder_input_ids.shape[0] != input_data_batch_size:
                    extension = torch.ones(
                        input_data_batch_size - decoder_input_ids.shape[0],
                        *decoder_input_ids.shape[1:],
                        dtype=decoder_input_ids.dtype,
                        device=torch_device,
                    )
                    decoder_input_ids = torch.cat((decoder_input_ids, extension), dim=0)
                    decoder_input_ids = decoder_input_ids.to(torch_device)

                # TODO: never an `attention_mask` arg here?
                processed_inputs.update(
                    {
                        "decoder_input_ids": decoder_input_ids,
                        "decoder_attention_mask": dummy_attention_mask,
                        "output_hidden_states": True,
                    }
                )
            else:
                processed_inputs.update(
                    {
                        "output_hidden_states": True,
                    }
                )

                # Otherwise fails for e.g. WhisperEncoderModel
                if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                    processed_inputs["attention_mask"] = dummy_attention_mask

                if self.has_attentions and "output_attentions" in inspect.signature(model_sdpa.forward).parameters:
                    processed_inputs["output_attentions"] = output_attentions
            if "bool_masked_pos" in inspect.signature(model_eager.forward).parameters:
                dummy_mask = torch.ones((self.model_tester.num_masks,))

                # In case of additional token (like class) we define a custom `mask_length`
                if hasattr(self.model_tester, "mask_length"):
                    mask_length = self.model_tester.mask_length - dummy_mask.size(0)
                else:
                    mask_length = self.model_tester.seq_length - dummy_mask.size(0)
                dummy_mask = torch.cat([dummy_mask, torch.zeros(mask_length)])
                dummy_bool_masked_pos = dummy_mask.expand(batch_size, -1).bool()
                processed_inputs["bool_masked_pos"] = dummy_bool_masked_pos.to(torch_device)

            if "noise" in inspect.signature(model_eager.forward).parameters:
                np.random.seed(2)
                num_patches = int((self.model_tester.image_size // self.model_tester.patch_size) ** 2)
                noise = np.random.uniform(size=(batch_size, num_patches))
                processed_inputs["noise"] = torch.from_numpy(noise)

            # TODO: test gradients as well (& for FA2 as well!)
            with torch.no_grad():
                with sdpa_kernel(
                    enable_flash=enable_kernels,
                    enable_math=True,
                    enable_mem_efficient=enable_kernels,
                ):
                    prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                    prepared_inputs = {
                        k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in prepared_inputs.items()
                    }
                    outputs_eager = model_eager(**prepared_inputs)
                    outputs_sdpa = model_sdpa(**prepared_inputs)

            if "logits_per_text" in outputs_eager:
                key = "logits_per_text"
            elif "vision_hidden_states" in outputs_eager:
                key = "vision_hidden_states"
            elif "audio_values" in outputs_eager:
                key = "audio_values"
            elif "decoder_hidden_states" in outputs_eager:
                key = "decoder_hidden_states"
            elif "logits" in outputs_eager and "Classification" in model_class.__name__:
                key = "logits"
            elif "language_model_outputs" in outputs_eager and "blip" in model_class.__name__.lower():
                outputs_eager = outputs_eager["language_model_outputs"]
                outputs_sdpa = outputs_sdpa["language_model_outputs"]
                key = "hidden_states" if "hidden_states" in outputs_eager else "decoder_hidden_states"
            else:
                key = "hidden_states"

            # TODO: rename logits -> hidden_states
            logits_eager = outputs_eager[key]
            logits_sdpa = outputs_sdpa[key]

            if key in ["vision_hidden_states", "decoder_hidden_states", "hidden_states"]:
                logits_eager = logits_eager[-1]
                logits_sdpa = logits_sdpa[-1]

            if key == "logits_per_text":
                nan_mask = torch.isnan(logits_eager)
                logits_eager[nan_mask] = 0
                logits_sdpa[nan_mask] = 0

            if torch_device in ["cpu", "cuda"]:
                atol = atols[torch_device, enable_kernels, dtype]
                rtol = rtols[torch_device, enable_kernels, dtype]
            elif torch_device == "hpu":
                atol = atols["cuda", enable_kernels, dtype]
                rtol = rtols["cuda", enable_kernels, dtype]
            elif torch_device == "xpu":
                # As of PyTorch 2.5 XPU backend supports only torch.nn.attention.SDPBackend.MATH
                # which is implemented on PyTorch level using aten operators and is
                # device agnostic with respect to implementation of each aten operator.
                atol = atols["cuda", False, dtype]
                rtol = rtols["cuda", False, dtype]
            else:
                atol = 1e-7
                rtol = 1e-4

            # Masked tokens output slightly deviates - we don't mind that.
            if use_attention_mask:
                _logits_sdpa = torch.zeros_like(input=logits_sdpa)
                _logits_eager = torch.zeros_like(input=logits_eager)

                _logits_sdpa[:-1] = logits_sdpa[:-1]
                _logits_eager[:-1] = logits_eager[:-1]

                if padding_side == "left":
                    _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, 2:]
                    _logits_eager[-1:, 2:] = logits_eager[-1:, 2:]

                elif padding_side == "right":
                    _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, :-2]
                    _logits_eager[-1:, 2:] = logits_eager[-1:, :-2]

                logits_sdpa = _logits_sdpa
                logits_eager = _logits_eager

            results = [
                torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
            ]
            # If 80% batch elements have matched results, it's fine
            if np.mean(results) < 0.8:
                mean_relative_diff = ((logits_sdpa - logits_eager).abs() / (logits_eager.abs() + 1e-12)).mean()
                raise ValueError(
                    f"mean relative difference for {key}: {mean_relative_diff:.3e}, torch atol = {atol}, torch rtol = "
                    f"{rtol}"
                )


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__:
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


def _mock_init_weights(self, module):
    for name, param in module.named_parameters(recurse=False):
        # Use the first letter of the name to get a value and go from a <> -13 to z <> 12
        value = ord(name[0].lower()) - 110
        param.data.fill_(value)


def _mock_all_init_weights(self):
    # Prune heads if needed
    if self.config.pruned_heads:
        self.prune_heads(self.config.pruned_heads)

    import transformers.modeling_utils

    if transformers.modeling_utils._init_weights:
        for module in self.modules():
            module._is_hf_initialized = False
        # Initialize weights
        self.apply(self._initialize_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()


@contextmanager
def _deepspeed_zero3(ds_config):
    dschf = HfDeepSpeedConfig(ds_config)
    try:
        yield dschf
    finally:
        unset_hf_deepspeed_config()


def sdpa_kernel(enable_flash, enable_math, enable_mem_efficient):
    if version.parse(torch.__version__).release < version.parse("2.3").release:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient
        )

    backends = []
    if enable_flash:
        backends += [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
    if enable_math:
        backends += [torch.nn.attention.SDPBackend.MATH]
    if enable_mem_efficient:
        backends += [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    return torch.nn.attention.sdpa_kernel(backends)


@require_torch
class ModelTesterMixin:
    model_tester = None
    all_model_classes = ()
    fx_compatible = False
    test_torchscript = True
    test_pruning = True
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_head_masking = True
    test_mismatched_shapes = True
    test_missing_keys = True
    test_model_parallel = False
    test_torch_exportable = False
    # Used in `check_training_gradient_checkpointing` to NOT check all params having gradient (e.g. for some MOE models)
    test_all_params_have_gradient = True
    is_encoder_decoder = False
    has_attentions = True
    _is_composite = False
    model_split_percents = [0.5, 0.7, 0.9]

    # Note: for all mixins that utilize the Hub in some way, we should ensure that
    # they contain the `hub_retry` decorator in case of failures.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr_name in dir(cls):
            if attr_name.startswith("test_"):
                attr = getattr(cls, attr_name)
                if callable(attr):
                    setattr(cls, attr_name, hub_retry()(attr))

    @property
    def all_generative_model_classes(self):
        return tuple(model_class for model_class in self.all_model_classes if model_class.can_generate())

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
            inputs_dict = {
                k: v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
                if isinstance(v, torch.Tensor) and v.ndim > 1
                else v
                for k, v in inputs_dict.items()
            }
        elif model_class.__name__ in get_values(MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES):
            inputs_dict.pop("attention_mask")
        elif model_class.__name__ == MODEL_FOR_PRETRAINING_MAPPING_NAMES["hiera"]:
            config = self.model_tester.get_config()
            mask_spatial_shape = [
                i // s // ms for i, s, ms in zip(config.image_size, config.patch_stride, config.masked_unit_size)
            ]
            num_windows = math.prod(mask_spatial_shape)
            torch.manual_seed(0)
            inputs_dict["noise"] = torch.rand(self.model_tester.batch_size, num_windows)

        if return_labels:
            if model_class.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
                inputs_dict["labels"] = torch.ones(self.model_tester.batch_size, dtype=torch.long, device=torch_device)
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
                *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
            ]:
                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in [
                *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES),
                *get_values(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES),
                *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
                *get_values(MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in get_values(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES):
                num_patches = self.model_tester.image_size // self.model_tester.patch_size
                inputs_dict["bool_masked_pos"] = torch.zeros(
                    (self.model_tester.batch_size, num_patches**2), dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES):
                batch_size, num_channels, height, width = inputs_dict["pixel_values"].shape
                inputs_dict["labels"] = torch.zeros(
                    [self.model_tester.batch_size, height, width], device=torch_device
                ).long()

        return inputs_dict

    def test_save_load(self):
        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0
            out_2 = out_2[~np.isneginf(out_2)]

            out_1 = out1.cpu().numpy()
            out_1[np.isnan(out_1)] = 0
            out_1 = out_1[~np.isneginf(out_1)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(
                    model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME))
                )

                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

                # Save and load second time because `from_pretrained` adds a bunch of new config fields
                # so we need to make sure those fields can be loaded back after saving
                # Simply init as `model(config)` doesn't add those fields
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    def test_from_pretrained_no_checkpoint(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            state_dict = model.state_dict()

            new_model = model_class.from_pretrained(
                pretrained_model_name_or_path=None, config=config, state_dict=state_dict
            )
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_keep_in_fp32_modules(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if model_class._keep_in_fp32_modules is None:
                self.skipTest(reason="Model class has no _keep_in_fp32_modules attribute defined")

            model = model_class(copy.deepcopy(config))
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                model = model_class.from_pretrained(tmpdirname, dtype=torch.float16)

                for name, param in model.named_parameters():
                    if any(n in model_class._keep_in_fp32_modules for n in name.split(".")):
                        self.assertTrue(param.dtype == torch.float32)
                    else:
                        self.assertTrue(param.dtype == torch.float16, name)

    def test_save_load_keys_to_ignore_on_save(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            _keys_to_ignore_on_save = getattr(model, "_keys_to_ignore_on_save", None)
            if _keys_to_ignore_on_save is None:
                continue

            # check the keys are in the original state_dict
            for k in _keys_to_ignore_on_save:
                self.assertIn(k, model.state_dict().keys(), "\n".join(model.state_dict().keys()))

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                output_model_file = os.path.join(tmpdirname, SAFE_WEIGHTS_NAME)
                state_dict_saved = safe_load_file(output_model_file)

                for k in _keys_to_ignore_on_save:
                    self.assertNotIn(k, state_dict_saved.keys(), "\n".join(state_dict_saved.keys()))

                # Test we can load the state dict in the model, necessary for the checkpointing API in Trainer.
                load_result = model.load_state_dict(state_dict_saved, strict=False)
                keys_to_ignore = set(model._keys_to_ignore_on_save)

                if hasattr(model, "_tied_weights_keys"):
                    keys_to_ignore.update(set(model._tied_weights_keys))

                self.assertTrue(len(load_result.missing_keys) == 0 or set(load_result.missing_keys) == keys_to_ignore)
                self.assertTrue(len(load_result.unexpected_keys) == 0)

    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.gradient_checkpointing = True
            model = model_class(copy.deepcopy(config))
            self.assertTrue(model.is_gradient_checkpointing)

    def test_gradient_checkpointing_enable_disable(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(copy.deepcopy(config))
            self.assertFalse(model.is_gradient_checkpointing)

            # check enable works
            model.gradient_checkpointing_enable()
            self.assertTrue(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to True
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertTrue(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to True"
                    )

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to False
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertFalse(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to False"
                    )

    def test_peft_gradient_checkpointing_enable_disable(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(copy.deepcopy(config))
            self.assertFalse(model.is_gradient_checkpointing)

            # check enable works
            model._hf_peft_config_loaded = True
            try:
                model.gradient_checkpointing_enable()
            except NotImplementedError:
                continue

            self.assertTrue(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to True
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertTrue(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to True"
                    )

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to False
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertFalse(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to False"
                    )

    def test_can_init_all_missing_weights(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # This is used to get the addition year of the model
        filename = inspect.getfile(config.__class__)
        # No easy way to get model addition date -> check copyright year on top of file
        with open(filename) as file:
            source_code = file.read()
        addition_year = 0  # if we cannot find it, set it to 0 (i.e. oldest)
        if match_object := re.search(r"^# Copyright (\d{4})", source_code, re.MULTILINE | re.IGNORECASE):
            addition_year = int(match_object.group(1))

        for model_class in self.all_model_classes:
            # For now, skip everything older than 2024 and "important models" (too much models to patch otherwise)
            # TODO: relax this as we patch more and more models
            if addition_year < 2023:
                self.skipTest(reason=f"{model_class} is not a priorited model for now.")

            # Monkey patch the method to add a seed (we do it on PreTrainedModel._initialize_weights, which wraps
            # `_init_weights` so that it can add the seed for composite models as well)
            original_initialize_weights = PreTrainedModel._initialize_weights

            def seeded_initialize_weights(self, module):
                set_seed(0)
                original_initialize_weights(self, module)

            PreTrainedModel._initialize_weights = seeded_initialize_weights

            # First, initialize the model from config -> this ensure everything is correctly initialized, even if
            # _init_weights() does not take all weights into account correctly
            model_from_config = model_class(copy.deepcopy(config))
            # Here, passing an empty state dict will force all weights to be moved from meta to cpu, then be initialized
            # by _init_weights()
            model_from_pretrained = model_class.from_pretrained(None, config=config, state_dict={})

            # Back to original method to avoid issues if running several other tests
            PreTrainedModel._initialize_weights = original_initialize_weights

            # First, check if any parameters are still on meta -> this is usually an issue with tied weights
            params_on_meta = []
            for k, v in model_from_pretrained.named_parameters():
                if v.device.type == "meta":
                    params_on_meta.append(k)

            self.assertTrue(
                len(params_on_meta) == 0,
                f"The following keys are still on the meta device, it probably comes from an issue in the tied weights:\n{params_on_meta}",
            )

            # Everything must be exactly the same as we set the same seed for each init
            different_weights = []
            for (k1, v1), (k2, v2) in zip(
                model_from_config.state_dict().items(), model_from_pretrained.state_dict().items()
            ):
                self.assertEqual(k1, k2, "The keys from each model should be the same")

                # In case using torch.nn.utils.parametrizations on a module, we should skip the resulting keys
                if re.search(r"\.parametrizations\..*?\.original[01]", k1):
                    continue

                # Since we added the seed, they should be exactly the same (i.e. using allclose maybe be wrong due
                # to very low std in init function)
                if not (v1 == v2).all():
                    different_weights.append(k1)

            # Buffers that are initialized randomly are ignored as they are not initialized on meta device anyway
            buffer_names = {name for name, _ in model_from_config.named_buffers()}
            different_weights = [k for k in different_weights if k not in buffer_names]

            self.assertTrue(
                len(different_weights) == 0,
                f"The following keys are not properly handled by `_init_weights()`:\n{different_weights}",
            )

    def test_torch_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if config.__class__ not in MODEL_MAPPING:
            self.skipTest(reason=f"{config.__class__.__name__} not in MODEL_MAPPING")

        base_class = MODEL_MAPPING[config.__class__]

        if isinstance(base_class, tuple):
            base_class = base_class[0]

        for model_class in self.all_model_classes:
            if model_class == base_class:
                continue

            # make a copy of model class to not break future tests
            # from https://stackoverflow.com/questions/9541025/how-to-copy-a-python-class
            class CopyClass(base_class):
                pass

            base_class_copy = CopyClass

            # make sure that all keys are expected for test
            base_class_copy._keys_to_ignore_on_load_missing = []

            # make init deterministic, but make sure that
            # non-initialized weights throw errors nevertheless
            base_class_copy._init_weights = _mock_init_weights
            base_class_copy.init_weights = _mock_all_init_weights

            model = model_class(copy.deepcopy(config))
            state_dict = model.state_dict()

            def check_equal(loaded):
                for key in state_dict:
                    max_diff = torch.max(
                        state_dict()[key] ^ loaded[key]
                        if isinstance(state_dict[key], torch.BoolTensor)
                        else torch.abs(state_dict[key] - loaded[key])
                    ).item()
                    self.assertLessEqual(max_diff, 1e-6, msg=f"{key} not identical")

            # check that certain keys didn't get saved with the model
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pytorch_model.bin")
                torch.save(state_dict, pt_checkpoint_path, _use_new_zipfile_serialization=True)
                check_equal(load_state_dict(pt_checkpoint_path))
                torch.save(state_dict, pt_checkpoint_path, _use_new_zipfile_serialization=False)
                check_equal(load_state_dict(pt_checkpoint_path))

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=copy.deepcopy(configs_no_init))
            for name, param in model.named_parameters():
                if param.requires_grad:
                    data = torch.flatten(param.data)
                    n_elements = torch.numel(data)
                    # skip 2.5% of elements on each side to avoid issues caused by `nn.init.trunc_normal_` described in
                    # https://github.com/huggingface/transformers/pull/27906#issuecomment-1846951332
                    n_elements_to_skip_on_each_side = int(n_elements * 0.025)
                    data_to_check = torch.sort(data).values
                    if n_elements_to_skip_on_each_side > 0:
                        data_to_check = data_to_check[n_elements_to_skip_on_each_side:-n_elements_to_skip_on_each_side]
                    self.assertIn(
                        ((data_to_check.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            out_1 = out_1[~np.isneginf(out_1)]
            out_2 = out_2[~np.isneginf(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_batching_equivalence(self, atol=1e-5, rtol=1e-5):
        """
        Tests that the model supports batching and that the output is the nearly the same for the same input in
        different batch sizes.
        (Why "nearly the same" not "exactly the same"? Batching uses different matmul shapes, which often leads to
        different results: https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535)
        """

        def recursive_check(batched_object, single_row_object, model_name, key):
            if isinstance(batched_object, (list, tuple)):
                for batched_object_value, single_row_object_value in zip(batched_object, single_row_object):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            elif isinstance(batched_object, dict):
                for batched_object_value, single_row_object_value in zip(
                    batched_object.values(), single_row_object.values()
                ):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            # do not compare returned loss (0-dim tensor) / codebook ids (int) / caching objects
            elif batched_object is None or not isinstance(batched_object, torch.Tensor):
                return
            elif batched_object.dim() == 0:
                return
            # do not compare int or bool outputs as they are mostly computed with max/argmax/topk methods which are
            # very sensitive to the inputs (e.g. tiny differences may give totally different results)
            elif not torch.is_floating_point(batched_object):
                return
            else:
                # indexing the first element does not always work
                # e.g. models that output similarity scores of size (N, M) would need to index [0, 0]
                slice_ids = [slice(0, index) for index in single_row_object.shape]
                batched_row = batched_object[slice_ids]
                self.assertFalse(
                    torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isnan(single_row_object).any(), f"Single row output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(single_row_object).any(), f"Single row output has `inf` in {model_name} for key={key}"
                )
                try:
                    torch.testing.assert_close(batched_row, single_row_object, atol=atol, rtol=rtol)
                except AssertionError as e:
                    msg = f"Batched and Single row outputs are not equal in {model_name} for key={key}.\n\n"
                    msg += str(e)
                    raise AssertionError(msg)

        set_model_tester_for_less_flaky_test(self)

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
                config, batched_input = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            set_model_for_less_flaky_test(model)

            batch_size = self.model_tester.batch_size
            single_row_input = {}
            for key, value in batched_input_prepared.items():
                if isinstance(value, torch.Tensor) and value.shape[0] % batch_size == 0:
                    # e.g. musicgen has inputs of size (bs*codebooks). in most cases value.shape[0] == batch_size
                    single_batch_shape = value.shape[0] // batch_size
                    single_row_input[key] = value[:single_batch_shape]
                else:
                    single_row_input[key] = value

            with torch.no_grad():
                model_batched_output = model(**batched_input_prepared)
                model_row_output = model(**single_row_input)

            if isinstance(model_batched_output, torch.Tensor):
                model_batched_output = {"model_output": model_batched_output}
                model_row_output = {"model_output": model_row_output}

            for key in model_batched_output:
                # DETR starts from zero-init queries to decoder, leading to cos_similarity = `nan`
                if hasattr(self, "zero_init_hidden_state") and "decoder_hidden_states" in key:
                    model_batched_output[key] = model_batched_output[key][1:]
                    model_row_output[key] = model_row_output[key][1:]
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if (
                    model_class.__name__
                    in [
                        *get_values(MODEL_MAPPING_NAMES),
                        *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
                    ]
                    or not model_class.supports_gradient_checkpointing
                ):
                    # TODO (ydshieh): use `skipTest` once pytest-dev/pytest-subtests/pull/169 is merged
                    # self.skipTest(reason=f"`supports_gradient_checkpointing` is False for {model_class.__name__}.")
                    continue

                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                config.use_cache = False
                config.return_dict = True
                model = model_class(config)

                model.to(torch_device)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                model.train()

                # unfreeze additional layers
                for p in model.parameters():
                    p.requires_grad_(True)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                loss = model(**inputs).loss
                loss.backward()
                optimizer.step()

                if self.test_all_params_have_gradient:
                    for k, v in model.named_parameters():
                        if v.requires_grad:
                            self.assertTrue(v.grad is not None, f"{k} in {model_class.__name__} has no gradient!")

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
                *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
            ]:
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_causal_lm_can_accept_kwargs(self):
        if not getattr(self.model_tester, "is_training", False):
            self.skipTest(reason="ModelTester is not configured to run training tests")

        valid_model_class = False
        incompatible_models = (
            "MusicgenForCausalLM",
            "MusicgenMelodyForCausalLM",
            "MllamaForCausalLM",
            "CpmAntForCausalLM",
            "GotOcr2ForConditionalGeneration",
        )
        for model_class in self.all_model_classes:
            if (
                model_class.__name__ in get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
                and model_class.__name__ not in incompatible_models
            ):
                valid_model_class = True
        if not valid_model_class:
            self.skipTest(reason="No causal lm model classes found")
        for model_class in self.all_model_classes:
            model_name = model_class.__name__
            if model_name in get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES) and model_name not in incompatible_models:
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

                with tempfile.TemporaryDirectory() as tmpdir:
                    with torch.device(torch_device):
                        model_eager = AutoModelForCausalLM.from_config(config, dtype=torch.float32)

                    model_eager.save_pretrained(tmpdir)
                    model = AutoModelForCausalLM.from_pretrained(tmpdir, dtype=torch.float32, device_map=torch_device)
                    inputs_dict["num_items_in_batch"] = torch.tensor(inputs_dict["input_ids"].shape[0])
                    inputs_dict["labels"] = inputs_dict["input_ids"]
                    _ = model(**inputs_dict, return_dict=False)

    def test_training_gradient_checkpointing(self):
        # Scenario - 1 default behaviour
        self.check_training_gradient_checkpointing()

    def test_training_gradient_checkpointing_use_reentrant(self):
        # Scenario - 2 with `use_reentrant=True` - this is the default value that is used in pytorch's
        # torch.utils.checkpoint.checkpoint
        self.check_training_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": True})

    def test_training_gradient_checkpointing_use_reentrant_false(self):
        # Scenario - 3 with `use_reentrant=False` pytorch suggests users to use this value for
        # future releases: https://pytorch.org/docs/stable/checkpoint.html
        self.check_training_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            for k in config.sub_configs:
                getattr(config, k).output_attentions = True

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Question Answering model returns start_logits and end_logits
                if model_class.__name__ in [
                    *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
                    *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
                ]:
                    correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    @unittest.skip("many failing tests after #39120. Will fix when the community ask for it.")
    @slow
    def test_torchscript_simple(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self._create_and_check_torchscript(config, inputs_dict)

    @unittest.skip("many failing tests after #39120. Will fix when the community ask for it.")
    @slow
    def test_torchscript_output_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_attentions = True
        self._create_and_check_torchscript(config, inputs_dict)

    @unittest.skip("many failing tests after #39120. Will fix when the community ask for it.")
    @slow
    def test_torchscript_output_hidden_state(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        self._create_and_check_torchscript(config, inputs_dict)

    # This is copied from `torch/testing/_internal/jit_utils.py::clear_class_registry`
    def clear_torch_jit_class_registry(self):
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        # torch 1.8 has no `_clear_class_state` in `torch.jit._state`
        if hasattr(torch.jit._state, "_clear_class_state"):
            torch.jit._state._clear_class_state()

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to `False`")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            for attn_implementation in ["eager", "sdpa"]:
                if attn_implementation == "sdpa" and not model_class._supports_sdpa or config.output_attentions:
                    continue

                configs_no_init._attn_implementation = attn_implementation
                model = model_class(config=configs_no_init)
                model.to(torch_device)
                model.eval()
                inputs = self._prepare_for_class(inputs_dict, model_class)

                main_input_name = model_class.main_input_name

                try:
                    if model.config.is_encoder_decoder:
                        model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                        main_input = inputs[main_input_name]
                        attention_mask = inputs["attention_mask"]
                        decoder_input_ids = inputs["decoder_input_ids"]
                        decoder_attention_mask = inputs["decoder_attention_mask"]
                        outputs = model(main_input, attention_mask, decoder_input_ids, decoder_attention_mask)
                        # `torchscript` doesn't work with outputs containing `Cache` object. However, #35235 makes
                        # several models to use `Cache` by default instead of the legacy cache (tuple), and
                        # their `torchscript` tests are failing. We won't support them anyway, but we still want to keep
                        # the tests for encoder models like `BERT`. So we skip the checks if the model's output contains
                        # a `Cache` object.
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(
                            model, (main_input, attention_mask, decoder_input_ids, decoder_attention_mask)
                        )
                    elif "bbox" in inputs and "image" in inputs:  # LayoutLMv2 requires additional inputs
                        input_ids = inputs["input_ids"]
                        bbox = inputs["bbox"]
                        image = inputs["image"].tensor
                        outputs = model(input_ids, bbox, image)
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(
                            model, (input_ids, bbox, image), check_trace=False
                        )  # when traced model is checked, an error is produced due to name mangling
                    elif "bbox" in inputs:  # Bros requires additional inputs (bbox)
                        input_ids = inputs["input_ids"]
                        bbox = inputs["bbox"]
                        outputs = model(input_ids, bbox)
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(
                            model, (input_ids, bbox), check_trace=False
                        )  # when traced model is checked, an error is produced due to name mangling
                    elif (
                        "pixel_values" in inputs and "prompt_pixel_values" in inputs and "prompt_masks" in inputs
                    ):  # SegGpt requires additional inputs
                        pixel_values = inputs["pixel_values"]
                        prompt_pixel_values = inputs["prompt_pixel_values"]
                        prompt_masks = inputs["prompt_masks"]
                        outputs = model(pixel_values, prompt_pixel_values, prompt_masks)
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(
                            model, (pixel_values, prompt_pixel_values, prompt_masks), check_trace=False
                        )  # when traced model is checked, an error is produced due to name mangling
                    elif "Siglip2" in model_class.__name__:
                        outputs = model(**inputs)
                        example_inputs = [t for t in inputs.values() if isinstance(t, torch.Tensor)]
                        traced_model = torch.jit.trace(model, example_inputs, check_trace=False)
                    else:
                        main_input = inputs[main_input_name]
                        outputs = model(main_input)
                        if any(isinstance(x, Cache) for x in outputs):
                            continue
                        traced_model = torch.jit.trace(model, (main_input,))
                except RuntimeError:
                    self.fail("Couldn't trace module.")

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                    try:
                        torch.jit.save(traced_model, pt_file_name)
                    except Exception:
                        self.fail("Couldn't save module.")

                    try:
                        loaded_model = torch.jit.load(pt_file_name)
                    except Exception:
                        self.fail("Couldn't load module.")

                model.to(torch_device)
                model.eval()

                loaded_model.to(torch_device)
                loaded_model.eval()

                model_state_dict = model.state_dict()
                loaded_model_state_dict = loaded_model.state_dict()

                non_persistent_buffers = {}
                for key in loaded_model_state_dict:
                    if key not in model_state_dict:
                        non_persistent_buffers[key] = loaded_model_state_dict[key]

                loaded_model_state_dict = {
                    key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
                }

                self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

                model_buffers = list(model.buffers())
                for non_persistent_buffer in non_persistent_buffers.values():
                    found_buffer = False
                    for i, model_buffer in enumerate(model_buffers):
                        if torch.equal(non_persistent_buffer, model_buffer):
                            found_buffer = True
                            break

                    self.assertTrue(found_buffer)
                    model_buffers.pop(i)

                models_equal = True
                for layer_name, p1 in model_state_dict.items():
                    if layer_name in loaded_model_state_dict:
                        p2 = loaded_model_state_dict[layer_name]
                        if p1.data.ne(p2.data).sum() > 0:
                            models_equal = False

                self.assertTrue(models_equal)

                # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
                # (Even with this call, there are still memory leak by ~0.04MB)
                self.clear_torch_jit_class_registry()

    def test_torch_fx(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self._create_and_check_torch_fx_tracing(config, inputs_dict)

    def test_torch_fx_output_loss(self):
        if self.all_model_classes[0].__name__ == "BloomModel":
            self.skipTest(reason="Bloom currently has issues, @michaelbenayoun")
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self._create_and_check_torch_fx_tracing(config, inputs_dict, output_loss=True)

    def _create_and_check_torch_fx_tracing(self, config, inputs_dict, output_loss=False):
        if not self.fx_compatible:
            self.skipTest(f"The model type {config.model_type} is not compatible with torch.fx")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.return_dict = False

        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=output_loss)

            # We may want to test several inputs (various shapes, etc.).
            inputs_to_test = [inputs]

            if model.config.is_encoder_decoder:
                model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                labels = inputs.get("labels", None)
                input_names = [
                    "attention_mask",
                    "decoder_attention_mask",
                    "decoder_input_ids",
                    "input_features",
                    "input_ids",
                    "input_values",
                ]
                if labels is not None:
                    input_names.append("labels")
            else:
                input_names = [
                    "attention_mask",
                    "bbox",
                    "input_features",
                    "input_ids",
                    "input_values",
                    "inputs_embeds",
                    "pixel_values",
                    "pixel_values_videos",
                    "token_type_ids",
                    "visual_feats",
                    "visual_pos",
                    "noise",
                ]

                labels = inputs.get("labels", None)
                start_positions = inputs.get("start_positions", None)
                end_positions = inputs.get("end_positions", None)
                if labels is not None:
                    input_names.append("labels")
                if start_positions is not None:
                    input_names.append("start_positions")
                if end_positions is not None:
                    input_names.append("end_positions")

                if model.config.model_type in _FX_SUPPORTED_MODELS_WITH_KV_CACHE:
                    input_names.append("past_key_values")

                    # Generally model_tester.prepare_config_and_inputs_for_common seem not to generate past key values inputs.
                    if "past_key_values" not in inputs:
                        batch_size = inputs[next(iter(inputs))].shape[0]
                        num_heads = model.config.num_attention_heads
                        head_dim = model.config.hidden_size // model.config.num_attention_heads

                        cache_shape = (batch_size, num_heads, 0, head_dim)
                        empty_pkv = DynamicCache(config=model.config)

                        cache_length = 9
                        cache_shape = (batch_size, num_heads, cache_length, head_dim)
                        non_empty_pkv = tuple(
                            (
                                torch.rand(cache_shape, dtype=torch.float, device=torch_device),
                                torch.rand(cache_shape, dtype=torch.float, device=torch_device),
                            )
                            for i in range(model.config.num_hidden_layers)
                        )
                        non_empty_pkv = DynamicCache.from_legacy_cache(non_empty_pkv)

                        inps = copy.deepcopy(inputs_to_test[0])

                        inputs_to_test[0]["past_key_values"] = empty_pkv

                        inps["past_key_values"] = non_empty_pkv
                        inputs_to_test.append(inps)

                        past_mask = torch.ones(batch_size, cache_length, device=torch_device, dtype=torch.float)
                        inputs_to_test[1]["attention_mask"] = torch.cat(
                            (past_mask, inputs_to_test[1]["attention_mask"]), dim=1
                        )

                forward_parameters = inspect.signature(model.forward).parameters
                if "input_ids" in forward_parameters and "inputs_embeds" in forward_parameters:
                    inps = copy.deepcopy(inputs_to_test[0])

                    embedding_size = (
                        model.config.embedding_size
                        if getattr(model.config, "embedding_size", None) is not None
                        and model.config.model_type != "megatron-bert"
                        else model.config.hidden_size
                    )

                    if (
                        model.config.model_type in MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
                        and model.__class__.__name__
                        == MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES[model.config.model_type]
                    ):
                        batch_size, num_choices, sequence_length = inputs["input_ids"].shape
                        shape = (batch_size, num_choices, sequence_length, embedding_size)
                    elif inps["input_ids"].ndim == 2:
                        batch_size, sequence_length = inputs["input_ids"].shape
                        shape = (batch_size, sequence_length, embedding_size)
                    else:
                        self.skipTest("Unknown case")

                    del inps["input_ids"]
                    inps["inputs_embeds"] = torch.rand(shape, dtype=torch.float, device=torch_device)
                    inputs_to_test.append(inps)

            for inps in inputs_to_test:
                filtered_inputs = {k: v for (k, v) in inps.items() if k in input_names}
                input_names_to_trace = list(filtered_inputs.keys())

                if model.__class__.__name__ in set(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()) and (
                    not hasattr(model.config, "problem_type") or model.config.problem_type is None
                ):
                    model.config.problem_type = "single_label_classification"

                model.config.use_cache = "past_key_values" in input_names_to_trace

                traced_model = symbolic_trace(model, input_names_to_trace)

                with torch.no_grad():
                    traced_output = traced_model(**filtered_inputs)
                    model_output = model(**filtered_inputs)

                def flatten_output(output):
                    flatten = []
                    for x in output:
                        if isinstance(x, (tuple, list)):
                            flatten += flatten_output(x)
                        elif not isinstance(x, torch.Tensor):
                            continue
                        else:
                            flatten.append(x)
                    return flatten

                model_output = flatten_output(model_output)
                traced_output = flatten_output(traced_output)
                num_outputs = len(model_output)

                for i in range(num_outputs):
                    self.assertTrue(
                        torch.allclose(model_output[i], traced_output[i]),
                        f"traced {i}th output doesn't match model {i}th output for {model_class}",
                    )

                # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
                # (Even with this call, there are still memory leak by ~0.04MB)
                self.clear_torch_jit_class_registry()

    def test_headmasking(self):
        if not self.test_head_masking:
            self.skipTest(reason="Model does not support head masking")

        global_rng.seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        global_rng.seed()

        inputs_dict["output_attentions"] = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init._attn_implementation = "eager"  # head mask works only in eager mode and will be removed soon
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            # Prepare head_mask
            # Set require_grad after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior)
            head_mask = torch.ones(
                self.model_tester.num_hidden_layers,
                self.model_tester.num_attention_heads,
                device=torch_device,
            )
            head_mask[0, 0] = 0
            head_mask[-1, :-1] = 0
            head_mask.requires_grad_(requires_grad=True)
            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs["head_mask"] = head_mask
            if model.config.is_encoder_decoder:
                signature = inspect.signature(model.forward)
                arg_names = [*signature.parameters.keys()]
                if "decoder_head_mask" in arg_names:  # necessary differentiation because of T5 model
                    inputs["decoder_head_mask"] = head_mask
                if "cross_attn_head_mask" in arg_names:
                    inputs["cross_attn_head_mask"] = head_mask
            outputs = model(**inputs, return_dict=True)

            # Test that we can get a gradient back for importance score computation
            output = sum(t.sum() for t in outputs[0])
            output = output.sum()
            output.backward()
            multihead_outputs = head_mask.grad

            self.assertIsNotNone(multihead_outputs)
            self.assertEqual(len(multihead_outputs), self.model_tester.num_hidden_layers)

            def check_attentions_validity(attentions):
                # Remove Nan
                for t in attentions:
                    self.assertLess(
                        torch.sum(torch.isnan(t)), t.numel() / 4
                    )  # Check we don't have more than 25% nans (arbitrary)
                attentions = [
                    t.masked_fill(torch.isnan(t), 0.0) for t in attentions
                ]  # remove them (the test is less complete)

                self.assertAlmostEqual(attentions[0][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(attentions[0][..., -1, :, :].flatten().sum().item(), 0.0)
                if len(attentions) > 2:  # encoder-decoder models have only 2 layers in each module
                    self.assertNotEqual(attentions[1][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertAlmostEqual(attentions[-1][..., -2, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(attentions[-1][..., -1, :, :].flatten().sum().item(), 0.0)

            if model.config.is_encoder_decoder:
                check_attentions_validity(outputs.encoder_attentions)
                check_attentions_validity(outputs.decoder_attentions)
                check_attentions_validity(outputs.cross_attentions)
            else:
                check_attentions_validity(outputs.attentions)

    def test_head_pruning(self):
        if not self.test_pruning:
            self.skipTest(reason="Pruning is not activated")

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            model.set_attn_implementation("eager")
            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            model.prune_heads(heads_to_prune)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            # TODO: To have this check, we will need at least 3 layers. Do we really need it?
            # self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_pretrained(self):
        if not self.test_pruning:
            self.skipTest(reason="Pruning is not activated")

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            model.set_attn_implementation("eager")
            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            model.prune_heads(heads_to_prune)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name, attn_implementation="eager")
                model.to(torch_device)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]
            self.assertEqual(attentions[0].shape[-3], 1)
            # TODO: To have this check, we will need at least 3 layers. Do we really need it?
            # self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_config_init(self):
        if not self.test_pruning:
            self.skipTest(reason="Pruning is not activated")

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False

            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            model.set_attn_implementation("eager")

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            # TODO: To have this check, we will need at least 3 layers. Do we really need it?
            # self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_integration(self):
        if not self.test_pruning:
            self.skipTest(reason="Pruning is not activated")

        for model_class in self.all_model_classes:
            (
                config,
                inputs_dict,
            ) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False

            heads_to_prune = {1: [1, 2]}
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            model.set_attn_implementation("eager")

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 0)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name, attn_implementation="eager")
                model.to(torch_device)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 0)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)

            heads_to_prune = {0: [0], 1: [1, 2]}
            model.prune_heads(heads_to_prune)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)

            self.assertDictEqual(model.config.pruned_heads, {0: [0], 1: [1, 2]})

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            for k in config.sub_configs:
                getattr(config, k).output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for k in config.sub_configs:
            getattr(config, k).output_hidden_states = True

        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        for k in config.sub_configs:
            getattr(config, k).output_attentions = self.has_attentions

        # force eager attention to support output attentions
        if self.has_attentions:
            config._attn_implementation = "eager"

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        if config.is_encoder_decoder:
            # Seq2Seq models
            encoder_hidden_states = outputs.encoder_hidden_states[0]
            encoder_hidden_states.retain_grad()

            decoder_hidden_states = outputs.decoder_hidden_states[0]
            decoder_hidden_states.retain_grad()

            if self.has_attentions:
                encoder_attentions = outputs.encoder_attentions[0]
                encoder_attentions.retain_grad()

                decoder_attentions = outputs.decoder_attentions[0]
                decoder_attentions.retain_grad()

                cross_attentions = outputs.cross_attentions[0]
                cross_attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(encoder_hidden_states.grad)
            self.assertIsNotNone(decoder_hidden_states.grad)

            if self.has_attentions:
                self.assertIsNotNone(encoder_attentions.grad)
                self.assertIsNotNone(decoder_attentions.grad)
                self.assertIsNotNone(cross_attentions.grad)
        else:
            # Encoder-/Decoder-only models
            hidden_states = outputs.hidden_states[0]
            hidden_states.retain_grad()

            if self.has_attentions:
                attentions = outputs.attentions[0]
                attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(hidden_states.grad)

            if self.has_attentions:
                self.assertIsNotNone(attentions.grad)

    def test_feed_forward_chunking(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            torch.manual_seed(0)
            model = model_class(copy.deepcopy(original_config))
            model.to(torch_device)
            model.eval()

            hidden_states_no_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            torch.manual_seed(0)
            original_config.chunk_size_feed_forward = 1
            model = model_class(copy.deepcopy(original_config))
            model.to(torch_device)
            model.eval()

            hidden_states_with_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            torch.testing.assert_close(hidden_states_no_chunk, hidden_states_with_chunk, rtol=1e-3, atol=1e-3)

    def test_resize_position_vector_embeddings(self):
        if not self.test_resize_position_embeddings:
            self.skipTest(reason="Model does not have position embeddings")

        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            max_position_embeddings = config.max_position_embeddings

            # Retrieve the embeddings and clone theme
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                encoder_cloned_embeddings = encoder_model_embed.weight.clone()
                decoder_cloned_embeddings = decoder_model_embed.weight.clone()
            else:
                model_embed = model.get_position_embeddings()
                cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the position embeddings with a larger max_position_embeddings increases
            # the model's position embeddings size
            model.resize_position_embeddings(max_position_embeddings + 10)
            self.assertEqual(model.config.max_position_embeddings, max_position_embeddings + 10)

            # Check that it actually resizes the embeddings matrix
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                self.assertEqual(encoder_model_embed.weight.shape[0], encoder_cloned_embeddings.shape[0] + 10)
                self.assertEqual(decoder_model_embed.weight.shape[0], decoder_cloned_embeddings.shape[0] + 10)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the position embeddings with a smaller max_position_embeddings decreases
            # the model's max_position_embeddings
            model.resize_position_embeddings(max_position_embeddings - 5)
            self.assertEqual(model.config.max_position_embeddings, max_position_embeddings - 5)

            # Check that it actually resizes the embeddings matrix
            if model.config.is_encoder_decoder:
                encoder_model_embed, decoder_model_embed = model.get_position_embeddings()
                self.assertEqual(encoder_model_embed.weight.shape[0], encoder_cloned_embeddings.shape[0] - 5)
                self.assertEqual(decoder_model_embed.weight.shape[0], decoder_cloned_embeddings.shape[0] - 5)
            else:
                model_embed = model.get_position_embeddings()
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 5)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True

            if model.config.is_encoder_decoder:
                for p1, p2 in zip(encoder_cloned_embeddings, encoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
                for p1, p2 in zip(decoder_cloned_embeddings, decoder_model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False
            else:
                for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

    def test_resize_tokens_embeddings(self):
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is set to `False`")
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict.pop("labels", None)

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_embed_pre_resize = model.get_input_embeddings()
            type_model_embed_pre_resize = type(model_embed_pre_resize)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.get_text_config().vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check to make sure the type of embeddings returned post resizing is same as type of input
            type_model_embed_post_resize = type(model_embed)
            self.assertEqual(type_model_embed_pre_resize, type_model_embed_post_resize)
            # Check that added embeddings mean is close to the old embeddings mean
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(model_embed.weight, modifier_rank=None):
                    old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                    new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            else:
                old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            torch.testing.assert_close(old_embeddings_mean, new_embeddings_mean, rtol=1e-3, atol=1e-3)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                model_inputs = self._prepare_for_class(inputs_dict, model_class)
                model(**model_inputs)

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            # make sure that decoder_input_ids are resized as well
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                if "decoder_input_ids" in inputs_dict:
                    inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
                model_inputs = self._prepare_for_class(inputs_dict, model_class)
                model(**model_inputs)

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

            del model
            del config
            # Copy again. config changed with embedding resizing (`vocab_size` changed)
            config = copy.deepcopy(original_config)
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_vocab_size = config.get_text_config().vocab_size
            model.resize_token_embeddings(model_vocab_size + 10, pad_to_multiple_of=1)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertTrue(new_model_vocab_size + 10, model_vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=64)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            self.assertTrue(model_embed.weight.shape[0], new_model_vocab_size)
            self.assertTrue(new_model_vocab_size, model.vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size + 13, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            # Check that resizing a model to a multiple of pad_to_multiple leads to a model of exactly that size
            target_dimension = 128
            model_embed = model.resize_token_embeddings(target_dimension, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0], target_dimension)

            with self.assertRaisesRegex(
                ValueError,
                "Asking to pad the embedding matrix to a multiple of `1.3`, which is not and integer. Please make sure to pass an integer",
            ):
                model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=1.3)

            # Test when `vocab_size` is smaller than `hidden_size`.
            del model
            del config
            # Copy again. config changed with embedding resizing (`vocab_size` changed)
            config = copy.deepcopy(original_config)
            config.vocab_size = 4
            config.pad_token_id = 3
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config)
                model.to(torch_device)

            model_vocab_size = config.get_text_config().vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check to make sure the type of embeddings returned post resizing is same as type of input
            type_model_embed_post_resize = type(model_embed)
            self.assertEqual(type_model_embed_pre_resize, type_model_embed_post_resize)
            # Check that added embeddings mean is close to the old embeddings mean
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(model_embed.weight, modifier_rank=None):
                    old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                    new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            else:
                old_embeddings_mean = torch.mean(model_embed.weight.data[:-10, :], axis=0)
                new_embeddings_mean = torch.mean(model_embed.weight.data[-10:, :], axis=0)
            torch.testing.assert_close(old_embeddings_mean, new_embeddings_mean, rtol=1e-3, atol=1e-3)

    @require_deepspeed
    @require_torch_accelerator
    def test_resize_tokens_embeddings_with_deepspeed(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_tokens_embeddings()

    @require_deepspeed
    @require_torch_multi_accelerator
    def test_resize_tokens_embeddings_with_deepspeed_multi_gpu(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_tokens_embeddings()

    def test_resize_embeddings_untied(self):
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is set to `False`")

        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        original_config.tie_word_embeddings = False
        inputs_dict.pop("labels", None)

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            self.skipTest(reason="Model cannot untied embeddings")

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.Init():
                    model = model_class(config)
            else:
                model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.get_text_config().vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                model(**self._prepare_for_class(inputs_dict, model_class))

            # Test multivariate resizing.
            model.resize_token_embeddings(model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            # Check that added embeddings mean is close to the old embeddings mean
            if is_deepspeed_zero3_enabled():
                with deepspeed.zero.GatheredParameters(output_embeds.weight, modifier_rank=None):
                    old_embeddings_mean = torch.mean(output_embeds.weight.data[:-10, :], axis=0)
                    new_embeddings_mean = torch.mean(output_embeds.weight.data[-10:, :], axis=0)
            else:
                old_embeddings_mean = torch.mean(output_embeds.weight.data[:-10, :], axis=0)
                new_embeddings_mean = torch.mean(output_embeds.weight.data[-10:, :], axis=0)
            torch.testing.assert_close(old_embeddings_mean, new_embeddings_mean, rtol=1e-3, atol=1e-3)
            # check if the old bias mean close to added bias mean.
            if output_embeds.bias is not None:
                if is_deepspeed_zero3_enabled():
                    with deepspeed.zero.GatheredParameters(output_embeds.bias, modifier_rank=None):
                        old_bias_mean = torch.mean(output_embeds.bias.data[:-10], axis=0)
                        new_bias_mean = torch.mean(output_embeds.bias.data[-10:], axis=0)
                else:
                    old_bias_mean = torch.mean(output_embeds.bias.data[:-10], axis=0)
                    new_bias_mean = torch.mean(output_embeds.bias.data[-10:], axis=0)

                torch.testing.assert_close(old_bias_mean, new_bias_mean, rtol=1e-5, atol=1e-5)

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            new_model_vocab_size = model.config.get_text_config().vocab_size
            self.assertEqual(new_model_vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            if not is_deepspeed_zero3_enabled():
                # A distriputed launcher is needed for the forward pass when deepspeed is enabled
                model(**self._prepare_for_class(inputs_dict, model_class))

    @require_deepspeed
    @require_torch_accelerator
    def test_resize_embeddings_untied_with_deepspeed(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_embeddings_untied()

    @require_deepspeed
    @require_torch_multi_accelerator
    def test_resize_embeddings_untied_with_deepspeed_multi_gpu(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_embeddings_untied()

    def test_model_get_set_embeddings(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            self.assertIsInstance(model.get_input_embeddings(), nn.Embedding)

            new_input_embedding_layer = nn.Embedding(10, 10)
            model.set_input_embeddings(new_input_embedding_layer)
            self.assertEqual(model.get_input_embeddings(), new_input_embedding_layer)

            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "forward"))
            # The main input is the name of the argument after `self`
            observed_main_input_name = list(model_signature.parameters.keys())[1]
            self.assertEqual(model_class.main_input_name, observed_main_input_name)

    def test_correct_missing_keys(self):
        if not self.test_missing_keys:
            self.skipTest(reason="test_missing_keys is set to `False`")

        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            base_model_prefix = model.base_model_prefix

            if hasattr(model, base_model_prefix):
                extra_params = {k: v for k, v in model.named_parameters() if not k.startswith(base_model_prefix)}
                extra_params.update({k: v for k, v in model.named_buffers() if not k.startswith(base_model_prefix)})
                # Some models define this as None
                if model._keys_to_ignore_on_load_missing:
                    for key in model._keys_to_ignore_on_load_missing:
                        extra_params.pop(key, None)

                if not extra_params:
                    # In that case, we *are* on a head model, but every single key is not actual parameters
                    continue

                with tempfile.TemporaryDirectory() as temp_dir_name:
                    model.base_model.save_pretrained(temp_dir_name)
                    model, loading_info = model_class.from_pretrained(temp_dir_name, output_loading_info=True)
                    self.assertGreater(len(loading_info["missing_keys"]), 0, model.__class__.__name__)

    def test_tie_model_weights(self):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to `False`")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(copy.deepcopy(config))
            if model_not_tied.get_output_embeddings() is None:
                continue

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())
            # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            vocab_size = config.get_text_config().vocab_size
            model_tied.resize_token_embeddings(vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertEqual(len(params_tied_2), len(params_tied))

    @require_safetensors
    def test_can_use_safetensors(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model_tied = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                try:
                    model_tied.save_pretrained(d, safe_serialization=True)
                except Exception as e:
                    raise Exception(f"Class {model_class.__name__} cannot be saved using safetensors: {e}")

                model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                # Checking the state dicts are correct
                reloaded_state = model_reloaded.state_dict()
                for k, v in model_tied.state_dict().items():
                    self.assertIn(k, reloaded_state, f"Key {k} is missing from reloaded")
                    torch.testing.assert_close(
                        v, reloaded_state[k], msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}"
                    )
                # Checking there was no complain of missing weights
                self.assertEqual(infos["missing_keys"], [])

                # Checking the tensor sharing are correct
                ptrs = defaultdict(list)
                for k, v in model_tied.state_dict().items():
                    ptrs[v.data_ptr()].append(k)

                shared_ptrs = {k: v for k, v in ptrs.items() if len(v) > 1}

                for shared_names in shared_ptrs.values():
                    reloaded_ptrs = {reloaded_state[k].data_ptr() for k in shared_names}
                    self.assertEqual(
                        len(reloaded_ptrs),
                        1,
                        f"The shared pointers are incorrect, found different pointers for keys {shared_names}",
                    )

    def test_load_save_without_tied_weights(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            config.tie_word_embeddings = False
            model = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)

                model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                # Checking the state dicts are correct
                reloaded_state = model_reloaded.state_dict()
                for k, v in model.state_dict().items():
                    self.assertIn(k, reloaded_state, f"Key {k} is missing from reloaded")
                    torch.testing.assert_close(
                        v, reloaded_state[k], msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}"
                    )
                # Checking there was no complain of missing weights
                self.assertEqual(infos["missing_keys"], [])

    def test_tied_weights_keys(self):
        original_config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            copied_config = copy.deepcopy(original_config)
            copied_config.get_text_config().tie_word_embeddings = True
            model_tied = model_class(copied_config)

            tied_weight_keys = _get_tied_weight_keys(model_tied)
            # If we don't find any tied weights keys, and by default we don't tie the embeddings, it's because the model
            # does not tie them
            if len(tied_weight_keys) == 0 and not original_config.tie_word_embeddings:
                continue

            ptrs = collections.defaultdict(list)
            for name, tensor in model_tied.state_dict().items():
                ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]

            # Detect we get a hit for each key
            for key in tied_weight_keys:
                is_tied_key = any(re.search(key, p) for group in tied_params for p in group)
                self.assertTrue(is_tied_key, f"{key} is not a tied weight key for {model_class}.")

            # Removed tied weights found from tied params -> there should only be one left after
            for key in tied_weight_keys:
                for i in range(len(tied_params)):
                    tied_params[i] = [p for p in tied_params[i] if re.search(key, p) is None]

            tied_params = [group for group in tied_params if len(group) > 1]
            self.assertListEqual(
                tied_params,
                [],
                f"Missing `_tied_weights_keys` for {model_class}: add all of {tied_params} except one.",
            )

    def test_model_weights_reload_no_missing_tied_weights(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)

                # We are nuking ALL weights on file, so every parameter should
                # yell on load. We're going to detect if we yell too much, or too little.
                placeholder_dict = {"tensor": torch.tensor([1, 2])}
                safe_save_file(placeholder_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
                model_reloaded, infos = model_class.from_pretrained(tmp_dir, output_loading_info=True)

                params = dict(model_reloaded.named_parameters())
                params.update(dict(model_reloaded.named_buffers()))
                param_names = set(params.keys())

                missing_keys = set(infos["missing_keys"])

                extra_missing = missing_keys - param_names
                # Remove tied weights from extra missing: they are normally not warned as missing if their tied
                # counterpart is present but here there are no weights at all so we do get the warning.
                ptrs = collections.defaultdict(list)
                for name, tensor in model_reloaded.state_dict().items():
                    ptrs[id_tensor_storage(tensor)].append(name)
                tied_params = [names for _, names in ptrs.items() if len(names) > 1]
                for group in tied_params:
                    # We remove the group from extra_missing if not all weights from group are in it
                    if len(set(group) - extra_missing) > 0:
                        extra_missing = extra_missing - set(group)

                self.assertEqual(
                    extra_missing,
                    set(),
                    f"This model {model_class.__name__} might be missing some `keys_to_ignore`: {extra_missing}. "
                    f"For debugging, tied parameters are {tied_params}",
                )

                missed_missing = param_names - missing_keys
                # Remove nonpersistent buffers from missed_missing
                buffers = [n for n, _ in model_reloaded.named_buffers()]
                nonpersistent_buffers = {n for n in buffers if n not in model_reloaded.state_dict()}
                missed_missing = missed_missing - nonpersistent_buffers

                if model_reloaded._keys_to_ignore_on_load_missing is None:
                    expected_missing = set()
                else:
                    expected_missing = set()
                    for pattern in model_reloaded._keys_to_ignore_on_load_missing:
                        expected_missing.update({k for k in param_names if re.search(pattern, k) is not None})
                self.assertEqual(
                    missed_missing,
                    expected_missing,
                    f"This model {model_class.__name__} ignores keys {missed_missing} but they look like real"
                    " parameters. If they are non persistent buffers make sure to instantiate them with"
                    " `persistent=False`",
                )

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (list, tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    # model might return non-tensors objects (e.g. Cache class)
                    elif isinstance(tuple_object, torch.Tensor):
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )

    # Don't copy this method to model specific test file!
    # TODO: remove this method once the issues are all fixed!
    def _make_attention_mask_non_null(self, inputs_dict):
        """Make sure no sequence has all zeros as attention mask"""

        for k in ["attention_mask", "encoder_attention_mask", "decoder_attention_mask"]:
            if k in inputs_dict:
                attention_mask = inputs_dict[k]

                # Make sure no all 0s attention masks - to avoid failure at this moment.
                # Put `1` at the beginning of sequences to make it still work when combining causal attention masks.
                # TODO: remove this line once a fix regarding large negative values for attention mask is done.
                attention_mask = torch.cat(
                    [torch.ones_like(attention_mask[:, :1], dtype=attention_mask.dtype), attention_mask[:, 1:]], dim=-1
                )

                # Here we make the first sequence with all 0s as attention mask.
                # Currently, this will fail for `TFWav2Vec2Model`. This is caused by the different large negative
                # values, like `1e-4`, `1e-9`, `1e-30` and `-inf` for attention mask across models/frameworks.
                # TODO: enable this block once the large negative values thing is cleaned up.
                # (see https://github.com/huggingface/transformers/issues/14859)
                # attention_mask = torch.cat(
                #     [torch.zeros_like(attention_mask[:1], dtype=attention_mask.dtype), attention_mask[1:]],
                #     dim=0
                # )

                inputs_dict[k] = attention_mask

    # Don't copy this method to model specific test file!
    # TODO: remove this method once the issues are all fixed!
    def _postprocessing_to_ignore_test_cases(self, tf_outputs, pt_outputs, model_class):
        """For temporarily ignoring some failed test cases (issues to be fixed)"""

        tf_keys = {k for k, v in tf_outputs.items() if v is not None}
        pt_keys = {k for k, v in pt_outputs.items() if v is not None}

        key_differences = tf_keys.symmetric_difference(pt_keys)

        if model_class.__name__ in [
            "FlaubertWithLMHeadModel",
            "FunnelForPreTraining",
            "ElectraForPreTraining",
            "XLMWithLMHeadModel",
        ]:
            for k in key_differences:
                if k in ["loss", "losses"]:
                    tf_keys.discard(k)
                    pt_keys.discard(k)
        elif model_class.__name__.startswith("GPT2"):
            # `TFGPT2` has `past_key_values` as a tensor while `GPT2` has it as a tuple.
            tf_keys.discard("past_key_values")
            pt_keys.discard("past_key_values")

        # create new outputs from the remaining fields
        new_tf_outputs = type(tf_outputs)(**{k: tf_outputs[k] for k in tf_keys})
        new_pt_outputs = type(pt_outputs)(**{k: pt_outputs[k] for k in pt_keys})

        return new_tf_outputs, new_pt_outputs

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            model_forward_args = inspect.signature(model.forward).parameters
            if "inputs_embeds" not in model_forward_args:
                self.skipTest(reason="This model doesn't use `inputs_embeds`")

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ not in get_values(MODEL_MAPPING_NAMES):
                continue
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            model_forward_args = inspect.signature(model.forward).parameters
            if "inputs_embeds" not in model_forward_args:
                self.skipTest(reason="This model doesn't use `inputs_embeds`")

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 1
            )

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                # some models infer position ids/attn mask differently when input ids
                # by check if pad_token let's make sure no padding is in input ids
                not_pad_token_id = pad_token_id + 1 if max(0, pad_token_id - 1) == 0 else pad_token_id - 1
                input_ids[input_ids == pad_token_id] = not_pad_token_id
                del inputs["input_ids"]
                inputs_embeds = wte(input_ids)
                with torch.no_grad():
                    out_ids = model(input_ids=input_ids, **inputs)[0]
                    out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                encoder_input_ids[encoder_input_ids == pad_token_id] = max(0, pad_token_id + 1)
                decoder_input_ids[decoder_input_ids == pad_token_id] = max(0, pad_token_id + 1)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)
                inputs_embeds = wte(encoder_input_ids)
                decoder_inputs_embeds = wte(decoder_input_ids)
                with torch.no_grad():
                    out_ids = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids, **inputs)[0]
                    out_embeds = model(
                        inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **inputs
                    )[0]
            torch.testing.assert_close(out_embeds, out_ids)

    @require_torch_gpu
    @require_torch_multi_gpu
    def test_multi_gpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask", "decoder_head_mask", "cross_attn_head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to accelerator O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = nn.DataParallel(model)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

    @require_torch_gpu
    @require_torch_multi_gpu
    def test_model_parallelization(self):
        if not self.test_model_parallel:
            self.skipTest(reason="test_model_parallel is set to False")

        # a candidate for testing_utils
        def get_current_gpu_memory_use():
            """returns a list of VRAM allocations per GPU in MBs"""

            per_device_memory = []
            for id in range(backend_device_count(torch_device)):
                with backend_torch_accelerator_module(torch_device).device(id):
                    per_device_memory.append(backend_memory_allocated(torch_device) >> 20)

            return per_device_memory

        # Needs a large model to see the difference.
        config = self.model_tester.get_large_model_config()

        for model_class in self.all_parallelizable_model_classes:
            backend_empty_cache(torch_device)

            # 1. single gpu memory load + unload + memory measurements
            # Retrieve initial memory usage (can easily be ~0.6-1.5GB if cuda-kernels have been preloaded by previous tests)
            memory_at_start = get_current_gpu_memory_use()

            # Put model on device 0 and take a memory snapshot
            model = model_class(config)
            model.to(f"{torch_device}:0")
            memory_after_model_load = get_current_gpu_memory_use()

            # The memory use on device 0 should be higher than it was initially.
            self.assertGreater(memory_after_model_load[0], memory_at_start[0])

            del model
            gc.collect()
            backend_empty_cache(torch_device)

            # 2. MP test
            # it's essential to re-calibrate the usage before the next stage
            memory_at_start = get_current_gpu_memory_use()

            # Spread model layers over multiple devices
            model = model_class(config)
            model.parallelize()
            memory_after_parallelization = get_current_gpu_memory_use()

            # Assert that the memory use on all devices is higher than it was when loaded only on CPU
            for n in range(len(model.device_map.keys())):
                self.assertGreater(memory_after_parallelization[n], memory_at_start[n])

            # Assert that the memory use of device 0 is lower than it was when the entire model was loaded on it
            self.assertLess(memory_after_parallelization[0], memory_after_model_load[0])

            # Assert that the memory use of device 1 is higher than it was when the entire model was loaded
            # on device 0 and device 1 wasn't used at all
            self.assertGreater(memory_after_parallelization[1], memory_after_model_load[1])

            del model
            gc.collect()
            backend_empty_cache(torch_device)

    @require_torch_gpu
    @require_torch_multi_gpu
    def test_model_parallel_equal_results(self):
        if not self.test_model_parallel:
            self.skipTest(reason="test_model_parallel is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_parallelizable_model_classes:
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)

            def cast_to_device(dictionary, device):
                output = {}
                for k, v in dictionary.items():
                    if isinstance(v, torch.Tensor):
                        output[k] = v.to(device)
                    else:
                        output[k] = v

                return output

            model = model_class(config)
            output = model(**cast_to_device(inputs_dict, "cpu"))

            model.parallelize()

            parallel_output = model(**cast_to_device(inputs_dict, f"{torch_device}:0"))

            for value, parallel_value in zip(output, parallel_output):
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, parallel_value.to("cpu"), rtol=1e-7, atol=1e-7)
                elif isinstance(value, (tuple, list)):
                    for value_, parallel_value_ in zip(value, parallel_value):
                        torch.testing.assert_close(value_, parallel_value_.to("cpu"), rtol=1e-7, atol=1e-7)

    def check_device_map_is_respected(self, model, device_map):
        for param_name, param in model.named_parameters():
            # Find device in device_map
            while len(param_name) > 0 and param_name not in device_map:
                param_name = ".".join(param_name.split(".")[:-1])
            if param_name not in device_map:
                raise ValueError("device map is incomplete, it does not contain any device for `param_name`.")

            param_device = device_map[param_name]
            if param_device in ["cpu", "disk"]:
                self.assertEqual(param.device, torch.device("meta"))
            elif param_device in ["mps"]:
                self.assertEqual(param.device, torch.device("mps"))
            else:
                # when loaded with device_map, `param_device` are integer values for cuda/xpu/hpu/npu/mlu
                self.assertEqual(param.device, torch.device(f"{torch_device}:{param_device}"))

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_disk_offload_bin(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(copy.deepcopy(config)).eval()
            model = model.to(torch_device)
            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir, safe_serialization=False)

                with self.assertRaises(ValueError):
                    max_size = int(self.model_split_percents[0] * model_size)
                    max_memory = {0: max_size, "cpu": max_size}
                    # This errors out cause it's missing an offload folder
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)

                max_size = int(self.model_split_percents[1] * model_size)
                max_memory = {0: max_size, "cpu": max_size}
                new_model = model_class.from_pretrained(
                    tmp_dir, device_map="auto", max_memory=max_memory, offload_folder=tmp_dir
                )

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict_class)

                if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                    [
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
                        for a, b in zip(base_output[0], new_output[0])
                    ]
                else:
                    torch.testing.assert_close(base_output[0], new_output[0], rtol=1e-5, atol=1e-5)

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_disk_offload_safetensors(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(copy.deepcopy(config)).eval()
            model = model.to(torch_device)
            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                max_size = int(self.model_split_percents[1] * model_size)
                max_memory = {0: max_size, "cpu": max_size}

                # This doesn't error out as it's in safetensors and doesn't need an offload folder
                new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict_class)

                if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                    [
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
                        for a, b in zip(base_output[0], new_output[0])
                    ]
                else:
                    torch.testing.assert_close(base_output[0], new_output[0], rtol=1e-5, atol=1e-5)

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_accelerator
    def test_cpu_offload(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(copy.deepcopy(config)).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            # We test several splits of sizes to make sure it works.
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                for max_size in max_gpu_sizes:
                    max_memory = {0: max_size, "cpu": model_size * 2}
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                    # Making sure part of the model will actually end up offloaded
                    self.assertSetEqual(set(new_model.hf_device_map.values()), {0, "cpu"})

                    self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                    torch.manual_seed(0)
                    new_output = new_model(**inputs_dict_class)

                    if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                        [
                            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
                            for a, b in zip(base_output[0], new_output[0])
                        ]
                    else:
                        torch.testing.assert_close(base_output[0], new_output[0], rtol=1e-5, atol=1e-5)

    @require_non_hpu
    @require_accelerate
    @mark.accelerate_tests
    @require_torch_multi_accelerator
    def test_model_parallelism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class._no_split_modules is None:
                continue

            inputs_dict_class = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).eval()
            model = model.to(torch_device)

            torch.manual_seed(0)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[""]
            # We test several splits of sizes to make sure it works.
            max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                for max_size in max_gpu_sizes:
                    max_memory = {0: max_size, 1: model_size * 2, "cpu": model_size * 2}
                    new_model = model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                    # Making sure part of the model will actually end up offloaded
                    self.assertSetEqual(set(new_model.hf_device_map.values()), {0, 1})
                    self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                    torch.manual_seed(0)
                    new_output = new_model(**inputs_dict_class)

                    if isinstance(base_output[0], tuple) and isinstance(new_output[0], tuple):
                        [
                            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
                            for a, b in zip(base_output[0], new_output[0])
                        ]
                    else:
                        torch.testing.assert_close(base_output[0], new_output[0], rtol=1e-5, atol=1e-5)

    def test_problem_types(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        problem_types = [
            {"title": "multi_label_classification", "num_labels": 2, "dtype": torch.float},
            {"title": "single_label_classification", "num_labels": 1, "dtype": torch.long},
            {"title": "regression", "num_labels": 1, "dtype": torch.float},
        ]

        for model_class in self.all_model_classes:
            if model_class.__name__ not in [
                *get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES),
                *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
            ]:
                continue

            for problem_type in problem_types:
                with self.subTest(msg=f"Testing {model_class} with {problem_type['title']}"):
                    config.problem_type = problem_type["title"]
                    config.num_labels = problem_type["num_labels"]

                    model = model_class(config)
                    model.to(torch_device)
                    model.train()

                    inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                    if problem_type["num_labels"] > 1:
                        inputs["labels"] = inputs["labels"].unsqueeze(1).repeat(1, problem_type["num_labels"])

                    inputs["labels"] = inputs["labels"].to(problem_type["dtype"])

                    # This tests that we do not trigger the warning form PyTorch "Using a target size that is different
                    # to the input size. This will likely lead to incorrect results due to broadcasting. Please ensure
                    # they have the same size." which is a symptom something in wrong for the regression problem.
                    # See https://github.com/huggingface/transformers/issues/11780
                    with warnings.catch_warnings(record=True) as warning_list:
                        loss = model(**inputs).loss
                    for w in warning_list:
                        if "Using a target size that is different to the input size" in str(w.message):
                            raise ValueError(
                                f"Something is going wrong in the regression problem: intercepted {w.message}"
                            )

                    loss.backward()

    def test_load_with_mismatched_shapes(self):
        if not self.test_mismatched_shapes:
            self.skipTest(reason="test_missmatched_shapes is set to False")
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ not in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
                continue

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(RuntimeError):
                        new_model = AutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42)
                    with self.assertRaises(RuntimeError):
                        new_model_without_prefix = AutoModel.from_pretrained(tmp_dir, vocab_size=10)

                    logger = logging.get_logger("transformers.modeling_utils")

                    with CaptureLogger(logger) as cl:
                        new_model = AutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)
                    new_model.to(torch_device)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = AutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)
                    input_ids = ids_tensor((2, 8), 10)
                    new_model_without_prefix.to(torch_device)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        if not self.test_mismatched_shapes:
            self.skipTest(reason="test_missmatched_shapes is set to False")
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)

        for model_class in self.all_model_classes:
            mappings = [
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
                MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
                MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
                MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
            ]
            is_classication_model = any(model_class.__name__ in get_values(mapping) for mapping in mappings)

            if not is_classication_model:
                continue

            # TODO: ydshieh
            is_special_classes = model_class.__name__ in [
                "wav2vec2.masked_spec_embed",
                "Wav2Vec2ForSequenceClassification",
                "CLIPForImageClassification",
                "MetaClip2ForImageClassification",
                "Siglip2ForImageClassification",
                "RegNetForImageClassification",
                "ResNetForImageClassification",
                "UniSpeechSatForSequenceClassification",
                "Wav2Vec2BertForSequenceClassification",
                "PvtV2ForImageClassification",
                "Wav2Vec2ConformerForSequenceClassification",
                "WavLMForSequenceClassification",
                "SwiftFormerForImageClassification",
                "SEWForSequenceClassification",
                "BitForImageClassification",
                "SEWDForSequenceClassification",
                "SiglipForImageClassification",
                "HubertForSequenceClassification",
                "Swinv2ForImageClassification",
                "Data2VecAudioForSequenceClassification",
                "UniSpeechForSequenceClassification",
                "PvtForImageClassification",
                "ModernBertForSequenceClassification",
                "ModernBertForTokenClassification",
                "TimmWrapperForImageClassification",
                "ModernBertForQuestionAnswering",
                "ModernBertDecoderForSequenceClassification",
                "ModernBertDecoderForCausalLM",
            ]
            special_param_names = [
                r"^bit\.",
                r"^classifier\.weight",
                r"^classifier\.bias",
                r"^classifier\..+\.weight",
                r"^classifier\..+\.bias",
                r"^data2vec_audio\.",
                r"^dist_head\.",
                r"^head\.",
                r"^hubert\.",
                r"^pvt\.",
                r"^pvt_v2\.",
                r"^regnet\.",
                r"^resnet\.",
                r"^sew\.",
                r"^sew_d\.",
                r"^swiftformer\.",
                r"^swinv2\.",
                r"^transformers\.models\.swiftformer\.",
                r"^timm_model\.",
                r"^unispeech\.",
                r"^unispeech_sat\.",
                r"^vision_model\.",
                r"^wav2vec2\.",
                r"^wav2vec2_bert\.",
                r"^wav2vec2_conformer\.",
                r"^wavlm\.",
            ]

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(configs_no_init)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(RuntimeError):
                        new_model = model_class.from_pretrained(tmp_dir, num_labels=42)

                    logger = logging.get_logger("transformers.modeling_utils")

                    with CaptureLogger(logger) as cl:
                        new_model = model_class.from_pretrained(tmp_dir, num_labels=42, ignore_mismatched_sizes=True)
                    self.assertIn("the shapes did not match", cl.out)

                    for name, param in new_model.named_parameters():
                        if param.requires_grad:
                            param_mean = ((param.data.mean() * 1e9).round() / 1e9).item()
                            if not (
                                is_special_classes
                                and any(len(re.findall(target, name)) > 0 for target in special_param_names)
                            ):
                                self.assertIn(
                                    param_mean,
                                    [0.0, 1.0],
                                    msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                                )
                            else:
                                # Here we allow the parameters' mean to be in the range [-5.0, 5.0] instead of being
                                # either `0.0` or `1.0`, because their initializations are not using
                                # `config.initializer_factor` (or something similar). The purpose of this test is simply
                                # to make sure they are properly initialized (to avoid very large value or even `nan`).
                                self.assertGreaterEqual(
                                    param_mean,
                                    -5.0,
                                    msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                                )
                                self.assertLessEqual(
                                    param_mean,
                                    5.0,
                                    msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                                )

    def test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist(self):
        # 1. Create a dummy class. Should have buffers as well? To make sure we test __init__
        class MyClass(PreTrainedModel):
            config_class = PretrainedConfig

            def __init__(self, config=None):
                super().__init__(config if config is not None else PretrainedConfig())
                self.linear = nn.Linear(10, config.num_labels, bias=True)
                self.embedding = nn.Embedding(10, 10)
                self.std = 1

            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    module.weight.data = nn.init.kaiming_uniform_(module.weight.data, np.sqrt(5))
                    if module.bias is not None:
                        module.bias.data = module.bias.data.normal_(mean=0.0, std=self.std)

        # Used to make sure the weights with matched shape are loaded correctly
        config = PretrainedConfig()
        config.num_labels = 3
        model = MyClass(config=config)

        # Used to make sure the weights with mismatched shape are properly initialized
        set_seed(0)
        config = PretrainedConfig()
        config.num_labels = 4
        # not to init. the weights during the creation: to match the logic in `from_pretrained`, so we can keep the
        # same sequence of random ops in the execution path to allow us to compare `target_model` and `new_model` below
        # for `linear` part.
        with ContextManagers([no_init_weights()]):
            target_model = MyClass(config=config)
        target_model.apply(target_model._initialize_weights)

        with tempfile.TemporaryDirectory() as tmpdirname:
            state_dict = model.state_dict()
            del state_dict["linear.weight"]

            model.config.save_pretrained(tmpdirname)
            torch.save(state_dict, os.path.join(tmpdirname, "pytorch_model.bin"))

            set_seed(0)
            new_model = MyClass.from_pretrained(tmpdirname, num_labels=4, ignore_mismatched_sizes=True)

            for key in new_model.state_dict():
                # check weight values for weights with matched shapes are identical
                # (i.e. correctly loaded from the checkpoint)
                if key not in ["linear.weight", "linear.bias"]:
                    max_diff = torch.max(torch.abs(model.state_dict()[key] - new_model.state_dict()[key]))
                    self.assertLessEqual(
                        max_diff.item(),
                        1e-6,
                        msg=f"the weight values for `{key}` in `new_model` and `model` are  not identical",
                    )
                else:
                    # check we have some mismatched shapes
                    self.assertNotEqual(
                        model.state_dict()[key].shape,
                        new_model.state_dict()[key].shape,
                        msg=f"the weight shapes for {key} in `model` and `new_model` should differ",
                    )
                    # check the weights with mismatched shape are properly initialized
                    max_diff = torch.max(torch.abs(new_model.state_dict()[key] - target_model.state_dict()[key]))
                    self.assertLessEqual(
                        max_diff.item(),
                        1e-6,
                        msg=f"the weight values for `{key}` in `new_model` and `target_model` are not identical",
                    )

    def test_model_is_small(self):
        # Just a consistency check to make sure we are not running tests on 80M parameter models.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            num_params = model.num_parameters()
            assert num_params < 1000000, (
                f"{model_class} is too big for the common tests ({num_params})! It should have 1M max."
            )

    def flash_attn_inference_equivalence(self, attn_implementation: str, padding_side: str):
        r"""
        Tests the equivalence between the eager and flash attention implementations.
        This test is only for inference and runs with `dtype=torch.bfloat16`.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        # This flag is used to know if the test was skipped for all `self.all_model_classes` or not
        _has_run_at_least_one_model = False

        for model_class in self.all_model_classes:
            # Custom kernel which needs the mask interface to be properly usable on these models
            if not model_class._supports_attention_backend and not attn_implementation.startswith("flash_attention"):
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # flash attention variants does not always support arbitrary headim
            config = self._prepare_config_headdim(config, 16)

            # forcing the prefill size to go over sliding window size to check for SWA correctness
            if getattr(config, "sliding_window", None):
                config.sliding_window = 2

            model = model_class(config)
            if not all(
                submodel._supports_flash_attn for submodel in model.modules() if isinstance(submodel, PreTrainedModel)
            ):
                continue

            _has_run_at_least_one_model = True
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Save the model so we can reload with correct attention
                model.save_pretrained(tmpdirname)

                # Create first inputs without attention mask
                dummy_input = inputs_dict[model.main_input_name][:1]
                if torch.is_floating_point(dummy_input):
                    dummy_input = dummy_input.to(torch.bfloat16)
                first_inputs = {model.main_input_name: dummy_input, "output_hidden_states": True}
                if "pixel_values" in inputs_dict:
                    first_inputs["pixel_values"] = inputs_dict["pixel_values"][:1].to(torch.bfloat16)
                if model.config.is_encoder_decoder:
                    first_inputs["decoder_input_ids"] = inputs_dict.get("decoder_input_ids", dummy_input)[:1]

                # Create attention mask with padding
                dummy_attention_mask = inputs_dict.get("attention_mask", None)
                if dummy_attention_mask is not None:
                    dummy_attention_mask = dummy_attention_mask[:1]
                    if padding_side == "left":
                        dummy_attention_mask[:, 1:] = 1
                        dummy_attention_mask[:, 0] = 0
                    else:
                        dummy_attention_mask[:, :-1] = 1
                        dummy_attention_mask[:, -1] = 0

                # Create second inputs with attention mask and padding
                second_inputs = copy.deepcopy(first_inputs)
                if dummy_attention_mask is not None:
                    second_inputs["attention_mask"] = dummy_attention_mask
                    if model.config.is_encoder_decoder:
                        second_inputs["decoder_attention_mask"] = dummy_attention_mask

                model = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="eager", device_map=torch_device
                )

                # First run without attention mask
                outputs = model(**first_inputs)
                logits_1_eager = (
                    outputs.hidden_states[-1]
                    if "hidden_states" in outputs
                    else outputs.logits_per_image
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                # Second run with attention mask and padding
                outputs = model(**second_inputs)
                logits_2_eager = (
                    outputs.hidden_states[-1]
                    if "hidden_states" in outputs
                    else outputs.logits_per_image
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )

                # Switch to FA
                del model
                model = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation=attn_implementation, device_map=torch_device
                )
                outputs = model(**first_inputs)
                logits_1_fa = (
                    outputs.hidden_states[-1]
                    if "hidden_states" in outputs
                    else outputs.logits_per_image
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )
                # Second run with attention mask and padding
                outputs = model(**second_inputs)
                logits_2_fa = (
                    outputs.hidden_states[-1]
                    if "hidden_states" in outputs
                    else outputs.logits_per_image
                    if not model.config.is_encoder_decoder
                    else outputs.decoder_hidden_states[-1]
                )

                # Check the results
                torch.testing.assert_close(logits_1_eager, logits_1_fa, atol=4e-2, rtol=4e-2)
                if padding_side == "left":
                    torch.testing.assert_close(logits_2_eager[1:], logits_2_fa[1:], atol=4e-2, rtol=4e-2)
                    # Check it can run in training mode
                    model.train()
                    _ = model(**second_inputs)
                else:
                    torch.testing.assert_close(logits_2_eager[:-1], logits_2_fa[:-1], atol=4e-2, rtol=4e-2)

        # In this case, the test should appear as skipped, not successful
        if not _has_run_at_least_one_model:
            self.skipTest(
                f"Model architecture does not support {attn_implementation}, or setting its attention dynamically"
            )

    @require_kernels
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_kernels_inference_equivalence(self):
        self.flash_attn_inference_equivalence(attn_implementation="kernels-community/flash-attn3", padding_side="left")

    @require_torch_mps
    @require_kernels
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_kernels_mps_inference_equivalence(self):
        self.flash_attn_inference_equivalence(
            attn_implementation="kernels-community/metal-flash-sdpa", padding_side="left"
        )

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence(self):
        self.flash_attn_inference_equivalence(attn_implementation="flash_attention_2", padding_side="left")

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.flash_attn_inference_equivalence(attn_implementation="flash_attention_2", padding_side="right")

    @require_flash_attn_3
    @require_torch_gpu
    @mark.flash_attn_3_test
    @slow
    @is_flaky()
    def test_flash_attn_3_inference_equivalence(self):
        self.flash_attn_inference_equivalence(attn_implementation="flash_attention_3", padding_side="left")

    @require_flash_attn_3
    @require_torch_gpu
    @mark.flash_attn_3_test
    @slow
    @is_flaky()
    def test_flash_attn_3_inference_equivalence_right_padding(self):
        self.flash_attn_inference_equivalence(attn_implementation="flash_attention_3", padding_side="right")

    def test_attn_implementation_composite_models(self):
        """
        Tests if composite models can receive a dict object as attn_implementation, where each key should be
        one of the sub-configs from the model's config.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_model_classes:
            if not self._is_composite:
                self.skipTest("Model is not a composite model.")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # set eager as it will be the one supported in all models
            # we just need to test if passing 'attn_implementation' as a dict fails or not
            attn_implementation_per_subconfig = {"": "eager"}
            for key in config.sub_configs:
                attn_implementation_per_subconfig[key] = "eager"

            config._attn_implementation = attn_implementation_per_subconfig
            model = model_class(config)
            for key in config.sub_configs:
                sub_config = getattr(model.config, key)
                self.assertTrue(sub_config._attn_implementation == "eager")

            for name, submodule in model.named_modules():
                class_name = submodule.__class__.__name__
                if (
                    class_name.endswith("Attention")
                    and getattr(submodule, "config", None)
                    and submodule.config._attn_implementation != "eager"
                ):
                    raise ValueError(
                        f"The eager model should not have SDPA/FA2 attention layers but got `{class_name}.config._attn_implementation={submodule.config._attn_implementation}`"
                    )

            # Set the attention to default `None` but the text config to `eager`
            # The model should load encoders in SDPA but not the text attention
            config._attn_implementation = None
            config.get_text_config(decoder=True)._attn_implementation = "eager"
            model = model_class(config)
            self.assertTrue(model.config.get_text_config(decoder=True)._attn_implementation == "eager")

            # Test that using `dict` atttention implementation works with `from_pretrained`
            #  Set all backbones to "eager" because "eager" attention is always available
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                new_model = model.from_pretrained(tmpdirname, attn_implementation=attn_implementation_per_subconfig)
                self.assertTrue(new_model.config._attn_implementation == "eager")
                for submodule in new_model.modules():
                    if (
                        submodule is not new_model
                        and isinstance(submodule, PreTrainedModel)
                        and submodule.config.__class__ != new_model.config.__class__
                    ):
                        self.assertTrue(submodule.config._attn_implementation == "eager")

    def test_sdpa_can_dispatch_non_composite_models(self):
        """
        Tests if non-composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are called "SDPAAttention".
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self.all_model_classes[0]._supports_sdpa or self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if (
                        class_name.endswith("Attention")
                        and getattr(submodule, "config", None)
                        and submodule.config._attn_implementation == "sdpa"
                    ):
                        raise ValueError(
                            f"The eager model should not have SDPA attention layers but got `{class_name}.config._attn_implementation={submodule.config._attn_implementation}`"
                        )

    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are called "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implamentation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                vision_model_names = {"visual", "image_tower", "vision_tower", "vision_model"}
                language_model_names = {"language_model", "model", "text_model"}
                vision_model_name = [name for name in vision_model_names if hasattr(model_sdpa, name)][0]
                language_model_name = [name for name in language_model_names if hasattr(model_sdpa, name)][0]

                vision_model_sdpa = getattr(model_sdpa, vision_model_name)
                language_model_sdpa = getattr(model_sdpa, language_model_name)
                text_attn = "sdpa" if language_model_sdpa._supports_sdpa else "eager"
                vision_attn = "sdpa" if vision_model_sdpa._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(language_model_sdpa.config._attn_implementation == text_attn)
                self.assertTrue(vision_model_sdpa.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(getattr(model_eager, language_model_name).config._attn_implementation == "eager")
                self.assertTrue(getattr(model_eager, vision_model_name).config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if (
                        class_name.endswith("Attention")
                        and getattr(submodule, "config", None)
                        and submodule.config._attn_implementation == "sdpa"
                    ):
                        raise ValueError("The eager model should not have SDPA attention layers")

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        _test_eager_matches_sdpa_inference(
            self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
        )

    @require_torch_accelerator
    @slow
    def test_sdpa_can_dispatch_on_flash(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        device_type, major, minor = get_device_properties()
        if device_type == "cuda" and major < 8:
            self.skipTest(reason="This test requires an NVIDIA GPU with compute capability >= 8.0")
        elif device_type == "rocm" and major < 9:
            self.skipTest(reason="This test requires an AMD GPU with compute capability >= 9.0")
        elif device_type not in ["cuda", "rocm", "xpu"]:
            self.skipTest(reason="This test requires a Nvidia or AMD GPU, or an Intel XPU")

        torch.compiler.reset()

        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(f"{model_class.__name__} does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            if config.model_type in ["paligemma"]:
                self.skipTest(
                    "PaliGemma-like models currently (transformers==4.41.0) requires an attention_mask input"
                )
            if config.model_type in [
                "modernbert",
                "gemma3",
                "t5gemma",
                "diffllama",
                "dpr",
                "eomt",
                "gpt_bigcode",
                "jamba",
                "kosmos-2",
                "mllama",
                "pixtral",
                "sam",
                "sam_hq",
                "zamba2",
                "sam_vision_model",
                "sam2_vision_model",
                "sam_hq_vision_model",
            ]:
                self.skipTest(
                    reason=f"{config.model_type} currently (transformers==4.52.0) automatically adds an attention_mask input"
                )
            if config.model_type in ["idefics", "idefics2", "idefics3"]:
                self.skipTest(reason="Idefics currently (transformers==4.39.1) requires an image_attention_mask input")
            if config.model_type in ["sam"]:
                self.skipTest(reason="SAM requires an attention_mask input for relative positional embeddings")

            model = model_class(config)

            sub_models_supporting_sdpa = [
                module._supports_sdpa
                for name, module in model.named_modules()
                if isinstance(module, PreTrainedModel) and name != ""
            ]
            supports_sdpa_all_modules = (
                all(sub_models_supporting_sdpa) if len(sub_models_supporting_sdpa) > 0 else model._supports_sdpa
            )
            if not supports_sdpa_all_modules:
                self.skipTest(reason="This models' submodels does not support sdpa")

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, dtype=torch.float16, attn_implementation="sdpa")
                model.to(torch_device)

                inputs_dict.pop("attention_mask", None)
                inputs_dict.pop("decoder_attention_mask", None)

                for name, inp in inputs_dict.items():
                    if isinstance(inp, torch.Tensor) and inp.dtype in [torch.float32, torch.float16]:
                        inputs_dict[name] = inp.to(torch.float16)

                with sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                    _ = model(**inputs_dict)

    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    @slow
    def test_sdpa_can_compile_dynamic(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        device_type, major, minor = get_device_properties()
        if device_type == "cuda" and major < 8:
            self.skipTest(reason="This test requires an NVIDIA GPU with compute capability >= 8.0")
        elif device_type == "rocm" and major < 9:
            self.skipTest(reason="This test requires an AMD GPU with compute capability >= 9.0")
        elif device_type not in ["cuda", "rocm", "xpu"]:
            self.skipTest(reason="This test requires a Nvidia or AMD GPU, or an Intel XPU")

        torch.compiler.reset()

        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(f"{model_class.__name__} does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            if config.model_type in ["dbrx"]:
                self.skipTest(
                    "DBRX (transformers==4.40) requires a modification to support dynamic shapes with compile."
                )
            if getattr(config, "cache_implementation", None) == "hybrid":
                self.skipTest(
                    "Cannot compile forward without an existing cache with Hybrid, as `torch._dynamo.mark_static_address` "
                    "is a forbidden call."
                )

            model = model_class(config)

            sub_models_supporting_sdpa = [
                module._supports_sdpa
                for name, module in model.named_modules()
                if isinstance(module, PreTrainedModel) and name != ""
            ]
            supports_sdpa_all_modules = (
                all(sub_models_supporting_sdpa) if len(sub_models_supporting_sdpa) > 0 else model._supports_sdpa
            )
            if not supports_sdpa_all_modules:
                self.skipTest(reason="This models' submodels does not support sdpa")

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, dtype=torch.float16, attn_implementation="sdpa")
                model.to(torch_device)

                # For PyTorch 2.1 - 2.3.0 set `dynamic=True`. In the future setting `dynamic=None` and using `torch._dynamo.mark_dynamic()`
                # on input tensors will be required. `mark_dynamic` currently raises inconsistent shape errors.
                model = torch.compile(model, dynamic=True)

                inputs_dict.pop("attention_mask", None)
                inputs_dict.pop("decoder_attention_mask", None)
                for name, inp in inputs_dict.items():
                    if isinstance(inp, torch.Tensor) and inp.dtype in [torch.float32, torch.float16]:
                        inputs_dict[name] = inp.to(torch.float16)

                # use no_grad to save some memory
                with torch.no_grad():
                    _ = model(**inputs_dict)

    def test_sdpa_matches_eager_sliding_window(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        WINDOW_ATTENTION_MODELS = ["mistral", "mixtral", "minimax", "qwen2", "qwen_moe", "starcoder2"]

        if len(self.all_generative_model_classes) == 0:
            self.skipTest(f"No generative model classes for {self.__class__.__name__}")

        for model_class in self.all_generative_model_classes:
            if model_class._supports_sdpa:
                self.skipTest(reason="Model architecture does not support attentions")
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            if config.model_type not in WINDOW_ATTENTION_MODELS:
                self.skipTest(f"{config.model_type} does not use window attention")

            config.sliding_window = 2

            dummy_input = inputs_dict[model_class.main_input_name]
            attention_mask = inputs_dict["attention_mask"]

            self.assertTrue(dummy_input.ndim == 2)
            self.assertTrue(dummy_input.shape[1] > 6)

            with tempfile.TemporaryDirectory() as tmpdir:
                with torch.device(torch_device):
                    model_eager = AutoModelForCausalLM.from_config(
                        config, attn_implementation="eager", dtype=torch.float32
                    )

                model_eager.save_pretrained(tmpdir)

                with torch.device(torch_device):
                    model_sdpa = AutoModelForCausalLM.from_pretrained(
                        tmpdir, attn_implementation="sdpa", dtype=torch.float32
                    )

                model_eager = model_eager.eval()
                model_sdpa = model_sdpa.eval()

                with torch.no_grad():
                    with sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        res_eager = model_eager(**inputs_dict, return_dict=False)[0]
                        res_sdpa = model_sdpa(**inputs_dict, return_dict=False)[0]

                # Only non-padding tokens are expected to match.
                self.assertTrue(
                    torch.allclose(res_eager[attention_mask == 1], res_sdpa[attention_mask == 1], rtol=1e-4, atol=1e-4)
                )

    def flash_attn_can_dispatch_composite_models(self, attn_implementation: str):
        """
        Tests if composite models can dispatch on flash attention if the sub-models support it.
        The tests is needed as we handle differently composite models and we cannot check them
        with above tests. If any of the sub-models does not support flash attention, we'll raise an error when dispatching
        that particular sub-model. Otherwise we dispatch safely in all sub-models, where "sub-models" are specific
        backbone models (LM/vision/audio/etc)
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(f"bfloat16 not supported on {torch_device} (on the specific device currently used)")

        dtype = torch.bfloat16
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            if not self._is_composite:
                self.skipTest("This model is not a composite model!")

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, dtype=dtype)

                sub_models_supporting_fa = [
                    module._supports_flash_attn
                    for name, module in model.named_modules()
                    if isinstance(module, PreTrainedModel) and name != ""
                ]
                supports_fa_all_modules = (
                    all(sub_models_supporting_fa) if len(sub_models_supporting_fa) > 0 else model._supports_flash_attn
                )
                if not supports_fa_all_modules:
                    with self.assertRaises(ValueError):
                        model_fa = model_class.from_pretrained(
                            tmpdirname,
                            dtype=dtype,
                            attn_implementation=attn_implementation,
                        )
                else:
                    model_fa = model_class.from_pretrained(
                        tmpdirname, dtype=dtype, attn_implementation=attn_implementation
                    )
                    for key in model_fa.config:
                        if isinstance(getattr(model_fa.config, key), PretrainedConfig):
                            sub_config = getattr(model_fa.config, key)
                            self.assertTrue(sub_config._attn_implementation == attn_implementation)

                    has_fa = False
                    for name, submodule in model_fa.named_modules():
                        class_name = submodule.__class__.__name__
                        if (
                            "Attention" in class_name
                            and getattr(submodule, "config", None)
                            and submodule.config._attn_implementation == attn_implementation
                        ):
                            has_fa = True
                            break
                    if not has_fa:
                        raise ValueError(f"The {attn_implementation} model should have {attn_implementation} layers")

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    def test_flash_attn_2_can_dispatch_composite_models(self):
        self.flash_attn_can_dispatch_composite_models(attn_implementation="flash_attention_2")

    @require_flash_attn_3
    @require_torch_gpu
    @mark.flash_attn_3_test
    def test_flash_attn_3_can_dispatch_composite_models(self):
        self.flash_attn_can_dispatch_composite_models(attn_implementation="flash_attention_3")

    @require_flash_attn
    @require_torch_gpu
    @require_bitsandbytes
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_fp32_ln(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                batch_size = dummy_attention_mask.shape[0]

                is_padding_right = dummy_attention_mask[:, -1].sum().item() != batch_size

                # To avoid errors with padding_side=="right"
                if is_padding_right:
                    dummy_attention_mask = torch.ones_like(dummy_input)

                model = model_class.from_pretrained(
                    tmpdirname,
                    dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    load_in_4bit=True,
                )

                for _, param in model.named_parameters():
                    # upcast only layer norms
                    if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                        param.data = param.data.to(torch.float32)

                if model.config.is_encoder_decoder:
                    dummy_decoder_input_ids = inputs_dict["decoder_input_ids"]
                    dummy_decoder_attention_mask = inputs_dict["decoder_attention_mask"]

                    _ = model(dummy_input, decoder_input_ids=dummy_decoder_input_ids)
                    # with attention mask
                    _ = model(
                        dummy_input,
                        attention_mask=dummy_attention_mask,
                        decoder_input_ids=dummy_decoder_input_ids,
                        decoder_attention_mask=dummy_decoder_attention_mask,
                    )
                else:
                    _ = model(dummy_input)
                    # with attention mask
                    _ = model(dummy_input, attention_mask=dummy_attention_mask)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @pytest.mark.torch_compile_test
    @slow
    def test_flash_attn_2_can_compile_with_attention_mask_None_without_graph_break(self):
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        if not hasattr(self, "_torch_compile_train_cls"):
            self.skipTest(f"{self.__class__.__name__} doesn't have the attribute `_torch_compile_train_cls`.")

        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        torch.compiler.reset()
        dtype = torch.float16

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cls = self._torch_compile_train_cls  # e.g. LlamaFroCausalLM
        model = cls._from_config(config, attn_implementation="flash_attention_2").to(device=torch_device, dtype=dtype)

        inputs = {
            "input_ids": torch.randint(low=1, high=model.config.vocab_size, size=(2, 10), device=torch_device),
            "labels": torch.randint(low=1, high=model.config.vocab_size, size=(2, 10), device=torch_device),
        }

        model = torch.compile(model, fullgraph=True)
        # forward compilation
        set_seed(42)
        loss = model(**inputs).loss
        # backward compilation
        loss.backward()

        assert not loss.isnan().any()

    def attention_mask_padding_matches_padding_free_with_position_ids(
        self, attn_implementation: str, fa_kwargs: bool = False
    ):
        """
        Tests that the given attention implementation can work with packed sequences and infers the mask
        from position ids. This test requires the model to use new attention mask API which handles packing.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        max_new_tokens = 30
        support_flag = {
            "sdpa": "_supports_sdpa",
            "flash_attention_2": "_supports_flash_attn",
            "flash_attention_3": "_supports_flash_attn",
        }

        for model_class in self.all_generative_model_classes:
            if attn_implementation != "eager" and not getattr(model_class, support_flag[attn_implementation]):
                self.skipTest(f"{model_class.__name__} does not support {attn_implementation}")

            # can't infer if new attn mask API is supported by assume that only model with attention backend support it
            if not model_class._supports_attention_backend:
                self.skipTest(f"{model_class.__name__} does not support new attention mask API")

            if model_class._is_stateful:  # non-transformer models most probably have no packing support
                self.skipTest(f"{model_class.__name__} doesn't support packing!")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if config.is_encoder_decoder:
                self.skipTest("Model is an encoder-decoder")

            if 0 not in inputs_dict.get("attention_mask", []) or "attention_mask" not in inputs_dict:
                self.skipTest("Model dummy inputs should contain padding in their attention mask")

            if "input_ids" not in inputs_dict or inputs_dict["input_ids"].ndim != 2:
                self.skipTest("Model dummy inputs should contain text input ids")

            # make sure that all models have enough positions for generation
            dummy_input_ids = inputs_dict["input_ids"]
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input_ids.shape[1] + 1

            model = model_class(config)
            if "position_ids" not in inspect.signature(model.forward).parameters:
                self.skipTest("Model does not support position_ids")

            if (not fa_kwargs) and "position_ids" not in inspect.signature(model.forward).parameters:
                continue  # this model doesn't accept position ids as input

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Drop all keys except for the minimal set. Hard to manipulate with multimodals/head_mask/etc
                inputs_dict = {k: v for k, v in inputs_dict.items() if k in ["input_ids", "attention_mask"]}

                # Ensure left padding, to adapt for some models
                if 0 in inputs_dict["attention_mask"][:, -1]:
                    inputs_dict["attention_mask"] = inputs_dict["attention_mask"].flip(1)
                dummy_attention_mask = inputs_dict["attention_mask"]
                dummy_input_ids[~dummy_attention_mask.bool()] = config.get_text_config().pad_token_id

                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        dtype=torch.bfloat16,
                        attn_implementation=attn_implementation,
                    )
                    .to(torch_device)
                    .eval()
                )

                if fa_kwargs:
                    # flatten
                    features = [
                        {"input_ids": i[a.bool()].tolist()} for i, a in zip(dummy_input_ids, dummy_attention_mask)
                    ]

                    # add position_ids + fa_kwargs
                    data_collator = DataCollatorWithFlattening(return_tensors="pt", return_flash_attn_kwargs=True)
                    batch = data_collator(features)
                    padfree_inputs_dict = {
                        k: t.to(torch_device) if torch.is_tensor(t) else t for k, t in batch.items()
                    }
                else:
                    # create packed position_ids
                    position_ids = (
                        torch.cat([torch.arange(length) for length in dummy_attention_mask.sum(1).tolist()])
                        .long()
                        .unsqueeze(0)
                        .to(torch_device)
                    )
                    padfree_inputs_dict = {
                        "input_ids": dummy_input_ids[dummy_attention_mask.bool()].unsqueeze(0),
                        "position_ids": position_ids,
                    }

                # We need to do simple forward without cache in order to trigger packed SDPA/flex/eager attention path
                res_padded = model(**inputs_dict, use_cache=False)
                res_padfree = model(**padfree_inputs_dict, use_cache=False)

                logits_padded = res_padded.logits[dummy_attention_mask.bool()]
                logits_padfree = res_padfree.logits[0]

                # acceptable numerical instability
                tol = torch.finfo(torch.bfloat16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, rtol=tol, atol=tol)

    def test_eager_padding_matches_padding_free_with_position_ids(self):
        self.attention_mask_padding_matches_padding_free_with_position_ids(attn_implementation="eager")

    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        self.attention_mask_padding_matches_padding_free_with_position_ids(attn_implementation="sdpa")

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        self.attention_mask_padding_matches_padding_free_with_position_ids(attn_implementation="flash_attention_2")

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        self.attention_mask_padding_matches_padding_free_with_position_ids(
            attn_implementation="flash_attention_2", fa_kwargs=True
        )

    @require_flash_attn_3
    @require_torch_gpu
    @mark.flash_attn_3_test
    @slow
    def test_flash_attention_3_padding_matches_padding_free_with_position_ids(self):
        self.attention_mask_padding_matches_padding_free_with_position_ids(attn_implementation="flash_attention_3")

    @require_flash_attn_3
    @require_torch_gpu
    @mark.flash_attn_3_test
    @slow
    def test_flash_attention_3_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        self.attention_mask_padding_matches_padding_free_with_position_ids(
            attn_implementation="flash_attention_3", fa_kwargs=True
        )

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    def test_flash_attention_2_continue_generate_with_position_ids(self):
        """
        Tests whether flash attention can continue its generation from given position ids.

        NOTE: This serves as regression check as we had instances where flash attention entered the varlen
        path here. It should now always enter the base `flash_fn`.
        """

        max_new_tokens = 2
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention.")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if config.is_encoder_decoder:
                self.skipTest("Model is an encoder-decoder")

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(f"{model_class.__name__} doesn't support caching")

            if "input_ids" not in inputs_dict or inputs_dict["input_ids"].ndim != 2:
                self.skipTest("Model dummy inputs should contain text input ids")

            # make sure that all models have enough positions for generation
            dummy_input_ids = inputs_dict["input_ids"]
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input_ids.shape[1] + 1

            model = model_class(config)
            if "position_ids" not in inspect.signature(model.forward).parameters:
                self.skipTest("Model does not support position_ids")

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                    )
                    .to(torch_device)
                    .eval()
                )

                # Drop all keys except for `input_ids`. Hard to manipulate with multimodals/head_mask/etc
                dummy_input_ids = inputs_dict["input_ids"]
                dummy_position_ids = torch.arange(dummy_input_ids.shape[1], device=torch_device)
                dummy_position_ids = dummy_position_ids.unsqueeze(0).repeat(dummy_input_ids.shape[0], 1)

                # Store cache for the input prompt
                output = model(dummy_input_ids, position_ids=dummy_position_ids, use_cache=True)
                if "past_key_values" not in output:
                    self.skipTest("This model doesn't return `past_key_values`")

                # create new input_ids and position_ids to continue generation re-using the cache
                new_input_ids = output.logits[:, -1, :].float().argmax(-1)[:, None]
                past_length = dummy_input_ids.shape[1]
                position_ids = torch.arange(past_length, past_length + new_input_ids.shape[1], device=torch_device)
                position_ids = position_ids.unsqueeze(0).repeat(new_input_ids.shape[0], 1)

                output = model(
                    input_ids=new_input_ids,
                    past_key_values=output.past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                )
                next_token_logits = output.logits[:, -1, :].float()

                generate_kwargs = {
                    "pad_token_id": -1,
                    "eos_token_id": -1,
                    "forced_eos_token_id": None,
                    "use_cache": True,
                    "do_sample": False,
                    "return_dict_in_generate": True,
                    "output_logits": True,
                    "max_new_tokens": max_new_tokens,
                }
                generation_out = model.generate(dummy_input_ids, **generate_kwargs)
                next_token_logits_from_generate = generation_out.logits[-1]

                # acceptable numerical instability
                tol = torch.finfo(torch.bfloat16).eps
                torch.testing.assert_close(next_token_logits_from_generate, next_token_logits, rtol=tol, atol=tol)

    def flash_attn_from_config(self, attn_implementation: str):
        r"""
        Tests if the model can be loaded with `attn_implementation` from the config and if the
        weights are not randomly initialized.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support {attn_implementation}")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            # TODO: to change it in the future with other relevant auto classes
            fa_model = model_class._from_config(
                config, attn_implementation=attn_implementation, dtype=torch.bfloat16
            ).to(torch_device)

            dummy_input = inputs_dict[fa_model.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.float16]:
                dummy_input = dummy_input.to(torch.bfloat16)
            dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))

            if fa_model.config.is_encoder_decoder:
                dummy_decoder_input_ids = inputs_dict["decoder_input_ids"]
                dummy_decoder_attention_mask = inputs_dict["decoder_attention_mask"]
                _ = fa_model(
                    dummy_input,
                    attention_mask=dummy_attention_mask,
                    decoder_input_ids=dummy_decoder_input_ids,
                    decoder_attention_mask=dummy_decoder_attention_mask,
                )
            else:
                _ = fa_model(dummy_input, attention_mask=dummy_attention_mask)

            with tempfile.TemporaryDirectory() as tmpdirname:
                fa_model.save_pretrained(tmpdirname)
                model_from_pretrained = model_class.from_pretrained(tmpdirname)
                self.assertTrue(model_from_pretrained.config._attn_implementation != attn_implementation)

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_from_config(self):
        self.flash_attn_from_config(attn_implementation="flash_attention_2")

    @require_flash_attn_3
    @require_torch_gpu
    @mark.flash_attn_3_test
    @slow
    def test_flash_attn_3_from_config(self):
        self.flash_attn_from_config(attn_implementation="flash_attention_3")

    def _get_custom_4d_mask_test_data(self):
        # Sequence in which all but the last token is the same
        input_ids = torch.tensor(
            [[10, 11, 12, 13], [10, 11, 12, 14], [10, 11, 12, 15]], device=torch_device, dtype=torch.int64
        )
        position_ids = torch.tensor([[0, 1, 2, 3]] * 3, device=torch_device, dtype=torch.int64)

        # Combining common prefix with the unique ending tokens:
        input_ids_shared_prefix = torch.cat([input_ids[0][:-1], input_ids[:, -1]]).unsqueeze(0)

        # Creating a 4D mask where each of the last 3 tokens do not attend to each other.
        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 1, 0],
                        [1, 1, 1, 0, 0, 1],
                    ]
                ]
            ],
        )
        # inverting the attention mask
        mask_dtype = torch.float32
        min_dtype = torch.finfo(mask_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=mask_dtype, device=torch_device) * min_dtype

        # Creating a position_ids tensor. note the repeating figures in the end.
        position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 3, 3]], device=torch_device, dtype=torch.int64)

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix

    def test_sliding_window_mask(self):
        """Tests that we can control the sliding window attention behavior of a model."""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        if not self.has_attentions:
            self.skipTest(reason="Model does not support output_attentions")

        if not (hasattr(config, "sliding_window") and hasattr(config, "use_sliding_window")):
            self.skipTest(reason="Model does not support sliding window mask")

        seq_len = self.model_tester.seq_length
        batch_size = self.model_tester.batch_size
        sliding_window = 3  # set to arbitrary small number

        sliding_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - sliding_window + 1)
            sliding_mask[i, start : i + 1] = True
        sliding_mask = sliding_mask.to(torch_device)

        config.sliding_window = sliding_window
        inputs["attention_mask"] = torch.ones(batch_size, seq_len).to(torch.int64).to(torch_device)
        for model_class in self.all_model_classes:
            # Set sliding window to `True` and check that all tokens beyond window size are masked
            config.use_sliding_window = True
            config_dict = config.to_diff_dict()
            if hasattr(config, "layer_types"):
                del config_dict["layer_types"]
            new_config = config.__class__(**config_dict)
            # We need to set eager as otherwise `output_attentions` is not supported
            model = model_class._from_config(new_config, attn_implementation="eager").to(torch_device)
            model.eval()
            layer_types = getattr(model.config, "layer_types", ["sliding_attention"] * config.num_hidden_layers)
            attentions = model(**inputs, output_attentions=True).attentions
            for layer_attention, layer_type in zip(attentions, layer_types):
                if layer_type == "sliding_attention":
                    self.assertTrue((layer_attention[:, :, ~sliding_mask] == 0).all().item())
                else:
                    self.assertFalse((layer_attention[:, :, ~sliding_mask] == 0).all().item())

            # Set sliding window to `False` while keeping `sliding_window=3`
            # Check that all tokens beyond window size are not masked
            config.use_sliding_window = False
            config_dict = config.to_diff_dict()
            if hasattr(config, "layer_types"):
                del config_dict["layer_types"]
            new_config = config.__class__(**config_dict)
            # We need to set eager as otherwise `output_attentions` is not supported
            model = model_class._from_config(new_config, attn_implementation="eager").to(torch_device)
            model.eval()
            attentions_not_sliding = model(**inputs, output_attentions=True).attentions
            for layer_attention in attentions_not_sliding:
                self.assertFalse((layer_attention[:, :, ~sliding_mask] == 0).all().item())

    def test_custom_4d_attention_mask(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if len(self.all_generative_model_classes) == 0:
            self.skipTest(
                reason="Model architecture has no generative classes, and thus not necessarily supporting 4D masks"
            )

        set_model_tester_for_less_flaky_test(self)

        for model_class in self.all_generative_model_classes:
            if not model_class._can_compile_fullgraph:
                self.skipTest(f"{model_class.__name__} is not guaranteed to work with custom 4D attention masks")
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            if getattr(config, "sliding_window", 0) is not None and getattr(config, "sliding_window", 0) > 0:
                self.skipTest(f"{model_class.__name__} with sliding window attention is not supported by this test")
            model = model_class(config).to(device=torch_device, dtype=torch.float32).eval()
            set_model_for_less_flaky_test(model)
            if "position_ids" not in inspect.signature(model.forward).parameters:
                continue  # model doesn't accept position ids and probably has special way to model positions

            if "position_ids" not in inspect.signature(model.forward).parameters:
                continue  # this model doesn't accept position ids as input

            (
                input_ids,
                position_ids,
                input_ids_shared_prefix,
                mask_shared_prefix,
                position_ids_shared_prefix,
            ) = self._get_custom_4d_mask_test_data()

            logits = model.forward(input_ids, position_ids=position_ids).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                input_ids_shared_prefix,
                attention_mask=mask_shared_prefix,
                position_ids=position_ids_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = F.softmax(out_last_tokens, dim=-1)
            normalized_1 = F.softmax(out_shared_prefix_last_tokens, dim=-1)
            torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-3)

    @slow
    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_torch_compile_for_training(self):
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        if getattr(self, "_torch_compile_train_cls", None) is None:
            self.skipTest(f"{self.__class__.__name__} doesn't have the attribute `_torch_compile_train_cls`.")

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cls = self._torch_compile_train_cls
        attn_implementation = getattr(self, "_torch_compile_train_attn_implementation", None)
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation

        model = cls(config).to(torch_device)

        inputs = {
            "input_ids": torch.randint(low=1, high=model.config.vocab_size, size=(2, 10), device=torch_device),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                dtype=torch.int64,
                device=torch_device,
            ),
            "position_ids": torch.arange(0, 10, device=torch_device).unsqueeze(0),
            "labels": torch.randint(low=1, high=model.config.vocab_size, size=(2, 10), device=torch_device),
            "use_cache": False,
        }

        # eager backward
        set_seed(42)
        loss = model(**inputs).loss
        loss.backward()

        params = {name: param.grad.detach().to(device="cpu", copy=True) for name, param in model.named_parameters()}
        model.zero_grad()
        del loss

        model = torch.compile(model, fullgraph=True, mode="reduce-overhead")

        # forward compilation
        set_seed(42)
        loss = model(**inputs).loss
        # backward compilation
        loss.backward()
        # check grad matches
        for name, param in model._orig_mod.named_parameters():
            torch.testing.assert_close(param.grad.detach().cpu(), params[name], rtol=1e-4, atol=1e-4)

    def test_forward_with_logits_to_keep(self):
        for model_class in self.all_generative_model_classes:
            if "logits_to_keep" not in set(inspect.signature(model_class.forward).parameters.keys()):
                self.skipTest(reason="This model does not support `logits_to_keep` argument.")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
            batch_size, sequence_length = inputs["input_ids"].shape[:2]
            vocab_size = config.get_text_config().vocab_size
            model = model_class(config).to(device=torch_device).eval()
            # some models have labels but `logits_to_keep` should not be used in train mode
            _ = inputs.pop("labels", None)

            # logits_to_keep=0 is a special case meaning "keep all logits"
            all_logits = model(**inputs, logits_to_keep=0).logits
            last_token_logits = model(**inputs, logits_to_keep=1).logits

            # Assert all shapes are correct
            self.assertEqual(tuple(all_logits.shape), (batch_size, sequence_length, vocab_size))
            self.assertEqual(tuple(last_token_logits.shape), (batch_size, 1, vocab_size))

            # Assert the last tokens are actually the same (except for the natural fluctuation due to order of FP ops)
            torch.testing.assert_close(all_logits[:, -1:, :], last_token_logits, rtol=1e-5, atol=1e-5)

    @slow
    @require_torch_greater_or_equal("2.5")
    @pytest.mark.torch_export_test
    def test_torch_export(self, config=None, inputs_dict=None, tolerance=1e-4):
        """
        Test if model can be exported with torch.export.export()

        Args:
            config (PretrainedConfig):
                Config to use for the model, if None, use default config from model_tester
            inputs_dict (dict):
                Inputs to use for the model, if None, use default inputs from model_tester
            tolerance (float):
                `atol` for torch.allclose(), defined in signature for test overriding
        """
        if not self.test_torch_exportable:
            self.skipTest(reason="test_torch_exportable=False for this model.")

        def recursively_check(eager_outputs, exported_outputs):
            is_tested = False
            if isinstance(eager_outputs, torch.Tensor):
                torch.testing.assert_close(eager_outputs, exported_outputs, atol=tolerance, rtol=tolerance)
                return True
            elif isinstance(eager_outputs, (tuple, list)):
                for eager_output, exported_output in zip(eager_outputs, exported_outputs):
                    is_tested = is_tested or recursively_check(eager_output, exported_output)
                return is_tested
            elif isinstance(eager_outputs, dict):
                for key in eager_outputs:
                    is_tested = is_tested or recursively_check(eager_outputs[key], exported_outputs[key])
                return is_tested
            return is_tested

        default_config, default_inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config = config or default_config
        inputs_dict = inputs_dict or default_inputs_dict

        for model_class in self.all_model_classes:
            if model_class.__name__.endswith("ForPreTraining"):
                continue

            with self.subTest(model_class.__name__):
                model = model_class(config).eval().to(torch_device)

                # Export model
                exported_model = torch.export.export(
                    model, args=(), kwargs=inputs_dict, strict=getattr(self, "test_torch_exportable_strictly", True)
                )

                # Run exported model and eager model
                with torch.no_grad():
                    # set seed in case anything is not deterministic in model (e.g. vit_mae noise)
                    torch.manual_seed(1234)
                    eager_outputs = model(**inputs_dict)
                    torch.manual_seed(1234)
                    exported_outputs = exported_model.module().forward(**inputs_dict)

                # Check if outputs are close:
                # is_tested is a boolean flag indicating if we compare any outputs,
                # e.g. there might be a situation when outputs are empty list, then is_tested will be False.
                # In case of outputs are different the error will be raised in `recursively_check` function.
                is_tested = recursively_check(eager_outputs, exported_outputs)
                self.assertTrue(is_tested, msg=f"No outputs were compared for {model_class.__name__}")

    @staticmethod
    def _prepare_config_headdim(config, requested_dim):
        """
        This method allows to update the head dim for all model types including
        composite models and models that do not support head dim by themselves.

        Why? A lot of kernels including flex attention rely on triton for compilation.
        However, triton cannot handle hidden dimensions of less than 16 for example.
        (There are many more examples especially now that the `kernels` library is
        supported)
        """
        config = copy.deepcopy(config)

        def update_config_headdim(config, requested_dim):
            # Flex Attention cannot use dropout
            if hasattr(config, "attention_dropout"):
                config.attention_dropout = 0
            if hasattr(config, "attention_probs_dropout_prob"):
                config.attention_probs_dropout_prob = 0

            # Update the head dim and try to update hidden size as well if present in config
            # NOTE: some models may have none if the values in sub-config, thus we check for `Noneness`
            head_dim = None
            if hasattr(config, "head_dim") and config.head_dim is not None:
                head_dim = config.head_dim
                config.head_dim = max(requested_dim, config.head_dim)

            cross_head_dim = None
            if hasattr(config, "cross_head_dim") and config.cross_head_dim is not None:
                cross_head_dim = config.cross_head_dim
                config.cross_head_dim = max(requested_dim, config.cross_head_dim)

            if (
                getattr(config, "hidden_size", None) is not None
                and getattr(config, "num_attention_heads", None) is not None
            ):
                head_dim = head_dim if head_dim is not None else config.hidden_size // config.num_attention_heads
                config.hidden_size *= max(requested_dim // head_dim, 1)

            if (
                getattr(config, "decoder_hidden_size", None) is not None
                and getattr(config, "decoder_num_attention_heads", None) is not None
            ):
                decoder_head_dim = config.decoder_hidden_size // config.decoder_num_attention_heads
                config.decoder_hidden_size *= max(requested_dim // decoder_head_dim, 1)

            if (
                getattr(config, "cross_hidden_size", None) is not None
                and getattr(config, "cross_num_attention_heads", None) is not None
            ):
                cross_head_dim = (
                    cross_head_dim
                    if cross_head_dim is not None
                    else config.cross_hidden_size // config.cross_num_attention_heads
                )
                config.cross_hidden_size *= max(requested_dim // cross_head_dim, 1)

            # 3d rope also depends on the head dim
            # (we assume easy shapes here where we get to the requested head dim at least)
            if (
                getattr(config, "rope_scaling", None) is not None
                and len(config.rope_scaling.get("mrope_section", None)) > 0
            ):
                scaling_factor = max(requested_dim // (sum(config.rope_scaling["mrope_section"]) * 2), 1)
                config.rope_scaling["mrope_section"] = [
                    section * scaling_factor for section in config.rope_scaling["mrope_section"]
                ]

        # Update config values
        update_config_headdim(config, requested_dim)
        for key in config.sub_configs:
            sub_config = getattr(config, key)
            update_config_headdim(sub_config, requested_dim)

        return config

    @require_torch_gpu
    def test_flex_attention_with_grads(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config).to(device=torch_device)

            # If not all sub-models support flex, skip the test
            if not all(
                submodel._supports_flex_attn for submodel in model.modules() if isinstance(submodel, PreTrainedModel)
            ):
                self.skipTest(reason="At least some parts of this model do not support flex attention")

            # Set default attention to flex and update config values
            config = self._prepare_config_headdim(config, 16)  # specific to triton

            if model_class._can_set_attn_implementation():
                model = model_class(config).to(device=torch_device)
                model.set_attn_implementation("flex_attention")
                self.assertTrue(model.config._attn_implementation == "flex_attention")
            else:
                config._attn_implementation = "flex_attention"
                model = model_class(config).to(device=torch_device)

            # Elaborate workaround for encoder-decoder models as some do not specify their main input
            dummy_inputs = {model.main_input_name: inputs_dict[model.main_input_name].to(torch_device)}
            for key in getattr(self, "additional_model_inputs", []):
                # Some models don't have all `additional_model_inputs`, especially when we
                # craft cases to test model in different settings
                if key in inputs_dict:
                    dummy_inputs[key] = inputs_dict[key].to(torch_device)

            if config.get_text_config(decoder=True).is_encoder_decoder:
                dummy_inputs["decoder_input_ids"] = inputs_dict["decoder_input_ids"].to(torch_device)
                dummy_inputs["decoder_attention_mask"] = inputs_dict["decoder_attention_mask"].to(torch_device)

            # If this does not raise an error, the test passes (see https://github.com/huggingface/transformers/pull/35605)
            _ = model(**dummy_inputs)

    def test_generation_tester_mixin_inheritance(self):
        """
        Ensures that we have the generation tester mixin if the model can generate. The test will fail otherwise,
        forcing the mixin to be added -- and ensuring proper test coverage
        """
        if len(self.all_generative_model_classes) > 0:
            self.assertTrue(
                issubclass(self.__class__, GenerationTesterMixin),
                msg=(
                    "This model can call `generate` from `GenerationMixin`, so one of two things must happen: 1) the "
                    "tester must inherit from `GenerationTesterMixin` to run `generate` tests, or 2) if the model "
                    "doesn't fully support the original `generate` or has a custom `generate` with partial feature "
                    "support, the tester must overwrite `all_generative_model_classes` to skip the failing classes "
                    "(make sure to comment why). If `all_generative_model_classes` is overwritten as `()`, then we "
                    "need to remove the `GenerationTesterMixin` inheritance -- no `generate` tests are being run."
                ),
            )
        else:
            self.assertFalse(
                issubclass(self.__class__, GenerationTesterMixin),
                msg=(
                    "This model can't call `generate`, so its tester can't inherit `GenerationTesterMixin`. (If you "
                    "think the model should be able to `generate`, the model may be missing the `GenerationMixin` "
                    "inheritance)"
                ),
            )

    def test_can_be_initialized_on_meta(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # If it does not raise here, the test passes
            with torch.device("meta"):
                _ = model_class(copy.deepcopy(config))

    @require_torch_accelerator
    def test_can_load_with_device_context_manager(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        # Need to specify index 0 here, as `torch_device` is simply the str of the type, e.g. "cuda"
        device = torch.device(torch_device, index=0)
        for model_class in self.all_model_classes:
            # Need to deepcopy here as it is modified in-place in save_pretrained (it sets sdpa for default attn, which
            # is not supported for e.g. dpt_hybrid)
            model = model_class(copy.deepcopy(config))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                with device:
                    new_model = model_class.from_pretrained(tmpdirname)
                unique_devices = {param.device for param in new_model.parameters()} | {
                    buffer.device for buffer in new_model.buffers()
                }

            self.assertEqual(
                unique_devices, {device}, f"All parameters should be on {device}, but found {unique_devices}."
            )

    # Here we need to run with a subprocess as otherwise setting back the default device to the default value ("cpu")
    # may bring unwanted consequences on other tests. See PR #37553
    @run_first
    @run_test_using_subprocess
    @require_torch_accelerator
    def test_can_load_with_global_device_set(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        # Need to specify index 0 here, as `torch_device` is simply the str of the type, e.g. "cuda"
        device = torch.device(torch_device, index=0)
        default_device = torch.get_default_device()
        for model_class in self.all_model_classes:
            # Need to deepcopy here as it is modified in-place in save_pretrained (it sets sdpa for default attn, which
            # is not supported for e.g. dpt_hybrid)
            model = model_class(copy.deepcopy(config))

            # set a global gpu device
            torch.set_default_device(device)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                new_model = model_class.from_pretrained(tmpdirname)
                unique_devices = {param.device for param in new_model.parameters()} | {
                    buffer.device for buffer in new_model.buffers()
                }

            # set back the correct device
            torch.set_default_device(default_device)

            self.assertEqual(
                unique_devices, {device}, f"All parameters should be on {device}, but found {unique_devices}."
            )

    def test_can_load_with_meta_device_context_manager(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # Need to deepcopy here as it is modified in-place in save_pretrained (it sets sdpa for default attn, which
            # is not supported for e.g. dpt_hybrid)
            model = model_class(copy.deepcopy(config))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                with torch.device("meta"):
                    new_model = model_class.from_pretrained(tmpdirname)
                unique_devices = {param.device for param in new_model.parameters()} | {
                    buffer.device for buffer in new_model.buffers()
                }

            self.assertEqual(
                unique_devices,
                {torch.device("meta")},
                f"All parameters should be on meta device, but found {unique_devices}.",
            )

    def test_config_attn_implementation_setter(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        def check_attn_implementation_setter(config: PretrainedConfig, attn_implementation: str):
            if not config._attn_implementation == attn_implementation:
                raise ValueError(
                    f"Unexpected attn_implementation for config {config.__class__.__name__}: "
                    f"{config._attn_implementation} != {attn_implementation}"
                )
            for attribute_value in config.__dict__.values():
                if isinstance(attribute_value, PretrainedConfig):
                    check_attn_implementation_setter(attribute_value, attn_implementation)

        config._attn_implementation = "eager"
        check_attn_implementation_setter(config, "eager")

        config._attn_implementation = "sdpa"
        check_attn_implementation_setter(config, "sdpa")

    def test_internal_model_config_and_subconfig_are_same(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        subconfig_keys = list(config.sub_configs.keys())
        for model_class in self.all_model_classes:
            if len(config.sub_configs) == 0:
                self.skipTest(reason="No subconfigs so the test does not make sense")
            # Need to deepcopy here to avoid changing the _attn_implementation in-place
            model = model_class(copy.deepcopy(config))

            for submodule in model.modules():
                # This is a submodel
                if isinstance(submodule, PreTrainedModel) and submodule.config.__class__ != model.config.__class__:
                    subconfig_from_model_internal = submodule.config
                    matching_sub_configs = []
                    for subconfig_key in subconfig_keys:
                        # Get the subconfig from the model config
                        subconfig_from_model_config = getattr(model.config, subconfig_key)
                        if subconfig_from_model_config.__class__ == subconfig_from_model_internal.__class__:
                            # Since some composite models have different submodels parameterized by 2 of the same config
                            # class instances, we need to check against a list of matching classes, and check that at least
                            # 1 is the exact object (instead of checking immediately for similar object)
                            matching_sub_configs.append(subconfig_from_model_config)

                    # Both should be exactly the same object, that is when instantiating the submodel when should
                    # absolutely not copy the subconfig
                    if len(matching_sub_configs) > 0:
                        self.assertTrue(
                            any(
                                subconfig_from_model_config is subconfig_from_model_internal
                                for subconfig_from_model_config in matching_sub_configs
                            )
                        )

    def test_can_set_attention_dynamically(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if not model_class._can_set_attn_implementation():
                self.skipTest(reason="This model does not support setting its attention dynamically")

            # Need to deepcopy here to avoid changing the _attn_implementation in-place
            model_config = copy.deepcopy(config)
            # Set eager everywhere (it sets it recursively on subconfigs)
            model_config._attn_implementation = "eager"
            model = model_class(model_config)

            # sanity check to make sure everything is correctly eager
            self.assertTrue(model.config._attn_implementation == "eager")
            for subconfig_key in model.config.sub_configs:
                self.assertTrue(getattr(model.config, subconfig_key)._attn_implementation == "eager")

            if not all(
                submodule._can_set_attn_implementation()
                for submodule in model.modules()
                if isinstance(submodule, PreTrainedModel)
            ):
                self.skipTest(reason="Parts of this model cannot set attention dynamically")
            # Some old models technically should support switching, but don't have the flags active...
            if not all(
                submodule._supports_sdpa for submodule in model.modules() if isinstance(submodule, PreTrainedModel)
            ):
                self.skipTest(reason="Parts of this model don't support sdpa")

            # Now, set it to sdpa
            model.set_attn_implementation("sdpa")

            # Check everything was correctly changed
            self.assertTrue(model.config._attn_implementation == "sdpa")
            for subconfig_key in model.config.sub_configs:
                self.assertTrue(getattr(model.config, subconfig_key)._attn_implementation == "sdpa")

            # Check we cannot set it to random values, and it raises an error
            with self.assertRaisesRegex(ValueError, 'Specified `attn_implementation="foo"` is not supported'):
                model.set_attn_implementation("foo")

            # Should still be sdpa everywhere
            self.assertTrue(model.config._attn_implementation == "sdpa")
            for subconfig_key in model.config.sub_configs:
                self.assertTrue(getattr(model.config, subconfig_key)._attn_implementation == "sdpa")

    def test_can_set_attention_dynamically_composite_model(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if not model_class._can_set_attn_implementation():
                self.skipTest(reason="This model does not support setting its attention dynamically")
            if not self._is_composite:
                self.skipTest(reason="This model is not composite")

            # Need to deepcopy here to avoid changing the _attn_implementation in-place
            model_config = copy.deepcopy(config)
            # Set eager everywhere (it sets it recursively on subconfigs)
            model_config._attn_implementation = "eager"
            model = model_class(model_config)

            # sanity check to make sure everything is correctly eager
            self.assertTrue(model.config._attn_implementation == "eager")
            for subconfig_key in model.config.sub_configs:
                self.assertTrue(getattr(model.config, subconfig_key)._attn_implementation == "eager")

            if not all(
                submodule._can_set_attn_implementation()
                for submodule in model.modules()
                if isinstance(submodule, PreTrainedModel)
            ):
                self.skipTest(reason="Parts of this model cannot set attention dynamically")

            # Now, set only top-most to sdpa (should support it if it supports the dynamic switch)
            model.set_attn_implementation({"": "sdpa"})

            # Check only top-most was correctly changed
            self.assertTrue(model.config._attn_implementation == "sdpa")
            for subconfig_key in model.config.sub_configs:
                self.assertTrue(getattr(model.config, subconfig_key)._attn_implementation == "eager")

    @require_torch
    def test_bc_torch_dtype(self):
        """
        Test that we can still use `torch_dtype` argument correctly, for BC.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if "TimmBackbone" in model_class.__name__:
                self.skipTest("TimmBackbone should not run this test")
            # First check that it works correctly
            model = model_class(copy.deepcopy(config))
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Check that it works for all dtypes
                for dtype in ["float16", "bfloat16", "float32", "auto", torch.float16, torch.bfloat16, torch.float32]:
                    model_torch_dtype = model_class.from_pretrained(tmpdirname, torch_dtype=dtype)
                    model_dtype = model_class.from_pretrained(tmpdirname, dtype=dtype)
                    for (k1, v1), (k2, v2) in zip(
                        model_torch_dtype.named_parameters(), model_dtype.named_parameters()
                    ):
                        self.assertEqual(k1, k2)
                        self.assertEqual(v1.dtype, v2.dtype)
                        self.assertTrue((v1 == v2).all())


global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()


def random_attention_mask(shape, rng=None, name=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None)
    # make sure that at least one token is attended to for each batch
    # we choose the 1st token so this property of `at least one being non-zero` still holds after applying causal mask
    attn_mask[:, 0] = 1
    return attn_mask


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()
