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
import inspect
import math
import os
import os.path
import random
import re
import tempfile
import unittest.mock
import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pytest
from parameterized import parameterized
from pytest import mark
from safetensors.torch import load_file

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    PreTrainedConfig,
    PreTrainedModel,
    is_torch_available,
    logging,
    set_seed,
)
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightRenaming, process_target_pattern
from transformers.integrations import HfDeepSpeedConfig
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)
from transformers.integrations.moe import batched_mm_experts_forward, grouped_mm_experts_forward
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import FLASH_ATTN_KERNEL_FALLBACK, _get_tied_weight_keys
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
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
    MODEL_MAPPING_NAMES,
)
from transformers.testing_utils import (
    CaptureLogger,
    force_serialization_as_bin_files,
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
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_mps,
    require_torch_multi_accelerator,
    require_torch_multi_gpu,
    run_first,
    run_test_using_subprocess,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import (
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    ModelOutput,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
)
from transformers.utils.output_capturing import CompileableContextVar

from .generation.test_utils import GenerationTesterMixin


if is_torch_available():
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
    from torch import nn

    from transformers import MODEL_MAPPING
    from transformers.integrations.accelerate import compute_module_sizes
    from transformers.integrations.tensor_parallel import _get_parameter_tp_plan
    from transformers.modeling_utils import load_state_dict
    from transformers.pytorch_utils import id_tensor_storage

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
    Otherwise, `parameterize.expand` prevents it as it removes the original function from the namespace.
    """
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

    def _can_output_attn(model):
        parameters = inspect.signature(model.forward).parameters
        if "output_attentions" in parameters:
            return True

        kwargs_param = parameters.get("kwargs")
        if kwargs_param is not None:
            try:
                annotation = kwargs_param.annotation.__args__
                return "output_attentions" in annotation[0].__annotations__
            except AttributeError:
                return False
        return False

    for model_class in self.all_model_classes:
        # Set seed for deterministic test - ensures reproducible model initialization and inputs
        set_seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)

        # If it's a model with sliding window attention, let's test it with sliding window
        if hasattr(config, "sliding_window"):
            config.sliding_window = 2

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

            try:
                model_eager = deepcopy(model_sdpa)
                model_eager.set_attn_implementation("eager")
            except Exception as _:
                model_eager = model_class.from_pretrained(**model_from_pretrained_kwargs, attn_implementation="eager")
            model_eager = model_eager.eval().to(torch_device)

        set_model_for_less_flaky_test(model_eager)
        set_model_for_less_flaky_test(model_sdpa)

        can_output_attn = _can_output_attn(model_sdpa)
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

                if self.has_attentions and _can_output_attn(model_sdpa):
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
            elif torch_device in ["hpu", "npu"]:
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

            # Avoid test flakiness with bf16!
            # bf16 is not good at precision when the magnitude is larger. We have some models like `SiglipVision` with
            # this test passing all the time for fp32/fp16 but flaky with bf16. Furthermore, `llama` and `clip` have
            # this test passing all the time for bf16: it turns out their outputs are of smaller size (0.1 and 1.0)
            # while `siglip` has outputs with maximal values around 3.0/4.0.
            outputs_magnitude = float(
                (torch.max(logits_sdpa.abs().amax(), logits_eager.abs().amax())).detach().to("cpu")
            )
            # The choice of `3e-2` in `outputs_magnitude * 1e-2` might not work if a model has even more larger outputs.
            # (we can try to analyze the `rtol` more closely element-wise in the future and adjust the `rtol` instead of `atol`).
            computed_atol = outputs_magnitude * 3e-2
            if dtype == torch.bfloat16:
                atol = max(atol, computed_atol)

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


TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION = [
    (
        # test name for the test runner
        f"{dtype}",
        # parameterization
        *(dtype,),
    )
    for dtype in ("fp16", "fp32", "bf16")
]


def _get_output_tensors(outputs):
    output_tensors = []

    if hasattr(outputs, "logits"):
        output_tensors.append(outputs.logits)
    if hasattr(outputs, "last_hidden_state"):
        output_tensors.append(outputs.last_hidden_state)
    if hasattr(outputs, "start_logits"):
        output_tensors.append(outputs.start_logits)
    if hasattr(outputs, "end_logits"):
        output_tensors.append(outputs.end_logits)

    return output_tensors


def _test_eager_matches_batched_and_grouped_inference(self, name, dtype):
    if not self.all_model_classes[0]._can_set_experts_implementation():
        self.skipTest(f"{self.all_model_classes[0].__name__} does not support grouped_mm")

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

    for model_class in self.all_model_classes:
        # Set seed for deterministic test - ensures reproducible model initialization and inputs
        set_seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        set_config_for_less_flaky_test(config)
        model = model_class(config).eval().to(torch_device).to(dtype)
        set_model_for_less_flaky_test(model)

        # Reload to find any buffer misalignments after saving/loading
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = model_class.from_pretrained(tmpdirname).eval().to(torch_device).to(dtype)

        with torch.no_grad():
            inputs_dict = {k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in inputs_dict.items()}
            prepared_inputs = self._prepare_for_class(inputs_dict, model_class)

            mock_batched_mm_forward = Mock(wraps=batched_mm_experts_forward)
            mock_grouped_mm_forward = Mock(wraps=grouped_mm_experts_forward)
            with (
                # This is needed because we call the functions through the interface's global mapping
                patch.dict(
                    "transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS._global_mapping",
                    {"batched_mm": mock_batched_mm_forward, "grouped_mm": mock_grouped_mm_forward},
                ),
            ):
                model.set_experts_implementation("eager")
                self.assertEqual(model.config._experts_implementation, "eager")
                outputs_eager = model(**prepared_inputs)
                mock_batched_mm_forward.assert_not_called()
                mock_grouped_mm_forward.assert_not_called()

                mock_batched_mm_forward.reset_mock()
                mock_grouped_mm_forward.reset_mock()

                model.set_experts_implementation("batched_mm")
                self.assertEqual(model.config._experts_implementation, "batched_mm")
                outputs_batched_mm = model(**prepared_inputs)
                mock_grouped_mm_forward.assert_not_called()
                mock_batched_mm_forward.assert_called()

                mock_batched_mm_forward.reset_mock()
                mock_grouped_mm_forward.reset_mock()

                model.set_experts_implementation("grouped_mm")
                self.assertEqual(model.config._experts_implementation, "grouped_mm")
                outputs_grouped_mm = model(**prepared_inputs)
                mock_batched_mm_forward.assert_not_called()
                mock_grouped_mm_forward.assert_called()

                mock_batched_mm_forward.reset_mock()
                mock_grouped_mm_forward.reset_mock()

        # extract output tensors for comparison
        outputs_eager = _get_output_tensors(outputs_eager)
        outputs_batched_mm = _get_output_tensors(outputs_batched_mm)
        outputs_grouped_mm = _get_output_tensors(outputs_grouped_mm)

        # make sure we have collected some tensors from the outputs
        self.assertTrue(outputs_eager, "No outputs from eager implementation")
        self.assertTrue(outputs_batched_mm, "No outputs from batched_mm implementation")
        self.assertTrue(outputs_grouped_mm, "No outputs from grouped_mm implementation")

        # make sure all implementations give numerically close outputs
        torch.testing.assert_close(outputs_eager, outputs_batched_mm, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(outputs_eager, outputs_grouped_mm, rtol=1e-4, atol=1e-4)


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__:
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PreTrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init


def _mock_init_weights(self, module):
    for name, param in module.named_parameters(recurse=False):
        # Use the first letter of the name to get a value and go from a <> -13 to z <> 12
        value = ord(name[0].lower()) - 110
        param.data.fill_(value)


def _mock_all_init_weights(self):
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
    test_resize_embeddings = True
    test_resize_position_embeddings = False
    test_mismatched_shapes = True
    test_missing_keys = True
    test_torch_exportable = True
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
            inputs_dict.pop("attention_mask", None)
        elif model_class.__name__ == MODEL_FOR_PRETRAINING_MAPPING_NAMES["hiera"]:
            config = self.model_tester.get_config()
            mask_spatial_shape = [
                i // s // ms for i, s, ms in zip(config.image_size, config.patch_stride, config.masked_unit_size)
            ]
            num_windows = math.prod(mask_spatial_shape)
            set_seed(42)
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
                *get_values(MODEL_FOR_CTC_MAPPING_NAMES),
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

    def test_num_layers_is_small(self):
        # TODO (if possible): Avoid exceptional cases, especially for `OwlViT`.
        # ⛔ DO NOT edit this list (unless there is really nothing to tweak in the model tester class and approved by the reviewer) ⛔!
        exceptional_num_hidden_layers = {
            # TODO: There might be some way to fix
            "FunnelModelTest": 5,
            "FunnelBaseModelTest": 4,
            "GroupViTVisionModelTest": 12,
            "OwlViTModelTest": 12,
            "OwlViTTextModelTest": 12,
            "OwlViTForObjectDetectionTest": 12,
            "Owlv2ModelTest": 12,
            "Owlv2TextModelTest": 12,
            "Owlv2ForObjectDetectionTest": 12,
            "Qwen2_5OmniThinkerForConditionalGenerationModelTest": 4,
            "Qwen3OmniMoeThinkerForConditionalGenerationModelTest": 4,
            "SamHQModelTest": 12,
            "Swin2SRModelTest": 3,
            "XLNetModelTest": 3,
            "DPTModelTest": 4,  # `test_modeling_dpt_hybrid.py`: not able to get it work after change `num_hidden_layers` and `neck_hidden_sizes`
            # Nothing we can't do
            "Gemma3nTextModelTest": 4,  # need to test KV shared layer for both types: `full_attention` and `sliding_attention`
            "Gemma3nVision2TextModelTest": 4,  # need to test KV shared layer for both types: `full_attention` and `sliding_attention`
            "BeitModelTest": 4,  # BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers
            "ZambaModelTest": 5,  # The minimum number to test beyond the initial ["mamba", "mamba", "hybrid"] in `ZambaConfig._layers_block_type`
        }
        target_num_hidden_layers = exceptional_num_hidden_layers.get(type(self).__name__, 2)

        if hasattr(self.model_tester, "num_hidden_layers") and isinstance(self.model_tester.num_hidden_layers, int):
            assert self.model_tester.num_hidden_layers <= target_num_hidden_layers

        if hasattr(self.model_tester, "vision_config") and "num_hidden_layers" in self.model_tester.vision_config:
            if isinstance(self.model_tester.vision_config, dict):
                assert self.model_tester.vision_config["num_hidden_layers"] <= target_num_hidden_layers
            else:
                assert self.model_tester.vision_config.num_hidden_layers <= target_num_hidden_layers
        if hasattr(self.model_tester, "text_config") and "num_hidden_layers" in self.model_tester.text_config:
            if isinstance(self.model_tester.text_config, dict):
                assert self.model_tester.text_config["num_hidden_layers"] <= target_num_hidden_layers
            else:
                assert self.model_tester.text_config.num_hidden_layers <= target_num_hidden_layers

    def test_save_load(self):
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
                model.eval()
                with torch.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

                # Save and load second time because `from_pretrained` adds a bunch of new config fields
                # so we need to make sure those fields can be loaded back after saving
                # Simply init as `model(config)` doesn't add those fields
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    torch.testing.assert_close(
                        tensor1, tensor2, msg="Running save/load and forward yields different results"
                    )
            else:
                torch.testing.assert_close(first, second, msg="Running save/load and forward yields different results")

    def test_from_pretrained_no_checkpoint(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            state_dict = model.state_dict()

            new_model = model_class.from_pretrained(
                pretrained_model_name_or_path=None, config=config, state_dict=state_dict
            )
            new_state_dict = new_model.state_dict()
            assert state_dict.keys() == new_state_dict.keys()
            keys = state_dict.keys()
            for k in keys:
                p1, p2 = new_state_dict[k], state_dict[k]
                with self.subTest(k):
                    torch.testing.assert_close(p1, p2, msg=f"failed on {k}")

            new_params = dict(new_model.named_parameters())
            for k, v in list(model.named_parameters()):
                with self.subTest(k):
                    torch.testing.assert_close(v, new_params[k], msg=f"failed on {k}")

    def test_keep_in_fp32_modules_exist(self):
        """Test that both the `_keep_in_fp32` and `_keep_in_fp32_strict` targets match some layers, to avoid any typo"""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))
                # Make sure the modules correctly exist if the flag is active
                if len(model._keep_in_fp32_modules) == 0 and len(model._keep_in_fp32_modules_strict) == 0:
                    self.skipTest(
                        reason=f"{model_class.__name__} has no _keep_in_fp32_modules nor _keep_in_fp32_modules_strict attribute defined"
                    )

                state_dict_names = {k for k, v in model.state_dict().items()}
                # Check that every module in the keep_in_fp32 list is part of the module graph
                if len(model._keep_in_fp32_modules) > 0:
                    non_existent = []
                    for module in model._keep_in_fp32_modules:
                        if not any(re.search(rf"(?:^|\.){module}(?:\.|$)", name) for name in state_dict_names):
                            non_existent.append(module)
                    self.assertTrue(
                        len(non_existent) == 0,
                        f"{non_existent} were specified in the `_keep_in_fp32_modules` list, but are not part of the modules in"
                        f" {model_class.__name__}",
                    )

                if len(model._keep_in_fp32_modules_strict) > 0:
                    non_existent = []
                    for module in model._keep_in_fp32_modules_strict:
                        if not any(re.search(rf"(?:^|\.){module}(?:\.|$)", name) for name in state_dict_names):
                            non_existent.append(module)
                    self.assertTrue(
                        len(non_existent) == 0,
                        f"{non_existent} were specified in the `_keep_in_fp32_modules_strict` list, but are not part of the "
                        f"modules in {model_class.__name__}",
                    )

    def test_keep_in_fp32_modules(self):
        """Test that the flag `_keep_in_fp32_modules` and `_keep_in_fp32_modules_strict`  is correctly respected."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))
                if len(model._keep_in_fp32_modules) == 0 and len(model._keep_in_fp32_modules_strict) == 0:
                    self.skipTest(
                        reason=f"{model_class.__name__} class has no _keep_in_fp32_modules or _keep_in_fp32_modules_strict attribute defined"
                    )

                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)

                    model = model_class.from_pretrained(tmpdirname, dtype=torch.float16)
                    self.assertFalse(
                        model._keep_in_fp32_modules & model._keep_in_fp32_modules_strict,
                        "We found a layer in both the `_keep_in_fp32_modules` and `_keep_in_fp32_modules_strict` lists. Please remove it from one of the two lists.",
                    )
                    # When reloading in fp16, keep_in_fp32_modules AND keep_in_fp32_modules_strict should be upcasted
                    all_fp32_modules = model._keep_in_fp32_modules | model._keep_in_fp32_modules_strict
                    for name, param in model.state_dict().items():
                        if any(re.search(rf"(?:^|\.){k}(?:\.|$)", name) for k in all_fp32_modules):
                            self.assertTrue(param.dtype == torch.float32, f"{name} not upcasted to fp32")
                        else:
                            self.assertTrue(param.dtype == torch.float16, f"{name} was upcasted but it should NOT be")

                    # When reloading in bf16, only keep_in_fp32_modules_strict should be upcasted
                    model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16)
                    for name, param in model.state_dict().items():
                        if any(re.search(rf"(?:^|\.){k}(?:\.|$)", name) for k in model._keep_in_fp32_modules_strict):
                            self.assertTrue(param.dtype == torch.float32, f"{name} not upcasted to fp32")
                        else:
                            self.assertTrue(param.dtype == torch.bfloat16, f"{name} was upcasted but it should NOT be")

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

                if getattr(model, "_tied_weights_keys", None):
                    keys_to_ignore.update(set(model._tied_weights_keys))
                with self.subTest(model=model_class.__name__):
                    self.assertTrue(
                        len(load_result.missing_keys) == 0 or set(load_result.missing_keys) == keys_to_ignore,
                        msg=f"Missing keys: {load_result.missing_keys}\nKeys to ignore: {keys_to_ignore}",
                    )
                    self.assertTrue(len(load_result.unexpected_keys) == 0)

    def test_load_contiguous_weights(self):
        """
        Checks whether the loaded weights are contiguous or not; inherently checking whether a conversion
        operation from `core_model_loading` may have affected the original weights.
        """
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()

            model = model_class(config)
            self.assertTrue(all(param.is_contiguous() for param in list(model.parameters())))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                model = model_class.from_pretrained(tmpdirname)
                self.assertTrue(all(param.is_contiguous() for param in list(model.parameters())))

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

            # Gradient checkpointing is implemented via GradientCheckpointingLayer, if none is present this is likely
            # an implementation issue. Note we exclude clvp for now since they are still not using
            # GradientCheckpointingLayer.
            if config.model_type not in ["clvp", "clvp_decoder"]:
                self.assertTrue([m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)])

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

    def test_enable_input_require_grads(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            if not hasattr(model, "get_input_embeddings"):
                continue
            try:
                model.enable_input_require_grads()
            except NotImplementedError as error:
                self.fail(f"enable_input_require_grads raised NotImplementedError for {model_class.__name__}: {error}")
            finally:
                model.disable_input_require_grads()

    def test_enable_input_require_grads_with_gradient_checkpointing(self):
        if not getattr(self.model_tester, "is_training", False):
            self.skipTest(reason="ModelTester is not configured to run training tests")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if hasattr(config, "use_cache"):
            config.use_cache = False

        has_verified_model = False

        for model_class in self.all_model_classes:
            if not getattr(model_class, "supports_gradient_checkpointing", False):
                continue

            model = model_class(copy.deepcopy(config))
            try:
                embeddings_module = model.get_input_embeddings()
            except NotImplementedError:
                continue
            if embeddings_module is None:
                continue

            embedding_param = getattr(embeddings_module, "weight", None)
            if embedding_param is None and isinstance(embeddings_module, (tuple, list)):
                for candidate in embeddings_module:
                    if hasattr(candidate, "weight"):
                        embedding_param = candidate.weight
                        break
            if embedding_param is None or not isinstance(embedding_param, torch.Tensor):
                continue

            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

            model.to(torch_device)
            model.train()

            set_seed(42)
            outputs = model(**inputs)
            loss_tensor = outputs.loss if getattr(outputs, "loss", None) is not None else outputs[0]
            if isinstance(loss_tensor, (tuple, list)):
                loss_tensor = loss_tensor[0]
            if loss_tensor is None or not isinstance(loss_tensor, torch.Tensor) or not loss_tensor.requires_grad:
                model.zero_grad(set_to_none=True)
                continue
            loss = loss_tensor.sum()
            loss.backward()

            baseline_grad = embedding_param.grad
            if (
                baseline_grad is None
                or baseline_grad.abs().sum().item() == 0
                or not torch.isfinite(baseline_grad).all()
            ):
                model.zero_grad(set_to_none=True)
                continue

            model.zero_grad(set_to_none=True)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

            set_seed(42)
            outputs = model(**inputs)
            loss_tensor = outputs.loss if getattr(outputs, "loss", None) is not None else outputs[0]
            if isinstance(loss_tensor, (tuple, list)):
                loss_tensor = loss_tensor[0]
            if loss_tensor is None or not isinstance(loss_tensor, torch.Tensor) or not loss_tensor.requires_grad:
                model.zero_grad(set_to_none=True)
                continue
            loss = loss_tensor.sum()
            loss.backward()

            grad_after_gc = embedding_param.grad
            self.assertIsNotNone(
                grad_after_gc,
                f"{model_class.__name__} should produce embedding gradients when gradient checkpointing is enabled. "
                "This typically means the model is not exposing its embeddings via `get_input_embeddings()` or "
                "a properly configured `_input_embed_layer` attribute.",
            )
            self.assertTrue(
                torch.isfinite(grad_after_gc).all(),
                f"{model_class.__name__} produced non-finite gradients with gradient checkpointing enabled.",
            )
            self.assertGreater(
                grad_after_gc.abs().sum().item(),
                0,
                f"{model_class.__name__} should keep non-zero embedding gradients with gradient checkpointing enabled.",
            )
            has_verified_model = True

        if not has_verified_model:
            self.skipTest(
                reason="No model with a differentiable loss was available to verify enable_input_require_grads with gradient checkpointing."
            )

    def test_can_init_all_missing_weights(self):
        """Ensure that all weights are correctly taken into account in `_init_weights`"""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # This is used to get the addition year of the model
        filename = inspect.getfile(config.__class__)
        # No easy way to get model addition date -> check copyright year on top of file
        with open(filename) as file:
            source_code = file.read()
        addition_year = 0  # if we cannot find it, set it to 0 (i.e. oldest)
        if match_object := re.search(r"^# Copyright (\d{4})", source_code, re.MULTILINE | re.IGNORECASE):
            addition_year = int(match_object.group(1))
        # For now, skip everything older than 2023 and "important models" (too many models to patch otherwise)
        # TODO: relax this as we patch more and more models
        if addition_year < 2023:
            self.skipTest(reason="Not a prioritized model for now.")

        for model_class in self.all_model_classes:
            # This context manager makes sure that we get the same results deterministically for random new weights
            with seeded_weight_init():
                # First, initialize the model from __init__ -> this ensure everything is correctly initialized, even if
                # _init_weights() does not take all weights into account correctly
                model_from_init = model_class(copy.deepcopy(config))
                # Here, passing an empty state dict will force all weights to be moved from meta to cpu, then be initialized
                # by _init_weights()
                model_from_pretrained = model_class.from_pretrained(None, config=copy.deepcopy(config), state_dict={})

            # First, check if any parameters/buffers are still on meta -> this is usually an issue with tied weights
            params_on_meta = []
            for k, v in model_from_pretrained.named_parameters():
                if v.device.type == "meta":
                    params_on_meta.append(k)
            for k, v in model_from_pretrained.named_buffers():
                if v.device.type == "meta":
                    params_on_meta.append(k)

            self.assertTrue(
                len(params_on_meta) == 0,
                f"The following keys are still on the meta device, it probably comes from an issue in the tied weights or buffers:\n{params_on_meta}",
            )

            from_pretrained_state_dict = model_from_pretrained.state_dict()
            from_init_state_dict = model_from_init.state_dict()
            self.assertEqual(
                sorted(from_pretrained_state_dict.keys()),
                sorted(from_init_state_dict.keys()),
                "The keys from each model should be the exact same",
            )

            # Everything must be exactly the same as we set the same seed for each init
            different_weights = set()
            for k1, v1 in from_init_state_dict.items():
                # In case using torch.nn.utils.parametrizations on a module, we should skip the resulting keys
                if re.search(r"\.parametrizations\..*?\.original[01]", k1):
                    continue
                v2 = from_pretrained_state_dict[k1]
                # Since we added the seed, they should be exactly the same (i.e. using allclose maybe be wrong due
                # to very low std in init function)
                if not (v1 == v2).all():
                    different_weights.add(k1)

            # Find the parent structure of the weights/buffers that are different for explicit error messages
            unique_bad_module_traceback = set()
            for weight in different_weights.copy():
                weight_name, immediate_parent_class, pretrained_parent_class = find_parent_traceback(
                    weight, model_from_init
                )

                # We cannot control timm model weights initialization, so skip in this case
                if (pretrained_parent_class == "TimmWrapperPreTrainedModel" and "timm_model." in weight) or (
                    pretrained_parent_class == "TimmBackbone" and "_backbone." in weight
                ):
                    different_weights.discard(weight)
                    continue

                # Add it to the traceback
                traceback = (
                    f"`{weight_name}` in module `{immediate_parent_class}` called from `{pretrained_parent_class}`\n"
                )
                unique_bad_module_traceback.add(traceback)

            self.assertTrue(
                len(different_weights) == 0,
                f"The following weights are not properly handled in `_init_weights()` (the model should be able to reinitialize "
                f"them correctly if the model is on meta device)::\n{unique_bad_module_traceback}",
            )

    def test_init_weights_can_init_buffers(self):
        """Ensure that all buffers (persistent and non-persistent) are correctly taken into account in `_init_weights`"""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Usually, buffers are not initialized randomly (it's kind of the point of having a Buffer instead of a Parameter...)
        # However, some PositionalEmbedding modules have a `positional_embedding` buffer, initialized randomly with normal
        # distribution and std `config.scale` - set it at 0 here to avoid randomness
        if hasattr(config, "scale"):
            config.scale = 0
        for sub_key in config.sub_configs:
            subconfig = getattr(config, sub_key)
            if hasattr(subconfig, "scale"):
                subconfig.scale = 0

        for model_class in self.all_model_classes:
            # First, initialize the model directly with `__init__`, with the context manager making sure that we do
            # not run `initialiaze_weights()`, i.e. buffers are the same as in the modules's `__init__` initial definition
            with skip_weight_init():
                model_from_init = model_class(copy.deepcopy(config))
            # Second, initialize the model fully on meta device, then move everything to cpu and run `init_weights`
            with torch.device("meta"):
                model_from_meta_init = model_class(copy.deepcopy(config))
            # move everything randomly to cpu
            model_from_meta_init.to_empty(device="cpu")
            # Now, run all the inits
            model_from_meta_init.init_weights()

            buffers_from_init = dict(model_from_init.named_buffers())
            buffers_from_meta_init = dict(model_from_meta_init.named_buffers())

            self.assertEqual(
                sorted(buffers_from_init.keys()),
                sorted(buffers_from_meta_init.keys()),
                "The name of the buffers from each model should be the exact same",
            )

            # Buffers are not random usually, so everything must match exactly
            different_buffers = set()
            for k1, v1 in buffers_from_init.items():
                v2 = buffers_from_meta_init[k1]
                if not (v1 == v2).all():
                    different_buffers.add(k1)

            # Find the parent structure of the buffers that are different for explicit error messages
            unique_bad_module_traceback = set()
            for buffer in different_buffers.copy():
                buf_name, immediate_parent_class, pretrained_parent_class = find_parent_traceback(
                    buffer, model_from_init
                )
                # Add it to the traceback
                traceback = (
                    f"`{buf_name}` in module `{immediate_parent_class}` called from `{pretrained_parent_class}`\n"
                )
                unique_bad_module_traceback.add(traceback)

            unique_bad_module_traceback = "".join(unique_bad_module_traceback)
            self.assertTrue(
                len(different_buffers) == 0,
                f"The following buffers are not properly handled in `_init_weights()` (the model should be able to reinitialize "
                f"them correctly if the model is on meta device):\n{unique_bad_module_traceback}",
            )

    def test_all_tensors_are_parameter_or_buffer(self) -> None:
        """Check that all tensors are registered as Parameter or Buffer, i.e. we don't have simple assignments such
        as `self.x = torch.tensor(...)` in a Module (as we cannot correctly recover from meta device if it's not
        registered as parameter/buffer). To test this, we initialize the model on a meta device and then move it onto
        the torch_device and perform a forward pass."""
        # Set seed to ensure stable model initialization - avoids numerical issues (NaN) with some models
        set_seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # Apparently this model cannot correctly create its inputs and has to use another function....
            if "modeling_perceiver.py" in inspect.getfile(model_class):
                _, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            # Initialize the model fully on meta device, then move everything to torch_device and run `init_weights`
            with torch.device("meta"):
                model = model_class(copy.deepcopy(config)).eval()
            # Move everything randomly to torch_device
            model.to_empty(device=torch_device)
            # Now, run all the inits
            model.init_weights()

            # Prepare inputs
            inputs = self._prepare_for_class(inputs_dict, model_class)
            # Try running a forward, to see if a tensor stayed on meta somewhere
            try:
                _ = model(**inputs)
            except (RuntimeError, NotImplementedError) as e:
                # Re-raise a more friendly exception (unfortunately, we cannot know which tensor it was...)
                if "Cannot copy out of meta tensor; no data!" in str(
                    e
                ) or "Tensor on device meta is not on the expected device cpu!" in str(e):
                    raise ValueError(
                        "A tensor is still on meta device. It means it was not properly registered as a Parameter or "
                        "Buffer.\nMost of the time, it should be added as a non-persistent buffer if you don't want to include "
                        "it in the model's state dict. It can also be a scalar that was added as a torch.Tensor, consider making it "
                        "a Python scalar in this case and use it as such in forward"
                    ) from e
                else:
                    raise e

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

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # Simply don't compare if both tensors only contain `nan` elements
            # See: https://github.com/huggingface/transformers/pull/40661
            if torch.all(torch.isnan(first)) and torch.all(torch.isnan(second)):
                return

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
                slice_ids = tuple(slice(0, index) for index in single_row_object.shape)
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

        # Set seed for deterministic test - ensures reproducible model initialization and inputs
        set_seed(42)
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

    def test_model_forward_default_config_values(
        self,
    ):
        """
        Tests that the model can run forward pass when config is intialized without common attributes.
        We expect that these attributes have a default value and will not cause errors. See #41541
        where the attributes were removed from `PreTrainedConfig` and moved to each model's config
        class.
        """
        common_config_properties = [
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
            "sep_token_id",
            "tie_word_embeddings",
        ]
        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()
        batch_size = self.model_tester.batch_size

        config_dict = config.to_diff_dict()
        for common_config_property in common_config_properties:
            config_dict.pop(common_config_property, None)
            for subconfig_key in config.sub_configs:
                subconfig = config_dict.get(subconfig_key, {})
                if subconfig:
                    subconfig.pop(common_config_property, None)
        config = config.__class__(**config_dict)

        # Set special tokens to `0` so it is guaranteed to be in vocab range
        for special_token in ["pad_token_id", "eos_token_id", "bos_token_id", "sep_token_id"]:
            if hasattr(config, special_token):
                setattr(config, special_token, 0)
            for subconfig_key in config.sub_configs:
                subconfig = getattr(config, subconfig_key, None)
                if subconfig and hasattr(subconfig, special_token):
                    setattr(subconfig, special_token, 0)

        for model_class in self.all_model_classes:
            if model_class.__name__ not in [
                *get_values(MODEL_MAPPING_NAMES),
            ]:
                continue

            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            single_batch_input = {}
            for key, value in batched_input.items():
                if isinstance(value, torch.Tensor) and value.shape[0] % batch_size == 0:
                    # e.g. musicgen has inputs of size (bs*codebooks). in most cases value.shape[0] == batch_size
                    single_batch_shape = value.shape[0] // batch_size
                    single_batch_input[key] = value[:single_batch_shape]
                else:
                    single_batch_input[key] = value

            with torch.no_grad():
                model(**single_batch_input)

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

                # make sure that test runs are consistent by disabling dropout
                #
                # Note: attention_probs_dropout_prob seem to influence classifier.bias in BertForMultipleChoice
                # (and other Bert derived models). Sometimes classifier.bias is None when
                # attention_probs_dropout_prob > 0. This might indicate a bug somewhere.
                if hasattr(config, "hidden_dropout_prob"):
                    config.hidden_dropout_prob = 0.0
                if hasattr(config, "attention_probs_dropout_prob"):
                    config.attention_probs_dropout_prob = 0.0

                inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                set_seed(42)
                model = model_class(config)
                model.to(torch_device)
                model.train()

                # unfreeze additional layers
                for p in model.parameters():
                    p.requires_grad_(True)

                # do a non-checkpointing run, so we can compare the set of non-zero gradients later. we skip None
                # grads here to collect a reference set of modules that have non-zero gradients (to filter layers like
                # MoE that drop out parts of the model).
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                set_seed(42)
                loss = model(**inputs).loss
                loss.backward()
                grad_expected_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
                non_zero_grads_normal = {n for n, p in grad_expected_params if p.grad.abs().sum() > 0}

                # reset all gradients to zero for the comparison with the gradient checkpointing run
                optimizer.zero_grad()

                # now enable gradient checkpointing and compare the gradients
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

                checkpointing_layer = next(m for m in model.modules() if isinstance(m, GradientCheckpointingLayer))

                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                with unittest.mock.patch.object(
                    checkpointing_layer, "forward", wraps=checkpointing_layer.forward
                ) as forward_mock:
                    set_seed(42)
                    loss = model(**inputs).loss
                    loss.backward()
                    optimizer.step()

                    # test that gradient checkpointing is active as it would call the gradient checkpointing layer's
                    # forward more than once.
                    self.assertGreater(forward_mock.call_count, 1)

                # check that all the parameters that had non-zero gradients before, have non-zero grads with gradient
                # checkpointing. divergence indicates a different forward-pass environment that needs special handling.
                non_zero_grads_gradcp = {n for n, p in grad_expected_params if p.grad.abs().sum() > 0}
                self.assertEqual(non_zero_grads_gradcp, non_zero_grads_normal)

                if self.test_all_params_have_gradient:
                    for k, v in model.named_parameters():
                        if v.requires_grad and v.grad is None:
                            if "expert" in k:
                                print(
                                    f"None for {k}, Probaby running a MOE, make sure grad is not NONE on EVERY layer. At LEAST 1 of the expert layer should have grads!"
                                )
                            else:
                                with self.subTest(f"{k}"):
                                    self.assertTrue(
                                        v.grad is not None, f"{k} in {model_class.__name__} has no gradient!"
                                    )

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

    def test_training_gradient_checkpointing(self):
        # Scenario - 1 default behaviour
        self.check_training_gradient_checkpointing()

    def test_training_gradient_checkpointing_use_reentrant_false(self):
        # Scenario - 2 with `use_reentrant=False` - this is the default value that is used in pytorch's
        # torch.utils.checkpoint.checkpoint
        self.check_training_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})

    def test_training_gradient_checkpointing_use_reentrant_true(self):
        # Scenario - 3 with `use_reentrant=True` (old default behaviour, not recommended)
        self.check_training_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": True})

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
                if (
                    self._is_composite and k == "vision_config"
                ):  # skip because it's not needed and causes errors e.g with Timm
                    continue
                if getattr(config, k) is not None:
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
                if getattr(config, k) is not None:
                    getattr(config, k).output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self._prepare_config_and_inputs_for_retain_grad_hidden_states_attentions()
        for k in config.sub_configs:
            if getattr(config, k) is not None:
                getattr(config, k).output_hidden_states = True

        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        for k in config.sub_configs:
            if (
                self._is_composite and k == "vision_config"
            ):  # skip because it's not needed and causes errors e.g with Timm
                continue
            if getattr(config, k) is not None:
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

    def _prepare_config_and_inputs_for_retain_grad_hidden_states_attentions(self):
        return self.model_tester.prepare_config_and_inputs_for_common()

    def test_feed_forward_chunking(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            set_seed(42)
            model = model_class(copy.deepcopy(original_config))
            model.to(torch_device)
            model.eval()

            hidden_states_no_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            set_seed(42)
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
        try:
            original_config.get_text_config().tie_word_embeddings = False
        except Exception as e:
            model_type = getattr(original_config, "model_type", "unknown")
            # Config may not have a text config
            print(f"Could not set text config's `tie_word_embeddings` for model type `{model_type}`: {e}")
        inputs_dict.pop("labels", None)

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            self.skipTest(reason="Model cannot untied embeddings")

        for model_class in self.all_model_classes:
            with self.subTest(model_class):
                config = copy.deepcopy(original_config)
                if is_deepspeed_zero3_enabled():
                    with deepspeed.zero.Init():
                        model = model_class(config)
                else:
                    model = model_class(config).to(torch_device)
                model.eval()

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

    def test_model_base_model_prefix(self):
        """
        Normally a generative model is a base model + lm_head on top. If this test
        fails for new model, probably the model has incorrect `base_model_prefix` or
        the you are re-defining base blocks for a generative model.
        There are some models which might not fit this assumption, if the model
        has a special architecture. Feel free to skip the test in that case with
        a reason in description.
        """
        for model_class in self.all_generative_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            self.assertTrue(model.base_model is not model)

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

    def test_can_use_safetensors(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            model_tied = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                try:
                    model_tied.save_pretrained(d)
                except Exception as e:
                    raise Exception(f"Class {model_class.__name__} cannot be saved using safetensors: {e}")
                with self.subTest(model_class):
                    model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                    # Checking the state dicts are correct
                    reloaded_state = model_reloaded.state_dict()
                    for k, v in model_tied.state_dict().items():
                        with self.subTest(f"{model_class.__name__}.{k}"):
                            torch.testing.assert_close(
                                v,
                                reloaded_state[k],
                                msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}.\n{v}\nvs\n{reloaded_state[k]}\n"
                                "This probably means that it was not set with the correct value when tying.",
                            )

                    # Checking the tensor sharing are correct on the new model (weights are properly tied in both cases)
                    ptrs = defaultdict(list)
                    for k, v in model_tied.state_dict().items():
                        ptrs[v.data_ptr()].append(k)

                    shared_ptrs = {k: v for k, v in ptrs.items() if len(v) > 1}

                    for shared_names in shared_ptrs.values():
                        reloaded_ptrs = {reloaded_state[k].data_ptr() for k in shared_names}
                        self.assertEqual(
                            len(reloaded_ptrs),
                            1,
                            f"The shared pointers are incorrect, found different pointers for keys {shared_names}. `__init__` and `from_pretrained` end up not tying the weights the same way.",
                        )

                    # Checking there was no complain of missing weights
                    self.assertEqual(
                        infos["missing_keys"],
                        set(),
                        "These keys were removed when serializing, and were not properly loaded by `from_pretrained`.",
                    )

    def test_load_save_without_tied_weights(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            config.tie_word_embeddings = False

            model = model_class(config)  # we init the model without tie
            # if this test fails later on, it means init tied the weights
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)
                with safe_open(f"{d}/model.safetensors", framework="pt") as f:
                    serialized_keys = f.keys()

                    model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                    # Checking the state dicts are correct

                    reloaded_state = model_reloaded.state_dict()
                    for k, v in model.state_dict().items():
                        with self.subTest(k):
                            torch.testing.assert_close(
                                v,
                                reloaded_state[k],
                                msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}. Key {k} was serialized: {k in serialized_keys}. If `False`, this means it was probably aliased and safetensors removed it. If `True` it means `_init_weights` overwrote that key",
                            )

                # Checking there was no complain of missing weights
                self.assertEqual(
                    infos["missing_keys"],
                    set(),
                    "Given that the loaded weights are the same, the issue is in `tie_weights`: it tied these keys and removed them from serialization. But because of tiying (hardcoded or not) the previous check is fine.\
                        This can happen if `save_pretrained` remove the targets and not the keys from serialiazation, or you hardcoded `self.xxx = yyy` thus forcing to always tie -> they are removed from serialization.",
                )

    def test_tied_weights_keys(self):
        original_config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            copied_config = copy.deepcopy(original_config)
            copied_config.get_text_config().tie_word_embeddings = True
            copied_config.tie_word_embeddings = True
            model_tied = model_class(copied_config)

            tied_weight_keys = _get_tied_weight_keys(model_tied)
            # If we don't find any tied weights keys, and by default we don't tie the embeddings, it's because the model
            # does not tie them or does not have embedding layer (non-text model)
            if len(tied_weight_keys) == 0 and not getattr(original_config, "tie_word_embeddings", None):
                continue

            ptrs = collections.defaultdict(list)
            for name, tensor in model_tied.state_dict().items():
                ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]

            # Detect we get a hit for each key
            for key in tied_weight_keys:
                is_tied_key = any(re.search(key, p) for group in tied_params for p in group)
                self.assertTrue(
                    is_tied_key,
                    f"{key} is not a tied weight key pattern for {model_class}: {is_tied_key}. With same params: {tied_params}",
                )

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
                # IMPORTANT Remove tied weights from extra missing: they are normally not warned as missing if their tied
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

        # move input tensors to accelerator O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            if model.config._experts_implementation == "grouped_mm":
                # DataParallel does not respect buffer alignment when replicating the model on
                # multiple GPUs, which can cause errors in grouped_mm experts implementation.
                model.set_experts_implementation("eager")

            # Wrap model in nn.DataParallel
            model = nn.DataParallel(model)
            torch.cuda.synchronize()  # otherwise the transfer might not be complete
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

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
            elif param_device == "mps":
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
            set_seed(42)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[0][""]
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Since we don't support saving with bins files anymore, but still support loading we use this context
                # to easily create the bins files and try to load them
                with force_serialization_as_bin_files():
                    model.cpu().save_pretrained(tmp_dir)

                with self.assertRaises(ValueError):
                    max_size = int(self.model_split_percents[0] * model_size)
                    max_memory = {0: max_size, "cpu": max_size}
                    # This errors out cause it's missing an offload folder
                    new_model = model_class.from_pretrained(
                        tmp_dir, device_map="auto", max_memory=max_memory, use_safetensors=False
                    )

                max_size = int(self.model_split_percents[1] * model_size)
                max_memory = {0: max_size, "cpu": max_size}
                new_model = model_class.from_pretrained(
                    tmp_dir, device_map="auto", max_memory=max_memory, offload_folder=tmp_dir, use_safetensors=False
                )

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                set_seed(42)
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
            set_seed(42)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[0][""]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)

                max_size = int(self.model_split_percents[1] * model_size)
                max_memory = {0: max_size, "cpu": max_size}

                # This doesn't error out as it's in safetensors and doesn't need an offload folder
                new_model = model_class.from_pretrained(
                    tmp_dir, device_map="auto", max_memory=max_memory, offload_folder=tmp_dir
                )

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                set_seed(42)
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

            set_seed(42)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[0][""]
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

                    set_seed(42)
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

            set_seed(42)
            base_output = model(**inputs_dict_class)

            model_size = compute_module_sizes(model)[0][""]
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

                    set_seed(42)
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
            self.skipTest(reason="test_mismatched_shapes is set to False")
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
                    self.assertIn("Reinit due to size mismatch", cl.out)
                    new_model.to(torch_device)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = AutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True
                        )
                    self.assertIn("Reinit due to size mismatch", cl.out)
                    input_ids = ids_tensor((2, 8), 10)
                    new_model_without_prefix.to(torch_device)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    def test_can_load_ignoring_mismatched_shapes(self):
        if not self.test_mismatched_shapes:
            self.skipTest(reason="test_mismatched_shapes is set to False")

        # Set seed for deterministic weight initialization
        set_seed(42)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        configs_no_init.num_labels = 3

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
                    self.assertIn("Reinit due to size mismatch", cl.out)

                    # Find the name of the module with the mismatched size
                    top_linear_modules = [
                        (name, module) for name, module in new_model.named_children() if isinstance(module, nn.Linear)
                    ]
                    # Some old model have the Linear classification layer inside a ClassificationHead module or nn.Sequential
                    if len(top_linear_modules) == 0:
                        # ClassificationHead case
                        if any(
                            module.__class__.__name__.endswith("ClassificationHead") for module in new_model.children()
                        ):
                            head_name, head_module = next(
                                (name, module)
                                for name, module in new_model.named_children()
                                if module.__class__.__name__.endswith("ClassificationHead")
                            )
                        # nn.Sequential case
                        elif any(isinstance(module, nn.Sequential) for module in new_model.children()):
                            head_name, head_module = next(
                                (name, module)
                                for name, module in new_model.named_children()
                                if isinstance(module, nn.Sequential)
                            )
                        # Unknown at this point -> skip (only xlm, perceiver, levit, flaubert, audio_spectrogram_transformer as of 23/09/2025)
                        else:
                            self.skipTest("Could not locate the classification Linear layer.")
                        top_linear_modules = [
                            (f"{head_name}.{name}", module)
                            for name, module in head_module.named_children()
                            if isinstance(module, nn.Linear)
                        ]
                    # Usually we have only 1, but swiftformer and deit have 2 Linear layers using `num_labels`
                    mismatched_modules = [name for name, module in top_linear_modules if module.out_features == 42]
                    old = dict(model.named_parameters())
                    new = dict(new_model.named_parameters())
                    assert not set(old.keys()) - set(new.keys())
                    for k1 in new.keys():
                        k2 = k1
                        v1 = old[k1]
                        v2 = new[k2]
                        # Each param except the mismatched ones must be exactly similar
                        if not any(k1.startswith(mismatched_module) for mismatched_module in mismatched_modules):
                            torch.testing.assert_close(v1, v2, msg=f"{k1} and  {k2} do not match: {v1} != {v2}")
                        # Check that the dims are indeed mismatched between old and new models
                        else:
                            # The old model should have `num_labels=3` (here it's the first dim of shape, as Linear layers
                            # are transposed)
                            self.assertEqual(v2.shape[0], 42)
                            # Make sure the mean of the new Linear layer is correctly centered around 0 (we cannot use
                            # a lower value for the check as some models hardcode a std of 0.02 instead of using the
                            # config, which we set very small with `config_no_init`)
                            self.assertLessEqual(v1.data.mean().item(), 1e-1, f"Issue with {k1}")

    def test_model_is_small(self):
        # Just a consistency check to make sure we are not running tests on 1M parameter models.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            num_params = model.num_parameters()
            assert num_params < 1000000, (
                f"{model_class} is too big for the common tests ({num_params})! It should have 1M max."
            )

    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ) -> None:
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

            # Set seed for deterministic test - ensures reproducible model initialization and inputs
            set_seed(42)
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

            # If we end up here, at least one model class was not skipped
            _has_run_at_least_one_model = True
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Save the model so we can reload with correct attention
                model.save_pretrained(tmpdirname)

                # Create first inputs without attention mask
                main_input = inputs_dict[model.main_input_name]
                # Only keep first batch sequence
                if isinstance(main_input, torch.Tensor):
                    main_input = main_input[:1]
                    # Fix the dtype
                    if torch.is_floating_point(main_input):
                        main_input = main_input.to(torch.bfloat16)
                first_inputs = {model.main_input_name: main_input, "output_hidden_states": True}
                # Some models have main input name which is different from input_ids, but require input_ids... e.g. BarkFine
                if model.main_input_name != "input_ids" and "input_ids" in inputs_dict:
                    first_inputs["input_ids"] = inputs_dict["input_ids"][:1]
                # If we have some pixel values, use them as well
                if model.main_input_name != "pixel_values" and "pixel_values" in inputs_dict:
                    # NOTE: this fixes qwen2_5_vl/omni because test break w/ pixel values
                    if "image_grid_thw" in inputs_dict:
                        continue
                    first_inputs["pixel_values"] = inputs_dict["pixel_values"][:1].to(torch.bfloat16)
                # Some VLMs require image_sizes alongside pixel_values, e.g. lighton_ocr, llava_onevision
                if "image_sizes" in inputs_dict:
                    first_inputs["image_sizes"] = inputs_dict["image_sizes"][:1]
                if model.config.is_encoder_decoder:
                    decoder_input_ids = inputs_dict.get("decoder_input_ids", first_inputs.get("input_ids"))
                    if decoder_input_ids is not None:
                        first_inputs["decoder_input_ids"] = decoder_input_ids[:1]

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

                # Use prepare for class to account for special attributes (e.g. in QnA models)
                first_inputs = self._prepare_for_class(first_inputs, model_class)
                first_inputs = {
                    k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in first_inputs.items()
                }
                second_inputs = self._prepare_for_class(second_inputs, model_class)
                second_inputs = {
                    k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in second_inputs.items()
                }

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
                torch.testing.assert_close(logits_1_eager, logits_1_fa, atol=atol, rtol=rtol)
                if padding_side == "left":
                    torch.testing.assert_close(logits_2_eager[1:], logits_2_fa[1:], atol=atol, rtol=rtol)
                else:
                    torch.testing.assert_close(logits_2_eager[:-1], logits_2_fa[:-1], atol=atol, rtol=rtol)

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
    @require_torch_accelerator
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence(self):
        self.flash_attn_inference_equivalence(attn_implementation="flash_attention_2", padding_side="left")

    @require_flash_attn
    @require_torch_accelerator
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
                if getattr(config, key) is not None:
                    attn_implementation_per_subconfig[key] = "eager"

            config._attn_implementation = attn_implementation_per_subconfig
            model = model_class(config)
            for key in config.sub_configs:
                if getattr(config, key) is not None:
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

            # Test that using `dict` attention implementation works with `from_pretrained`
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
        In contrast to the above test, this one checks if the "config._attn_implementation" is a dict after the model
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
                model_sdpa = model_sdpa.base_model

                vision_model_names = {"visual", "image_tower", "vision_tower", "vision_model"}
                language_model_names = {"language_model", "model", "text_model"}
                vision_model_name = [name for name in vision_model_names if hasattr(model_sdpa, name)]
                vision_model_name = vision_model_name[0] if len(vision_model_name) > 0 else None
                language_model_name = [name for name in language_model_names if hasattr(model_sdpa, name)]
                language_model_name = language_model_name[0] if len(language_model_name) > 0 else None
                if language_model_name is None or vision_model_name is None:
                    self.skipTest(
                        reason="Model does not have both vision and language sub-models, cannot test composite SDPA dispatch"
                    )
                vision_model_sdpa = getattr(model_sdpa, vision_model_name)
                language_model_sdpa = getattr(model_sdpa, language_model_name)
                text_attn = "sdpa" if language_model_sdpa._supports_sdpa else "eager"
                vision_attn = "sdpa" if vision_model_sdpa._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(language_model_sdpa.config._attn_implementation == text_attn)
                self.assertTrue(vision_model_sdpa.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.base_model
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

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        _test_eager_matches_batched_and_grouped_inference(self, name, dtype)

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
            if config.model_type == "paligemma":
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
            if config.model_type == "sam":
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
            if config.model_type == "dbrx":
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
                model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16, attn_implementation="sdpa")
                model.to(torch_device)

                # For PyTorch 2.1 - 2.3.0 set `dynamic=True`. In the future setting `dynamic=None` and using `torch._dynamo.mark_dynamic()`
                # on input tensors will be required. `mark_dynamic` currently raises inconsistent shape errors.
                model = torch.compile(model, dynamic=True)

                inputs_dict.pop("attention_mask", None)
                inputs_dict.pop("decoder_attention_mask", None)
                for name, inp in inputs_dict.items():
                    if isinstance(inp, torch.Tensor) and inp.dtype in [torch.float32, torch.float16]:
                        inputs_dict[name] = inp.to(torch.bfloat16)

                # use no_grad to save some memory
                with torch.no_grad():
                    _ = model(**inputs_dict)

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

        def _expected_attn_implementations(attention_implementation: str) -> set[str]:
            # Allow kernels fallbacks for flash attention tests.
            requested = attention_implementation
            base = requested.removeprefix("paged|")
            prefix = "paged|" if requested.startswith("paged|") else ""

            expected = {requested}
            if base in FLASH_ATTN_KERNEL_FALLBACK:
                expected.add(f"{prefix}{FLASH_ATTN_KERNEL_FALLBACK[base]}")
            return expected

        expected_attn_implementations = _expected_attn_implementations(attn_implementation)

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
                        if isinstance(getattr(model_fa.config, key), PreTrainedConfig):
                            sub_config = getattr(model_fa.config, key)
                            self.assertIn(sub_config._attn_implementation, expected_attn_implementations)

                    has_fa = False
                    for name, submodule in model_fa.named_modules():
                        class_name = submodule.__class__.__name__
                        if (
                            "Attention" in class_name
                            and getattr(submodule, "config", None)
                            and submodule.config._attn_implementation in expected_attn_implementations
                        ):
                            has_fa = True
                            break
                    if not has_fa:
                        raise ValueError(f"The {attn_implementation} model should have {attn_implementation} layers")

    @require_flash_attn
    @require_torch_accelerator
    @mark.flash_attn_test
    def test_flash_attn_2_can_dispatch_composite_models(self):
        self.flash_attn_can_dispatch_composite_models(attn_implementation="flash_attention_2")

    @require_flash_attn_3
    @require_torch_gpu
    @mark.flash_attn_3_test
    def test_flash_attn_3_can_dispatch_composite_models(self):
        self.flash_attn_can_dispatch_composite_models(attn_implementation="flash_attention_3")

    @require_flash_attn
    @require_torch_accelerator
    @require_bitsandbytes
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_fp32_ln(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_generative_model_classes:  # TODO: this test should run on all classes instead
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            if not all(
                submodel._supports_flash_attn for submodel in model.modules() if isinstance(submodel, PreTrainedModel)
            ):
                self.skipTest(reason="At least some parts of this model do not support flash attention")

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
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
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
    @require_torch_accelerator
    @mark.flash_attn_test
    @pytest.mark.torch_compile_test
    @slow
    def test_flash_attn_2_can_compile_with_attention_mask_None_without_graph_break(self):
        if not hasattr(self, "_torch_compile_train_cls"):
            self.skipTest(f"{self.__class__.__name__} doesn't have the attribute `_torch_compile_train_cls`.")

        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_device == "xpu":
            self.skipTest("XPU FA2 currently does not support backward.")

        torch.compiler.reset()
        dtype = torch.float16

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cls = self._torch_compile_train_cls  # e.g. LlamaFroCausalLM
        if not cls._supports_flash_attn:
            self.skipTest(f"{cls.__name__} does not support Flash Attention 2")

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

    def flash_attn_from_config(self, attn_implementation: str, test_fwd_in_train: bool = True):
        r"""
        Tests if the model can be loaded with `attn_implementation` from the config and if the
        weights are not randomly initialized.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_generative_model_classes:  # TODO: this test should run on all classes instead
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support {attn_implementation}")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)  # let's construct it here to see if any submodels can't support flash attn
            if not all(
                submodel._supports_flash_attn for submodel in model.modules() if isinstance(submodel, PreTrainedModel)
            ):
                self.skipTest(reason=f"At least some parts of this model do not support {attn_implementation}")

            # TODO: to change it in the future with other relevant auto classes
            fa_model = model_class._from_config(
                config, attn_implementation=attn_implementation, dtype=torch.bfloat16
            ).to(torch_device)

            # By default, we perform the forward pass in train mode, because it's more sctrict than eval mode. If the
            # forward pass is successful in train mode, it will also be successful in eval mode. But since some models
            # (eg. gemma3) need different inputs in train mode we have the option to test the forward pass in eval mode.
            if test_fwd_in_train:
                fa_model = fa_model.train()
            else:
                fa_model = fa_model.eval()

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
    @require_torch_accelerator
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
            config_dict.pop("layer_types", None)
            config_dict.pop("rope_parameters", None)
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
            config_dict.pop("layer_types", None)
            config_dict.pop("rope_parameters", None)
            new_config = config.__class__(**config_dict)
            # We need to set eager as otherwise `output_attentions` is not supported
            model = model_class._from_config(new_config, attn_implementation="eager").to(torch_device)
            model.eval()
            attentions_not_sliding = model(**inputs, output_attentions=True).attentions
            for layer_attention in attentions_not_sliding:
                self.assertFalse((layer_attention[:, :, ~sliding_mask] == 0).all().item())

    @slow
    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_torch_compile_for_training(self):
        if getattr(self, "_torch_compile_train_cls", None) is None:
            self.skipTest(f"{self.__class__.__name__} doesn't have the attribute `_torch_compile_train_cls`.")

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        cls = self._torch_compile_train_cls
        attn_implementation = getattr(self, "_torch_compile_train_attn_implementation", None)
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation

        model = cls(config).to(device=torch_device)

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

    @slow
    @pytest.mark.torch_export_test
    def test_torch_export(self, atol=1e-4, rtol=1e-4):
        """
        Test if model can be exported with torch.export.export()

        Args:
            atol (`float`, *optional*, defaults to 1e-4): absolute tolerance for output comparison
            rtol (`float`, *optional*, defaults to 1e-4): relative tolerance for output comparison
        """

        if not self.test_torch_exportable:
            self.skipTest(reason="Model architecture is not torch exportable")

        with open(inspect.getfile(self.all_model_classes[0]), "r") as f:
            source_code = f.read()
            # Skip model if it uses a chunked attention implementation which is not torch exportable
            if "for q, k, v in zip(*splits)" in source_code:
                self.skipTest(reason="Model architecture uses chunked attention which is not torch exportable")
            # Skip MoEs that don't support batched_mm experts implementation
            if "for expert" in source_code and "use_experts_implementation" not in source_code:
                self.skipTest(reason="Model architecture uses eager MoE implementation which is not torch exportable")
            # Skip models that use get_rope_index which is not torch exportable
            if "get_rope_index" in source_code:
                self.skipTest(reason="Model architecture uses get_rope_index which is not torch exportable")

        def _is_pure_python_object(obj) -> bool:
            if isinstance(obj, (int, float, bool, str)) or obj is None:
                return True
            elif isinstance(obj, (list, tuple, set)):
                return all(_is_pure_python_object(o) for o in obj)
            elif isinstance(obj, dict):
                return all(_is_pure_python_object(o) for o in obj.values())
            else:
                return False

        def _get_leaf_tensors(obj) -> dict[str, torch.Tensor]:
            if _is_pure_python_object(obj):
                return {}
            elif isinstance(obj, torch.Tensor):
                return {"": obj}
            elif isinstance(obj, (list, tuple, set)):
                return _get_leaf_tensors(dict(enumerate(obj)))
            elif isinstance(obj, dict):
                leaf_tensors = {}
                for key, value in obj.items():
                    for sub_key, tensor in _get_leaf_tensors(value).items():
                        full_key = f"{key}.{sub_key}" if sub_key else str(key)
                        leaf_tensors[full_key] = tensor
                return leaf_tensors
            else:
                raise ValueError(f"Unexpected object type: {type(obj)}")

        def _prepare_for_export(model, inputs_dict):
            # we don't test outputing a cache class for now
            inputs_dict.pop("use_cache", None)
            # we don't test loss computation for now
            inputs_dict.pop("return_loss", None)
            # we don't test loss computation for now
            inputs_dict.pop("future_values", None)

            # set experts implementation to batched_mm for export
            if model._can_set_experts_implementation():
                model.set_experts_implementation("batched_mm")

            # set attention implementation to sdpa for export
            if model._can_set_attn_implementation() and model.config.model_type != "videomae":
                try:
                    model.set_attn_implementation("sdpa")
                except Exception as e:
                    print(
                        f"Could not set attention implementation to sdpa for {model} of type {model.config.model_type} : {e}"
                    )

            for module in model.modules():
                if hasattr(module, "config"):
                    # disable cache usage for every submodel
                    if hasattr(module.config, "use_cache"):
                        module.config.use_cache = False
                    # disable returning loss for every submodel
                    if hasattr(module.config, "return_loss"):
                        module.config.return_loss = False
                    # disable mamba kernels for every submodel (mamba, jamba)
                    if hasattr(module.config, "use_mamba_kernels"):
                        module.config.use_mamba_kernels = False
                # disable classifier cast for nllb-moe
                if hasattr(module, "_cast_classifier"):
                    module._cast_classifier = lambda *args, **kwargs: None
                # disable mamba mask update for ssms
                if hasattr(module, "_update_mamba_mask"):
                    module._update_mamba_mask = lambda attention_mask, *args, **kwargs: attention_mask
                if hasattr(module, "_update_linear_attn_mask"):
                    module._update_linear_attn_mask = lambda attention_mask, *args, **kwargs: attention_mask

            return model, inputs_dict

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
                    config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
                else:
                    config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                set_config_for_less_flaky_test(config)
                model = model_class(config).eval().to(torch_device)
                set_model_for_less_flaky_test(model)

                # Prepare model and inputs for export
                model, inputs_dict = _prepare_for_export(model, inputs_dict)

                with torch.no_grad():
                    # Running the eager inference before the export to catch model/inputs comatibility issues, also sometimes after
                    # the export, the model used for export will return FakeTensors instead of real ones (torch compiler issue).
                    # This happens on cuda for example with (codegen, clvp, esm, gptj, levit, wav2vec2_bert and wav2vec2_conformer)
                    set_seed(1234)
                    eager_outputs = model(**copy.deepcopy(inputs_dict))
                    eager_outputs = _get_leaf_tensors(eager_outputs)
                    self.assertTrue(eager_outputs, "Eager outputs is empty.")

                try:
                    exported_program = torch.export.export(model, args=(), kwargs=copy.deepcopy(inputs_dict))
                except Exception as e:
                    raise e

                with torch.no_grad():
                    set_seed(1234)
                    exported_outputs = exported_program.module()(**copy.deepcopy(inputs_dict))
                    exported_outputs = _get_leaf_tensors(exported_outputs)
                    self.assertTrue(exported_outputs, "Exported outputs is empty.")

                # Check outputs closeness:
                torch.testing.assert_close(exported_outputs, eager_outputs, atol=atol, rtol=rtol)

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
                # For some models, num_attention_heads is a list of ints: we take the max to maximize the multiplier
                num_attn_heads = getattr(config, "num_attention_heads")
                num_attn_heads = num_attn_heads if isinstance(num_attn_heads, int) else max(num_attn_heads)
                head_dim = head_dim if head_dim is not None else config.hidden_size // num_attn_heads
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
                getattr(config, "rope_parameters", None) is not None
                and len(config.rope_parameters.get("mrope_section", [])) > 0
            ):
                scaling_factor = max(requested_dim // (sum(config.rope_parameters["mrope_section"]) * 2), 1)
                config.rope_parameters["mrope_section"] = [
                    section * scaling_factor for section in config.rope_parameters["mrope_section"]
                ]

        # Update config values
        update_config_headdim(config, requested_dim)
        for key in config.sub_configs:
            if getattr(config, key) is not None:
                sub_config = getattr(config, key)
                update_config_headdim(sub_config, requested_dim)

        return config

    @require_torch_accelerator
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

            if config.is_encoder_decoder:
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

    def test_cannot_load_with_meta_device_context_manager(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # Need to deepcopy here as it is modified in-place in save_pretrained (it sets sdpa for default attn, which
            # is not supported for e.g. dpt_hybrid)
            model = model_class(copy.deepcopy(config))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                with torch.device("meta"):
                    with self.assertRaisesRegex(
                        RuntimeError, "You are using `from_pretrained` with a meta device context manager"
                    ):
                        _ = model_class.from_pretrained(tmpdirname)

    def test_config_attn_implementation_setter(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        def check_attn_implementation_setter(config: PreTrainedConfig, attn_implementation: str):
            if not config._attn_implementation == attn_implementation:
                raise ValueError(
                    f"Unexpected attn_implementation for config {config.__class__.__name__}: "
                    f"{config._attn_implementation} != {attn_implementation}"
                )
            for attribute_value in config.__dict__.values():
                if isinstance(attribute_value, PreTrainedConfig):
                    check_attn_implementation_setter(attribute_value, attn_implementation)

        # Check that attention implementation can be passed with init args
        config_dict = config.to_diff_dict()
        config_dict.pop("_attn_implementation_internal", None)
        config_dict.pop("_attn_implementation", None)
        config_dict["attn_implementation"] = "eager"
        config = type(config)(**config_dict)
        check_attn_implementation_setter(config, "eager")

        # Check that attention implementation can be set to different value
        config._attn_implementation = "sdpa"
        check_attn_implementation_setter(config, "sdpa")

        config._attn_implementation = "eager"
        check_attn_implementation_setter(config, "eager")

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
                        if (
                            subconfig_from_model_config is not None
                            and subconfig_from_model_config.__class__ == subconfig_from_model_internal.__class__
                        ):
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
                if getattr(config, subconfig_key) is not None:
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
                if getattr(config, subconfig_key) is not None:
                    self.assertTrue(getattr(model.config, subconfig_key)._attn_implementation == "sdpa")

            # Check we cannot set it to random values, and it raises an error
            with self.assertRaisesRegex(ValueError, 'Specified `attn_implementation="foo"` is not supported'):
                model.set_attn_implementation("foo")

            # Should still be sdpa everywhere
            self.assertTrue(model.config._attn_implementation == "sdpa")
            for subconfig_key in model.config.sub_configs:
                if getattr(config, subconfig_key) is not None:
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
                if getattr(config, subconfig_key) is not None:
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
                if getattr(config, subconfig_key) is not None:
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
                        with self.subTest(f"{dtype} for {model_class.__name__}.{k1}"):
                            self.assertEqual(k1, k2)
                            self.assertEqual(v1.dtype, v2.dtype)
                            torch.testing.assert_close(v1, v2, msg=f"{k1} and  {k2} do not match: {v1} != {v2}")

    def test_tp_plan_matches_params(self):
        """Make sure that each entry of the tp plan matches at least one param (this avoid typos and/or edge cases
        with regexes)"""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        # If none of the config and subconfigs have a tp_plan, then skip (otherwise we should make sure to respect the plan)
        if config.base_model_tp_plan is None and all(
            getattr(getattr(config, key), "base_model_tp_plan", None) is None for key in config.sub_configs
        ):
            self.skipTest("Model does not have a TP plan.")

        # Some MoE models alternate between a classic MLP and a MoE layer, in which case we want to have each one
        # in order to test the whole tp plan
        config_to_set = config.get_text_config()
        config_to_set.first_k_dense_replace = 1  # means that the first layer (idx 0) will be MLP, then MoE
        config_to_set.moe_layer_start_index = 1  # same as above but for Ernie 4.5...
        config_to_set.mlp_only_layers = [0]  # same but for qwens

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            param_names = {name for name, _ in model.named_parameters()} | {name for name, _ in model.named_buffers()}
            module_names = {name for name, _ in model.named_modules()}
            tp_plan = model.tp_plan
            # Make sure the plan is not empty
            self.assertTrue(
                len(tp_plan) > 0,
                f"No TP-plan found for class {model_class.__name__} even though the associated config has one",
            )
            pattern_usage = {}
            for pattern in tp_plan:
                # Check if this given pattern matches any param or module (the value attributed to the pattern does not matter)
                pattern_usage[pattern] = any(
                    _get_parameter_tp_plan(param, {pattern: ""}, is_weight=True) is not None for param in param_names
                ) or any(
                    _get_parameter_tp_plan(module, {pattern: ""}, is_weight=False) is not None
                    for module in module_names
                )

            unused_entries = {k for k, v in pattern_usage.items() if not v}
            self.assertTrue(
                len(unused_entries) == 0, f"The following entries of the TP-plan are not valid: {unused_entries}"
            )

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        """Make sure we can load and save correctly the models having any weight renaming mapping or weight conversion
        mapping.
        Note that this test would be better if we could start from the serialized keys, and check that the model
        keys correspond to the weight converions. However, when instantiating a model, it already has the "target"
        keys (or modified keys after mapping) of the conversion mapping, so we have to do it the other way, i.e.
        reverse the conversion and then check that those converted keys match correctly the conversions.

        However, all the checks performed here should ensure everything is going as it should.

        Args:
            check_keys_were_modified (`bool`, *optional*, defaults to `True`):
                Whether to expect keys being modified or not. In some cases, models do not change keys but
                their weights, e.g. via transpose, memory alignment, etc.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        #  Some MoE models alternate between a classic MLP and a MoE layer, in which case we want to have at
        # lest one MoE layer here to check the mapping
        config_to_set = config.get_text_config(decoder=True)
        config_to_set.first_k_dense_replace = 1  # means that the first layer (idx 0) will be MLP, then MoE
        config_to_set.moe_layer_start_index = 1  # same as above but for Ernie 4.5...
        config_to_set.mlp_only_layers = [0]  # same but for qwens
        config_to_set.num_dense_layers = 1  # lfm2_moe

        for model_class in self.all_model_classes:
            # Each individual model is a subtest
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))
                # Skip if no conversions
                conversions = get_model_conversion_mapping(model, add_legacy=False)
                if len(conversions) == 0:
                    self.skipTest("No conversion found for this model")

                # Find the model keys, so the targets according to the conversions
                model_keys = list(model.state_dict().keys())

                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Serialize with reverse mapping
                    model.save_pretrained(tmpdirname)
                    state_dict = load_file(os.path.join(tmpdirname, "model.safetensors"))
                    # Get all the serialized keys that we just saved according to the reverse mapping
                    serialized_keys = list(state_dict.keys())

                if check_keys_were_modified:
                    # They should be different, otherwise we did not perform any mapping
                    self.assertNotEqual(sorted(serialized_keys), sorted(model_keys), "No key mapping was performed!")

                # Check that for each conversion entry, we at least map to one key
                for conversion in conversions:
                    for source_pattern in conversion.source_patterns:
                        # Sometimes the mappings specify keys that are tied, so absent from the saved state dict
                        if isinstance(conversion, WeightRenaming):
                            # We need to revert the target pattern to make it compatible with regex search
                            target_pattern_reversed = conversion.target_patterns[0]
                            captured_group = process_target_pattern(source_pattern)[1]
                            if captured_group:
                                target_pattern_reversed = target_pattern_reversed.replace(r"\1", captured_group)
                            if any(re.search(target_pattern_reversed, k) for k in model.all_tied_weights_keys.keys()):
                                continue
                        num_matches = sum(re.search(source_pattern, key) is not None for key in serialized_keys)
                        self.assertTrue(
                            num_matches > 0,
                            f"`{source_pattern}` in `{conversion}` did not match any of the source keys. "
                            "This indicates whether that the pattern is not properly written, ot that it could not be reversed correctly",
                        )

                # If everything is still good at this point, let's test that we perform the same operations both when
                # reverting ops from `from_pretrained` and from `__init__`
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # The model was instantiated from __init__ before being saved
                    model.save_pretrained(tmpdirname)
                    state_dict_saved_from_init = load_file(os.path.join(tmpdirname, "model.safetensors"))

                    # Now reload it
                    model_reloaded = model_class.from_pretrained(tmpdirname)

                    # Make sure both loaded state_dict are identical
                    self.assertTrue(compare_state_dicts(model_reloaded.state_dict(), model.state_dict()))

                    # The model was instantiated from `from_pretrained` before being saved
                    model_reloaded.save_pretrained(tmpdirname)
                    state_dict_saved_from_pretrained = load_file(os.path.join(tmpdirname, "model.safetensors"))

                    # Make sure both saved state_dict are identical
                    self.assertTrue(compare_state_dicts(state_dict_saved_from_init, state_dict_saved_from_pretrained))

    def test_can_load_from_already_mapped_keys(self):
        """Test that we can correctly reload a model if we chose `save_original_format=False` in `save_pretrained`,
        i.e. we do not reapply weight conversions when reloading if it was saved correctly already.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # Each individual model is a subtest
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))

                # Skip if no conversions
                conversions = get_model_conversion_mapping(model, add_legacy=False)
                if len(conversions) == 0:
                    self.skipTest("No conversion found for this model")

                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Serialize without reverting the mapping
                    model.save_pretrained(tmpdirname, save_original_format=False)
                    model_reloaded = model_class.from_pretrained(tmpdirname)
                    # Make sure both saved state_dict are identical
                    self.assertTrue(compare_state_dicts(model.state_dict(), model_reloaded.state_dict()))

    def _text_features_prepare_config_and_inputs(self):
        """
        Helper method to extract only text-related inputs from the full set of inputs, for testing `get_text_features`.

        Specifically, it tests both the model_tester and its text_model_tester (if any),
        and filters for "input_ids", "token_type_ids", and "attention_mask" keys.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if hasattr(self.model_tester, "text_model_tester"):
            _, inputs_dict = self.model_tester.text_model_tester.prepare_config_and_inputs_for_common()
        else:
            inputs_dict = {
                key: value
                for key, value in inputs_dict.items()
                if key in ["input_ids", "token_type_ids", "attention_mask"]
            }
        return config, inputs_dict

    def _image_features_prepare_config_and_inputs(self):
        """
        Helper method to extract only image-related inputs from the full set of inputs, for testing `get_image_features`.

        Specifically, it tests both the model_tester and its vision_model_tester (if any),
        and filters for keys related to images. It excludes video-related keys, but allows
        "spatial_shapes" and "qformer_input_ids" keys as required by some architectures.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if hasattr(self.model_tester, "vision_model_tester"):
            _, inputs_dict = self.model_tester.vision_model_tester.prepare_config_and_inputs_for_common()
        else:
            inputs_dict = {
                key: value
                for key, value in inputs_dict.items()
                if ("pixel" in key or "image" in key)
                and "video" not in key
                or key in ["spatial_shapes", "qformer_input_ids"]
            }
        return config, inputs_dict

    def _audio_features_prepare_config_and_inputs(self):
        """
        Helper method to extract only audio-related inputs from the full set of inputs, for testing `get_audio_features`.

        Specifically, it tests both the model_tester and its audio_model_tester (if any),
        and filters for keys related to audio.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if hasattr(self.model_tester, "audio_model_tester"):
            _, inputs_dict = self.model_tester.audio_model_tester.prepare_config_and_inputs_for_common()
        else:
            inputs_dict = {
                key: value
                for key, value in inputs_dict.items()
                if "audio" in key
                or "input_values" in key
                or "input_features" in key
                or key in ["padding_mask", "is_longer", "feature_attention_mask"]
            }
        return config, inputs_dict

    def _video_features_prepare_config_and_inputs(self):
        """
        Helper method to extract only video-related inputs from the full set of inputs, for testing `get_video_features`.

        Specifically, it tests both the model_tester and its video_model_tester (if any),
        and filters for keys related to videos. It also handles key renaming for video inputs
        if there is no dedicated video_model_tester.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if hasattr(self.model_tester, "video_model_tester"):
            _, inputs_dict = self.model_tester.video_model_tester.prepare_config_and_inputs_for_common()
        else:
            key_mappings = {
                "pixel_values": "pixel_values_videos",
                "image_grid_thw": "video_grid_thw",
                "image_merge_sizes": "video_merge_sizes",
            }

            for src_key, dst_key in key_mappings.items():
                if src_key in inputs_dict and dst_key not in inputs_dict:
                    inputs_dict[dst_key] = inputs_dict.pop(src_key)

            allowed_non_video_keys = {"vision_feature_layer", "vision_feature_select_strategy", "cu_seqlens"}
            inputs_dict = {
                key: value for key, value in inputs_dict.items() if "video" in key or key in allowed_non_video_keys
            }
        return config, inputs_dict

    def _text_features_get_expected_num_attentions(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester

        if hasattr(model_tester, "text_model_tester"):
            return self._text_features_get_expected_num_attentions(model_tester.text_model_tester)
        if hasattr(model_tester, "expected_num_hidden_layers"):
            return model_tester.expected_num_hidden_layers - 1
        if hasattr(model_tester, "num_hidden_layers"):
            return model_tester.num_hidden_layers
        raise ValueError("Cannot determine the expected number of layers for text features")

    def _text_features_get_expected_num_hidden_states(self, model_tester=None):
        return self._text_features_get_expected_num_attentions(model_tester) + 1

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester
        if hasattr(model_tester, "vision_model_tester"):
            return self._image_features_get_expected_num_attentions(model_tester.vision_model_tester)
        elif (
            hasattr(model_tester, "vision_config")
            and isinstance(model_tester.vision_config, dict)
            and "num_hidden_layers" in model_tester.vision_config
        ):
            return model_tester.vision_config["num_hidden_layers"]

        if hasattr(model_tester, "expected_num_hidden_layers"):
            return model_tester.expected_num_hidden_layers - 1
        elif hasattr(model_tester, "num_hidden_layers"):
            return model_tester.num_hidden_layers
        raise ValueError("Cannot determine the expected number of layers for image features")

    def _image_features_get_expected_num_hidden_states(self, model_tester=None):
        return self._image_features_get_expected_num_attentions(model_tester) + 1

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester

        if hasattr(model_tester, "audio_model_tester"):
            return self._audio_features_get_expected_num_attentions(model_tester.audio_model_tester)
        elif (
            hasattr(model_tester, "audio_config")
            and isinstance(model_tester.audio_config, dict)
            and "num_hidden_layers" in model_tester.audio_config
        ):
            return model_tester.audio_config["num_hidden_layers"]

        if hasattr(model_tester, "expected_num_hidden_layers"):
            return model_tester.expected_num_hidden_layers - 1
        elif hasattr(model_tester, "num_hidden_layers"):
            return model_tester.num_hidden_layers
        raise ValueError("Cannot determine the expected number of layers for audio features")

    def _audio_features_get_expected_num_hidden_states(self, model_tester=None):
        return self._audio_features_get_expected_num_attentions(model_tester) + 1

    def _video_features_get_expected_num_attentions(self, model_tester=None):
        if model_tester is None:
            model_tester = self.model_tester

        if hasattr(model_tester, "video_model_tester"):
            return self._video_features_get_expected_num_attentions(model_tester.video_model_tester)
        if hasattr(model_tester, "vision_model_tester"):
            return self._video_features_get_expected_num_attentions(model_tester.vision_model_tester)
        elif (
            hasattr(model_tester, "video_config")
            and isinstance(model_tester.video_config, dict)
            and "num_hidden_layers" in model_tester.video_config
        ):
            return model_tester.video_config["num_hidden_layers"]

        if hasattr(model_tester, "expected_num_hidden_layers"):
            return model_tester.expected_num_hidden_layers - 1
        elif hasattr(model_tester, "num_hidden_layers"):
            return model_tester.num_hidden_layers
        raise ValueError("Cannot determine the expected number of layers for video features")

    def _video_features_get_expected_num_hidden_states(self, model_tester=None):
        return self._video_features_get_expected_num_attentions(model_tester) + 1

    @parameterized.expand([True, False, None])
    def test_get_text_features_output(self, return_dict: bool | None):
        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_text_features"):
                continue

            config, inputs_dict = self._text_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            set_seed(42)
            with torch.no_grad():
                outputs = model.get_text_features(**inputs_dict)

            if return_dict in (True, None):
                self.assertTrue(isinstance(outputs, ModelOutput), "get_text_features() must return a BaseModelOutput")
                self.assertTrue(
                    hasattr(outputs, "last_hidden_state"),
                    "get_text_features() must return a BaseModelOutput with last_hidden_state",
                )
                self.assertTrue(
                    hasattr(outputs, "pooler_output"),
                    "get_text_features() must return a BaseModelOutput with pooler_output",
                )
                self.assertTrue(
                    hasattr(outputs, "hidden_states"),
                    "get_text_features() must return a BaseModelOutput with hidden_states",
                )
                if self.has_attentions:
                    self.assertTrue(
                        hasattr(outputs, "attentions"),
                        "get_text_features() must return a BaseModelOutput with attentions",
                    )

                # Test against (batch_size, seq_len, hidden_size)
                last_hidden_state = outputs.last_hidden_state
                expected_hidden_size = config.text_config.hidden_size
                expected_shape = (
                    inputs_dict["input_ids"].shape[0],
                    inputs_dict["input_ids"].shape[1],
                    expected_hidden_size,
                )
                self.assertEqual(last_hidden_state.shape, expected_shape, "last_hidden_state shape mismatch")

            else:
                self.assertIsInstance(outputs, tuple, "get_text_features() must return a tuple if return_dict=False")

    def test_get_text_features_hidden_states(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_text_features(**inputs_dict)
            # hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            hidden_states = outputs.hidden_states
            expected_num_hidden_states = self._text_features_get_expected_num_hidden_states()
            self.assertIsNotNone(hidden_states, "hidden_states should not be None")
            self.assertEqual(len(hidden_states), expected_num_hidden_states, "Number of hidden states layers mismatch")

        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_text_features"):
                continue

            config, inputs_dict = self._text_features_prepare_config_and_inputs()

            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_get_text_features_attentions(self):
        def check_attentions_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.set_attn_implementation("eager")
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_text_features(**inputs_dict)
            attentions = outputs.attentions
            # model.text_model(**inputs_dict) also no attentions for aimv2
            expected_num_attentions = self._text_features_get_expected_num_attentions()
            self.assertIsNotNone(attentions, "attentions should not be None")
            self.assertEqual(len(attentions), expected_num_attentions, "Number of attention layers mismatch")

        if not self.has_attentions:
            return

        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_text_features"):
                continue

            config, inputs_dict = self._text_features_prepare_config_and_inputs()
            inputs_dict["output_hidden_states"] = False
            inputs_dict["output_attentions"] = True
            check_attentions_output(inputs_dict, config, model_class)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_attentions = True

            check_attentions_output(inputs_dict, config, model_class)

    @parameterized.expand([True, False, None])
    def test_get_image_features_output(self, return_dict: bool | None):
        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_image_features"):
                continue

            config, inputs_dict = self._image_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            set_seed(42)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs_dict)

            if return_dict in (True, None):
                self.assertTrue(isinstance(outputs, ModelOutput), "get_image_features() must return a BaseModelOutput")
                self.assertTrue(
                    hasattr(outputs, "last_hidden_state"),
                    "get_image_features() must return a BaseModelOutput with last_hidden_state",
                )
                self.assertTrue(
                    hasattr(outputs, "pooler_output"),
                    "get_image_features() must return a BaseModelOutput with pooler_output",
                )
                self.assertTrue(
                    hasattr(outputs, "hidden_states"),
                    "get_image_features() must return a BaseModelOutput with hidden_states",
                )
                if self.has_attentions:
                    self.assertTrue(
                        hasattr(outputs, "attentions"),
                        "get_image_features() must return a BaseModelOutput with attentions",
                    )

                if getattr(self, "skip_test_image_features_output_shape", False):
                    return

                last_hidden_state_shape = outputs.last_hidden_state.shape
                batch_size = (
                    inputs_dict["pixel_values"].shape[0]
                    if "pixel_values" in inputs_dict
                    else inputs_dict["pixel_values_images"].shape[0]
                )
                self.assertEqual(
                    last_hidden_state_shape[0],
                    batch_size,
                    f"batch_size mismatch, full shape: {last_hidden_state_shape}",
                )

                vision_config = config.vision_config if hasattr(config, "vision_config") else config
                vision_config = (
                    vision_config.backbone_config if hasattr(vision_config, "backbone_config") else vision_config
                )
                vision_config = vision_config.vq_config if hasattr(vision_config, "vq_config") else vision_config
                vision_config = vision_config.model_args if hasattr(vision_config, "model_args") else vision_config
                attribute_candidates = [
                    "embed_dim_per_stage",
                    "embed_dim",
                    "embed_dims",
                    "out_hidden_size",
                    "hidden_size",
                    "hidden_dim",
                ]
                hidden_size = None
                for attr in attribute_candidates:
                    if hasattr(vision_config, attr):
                        hidden_size = getattr(vision_config, attr)
                        break
                    elif isinstance(vision_config, dict) and attr in vision_config:
                        hidden_size = vision_config[attr]
                        break
                else:
                    raise ValueError("Cannot find the hidden size attribute in vision_config")
                if isinstance(hidden_size, (list, tuple)):
                    hidden_size = hidden_size[-1]
                self.assertEqual(
                    last_hidden_state_shape[-1],
                    hidden_size,
                    f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
                )

            else:
                self.assertIsInstance(outputs, tuple, "get_image_features() must return a tuple if return_dict=False")

    def test_get_image_features_hidden_states(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_image_features(**inputs_dict)
            # hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            hidden_states = outputs.hidden_states
            expected_num_hidden_states = self._image_features_get_expected_num_hidden_states()
            self.assertIsNotNone(hidden_states, "hidden_states should not be None")
            self.assertEqual(len(hidden_states), expected_num_hidden_states, "Number of hidden states layers mismatch")

        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_image_features"):
                continue

            config, inputs_dict = self._image_features_prepare_config_and_inputs()

            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]

            def set_value_subconfigs(config, key, value):
                setattr(config, key, value)
                for k in config.sub_configs:
                    if (subconfig := getattr(config, k)) is not None:
                        set_value_subconfigs(subconfig, key, value)

            set_value_subconfigs(config, "output_hidden_states", True)
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_get_image_features_attentions(self):
        def check_attentions_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.set_attn_implementation("eager")
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_image_features(**inputs_dict)
            attentions = outputs.attentions
            # model.text_model(**inputs_dict) also no attentions for aimv2
            expected_num_attentions = self._image_features_get_expected_num_attentions()
            self.assertIsNotNone(attentions, "attentions should not be None")
            self.assertEqual(len(attentions), expected_num_attentions, "Number of attention layers mismatch")

        if not self.has_attentions:
            return

        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_image_features"):
                continue

            config, inputs_dict = self._image_features_prepare_config_and_inputs()
            inputs_dict["output_hidden_states"] = False
            inputs_dict["output_attentions"] = True
            check_attentions_output(inputs_dict, config, model_class)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]

            def set_value_subconfigs(config, key, value):
                setattr(config, key, value)
                for k in config.sub_configs:
                    if (subconfig := getattr(config, k)) is not None:
                        set_value_subconfigs(subconfig, key, value)

            set_value_subconfigs(config, "output_attentions", True)
            check_attentions_output(inputs_dict, config, model_class)

    @parameterized.expand([True, False, None])
    def test_get_audio_features_output(self, return_dict: bool | None):
        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_audio_features"):
                continue

            config, inputs_dict = self._audio_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            set_seed(42)
            with torch.no_grad():
                outputs = model.get_audio_features(**inputs_dict)

            if return_dict in (True, None):
                self.assertTrue(isinstance(outputs, ModelOutput), "get_audio_features() must return a BaseModelOutput")
                self.assertTrue(
                    hasattr(outputs, "last_hidden_state"),
                    "get_audio_features() must return a BaseModelOutput with last_hidden_state",
                )
                self.assertTrue(
                    hasattr(outputs, "pooler_output"),
                    "get_audio_features() must return a BaseModelOutput with pooler_output",
                )
                self.assertTrue(
                    hasattr(outputs, "hidden_states"),
                    "get_audio_features() must return a BaseModelOutput with hidden_states",
                )
                if self.has_attentions:
                    self.assertTrue(
                        hasattr(outputs, "attentions"),
                        "get_audio_features() must return a BaseModelOutput with attentions",
                    )

                if getattr(self, "skip_test_audio_features_output_shape", False):
                    return

                last_hidden_state_shape = outputs.last_hidden_state.shape
                batch_size = inputs_dict["input_features"].shape[0]
                self.assertEqual(
                    last_hidden_state_shape[0],
                    batch_size,
                    f"batch_size mismatch, full shape: {last_hidden_state_shape}",
                )

                audio_config = config.audio_config if hasattr(config, "audio_config") else config
                if hasattr(audio_config, "projection_dim"):
                    hidden_size = audio_config.projection_dim
                elif hasattr(audio_config, "hidden_size"):
                    hidden_size = audio_config.hidden_size
                elif hasattr(audio_config, "encoder_config"):
                    hidden_size = audio_config.encoder_config.hidden_dim
                elif hasattr(audio_config, "encoder_ffn_dim"):
                    hidden_size = audio_config.encoder_ffn_dim
                self.assertEqual(
                    last_hidden_state_shape[-1],
                    hidden_size,
                    f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
                )

            else:
                self.assertIsInstance(outputs, tuple, "get_audio_features() must return a tuple if return_dict=False")

    def test_get_audio_features_hidden_states(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_audio_features(**inputs_dict)
            hidden_states = outputs.hidden_states
            expected_num_hidden_states = self._audio_features_get_expected_num_hidden_states()
            self.assertIsNotNone(hidden_states, "hidden_states should not be None")
            self.assertEqual(len(hidden_states), expected_num_hidden_states, "Number of hidden states layers mismatch")

        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_audio_features"):
                continue

            config, inputs_dict = self._audio_features_prepare_config_and_inputs()

            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_get_audio_features_attentions(self):
        def check_attentions_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.set_attn_implementation("eager")
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_audio_features(**inputs_dict)
            attentions = outputs.attentions
            expected_num_attentions = self._audio_features_get_expected_num_attentions()
            self.assertIsNotNone(attentions, "attentions should not be None")
            self.assertEqual(len(attentions), expected_num_attentions, "Number of attention layers mismatch")

        if not self.has_attentions:
            return

        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_audio_features"):
                continue

            config, inputs_dict = self._audio_features_prepare_config_and_inputs()
            inputs_dict["output_hidden_states"] = False
            inputs_dict["output_attentions"] = True
            check_attentions_output(inputs_dict, config, model_class)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_attentions = True

            check_attentions_output(inputs_dict, config, model_class)

    @parameterized.expand([True, False, None])
    def test_get_video_features_output(self, return_dict: bool | None):
        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_video_features"):
                continue

            config, inputs_dict = self._video_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            set_seed(42)
            with torch.no_grad():
                outputs = model.get_video_features(**inputs_dict)

            if return_dict in (True, None):
                self.assertTrue(isinstance(outputs, ModelOutput), "get_video_features() must return a BaseModelOutput")
                self.assertTrue(
                    hasattr(outputs, "last_hidden_state"),
                    "get_video_features() must return a BaseModelOutput with last_hidden_state",
                )
                self.assertTrue(
                    hasattr(outputs, "pooler_output"),
                    "get_video_features() must return a BaseModelOutput with pooler_output",
                )
                self.assertTrue(
                    hasattr(outputs, "hidden_states"),
                    "get_video_features() must return a BaseModelOutput with hidden_states",
                )
                if self.has_attentions:
                    self.assertTrue(
                        hasattr(outputs, "attentions"),
                        "get_video_features() must return a BaseModelOutput with attentions",
                    )

                if getattr(self, "skip_test_video_features_output_shape", False):
                    return

                last_hidden_state_shape = outputs.last_hidden_state.shape
                if "pixel_values_videos" in inputs_dict:
                    batch_size = inputs_dict["pixel_values_videos"].shape[0]
                elif "pixel_values" in inputs_dict:
                    batch_size = inputs_dict["pixel_values"].shape[0]
                self.assertEqual(
                    last_hidden_state_shape[0],
                    batch_size,
                    f"batch_size mismatch, full shape: {last_hidden_state_shape}",
                )
                video_config = config
                if hasattr(config, "video_config"):
                    video_config = config.video_config
                elif hasattr(config, "vision_config"):
                    video_config = config.vision_config
                if hasattr(video_config, "out_hidden_size"):
                    hidden_size = video_config.out_hidden_size
                else:
                    hidden_size = video_config.hidden_size
                self.assertEqual(
                    last_hidden_state_shape[-1],
                    hidden_size,
                    f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
                )

            else:
                self.assertIsInstance(outputs, tuple, "get_video_features() must return a tuple if return_dict=False")

    def test_get_video_features_hidden_states(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_video_features(**inputs_dict)
            hidden_states = outputs.hidden_states
            expected_num_hidden_states = self._video_features_get_expected_num_hidden_states()
            self.assertIsNotNone(hidden_states, "hidden_states should not be None")
            self.assertEqual(len(hidden_states), expected_num_hidden_states, "Number of hidden states layers mismatch")

        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_video_features"):
                continue

            config, inputs_dict = self._video_features_prepare_config_and_inputs()

            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_get_video_features_attentions(self):
        def check_attentions_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.set_attn_implementation("eager")
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model.get_video_features(**inputs_dict)
            attentions = outputs.attentions
            expected_num_attentions = self._video_features_get_expected_num_attentions()
            self.assertIsNotNone(attentions, "attentions should not be None")
            self.assertEqual(len(attentions), expected_num_attentions, "Number of attention layers mismatch")

        if not self.has_attentions:
            return

        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_video_features"):
                continue

            config, inputs_dict = self._video_features_prepare_config_and_inputs()
            inputs_dict["output_hidden_states"] = False
            inputs_dict["output_attentions"] = True
            check_attentions_output(inputs_dict, config, model_class)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_attentions = True

            check_attentions_output(inputs_dict, config, model_class)

    def test_capture_outputs_decorator(self):
        """Test that the decorator `capture_outputs` is not chained, and that only the base models use it.
        Also test that we can return all the needed outputs, i.e. the kwargs are passed and the custom `XXXOutput`
        classes accept the necessary keys.
        Chaining the calls to `capture_outputs` for the same output is not allowed because:
            1) useless - because the class above in the graph can simply reuse the already collected outputs
            2) dangerous - as outputs WILL be mixed up between the callers, i.e. the first call to the decorator will
                capture and return only the portion of the outputs that was not captured by the second `capture_outputs`
                call for that output.
        Note that chaining on different outputs (i.e. first call is set to capture "hidden_states" and 2nd to capture "attentions"
        is allowed, as we do not mix up outputs in this case.)
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        COUNTER = defaultdict(lambda: 0)
        origional_set = CompileableContextVar.set
        origional_reset = CompileableContextVar.reset

        # Every time we enter the `capture_outputs` decorator, we first call `set`, and then `reset`. So if we end
        # up calling `set` twice in a row before `reset`, it means we chained the calls to `capture_outputs` which is
        # an illegal practice
        def new_set(self, value):
            nonlocal COUNTER
            for k in value.keys():
                COUNTER[k] += 1
            if any(v > 1 for v in COUNTER.values()):
                raise ValueError("You're calling `capture_outputs` several time in a chain!")
            return origional_set(self, value)

        def new_reset(self, token):
            nonlocal COUNTER
            current_val = self.context_var.get()
            for k in current_val.keys():
                COUNTER[k] -= 1
            origional_reset(self, token)

        for model_class in self.all_model_classes:
            # Reset the counter in case one subtest fails and thus does not clean it up correctly
            COUNTER = defaultdict(lambda: 0)
            # Each individual model is a subtest
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config)).to(device=torch_device)
                model.eval()

                recordable_outputs = [
                    (module._can_record_outputs or {}).keys()
                    for module in model.modules()
                    if isinstance(module, PreTrainedModel)
                ]
                recordable_outputs = set().union(*recordable_outputs)
                # If we don't use the `capture_outputs` decorator, this test has no use
                if len(recordable_outputs) == 0:
                    self.skipTest("No usage of the `capture_outputs` decorator.")

                # Prepare inputs
                inputs = self._prepare_for_class(inputs_dict, model_class)
                return_all = {}
                # For attentions, any of those capturable are captured by `output_attentions`
                if any(x in recordable_outputs for x in ("attentions", "cross_attentions", "mask_decoder_attentions")):
                    return_all["output_attentions"] = True
                if "hidden_states" in recordable_outputs:
                    return_all["output_hidden_states"] = True
                if "router_logits" in recordable_outputs:
                    return_all["output_router_logits"] = True

                # Merge them (SwitchTransformers provides `output_router_logits` in `inputs` as well so we need to avoid
                # passing it twice)
                all_inputs = {**inputs, **return_all}

                # If we don't trigger the exception of the new set, then all good
                with patch.object(CompileableContextVar, "set", new=new_set):
                    with patch.object(CompileableContextVar, "reset", new=new_reset):
                        with torch.no_grad():
                            _ = model(**all_inputs)


global_rng = random.Random()


def compare_state_dicts(state_dict1, state_dict2) -> bool:
    """Make sure 2 state dicts are the exact same"""
    # Make sure the keys are the exact same
    if sorted(state_dict1.keys()) != sorted(state_dict2.keys()):
        raise ValueError("The keys of both state dict are not the same")

    for k, v1 in state_dict1.items():
        v2 = state_dict2[k]
        try:
            torch.testing.assert_close(v1, v2)
        except Exception as e:
            raise AssertionError(f"For key {k}: {e}")

    return True


@contextmanager
def seeded_weight_init():
    """Add a seed before weight initialization, to get the same random weights deterministically"""
    try:
        # Monkey patch the method to add a seed (we do it on PreTrainedModel._initialize_weights, which wraps
        # `_init_weights` so that it can add the seed for composite models as well)
        original_initialize_weights = PreTrainedModel._initialize_weights

        def seeded_initialize_weights(*args, **kwargs):
            set_seed(42)
            original_initialize_weights(*args, **kwargs)

        PreTrainedModel._initialize_weights = seeded_initialize_weights

        yield
    finally:
        # Restore it
        PreTrainedModel._initialize_weights = original_initialize_weights


@contextmanager
def skip_weight_init():
    """Skip weight initialization by `_init_weights` altogether."""
    try:
        original_initialize_weights = PreTrainedModel._initialize_weights

        # Just do nothing instead
        def skip_initialize_weights(*args, **kwargs):
            pass

        PreTrainedModel._initialize_weights = skip_initialize_weights

        yield
    finally:
        # Restore it
        PreTrainedModel._initialize_weights = original_initialize_weights


def find_parent_traceback(full_param_name: str, model: PreTrainedModel) -> tuple[str, str, str]:
    """From a given parameter or buffer `full_param_name`, find its immediate parent class name and immediate
    PreTrainedModel parent class name."""
    parent_name, name = full_param_name.rsplit(".", 1) if "." in full_param_name else ("", full_param_name)
    parent = model.get_submodule(parent_name)
    immediate_parent_class = type(parent).__name__
    # Go back recursively to find the first PreTrainedModel from which we inherit
    while not isinstance(parent, PreTrainedModel):
        parent_name = parent_name.rsplit(".", 1)[0] if "." in parent_name else ""
        parent = model.get_submodule(parent_name)
    # Get the exact XXXPreTrainedModel
    pretrained_parent_class = next(x.__name__ for x in type(parent).mro() if "PreTrainedModel" in x.__name__)
    # Some models directly inherit from `PreTrainedModel` instead of `XXXPreTrainedModel`
    if pretrained_parent_class == "PreTrainedModel":
        pretrained_parent_class = type(parent).__name__

    return name, immediate_parent_class, pretrained_parent_class


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
