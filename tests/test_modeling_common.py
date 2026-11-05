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
import copy
import inspect
import itertools
import math
import re
import tempfile
import unittest.mock
import warnings
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from typing import get_args
from unittest.mock import Mock, patch

import numpy as np
import pytest
from parameterized import parameterized
from pytest import mark

from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    is_torch_available,
    set_seed,
)
from transformers.integrations import HfDeepSpeedConfig
from transformers.integrations.deepgemm import is_deepgemm_loadable
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
    unset_hf_deepspeed_config,
)
from transformers.integrations.moe import (
    batched_mm_experts_forward,
    deepgemm_bf16_experts_forward,
    grouped_mm_experts_forward,
    sonicmoe_experts_forward,
)
from transformers.integrations.sonicmoe import is_sonicmoe_loadable
from transformers.modeling_layers import GradientCheckpointingLayer
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
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.testing_utils import (
    force_serialization_as_bin_files,
    get_device_properties,
    hub_retry,
    is_flaky,
    require_accelerate,
    require_bitsandbytes,
    require_deepspeed,
    require_flash_attn,
    require_flash_attn_3,
    require_flash_attn_4,
    require_flex_attention,
    require_kernels,
    require_non_hpu,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_mps,
    require_torch_multi_accelerator,
    rocm_has_sdpa_flash_backend,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import (
    ModelOutput,
    is_rocm_platform,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
)
from transformers.utils.output_capturing import CompileableContextVar

from .exporters.test_export import ExportTesterMixin
from .test_modeling_mixins._helpers import (
    _config_zero_init,  # noqa
    _set_subconfig_attributes,
    floats_tensor,  # noqa
    global_rng,  # noqa
    ids_tensor,  # noqa
    prepare_config_headdim,
    random_attention_mask,  # noqa
)
from .test_modeling_mixins.test_pretraned_modeling import (
    AttnImplementationTesterMixin,
    ModelInterfaceTesterMixin,
    ModelRotaryEmbeddingTesterMixin,
    ModelWeightsIOTesterMixin,
    TesterInterfaceMixin,
)


if is_torch_available():
    import torch
    from torch import nn

    from transformers.integrations.accelerate import compute_module_sizes


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


TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION = [
    (
        # test name for the test runner
        f"{dtype}",
        # parameterization
        *(dtype,),
    )
    for dtype in ("fp16", "fp32", "bf16")
]


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


# Language modeling heads that score every position against its own target instead of shifting `labels` by one:
# the standalone decoders of encoder-decoder models (driven with already right-shifted `decoder_input_ids`), the
# permutation/masked LM heads, and the per-codebook audio LMs. `shift_labels` -- pre-shifted targets, see
# `test_causal_lm_loss_honors_shift_labels` -- carries no meaning for them, so that test does not apply.
UNSHIFTED_LM_HEADS = {
    "BartForCausalLM",
    "BigBirdPegasusForCausalLM",
    "BlenderbotForCausalLM",
    "BlenderbotSmallForCausalLM",
    "CpmAntForCausalLM",
    "MBartForCausalLM",
    "MarianForCausalLM",
    "MusicgenForCausalLM",
    "MusicgenMelodyForCausalLM",
    "MvpForCausalLM",
    "PLBartForCausalLM",
    "PegasusForCausalLM",
    "ProphetNetForCausalLM",
    "TrOCRForCausalLM",
    "WhisperForCausalLM",
    "XLMWithLMHeadModel",
    "XLNetLMHeadModel",
}


@require_torch
class ModelTesterMixin(
    ExportTesterMixin,
    AttnImplementationTesterMixin,
    ModelInterfaceTesterMixin,
    ModelRotaryEmbeddingTesterMixin,
    TesterInterfaceMixin,
    ModelWeightsIOTesterMixin,
):
    model_tester = None
    all_model_classes = ()
    test_resize_embeddings = True
    test_torch_exportable = True
    # Used in `check_training_gradient_checkpointing` to NOT check all params having gradient (e.g. for some MOE models)
    test_all_params_have_gradient = True
    has_attentions = True
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

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__), tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(copy.deepcopy(config)).to(torch_device).eval()
                inputs = self._prepare_for_class(inputs_dict, model_class)
                with torch.no_grad():
                    first = model(**inputs)[0]

                model.save_pretrained(tmp_dir)
                model = model_class.from_pretrained(tmp_dir).to(torch_device).eval()
                with torch.no_grad():
                    second = model(**inputs)[0]

                for t1, t2 in zip(first, second) if isinstance(first, tuple) else [(first, second)]:
                    torch.testing.assert_close(t1, t2, msg="Running save/load and forward yields different results")

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
            model.activation_checkpointing_enable()
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
            with torch.no_grad():
                model(**batched_input)

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

                # unfreeze additional layers, except those the model freezes on purpose
                intentionally_frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
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
                normal_grad_sums = {n: p.grad.abs().sum().item() for n, p in grad_expected_params}
                non_zero_grads_normal = {n for n, s in normal_grad_sums.items() if s > 0}

                # reset all gradients to zero for the comparison with the gradient checkpointing run
                optimizer.zero_grad()

                # now enable gradient checkpointing and compare the gradients
                model.activation_checkpointing_enable(activation_checkpointing_kwargs=gradient_checkpointing_kwargs)

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
                gradcp_grad_sums = {n: p.grad.abs().sum().item() for n, p in grad_expected_params}
                non_zero_grads_gradcp = {n for n, s in gradcp_grad_sums.items() if s > 0}

                if non_zero_grads_gradcp != non_zero_grads_normal:
                    only_in_normal = non_zero_grads_normal - non_zero_grads_gradcp
                    only_in_gradcp = non_zero_grads_gradcp - non_zero_grads_normal

                    # Observed flakiness (see #48332): a parameter's gradient is exactly 0.0 in one
                    # run and a tiny value (≤1e-6, orders of magnitude below real gradients) in the
                    # other. Either side can be the near-zero one. Treat such pairs as not a mismatch.
                    _fp_noise = 1e-6
                    only_in_normal = {
                        n
                        for n in only_in_normal
                        if not (gradcp_grad_sums[n] == 0.0 and normal_grad_sums[n] <= _fp_noise)
                    }
                    only_in_gradcp = {
                        n
                        for n in only_in_gradcp
                        if not (normal_grad_sums[n] == 0.0 and gradcp_grad_sums[n] <= _fp_noise)
                    }
                    self.assertEqual(
                        # set union
                        only_in_gradcp | only_in_normal,
                        set(),
                        f"non_zero_grads mismatch after filtering fp noise: only_in_normal={only_in_normal}, only_in_gradcp={only_in_gradcp}",
                    )

                if self.test_all_params_have_gradient:
                    for k, v in model.named_parameters():
                        if k in intentionally_frozen:
                            continue
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

    def test_encoder_decoder_loss_no_double_shift(self):
        """
        Encoder-decoder LM heads are fed right-shifted ``decoder_input_ids``, so their logits are already
        position-aligned with the targets: the training loss must be the cross-entropy against ``labels``
        directly. If it shifts a second time (e.g. by routing through ``ForCausalLMLoss`` without passing
        ``shift_labels``) the model trains against ``labels[..., 1:]`` -- an off-by-one in the loss.

        The right-shifted ``decoder_input_ids`` are built and supplied here, so every model is driven the same
        way regardless of how its own ``forward`` would have derived them.

        The oracle compares the *gradient* of the reported loss w.r.t. ``output.logits`` against the gradients
        of the aligned and of the shifted cross-entropy. Gradients carry the target positions, so they stay
        decisive even where the two loss values coincide numerically -- which happens whenever a tester's
        initialization makes the logits near-uniform (the T5-family testers set ``initializer_factor=0.002``).

        Scope: encoder-decoder models built on ``ModelTesterMixin`` whose ``forward`` owns the loss. The
        ``EncoderDecoderModel`` / ``VisionEncoderDecoderModel`` / ``SpeechEncoderDecoderModel`` wrappers are
        covered by separate ``EncoderDecoderMixin`` test suites; those wrapper models' ``forward`` already
        computes an aligned cross-entropy independently of the decoder, so they cannot hit this path.
        """
        # Class-based applicability gate first: this oracle needs a sequence-to-sequence LM head with one logit
        # vector per label token. Decide from `all_model_classes` alone -- some testers have no `get_config`, so
        # the config is not touched until after this gate.
        seq2seq_lm_names = [
            *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
            *get_values(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES),
            *get_values(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES),
        ]
        seq2seq_lm_classes = [c for c in self.all_model_classes if c.__name__ in seq2seq_lm_names]
        if not seq2seq_lm_classes:
            self.skipTest(reason="No sequence-to-sequence language modeling head to check")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not config.is_encoder_decoder:
            self.skipTest(reason="Model is not an encoder-decoder")

        vocab_size = config.get_text_config(decoder=True).vocab_size
        # Targets are >= 2 tokens long so that `labels[..., 1:]` is non-empty and the two oracles differ.
        tester_decoder_input_ids = inputs_dict.get("decoder_input_ids")
        seq_len = (
            tester_decoder_input_ids.shape[-1]
            if isinstance(tester_decoder_input_ids, torch.Tensor)
            else getattr(self.model_tester, "decoder_seq_length", getattr(self.model_tester, "seq_length", 2)) or 2
        )
        seq_len = max(2, seq_len)
        set_seed(42)
        labels = torch.randint(3, vocab_size, (self.model_tester.batch_size, seq_len), device=torch_device)

        for model_class in seq2seq_lm_classes:
            with self.subTest(model_class.__name__):
                # Deep-copy per class so nothing below mutates the shared config across iterations.
                model_config = copy.deepcopy(config)
                model_config.use_cache = False
                model_config.return_dict = True
                text_config = model_config.get_text_config(decoder=True)
                pad_fallback = getattr(self.model_tester, "pad_token_id", None) or 0
                for cfg in (model_config, text_config):
                    if getattr(cfg, "pad_token_id", None) is None:
                        # A pad id is still read while building decoder masks, and some models take it from the
                        # top config while others take it from the decoder text config.
                        cfg.pad_token_id = pad_fallback

                # The first decoder token moves the logits but cannot change which cross-entropy the reported
                # loss matches, so any in-vocabulary id will do. Use the model's own start id when the tester
                # sets one and 0 otherwise, since several testers leave `decoder_start_token_id` unset. The
                # range check is defensive: it keeps an id past the tester's vocabulary from failing in an
                # embedding lookup before the alignment assertions run.
                start_token_id = getattr(model_config, "decoder_start_token_id", None)
                if not isinstance(start_token_id, int) or not 0 <= start_token_id < vocab_size:
                    start_token_id = 0
                decoder_input_ids = labels.roll(shifts=1, dims=1)
                decoder_input_ids[:, 0] = start_token_id

                inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=False)
                # Replace the tester's decoder inputs with targets the model cannot have derived differently:
                # `decoder_input_ids` is `labels` right-shifted, so the logits are aligned with `labels`.
                inputs.pop("decoder_attention_mask", None)
                inputs["decoder_input_ids"] = decoder_input_ids
                inputs["labels"] = labels

                model = model_class(model_config).to(torch_device).eval()
                # No `torch.no_grad()`: the oracle below differentiates the reported loss w.r.t. the logits.
                output = model(**inputs)

                logits = output.logits
                vocab = logits.shape[-1]
                float_logits = logits.float()
                aligned = nn.functional.cross_entropy(float_logits.reshape(-1, vocab), labels.reshape(-1))
                double_shifted = nn.functional.cross_entropy(
                    float_logits[:, :-1, :].reshape(-1, vocab), labels[:, 1:].reshape(-1)
                )
                # The three gradients are taken from graphs that share nodes, so keep the graph alive until the
                # last one. `allow_unused` turns "the loss does not depend on `output.logits`" into a readable
                # failure instead of an autograd error.
                grad_reported = torch.autograd.grad(output.loss, logits, retain_graph=True, allow_unused=True)[0]
                self.assertIsNotNone(
                    grad_reported,
                    msg=f"{model_class.__name__}: the reported loss is not a function of `output.logits`",
                )
                grad_aligned = torch.autograd.grad(aligned, logits, retain_graph=True)[0]
                grad_shifted = torch.autograd.grad(double_shifted, logits)[0]
                # Guard the oracle: if both targets produce the same gradient the assertions below are vacuous.
                self.assertFalse(
                    torch.allclose(grad_aligned, grad_shifted),
                    msg=(
                        f"{model_class.__name__}: the aligned and shifted gradients are indistinguishable "
                        f"(vacuous oracle)"
                    ),
                )
                self.assertTrue(
                    torch.allclose(grad_reported, grad_aligned),
                    msg=f"{model_class.__name__}: training loss is not the cross-entropy against `labels`",
                )
                self.assertFalse(
                    torch.allclose(grad_reported, grad_shifted),
                    msg=(
                        f"{model_class.__name__}: training loss is the cross-entropy against `labels[..., 1:]`. "
                        f"`decoder_input_ids` are already right-shifted, so the loss must not shift `labels` again."
                    ),
                )

    def test_causal_lm_loss_honors_shift_labels(self):
        """
        Decoder-only LM heads shift ``labels`` by one internally to align them with the logits. Sequence and
        context parallel training (accelerate CP, Ulysses SP, DeepSpeed ALST) cannot rely on that: the shift has
        to happen on the full sequence, before it is sharded, so the trainer hands the model the already-aligned
        targets as ``shift_labels`` and the model must train against those. A model that drops ``shift_labels`` --
        by not forwarding ``**kwargs`` to ``self.loss_function``, or by shifting ``labels`` by hand -- silently
        trains every shard against the wrong targets, and the last token of each shard gets no target at all.

        The oracle is differential, so it assumes nothing about how a given head builds its loss: with
        ``shift_labels`` supplied, the reported loss has to move when ``shift_labels`` changes and has to stay put
        when only ``labels`` changes.
        """
        decoder_lm_names = [
            *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
            *get_values(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES),
        ]
        decoder_lm_classes = [
            c
            for c in self.all_model_classes
            if c.__name__ in decoder_lm_names
            and c.__name__ not in UNSHIFTED_LM_HEADS
            and "labels" in inspect.signature(c.forward).parameters
        ]
        if not decoder_lm_classes:
            self.skipTest(reason="No decoder-only language modeling head to check")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if config.is_encoder_decoder:
            self.skipTest(reason="Encoder-decoder logits are already aligned with the targets, nothing to shift")

        for model_class in decoder_lm_classes:
            with self.subTest(model_class.__name__):
                model_config = copy.deepcopy(config)
                model_config.use_cache = False
                model_config.return_dict = True
                # Heads that damp their logits (Cohere scales them by `logit_scale`) report a loss that moves by
                # a few 1e-5 when the targets change, so seed the init to keep that margin the same on every run.
                set_seed(42)
                model = model_class(model_config).to(torch_device).eval()

                # `_prepare_for_class` builds targets only for the heads it knows about; read the shape off the
                # logits for the others.
                target_shape = self._prepare_for_class(inputs_dict, model_class, return_labels=True).get("labels")
                if isinstance(target_shape, torch.Tensor):
                    target_shape = target_shape.shape
                else:
                    set_seed(42)
                    with torch.no_grad():
                        probe_logits = getattr(
                            model(**self._prepare_for_class(inputs_dict, model_class)), "logits", None
                        )
                    if not isinstance(probe_logits, torch.Tensor) or probe_logits.ndim != 3:
                        self.skipTest(reason="Tester does not build targets for this head")
                    target_shape = probe_logits.shape[:2]

                def targets(seed):
                    # Vary the target id from one position to the next, so that changing the targets moves the
                    # reported loss. Ids 0 and 1 are in range for every vocabulary.
                    generator = torch.Generator().manual_seed(seed)
                    return torch.randint(2, target_shape, generator=generator, dtype=torch.long).to(torch_device)

                def reported_loss(label_seed, shift_label_seed, model_class=model_class, model=model):
                    inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                    inputs["labels"] = targets(label_seed)
                    inputs["shift_labels"] = targets(shift_label_seed)
                    # Some heads run a stochastic front end (audio tokenizers sample); reseed so that the three
                    # forward passes below differ only in the targets they are given.
                    set_seed(42)
                    with torch.no_grad():
                        return model(**inputs).loss

                baseline = reported_loss(0, 0)
                self.assertIsNotNone(
                    baseline, msg=f"{model_class.__name__}: no loss is returned when targets are passed"
                )
                self.assertFalse(
                    torch.equal(baseline, reported_loss(0, 1)),
                    msg=(
                        f"{model_class.__name__}: the training loss does not depend on `shift_labels`. Pre-shifted "
                        f"targets are the only correct ones under sequence/context parallelism, so the loss must "
                        f"be taken against them."
                    ),
                )
                self.assertTrue(
                    torch.equal(baseline, reported_loss(1, 0)),
                    msg=(
                        f"{model_class.__name__}: the training loss still depends on `labels` although "
                        f"`shift_labels` was passed. `shift_labels` holds the targets already aligned with the "
                        f"logits and must fully replace `labels` in the loss."
                    ),
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
            attentions = (
                outputs.encoder_attentions
                if config.get_text_config(decoder=True).is_encoder_decoder
                else outputs.attentions
            )
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            _set_subconfig_attributes(config, "output_attentions", True)
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = (
                outputs.encoder_attentions
                if config.get_text_config(decoder=True).is_encoder_decoder
                else outputs.attentions
            )
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

            if config.get_text_config(decoder=True).is_encoder_decoder:
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
            elif config.get_text_config(decoder=True).is_encoder_decoder:
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
            _set_subconfig_attributes(config, "output_hidden_states", True)
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self._prepare_config_and_inputs_for_retain_grad_hidden_states_attentions()
        _set_subconfig_attributes(config, "output_hidden_states", True)

        config.output_hidden_states = True
        config.output_attentions = self.has_attentions
        _set_subconfig_attributes(config, "output_attentions", self.has_attentions)

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
                # `_from_config` properly handles the ZeRO-3 init, i.e. it defers the weight initialization until
                # the params can be gathered (they are empty shards during `__init__`)
                model = model_class._from_config(config)
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
                # Input ids should be expanded to the new maximum size of the vocabulary
                inputs_dict["input_ids"][:, -2] = new_model_vocab_size - 1

                # A distributed launcher is needed for the forward pass when deepspeed is enabled
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

            # adapt other vocab-based inputs accordingly
            if not is_deepspeed_zero3_enabled():
                for key in ["decoder_input_ids"]:
                    if key in inputs_dict:
                        inputs_dict[key].clamp_(max=model_vocab_size - 15 - 1)
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
                model = model_class._from_config(config)
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
                model = model_class._from_config(config)
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
            "train_micro_batch_size_per_gpu": 1,
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
            "train_micro_batch_size_per_gpu": 1,
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
                    model = model_class._from_config(config)
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
            "train_micro_batch_size_per_gpu": 1,
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
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
            },
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_embeddings_untied()

    def test_resize_embeddings_untied_no_reinit_on_post_init(self):
        if not self.test_resize_embeddings:
            self.skipTest(reason="test_resize_embeddings is set to `False`")

        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        original_config.tie_word_embeddings = False
        try:
            original_config.get_text_config().tie_word_embeddings = False
        except Exception as e:
            model_type = getattr(original_config, "model_type", "unknown")
            print(f"Could not set text config's `tie_word_embeddings` for model type `{model_type}`: {e}")

        if original_config.tie_word_embeddings:
            self.skipTest(reason="Model cannot untie embeddings")

        for model_class in self.all_model_classes:
            with self.subTest(model_class):
                config = copy.deepcopy(original_config)
                model = model_class(config).to(torch_device)
                model.eval()

                # The bug only affects nn.Linear LM heads created by _get_resized_lm_head
                output_embeds = model.get_output_embeddings()
                if not isinstance(output_embeds, nn.Linear):
                    continue

                model_vocab_size = config.get_text_config().vocab_size
                try:
                    model.resize_token_embeddings(model_vocab_size + 10)
                except (NotImplementedError, AttributeError):
                    continue

                output_embeds = model.get_output_embeddings()
                weights_before = output_embeds.weight.data.clone()
                bias_before = output_embeds.bias.data.clone() if output_embeds.bias is not None else None

                model.post_init()

                output_embeds_after = model.get_output_embeddings()
                self.assertTrue(
                    torch.equal(weights_before, output_embeds_after.weight.data),
                    "Output embedding weights were reinitialized by post_init() after resize_token_embeddings()",
                )
                if bias_before is not None:
                    self.assertTrue(
                        torch.equal(bias_before, output_embeds_after.bias.data),
                        "Output embedding bias was reinitialized by post_init() after resize_token_embeddings()",
                    )

    def test_base_model_prefix_loads_base_weights(self):
        """
        `base_model_prefix` has to be the attribute a head stores the base model under: the loader strips or prepends
        it to bridge base and head checkpoints, so any other value silently leaves every base weight random.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_model_names = MODEL_MAPPING_NAMES.get(config.model_type, ())
        base_model_names = (base_model_names,) if isinstance(base_model_names, str) else base_model_names
        for model_class in self.all_model_classes:
            model = model_class(config)
            for name, child in model.named_children():
                if type(child).__name__ not in base_model_names:
                    continue
                self.assertEqual(model.base_model_prefix, name)
                base_model, child_state_dict = type(child)(config), child.state_dict()
                # Heads may build their base model differently (no pooler, a mask token...): check the shared weights
                expected_keys = {
                    f"{name}.{key}"
                    for key, tensor in base_model.state_dict().items()
                    if key in child_state_dict and child_state_dict[key].shape == tensor.shape
                }
                self.assertTrue(expected_keys)
                with tempfile.TemporaryDirectory() as tmp:
                    base_model.save_pretrained(tmp)
                    _, info = model_class.from_pretrained(tmp, output_loading_info=True, ignore_mismatched_sizes=True)
                self.assertTrue(info["missing_keys"].isdisjoint(expected_keys), info["missing_keys"] & expected_keys)

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

            if not config.get_text_config(decoder=True).is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not config.get_text_config(decoder=True).is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_inputs_embeds_matches_input_ids(self, **model_specific_kwargs):
        """
        Specific model testing classes can override and pass custom kwargs, these
        are forwarded to model as is. For example: some models prepare position ids
        differently with input IDs or embeds, so passing prepared positions is needed.
        """
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
            inputs.update(**model_specific_kwargs)

            # Some models prepare position IDs based on input IDs, and skip if embeddings
            # are used. Precompute in that case to force matching
            if hasattr(model.base_model, "get_rope_index"):
                inputs["position_ids"] = model.base_model.get_rope_index(**inputs)[0]

            wte = model.get_input_embeddings()
            if not config.get_text_config(decoder=True).is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
                inputs_embeds = wte(input_ids)
                with torch.no_grad():
                    out_ids = model(input_ids=input_ids, **inputs)[0]
                    out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
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

    def test_forward_seq_classifier_with_num_labels(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ not in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                model_class(copy.deepcopy(config)).save_pretrained(tmp_dir)
                new_model = model_class.from_pretrained(tmp_dir, num_labels=42, ignore_mismatched_sizes=True)
                new_model.to(torch_device).eval()
                with torch.no_grad():
                    logits = new_model(**self._prepare_for_class(inputs_dict, model_class)).logits
                self.assertEqual(logits.shape[-1], 42)

    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ) -> None:
        r"""
        Tests the equivalence between the eager and flash attention implementations.
        This test is only for inference and runs with `dtype=torch.bfloat16`.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        dtype = torch.bfloat16

        # This flag is used to know if the test was skipped for all `self.all_model_classes` or not
        _has_run_at_least_one_model = False

        for model_class in self.all_model_classes:
            # Custom kernel which needs the mask interface to be properly usable on these models
            if not model_class._supports_attention_backend and not attn_implementation.startswith("flash_attention"):
                continue

            # Set seed for deterministic test - ensures reproducible model initialization and inputs
            set_seed(42)
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # disable flakiness via norms as much as possible
            set_config_for_less_flaky_test(config)

            # flash attention variants does not always support arbitrary headim
            config = prepare_config_headdim(config, 16)

            # forcing the prefill size to go over sliding window size to check for SWA correctness
            if getattr(config, "sliding_window", None):
                config.sliding_window = 2

            model = model_class(config)
            if not all(
                submodel._supports_flash_attn for submodel in model.modules() if isinstance(submodel, PreTrainedModel)
            ):
                continue

            # Some models only support a sub set of all FA implementations
            valid_fa_implementations = model._compatible_flash_implementations
            if valid_fa_implementations is not None and attn_implementation not in valid_fa_implementations:
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
                        main_input = main_input.to(dtype)
                first_inputs = {model.main_input_name: main_input, "output_hidden_states": True}
                # Some models have main input name which is different from input_ids, but require input_ids... e.g. BarkFine
                if model.main_input_name != "input_ids" and "input_ids" in inputs_dict:
                    first_inputs["input_ids"] = inputs_dict["input_ids"][:1]
                # If we have some pixel values, use them as well
                if model.main_input_name != "pixel_values" and "pixel_values" in inputs_dict:
                    # NOTE: this fixes qwen2_5_vl/omni because test break w/ pixel values
                    if "image_grid_thw" in inputs_dict:
                        continue
                    first_inputs["pixel_values"] = inputs_dict["pixel_values"][:1].to(dtype)
                # Some VLMs require image_sizes alongside pixel_values, e.g. lighton_ocr, llava_onevision
                if "image_sizes" in inputs_dict:
                    first_inputs["image_sizes"] = inputs_dict["image_sizes"][:1]

                # Add additional model inputs if there is any
                for key in getattr(self, "additional_model_inputs", []):
                    if key in inputs_dict:
                        first_inputs[key] = inputs_dict[key][:1]
                        if torch.is_tensor(first_inputs[key]) and torch.is_floating_point(first_inputs[key]):
                            first_inputs[key] = first_inputs[key].to(dtype)

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
                    tmpdirname, dtype=dtype, attn_implementation="eager", device_map=torch_device
                )
                # disable flakiness via norms as much as possible
                set_model_for_less_flaky_test(model)

                def _get_output_logits(outputs):
                    if "hidden_states" in outputs:
                        return outputs.hidden_states[-1]
                    elif model.config.is_encoder_decoder:
                        return outputs.decoder_hidden_states[-1]
                    elif "logits_per_image" in outputs:
                        return outputs.logits_per_image
                    elif "logits_per_video" in outputs:
                        return outputs.logits_per_video
                    else:
                        return outputs.logits

                # First run without attention mask
                outputs = model(**first_inputs)
                logits_1_eager = _get_output_logits(outputs)
                # Second run with attention mask and padding
                outputs = model(**second_inputs)
                logits_2_eager = _get_output_logits(outputs)

                # Switch to FA
                del model
                model = model_class.from_pretrained(
                    tmpdirname, dtype=dtype, attn_implementation=attn_implementation, device_map=torch_device
                )
                set_model_for_less_flaky_test(model)

                outputs = model(**first_inputs)
                logits_1_fa = _get_output_logits(outputs)
                outputs = model(**second_inputs)
                logits_2_fa = _get_output_logits(outputs)

                # Check the results
                # No mask
                torch.testing.assert_close(logits_1_eager, logits_1_fa, atol=atol, rtol=rtol)

                # With mask
                if dummy_attention_mask is not None:
                    attention_mask = slice(1, None) if padding_side == "left" else slice(None, -1)
                    logits_2_fa = logits_2_fa[:, attention_mask]
                    logits_2_eager = logits_2_eager[:, attention_mask]
                torch.testing.assert_close(logits_2_eager, logits_2_fa, atol=atol, rtol=rtol)

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
        # flash-attn3 is CUDA-only; on ROCm use aiter-flash-attn, which serves FA3 there.
        attn_implementation = (
            "kernels-community/aiter-flash-attn" if is_rocm_platform() else "kernels-community/flash-attn3"
        )
        self.flash_attn_inference_equivalence(attn_implementation=attn_implementation, padding_side="left")

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

    @require_flash_attn_4
    @require_torch_gpu
    @mark.flash_attn_4_test
    @slow
    @is_flaky()
    def test_flash_attn_4_inference_equivalence(self):
        self.flash_attn_inference_equivalence(attn_implementation="flash_attention_4", padding_side="left")

    @require_flash_attn_4
    @require_torch_gpu
    @mark.flash_attn_4_test
    @slow
    @is_flaky()
    def test_flash_attn_4_inference_equivalence_right_padding(self):
        self.flash_attn_inference_equivalence(attn_implementation="flash_attention_4", padding_side="right")

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
        rtols=None,
        atols=None,
    ):
        """
        This test is written as a regular function to be able to overload it easily with different tolerances.
        Otherwise, `parameterize.expand` prevents it as it removes the original function from the namespace.
        """
        self._test_eager_matches_sdpa_inference(
            name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels, rtols=rtols, atols=atols
        )

    def _test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
        rtols=None,
        atols=None,
    ):
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

                if (
                    hasattr(config, "use_mask_token")
                    or "use_mask_token" in inspect.signature(model.__init__).parameters
                ):
                    model_from_pretrained_kwargs["use_mask_token"] = True

                # TODO: remove this try/except, models should have a shared API
                try:
                    model_sdpa = model_class.from_pretrained(
                        **model_from_pretrained_kwargs, attn_implementation="sdpa"
                    )
                except ValueError:
                    model_sdpa = model_class.from_pretrained(**model_from_pretrained_kwargs)
                model_sdpa = model_sdpa.eval().to(torch_device)

                try:
                    model_eager = deepcopy(model_sdpa)
                    model_eager.set_attn_implementation("eager")
                except Exception as _:
                    model_eager = model_class.from_pretrained(
                        **model_from_pretrained_kwargs, attn_implementation="eager"
                    )
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
                            seqlen = inputs_dict.get(
                                "decoder_input_ids", processed_inputs[model.main_input_name]
                            ).shape[-1]
                        elif model.main_input_name in ("pixel_values", "input_values") and hasattr(
                            self.model_tester, "seq_length"
                        ):
                            # For models where the main input's last dimension is not the token sequence length
                            # (e.g. pixel_values.shape[-1] is image width, input_values.shape[-1] is num_mel_bins).
                            # Use model_tester.seq_length (num_patches + 1/2 for ViT/DeiT/AST) instead.
                            seqlen = self.model_tester.seq_length
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
                            k: v.to(torch_device) if isinstance(v, torch.Tensor) else v
                            for k, v in prepared_inputs.items()
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
                elif "decoder_output" in outputs_eager and "clipseg" in model_class.__name__.lower():
                    outputs_eager = outputs_eager["decoder_output"]
                    outputs_sdpa = outputs_sdpa["decoder_output"]
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
                elif torch_device in ["hpu", "npu", "tpu"]:
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
                    # Keep in float to avoid any under/overflows in e.g. fp16
                    mean_relative_diff = (
                        (logits_sdpa.float() - logits_eager.float()).abs() / (logits_eager.float().abs() + 1e-12)
                    ).mean()
                    raise ValueError(
                        f"mean relative difference for {key}: {mean_relative_diff:.3e}, torch atol = {atol}, torch rtol = "
                        f"{rtol}"
                    )

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
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

            inputs_dict = {k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in inputs_dict.items()}
            prepared_inputs = self._prepare_for_class(inputs_dict, model_class)

            implementations = ["eager", "batched_mm", "grouped_mm"]
            mocks = {
                "batched_mm": Mock(wraps=batched_mm_experts_forward),
                "grouped_mm": Mock(wraps=grouped_mm_experts_forward),
            }

            # sonicmoe and deepgemm do not yet support post-expert norms
            has_post_expert_norm_attribute = any(
                getattr(module, "has_post_expert_norm", False) for module in model.modules()
            )

            # `is_sonicmoe_loadable` checks for `kernels`, a Hopper+ GPU and the `nvidia-cutlass-dsl` / `apache-tvm-ffi`
            # build dependencies, so the kernel is only exercised where it can actually be loaded
            if dtype != torch.float32 and is_sonicmoe_loadable() and not has_post_expert_norm_attribute:
                # we also need nvidia-cutlass-dsl and apache-tvm-ffi
                mocks["sonicmoe"] = Mock(wraps=sonicmoe_experts_forward)
                implementations.append("sonicmoe")

            # `is_deepgemm_loadable` checks for kernels availibility and NVCC version
            if dtype == torch.bfloat16 and is_deepgemm_loadable() and not has_post_expert_norm_attribute:
                # DeepGEMM BF16 grouped forward requires Hopper+, a new-enough nvcc toolkit, and bf16 hidden states
                mocks["deepgemm"] = Mock(wraps=deepgemm_bf16_experts_forward)
                implementations.append("deepgemm")

            outputs = {}
            # This is needed because we call the functions through the interface's global mapping
            with patch.dict("transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS._global_mapping", mocks):
                for impl in implementations:
                    model.set_experts_implementation(impl)
                    self.assertEqual(model.config._experts_implementation, impl)

                    with torch.no_grad():
                        impl_outputs = model(**prepared_inputs)

                        output_tensors = []
                        if hasattr(impl_outputs, "logits"):
                            output_tensors.append(impl_outputs.logits)
                        if hasattr(impl_outputs, "last_hidden_state"):
                            output_tensors.append(impl_outputs.last_hidden_state)
                        if hasattr(impl_outputs, "start_logits"):
                            output_tensors.append(impl_outputs.start_logits)
                        if hasattr(impl_outputs, "end_logits"):
                            output_tensors.append(impl_outputs.end_logits)
                        outputs[impl] = output_tensors

                    self.assertTrue(outputs[impl], f"No outputs from {impl} implementation")

                    for name, mock in mocks.items():
                        if name == impl:
                            mock.assert_called()
                        else:
                            mock.assert_not_called()

                        mock.reset_mock()

            # all non-eager implementations must numerically match eager
            eager_outputs = outputs.pop("eager")
            for impl, impl_outputs in outputs.items():
                torch.testing.assert_close(eager_outputs, impl_outputs, rtol=1e-4, atol=1e-4)

    @require_torch_accelerator
    @slow
    def test_sdpa_can_dispatch_on_flash(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        device_type, major, minor = get_device_properties()
        if device_type == "cuda" and major < 8:
            self.skipTest(reason="This test requires an NVIDIA GPU with compute capability >= 8.0")
        elif device_type == "rocm" and not rocm_has_sdpa_flash_backend(major):
            self.skipTest(reason="This AMD GPU has no SDPA flash backend available")
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
                "evolla",
                "modernbert",
                "gemma3",
                # gemma4: the block-overlay mask in create_masks_for_vision_model forces mask
                # materialization unconditionally, so SDPA can never use the FA backend when the
                # vision portion is involved. This is by design and not fixable on the modeling side.
                "gemma4",
                "t5gemma",
                "diffllama",
                "dpr",
                "eomt",
                "gpt_bigcode",
                "jamba",
                "kosmos-2",
                "mllama",
                "lighton_ocr",
                "parakeet_encoder",
                "parakeet_ctc",
                "pi0",
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
            # This test is about the `flash_attn` package: skip models that can't use it
            valid_fa_implementations = model_class._compatible_flash_implementations
            if valid_fa_implementations is not None and "flash_attention_2" not in valid_fa_implementations:
                self.skipTest(f"{model_class.__name__} only supports {valid_fa_implementations}")
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

            # Some models only support a sub set of all FA implementations
            valid_fa_implementations = model_class._compatible_flash_implementations
            if valid_fa_implementations is not None and attn_implementation not in valid_fa_implementations:
                self.skipTest(f"{model_class.__name__} only supports {valid_fa_implementations}")

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

    @require_flash_attn_4
    @require_torch_gpu
    @mark.flash_attn_4_test
    @slow
    def test_flash_attn_4_from_config(self):
        self.flash_attn_from_config(attn_implementation="flash_attention_4")

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

    @require_flex_attention
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
            config = prepare_config_headdim(config, 16)  # specific to triton

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

    def _check_text_features_shape(self, model_class, config, inputs_dict, outputs):
        # Test against (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        expected_hidden_size = config.text_config.hidden_size
        expected_shape = (
            inputs_dict["input_ids"].shape[0],
            inputs_dict["input_ids"].shape[1],
            expected_hidden_size,
        )
        self.assertEqual(last_hidden_state.shape, expected_shape, "last_hidden_state shape mismatch")

    def _check_image_features_shape(self, model_class, config, inputs_dict, outputs):
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
        vision_config = vision_config.backbone_config if hasattr(vision_config, "backbone_config") else vision_config
        vision_config = vision_config.vq_config if hasattr(vision_config, "vq_config") else vision_config
        vision_config = vision_config.model_args if hasattr(vision_config, "model_args") else vision_config
        attribute_candidates = [
            "embed_dim_per_stage",
            "embed_dim",
            "embed_dims",
            "out_hidden_size",
            "hidden_size",
            "hidden_dim",
            "mm_embed_dim",  # gemma4-only
        ]
        if "Fuyu" in model_class.__name__:
            # very old model without an encoder - simple MLP as vision backbone
            # add a knob here to not overwrite the whole test
            hidden_size = config.get_text_config().hidden_size
        else:
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

    def _check_audio_features_shape(self, model_class, config, inputs_dict, outputs):
        last_hidden_state_shape = outputs.last_hidden_state.shape

        if "input_features" in inputs_dict:
            batch_size = inputs_dict["input_features"].shape[0]
        else:
            batch_size = inputs_dict["input_values"].shape[0]
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
            if hasattr(audio_config.encoder_config, "hidden_size"):
                hidden_size = audio_config.encoder_config.hidden_size
            else:
                hidden_size = audio_config.encoder_config.hidden_dim
        elif hasattr(audio_config, "encoder_ffn_dim"):
            hidden_size = audio_config.encoder_ffn_dim
        elif hasattr(audio_config, "out_hidden_size"):
            hidden_size = audio_config.out_hidden_size
        self.assertEqual(
            last_hidden_state_shape[-1],
            hidden_size,
            f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
        )

    def _features_prepare_config_and_inputs(self, modality):
        _FEATURE_INPUT_PATTERNS = {
            "text": r"^(input_ids|token_type_ids|attention_mask)$",
            "image": r"^(?!.*video).*(pixel|image)|^(spatial_shapes|qformer_input_ids)$",
            "audio": r"audio|input_values|input_features|^(padding_mask|is_longer|feature_attention_mask)$",
            "video": r"video|^(vision_feature_layer|vision_feature_select_strategy|cu_seqlens)$",
        }
        _IMAGE_TO_VIDEO_KEYS = {
            "pixel_values": "pixel_values_videos",
            "image_grid_thw": "video_grid_thw",
            "image_merge_sizes": "video_merge_sizes",
        }

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        sub_tester = getattr(self.model_tester, f"{'vision' if modality == 'image' else modality}_model_tester", None)
        if sub_tester is not None:
            return config, sub_tester.prepare_config_and_inputs_for_common()[1]

        # FIXME: raushan when we get proper VideoTester!
        if modality == "video":
            for src, dst in _IMAGE_TO_VIDEO_KEYS.items():
                if src in inputs_dict and dst not in inputs_dict:
                    inputs_dict[dst] = inputs_dict.pop(src)

        pattern = _FEATURE_INPUT_PATTERNS[modality]
        if modality == "audio" and config.model_type == "musicflamingo":
            pattern += r"|^input_ids$"
        return config, {k: v for k, v in inputs_dict.items() if re.search(pattern, k)}

    def _features_get_expected_num_attentions(self, modality, model_tester=None):
        model_tester = model_tester or self.model_tester
        sub_names = {"image": ["vision"], "video": ["video", "vision"]}.get(modality, [modality])
        for name in sub_names:
            if hasattr(model_tester, f"{name}_model_tester"):
                return self._features_get_expected_num_attentions(
                    modality, getattr(model_tester, f"{name}_model_tester")
                )
        sub_config = getattr(model_tester, f"{sub_names[0]}_config", None)
        if modality != "text" and isinstance(sub_config, dict) and "num_hidden_layers" in sub_config:
            return sub_config["num_hidden_layers"]
        if hasattr(model_tester, "expected_num_hidden_layers"):
            return model_tester.expected_num_hidden_layers - 1
        if hasattr(model_tester, "num_hidden_layers"):
            return model_tester.num_hidden_layers
        raise ValueError(f"Cannot determine the expected number of layers for {modality} features")

    @parameterized.expand(list(itertools.product(["text", "image", "audio", "video"], [True, False, None])))
    def test_get_features_output(self, modality, return_dict):
        for model_class in self.all_model_classes:
            if not hasattr(model_class, f"get_{modality}_features"):
                continue

            config, inputs_dict = self._features_prepare_config_and_inputs(modality)
            if len(inputs_dict) == 0:
                self.skipTest(f"We didn't find any inputs for {modality}!")

            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval().to(torch_device)
            set_seed(42)
            with torch.no_grad():
                outputs = getattr(model, f"get_{modality}_features")(**inputs_dict)

            if return_dict is False:
                self.assertIsInstance(
                    outputs, tuple, f"get_{modality}_features() must return a tuple if return_dict=False"
                )
                continue

            self.assertIsInstance(outputs, ModelOutput, f"get_{modality}_features() must return a ModelOutput")
            for field in ("last_hidden_state", "pooler_output"):
                self.assertTrue(hasattr(outputs, field), f"get_{modality}_features() must return `{field}`")
            if getattr(self, f"skip_test_{modality}_features_output_shape", False):
                continue
            getattr(self, f"_check_{modality}_features_shape")(model_class, config, inputs_dict, outputs)

    @parameterized.expand(["text", "image", "audio", "video"])
    def test_get_features_hidden_states(self, modality):
        def check(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                outputs = getattr(model, f"get_{modality}_features")(**inputs_dict)
            self.assertTrue(hasattr(outputs, "hidden_states"), "output must have hidden_states")
            self.assertIsNotNone(outputs.hidden_states, "hidden_states should not be None")
            self.assertEqual(
                len(outputs.hidden_states),
                self._features_get_expected_num_attentions(modality) + 1,
                "Number of hidden states layers mismatch",
            )

        for model_class in self.all_model_classes:
            if not hasattr(model_class, f"get_{modality}_features"):
                continue

            config, inputs_dict = self._features_prepare_config_and_inputs(modality)
            if len(inputs_dict) == 0:
                self.skipTest(f"We didn't find any inputs for {modality}!")

            inputs_dict["output_hidden_states"] = True
            check(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            _set_subconfig_attributes(config, "output_hidden_states", True)
            check(inputs_dict, config, model_class)

    @parameterized.expand(["text", "image", "audio", "video"])
    def test_get_features_attentions(self, modality):
        if not self.has_attentions:
            self.skipTest("Model architecture does not support attentions")

        def check(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.set_attn_implementation("eager")
            model.to(torch_device).eval()
            with torch.no_grad():
                outputs = getattr(model, f"get_{modality}_features")(**inputs_dict)
            self.assertTrue(hasattr(outputs, "attentions"), "output must have attentions")
            self.assertIsNotNone(outputs.attentions, "attentions should not be None")
            self.assertEqual(
                len(outputs.attentions),
                self._features_get_expected_num_attentions(modality),
                "Number of attention layers mismatch",
            )

        def has_timm_config(config):
            "Recursive check over all sub-configs"
            for key in config.sub_configs_defaults:
                sub_config = getattr(config, key)
                if "Timm" in sub_config.__class__.__name__:
                    return True
                return has_timm_config(sub_config)
            return False

        for model_class in self.all_model_classes:
            if not hasattr(model_class, f"get_{modality}_features"):
                continue

            config, inputs_dict = self._features_prepare_config_and_inputs(modality)
            if has_timm_config(config):
                self.skipTest("Cannot set `output_attentions` for timm models.")

            if len(inputs_dict) == 0:
                self.skipTest(f"We didn't find any inputs for {modality}!")

            inputs_dict["output_hidden_states"] = False
            inputs_dict["output_attentions"] = True
            check(inputs_dict, config, model_class)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            _set_subconfig_attributes(config, "output_attentions", True)
            check(inputs_dict, config, model_class)

    @parameterized.expand(["image", "video"])
    def test_get_features_pooler_output(self, modality):
        """
        Generation assumes that encoded images always follow certain layout. This test is IMPORTANT
        and should NEVER be skipped unless your model override beam-search expansion on `generate()`
        """
        for model_class in self.all_model_classes:
            if not hasattr(model_class, f"get_{modality}_features"):
                continue
            if model_class.__name__ not in get_values(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES):
                continue
            config, inputs_dict = self._features_prepare_config_and_inputs(modality)
            model = model_class(config).eval().to(torch_device)
            with torch.no_grad():
                pooler_output = getattr(model, f"get_{modality}_features")(
                    **inputs_dict, return_dict=True
                ).pooler_output
            if pooler_output is None:
                continue
            bs = self.model_tester.batch_size
            self.assertEqual(
                len(pooler_output),
                bs,
                f"`pooler_output` should have one entry per sample: {len(pooler_output)} != {bs}",
            )
            self.assertEqual(
                pooler_output[0].ndim,
                2,
                f"each sample in `pooler_output` should be 2D but got {pooler_output[0].ndim}",
            )

    def _check_video_features_shape(self, model_class, config, inputs_dict, outputs):
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

    def test_output_router_logits_from_config(self):
        """`config.output_router_logits` turns the router logits on, and an explicit forward argument wins over it.
        A head that wraps a MoE backbone has to resolve the flag against the config like the backbone does, otherwise
        the config setting is silently ignored."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not hasattr(config.get_text_config(decoder=True), "output_router_logits"):
            self.skipTest("This model has no `output_router_logits` in its config.")

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                return_type = model_class.forward.__annotations__.get("return")
                output_fields = set().union(
                    *(getattr(t, "__dataclass_fields__", {}).keys() for t in (get_args(return_type) or (return_type,)))
                )
                if "router_logits" not in output_fields:
                    continue

                model = model_class(copy.deepcopy(config)).to(device=torch_device)
                model.eval()
                model.config.get_text_config(decoder=True).output_router_logits = False
                inputs = self._prepare_for_class(inputs_dict, model_class)
                inputs.pop("output_router_logits", None)

                with torch.no_grad():
                    explicit = model(**inputs, output_router_logits=True)
                    self.assertFalse(model(**inputs).router_logits, "router logits returned with the flag off")

                    model.config.get_text_config(decoder=True).output_router_logits = True
                    from_config = model(**inputs)
                    self.assertTrue(from_config.router_logits, "`config.output_router_logits=True` was ignored")
                    if getattr(explicit, "aux_loss", None) is not None:
                        self.assertIsNotNone(
                            from_config.aux_loss, "`config.output_router_logits=True` skipped the aux loss"
                        )
                    self.assertFalse(
                        model(**inputs, output_router_logits=False).router_logits,
                        "an explicit `output_router_logits=False` did not win over the config",
                    )

    def test_can_capture_specific_layers_hidden_states(self):
        """
        Test that we can capture only a subset of the hidden states with `output_hidden_states`.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # Each individual model is a subtest
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config)).to(device=torch_device)
                model.eval()

                can_set_hidden_states_layers = any(
                    "hidden_states" in (module._can_record_outputs or {})
                    and not isinstance(module._can_record_outputs["hidden_states"], list)
                    for module in model.modules()
                    if isinstance(module, PreTrainedModel)
                )
                if not can_set_hidden_states_layers:
                    self.skipTest(
                        "Cannot capture hidden states, or cannot correctly set layer indices due to several classes capturing at the same time"
                    )

                # Prepare inputs
                inputs = self._prepare_for_class(inputs_dict, model_class)

                # Despite the previous skip based on decorators, some models do not propagate the outputs correctly.......
                out = model(**inputs, output_hidden_states=True)
                if "hidden_states" in out:
                    hidden_states = out.hidden_states
                elif "text_model_output" in out and "hidden_states" in out.text_model_output:
                    hidden_states = out.text_model_output.hidden_states
                else:
                    self.skipTest(
                        f"{model_class.__name__} does not propagate hidden states correctly or uses another name"
                    )

                skips_first_input = any(
                    not module._can_record_outputs["hidden_states"].capture_initial_hidden_state
                    for module in model.modules()
                    if isinstance(module, PreTrainedModel)
                    and hasattr(
                        (module._can_record_outputs or {}).get("hidden_states", None), "capture_initial_hidden_state"
                    )
                )
                N_layers = len(hidden_states) if skips_first_input else len(hidden_states) - 1
                # Capture one every 2 layers
                indices_to_capture = list(range(0, N_layers, 2))
                out = model(**inputs, output_hidden_states=indices_to_capture)

                hidden_states = out.hidden_states if "hidden_states" in out else out.text_model_output.hidden_states
                # hidden_states = out.hidden_states
                # We should have the same number of captured hidden_states, i.e. we do not add the first input
                self.assertEqual(len(hidden_states), N_layers)
                # assert we correctly captured the layers
                self.assertTrue(all(hidden_states[i] is None for i in range(N_layers) if i not in indices_to_capture))
                self.assertTrue(
                    all(isinstance(hidden_states[i], torch.Tensor) for i in range(N_layers) if i in indices_to_capture)
                )
