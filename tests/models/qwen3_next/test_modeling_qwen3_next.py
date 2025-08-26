# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
import unittest

import pytest
from parameterized import parameterized

from transformers import Qwen3NextConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        Qwen3NextForCausalLM,
        Qwen3NextForQuestionAnswering,
        Qwen3NextForSequenceClassification,
        Qwen3NextForTokenClassification,
        Qwen3NextModel,
    )
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDynamicCache

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    _config_zero_init,
    _test_eager_matches_sdpa_inference,
)


class Qwen3NextModelTester(CausalLMModelTester):
    config_class = Qwen3NextConfig
    if is_torch_available():
        base_model_class = Qwen3NextModel
        causal_lm_class = Qwen3NextForCausalLM
        sequence_class = Qwen3NextForSequenceClassification
        token_class = Qwen3NextForTokenClassification
        question_answering_class = Qwen3NextForQuestionAnswering

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["linear_attention", "full_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8


@require_torch
class Qwen3NextModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Qwen3NextModel,
            Qwen3NextForCausalLM,
            Qwen3NextForSequenceClassification,
            Qwen3NextForTokenClassification,
            Qwen3NextForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Qwen3NextModel,
            "text-classification": Qwen3NextForSequenceClassification,
            "token-classification": Qwen3NextForTokenClassification,
            "text-generation": Qwen3NextForCausalLM,
            "question-answering": Qwen3NextForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    model_tester_class = Qwen3NextModelTester

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        "Qwen3-Next has a special Cache as it alternates with Mamba layers"
        self.assertIsInstance(decoder_past_key_values, Qwen3NextDynamicCache)

        # (batch, head, seq_length, head_features)
        expected_shape = (
            batch_size,
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
            cache_length,
            config.hidden_size // config.num_attention_heads,
        )

        attention_layer_indices = decoder_past_key_values.transformer_layers
        self.assertListEqual(
            [decoder_past_key_values.key_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )
        self.assertListEqual(
            [decoder_past_key_values.value_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )

    @pytest.mark.generate
    def test_past_key_values_format(self):
        "Needs to be overwritten as Qwen3-Next alternates between attention layers and mamba layers."
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            model = model_class(config).to(torch_device)
            model = model.eval()
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            past_kv = outputs["past_key_values"]

            num_query_attention_heads = config.num_attention_heads
            embed_dim = config.hidden_size
            per_head_embed_dim = embed_dim // num_query_attention_heads
            num_key_value_heads = getattr(config, "num_key_value_heads", num_query_attention_heads)

            batch_size, seq_length = inputs["input_ids"].shape[:2]
            default_self_attention_shape = (batch_size, num_key_value_heads, seq_length, per_head_embed_dim)

            num_cache_decoder_layers = len(past_kv)
            self.assertEqual(num_cache_decoder_layers, config.num_hidden_layers)

            for i in range(config.num_hidden_layers):
                if config.layer_types[i] == "full_attention":
                    self_attention_layer_keys = past_kv.key_cache[i]
                    self_attention_layer_values = past_kv.value_cache[i]
                    self.assertEqual(self_attention_layer_keys.shape, default_self_attention_shape)
                    self.assertEqual(self_attention_layer_values.shape, default_self_attention_shape)

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3-Next alternates between attention layers and mamba layers."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

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
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    def test_initialization(self):
        "Some parameters need to be skipped."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=copy.deepcopy(configs_no_init))
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # this one need to be skipped, it's initialized as log(uniform(0, 16))
                    if "A_log" in name:
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @unittest.skip("Redundant with `test_initialization`, and fails because of the same param (`A_log`)")
    def test_mismatched_shapes_have_properly_initialized_weights(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        """
        We need to overwrite this without the fp16 part of the dtype, because the slow path `torch_chunk_gated_delta_rule`
        is not robust enough (flaky test) in fp16 due to upscaling in fp32 and then downscaling to fp16 at the end
        """
        if dtype == "fp16":
            self.skipTest("Not robust in fp16")
        _test_eager_matches_sdpa_inference(
            self,
            name,
            dtype,
            padding_side,
            use_attention_mask,
            output_attentions,
            enable_kernels,
        )

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@slow
class Qwen3MoeIntegrationTest(unittest.TestCase):
    pass
