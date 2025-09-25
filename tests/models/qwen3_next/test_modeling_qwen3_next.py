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
import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import Qwen3NextConfig, is_torch_available
from transformers.testing_utils import require_torch, require_torch_multi_gpu, slow, torch_device


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
from ...generation.test_utils import has_similar_generate_outputs
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
        "Qwen3-Next has a special Cache as it alternates with gated deltanet layers"
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
        "Needs to be overwritten as Qwen3-Next alternates between attention layers and gated deltanet layers."
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

    @pytest.mark.generate
    def test_generate_continue_from_past_key_values(self):
        "Needs to be overwritten as Qwen3-Next has non-standard cache."
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config).to(torch_device)
            model.eval()

            generate_kwargs = {
                "pad_token_id": -1,
                "eos_token_id": -1,
                "forced_eos_token_id": None,
                "encoder_no_repeat_ngram_size": 0,
                "use_cache": True,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            outputs = model.generate(**inputs, **generate_kwargs, max_new_tokens=4)
            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=3)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[-1]

            inputs["input_ids"] = outputs_cached.sequences
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.nn.functional.pad(
                    inputs["attention_mask"],
                    (0, new_attention_len - inputs["attention_mask"].shape[1]),
                    mode="constant",
                    value=1,
                )
            first_caches_scores = outputs_cached.scores
            outputs_cached = model.generate(**inputs, **generate_kwargs, max_new_tokens=1)
            full_cached_scores = first_caches_scores + outputs_cached.scores
            outputs_cached.scores = full_cached_scores

            # The two sets of generated text and past kv should be equal to each other
            self.assertTrue(has_similar_generate_outputs(outputs, outputs_cached))
            for layer_idx in range(len(outputs_cached.past_key_values)):
                for kv_idx in range(len(outputs_cached.past_key_values[layer_idx])):
                    # Diff with the main test: we need to skip layers where it stays None
                    if outputs.past_key_values[layer_idx][kv_idx] is not None:
                        self.assertTrue(
                            torch.allclose(
                                outputs.past_key_values[layer_idx][kv_idx],
                                outputs_cached.past_key_values[layer_idx][kv_idx],
                            )
                        )

    @pytest.mark.generate
    def test_generate_continue_from_inputs_embeds(self):
        "Needs to be overwritten as Qwen3-Next has non-standard cache."
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).to(torch_device).eval()
            input_ids = inputs_dict.pop("input_ids")
            model.generation_config.pad_token_id = model.generation_config.eos_token_id = -1
            model.generation_config.forced_eos_token_id = None
            model.config.is_decoder = True
            model.generation_config.use_cache = True

            generation_kwargs = {
                "return_dict_in_generate": True,
                "do_sample": False,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values.
            input_embeds = model.get_input_embeddings()(input_ids)
            outputs = model.generate(inputs_embeds=input_embeds, max_new_tokens=4, **generation_kwargs)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens)
            initial_output = model.generate(inputs_embeds=input_embeds, max_new_tokens=3, **generation_kwargs)
            continued_embeds = torch.cat([input_embeds, model.get_input_embeddings()(initial_output.sequences)], dim=1)
            cached_output = model.generate(
                inputs_embeds=continued_embeds,
                max_new_tokens=1,
                past_key_values=initial_output.past_key_values,
                **generation_kwargs,
            )

            # Combine the (3 + 1) generated tokens and verify it matches with full generation.
            combined_output_sequences = torch.concat([initial_output.sequences, cached_output.sequences], axis=1)
            self.assertListEqual(outputs.sequences.tolist(), combined_output_sequences.tolist())
            # The two sets of past kv should be equal to each other
            for layer_idx in range(len(cached_output.past_key_values)):
                for kv_idx in range(len(cached_output.past_key_values[layer_idx])):
                    # Diff with the main test: we need to skip layers where it stays None
                    if outputs.past_key_values[layer_idx][kv_idx] is not None:
                        self.assertTrue(
                            torch.allclose(
                                outputs.past_key_values[layer_idx][kv_idx],
                                cached_output.past_key_values[layer_idx][kv_idx],
                            )
                        )

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3-Next alternates between attention layers and gated deltanet layers."
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

    @require_torch_multi_gpu
    def test_can_use_device_map(self):
        """
        Test that this model can be dispatched on multiple gpus. It's not obvious as the Cache is not standard,
        ant each layer need to use the correct device on which it reside (i.e. it needs to be lazy initialized).
        """
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            inputs_dict = {k: v.to(0) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}
            # We want the linear attention layer to reside on device 1 with the device map (i.e. not the first/default device),
            # to check if cache initialization is on the correct device
            config.layer_types = ["full_attention", "linear_attention"]
            model = model_class(config).eval()

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                del model
                model = model_class.from_pretrained(
                    tmpdirname,
                    device_map={
                        "lm_head": 0,
                        "model.embed_tokens": 0,
                        "model.norm": 0,
                        "model.layers.0": 0,
                        "model.layers.1": 1,
                    },
                )

                # Check that we indeed use 2 different devices for each layer
                self.assertTrue({param.device for param in model.model.layers[0].parameters()} == {torch.device(0)})
                self.assertTrue({param.device for param in model.model.layers[1].parameters()} == {torch.device(1)})

                # This should not crash
                _ = model.generate(**inputs_dict, max_new_tokens=5, min_new_tokens=5)


@slow
class Qwen3NextIntegrationTest(unittest.TestCase):
    pass
