# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch LLaMA model."""

import unittest

import pytest

from transformers import AutoTokenizer, is_torch_available, set_seed
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM, Lfm2MoeModel


class Lfm2MoeModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Lfm2MoeConfig
        base_model_class = Lfm2MoeModel
        causal_lm_class = Lfm2MoeForCausalLM

    def __init__(
        self,
        parent,
        layer_types=["full_attention", "conv"],
    ):
        super().__init__(parent)
        self.layer_types = layer_types


@require_torch
class Lfm2MoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (Lfm2MoeModel, Lfm2MoeForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Lfm2MoeModel,
            "text-generation": Lfm2MoeForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = Lfm2MoeModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Lfm2MoeForCausalLM if is_torch_available() else None

    def test_attention_outputs(self):
        """Lfm2Moe alternates between attention and short-conv layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            config = model.config
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    @pytest.mark.generate
    def test_past_key_values_format(self):
        """Lfm2Moe has a special cache format as it alternates between attention and conv layers"""
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            model = model_class(config).to(torch_device).eval()
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
            default_conv_shape = (batch_size, config.hidden_size, config.conv_L_cache)

            num_cache_decoder_layers = len(past_kv)
            self.assertEqual(num_cache_decoder_layers, config.num_hidden_layers)

            for i in range(config.num_hidden_layers):
                if config.layer_types[i] == "full_attention":
                    self_attention_layer_keys = past_kv.key_cache[i]
                    self_attention_layer_values = past_kv.value_cache[i]
                    self.assertEqual(self_attention_layer_keys.shape, default_self_attention_shape)
                    self.assertEqual(self_attention_layer_values.shape, default_self_attention_shape)
                else:
                    conv_layer = past_kv.conv_cache[i]
                    self.assertEqual(conv_layer.shape, default_conv_shape)


@require_torch_accelerator
@require_read_token
@slow
class Lfm2MoeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = Lfm2MoeForCausalLM.from_pretrained(
                "LiquidAI/LFM2-8B-A1B", device_map="auto", dtype=torch.bfloat16
            )
        return cls.model

    @slow
    def test_model_1a8b_logits(self):
        set_seed(1789)
        input_ids = [1, 22998, 768, 1947, 797, 22017, 811, 6332, 928, 5743, 797, 779, 48123, 772, 33551, 60996, 523]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor(
            [
                [
                    -1.3855,
                    -0.5123,
                    -1.3143,
                    -1.2144,
                    -1.0791,
                    -1.2117,
                    -1.4704,
                    -0.7648,
                    -0.6175,
                    -1.2402,
                    -1.1459,
                    -1.0083,
                    -1.0247,
                    -0.8830,
                    -1.5643,
                    -1.7266,
                    -1.6254,
                ]
            ]
        )
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # Expected portion of the logits
        EXPECTED_SLICE = torch.tensor(
            [-1.2656, 2.4844, 5.5000, -1.3359, -1.3203, -1.3438, 1.9375, 5.8438, -0.6523, -1.2891]
        )
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @slow
    def test_model_1a8b_generation(self):
        EXPECTED_TEXT_COMPLETION = """In 1st century A.D., the Roman Empire controlled much of Europe, North Africa, and parts of the Middle East."""
        set_seed(1789)
        prompt = "In 1st century A.D., the Roman Empire"
        tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-8B-A1B", use_fast=False)
        model = self.get_model()
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(
            model.model.embed_tokens.weight.device
        )
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=15, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_model_1a8b_batched_chat_generation(self):
        prompts = ["Who are you?", "Complete the text: Lorem ipsum dolor ", "The Meji Restoration in Japan ended"]
        EXPECTED_TEXT_COMPLETIONS = [
            "Who are you??  \nI am an artificial intelligence assistant designed to provide information, answer questions",
            "Complete the text: Lorem ipsum dolor ipsum dolor ipsum dolor ipsum dolor ipsum dolor",
            "The Meji Restoration in Japan ended (1868) marked the:  \nA) Establishment of a constitutional",
        ]
        set_seed(1789)
        tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-8B-A1B", use_fast=False)
        model = self.get_model()
        batched_input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to(
            model.model.embed_tokens.weight.device
        )
        with torch.no_grad():
            generated_ids = model.generate(**batched_input_ids, max_new_tokens=15, do_sample=False)
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETIONS, text)
