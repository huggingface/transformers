# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers.models.cwm import (
        CwmConfig,
        CwmForCausalLM,
        CwmModel,
    )


class CwmModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = CwmConfig
        base_model_class = CwmModel
        causal_lm_class = CwmForCausalLM

    def get_config(self):
        config = super().get_config()

        config.sliding_window = 8192
        config.rope_scaling = {
            "factor": 16.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }

        return config


@require_torch
class CwmModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            CwmModel,
            CwmForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": CwmModel,
            "text-generation": CwmForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor
    model_tester_class = CwmModelTester

    # 0.8 for `test_cpu_offload`
    model_split_percents = [0.5, 0.7, 0.8]

    # for `test_torch_compile_for_training`
    _torch_compile_train_cls = CwmForCausalLM if is_torch_available() else None

    def test_cwm_forward_with_sliding_window(self):
        config = self.model_tester.get_config()
        model = CwmModel(config)
        model.to(torch_device)
        model.eval()

        # input longer than sliding window
        seq_length = config.sliding_window + 10
        input_ids = torch.randint(0, config.vocab_size, (1, seq_length), device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (1, seq_length, config.hidden_size))

    def test_cwm_attention_mask_mapping(self):
        config = self.model_tester.get_config()
        model = CwmModel(config)
        model.to(torch_device)
        model.eval()

        seq_length = 20
        input_ids = torch.randint(0, config.vocab_size, (1, seq_length), device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids)

        # no errors
        self.assertIsNotNone(outputs.last_hidden_state)


class CwmIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_cwm_small_model_forward(self):
        config = CwmConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            sliding_window=32,
        )

        model = CwmForCausalLM(config)
        model.to(torch_device)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (2, 50), device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.logits.shape, (2, 50, config.vocab_size))

    def test_cwm_with_cache(self):
        config = CwmConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            sliding_window=32,
            use_cache=True,
        )

        model = CwmForCausalLM(config)
        model.to(torch_device)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 10), device=torch_device)

        with torch.no_grad():
            outputs1 = model(input_ids, use_cache=True)
            self.assertIsNotNone(outputs1.past_key_values)

            new_input_ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)
            outputs2 = model(new_input_ids, past_key_values=outputs1.past_key_values, use_cache=True)
            self.assertIsNotNone(outputs2.past_key_values)

    @slow
    def test_cwm_rope_scaling_llama3(self):
        # Llama3 rope scaling
        config = CwmConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            sliding_window=64,
            rope_scaling={
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 128,
                "rope_type": "llama3",
            },
        )

        model = CwmModel(config)
        model.to(torch_device)
        model.eval()

        # sequence longer than original max position embeddings
        long_input = torch.randint(0, config.vocab_size, (1, 200), device=torch_device)

        with torch.no_grad():
            outputs = model(long_input)

        self.assertEqual(outputs.last_hidden_state.shape, (1, 200, config.hidden_size))

    def test_cwm_mixed_attention_layers(self):
        config = CwmConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            sliding_window=32,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        model = CwmModel(config)
        model.to(torch_device)
        model.eval()

        expected_types = config.layer_types
        for i, layer in enumerate(model.layers):
            self.assertEqual(layer.layer_type, expected_types[i])

        input_ids = torch.randint(0, config.vocab_size, (1, 60), device=torch_device)

        with torch.no_grad():
            outputs = model(input_ids)

        self.assertEqual(outputs.last_hidden_state.shape, (1, 60, config.hidden_size))

    @require_torch_accelerator
    @slow
    def test_cwm_compile_static_cache(self):
        NUM_TOKENS_TO_GENERATE = 20

        config = CwmConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=256,
            sliding_window=64,
        )

        model = CwmForCausalLM(config)
        model.to(torch_device)
        model.eval()

        input_ids = torch.randint(1, 100, (1, 5), device=torch_device)

        generated_ids_dynamic = model.generate(input_ids, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)

        generated_ids_static = model.generate(
            input_ids, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )

        self.assertEqual(generated_ids_dynamic.shape, generated_ids_static.shape)
