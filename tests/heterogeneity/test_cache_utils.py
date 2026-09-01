# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pytest

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    import torch
    from torch._dynamo.testing import CompileCounter

    from tests.heterogeneity.testing_utils import (
        build_model,
        dummy_input_ids,
        hetero_context,
        tiny_llama_config,
    )
    from transformers import DynamicCache, LlamaForCausalLM, StaticCache
    from transformers.generation import CompileConfig


@require_torch
class TestHeterogeneousCache(unittest.TestCase):
    def test_per_layer_kv_cache_shapes(self):
        """KV cache tensors should reflect per-layer num_key_value_heads after a cached forward pass."""
        config = tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 2: {"num_key_value_heads": 1}})
        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM)
        with torch.no_grad():
            cache = model(dummy_input_ids(), use_cache=True).past_key_values
        # keys shape: [batch, num_heads, seq_len, head_dim]
        self.assertEqual(cache.layers[0].keys.shape[1], 2)
        self.assertEqual(cache.layers[1].keys.shape[1], 4)  # default
        self.assertEqual(cache.layers[2].keys.shape[1], 1)
        self.assertEqual(cache.layers[3].keys.shape[1], 4)  # default

    def test_cached_decoding_matches_uncached_with_layer_zero_attention_skipped(self):
        # `get_seq_length()` used to default to layer 0, which has no KV state when its attention is skipped.
        # Model code calls it without a `layer_idx` to derive position IDs and mask offsets, so it returned zero
        # instead of the cached sequence length and broke cached decoding.
        # This test verifies that the fix restores correct behavior.
        config = tiny_llama_config(per_layer_config={0: {"skip": ["attention"]}})
        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM).eval()
        input_ids = torch.tensor([[0, 0, 1, 2, 3]], device=model.device)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], device=model.device)
        caches = (
            DynamicCache(config=config),
            StaticCache(config=config, max_cache_len=input_ids.shape[1]),
        )

        with torch.no_grad():
            expected_logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1]
            for cache in caches:
                with self.subTest(cache_type=type(cache).__name__):
                    outputs = model(
                        input_ids[:, :-1],
                        attention_mask=attention_mask[:, :-1],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    self.assertEqual(cache.get_seq_length(), input_ids.shape[1] - 1)
                    self.assertEqual(cache.get_seq_length(layer_idx=0), 0)

                    actual_logits = model(
                        input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=outputs.past_key_values,
                        use_cache=True,
                    ).logits[:, -1]
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-5)

    @pytest.mark.torch_compile_test
    def test_static_cache_generation_with_skipped_attention_compiles_fullgraph(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={0: {"skip": ["attention"]}},
        )
        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM)
        input_ids = torch.tensor([[1, 2]], device=model.device)
        generation_kwargs = {"max_new_tokens": 2, "do_sample": False}

        dynamic_ids = model.generate(
            input_ids, cache_implementation="dynamic", disable_compile=True, **generation_kwargs
        )
        static_ids = model.generate(
            input_ids, cache_implementation="static", disable_compile=True, **generation_kwargs
        )

        torch.compiler.reset()  # prevent cached compilation from being used in the test
        compile_config = CompileConfig(fullgraph=True, dynamic=False)  # Error out on graph breaks and dynamic shapes
        compile_config._compile_all_devices = True
        compiled_ids = model.generate(
            input_ids, cache_implementation="static", compile_config=compile_config, **generation_kwargs
        )

        self.assertTrue(hasattr(model, "_compiled_call"))
        torch.testing.assert_close(static_ids, dynamic_ids)
        torch.testing.assert_close(compiled_ids, dynamic_ids)

    @pytest.mark.torch_compile_test
    def test_static_cache_compiles_once_with_custom_kv_cache_updater_skip(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={0: {"skip": ["xyz"]}},
        )
        with hetero_context("llama") as modeling_spec:
            # Use a non-semantic name to verify cache handling follows the descriptor metadata.
            modeling_spec.skip_descriptors["xyz"] = modeling_spec.skip_descriptors.pop("attention")
            model = build_model(config, LlamaForCausalLM)
        input_ids = torch.tensor([[1, 2]], device=model.device)
        cache = StaticCache(config=config, max_cache_len=input_ids.shape[1])

        cache.early_initialization(
            batch_size=input_ids.shape[0],
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=model.dtype,
            device=model.device,
        )
        torch.compiler.reset()
        compile_counter = CompileCounter()

        def cached_forward(current_input_ids):
            return model(current_input_ids, past_key_values=cache, use_cache=True).logits

        compiled_forward = torch.compile(cached_forward, backend=compile_counter, fullgraph=True)

        with torch.no_grad():
            expected_logits = model(input_ids, use_cache=False).logits
            actual_logits = torch.cat(
                [compiled_forward(input_ids[:, position : position + 1]) for position in range(input_ids.shape[1])],
                dim=1,
            )

        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-5)
        self.assertEqual(cache.get_seq_length(), input_ids.shape[1])
        self.assertEqual(cache.get_seq_length(layer_idx=0), 0)
        self.assertEqual(compile_counter.frame_count, 1)

    def test_static_cache_keeps_layer_with_mlp_skip_enabled(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={0: {"skip": ["mlp"]}},
        )
        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM)
        cache = StaticCache(config=config, max_cache_len=1)

        with torch.no_grad():
            model(torch.tensor([[1]], device=model.device), past_key_values=cache, use_cache=True)

        self.assertEqual(cache.get_representative_kv_layer_idx(range(config.num_hidden_layers)), 0)
        self.assertEqual(cache.get_seq_length(layer_idx=0), 1)

    def test_static_cache_with_skips_requires_model_initialization(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={0: {"skip": ["attention"]}},
        )

        with self.assertRaisesRegex(ValueError, "Initialize a model from this config first"):
            StaticCache(config=config, max_cache_len=1)

        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM)
        cache = StaticCache(config=model.config, max_cache_len=1)
        self.assertEqual(cache.get_representative_kv_layer_idx(range(config.num_hidden_layers)), 1)
