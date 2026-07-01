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

from transformers.testing_utils import cleanup, is_torch_available, require_torch, torch_device


if is_torch_available():
    import torch

    from tests.heterogeneity.testing_utils import (
        _build_model,
        _dummy_input_ids,
        _hetero_context,
        _tiny_llama_config,
    )
    from transformers import DynamicCache, LlamaForCausalLM, StaticCache
    from transformers.cache_utils import DynamicSlidingWindowLayer, StaticSlidingWindowLayer


@require_torch
class TestHeterogeneousCache(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_per_layer_kv_cache_shapes(self):
        """KV cache tensors should reflect per-layer num_key_value_heads after a cached forward pass."""
        config = _tiny_llama_config(per_layer_config={0: {"num_key_value_heads": 2}, 2: {"num_key_value_heads": 1}})
        with _hetero_context("llama"):
            model = _build_model(config, LlamaForCausalLM)
        with torch.no_grad():
            cache = model(_dummy_input_ids(), use_cache=True).past_key_values
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
        config = _tiny_llama_config(per_layer_config={0: {"skip": ["attention"]}})
        with _hetero_context("llama"):
            model = _build_model(config, LlamaForCausalLM).eval()
        input_ids = torch.tensor([[0, 0, 1, 2, 3]], device=torch_device)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], device=torch_device)
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

    def test_preloaded_cache_uses_populated_layer(self):
        empty_states = torch.empty(1, 1, 0, 4)
        populated_states = torch.randn(1, 1, 3, 4)

        cache = DynamicCache([(empty_states, empty_states), (populated_states, populated_states)])

        self.assertEqual(cache.get_seq_length(), 3)
        self.assertEqual(cache.get_updated_kv_layer_idx([0, 1]), 1)

    def test_dynamic_cache_heterogeneous_sliding_window(self):
        """DynamicCache should create sliding layers matching per-layer sliding_window."""
        config = _tiny_llama_config(
            sliding_window=None, per_layer_config={0: {"sliding_window": 32}, 2: {"sliding_window": 16}}
        )
        layers = DynamicCache(config=config).layers

        self.assertEqual(len(layers), 4)
        self.assertIsInstance(layers[0], DynamicSlidingWindowLayer)
        self.assertEqual(layers[0].sliding_window, 32)
        self.assertFalse(layers[1].is_sliding)
        self.assertIsInstance(layers[2], DynamicSlidingWindowLayer)
        self.assertEqual(layers[2].sliding_window, 16)
        self.assertFalse(layers[3].is_sliding)

    def test_static_cache_heterogeneous_sliding_window(self):
        """StaticCache should create sliding layers for the right layers."""
        config = _tiny_llama_config(
            sliding_window=None, per_layer_config={1: {"sliding_window": 24}, 3: {"sliding_window": 48}}
        )
        layers = StaticCache(config=config, batch_size=1, max_cache_len=64).layers

        self.assertEqual(len(layers), 4)
        self.assertFalse(layers[0].is_sliding)
        self.assertIsInstance(layers[1], StaticSlidingWindowLayer)
        # StaticSlidingWindowLayer caps max_cache_len = min(sliding_window, max_cache_len)
        self.assertEqual(layers[1].max_cache_len, 24)
        self.assertFalse(layers[2].is_sliding)
        self.assertIsInstance(layers[3], StaticSlidingWindowLayer)
        self.assertEqual(layers[3].max_cache_len, 48)
