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

from transformers.testing_utils import cleanup, is_torch_available, require_torch, torch_device


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
    from transformers.cache_utils import DynamicSlidingWindowLayer, StaticSlidingWindowLayer
    from transformers.integrations.executorch import export_with_dynamic_cache


@require_torch
class TestHeterogeneousCache(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

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

    def test_static_cache_generation_with_layer_zero_attention_skipped(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={0: {"skip": ["attention"]}},
        )
        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM)
        input_ids = torch.tensor([[1, 2]], device=model.device)
        generation_kwargs = {"max_new_tokens": 2, "do_sample": False, "disable_compile": True}

        expected_ids = model.generate(input_ids, cache_implementation="dynamic", **generation_kwargs)
        actual_ids = model.generate(input_ids, cache_implementation="static", **generation_kwargs)

        torch.testing.assert_close(actual_ids, expected_ids)

    @pytest.mark.torch_compile_test
    def test_static_cache_compiles_once_with_layer_zero_attention_skipped(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={0: {"skip": ["xyz"]}},
        )
        with hetero_context("llama") as modeling_spec:
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

    def test_static_cache_with_skips_requires_model_construction(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={0: {"skip": ["attention"]}},
        )

        with self.assertRaisesRegex(ValueError, "Construct the model before creating a `StaticCache`"):
            StaticCache(config=config, max_cache_len=1)

        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM)
        cache = StaticCache(config=model.config, max_cache_len=1)
        self.assertEqual(cache.get_representative_kv_layer_idx(range(config.num_hidden_layers)), 1)

    @pytest.mark.torch_export_test
    def test_dynamic_cache_export_preserves_skipped_layer_indices(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={0: {"skip": ["attention"]}},
        )
        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM)

        input_ids = torch.tensor([[1, 2]], device=model.device)
        attention_mask = torch.ones_like(input_ids)
        exported_program = export_with_dynamic_cache(model, input_ids, attention_mask)

        with torch.no_grad():
            exported_outputs = exported_program.module()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=DynamicCache(config=config),
                use_cache=True,
            )
            eager_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=DynamicCache(config=config),
                use_cache=True,
            )

        torch.testing.assert_close(exported_outputs.logits, eager_outputs.logits)
        exported_cache = exported_outputs.past_key_values
        eager_cache = eager_outputs.past_key_values
        self.assertEqual(len(exported_cache.layers), config.num_hidden_layers)
        self.assertIsNone(exported_cache.layers[0].keys)
        self.assertIsNone(exported_cache.layers[0].values)
        torch.testing.assert_close(exported_cache.layers[1].keys, eager_cache.layers[1].keys)
        torch.testing.assert_close(exported_cache.layers[1].values, eager_cache.layers[1].values)
        self.assertEqual(exported_cache.get_seq_length(layer_idx=0), 0)
        self.assertEqual(exported_cache.get_seq_length(), input_ids.shape[1])
        self.assertEqual(exported_cache.get_representative_kv_layer_idx(range(config.num_hidden_layers)), 1)

    def test_preloaded_cache_uses_populated_layer(self):
        empty_states = torch.empty(1, 1, 0, 4)
        populated_states = torch.randn(1, 1, 3, 4)

        cache = DynamicCache([(empty_states, empty_states), (populated_states, populated_states)])

        self.assertEqual(cache.get_seq_length(), 3)
        self.assertEqual(cache.get_representative_kv_layer_idx([0, 1]), 1)

    def test_dynamic_cache_heterogeneous_sliding_window(self):
        """DynamicCache should create sliding layers matching per-layer sliding_window."""
        config = tiny_llama_config(
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
        config = tiny_llama_config(
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
