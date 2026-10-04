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
from unittest.mock import patch

from parameterized import parameterized

from transformers.testing_utils import cleanup, is_torch_available, require_torch, torch_device


if is_torch_available():
    import torch

    from tests.heterogeneity.testing_utils import build_model, tiny_llama4_config, tiny_llama_config
    from transformers import DynamicCache, LlamaForCausalLM, ModernBertConfig, ModernBertModel, StaticCache
    from transformers.integrations.heterogeneity import (
        HeterogeneousModelingSpec,
        LayerIdxFromArgument,
        ReturnEntry,
        get_skip_replacement_factory,
    )
    from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByLayerIdx
    from transformers.masking_utils import (
        create_bidirectional_mask,
        create_bidirectional_sliding_window_mask,
        create_causal_mask,
        create_chunked_causal_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.models.modernbert.modeling_modernbert import ModernBertAttention, ModernBertEncoderLayer


@require_torch
class TestHeterogeneousMasking(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @parameterized.expand([("causal", True), ("bidirectional", False)])
    def test_sliding_window_masks_are_keyed_by_layer_idx(self, _name, is_causal):
        config = tiny_llama_config(
            sliding_window=None,
            is_causal=is_causal,
            per_layer_config={
                0: {"sliding_window": 2, "intermediate_size": 64},
                1: {"sliding_window": 2, "intermediate_size": 96},
                2: {"sliding_window": 3},
            },
        )
        config._attn_implementation = "sdpa"
        config._heterogeneity_spec.model_layer_configs = dict(enumerate(config.per_layer_config))

        inputs_embeds = torch.randn(1, 4, config.hidden_size)
        create_mask = create_sliding_window_causal_mask if is_causal else create_bidirectional_sliding_window_mask

        mask = create_mask(config, inputs_embeds, attention_mask=None, past_key_values=None)
        self.assertIsInstance(mask, AttentionMasksByLayerIdx)
        expected_masks = {
            0: torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, True, True, False],
                    [False, False, True, True],
                ]
            ),
            2: torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [False, True, True, True],
                ]
            ),
        }
        if not is_causal:
            expected_masks = {
                0: torch.tensor(
                    [
                        [True, True, True, False],
                        [True, True, True, True],
                        [True, True, True, True],
                        [False, True, True, True],
                    ]
                ),
                2: torch.ones(4, 4, dtype=torch.bool),
            }
        self.assertEqual(set(mask), {0, 1, 2})
        for layer_idx, expected_mask_idx in enumerate((0, 0, 2)):
            torch.testing.assert_close(mask[layer_idx], expected_masks[expected_mask_idx][None, None])

    @parameterized.expand([("full_attention", {0, 2}), ("sliding_attention", {1, 3})])
    def test_bidirectional_masks_respect_layer_types(self, layer_type, expected_layer_indices):
        config = tiny_llama_config(
            is_causal=False,
            sliding_window=2,
            layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
            per_layer_config={3: {"sliding_window": 3}},
        )
        config._attn_implementation = "sdpa"
        config._heterogeneity_spec.model_layer_configs = dict(enumerate(config.per_layer_config))
        create_mask = (
            create_bidirectional_mask if layer_type == "full_attention" else create_bidirectional_sliding_window_mask
        )

        mask = create_mask(
            config,
            inputs_embeds=torch.randn(1, 4, config.hidden_size),
            attention_mask=None,
        )

        self.assertEqual(set(mask), expected_layer_indices)

    @parameterized.expand([("causal", True), ("bidirectional", False)])
    def test_sliding_masks_use_each_layer_cache_geometry(self, _name, is_causal):
        config = tiny_llama_config(
            num_hidden_layers=2,
            sliding_window=None,
            is_causal=is_causal,
            per_layer_config={
                0: {"sliding_window": 3},
                1: {"sliding_window": 5},
            },
        )
        config._attn_implementation = "eager"
        config._heterogeneity_spec.model_layer_configs = dict(enumerate(config.per_layer_config))
        cache = DynamicCache(config=config)
        states = torch.randn(1, config.num_key_value_heads, 6, config.head_dim)
        for layer_idx in range(config.num_hidden_layers):
            cache.update(states, states, layer_idx=layer_idx)

        mask = create_sliding_window_causal_mask(
            config,
            inputs_embeds=torch.randn(1, 1, config.hidden_size),
            attention_mask=None,
            past_key_values=cache,
            allow_is_causal_skip=False,
        )

        self.assertEqual(mask[0].shape[-1], 3)
        self.assertEqual(mask[1].shape[-1], 5)

    def test_mask_reuse_does_not_recompile_when_cache_grows(self):
        config = tiny_llama_config(
            num_hidden_layers=2,
            sliding_window=None,
            per_layer_config={0: {"sliding_window": 16}, 1: {"sliding_window": 16}},
        )
        config._attn_implementation = "eager"
        config._heterogeneity_spec.model_layer_configs = dict(enumerate(config.per_layer_config))
        cache = StaticCache(config=config, max_cache_len=32)
        states = torch.randn(1, config.num_key_value_heads, 1, config.head_dim)
        inputs_embeds = torch.randn(1, 1, config.hidden_size)
        # Start from a clean compile cache, since other tests compile the same mask wrapper
        torch._dynamo.reset()
        # Without `dynamic=True`, the first change in cache length would recompile once, which is expected
        create_mask = torch.compile(create_sliding_window_causal_mask, backend="eager", dynamic=True)

        # Static sliding window layers track their length as a Python int, which grows with every update
        with torch._dynamo.config.patch(error_on_recompile=True):
            for _ in range(3):
                for layer_idx in range(config.num_hidden_layers):
                    cache.update(states, states, layer_idx=layer_idx)
                mask = create_mask(config, inputs_embeds, attention_mask=None, past_key_values=cache)

        self.assertIs(mask[0], mask[1])

    def test_static_cache_layers_share_masks_when_a_layer_skips_attention(self):
        config = tiny_llama_config(num_hidden_layers=3, per_layer_config={0: {"skip": ["attention"]}})
        config._attn_implementation = "eager"
        build_model(config, LlamaForCausalLM)
        # Once a static layer is written to, it reports its position as a device tensor
        cache = StaticCache(config=config, max_cache_len=8)
        states = torch.randn(1, config.num_key_value_heads, 2, config.head_dim)
        # Layer 0's attention is skipped, so nothing writes to its cache
        for layer_idx in (1, 2):
            cache.update(states, states, layer_idx=layer_idx)

        mask = create_causal_mask(config, torch.randn(1, 1, config.hidden_size), None, cache)

        # The layers that write to their caches stay at the same position, so they share a mask
        self.assertIs(mask[1], mask[2])
        self.assertIsNone(mask[0])

    def test_encoder_layer_with_skipped_attention_gets_no_mask(self):
        # ModernBERT's attention receives the attention mask but no cache
        spec = HeterogeneousModelingSpec(
            layer_cls=ModernBertEncoderLayer,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
            skip_descriptors={
                "attention": {
                    "attn": get_skip_replacement_factory(
                        ModernBertAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                    )
                }
            },
        )
        config = ModernBertConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            per_layer_config={0: {"skip": ["attention"]}},
        )
        with patch.object(ModernBertModel, "_heterogeneous_modeling_spec", spec, create=True):
            ModernBertModel(config)
        attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])

        mask = create_bidirectional_mask(config, torch.randn(2, 4, config.hidden_size), attention_mask)

        self.assertIsNone(mask[0])
        self.assertIsNotNone(mask[1])

    def test_attention_masks_round_trip_through_serialized_pytree(self):
        # With a compileable cache, `generate` passes the per-layer masks to the forward, so `torch.export` takes them
        # as an input, and `torch.export.load` rebuilds them from their serialized tree structure
        masks = AttentionMasksByLayerIdx({0: torch.ones(1, 1, 2, 2), 1: None})

        leaves, tree_spec = torch.utils._pytree.tree_flatten(masks)
        serialized_tree_spec = torch.utils._pytree.treespec_dumps(tree_spec)
        restored_masks = torch.utils._pytree.tree_unflatten(
            leaves, torch.utils._pytree.treespec_loads(serialized_tree_spec)
        )

        self.assertIsInstance(restored_masks, AttentionMasksByLayerIdx)
        self.assertIs(restored_masks[0], masks[0])
        self.assertIsNone(restored_masks[1])

    def test_chunked_attention_masks_are_keyed_by_layer_idx(self):
        config = tiny_llama4_config(
            attention_chunk_size=3,
            per_layer_config={
                0: {"attention_chunk_size": 2},
                2: {"attention_chunk_size": 2},
            },
        )
        config._attn_implementation = "sdpa"
        config._heterogeneity_spec.model_layer_configs = dict(enumerate(config.per_layer_config))

        inputs_embeds = torch.randn(1, 4, config.hidden_size)
        cache = DynamicCache(config=config)

        mask = create_chunked_causal_mask(config, inputs_embeds, attention_mask=None, past_key_values=cache)
        self.assertIsInstance(mask, AttentionMasksByLayerIdx)
        expected_masks = {
            0: torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, False, True, False],
                    [False, False, True, True],
                ]
            ),
            1: torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [False, False, False, True],
                ]
            ),
        }
        self.assertEqual(set(mask), {0, 1, 2})
        for layer_idx, expected_mask_idx in enumerate((0, 1, 0)):
            torch.testing.assert_close(mask[layer_idx], expected_masks[expected_mask_idx][None, None])
