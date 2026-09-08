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

from parameterized import parameterized

from transformers.testing_utils import cleanup, is_torch_available, require_torch, torch_device


if is_torch_available():
    import torch

    from tests.heterogeneity.testing_utils import tiny_gpt_oss_config, tiny_llama4_config, tiny_llama_config
    from transformers import DynamicCache
    from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByLayerIdx
    from transformers.masking_utils import (
        create_chunked_causal_mask,
        create_masks_for_generate,
        create_sliding_window_causal_mask,
    )


@require_torch
class TestHeterogeneousMasking(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_sliding_window_masks_are_keyed_by_layer_idx(self):
        config = tiny_llama_config(
            sliding_window=None,
            per_layer_config={
                0: {"sliding_window": 2, "intermediate_size": 64},
                1: {"sliding_window": 2, "intermediate_size": 96},
                2: {"sliding_window": 3},
            },
        )
        config._attn_implementation = "sdpa"

        inputs_embeds = torch.randn(1, 4, 64)
        cache = DynamicCache(config=config)

        mask = create_sliding_window_causal_mask(config, inputs_embeds, attention_mask=None, past_key_values=cache)
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
        self.assertEqual(set(mask), {0, 1, 2})
        self.assertIs(mask[0], mask[1])
        for layer_idx, expected_mask in expected_masks.items():
            torch.testing.assert_close(mask[layer_idx], expected_mask[None, None])

    @parameterized.expand([("causal", True), ("bidirectional", False)])
    def test_sliding_window_mask_uses_updated_layer_with_matching_window(self, _name, is_causal):
        config = tiny_llama_config(
            sliding_window=None,
            is_causal=is_causal,
            per_layer_config={
                0: {"sliding_window": 3},
                1: {"sliding_window": 5},
                2: {"sliding_window": 3},
            },
        )
        config._attn_implementation = "eager"
        cache = DynamicCache(config=config)
        key_states = torch.randn(1, config.num_key_value_heads, 4, config.head_dim)
        cache.update(key_states, key_states, layer_idx=1)
        cache.update(key_states, key_states, layer_idx=2)

        mask = create_sliding_window_causal_mask(
            config,
            inputs_embeds=torch.randn(1, 1, config.hidden_size),
            attention_mask=torch.ones(1, 5),
            past_key_values=cache,
        )

        self.assertEqual(mask[0].shape[-1], 3)

    def test_chunked_attention_masks_are_keyed_by_layer_idx(self):
        config = tiny_llama4_config(
            attention_chunk_size=3,
            per_layer_config={
                0: {"attention_chunk_size": 2},
                2: {"attention_chunk_size": 2},
            },
        )
        config._attn_implementation = "sdpa"

        inputs_embeds = torch.randn(1, 4, 64)
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
        self.assertEqual(set(mask), {0, 1, 2, 3})
        for layer_idx, expected_mask_idx in enumerate((0, 1, 0, 1)):
            torch.testing.assert_close(mask[layer_idx], expected_masks[expected_mask_idx][None, None])

    def test_masks_for_generate_keys_layer_idx_masks_by_pattern(self):
        config = tiny_gpt_oss_config(
            layer_types=["sliding_attention"] * 4,
            per_layer_config={0: {"sliding_window": 16}, 1: {"sliding_window": 8}},
        )
        config._attn_implementation = "sdpa"

        inputs_embeds = torch.randn(1, 8, 64)
        attention_masks = create_masks_for_generate(
            config,
            inputs_embeds,
            attention_mask=torch.ones(1, 8),
            past_key_values=DynamicCache(config=config),
        )

        # Model code selects a layer's mask from the result by pattern name, then the layer resolves its own index.
        self.assertEqual(set(attention_masks), {"sliding_attention"})
        sliding_masks = attention_masks["sliding_attention"]
        self.assertIsInstance(sliding_masks, AttentionMasksByLayerIdx)
        self.assertEqual(set(sliding_masks), {0, 1, 2, 3})
