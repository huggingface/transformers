# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniMax-M3-VL model."""

import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLVisionConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLVisionRotaryEmbedding,
        apply_rotary_pos_emb_vision,
    )
    from transformers.vision_utils import get_vision_position_ids


@require_torch
class MiniMaxM3VLVisionRotaryTest(unittest.TestCase):
    @parameterized.expand([(8, 2), (80, 26)])
    def test_three_axes_frequency_ladder_and_partial_rotation(self, head_dim, axis_dim):
        config = MiniMaxM3VLVisionConfig(
            hidden_size=2 * head_dim,
            num_attention_heads=2,
            spatial_merge_size=2,
            rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
        )
        rotary = MiniMaxM3VLVisionRotaryEmbedding(config)
        positions = get_vision_position_ids(torch.tensor([[2, 2, 2]]), 2, include_temporal=True)
        expected_positions = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        torch.testing.assert_close(positions, expected_positions)
        cosine, sine = rotary(torch.empty(8, head_dim), positions)
        bands = axis_dim // 2
        ladder = torch.logspace(0, -(bands - 1) / bands, bands, base=10000.0)
        angles = torch.cat([expected_positions[:, axis, None] * ladder for axis in range(3)], dim=-1).repeat(1, 2)
        self.assertEqual(cosine.shape, (8, 3 * axis_dim))
        self.assertEqual(sine.shape, (8, 3 * axis_dim))
        torch.testing.assert_close(cosine, angles.cos())
        torch.testing.assert_close(sine, angles.sin())
        for row in (1, 2, 4):
            self.assertFalse(torch.equal(sine[0], sine[row]))

        query = torch.randn(1, 8, 2, head_dim)
        key = torch.randn_like(query)
        rotated_query, rotated_key = apply_rotary_pos_emb_vision(query, key, cosine, sine)
        torch.testing.assert_close(rotated_query[..., 3 * axis_dim :], query[..., 3 * axis_dim :], rtol=0, atol=0)
        torch.testing.assert_close(rotated_key[..., 3 * axis_dim :], key[..., 3 * axis_dim :], rtol=0, atol=0)
        self.assertFalse(torch.equal(rotated_query[..., : 3 * axis_dim], query[..., : 3 * axis_dim]))
        self.assertEqual(rotary.state_dict(), {})
