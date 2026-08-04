# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import torch

from transformers.models.tiny_model.convert_tiny_model_weights_to_hf import _convert_state_dict


def make_original_state_dict(num_hidden_layers=2):
    def tensor(shape, offset):
        numel = torch.Size(shape).numel()
        return (torch.arange(numel, dtype=torch.float32).reshape(shape) + offset).to(torch.bfloat16)

    state_dict = {
        "embed.weight": tensor((10_000, 8), 1),
        "pos_embed": tensor((1, 8, 8), 2),
        "lm_head.weight": tensor((10_000, 8), 3),
        "lm_head.bias": tensor((10_000,), 4),
    }
    for layer_idx in range(num_hidden_layers):
        prefix = f"torso.{layer_idx}"
        state_dict.update(
            {
                f"{prefix}.attn.Q.weight": tensor((8, 8), 10 + layer_idx),
                f"{prefix}.attn.K.weight": tensor((8, 8), 20 + layer_idx),
                f"{prefix}.attn.V.weight": tensor((8, 8), 30 + layer_idx),
                f"{prefix}.attn.O.weight": tensor((8, 8), 40 + layer_idx),
                f"{prefix}.attn.O.bias": tensor((8,), 50 + layer_idx),
                f"{prefix}.mlp.read_in.weight": tensor((32, 8), 60 + layer_idx),
                f"{prefix}.mlp.read_in.bias": tensor((32,), 70 + layer_idx),
                f"{prefix}.mlp.write_out.weight": tensor((8, 32), 80 + layer_idx),
                f"{prefix}.mlp.write_out.bias": tensor((8,), 90 + layer_idx),
            }
        )
    return state_dict


class TinyModelStateDictConversionTest(unittest.TestCase):
    def test_infers_two_and_four_layer_configs(self):
        for num_hidden_layers in (2, 4):
            with self.subTest(num_hidden_layers=num_hidden_layers):
                config, converted = _convert_state_dict(
                    make_original_state_dict(num_hidden_layers),
                    num_attention_heads=2,
                    expected_num_hidden_layers=num_hidden_layers,
                )

                self.assertEqual(config.vocab_size, 10_000)
                self.assertEqual(config.hidden_size, 8)
                self.assertEqual(config.intermediate_size, 32)
                self.assertEqual(config.num_hidden_layers, num_hidden_layers)
                self.assertEqual(config.num_attention_heads, 2)
                self.assertEqual(config.max_position_embeddings, 8)
                self.assertEqual(len(converted), 4 + 9 * num_hidden_layers)

    def test_maps_every_tensor_without_transposing(self):
        original = make_original_state_dict()
        _, converted = _convert_state_dict(original, num_attention_heads=2)

        torch.testing.assert_close(
            converted["model.embed_positions.weight"], original["pos_embed"].squeeze(0), rtol=0, atol=0
        )
        torch.testing.assert_close(
            converted["model.layers.0.self_attn.q_proj.weight"],
            original["torso.0.attn.Q.weight"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            converted["model.layers.1.mlp.fc2.weight"],
            original["torso.1.mlp.write_out.weight"],
            rtol=0,
            atol=0,
        )

    def test_rejects_missing_and_unexpected_keys(self):
        original = make_original_state_dict()
        del original["torso.1.attn.O.bias"]
        original["torso.0.layer_norm.weight"] = torch.zeros(8, dtype=torch.bfloat16)

        with self.assertRaisesRegex(ValueError, "torso.1.attn.O.bias.*torso.0.layer_norm.weight"):
            _convert_state_dict(original, num_attention_heads=2)

    def test_rejects_noncontiguous_layers(self):
        original = make_original_state_dict()
        original = {key.replace("torso.1.", "torso.2."): value for key, value in original.items()}

        with self.assertRaisesRegex(ValueError, "contiguous from zero"):
            _convert_state_dict(original, num_attention_heads=2)

    def test_rejects_wrong_dtype_and_shape(self):
        original = make_original_state_dict()
        original["lm_head.bias"] = original["lm_head.bias"].float()
        with self.assertRaisesRegex(ValueError, "bfloat16.*lm_head.bias"):
            _convert_state_dict(original, num_attention_heads=2)

        original = make_original_state_dict()
        original["torso.0.attn.Q.weight"] = torch.zeros((8, 7), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, r"torso\.0\.attn\.Q\.weight.*\(8, 8\).*\(8, 7\)"):
            _convert_state_dict(original, num_attention_heads=2)
