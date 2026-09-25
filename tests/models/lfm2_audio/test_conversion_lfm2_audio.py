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

"""Tests for one-time conversion of the original Liquid Audio checkpoint."""

import unittest

from transformers import Lfm2AudioConfig
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.models.lfm2_audio.convert_lfm2_audio_to_hf import convert_encoder_config, convert_state_dict


@require_torch
class Lfm2AudioConversionTest(unittest.TestCase):
    def test_depth_attention_conversion(self):
        config = Lfm2AudioConfig(depthformer={"dim": 16, "num_attention_heads": 4, "num_key_value_heads": 2})
        packed = torch.arange(32 * 16).reshape(32, 16).float()
        norm = torch.arange(4).float()
        result = convert_state_dict(
            {
                "depthformer.layers.0.operator.qkv_proj.weight": packed,
                "depthformer.layers.0.operator.bounded_attention.q_layernorm.weight": norm,
                "depthformer.layers.0.operator.out_proj.weight": torch.eye(16),
                "codebook_offsets": torch.arange(8),
            },
            config,
        )
        prefix = "model.depthformer.layers.0.operator."
        torch.testing.assert_close(result[prefix + "q_proj.weight"], packed[:16])
        torch.testing.assert_close(result[prefix + "k_proj.weight"], packed[16:24])
        torch.testing.assert_close(result[prefix + "v_proj.weight"], packed[24:])
        torch.testing.assert_close(result[prefix + "q_norm.weight"], norm)
        torch.testing.assert_close(result[prefix + "o_proj.weight"], torch.eye(16))
        self.assertNotIn("model.codebook_offsets", result)

    def test_audio_embedding_conversion_removes_unused_projections(self):
        weight = torch.arange(32).reshape(8, 4).float()
        converted = convert_state_dict(
            {
                "audio_embedding.embedding.weight": weight,
                "audio_embedding.embedding_norm.weight": torch.ones(4),
                "audio_embedding.to_logits.weight": weight.clone(),
            },
            Lfm2AudioConfig(),
        )
        self.assertEqual(set(converted), {"model.audio_embedding.weight"})
        torch.testing.assert_close(converted["model.audio_embedding.weight"], weight)

    def test_encoder_configuration(self):
        encoder = {"d_model": 16, "n_layers": 2, "n_heads": 4, "feat_in": 8, "xscaling": False}
        config = convert_encoder_config(encoder)
        self.assertEqual(config.hidden_size, 16)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.num_mel_bins, 8)
        self.assertFalse(config.scale_input)
        self.assertEqual(config.layerdrop, 0.0)
        with self.assertRaises(ValueError):
            convert_encoder_config({**encoder, "reduction": "pooling"})

    def test_encoder_and_adapter_weight_names(self):
        weight = torch.ones(2, 2)
        result = convert_state_dict(
            {
                "conformer.pre_encode.conv.0.weight": weight,
                "conformer.layers.0.self_attn.linear_q.weight": weight,
                "conformer.layers.0.conv.batch_norm.running_mean": weight[0],
                "audio_adapter.model.1.weight": weight,
            },
            Lfm2AudioConfig(),
        )
        self.assertEqual(
            set(result),
            {
                "model.conformer.subsampling.layers.0.weight",
                "model.conformer.layers.0.self_attn.q_proj.weight",
                "model.conformer.layers.0.conv.norm.running_mean",
                "model.audio_adapter_linear_1.weight",
            },
        )
