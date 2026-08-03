# Copyright 2026 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniCPM model."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from transformers import AutoConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import MiniCPMConfig, MiniCPMForCausalLM, MiniCPMModel
    from transformers.models.minicpm.modeling_minicpm import MiniCPMRotaryEmbedding, apply_rotary_pos_emb


class MiniCPMModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniCPMModel


@require_torch
class MiniCPMModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    _torch_compile_train_cls = MiniCPMForCausalLM if is_torch_available() else None

    def test_embedding_scaling(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = MiniCPMModel(config).to(torch_device)

        with torch.no_grad():
            model.embed_tokens.weight.fill_(1)
            embeddings = model.embed_tokens(inputs["input_ids"])

        torch.testing.assert_close(embeddings, torch.full_like(embeddings, config.scale_emb))

    def test_residual_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = MiniCPMModel(config)
        expected_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)

        for layer in model.layers:
            self.assertEqual(layer.residual_scale, expected_scale)

    def test_logits_scaling(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = MiniCPMForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            hidden_states = model.model(**inputs).last_hidden_state
            expected_logits = model.lm_head(hidden_states / config.logits_scaling)
            logits = model(**inputs).logits

        torch.testing.assert_close(logits, expected_logits)

    def test_rope_uses_float32(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        rotary_emb = MiniCPMRotaryEmbedding(config)
        position_ids = torch.arange(3).unsqueeze(0)
        cos, sin = rotary_emb(torch.empty((), dtype=torch.bfloat16), position_ids)

        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)

        query = torch.randn(1, config.num_attention_heads, 3, config.head_dim).to(torch.bfloat16)
        key = torch.randn(1, config.num_key_value_heads, 3, config.head_dim).to(torch.bfloat16)
        rotated_query, rotated_key = apply_rotary_pos_emb(query, key, cos, sin)

        def rotate_half(hidden_states):
            first_half, second_half = hidden_states.chunk(2, dim=-1)
            return torch.cat((-second_half, first_half), dim=-1)

        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        expected_query = (query.float() * cos + rotate_half(query.float()) * sin).to(query.dtype)
        expected_key = (key.float() * cos + rotate_half(key.float()) * sin).to(key.dtype)
        torch.testing.assert_close(rotated_query, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(rotated_key, expected_key, rtol=0, atol=0)

    def test_longrope_matches_upstream_operation_order(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.max_position_embeddings = 8
        factor_count = config.head_dim // 2
        short_factors = [0.9977997, 1.0146583, 1.0349680, 1.0594292]
        long_factors = [1.1, 1.2, 1.3, 1.4]
        config.rope_parameters = {
            "rope_type": "longrope",
            "rope_theta": 10_000.0,
            "short_factor": [short_factors[index % len(short_factors)] for index in range(factor_count)],
            "long_factor": [long_factors[index % len(long_factors)] for index in range(factor_count)],
            "original_max_position_embeddings": 8,
            "factor": 1.0,
            "attention_factor": 1.0,
        }
        rotary_emb = MiniCPMRotaryEmbedding(config)
        position_ids = torch.tensor([[0, 1, 7]])
        cos, sin = rotary_emb(torch.empty((), dtype=torch.bfloat16), position_ids)

        ext_factors = torch.tensor(config.rope_parameters["short_factor"], dtype=torch.float32)
        inv_freq_shape = torch.arange(0, config.head_dim, 2, dtype=torch.int64).float() / config.head_dim
        base_inv_freq = 1.0 / (config.rope_parameters["rope_theta"] ** inv_freq_shape)
        freqs = position_ids.float().unsqueeze(-1) * (1.0 / ext_factors)
        freqs = freqs * base_inv_freq
        embeddings = torch.cat((freqs, freqs), dim=-1)

        torch.testing.assert_close(cos, embeddings.cos(), rtol=0, atol=0)
        torch.testing.assert_close(sin, embeddings.sin(), rtol=0, atol=0)

    def test_logits_are_float32(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = MiniCPMForCausalLM(config).to(device=torch_device, dtype=torch.bfloat16).eval()

        with torch.no_grad():
            logits = model(**inputs).logits

        self.assertEqual(logits.dtype, torch.float32)

    def test_sparse_attention_is_not_silently_ignored(self):
        config = MiniCPMConfig(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            sparse_config={"block_size": 64},
        )

        with self.assertRaisesRegex(NotImplementedError, "sparse attention is not implemented"):
            MiniCPMModel(config)

    def test_checkpoint_weight_names(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = MiniCPMForCausalLM(config)
        expected_keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}

        for layer_idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            expected_keys.update(
                {
                    f"{prefix}.input_layernorm.weight",
                    f"{prefix}.post_attention_layernorm.weight",
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.down_proj.weight",
                }
            )

        self.assertEqual(set(model.state_dict()), expected_keys)

    def test_config_without_model_type_uses_native_class(self):
        checkpoint_config = {
            "architectures": ["MiniCPMForCausalLM"],
            "auto_map": {"AutoConfig": "configuration_minicpm.MiniCPMConfig"},
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(checkpoint_config))
            config = AutoConfig.from_pretrained(directory, trust_remote_code=False)

        self.assertEqual(type(config).__name__, "MiniCPMConfig")
        self.assertTrue(config.tie_word_embeddings)
        self.assertIsNone(config.pad_token_id)
        self.assertIsNone(config.mup_denominator)
