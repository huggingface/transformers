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

    from transformers import MiniCPMForCausalLM, MiniCPMModel


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
