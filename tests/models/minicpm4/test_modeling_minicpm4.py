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
"""Testing suite for the PyTorch MiniCPM4 model."""

import json
import tempfile
import unittest
from pathlib import Path

from transformers import AutoConfig, is_torch_available
from transformers.testing_utils import require_torch, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer, MiniCPM4Config, MiniCPM4ForCausalLM, MiniCPM4Model


class MiniCPM4ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = MiniCPM4Model

    def __init__(
        self,
        parent,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        **kwargs,
    ):
        super().__init__(
            parent=parent,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            **kwargs,
        )


@require_torch
class MiniCPM4ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPM4ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    _torch_compile_train_cls = MiniCPM4ForCausalLM if is_torch_available() else None

    def test_legacy_config_identifiers_use_native_minicpm4_class(self):
        checkpoint_config = {
            "architectures": ["MiniCPMForCausalLM"],
            "auto_map": {"AutoConfig": "configuration_minicpm.MiniCPMConfig"},
            "vocab_size": 99,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "rope_scaling": {
                "rope_type": "longrope",
                "short_factor": [1.0] * 32,
                "long_factor": [1.0] * 32,
                "original_max_position_embeddings": 32768,
            },
            "scale_emb": 12,
            "scale_depth": 1.4,
            "dim_model_base": 256,
        }
        for legacy_model_type in (None, "minicpm"):
            with self.subTest(model_type=legacy_model_type), tempfile.TemporaryDirectory() as directory:
                if legacy_model_type is not None:
                    checkpoint_config["model_type"] = legacy_model_type
                else:
                    checkpoint_config.pop("model_type", None)
                Path(directory, "config.json").write_text(json.dumps(checkpoint_config))
                config = AutoConfig.from_pretrained(directory, trust_remote_code=False)

            self.assertIsInstance(config, MiniCPM4Config)
            self.assertEqual(config.model_type, "minicpm4")
            self.assertTrue(config.tie_word_embeddings)
            self.assertIsNone(config.pad_token_id)

    def test_bitcpm_config_is_not_routed_to_minicpm4(self):
        checkpoint_config = {
            "architectures": ["MiniCPMForCausalLM"],
            "auto_map": {"AutoConfig": "configuration_minicpm.MiniCPMConfig"},
            "hidden_size": 1536,
            "intermediate_size": 3840,
            "num_hidden_layers": 52,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "rope_scaling": {"rope_type": "longrope"},
            "scale_emb": 12,
            "scale_depth": 1.4,
            "dim_model_base": 256,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(checkpoint_config))
            with self.assertRaisesRegex(ValueError, "contains custom code"):
                AutoConfig.from_pretrained(directory, trust_remote_code=False)

    def test_older_minicpm_config_is_not_routed_to_minicpm4(self):
        checkpoint_config = {
            "architectures": ["MiniCPMForCausalLM"],
            "auto_map": {"AutoConfig": "configuration_minicpm.MiniCPMConfig"},
            "model_type": "minicpm",
            "hidden_size": 2304,
            "intermediate_size": 5760,
            "num_hidden_layers": 40,
            "num_attention_heads": 36,
            "num_key_value_heads": 36,
            "rope_scaling": None,
            "scale_emb": 12,
            "scale_depth": 1.4,
            "dim_model_base": 256,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(checkpoint_config))
            with self.assertRaisesRegex(ValueError, "contains custom code"):
                AutoConfig.from_pretrained(directory, trust_remote_code=False)


@slow
@require_torch
class MiniCPM4IntegrationTest(unittest.TestCase):
    model_id = "openbmb/MiniCPM4-0.5B"
    revision = "5253c7fcc5e29e1cf3eacb59a58adf1ba4df8630"
    checkpoint_configs = {
        "openbmb/MiniCPM4-0.5B": (
            "5253c7fcc5e29e1cf3eacb59a58adf1ba4df8630",
            {
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "max_position_embeddings": 32768,
                "tie_word_embeddings": True,
                "pad_token_id": None,
            },
        ),
        "openbmb/MiniCPM4-8B": (
            "bb2ae14cf59d4ca769c4e42ece54cc3b82a58ef7",
            {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "max_position_embeddings": 32768,
                "tie_word_embeddings": False,
                "pad_token_id": 2,
            },
        ),
        "openbmb/MiniCPM4.1-8B": (
            "3a8dfed9c79a45e07dbff95bcd49d792343fa1a3",
            {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "max_position_embeddings": 65536,
                "tie_word_embeddings": False,
                "pad_token_id": 2,
            },
        ),
    }

    def test_supported_checkpoint_configs(self):
        for model_id, (revision, expected_config) in self.checkpoint_configs.items():
            with self.subTest(model_id=model_id):
                config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)

            self.assertIsInstance(config, MiniCPM4Config)
            self.assertEqual(config.model_type, "minicpm4")
            for attribute, expected_value in expected_config.items():
                self.assertEqual(getattr(config, attribute), expected_value)

    def test_minicpm4_0_5b_logits(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
            dtype="auto",
        )
        self.assertIsInstance(model, MiniCPM4ForCausalLM)

        inputs = tokenizer("The capital of France is", return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.float()

        expected_logits = torch.tensor(
            [-5.375, 1.6328125, 5.78125, -6.5625, -6.0, 9.875, 1.4296875, -5.6875, -6.0625, -5.5625]
        )
        torch.testing.assert_close(logits[0, -1, :10].cpu(), expected_logits, rtol=1e-3, atol=1e-3)

    def test_minicpm4_0_5b_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
            dtype="auto",
        )
        inputs = tokenizer("The capital of France is", return_tensors="pt")
        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(
            generated_text,
            "The capital of France is Paris. Which of the following statements about Paris is true?\n"
            "[A]. Paris is the",
        )
