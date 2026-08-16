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

import json
import tempfile
import unittest
from pathlib import Path

from huggingface_hub.errors import StrictDataclassFieldValidationError

from transformers import AutoConfig, CLIPVisionConfig, MiniMaxConfig, MiniMaxVL01Config, MiniMaxVL01TextConfig


def get_legacy_text_config():
    return {
        "model_type": "minimax_text_01",
        "architectures": ["MiniMaxText01ForCausalLM"],
        "vocab_size": 97,
        "hidden_size": 32,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "num_local_experts": 2,
        "num_experts_per_tok": 2,
        "attn_type_list": [0, 1],
        "block_size": 4,
        "postnorm": True,
        "layernorm_full_attention_alpha": 3.5,
        "layernorm_full_attention_beta": 1.0,
        "layernorm_linear_attention_alpha": 3.25,
        "layernorm_linear_attention_beta": 0.75,
        "layernorm_mlp_alpha": 2.5,
        "layernorm_mlp_beta": 0.5,
        "rope_theta": 10_000_000.0,
        "rotary_dim": 4,
        "shared_intermediate_size": [0],
        "shared_moe_mode": "always",
    }


def get_vision_config_dict():
    return {
        "model_type": "clip_vision_model",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "image_size": 8,
        "patch_size": 4,
        "projection_dim": 32,
    }


class MiniMaxVL01ConfigTest(unittest.TestCase):
    def test_rejects_invalid_field_types(self):
        for invalid_field in (
            {"image_token_index": "bad"},
            {"ignore_index": "bad"},
            {"vision_feature_layer": 3.5},
        ):
            with self.subTest(invalid_field=invalid_field):
                with self.assertRaises(StrictDataclassFieldValidationError):
                    MiniMaxVL01Config(**invalid_field)

    def test_legacy_text_config_is_migrated_losslessly(self):
        config = MiniMaxVL01Config(
            text_config=get_legacy_text_config(),
            vision_config=get_vision_config_dict(),
            image_token_index=3,
            image_grid_pinpoints=[[8, 8]],
        )

        self.assertIsInstance(config.text_config, MiniMaxVL01TextConfig)
        self.assertIsInstance(config.vision_config, CLIPVisionConfig)
        self.assertEqual(config.text_config.layer_types, ["linear_attention", "full_attention"])
        self.assertEqual(config.text_config.full_attn_alpha_factor, 3.5)
        self.assertEqual(config.text_config.full_attn_beta_factor, 1.0)
        self.assertEqual(config.text_config.linear_attn_alpha_factor, 3.25)
        self.assertEqual(config.text_config.linear_attn_beta_factor, 0.75)
        self.assertEqual(config.text_config.mlp_alpha_factor, 2.5)
        self.assertEqual(config.text_config.mlp_beta_factor, 0.5)
        self.assertEqual(
            config.text_config.rope_parameters,
            {"rope_type": "default", "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.5},
        )
        self.assertEqual(config.text_config.architectures, ["MiniMaxVL01TextModel"])
        self.assertIsNone(config.text_config.bos_token_id)
        self.assertIsNone(config.text_config.eos_token_id)
        self.assertEqual(config.image_token_id, 3)
        self.assertEqual(config.image_token_index, 3)

        serialized_text_config = config.to_dict()["text_config"]
        legacy_only_keys = {
            "attn_type_list",
            "postnorm",
            "rotary_dim",
            "shared_intermediate_size",
            "shared_moe_mode",
        }
        self.assertTrue(legacy_only_keys.isdisjoint(serialized_text_config))
        self.assertEqual(serialized_text_config["model_type"], "minimax_vl_01_text")
        self.assertIsNone(serialized_text_config["bos_token_id"])
        self.assertIsNone(serialized_text_config["eos_token_id"])

    def test_auto_config_loads_a_legacy_local_manifest_without_remote_code(self):
        manifest = {
            "model_type": "minimax_vl_01",
            "architectures": ["MiniMaxVL01ForConditionalGeneration"],
            "auto_map": {
                "AutoConfig": "configuration_minimax_vl_01.MiniMaxVL01Config",
                "AutoModel": "modeling_minimax_vl_01.MiniMaxVL01ForConditionalGeneration",
            },
            "image_token_index": 3,
            "image_grid_pinpoints": [[8, 8]],
            "vision_feature_layer": -1,
            "vision_feature_select_strategy": "default",
            "text_config": get_legacy_text_config(),
            "vision_config": get_vision_config_dict(),
        }

        with tempfile.TemporaryDirectory() as tmpdirname:
            Path(tmpdirname, "config.json").write_text(json.dumps(manifest), encoding="utf-8")
            config = AutoConfig.from_pretrained(tmpdirname, trust_remote_code=False)

        self.assertIsInstance(config, MiniMaxVL01Config)
        self.assertEqual(config.text_config.model_type, "minimax_vl_01_text")
        self.assertEqual(config.text_config.layer_types, ["linear_attention", "full_attention"])

    def test_native_config_round_trip_uses_auto_config(self):
        config = MiniMaxVL01Config(
            text_config=MiniMaxConfig(
                vocab_size=97,
                hidden_size=32,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                num_local_experts=2,
                num_experts_per_tok=2,
                layer_types=["linear_attention", "full_attention"],
            ),
            vision_config=CLIPVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=8,
                patch_size=4,
            ),
            image_token_index=3,
            image_grid_pinpoints=[[8, 8]],
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            reloaded = AutoConfig.from_pretrained(tmpdirname)

        self.assertIsInstance(reloaded, MiniMaxVL01Config)
        expected = config.to_dict()
        observed = reloaded.to_dict()
        expected.pop("_name_or_path", None)
        observed.pop("_name_or_path", None)
        self.assertEqual(observed, expected)

    def test_unsupported_remote_code_branches_are_rejected(self):
        invalid_configs = [
            {"postnorm": False},
            {"shared_intermediate_size": [8]},
            {"attn_type_list": [0, 2]},
            {"rotary_dim": 3},
        ]
        for invalid_update in invalid_configs:
            with self.subTest(invalid_update=invalid_update):
                text_config = get_legacy_text_config()
                text_config.update(invalid_update)
                with self.assertRaises(ValueError):
                    MiniMaxVL01Config(text_config=text_config, vision_config=get_vision_config_dict())

        with self.assertRaises(ValueError):
            MiniMaxVL01Config(
                text_config=get_legacy_text_config(),
                vision_config=get_vision_config_dict(),
                vision_feature_select_strategy="full",
            )

        invalid_vision_config = get_vision_config_dict()
        invalid_vision_config["model_type"] = "siglip_vision_model"
        with self.assertRaises(TypeError):
            MiniMaxVL01Config(text_config=get_legacy_text_config(), vision_config=invalid_vision_config)

        invalid_text_config = {
            "model_type": "llama",
            "vocab_size": 97,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
        with self.assertRaises(TypeError):
            MiniMaxVL01Config(text_config=invalid_text_config, vision_config=get_vision_config_dict())


if __name__ == "__main__":
    unittest.main()
