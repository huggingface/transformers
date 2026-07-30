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

import unittest

from transformers.models.ovis2_5.convert_ovis2_5_weights_to_hf import (
    CHECKPOINT_SPECS,
    VISUAL_SPECIAL_TOKENS,
    _to_native_key,
    _validate_loading_info,
    build_native_config,
    detect_checkpoint_spec,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


def get_source_config(variant):
    spec = CHECKPOINT_SPECS[variant]
    hidden_size, num_layers, num_heads, num_key_value_heads, intermediate_size = spec.text_signature
    return {
        "model_type": "ovis2_5",
        "torch_dtype": "bfloat16",
        "visual_vocab_size": 65536,
        "llm_config": {
            "model_type": "qwen3",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_key_value_heads,
            "vocab_size": 151936,
            "hidden_act": "silu",
            "max_position_embeddings": 40960,
            "rope_theta": 1_000_000,
            "rms_norm_eps": 1e-6,
            "head_dim": 128,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "pad_token_id": None,
            "tie_word_embeddings": variant == "2b",
        },
        "vit_config": {
            "model_type": "siglip2_navit",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_patches": -1,
            "image_size": 512,
            "patch_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "hidden_stride": 2,
            "fullatt_block_indexes": None,
            "temporal_patch_size": 1,
            "window_size": 112,
            "preserve_original_pe": True,
            "use_rope": True,
        },
    }


class Ovis2_5ConversionTest(unittest.TestCase):
    def test_detects_both_released_layouts(self):
        self.assertEqual(detect_checkpoint_spec(get_source_config("2b")).variant, "2b")
        self.assertEqual(detect_checkpoint_spec(get_source_config("9b")).variant, "9b")
        self.assertEqual(detect_checkpoint_spec(get_source_config("2b")).max_pixels, 1344 * 1792)
        self.assertEqual(detect_checkpoint_spec(get_source_config("9b")).max_pixels, 1792 * 1792)

    def test_builds_native_config_without_remote_code_metadata(self):
        for variant in ("2b", "9b"):
            with self.subTest(variant=variant):
                source_config = get_source_config(variant)
                source_config["auto_map"] = {"AutoModel": "modeling_ovis2_5.Ovis"}
                source_config["llm_config"]["auto_map"] = {"AutoConfig": "configuration_qwen3.Qwen3Config"}
                config = build_native_config(source_config)

                self.assertIsInstance(config.text_config, Qwen3Config)
                self.assertEqual(config.vision_config.model_type, "ovis2_5_vision")
                self.assertEqual(
                    [
                        config.visual_atom_token_id,
                        config.image_start_token_id,
                        config.image_end_token_id,
                        config.video_start_token_id,
                        config.video_end_token_id,
                    ],
                    [token_id for _, token_id in VISUAL_SPECIAL_TOKENS],
                )
                self.assertNotIn("auto_map", config.to_dict())

    def test_all_six_source_prefixes_map_to_native_names(self):
        source_and_native_keys = {
            "llm.model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
            "llm.lm_head.weight": "lm_head.weight",
            "visual_tokenizer.vit.vision_model.embeddings.patch_embedding.weight": (
                "model.vision_tower.transformer.embeddings.patch_embedding.weight"
            ),
            "visual_tokenizer.head.0.weight": "model.vision_tower.head_linear.weight",
            "visual_tokenizer.head.1.weight": "model.vision_tower.head_norm.weight",
            "vte.weight": "model.visual_embeddings_table.weight",
        }
        for source_key, native_key in source_and_native_keys.items():
            with self.subTest(source_key=source_key):
                self.assertEqual(_to_native_key(source_key), native_key)

    def test_loading_info_only_allows_the_tied_2b_head(self):
        tied_config = build_native_config(get_source_config("2b"))
        _validate_loading_info(
            {
                "missing_keys": ["lm_head.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            tied_config,
        )

        untied_config = build_native_config(get_source_config("9b"))
        with self.assertRaisesRegex(ValueError, "missing=.*lm_head.weight"):
            _validate_loading_info(
                {
                    "missing_keys": ["lm_head.weight"],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                },
                untied_config,
            )


if __name__ == "__main__":
    unittest.main()
