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

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import (
    AutoConfig,
    MiMoV2VLAudioConfig,
    MiMoV2VLAudioTokenizerConfig,
    MiMoV2VLConfig,
    MiMoV2VLTextConfig,
    MiMoV2VLVisionConfig,
)


class MiMoV2VLConfigTest(unittest.TestCase):
    def test_defaults_match_the_released_checkpoints(self):
        config = MiMoV2VLConfig()
        text_config = config.text_config
        self.assertEqual(text_config.swa_num_key_value_heads, 2 * text_config.num_key_value_heads)
        self.assertEqual(text_config.rope_parameters["full_attention"]["rope_theta"], 10_000_000.0)
        self.assertEqual(text_config.rope_parameters["sliding_attention"]["rope_theta"], 10_000.0)
        self.assertEqual(config.vision_config.layer_types[:2], ["full_attention", "sliding_attention"])
        self.assertEqual(config.vision_config.layer_types[9], "full_attention")
        self.assertEqual(config.vision_config.patch_orders[:6], ["row_major"] * 5 + ["column_major"])
        self.assertEqual(config.audio_config.codebook_vocab_sizes, [1280] * config.audio_config.num_codebooks)
        self.assertEqual(config.audio_config.rope_parameters["rope_theta"], 640000.0)
        self.assertEqual(len(config.audio_tokenizer_config.codebook_sizes), 20)
        self.assertEqual(config.audio_tokenizer_config.layer_types[:2], ["sliding_attention", "full_attention"])
        self.assertEqual(config.pad_token_id, text_config.pad_token_id)
        self.assertEqual(config.eos_token_id, text_config.eos_token_id)

    def test_save_load_round_trip(self):
        config = MiMoV2VLConfig(
            text_config={"num_hidden_layers": 2, "sliding_window": 9},
            vision_config={"depth": 2},
            audio_config={"num_codebooks": 2, "codebook_vocab_sizes": [16, 32]},
            image_token_id=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = AutoConfig.from_pretrained(directory)
        self.assertIsInstance(restored, MiMoV2VLConfig)
        restored_dict, config_dict = restored.to_dict(), config.to_dict()
        restored_dict.pop("_name_or_path"), config_dict.pop("_name_or_path")
        self.assertEqual(restored_dict, config_dict)
        self.assertEqual(restored.text_config.sliding_window, 9)
        self.assertEqual(restored.audio_config.codebook_vocab_sizes, [16, 32])
        self.assertEqual(restored.image_token_id, 5)

    @parameterized.expand(
        [
            ("text_kv_heads", MiMoV2VLTextConfig, {"num_attention_heads": 6, "num_key_value_heads": 4}),
            ("text_sliding_kv_heads", MiMoV2VLTextConfig, {"swa_num_key_value_heads": 5}),
            ("text_expert_groups", MiMoV2VLTextConfig, {"n_routed_experts": 6, "n_group": 4}),
            ("vision_kv_heads", MiMoV2VLVisionConfig, {"num_heads": 6, "num_key_value_heads": 4}),
            ("vision_head_dim", MiMoV2VLVisionConfig, {"head_dim": 6}),
            ("vision_patch_orders", MiMoV2VLVisionConfig, {"depth": 2, "patch_orders": ["row_major", "diagonal"]}),
            ("audio_projection_layers", MiMoV2VLAudioConfig, {"projection_layers": 3}),
            ("audio_codebook_values", MiMoV2VLAudioConfig, {"num_codebooks": 2, "codebook_vocab_sizes": [1, 2, 3]}),
            ("codec_heads", MiMoV2VLAudioTokenizerConfig, {"hidden_size": 10, "encoder_attention_heads": 4}),
            ("output_size", MiMoV2VLConfig, {"vision_config": {"out_hidden_size": 8}}),
        ]
    )
    def test_invalid_architecture_raises(self, _, config_class, kwargs):
        with self.assertRaises(StrictDataclassClassValidationError):
            config_class(**kwargs)
