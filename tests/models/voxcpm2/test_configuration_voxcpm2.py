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

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import (
    AutoConfig,
    VoxCPM2AudioVAEConfig,
    VoxCPM2CfmConfig,
    VoxCPM2Config,
    VoxCPM2DiTConfig,
    VoxCPM2EncoderConfig,
    VoxCPM2TextConfig,
)

from ...test_configuration_common import ConfigTester


class VoxCPM2ModelTester:
    def get_config(self):
        return VoxCPM2Config(lm_config={"vocab_size": 128})


class VoxCPM2ConfigTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = VoxCPM2ModelTester()
        self.config_tester = ConfigTester(self, config_class=VoxCPM2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_sub_configs_are_normalized(self):
        config = VoxCPM2Config(
            lm_config={"hidden_size": 2048, "num_attention_heads": 16, "kv_channels": 128, "use_mup": False},
            encoder_config={"hidden_dim": 512, "ffn_dim": 1536, "num_heads": 8, "num_layers": 3, "kv_channels": 64},
            dit_config={
                "hidden_dim": 768,
                "ffn_dim": 2304,
                "num_heads": 12,
                "num_layers": 5,
                "kv_channels": 64,
                "mean_mode": False,
                "cfm_config": {"inference_cfg_rate": 2.0},
            },
            audio_vae_config={"latent_dim": 64},
        )

        self.assertIsInstance(config.lm_config, VoxCPM2TextConfig)
        self.assertIsInstance(config.encoder_config, VoxCPM2EncoderConfig)
        self.assertIsInstance(config.dit_config, VoxCPM2DiTConfig)
        self.assertIsInstance(config.dit_config.cfm_config, VoxCPM2CfmConfig)
        self.assertIsInstance(config.audio_vae_config, VoxCPM2AudioVAEConfig)
        self.assertIs(config.get_text_config(), config.lm_config)

        for sub_config, expected_values in (
            (
                config.encoder_config,
                {
                    "hidden_size": 512,
                    "intermediate_size": 1536,
                    "num_attention_heads": 8,
                    "num_hidden_layers": 3,
                    "head_dim": 64,
                },
            ),
            (
                config.dit_config,
                {
                    "hidden_size": 768,
                    "intermediate_size": 2304,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 5,
                    "head_dim": 64,
                },
            ),
        ):
            serialized_config = sub_config.to_dict()
            for name, value in expected_values.items():
                self.assertEqual(getattr(sub_config, name), value)
                self.assertEqual(serialized_config[name], value)
            for legacy_name in ("hidden_dim", "ffn_dim", "num_heads", "num_layers", "kv_channels"):
                self.assertNotIn(legacy_name, serialized_config)

    def test_released_config_fields_are_preserved(self):
        config = VoxCPM2Config(
            architecture="voxcpm2",
            device="cuda",
            max_length=8192,
            lm_config={
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "max_position_embeddings": 32768,
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "kv_channels": 128,
                "vocab_size": 73448,
                "use_mup": False,
                "scale_emb": 12,
                "dim_model_base": 256,
                "scale_depth": 1.4,
            },
            residual_lm_no_rope=True,
            encoder_config={"hidden_dim": 1024, "ffn_dim": 4096, "num_heads": 16, "num_layers": 12},
            dit_config={
                "hidden_dim": 1024,
                "ffn_dim": 4096,
                "num_heads": 16,
                "num_layers": 12,
                "mean_mode": False,
                "cfm_config": {"inference_cfg_rate": 2.0},
            },
            audio_vae_config={"latent_dim": 64, "sample_rate": 16000, "out_sample_rate": 48000},
        )

        self.assertFalse(config.lm_config.use_mup)
        self.assertEqual(config.lm_config.head_dim, 128)
        self.assertTrue(config.residual_lm_no_rope)
        self.assertFalse(config.dit_config.mean_mode)
        self.assertEqual(config.lm_config.kv_channels, config.lm_config.head_dim)
        self.assertNotIn("kv_channels", config.lm_config.to_dict())
        self.assertEqual(config.dit_config.cfm_config.inference_cfg_rate, 2.0)
        self.assertEqual(config.max_cache_length, 8192)
        self.assertFalse(hasattr(config, "device"))
        self.assertFalse(hasattr(config, "architecture"))

    def test_save_load_and_auto_config(self):
        config = VoxCPM2Config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)

            loaded_config = AutoConfig.from_pretrained(tmp_dir)
            loaded_text_config = VoxCPM2TextConfig.from_pretrained(tmp_dir)
            loaded_encoder_config = VoxCPM2EncoderConfig.from_pretrained(tmp_dir)

        self.assertIsInstance(loaded_config, VoxCPM2Config)
        self.assertIsInstance(loaded_config.lm_config, VoxCPM2TextConfig)
        self.assertIsInstance(loaded_text_config, VoxCPM2TextConfig)
        self.assertIsInstance(loaded_encoder_config, VoxCPM2EncoderConfig)
        loaded_dict = loaded_config.to_dict()
        config_dict = config.to_dict()
        loaded_dict.pop("_name_or_path")
        config_dict.pop("_name_or_path")
        self.assertEqual(loaded_dict, config_dict)

    def test_audio_vae_hop_lengths(self):
        config = VoxCPM2AudioVAEConfig()

        self.assertEqual(config.hop_length, 640)
        self.assertEqual(config.decode_hop_length, 1920)

    def test_output_sample_rate(self):
        config = VoxCPM2Config(audio_vae_config={"out_sample_rate": 24000})

        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.to_dict()["sample_rate"], 24000)
        with self.assertRaisesRegex(ValueError, "out_sample_rate"):
            VoxCPM2Config(sample_rate=16000, audio_vae_config={"out_sample_rate": 24000})

    def test_feat_dim_must_match_audio_vae_latent_dim(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "must match AudioVAE"):
            VoxCPM2Config(feat_dim=32, audio_vae_config={"latent_dim": 64})
