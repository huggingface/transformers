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
            lm_config={"hidden_size": 1024, "num_attention_heads": 16, "kv_channels": 64},
            encoder_config={"hidden_dim": 512, "ffn_dim": 1536, "num_heads": 8, "num_layers": 3, "kv_channels": 64},
            dit_config={
                "hidden_dim": 768,
                "ffn_dim": 2304,
                "num_heads": 12,
                "num_layers": 5,
                "kv_channels": 64,
                "cfm_config": {"inference_cfg_rate": 1.5},
            },
            audio_vae_config={"latent_dim": 64},
        )

        self.assertIsInstance(config.lm_config, VoxCPM2TextConfig)
        self.assertIsInstance(config.encoder_config, VoxCPM2EncoderConfig)
        self.assertIsInstance(config.dit_config, VoxCPM2DiTConfig)
        self.assertIsInstance(config.dit_config.cfm_config, VoxCPM2CfmConfig)
        self.assertIsInstance(config.audio_vae_config, VoxCPM2AudioVAEConfig)
        self.assertIs(config.get_text_config(), config.lm_config)
        self.assertEqual(config.dit_config.cfm_config.inference_cfg_rate, 1.5)

        for sub_config, expected_values in (
            (config.lm_config, {"hidden_size": 1024, "num_attention_heads": 16, "head_dim": 64}),
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

    def test_legacy_checkpoint_metadata(self):
        config = VoxCPM2Config(
            architecture="voxcpm2",
            device="cuda",
            max_length=4096,
        )

        self.assertEqual(config.max_cache_length, 4096)
        self.assertFalse(hasattr(config, "device"))
        self.assertFalse(hasattr(config, "architecture"))

    def test_auto_config_registration(self):
        self.assertIsInstance(AutoConfig.for_model("voxcpm2"), VoxCPM2Config)

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
