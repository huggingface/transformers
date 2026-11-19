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

from transformers import AutoConfig, DonutSwinConfig, MBartConfig, NougatConfig


class NougatConfigTest(unittest.TestCase):
    def check_round_trip(self, config):
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            reloaded = NougatConfig.from_pretrained(directory)
            auto_reloaded = AutoConfig.from_pretrained(directory)

        self.assertEqual(config.to_dict(), reloaded.to_dict())
        self.assertIsInstance(auto_reloaded, NougatConfig)
        self.assertEqual(config.encoder.to_dict(), auto_reloaded.encoder.to_dict())
        self.assertEqual(config.decoder.to_dict(), auto_reloaded.decoder.to_dict())

    def test_default_subconfigs_round_trip(self):
        config = NougatConfig()
        self.assertIsInstance(config.encoder, DonutSwinConfig)
        self.assertIsInstance(config.decoder, MBartConfig)
        self.check_round_trip(config)

    def test_partial_encoder_config_round_trip(self):
        config = NougatConfig(encoder={"image_size": 128, "embed_dim": 48})
        self.assertIsInstance(config.encoder, DonutSwinConfig)
        self.assertEqual(config.encoder.image_size, 128)
        self.assertEqual(config.encoder.embed_dim, 48)
        self.check_round_trip(config)

    def test_from_encoder_decoder_configs_round_trip(self):
        config = NougatConfig.from_encoder_decoder_configs(
            DonutSwinConfig(image_size=128, embed_dim=48),
            MBartConfig(vocab_size=101, d_model=64, encoder_attention_heads=4, decoder_attention_heads=4),
        )
        self.check_round_trip(config)
