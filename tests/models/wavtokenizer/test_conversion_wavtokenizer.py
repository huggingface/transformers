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

import math
import re
import tempfile
import unittest
from pathlib import Path

from transformers.testing_utils import require_torch


@require_torch
class WavTokenizerConversionTest(unittest.TestCase):
    def _state_dict(self, upsampling_ratios):
        import torch

        state_dict = {
            "feature_extractor.encodec.encoder.model.0.conv.conv.weight_v": torch.empty(32, 1, 7),
            "feature_extractor.encodec.encoder.model.13.lstm.weight_ih_l0": torch.empty(2048, 512),
            "feature_extractor.encodec.encoder.model.13.lstm.weight_ih_l1": torch.empty(2048, 512),
            "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed": torch.empty(4096, 512),
            "backbone.embed.weight": torch.empty(768, 512, 7),
            "backbone.norm.scale.weight": torch.empty(4, 768),
            "backbone.convnext.0.pwconv1.weight": torch.empty(2304, 768),
        }
        encoder_ratios = list(reversed(upsampling_ratios))
        channels = 32
        for layer_index, ratio in zip((3, 6, 9, 12), encoder_ratios):
            state_dict[f"feature_extractor.encodec.encoder.model.{layer_index}.conv.conv.weight_v"] = torch.empty(
                channels * 2, channels, ratio * 2
            )
            channels *= 2
        for layer_index in range(12):
            state_dict[f"backbone.convnext.{layer_index}.dwconv.weight"] = torch.empty(768, 1, 7)
        hop_length = math.prod(upsampling_ratios)
        state_dict["head.out.weight"] = torch.empty(4 * hop_length + 2, 768)
        return state_dict

    def _to_original_state_dict(self, state_dict):
        original_state_dict = {}
        for key, value in state_dict.items():
            original_key = key.replace(
                "encoder.layers.",
                "feature_extractor.encodec.encoder.model.",
            )
            original_key = original_key.replace(
                "quantizer.codebook.",
                "feature_extractor.encodec.quantizer.vq.layers.0._codebook.",
            )
            original_key = original_key.replace("head.linear.", "head.out.")
            if original_key.startswith("feature_extractor.encodec.encoder.model."):
                original_key = re.sub(
                    r"\.conv\.parametrizations\.weight\.original0$", ".conv.conv.weight_g", original_key
                )
                original_key = re.sub(
                    r"\.conv\.parametrizations\.weight\.original1$", ".conv.conv.weight_v", original_key
                )
                original_key = re.sub(r"\.conv\.bias$", ".conv.conv.bias", original_key)
            original_state_dict[original_key] = value
        return original_state_dict

    def test_infers_40_token_architecture(self):
        from transformers.models.wavtokenizer.convert_wavtokenizer_checkpoint import infer_wavtokenizer_config

        config = infer_wavtokenizer_config(self._state_dict([6, 5, 5, 4]))

        self.assertEqual(config.upsampling_ratios, [6, 5, 5, 4])
        self.assertEqual(config.hop_length, 600)
        self.assertEqual(config.n_fft, 2400)
        self.assertEqual(config.frame_rate, 40)
        self.assertEqual(config.num_filters, 32)
        self.assertEqual(config.num_lstm_layers, 2)
        self.assertEqual(config.codebook_dim, 512)
        self.assertEqual(config.decoder_num_layers, 12)

    def test_infers_75_token_architecture(self):
        from transformers.models.wavtokenizer.convert_wavtokenizer_checkpoint import infer_wavtokenizer_config

        config = infer_wavtokenizer_config(self._state_dict([8, 5, 4, 2]))

        self.assertEqual(config.upsampling_ratios, [8, 5, 4, 2])
        self.assertEqual(config.hop_length, 320)
        self.assertEqual(config.n_fft, 1280)
        self.assertEqual(config.frame_rate, 75)
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.codebook_size, 4096)
        self.assertEqual(config.decoder_hidden_size, 768)
        self.assertEqual(config.decoder_intermediate_size, 2304)
        self.assertEqual(config.adanorm_num_embeddings, 4)

    def test_rejects_inconsistent_istft_head(self):
        from transformers.models.wavtokenizer.convert_wavtokenizer_checkpoint import infer_wavtokenizer_config

        state_dict = self._state_dict([8, 5, 4, 2])
        state_dict["head.out.weight"] = state_dict["head.out.weight"][:-4]

        with self.assertRaisesRegex(ValueError, "ISTFT head implies"):
            infer_wavtokenizer_config(state_dict)

    def test_rejects_missing_downsampling_layer(self):
        from transformers.models.wavtokenizer.convert_wavtokenizer_checkpoint import infer_wavtokenizer_config

        state_dict = self._state_dict([8, 5, 4, 2])
        del state_dict["feature_extractor.encodec.encoder.model.12.conv.conv.weight_v"]

        with self.assertRaisesRegex(ValueError, "expected 4"):
            infer_wavtokenizer_config(state_dict)

    def test_converts_and_reloads_75_token_checkpoint(self):
        import torch

        from transformers import WavTokenizerConfig, WavTokenizerFeatureExtractor, WavTokenizerModel
        from transformers.models.wavtokenizer.convert_wavtokenizer_checkpoint import convert_checkpoint

        config = WavTokenizerConfig(
            num_filters=4,
            upsampling_ratios=[8, 5, 4, 2],
            hidden_size=64,
            codebook_size=64,
            codebook_dim=64,
            decoder_hidden_size=32,
            decoder_intermediate_size=64,
            decoder_num_layers=2,
        )
        source_model = WavTokenizerModel(config)
        original_state_dict = self._to_original_state_dict(source_model.state_dict())

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "original.ckpt"
            output_dir = Path(temporary_directory) / "converted"
            torch.save({"state_dict": original_state_dict}, checkpoint_path)

            convert_checkpoint(str(checkpoint_path), str(output_dir))

            converted_model = WavTokenizerModel.from_pretrained(output_dir)
            feature_extractor = WavTokenizerFeatureExtractor.from_pretrained(output_dir)

        self.assertEqual(converted_model.config.upsampling_ratios, [8, 5, 4, 2])
        self.assertEqual(converted_model.config.hop_length, 320)
        self.assertEqual(feature_extractor.hop_length, 320)
        self.assertEqual(feature_extractor.sampling_rate, converted_model.config.sampling_rate)
        self.assertEqual(source_model.state_dict().keys(), converted_model.state_dict().keys())
        for key, expected_value in source_model.state_dict().items():
            torch.testing.assert_close(converted_model.state_dict()[key], expected_value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
