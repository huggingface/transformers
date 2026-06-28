# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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

from tests.test_configuration_common import ConfigTester
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioTokenization,
    MossAudioTokenizerConfig,
    MossAudioTokenizerDecoderConfig,
    MossAudioTokenizerEncoderConfig,
    MossAudioTokenizerFeatureExtractor,
    MossAudioTokenizerQuantizerConfig,
)
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import MossAudioTokenizerModel
    from transformers.models.moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformer


def get_moss_audio_tokenizer_config(**kwargs):
    return MossAudioTokenizerConfig(
        sampling_rate=16000,
        downsample_rate=4,
        causal_transformer_context_duration=1.0,
        encoder_config=MossAudioTokenizerEncoderConfig(patch_sizes=[4], input_dimensions=[]),
        decoder_config=MossAudioTokenizerDecoderConfig(patch_sizes=[4], input_dimensions=[]),
        quantizer_config=MossAudioTokenizerQuantizerConfig(
            input_dim=4,
            rvq_dim=4,
            output_dim=4,
            num_quantizers=2,
            codebook_size=16,
            codebook_dim=2,
        ),
        **kwargs,
    )


@require_torch
class MossAudioTokenizerModelTest(unittest.TestCase):
    all_model_classes = (MossAudioTokenizerModel,) if is_torch_available() else ()

    def setUp(self):
        self.config_tester = ConfigTester(
            self, config_class=MossAudioTokenizerConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_auto_config_and_model(self):
        config = AutoConfig.for_model("moss-audio-tokenizer")
        self.assertIsInstance(config, MossAudioTokenizerConfig)

        model = AutoModel.from_config(get_moss_audio_tokenizer_config())
        self.assertIsInstance(model, MossAudioTokenizerModel)

    def test_sub_configs(self):
        config = MossAudioTokenizerConfig(
            sampling_rate=16000,
            downsample_rate=4,
            causal_transformer_context_duration=1.0,
            encoder_config=MossAudioTokenizerEncoderConfig(patch_sizes=[4], input_dimensions=[]),
            decoder_config=MossAudioTokenizerDecoderConfig(patch_sizes=[4], input_dimensions=[]),
            quantizer_config=MossAudioTokenizerQuantizerConfig(
                input_dim=4,
                rvq_dim=4,
                output_dim=4,
                num_quantizers=2,
                codebook_size=16,
                codebook_dim=2,
            ),
        )

        self.assertIsInstance(config.encoder_config, MossAudioTokenizerEncoderConfig)
        self.assertIsInstance(config.decoder_config, MossAudioTokenizerDecoderConfig)
        self.assertIsInstance(config.quantizer_config, MossAudioTokenizerQuantizerConfig)
        self.assertEqual(config.sampling_rate, 16000)
        self.assertEqual(config.encoder_config.patch_sizes, [4])
        self.assertEqual(config.encoder_config.input_dimensions, [])
        self.assertEqual(config.decoder_config.patch_sizes, [4])
        self.assertEqual(config.decoder_config.input_dimensions, [])

        config_dict = config.to_dict()
        self.assertIn("encoder_config", config_dict)
        self.assertIn("decoder_config", config_dict)
        self.assertIn("quantizer_config", config_dict)
        self.assertNotIn("quantizer_type", config_dict["quantizer_config"])

        transformer_encoder_config = MossAudioTokenizerEncoderConfig(
            patch_sizes=[],
            input_dimensions=[4],
            output_dimensions=[4],
            d_models=[4],
            num_heads=[1],
            num_layers=[1],
            dim_feedforward=[8],
            hidden_act="relu",
        )
        self.assertEqual(transformer_encoder_config.hidden_act, ["relu"])

    def test_encode_decode_and_forward(self):
        config = get_moss_audio_tokenizer_config()
        model = MossAudioTokenizerModel(config).to(torch_device).eval()
        input_values = torch.randn(2, 1, 32, device=torch_device)

        with torch.no_grad():
            encoded = model.encode(input_values, return_dict=True)
            self.assertEqual(encoded.audio_codes.shape, (2, config.num_quantizers, 8))
            self.assertEqual(encoded.audio_codes_lengths.tolist(), [8, 8])

            decoded = model.decode(encoded.audio_codes, return_dict=True)
            self.assertEqual(decoded.audio.shape, (2, 1, 32))
            self.assertEqual(decoded.audio_lengths.tolist(), [32, 32])

            outputs = model(input_values=input_values)
            self.assertEqual(outputs.audio.shape, (2, 1, 32))
            self.assertEqual(outputs.audio_lengths.tolist(), [32, 32])
            self.assertEqual(outputs.audio_codes.shape, (2, config.num_quantizers, 8))

    def test_feature_extractor_and_auto_model_for_audio_tokenization(self):
        config = get_moss_audio_tokenizer_config()
        model = MossAudioTokenizerModel(config).to(torch_device).eval()
        feature_extractor = MossAudioTokenizerFeatureExtractor(sampling_rate=16000, hop_length=4)
        wav_list = [torch.randn(32).numpy(), torch.randn(24).numpy()]

        inputs = feature_extractor(wav_list, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            encoded = model.encode(**inputs, return_dict=True)
            self.assertEqual(encoded.audio_codes.shape, (2, config.num_quantizers, 8))
            self.assertEqual(encoded.audio_codes_lengths.tolist(), [8, 6])

            decoded = model.decode(
                encoded.audio_codes, padding_mask=encoded.audio_codes.new_ones((2, 8)), return_dict=True
            )
            self.assertEqual(decoded.audio.shape, (2, 1, 32))
            self.assertEqual(decoded.audio_lengths.tolist(), [32, 32])

        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_extractor.save_pretrained(tmp_dir)
            config.save_pretrained(tmp_dir)
            loaded_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir)
        self.assertIsInstance(loaded_feature_extractor, MossAudioTokenizerFeatureExtractor)
        self.assertIsInstance(AutoModelForAudioTokenization.from_config(config), MossAudioTokenizerModel)

    def test_save_load(self):
        config = get_moss_audio_tokenizer_config()
        model = MossAudioTokenizerModel(config).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            loaded = MossAudioTokenizerModel.from_pretrained(tmpdirname).eval()

        input_values = torch.randn(1, 1, 32)
        with torch.no_grad():
            expected = model(input_values=input_values).audio_codes
            actual = loaded(input_values=input_values).audio_codes

        self.assertTrue(torch.equal(expected, actual))

    def test_transformer_rms_norm_forward(self):
        transformer_kwargs = {
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "causal": True,
            "context": 16,
            "norm": "rms_norm",
            "positional_embedding": "rope",
            "hidden_act": "relu",
            "layer_scale": None,
        }
        transformer = MossAudioTokenizerTransformer(**transformer_kwargs).to(torch_device).eval()
        hidden_states = torch.randn(1, 6, 8, device=torch_device)
        with torch.no_grad():
            output = transformer(hidden_states)
        self.assertEqual(output.shape, hidden_states.shape)

    def test_transformer_streaming_matches_full_forward(self):
        transformer = MossAudioTokenizerTransformer(
            d_model=8,
            num_heads=2,
            num_layers=1,
            dim_feedforward=16,
            causal=True,
            context=16,
            positional_embedding="rope",
            layer_scale=None,
        ).to(torch_device)
        transformer.eval()

        hidden_states = torch.randn(1, 6, 8, device=torch_device)
        with torch.no_grad():
            full_output = transformer(hidden_states)
            with transformer.streaming(batch_size=1):
                streamed_output = torch.cat(
                    [transformer(hidden_states[:, :3]), transformer(hidden_states[:, 3:])], dim=1
                )

        self.assertTrue(torch.allclose(full_output, streamed_output, atol=1e-5, rtol=1e-5))


@require_torch
class MossAudioTokenizerIntegrationTest(unittest.TestCase):
    model_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

    @slow
    def test_integration_encode_decode(self):
        model, loading_info = MossAudioTokenizerModel.from_pretrained(
            self.model_id, device_map="auto", output_loading_info=True
        )
        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        self.assertFalse(loading_info["error_msgs"])

        model.eval()
        device = next(model.parameters()).device
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        audio = torch.linspace(-0.25, 0.25, steps=24000).numpy()
        inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            encoded = model.encode(**inputs, num_quantizers=4, return_dict=True)
            self.assertEqual(encoded.audio_codes.shape, (1, 4, 13))
            self.assertEqual(encoded.audio_codes_lengths.tolist(), [13])
            self.assertEqual(
                encoded.audio_codes[0, :, :4].cpu().tolist(),
                [[245, 402, 936, 936], [996, 948, 948, 600], [154, 484, 484, 548], [119, 743, 751, 456]],
            )

            decoded = model.decode(encoded.audio_codes, return_dict=True)
            self.assertEqual(decoded.audio.shape, (1, 1, 24960))
            self.assertEqual(decoded.audio_lengths.tolist(), [24960])
            self.assertGreater(float(decoded.audio.abs().mean().cpu()), 1e-4)
