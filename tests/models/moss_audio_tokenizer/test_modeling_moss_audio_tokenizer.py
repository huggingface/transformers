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
    AutoModel,
    MossAudioTokenizerConfig,
    MossAudioTokenizerDecoderConfig,
    MossAudioTokenizerEncoderConfig,
    MossAudioTokenizerQuantizerConfig,
)
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import MossAudioTokenizerModel


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
            quantizer_type="rlfq",
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

    def test_legacy_config_kwargs(self):
        encoder_kwargs = [{"module_type": "PatchedPretransform", "patch_size": 4}]
        decoder_kwargs = [{"module_type": "PatchedPretransform", "patch_size": 4}]
        quantizer_kwargs = {
            "input_dim": 4,
            "rvq_dim": 4,
            "output_dim": 4,
            "num_quantizers": 2,
            "codebook_size": 16,
            "codebook_dim": 2,
            "quantizer_type": "rlfq",
        }

        config = MossAudioTokenizerConfig(
            sample_rate=16000,
            downsample_rate=4,
            causal_transformer_context_duration=1.0,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            quantizer_kwargs=quantizer_kwargs,
            code_dim=4,
            reversed_decoder_kwargs=encoder_kwargs,
        )

        self.assertIsInstance(config.encoder_config, MossAudioTokenizerEncoderConfig)
        self.assertIsInstance(config.decoder_config, MossAudioTokenizerDecoderConfig)
        self.assertIsInstance(config.quantizer_config, MossAudioTokenizerQuantizerConfig)
        self.assertEqual(config.sampling_rate, 16000)
        self.assertEqual(config.encoder_kwargs, encoder_kwargs)
        self.assertEqual(config.decoder_kwargs, decoder_kwargs)
        self.assertEqual(config.quantizer_kwargs, quantizer_kwargs)

        config_dict = config.to_dict()
        self.assertIn("encoder_config", config_dict)
        self.assertIn("decoder_config", config_dict)
        self.assertIn("quantizer_config", config_dict)
        self.assertNotIn("encoder_kwargs", config_dict)
        self.assertNotIn("decoder_kwargs", config_dict)
        self.assertNotIn("quantizer_kwargs", config_dict)

    def test_encode_decode_and_forward(self):
        config = get_moss_audio_tokenizer_config()
        model = MossAudioTokenizerModel(config).to(torch_device).eval()
        input_values = torch.randn(2, 1, 32, device=torch_device)

        with torch.no_grad():
            encoded = model.encode(input_values, return_dict=True)
            self.assertEqual(encoded.audio_codes.shape, (config.num_quantizers, 2, 8))
            self.assertEqual(encoded.audio_codes_lengths.tolist(), [8, 8])

            decoded = model.decode(encoded.audio_codes, return_dict=True)
            self.assertEqual(decoded.audio.shape, (2, 1, 32))
            self.assertEqual(decoded.audio_lengths.tolist(), [32, 32])

            outputs = model(input_values=input_values)
            self.assertEqual(outputs.audio.shape, (2, 1, 32))
            self.assertEqual(outputs.audio_lengths.tolist(), [32, 32])
            self.assertEqual(outputs.audio_codes.shape, (config.num_quantizers, 2, 8))

    def test_batch_encode_decode(self):
        config = get_moss_audio_tokenizer_config()
        model = MossAudioTokenizerModel(config).to(torch_device).eval()
        wav_list = [torch.randn(32, device=torch_device), torch.randn(24, device=torch_device)]

        with torch.no_grad():
            encoded = model.batch_encode(wav_list)
            self.assertEqual(encoded.audio_codes.shape, (config.num_quantizers, 2, 8))
            self.assertEqual(encoded.audio_codes_lengths.tolist(), [8, 6])

            codes_list = [encoded.audio_codes[:, 0, :8], encoded.audio_codes[:, 1, :6]]
            decoded = model.batch_decode(codes_list)
            self.assertEqual(decoded.audio.shape, (2, 1, 32))
            self.assertEqual(decoded.audio_lengths.tolist(), [32, 24])

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


@require_torch
class MossAudioTokenizerIntegrationTest(unittest.TestCase):
    model_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

    @slow
    def test_integration_encode_decode(self):
        model = MossAudioTokenizerModel.from_pretrained(self.model_id, device_map="auto").eval()
        device = next(model.parameters()).device
        input_values = torch.linspace(-0.25, 0.25, steps=24000, device=device).reshape(1, 1, -1)

        with torch.no_grad():
            encoded = model.encode(input_values, num_quantizers=4, return_dict=True)
            self.assertEqual(encoded.audio_codes.shape, (4, 1, 13))
            self.assertEqual(encoded.audio_codes_lengths.tolist(), [12])
            self.assertEqual(
                encoded.audio_codes[:, 0, :4].cpu().tolist(),
                [[245, 402, 936, 936], [996, 948, 948, 600], [154, 484, 484, 548], [119, 743, 751, 456]],
            )

            decoded = model.decode(encoded.audio_codes, return_dict=True)
            self.assertEqual(decoded.audio.shape, (1, 1, 24960))
            self.assertEqual(decoded.audio_lengths.tolist(), [24960])
            self.assertGreater(float(decoded.audio.abs().mean().cpu()), 1e-4)
