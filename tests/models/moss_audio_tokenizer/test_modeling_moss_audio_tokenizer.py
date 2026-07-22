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
    from transformers.models.moss_audio_tokenizer.convert_moss_audio_tokenizer_to_hf import _rename_state_dict_key
    from transformers.models.moss_audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformer


def get_moss_audio_tokenizer_config(**kwargs):
    return MossAudioTokenizerConfig(
        sampling_rate=16000,
        sliding_window_duration=1.0,
        downsampling_ratios=[4],
        input_hidden_sizes=[4],
        output_hidden_sizes=[4],
        hidden_sizes=[4],
        num_attention_heads=[1],
        num_hidden_layers=[1],
        intermediate_sizes=[8],
        quantizer_config=MossAudioTokenizerQuantizerConfig(
            input_hidden_size=4,
            hidden_size=4,
            output_hidden_size=4,
            n_codebooks=2,
            codebook_size=16,
            codebook_dim=2,
        ),
        **kwargs,
    )


def get_asymmetric_moss_audio_tokenizer_config(**kwargs):
    return MossAudioTokenizerConfig(
        sampling_rate=16000,
        sliding_window_duration=1.0,
        downsampling_ratios=[2, 3],
        input_hidden_sizes=[2, 9],
        output_hidden_sizes=[3, 4],
        hidden_sizes=[4, 6],
        num_attention_heads=[1, 1],
        num_hidden_layers=[1, 1],
        intermediate_sizes=[8, 12],
        quantizer_config=MossAudioTokenizerQuantizerConfig(
            input_hidden_size=4,
            hidden_size=4,
            output_hidden_size=4,
            n_codebooks=2,
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
            self,
            config_class=MossAudioTokenizerConfig,
            common_properties=["encoder_config", "decoder_config"],
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_auto_config_and_model(self):
        config = AutoConfig.for_model("moss-audio-tokenizer")
        self.assertIsInstance(config, MossAudioTokenizerConfig)
        self.assertIsInstance(config.quantizer_config, MossAudioTokenizerQuantizerConfig)

        model = AutoModel.from_config(get_moss_audio_tokenizer_config(layer_scale_init_value=0.02))
        self.assertIsInstance(model, MossAudioTokenizerModel)
        self.assertTrue(torch.all(model.encoder.layers[1].transformer.layers[0].layer_scale_1.scale == 0.02))

    def test_sub_configs(self):
        self.assertIsInstance(MossAudioTokenizerConfig().quantizer_config, MossAudioTokenizerQuantizerConfig)

        config = MossAudioTokenizerConfig(
            sampling_rate=16000,
            sliding_window_duration=1.0,
            downsampling_ratios=[4],
            input_hidden_sizes=[4],
            output_hidden_sizes=[4],
            hidden_sizes=[4],
            num_attention_heads=[1],
            num_hidden_layers=[1],
            intermediate_sizes=[8],
            quantizer_config=MossAudioTokenizerQuantizerConfig(
                input_hidden_size=4,
                hidden_size=4,
                output_hidden_size=4,
                n_codebooks=2,
                codebook_size=16,
                codebook_dim=2,
            ),
        )

        self.assertIsInstance(config.quantizer_config, MossAudioTokenizerQuantizerConfig)
        self.assertEqual(config.sampling_rate, 16000)
        self.assertIsInstance(config.sliding_window_duration, float)
        self.assertEqual(config.hop_length, 4)
        self.assertEqual(config.max_position_embeddings, 2048)
        self.assertEqual(config.layer_scale_init_value, 0.01)
        self.assertFalse(hasattr(config, "rope_parameters"))
        self.assertIsInstance(config.encoder_config, MossAudioTokenizerEncoderConfig)
        self.assertIsInstance(config.decoder_config, MossAudioTokenizerDecoderConfig)
        self.assertEqual(config.decoder_config.input_hidden_sizes, [4])
        self.assertEqual(config.decoder_config.output_hidden_sizes, [4])
        self.assertEqual(config.decoder_config.upsampling_ratios, [4])
        self.assertFalse(hasattr(config, "_encoder_config"))
        self.assertFalse(hasattr(config, "_decoder_config"))
        self.assertEqual(config.quantizer_config.n_codebooks, 2)

        multi_stage_config = get_asymmetric_moss_audio_tokenizer_config()
        self.assertEqual(multi_stage_config.decoder_config.downsampling_ratios, [2, 3])
        self.assertEqual(multi_stage_config.decoder_config.upsampling_ratios, [3, 2])

        config_dict = config.to_dict()
        self.assertNotIn("encoder_config", config_dict)
        self.assertNotIn("decoder_config", config_dict)
        self.assertIn("quantizer_config", config_dict)

    def test_encode_decode_and_forward(self):
        config = get_moss_audio_tokenizer_config()
        model = MossAudioTokenizerModel(config).to(torch_device).eval()
        input_values = torch.randn(2, 1, 32, device=torch_device)

        with torch.no_grad():
            encoded = model.encode(input_values, return_dict=True)
            self.assertEqual(encoded.audio_codes.shape, (2, config.quantizer_config.n_codebooks, 8))
            self.assertEqual(encoded.audio_codes_lengths.tolist(), [8, 8])

            decoded = model.decode(encoded.audio_codes, return_dict=True)
            self.assertEqual(decoded.audio.shape, (2, 1, 32))
            self.assertEqual(decoded.audio_lengths.tolist(), [32, 32])

            outputs = model(input_values=input_values)
            self.assertEqual(outputs.audio.shape, (2, 1, 32))
            self.assertEqual(outputs.audio_lengths.tolist(), [32, 32])
            self.assertEqual(outputs.audio_codes.shape, (2, config.quantizer_config.n_codebooks, 8))

    def test_decode_uses_upsampling_ratios(self):
        config = get_asymmetric_moss_audio_tokenizer_config()
        model = MossAudioTokenizerModel(config).to(torch_device).eval()
        input_values = torch.randn(2, 1, 24, device=torch_device)

        with torch.no_grad():
            encoded = model.encode(input_values, return_dict=True)
            self.assertEqual(encoded.audio_codes.shape, (2, config.quantizer_config.n_codebooks, 4))

            decoded = model.decode(encoded.audio_codes, return_dict=True)
            self.assertEqual(decoded.audio.shape, (2, 1, 24))
            self.assertEqual(decoded.audio_lengths.tolist(), [24, 24])

    def test_feature_extractor_and_auto_model_for_audio_tokenization(self):
        config = get_moss_audio_tokenizer_config()
        model = MossAudioTokenizerModel(config).to(torch_device).eval()
        feature_extractor = MossAudioTokenizerFeatureExtractor(sampling_rate=16000, hop_length=4)
        wav_list = [torch.randn(32), torch.randn(24)]

        inputs = feature_extractor(wav_list, sampling_rate=16000)
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            encoded = model.encode(**inputs, return_dict=True)
            self.assertEqual(encoded.audio_codes.shape, (2, config.quantizer_config.n_codebooks, 8))
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

    def test_convert_state_dict_renames_attention_projection_lists(self):
        self.assertEqual(
            _rename_state_dict_key("encoder.1.transformer.layers.0.self_attn.in_projs.0.weight"),
            "encoder.layers.1.transformer.layers.0.self_attn.in_proj.weight",
        )
        self.assertEqual(
            _rename_state_dict_key("encoder.1.transformer.layers.0.self_attn.out_projs.0.weight"),
            "encoder.layers.1.transformer.layers.0.self_attn.out_proj.weight",
        )

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

    def test_transformer_forward(self):
        transformer_kwargs = {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "context": 16,
            "layer_scale_init_value": 0.02,
            "max_position_embeddings": 32,
        }
        transformer = MossAudioTokenizerTransformer(**transformer_kwargs).to(torch_device).eval()
        self.assertEqual(transformer.rope.config.max_position_embeddings, 32)
        self.assertEqual(transformer.rope.config.rope_parameters, {"rope_theta": 10000.0, "rope_type": "default"})
        self.assertTrue(torch.all(transformer.layers[0].layer_scale_1.scale == 0.02))
        hidden_states = torch.randn(1, 6, 8, device=torch_device)
        with torch.no_grad():
            output = transformer(hidden_states)
        self.assertEqual(output.shape, hidden_states.shape)

    def test_transformer_streaming_matches_full_forward(self):
        transformer = MossAudioTokenizerTransformer(
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=16,
            context=16,
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
