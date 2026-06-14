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

import math
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
    from transformers.models.moss_audio_tokenizer.modeling_moss_audio_tokenizer import (
        MossAudioTokenizerRMSNorm,
        MossAudioTokenizerRotaryEmbedding,
        MossAudioTokenizerTransformer,
        apply_rotary_pos_emb,
    )


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

    def test_legacy_rms_norm_alpha_load(self):
        norm = MossAudioTokenizerRMSNorm(4)
        alpha = torch.arange(4, dtype=torch.float32).view(1, 1, 4)

        norm.load_state_dict({"alpha": alpha})

        self.assertTrue(torch.equal(norm.weight, alpha.reshape(4)))

    def test_legacy_transformer_rms_norm_state_dict_load(self):
        transformer_kwargs = {
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "causal": True,
            "context": 16,
            "norm": "rms_norm",
            "positional_embedding": "rope",
            "layer_scale": None,
        }
        reference = MossAudioTokenizerTransformer(**transformer_kwargs).to(torch_device).eval()
        legacy_state_dict = {}
        rms_norms = {
            name for name, module in reference.named_modules() if isinstance(module, MossAudioTokenizerRMSNorm)
        }
        for key, value in reference.state_dict().items():
            module_name, _, param_name = key.rpartition(".")
            if module_name in rms_norms and param_name == "weight":
                legacy_state_dict[f"{module_name}.alpha"] = value.reshape(1, 1, -1).clone()
            else:
                legacy_state_dict[key] = value.clone()

        loaded = MossAudioTokenizerTransformer(**transformer_kwargs).to(torch_device).eval()
        load_result = loaded.load_state_dict(legacy_state_dict, strict=True)

        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        hidden_states = torch.randn(1, 6, 8, device=torch_device)
        with torch.no_grad():
            self.assertTrue(torch.allclose(reference(hidden_states), loaded(hidden_states), atol=1e-6, rtol=1e-6))

    def test_rotary_embedding_preserves_legacy_attention_scores(self):
        batch_size, num_heads, sequence_length, head_dim = 2, 3, 5, 8
        query = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=torch_device)
        key = torch.randn(batch_size, num_heads, sequence_length, head_dim, device=torch_device)
        offset = torch.tensor([0, 7], device=torch_device)

        rotary_embedding = MossAudioTokenizerRotaryEmbedding(max_period=10000.0, head_dim=head_dim)
        position_ids = offset.view(batch_size, 1) + torch.arange(sequence_length, device=torch_device).view(1, -1)
        cos, sin = rotary_embedding(query, position_ids)
        query_half = torch.cat((query[..., ::2], query[..., 1::2]), dim=-1)
        key_half = torch.cat((key[..., ::2], key[..., 1::2]), dim=-1)
        query_llama, key_llama = apply_rotary_pos_emb(query_half, key_half, cos, sin)

        freqs = torch.exp(
            torch.arange(head_dim // 2, device=torch_device, dtype=torch.float32) * (-math.log(10000.0) * 2 / head_dim)
        )
        time = offset.float().view(batch_size, 1, 1, 1) + torch.arange(
            sequence_length, device=torch_device, dtype=torch.float32
        ).view(1, 1, sequence_length, 1)
        query = query.view(batch_size, num_heads, sequence_length, head_dim // 2, 2)
        key = key.view(batch_size, num_heads, sequence_length, head_dim // 2, 2)

        query_real, query_imaginary = query[..., 0].float(), query[..., 1].float()
        key_real, key_imaginary = key[..., 0].float(), key[..., 1].float()
        rot_real = torch.cos(freqs * time)
        rot_imaginary = torch.sin(freqs * time)

        query_legacy = torch.stack(
            [
                query_real * rot_real - query_imaginary * rot_imaginary,
                query_real * rot_imaginary + query_imaginary * rot_real,
            ],
            dim=-1,
        ).view(batch_size, num_heads, sequence_length, head_dim)
        key_legacy = torch.stack(
            [
                key_real * rot_real - key_imaginary * rot_imaginary,
                key_real * rot_imaginary + key_imaginary * rot_real,
            ],
            dim=-1,
        ).view(batch_size, num_heads, sequence_length, head_dim)

        expected = torch.matmul(query_legacy, key_legacy.transpose(-1, -2))
        actual = torch.matmul(query_llama, key_llama.transpose(-1, -2))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

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
