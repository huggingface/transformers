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
"""Testing suite for the PyTorch Liquid Codec model."""

import math
import os
import tempfile
import unittest

import safetensors.torch

from transformers import LiquidCodecConfig, LiquidCodecModel, is_torch_available
from transformers.testing_utils import Expectations, require_torch, slow, torch_device

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch


def tiny_backbone(attention_type, **kwargs):
    return {
        "vocab_size": 1,
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "layer_types": ["conv", attention_type, "conv", attention_type],
        "rope_parameters": {attention_type: {"rope_type": "default", "rope_theta": 1_000_000.0}},
        "use_cache": False,
        "tie_word_embeddings": False,
        **kwargs,
    }


def tiny_config(**kwargs):
    return LiquidCodecConfig(
        encoder_config=tiny_backbone("full_attention", sliding_window=None),
        decoder_config=tiny_backbone("sliding_attention", sliding_window=30),
        **{"encoder_conv_dim": 8, "quantizer_dim": 16, "n_q": 4, "fsq_levels": [3, 3], **kwargs},
    )


class LiquidCodecModelTester:
    def __init__(self, parent, batch_size=2, num_samples=1920 * 2 + 100):
        self.parent = parent
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.is_training = False

    def get_config(self):
        return tiny_config()

    def prepare_config_and_inputs_for_common(self):
        input_values = floats_tensor([self.batch_size, self.num_samples], scale=1.0)
        return self.get_config(), {"input_values": input_values}


@require_torch
class LiquidCodecModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LiquidCodecModel,)
    has_attentions = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = LiquidCodecModelTester(self)
        torch.manual_seed(0)
        self.config = tiny_config()
        self.codec = LiquidCodecModel(self.config).to(torch_device).eval()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # the codec records neither attentions nor hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        inputs_dict.pop("output_attentions", None)
        inputs_dict.pop("output_hidden_states", None)
        return inputs_dict

    @unittest.skip("The codec takes audio, not token embeddings")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("The codec takes audio, not token embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("The codec records no hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("The codec records no hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("The codec records no hidden states")
    def test_capture_outputs_decorator(self):
        pass

    @unittest.skip("The codec records no hidden states")
    def test_can_capture_specific_layers_hidden_states(self):
        pass

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)
            with torch.no_grad():
                tuple_output = model(**inputs, return_dict=False)
                dict_output = model(**inputs, return_dict=True)
            self.assertIsInstance(tuple_output, tuple)
            for tuple_value, dict_value in zip(tuple_output, dict_output.to_tuple()):
                torch.testing.assert_close(tuple_value, dict_value)

    def test_encode_decode_shapes(self):
        audio = torch.randn(2, 1920 * 3 + 100, device=torch_device)
        with torch.no_grad():
            codes = self.codec.encode(audio).audio_codes
            decoded = self.codec.decode(codes).audio_values
            outputs = self.codec(audio)
            reconstructed, same_codes = self.codec(audio, return_dict=False)
            encoded = self.codec.encode(audio, return_dict=False)
        self.assertIsInstance(encoded, tuple)
        self.assertTrue(torch.equal(encoded[0], codes))
        self.assertEqual(codes.shape, (2, 4, 4))
        self.assertTrue(bool((codes >= 0).all() and (codes < 9).all()))
        self.assertEqual(decoded.shape, (2, 4 * 1920))
        self.assertEqual(outputs.audio_values.shape, audio.shape)
        self.assertTrue(torch.equal(outputs.audio_codes, codes))
        self.assertTrue(torch.equal(reconstructed, outputs.audio_values))
        self.assertTrue(torch.equal(codes, same_codes))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            self.codec.encode(torch.zeros(0, device=torch_device))
        with self.assertRaises(ValueError):
            self.codec.encode(torch.zeros(2, 0, device=torch_device))
        with self.assertRaises(ValueError):
            self.codec.decode(torch.zeros(1, 3, 2, dtype=torch.long, device=torch_device))
        with self.assertRaises(ValueError):
            self.codec.decode(torch.full((1, 4, 2), 9, dtype=torch.long, device=torch_device))

    def test_streaming_decode_matches_offline(self):
        """Every chunk, the first and the final flush included, returns exactly the samples of the offline decode
        that no later frame changes; empty chunks leave the stream untouched."""
        codes = torch.randint(0, 9, (2, 4, 7), device=torch_device)
        with torch.no_grad():
            offline = self.codec.decode(codes).audio_values
            out = self.codec.decode(codes[..., :0], use_streaming=True, finalize=False)
            self.assertIsNone(out.overlap_cache.audio)
            chunks = [out.audio_values]
            for start, end in ((0, 1), (1, 1), (1, 4), (4, 7)):
                out = self.codec.decode(
                    codes[..., start:end], out.decoder_past_key_values, out.overlap_cache, finalize=False
                )
                chunks.append(out.audio_values)
                self.assertEqual(out.overlap_cache.audio.shape, (2, 960))
                self.assertEqual(out.overlap_cache.envelope.shape, (960,))
                self.assertEqual(out.decoder_past_key_values.get_seq_length(), end * 6)
            out = self.codec.decode(codes[..., :0], out.decoder_past_key_values, out.overlap_cache, finalize=True)
            chunks.append(out.audio_values)
        self.assertEqual([chunk.shape[-1] for chunk in chunks], [0, 1920 - 480, 0, 3 * 1920, 3 * 1920, 480])
        self.assertIsNone(out.overlap_cache.audio)
        torch.testing.assert_close(torch.cat(chunks, dim=-1), offline, atol=1e-6, rtol=0)
        with torch.no_grad():
            first = self.codec.decode(codes[..., :3], use_streaming=True, finalize=False)
            last = self.codec.decode(codes[..., 3:], first.decoder_past_key_values, first.overlap_cache)
        self.assertEqual((first.audio_values.shape[-1], last.audio_values.shape[-1]), (3 * 1920 - 480, 4 * 1920 + 480))
        torch.testing.assert_close(torch.cat([first.audio_values, last.audio_values], -1), offline, atol=1e-6, rtol=0)

    def test_quantizer_matches_the_original_fsq(self):
        """Codes and indices are those of the original implementation: bound, round, mixed-radix index."""
        quantizer = self.codec.quantizer
        levels, basis = quantizer.levels, quantizer.basis
        latents = torch.randn(2, 16, 5, device=torch_device) * 3
        with torch.no_grad():
            projected = quantizer.project_in(latents.movedim(1, -1)).reshape(2, 5, 4, 2)
            half_levels = (levels - 1) * (1 + 1e-3) / 2
            offset = torch.where(levels % 2 == 0, 0.5, 0.0)
            codes = (torch.tanh(projected + torch.atanh(offset / half_levels)) * half_levels - offset).round()
            expected = ((codes + levels // 2) * basis).sum(-1).long().transpose(1, 2)
            indices = quantizer.encode(latents)
            self.assertEqual(indices.dtype, torch.long)
            self.assertTrue(torch.equal(indices, expected))
            # the codebook round trip is exact; the latent projections are not inverses of each other
            self.assertTrue(torch.equal(quantizer(quantizer._indices_to_codes(expected))[1].long(), expected))
            self.assertEqual(quantizer.decode(expected).shape, (2, 16, 5))

    def test_released_checkpoint_keys_load(self):
        """The released codec stores the encoder backbone under `encoder.lfm.*`."""
        state_dict = {
            key.replace("encoder.", "encoder.lfm.", 1) if key.startswith("encoder.") else key: value
            for key, value in self.codec.state_dict().items()
        }
        self.assertNotIn("encoder.lfm.embed_tokens.weight", state_dict)
        with tempfile.TemporaryDirectory() as tmp:
            self.codec.config.save_pretrained(tmp)
            safetensors.torch.save_file(state_dict, os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
            reloaded, info = LiquidCodecModel.from_pretrained(tmp, output_loading_info=True)
        self.assertEqual({k: v for k, v in info.items() if v}, {})
        for key, value in self.codec.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), reloaded.state_dict()[key].cpu()), key)

    def test_attention_implementation_reaches_the_backbones(self):
        def implementations(model):
            return (
                model.config._attn_implementation,
                model.encoder.config._attn_implementation,
                model.decoder.lfm.config._attn_implementation,
            )

        self.assertEqual(
            implementations(LiquidCodecModel._from_config(self.config, attn_implementation="eager")), ("eager",) * 3
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.codec.save_pretrained(tmp)
            reloaded = LiquidCodecModel.from_pretrained(tmp, attn_implementation="eager")
        self.assertEqual(implementations(reloaded), ("eager",) * 3)
        reloaded.set_attn_implementation("sdpa")
        self.assertEqual(implementations(reloaded), ("sdpa",) * 3)


@require_torch
@slow
class LiquidCodecIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "LiquidAI/Liquid-Codec"

    def test_chirp_codes_and_audio(self):
        codec = LiquidCodecModel.from_pretrained(self.model_id, dtype=torch.float32)
        codec = codec.to(torch_device).eval()
        # a 0.5 s linear chirp from 100 Hz to 4 kHz at half scale: 7 codec frames, no dataset needed
        sampling_rate = codec.config.sampling_rate
        t = torch.arange(sampling_rate // 2, dtype=torch.float64) / sampling_rate
        chirp = (0.5 * torch.sin(2 * math.pi * (100 * t + 3900 * t**2))).float()[None].to(torch_device)
        with torch.no_grad():
            codes = codec.encode(chirp).audio_codes
            audio = codec.decode(codes).audio_values
            chunks, past_key_values, overlap_cache = [], None, None
            for i in range(codes.shape[-1]):
                out = codec.decode(
                    codes[..., i : i + 1], past_key_values, overlap_cache, use_streaming=True, finalize=False
                )
                chunks.append(out.audio_values)
                past_key_values, overlap_cache = out.decoder_past_key_values, out.overlap_cache
            chunks.append(codec.decode(codes[..., :0], past_key_values, overlap_cache, finalize=True).audio_values)

        # two codes sit on an FSQ boundary and round to the neighbouring level on CPU
        expected_codes = Expectations(
            {
                (None, None): [
                    [716, 1989, 757, 1490, 1029, 1994, 1141],
                    [194, 586, 1999, 254, 980, 1935, 585],
                    [1567, 1840, 1337, 1545, 1168, 412, 1281],
                    [503, 125, 1702, 581, 1054, 539, 439],
                    [416, 1478, 528, 1516, 535, 1247, 1238],
                    [1721, 1109, 1791, 1623, 1750, 1453, 996],
                    [1061, 722, 377, 1057, 475, 554, 1056],
                    [312, 577, 1310, 439, 576, 649, 1296],
                    [337, 1315, 1183, 1008, 618, 675, 1419],
                    [491, 1797, 363, 1005, 1732, 860, 603],
                    [1238, 1014, 338, 1183, 350, 677, 961],
                    [486, 943, 46, 599, 485, 1037, 658],
                ],
                ("cpu", None): [
                    [716, 1989, 757, 1490, 1029, 1994, 1141],
                    [194, 586, 1999, 254, 980, 1935, 585],
                    [1567, 1840, 1337, 1545, 1168, 412, 1281],
                    [503, 125, 1702, 917, 1054, 539, 439],
                    [416, 1478, 528, 1516, 535, 911, 1238],
                    [1721, 1109, 1791, 1623, 1750, 1453, 996],
                    [1061, 722, 377, 1057, 475, 554, 1056],
                    [312, 577, 1310, 439, 576, 649, 1296],
                    [337, 1315, 1183, 1008, 618, 675, 1419],
                    [491, 1797, 363, 1005, 1732, 860, 603],
                    [1238, 1014, 338, 1183, 350, 677, 961],
                    [486, 943, 46, 599, 485, 1037, 658],
                ],
            }
        ).get_expectation()
        self.assertEqual(codes[0].tolist(), expected_codes)
        self.assertEqual(audio.shape, (1, 7 * 1920))
        expected_slice = Expectations(
            {
                (None, None): [
                    -0.015839,
                    -0.016292,
                    -0.012318,
                    -0.00509,
                    0.003429,
                    0.010891,
                    0.015217,
                    0.015341,
                    0.011331,
                    0.004363,
                ],
                ("cpu", None): [
                    -0.017624,
                    -0.01696,
                    -0.011672,
                    -0.003301,
                    0.005867,
                    0.013302,
                    0.01693,
                    0.015876,
                    0.01054,
                    0.002469,
                ],
            }
        ).get_expectation()
        torch.testing.assert_close(audio[0, 6000:6010].cpu(), torch.tensor(expected_slice), atol=1e-3, rtol=0)
        torch.testing.assert_close(torch.cat(chunks, dim=-1), audio, atol=1e-5, rtol=0)
