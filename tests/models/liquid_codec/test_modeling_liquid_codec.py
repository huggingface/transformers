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

import json
import tempfile
import unittest
from pathlib import Path

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    LiquidCodecConfig,
    LiquidCodecDecoderConfig,
    LiquidCodecEncoderConfig,
    LiquidCodecFeatureExtractor,
    LiquidCodecModel,
    is_torch_available,
    set_seed,
)
from transformers.audio_utils import load_audio
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class LiquidCodecModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_frames=3,
        is_training=False,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        embedder_hidden_size=8,
        sliding_window=8,
        max_position_embeddings=512,
        quantization_dim=16,
        num_quantizers=4,
        quantization_levels=[3, 3],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.embedder_hidden_size = embedder_hidden_size
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.quantization_dim = quantization_dim
        self.num_quantizers = num_quantizers
        self.quantization_levels = quantization_levels

    def get_backbone_config(self, attention_type):
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_types": ["conv", attention_type],
            "rope_parameters": {attention_type: {"rope_type": "default", "rope_theta": 1_000_000.0}},
            "quantization_dim": self.quantization_dim,
        }

    def get_config(self):
        return LiquidCodecConfig(
            encoder_config={
                **self.get_backbone_config("full_attention"),
                "embedder_hidden_size": self.embedder_hidden_size,
            },
            decoder_config={**self.get_backbone_config("sliding_attention"), "sliding_window": self.sliding_window},
            quantization_dim=self.quantization_dim,
            num_quantizers=self.num_quantizers,
            quantization_levels=self.quantization_levels,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        feature_extractor = LiquidCodecFeatureExtractor()
        frame_size = feature_extractor.frame_stack * feature_extractor.hop_length

        # partial frames; last clip one frame shorter
        num_samples = [(self.num_frames - 1) * frame_size + 100] * self.batch_size
        num_samples[-1] -= frame_size
        clips = [floats_tensor([length], scale=1.0) for length in num_samples]

        inputs_dict = feature_extractor(
            clips, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", device=torch_device
        )
        return config, dict(inputs_dict)

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()

    def create_and_check_model_forward(self, config, inputs_dict):
        model = LiquidCodecModel(config=config).to(torch_device).eval()
        frame_size = config.decoder_config.frame_stack * config.decoder_config.hop_length

        with torch.no_grad():
            result = model(**inputs_dict)

        self.parent.assertEqual(result.audio_codes.shape, (self.batch_size, self.num_quantizers, self.num_frames))
        self.parent.assertTrue(
            bool((result.audio_codes >= 0).all() and (result.audio_codes < config.codebook_size).all())
        )
        self.parent.assertEqual(result.audio_values.shape, (self.batch_size, self.num_frames * frame_size))
        self.parent.assertEqual(result.audio_codes_mask.sum(-1).tolist(), [self.num_frames, self.num_frames - 1])
        self.parent.assertEqual(
            result.audio_values_mask.sum(-1).tolist(),
            [self.num_frames * frame_size, (self.num_frames - 1) * frame_size],
        )

        # padded clip == clip alone
        num_valid = int(inputs_dict["input_features_mask"][-1].sum())
        with torch.no_grad():
            alone = model(inputs_dict["input_features"][-1:, :, :num_valid])

        self.parent.assertIsNone(alone.audio_values_mask)
        self.parent.assertTrue(torch.equal(result.audio_codes[-1:, :, : self.num_frames - 1], alone.audio_codes))
        num_samples = alone.audio_values.shape[-1]
        torch.testing.assert_close(result.audio_values[-1:, :num_samples], alone.audio_values, atol=1e-6, rtol=0)
        self.parent.assertTrue(bool((result.audio_values[-1, num_samples:] == 0).all()))

        # forward == encode + decode, tuples too
        with torch.no_grad():
            encoded = model.encode(**inputs_dict, return_dict=False)
            decoded = model.decode(*encoded, return_dict=False)

        self.parent.assertIsInstance(encoded, tuple)
        self.parent.assertIsInstance(decoded, tuple)
        self.parent.assertTrue(torch.equal(encoded[0], result.audio_codes))
        self.parent.assertTrue(torch.equal(encoded[1], result.audio_codes_mask))
        self.parent.assertTrue(torch.equal(decoded[0], result.audio_values))
        self.parent.assertTrue(torch.equal(decoded[1], result.audio_values_mask))


@require_torch
class LiquidCodecModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LiquidCodecModel,) if is_torch_available() else ()
    _is_composite = True
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = LiquidCodecModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=LiquidCodecConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        set_seed(0)
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    @unittest.skip(reason="No input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="No attentions returned")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="No hidden states returned")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="No hidden states nor attentions returned")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Sliding-window mask always materialized: no flash kernel")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_model_outputs_equivalence(self):
        # common test subtracts tuple and dict values: fails on bool masks
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()

            with torch.no_grad():
                tuple_output = model(**self._prepare_for_class(inputs_dict, model_class), return_dict=False)
                dict_output = model(**self._prepare_for_class(inputs_dict, model_class), return_dict=True)

            self.assertIsInstance(tuple_output, tuple)
            self.assertEqual(len(tuple_output), len(dict_output.to_tuple()))
            for tuple_value, dict_value in zip(tuple_output, dict_output.to_tuple()):
                torch.testing.assert_close(tuple_value, dict_value)

    def test_streaming_decode_matches_offline(self):
        set_seed(0)
        config = self.model_tester.get_config()
        model = LiquidCodecModel(config).to(torch_device).eval()
        frame_stack, hop_length = config.decoder_config.frame_stack, config.decoder_config.hop_length
        frame_size, overlap = frame_stack * hop_length, config.decoder_config.n_fft - hop_length
        codes = ids_tensor([2, config.num_quantizers, 7], config.codebook_size)

        with torch.no_grad():
            offline = model.decode(codes).audio_values

            chunks, caches = [], {}
            for start, end in ((0, 1), (1, 4), (4, 7)):
                out = model.decode(codes[..., start:end], use_streaming=True, **caches)
                caches = {"decoder_cache": out.decoder_cache, "overlap_add_cache": out.overlap_add_cache}
                chunks.append(out.audio_values)

                self.assertIsNone(out.audio_values_mask)
                self.assertEqual(out.overlap_add_cache.audio.shape, (2, overlap))
                self.assertEqual(out.overlap_add_cache.envelope.shape, (2, overlap))
                self.assertEqual(out.decoder_cache.get_seq_length(), end * frame_stack)
            flushed = out.overlap_add_cache.flush()
            # no mask in, no mask out
            self.assertIsNone(flushed.audio_values_mask)
            chunks.append(flushed.audio_values)

        # half the overlap held back until flush
        self.assertEqual(
            [chunk.shape[-1] for chunk in chunks],
            [frame_size - overlap // 2, 3 * frame_size, 3 * frame_size, overlap // 2],
        )
        self.assertIsNone(out.overlap_add_cache.audio)
        torch.testing.assert_close(torch.cat(chunks, dim=-1), offline, atol=1e-6, rtol=0)
        with self.assertRaisesRegex(ValueError, "Nothing to flush"):
            out.overlap_add_cache.flush()

        # flushed cache starts a new stream
        with torch.no_grad():
            again = model.decode(codes, overlap_add_cache=out.overlap_add_cache, use_streaming=True)

        self.assertIs(again.overlap_add_cache, out.overlap_add_cache)
        audio_values = torch.cat([again.audio_values, again.overlap_add_cache.flush().audio_values], dim=-1)
        torch.testing.assert_close(audio_values, offline, atol=1e-6, rtol=0)

    def test_streaming_decode_with_mask_matches_offline(self):
        set_seed(0)
        config = self.model_tester.get_config()
        model = LiquidCodecModel(config).to(torch_device).eval()
        frame_size = config.decoder_config.frame_stack * config.decoder_config.hop_length
        codes = ids_tensor([2, config.num_quantizers, 7], config.codebook_size)

        # 2nd stream ends: mid-chunk, at a chunk end, in the 1st chunk, before its 1st frame
        for length in (3, 4, 1, 0):
            with self.subTest(length=length):
                codes_mask = torch.arange(7, device=torch_device) < torch.tensor([[7], [length]], device=torch_device)

                with torch.no_grad():
                    offline = model.decode(codes, codes_mask)

                    chunks, masks, caches = [], [], {}
                    for start, end in ((0, 2), (2, 4), (4, 7)):
                        out = model.decode(
                            codes[..., start:end], codes_mask[:, start:end], use_streaming=True, **caches
                        )
                        caches = {"decoder_cache": out.decoder_cache, "overlap_add_cache": out.overlap_add_cache}
                        chunks.append(out.audio_values)
                        masks.append(out.audio_values_mask)
                    flushed = out.overlap_add_cache.flush()
                    last_audio_values, last_audio_values_mask = flushed.audio_values, flushed.audio_values_mask

                audio_values = torch.cat([*chunks, last_audio_values], dim=-1)
                audio_values_mask = torch.cat([*masks, last_audio_values_mask], dim=-1)
                torch.testing.assert_close(audio_values, offline.audio_values, atol=1e-6, rtol=0)

                if length > 0:
                    with torch.no_grad():
                        alone = model.decode(codes[1:, :, :length]).audio_values
                    torch.testing.assert_close(audio_values[1:, : length * frame_size], alone, atol=1e-6, rtol=0)

                self.assertTrue(torch.equal(audio_values_mask, offline.audio_values_mask))
                self.assertEqual(offline.audio_values_mask.sum(-1).tolist(), [7 * frame_size, length * frame_size])

    def test_masks_must_be_right_padded(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = LiquidCodecModel(config).to(torch_device).eval()
        codes = ids_tensor([2, config.num_quantizers, 3], config.codebook_size)
        codes_mask = torch.tensor([[True, True, True], [True, False, False]], device=torch_device)

        with torch.no_grad():
            with self.assertRaisesRegex(ValueError, "The mask must be right-padded"):
                model.encode(inputs_dict["input_features"], inputs_dict["input_features_mask"].flip(-1))
            with self.assertRaisesRegex(ValueError, "The mask must be right-padded"):
                model.decode(codes, codes_mask.flip(-1))

            # an ended stream can't resume
            out = model.decode(codes, codes_mask, use_streaming=True)
            with self.assertRaisesRegex(ValueError, "does not resume"):
                model.decode(
                    codes,
                    torch.ones_like(codes_mask),
                    decoder_cache=out.decoder_cache,
                    overlap_add_cache=out.overlap_add_cache,
                    use_streaming=True,
                )

    def test_config_without_encoder_or_decoder(self):
        config = self.model_tester.get_config()
        config.encoder_config = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            config = LiquidCodecConfig.from_pretrained(tmp_dir)

        self.assertIsNone(config.encoder_config)
        self.assertIsInstance(config.decoder_config, LiquidCodecDecoderConfig)
        self.assertIsInstance(LiquidCodecConfig().encoder_config, LiquidCodecEncoderConfig)

        # halves must share quantization_dim
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "quantization_dim=16"):
            LiquidCodecConfig(decoder_config={"quantization_dim": 8}, quantization_dim=16)

    def test_model_without_encoder_or_decoder(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = LiquidCodecModel(config).to(torch_device).eval()

        with torch.no_grad():
            codes = model.encode(inputs_dict["input_features"]).audio_codes
            expected = model.decode(codes).audio_values

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            encoder_only, encoder_info = LiquidCodecModel.from_pretrained(
                tmp_dir, decoder_config=None, output_loading_info=True
            )
            decoder_only, decoder_info = LiquidCodecModel.from_pretrained(
                tmp_dir, encoder_config=None, output_loading_info=True
            )

        for info in (encoder_info, decoder_info):
            self.assertEqual({k: v for k, v in info.items() if v}, {})
        self.assertIsNone(encoder_only.decoder)
        self.assertIsNone(decoder_only.encoder)

        with torch.no_grad():
            encoded = encoder_only.to(torch_device).encode(inputs_dict["input_features"]).audio_codes
            self.assertTrue(torch.equal(encoded, codes))
            torch.testing.assert_close(decoder_only.to(torch_device).decode(codes).audio_values, expected)

            with self.assertRaisesRegex(ValueError, "without its decoder"):
                encoder_only.decode(codes)
            with self.assertRaisesRegex(ValueError, "without its encoder"):
                decoder_only.encode(inputs_dict["input_features"])


@require_torch
class LiquidCodecIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    Goldens: Hub remote code of LiquidAI/LFM3-Audio-2B @ 2d6c1f99, float32. Per clip: codes, decoded audio at 100
    random positions.
    reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8

    float32, not the checkpoint's bfloat16: there 4-7 % of the reference's own codes flip vs its float32 ones.
    Spectrogram on the codec's device, as the reference: on CPU, 3/888 and 1/456 codes flip.
    """

    # 2x largest diff measured (A100; CI A10G pending): 3.4e-6 vs reference, 4.2e-6 streamed vs offline
    RTOL = 0
    ATOL = 1e-5

    checkpoint = "eustlb/Liquid-Codec"
    revision = "refs/pr/1"
    samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
    clips = ["librispeech_mr_quilter.wav", "dude_where_is_my_car.wav"]
    other_num_samples = 24000 * 3 + 777  # second clip cut to 3 s

    @classmethod
    def setUpClass(cls):
        fixture = Path(__file__).parent.parent.parent / "fixtures/liquid_codec/expected_results.json"
        with open(fixture, encoding="utf-8") as f:
            expected = json.load(f)["codec"]

        cls.expected_codes = expected["audio_codes"]
        cls.expected_audio_values_indices = expected["audio_values_indices"]
        cls.expected_audio_values = expected["audio_values"]

    @slow
    @require_torch_accelerator
    def test_encode(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-codec-py
        """
        model = AutoModel.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.checkpoint, revision=self.revision)

        clips = [
            torch.tensor(load_audio(self.samples + self.clips[0], sampling_rate=24000)),
            torch.tensor(load_audio(self.samples + self.clips[1], sampling_rate=24000))[: self.other_num_samples],
        ]

        expected_codes = [torch.tensor(codes) for codes in self.expected_codes]
        # Hann window built on GPU (reference: CPU, then moved): 1 float32 step -> 1 code off by 1 FSQ level
        expected_codes[0][7, 35] = 780  # golden: 1116

        for index, clip in enumerate(clips):
            inputs = feature_extractor([clip], sampling_rate=24000, return_tensors="pt", device=model.device)
            inputs = inputs.to(model.device, model.dtype)

            with torch.no_grad():
                encoded = model.encode(**inputs)

            self.assertEqual(encoded.audio_codes[0].tolist(), expected_codes[index].tolist())

    @slow
    @require_torch_accelerator
    def test_encode_batch(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-codec-py
        """
        model = AutoModel.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.checkpoint, revision=self.revision)

        clips = [
            torch.tensor(load_audio(self.samples + self.clips[0], sampling_rate=24000)),
            torch.tensor(load_audio(self.samples + self.clips[1], sampling_rate=24000))[: self.other_num_samples],
        ]

        expected_codes = [torch.tensor(codes) for codes in self.expected_codes]
        # Hann window built on GPU (reference: CPU, then moved): 1 float32 step -> 1 code off by 1 FSQ level
        expected_codes[0][7, 35] = 780  # golden: 1116

        inputs = feature_extractor(clips, sampling_rate=24000, return_tensors="pt", device=model.device)
        inputs = inputs.to(model.device, model.dtype)

        with torch.no_grad():
            encoded = model.encode(**inputs)

        for index in range(len(clips)):
            audio_codes = encoded.audio_codes[index][:, encoded.audio_codes_mask[index]]
            self.assertEqual(audio_codes.tolist(), expected_codes[index].tolist())

    @slow
    def test_decode(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-codec-py
        """
        model = AutoModel.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )

        for index, expected_codes in enumerate(self.expected_codes):
            audio_codes = torch.tensor(expected_codes, device=torch_device)[None]

            with torch.no_grad():
                decoded = model.decode(audio_codes)

            self.assertIsNone(decoded.audio_values_mask)
            audio_values = decoded.audio_values[0].cpu()
            self.assertEqual(audio_values.shape, (audio_codes.shape[-1] * 1920,))
            torch.testing.assert_close(
                audio_values[self.expected_audio_values_indices[index]],
                torch.tensor(self.expected_audio_values[index]),
                rtol=self.RTOL,
                atol=self.ATOL,
            )

    @slow
    def test_decode_batch(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-codec-py
        """
        model = AutoModel.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )

        # golden codes, right-padded
        num_frames = torch.tensor([len(codes[0]) for codes in self.expected_codes], device=torch_device)
        audio_codes = torch.zeros(2, len(self.expected_codes[0]), int(num_frames.max()), dtype=torch.long)
        for index, expected_codes in enumerate(self.expected_codes):
            audio_codes[index, :, : num_frames[index]] = torch.tensor(expected_codes)
        audio_codes = audio_codes.to(torch_device)
        audio_codes_mask = torch.arange(audio_codes.shape[-1], device=torch_device) < num_frames[:, None]

        with torch.no_grad():
            decoded = model.decode(audio_codes, audio_codes_mask)

        self.assertTrue(bool((decoded.audio_values[~decoded.audio_values_mask] == 0).all()))

        for index in range(len(self.expected_codes)):
            audio_values = decoded.audio_values[index][decoded.audio_values_mask[index]].cpu()
            self.assertEqual(audio_values.shape, (int(num_frames[index]) * 1920,))
            torch.testing.assert_close(
                audio_values[self.expected_audio_values_indices[index]],
                torch.tensor(self.expected_audio_values[index]),
                rtol=self.RTOL,
                atol=self.ATOL,
            )

    @slow
    def test_decode_streaming(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-codec-py
        """
        model = AutoModel.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )

        for index, expected_codes in enumerate(self.expected_codes):
            audio_codes = torch.tensor(expected_codes, device=torch_device)[None]

            chunks, caches = [], {}
            with torch.no_grad():
                for start in range(0, audio_codes.shape[-1], 4):
                    out = model.decode(audio_codes[..., start : start + 4], use_streaming=True, **caches)
                    caches = {"decoder_cache": out.decoder_cache, "overlap_add_cache": out.overlap_add_cache}
                    chunks.append(out.audio_values)
                last_audio_values = out.overlap_add_cache.flush().audio_values

                offline = model.decode(audio_codes)

            audio_values = torch.cat([*chunks, last_audio_values], dim=-1)
            torch.testing.assert_close(audio_values, offline.audio_values, rtol=self.RTOL, atol=self.ATOL)

            audio_values = audio_values[0].cpu()
            self.assertEqual(audio_values.shape, (audio_codes.shape[-1] * 1920,))
            torch.testing.assert_close(
                audio_values[self.expected_audio_values_indices[index]],
                torch.tensor(self.expected_audio_values[index]),
                rtol=self.RTOL,
                atol=self.ATOL,
            )

    @slow
    def test_decode_streaming_batch(self):
        """
        reproducer: https://gist.github.com/eustlb/1befc95bf557e88c808d61c9f7b199d8#file-reproducer-codec-py
        """
        model = AutoModel.from_pretrained(
            self.checkpoint, revision=self.revision, dtype=torch.float32, device_map=torch_device
        )

        # golden codes, right-padded
        num_frames = torch.tensor([len(codes[0]) for codes in self.expected_codes], device=torch_device)
        audio_codes = torch.zeros(2, len(self.expected_codes[0]), int(num_frames.max()), dtype=torch.long)
        for index, expected_codes in enumerate(self.expected_codes):
            audio_codes[index, :, : num_frames[index]] = torch.tensor(expected_codes)
        audio_codes = audio_codes.to(torch_device)
        audio_codes_mask = torch.arange(audio_codes.shape[-1], device=torch_device) < num_frames[:, None]

        chunks, masks, caches = [], [], {}
        with torch.no_grad():
            for start in range(0, audio_codes.shape[-1], 4):
                out = model.decode(
                    audio_codes[..., start : start + 4],
                    audio_codes_mask[:, start : start + 4],
                    use_streaming=True,
                    **caches,
                )
                caches = {"decoder_cache": out.decoder_cache, "overlap_add_cache": out.overlap_add_cache}
                chunks.append(out.audio_values)
                masks.append(out.audio_values_mask)
            flushed = out.overlap_add_cache.flush()
            last_audio_values, last_audio_values_mask = flushed.audio_values, flushed.audio_values_mask

            offline = model.decode(audio_codes, audio_codes_mask)

        audio_values = torch.cat([*chunks, last_audio_values], dim=-1)
        audio_values_mask = torch.cat([*masks, last_audio_values_mask], dim=-1)
        torch.testing.assert_close(audio_values, offline.audio_values, rtol=self.RTOL, atol=self.ATOL)
        self.assertTrue(torch.equal(audio_values_mask, offline.audio_values_mask))

        for index in range(len(self.expected_codes)):
            clip_audio_values = audio_values[index][audio_values_mask[index]].cpu()
            self.assertEqual(clip_audio_values.shape, (int(num_frames[index]) * 1920,))
            torch.testing.assert_close(
                clip_audio_values[self.expected_audio_values_indices[index]],
                torch.tensor(self.expected_audio_values[index]),
                rtol=self.RTOL,
                atol=self.ATOL,
            )
