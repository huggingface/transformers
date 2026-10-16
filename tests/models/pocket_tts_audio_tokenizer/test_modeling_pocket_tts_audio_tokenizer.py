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
"""Testing suite for the PyTorch Pocket TTS audio tokenizer model."""

import math
import tempfile
import unittest
from pathlib import Path

from transformers import (
    PocketTTSAudioTokenizerConfig,
    PocketTTSAudioTokenizerDecoderConfig,
    PocketTTSAudioTokenizerEncoderConfig,
)
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import (
        AutoModel,
        PocketTTSAudioTokenizerDecoderModel,
        PocketTTSAudioTokenizerEncoderModel,
        PocketTTSAudioTokenizerModel,
    )


@require_torch
class PocketTTSAudioTokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_frames=5,
        latent_dim=8,
        hidden_size=32,
        num_filters=4,
        upsampling_ratios=[4, 2],
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        latent_stride=8,
        is_training=False,
        config_class=PocketTTSAudioTokenizerConfig,
    ):
        self.parent = parent
        self.config_class = config_class
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.upsampling_ratios = upsampling_ratios
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.latent_stride = latent_stride
        self.is_training = is_training
        self.samples_per_frame = math.prod(upsampling_ratios) * latent_stride
        # the transformers run at the SEANet rate: `latent_stride` positions per latent frame
        self.seq_length = num_frames * latent_stride

    def get_config(self):
        half = dict(  # noqa: C408
            latent_dim=self.latent_dim,
            hidden_size=self.hidden_size,
            num_filters=self.num_filters,
            upsampling_ratios=self.upsampling_ratios,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            upsample_groups=self.hidden_size,
            latent_stride=self.latent_stride,
        )
        return self.config_class(**half)

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, 1, self.num_frames * self.samples_per_frame], scale=0.1)
        padding_mask = torch.ones_like(input_values[:, 0], dtype=torch.long)
        # the second waveform is a frame and 10 samples shorter, right-padded: 4 latent frames instead of 5
        padding_mask[1, -self.samples_per_frame - 10 :] = 0
        input_values[1, :, -self.samples_per_frame - 10 :] = 0
        return self.get_config(), {"input_values": input_values, "padding_mask": padding_mask}

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()

    def create_and_check_model_forward(self, config, inputs_dict):
        model = PocketTTSAudioTokenizerModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            result = model(**inputs_dict)
        self.parent.assertEqual(result.latents.shape, (self.batch_size, self.num_frames, self.latent_dim))
        self.parent.assertListEqual(result.latents_mask.sum(-1).tolist(), [self.num_frames, self.num_frames - 1])
        self.parent.assertEqual(
            result.audio_values.shape,
            (self.batch_size, config.decoder_config.audio_channels, self.num_frames * self.samples_per_frame),
        )

    def create_and_check_causal_padding(self, config, inputs_dict):
        # the convolutions and transformers are causal: right padding leaves the latents of the audio frames unchanged
        model = PocketTTSAudioTokenizerModel(config=config).to(torch_device).eval()
        input_values = inputs_dict["input_values"][:1, :, : 3 * self.samples_per_frame]
        padded = torch.nn.functional.pad(input_values, (0, 2 * self.samples_per_frame))
        with torch.no_grad():
            latents = model.encode(input_values).latents
            padded_latents = model.encode(padded).latents
        torch.testing.assert_close(padded_latents[:, :3], latents, atol=1e-5, rtol=1e-5)


@require_torch
class PocketTTSAudioTokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PocketTTSAudioTokenizerModel,) if is_torch_available() else ()
    test_resize_embeddings = False
    _is_composite = True

    def setUp(self):
        self.model_tester = PocketTTSAudioTokenizerModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=PocketTTSAudioTokenizerConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_flat_config(self):
        # the halves are derived from the shared fields, and only a dropped half is saved
        config = self.model_tester.get_config()
        self.assertNotIn("encoder_config", config.to_diff_dict())
        self.assertEqual(config.encoder_config.latent_dim, config.latent_dim)
        self.assertEqual(config.decoder_config.frame_size, config.frame_size)
        dropped = PocketTTSAudioTokenizerConfig(**{**config.to_dict(), "encoder_config": None})
        self.assertIsNone(dropped.encoder_config)
        self.assertIsNone(dropped.to_diff_dict()["encoder_config"])
        with self.assertRaisesRegex(ValueError, "an encoder, a decoder, or both"):
            PocketTTSAudioTokenizerConfig(encoder_config=None, decoder_config=None)
        for removed in ("codebook_size", "num_quantizers"):
            self.assertFalse(hasattr(config, removed), removed)

    def test_optional_halves(self):
        # a codec without encoder only decodes, and loads the decoder of a full checkpoint without warnings
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = PocketTTSAudioTokenizerModel(config).to(torch_device).eval()
        latents = torch.randn(1, 3, self.model_tester.latent_dim, device=torch_device)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            decoder_only, info = PocketTTSAudioTokenizerModel.from_pretrained(
                tmp_dir, encoder_config=None, output_loading_info=True
            )
            # each half alone, from the full checkpoint
            encoder, encoder_info = PocketTTSAudioTokenizerEncoderModel.from_pretrained(
                tmp_dir, output_loading_info=True
            )
            decoder, decoder_info = PocketTTSAudioTokenizerDecoderModel.from_pretrained(
                tmp_dir, output_loading_info=True
            )
        for loading_info in (info, encoder_info, decoder_info):
            self.assertFalse(any(loading_info.values()), loading_info)
        decoder_only.to(torch_device).eval()
        self.assertIsNone(decoder_only.encoder)
        with torch.no_grad():
            expected = model.decode(latents).audio_values
            torch.testing.assert_close(decoder_only.decode(latents).audio_values, expected)
            # the halves work on `(batch_size, channels, num_frames)` latents; `latent_proj` belongs to the full codec
            projected = model.latent_proj(latents.transpose(1, 2))
            torch.testing.assert_close(decoder.to(torch_device)(projected).audio_values, expected)
            torch.testing.assert_close(
                encoder.to(torch_device)(inputs_dict["input_values"]).latents.transpose(1, 2),
                model.encode(inputs_dict["input_values"]).latents,
            )
        with self.assertRaises(ValueError):
            decoder_only.encode(inputs_dict["input_values"])

    def test_causal_padding(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_padding(*config_and_inputs)

    @unittest.skip(reason="The audio tokenizer has no token embeddings.")
    def test_model_get_set_embeddings(self):
        pass


@require_torch
class PocketTTSAudioTokenizerEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PocketTTSAudioTokenizerEncoderModel,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PocketTTSAudioTokenizerModelTester(self, config_class=PocketTTSAudioTokenizerEncoderConfig)
        self.config_tester = ConfigTester(
            self, config_class=PocketTTSAudioTokenizerEncoderConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="The audio tokenizer has no token embeddings.")
    def test_model_get_set_embeddings(self):
        pass


class PocketTTSAudioTokenizerDecoderModelTester(PocketTTSAudioTokenizerModelTester):
    def prepare_config_and_inputs(self):
        # the decoder half takes the latents projected to `hidden_size` by the full codec's `latent_proj`
        latents = floats_tensor([self.batch_size, self.hidden_size, self.num_frames], scale=1.0)
        return self.get_config(), {"latents": latents}


@require_torch
class PocketTTSAudioTokenizerDecoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PocketTTSAudioTokenizerDecoderModel,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PocketTTSAudioTokenizerDecoderModelTester(
            self, config_class=PocketTTSAudioTokenizerDecoderConfig
        )
        self.config_tester = ConfigTester(
            self, config_class=PocketTTSAudioTokenizerDecoderConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(
        reason="The latents have `hidden_size` channels, which the test changes after preparing them (head_dim 16)."
    )
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skip(reason="The audio tokenizer has no token embeddings.")
    def test_model_get_set_embeddings(self):
        pass


@require_torch
class PocketTTSAudioTokenizerIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """reproducer: ~/audio-model-work/pocket_tts/reproducers (`run_reproducers.sh` regenerates every fixture from the original)"""

    def setUp(self):
        super().setUp()
        self.audio_tokenizer_name = "eustlb/pocket-tts-audio-tokenizer"
        self.bucket = "hf-internal-testing/pocket-tts-integration-tests"

    def _load_expected_tensors(self, case):
        from huggingface_hub import download_bucket_files
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "expected.safetensors"
            download_bucket_files(self.bucket, files=[(f"{case}/expected.safetensors", str(local_path))])
            return load_file(local_path)

    @slow
    def test_encode_decode(self):
        """
        reproducer: reproducer_codec.py (the first LibriSpeech dummy sample at 24 kHz, encoded and decoded by the original)

        The audio tokenizer, loaded on its own through `AutoModel`, encodes the waveform (which the feature extractor
        zero-pads to a multiple of the frame size, as the original does) and decodes the original's latents.
        """
        from transformers import AutoFeatureExtractor

        expected = self._load_expected_tensors("codec")
        model = AutoModel.from_pretrained(self.audio_tokenizer_name, device_map=torch_device)
        self.assertIsInstance(model, PocketTTSAudioTokenizerModel)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.audio_tokenizer_name)
        inputs = feature_extractor(expected["input_audio"].numpy(), sampling_rate=24000, return_tensors="pt")
        with torch.no_grad():
            latents = model.encode(**inputs.to(torch_device)).latents
            audio = model.decode(expected["latents"][None].to(torch_device)).audio_values
        # 3.8e-6 (CPU) / 8.1e-6 (GPU) measured on latents up to 7.4
        torch.testing.assert_close(latents[0].cpu(), expected["latents"], atol=5e-5, rtol=0)
        # 8.7e-7 (CPU) / 1.0e-6 (GPU) measured
        torch.testing.assert_close(audio[0, 0].cpu(), expected["audio"], atol=2e-5, rtol=0)

    @slow
    def test_encode_decode_batch(self):
        """
        reproducer: reproducer_codec_batch.py (two LibriSpeech samples of different lengths, each encoded and decoded on
        its own by the original)

        The two waveforms go through the feature extractor as one right-padded batch; each row's valid latents, and the
        decoding of the padded latents trimmed to the row, match the per-sample references.
        """
        from transformers import AutoFeatureExtractor

        expected = self._load_expected_tensors("codec_batch")
        model = AutoModel.from_pretrained(self.audio_tokenizer_name, device_map=torch_device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.audio_tokenizer_name)
        audio = [expected["input_audio_0"].numpy(), expected["input_audio_1"].numpy()]
        inputs = feature_extractor(audio, sampling_rate=24000, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            encoded = model.encode(**inputs)
            decoded = model.decode(encoded.latents).audio_values
        for row in range(2):
            num_frames = int(encoded.latents_mask[row].sum())
            self.assertEqual(num_frames, expected[f"latents_{row}"].shape[0])
            # 3.8e-6 (CPU) / 8.1e-6 (GPU) measured on a single sample
            torch.testing.assert_close(
                encoded.latents[row, :num_frames].cpu(), expected[f"latents_{row}"], atol=5e-5, rtol=0
            )
            num_samples = expected[f"audio_{row}"].shape[0]
            # the decoder is causal: the padded latents after the row's frames do not change its samples
            torch.testing.assert_close(
                decoded[row, 0, :num_samples].cpu(), expected[f"audio_{row}"], atol=1e-4, rtol=0
            )
