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

import tempfile
import unittest
from pathlib import Path

from transformers import (
    AutoFeatureExtractor,
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
        latent_stride=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.upsampling_ratios = upsampling_ratios
        self.latent_stride = latent_stride
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.is_training = is_training
        # the transformers run at the SEANet rate: `latent_stride` positions per latent frame
        self.seq_length = num_frames * latent_stride

    def get_config(self):
        return PocketTTSAudioTokenizerConfig(
            latent_dim=self.latent_dim,
            hidden_size=self.hidden_size,
            num_filters=self.num_filters,
            upsampling_ratios=self.upsampling_ratios,
            latent_stride=self.latent_stride,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            upsample_groups=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_values = floats_tensor([self.batch_size, 1, self.num_frames * config.frame_size], scale=0.1)
        padding_mask = torch.ones_like(input_values[:, 0], dtype=torch.long)
        # the second waveform is a frame and 10 samples shorter, right-padded: 4 latent frames instead of 5
        padding_mask[1, -config.frame_size - 10 :] = 0
        input_values[1, :, -config.frame_size - 10 :] = 0
        return config, {"input_values": input_values, "padding_mask": padding_mask}

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()

    def create_and_check_model_forward(self, config, inputs_dict):
        model = PocketTTSAudioTokenizerModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            result = model(**inputs_dict)
        self.parent.assertEqual(result.latents.shape, (self.batch_size, self.num_frames, self.latent_dim))
        self.parent.assertListEqual(result.latents_mask.sum(-1).tolist(), [self.num_frames, self.num_frames - 1])
        self.parent.assertEqual(
            result.audio_values.shape, (self.batch_size, config.audio_channels, self.num_frames * config.frame_size)
        )

        # the model is causal: the right padding leaves the latents of the shorter waveform unchanged
        num_frames = self.num_frames - 1
        with torch.no_grad():
            single = model.encode(inputs_dict["input_values"][1:, :, : num_frames * config.frame_size]).latents
        torch.testing.assert_close(result.latents[1:, :num_frames], single, atol=1e-5, rtol=1e-5)


@require_torch
class PocketTTSAudioTokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PocketTTSAudioTokenizerModel,) if is_torch_available() else ()
    test_resize_embeddings = False

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

    @unittest.skip(reason="The audio tokenizer has no token embeddings.")
    def test_model_get_set_embeddings(self):
        pass

    def test_config_without_encoder_or_decoder(self):
        config = self.model_tester.get_config()
        self.assertNotIn("encoder_config", config.to_diff_dict())

        config.encoder_config = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            config = PocketTTSAudioTokenizerConfig.from_pretrained(tmp_dir)
        self.assertIsNone(config.encoder_config)
        self.assertIsInstance(config.decoder_config, PocketTTSAudioTokenizerDecoderConfig)

        with self.assertRaises(ValueError):
            config.decoder_config = None
        with self.assertRaises(ValueError):
            config.encoder_config = PocketTTSAudioTokenizerEncoderConfig()

    def test_model_without_encoder_or_decoder(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_values = inputs_dict["input_values"]
        model = PocketTTSAudioTokenizerModel(config).to(torch_device).eval()
        with torch.no_grad():
            latents = model.encode(input_values).latents
            audio_values = model.decode(latents).audio_values

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            encoder_only, encoder_info = PocketTTSAudioTokenizerModel.from_pretrained(
                tmp_dir, decoder_config=None, output_loading_info=True
            )
            decoder_only, decoder_info = PocketTTSAudioTokenizerModel.from_pretrained(
                tmp_dir, encoder_config=None, output_loading_info=True
            )
            # each half also loads on its own from the full checkpoint
            _, encoder_half_info = PocketTTSAudioTokenizerEncoderModel.from_pretrained(
                tmp_dir, output_loading_info=True
            )
            _, decoder_half_info = PocketTTSAudioTokenizerDecoderModel.from_pretrained(
                tmp_dir, output_loading_info=True
            )
        for loading_info in (encoder_info, decoder_info, encoder_half_info, decoder_half_info):
            self.assertFalse(any(loading_info.values()), loading_info)

        encoder_only.to(torch_device).eval()
        decoder_only.to(torch_device).eval()
        with torch.no_grad():
            torch.testing.assert_close(encoder_only.encode(input_values).latents, latents)
            torch.testing.assert_close(decoder_only.decode(latents).audio_values, audio_values)
        with self.assertRaises(ValueError):
            encoder_only.decode(latents)
        with self.assertRaises(ValueError):
            decoder_only.encode(input_values)


@require_torch
class PocketTTSAudioTokenizerIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """reproducer: ~/audio-model-work/pocket_tts/reproducers (`run_reproducers.sh` regenerates every fixture from the original)"""

    checkpoint = "eustlb/pocket-tts-audio-tokenizer"
    bucket = "hf-internal-testing/pocket-tts-integration-tests"

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

        The feature extractor zero-pads the waveform to a multiple of the frame size, as the original does. The decoder
        is fed the original's latents, so that each half is checked on its own.
        """
        expected = self._load_expected_tensors("codec")
        model = AutoModel.from_pretrained(self.checkpoint, device_map=torch_device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.checkpoint)
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
        expected = self._load_expected_tensors("codec_batch")
        model = AutoModel.from_pretrained(self.checkpoint, device_map=torch_device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.checkpoint)
        audio = [expected["input_audio_0"].numpy(), expected["input_audio_1"].numpy()]
        inputs = feature_extractor(audio, sampling_rate=24000, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            encoded = model.encode(**inputs)
            decoded = model.decode(encoded.latents).audio_values
        for row in range(2):
            num_frames = int(encoded.latents_mask[row].sum())
            self.assertEqual(num_frames, expected[f"latents_{row}"].shape[0])
            # same tolerance as the single sample
            torch.testing.assert_close(
                encoded.latents[row, :num_frames].cpu(), expected[f"latents_{row}"], atol=5e-5, rtol=0
            )
            num_samples = expected[f"audio_{row}"].shape[0]
            # the decoder is causal: the padded latents after the row's frames do not change its samples
            torch.testing.assert_close(
                decoded[row, 0, :num_samples].cpu(), expected[f"audio_{row}"], atol=1e-4, rtol=0
            )
