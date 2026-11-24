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

import json
import tempfile
import unittest
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import (
    AutoFeatureExtractor,
    YuE2AcousticTokenizerConfig,
    YuE2AcousticTokenizerDecoderConfig,
)
from transformers.testing_utils import is_torch_available, require_soundfile, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import (
        AutoModel,
        YuE2AcousticTokenizerDecoderModel,
        YuE2AcousticTokenizerEncoderModel,
        YuE2AcousticTokenizerModel,
    )


class YuE2AcousticTokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_frames=5,
        audio_channels=2,
        hidden_sizes=[2, 2, 2, 2, 2, 2, 2],
        downsampling_ratios=[2, 2, 2, 2, 2, 2],
        latent_dim=6,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.audio_channels = audio_channels
        self.hidden_sizes = hidden_sizes
        self.downsampling_ratios = downsampling_ratios
        self.latent_dim = latent_dim
        self.is_training = is_training

    def get_config(self):
        half_config = {
            "audio_channels": self.audio_channels,
            "hidden_sizes": self.hidden_sizes,
            "downsampling_ratios": self.downsampling_ratios,
            "latent_dim": self.latent_dim,
        }
        encoder_config = {**half_config, "latent_dim": 2 * self.latent_dim}
        return YuE2AcousticTokenizerConfig(encoder_config=encoder_config, decoder_config=half_config)

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_values = floats_tensor([self.batch_size, self.audio_channels, self.num_frames * config.hop_length])
        return config, {"input_values": input_values}

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()

    def create_and_check_model_forward(self, config, inputs_dict):
        model = YuE2AcousticTokenizerModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            result = model(**inputs_dict)
        self.parent.assertEqual(result.latents.shape, (self.batch_size, self.num_frames, self.latent_dim))
        # the transposed convolutions of the decoder crop `kernel_size - stride` samples per block
        self.parent.assertEqual(result.audio_values.shape[:2], (self.batch_size, self.audio_channels))


@require_torch
class YuE2AcousticTokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (YuE2AcousticTokenizerModel,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = YuE2AcousticTokenizerModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=YuE2AcousticTokenizerConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_encode_sample(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = YuE2AcousticTokenizerModel(config).to(torch_device).eval()
        with torch.no_grad():
            mean = model.encode(inputs_dict["input_values"]).latents
            sampled = model.encode(inputs_dict["input_values"], sample=True).latents
        self.assertEqual(sampled.shape, mean.shape)
        self.assertFalse(torch.allclose(sampled, mean))

    def test_config_without_encoder_or_decoder(self):
        config = YuE2AcousticTokenizerConfig(encoder_config=None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            config = YuE2AcousticTokenizerConfig.from_pretrained(tmp_dir)
        self.assertIsNone(config.encoder_config)
        self.assertIsInstance(config.decoder_config, YuE2AcousticTokenizerDecoderConfig)

        with self.assertRaises(StrictDataclassClassValidationError):
            YuE2AcousticTokenizerConfig(encoder_config=None, decoder_config=None)

    def _save_full_model(self, tmp_dir):
        """Save a full model to `tmp_dir` and return its inputs and reference outputs."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_values = inputs_dict["input_values"]
        model = YuE2AcousticTokenizerModel(config).to(torch_device).eval()
        with torch.no_grad():
            latents = model.encode(input_values).latents
            audio_values = model.decode(latents).audio_values
        model.save_pretrained(tmp_dir)
        return input_values, latents, audio_values

    @parameterized.expand(
        [
            ("full_class_config_dropped", YuE2AcousticTokenizerModel, {"decoder_config": None}),
            ("half_class", YuE2AcousticTokenizerEncoderModel, {}),
        ]
    )
    def test_model_without_decoder(self, _, model_class, kwargs):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_values, latents, _ = self._save_full_model(tmp_dir)
            encoder, loading_info = model_class.from_pretrained(tmp_dir, output_loading_info=True, **kwargs)

        # no missing, unexpected or mismatched weights
        self.assertFalse(any(loading_info.values()), loading_info)

        # reproduces the full model's latents (the full class exposes `encode`, the half class is called directly)
        encoder.to(torch_device).eval()
        encode = encoder.encode if hasattr(encoder, "encode") else encoder
        with torch.no_grad():
            torch.testing.assert_close(encode(input_values).latents, latents)

    def test_model_without_decoder_cannot_decode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, latents, _ = self._save_full_model(tmp_dir)
            encoder_only = YuE2AcousticTokenizerModel.from_pretrained(tmp_dir, decoder_config=None)

        with self.assertRaises(ValueError):
            encoder_only.decode(latents)

    @parameterized.expand(
        [
            ("full_class_config_dropped", YuE2AcousticTokenizerModel, {"encoder_config": None}),
            ("half_class", YuE2AcousticTokenizerDecoderModel, {}),
        ]
    )
    def test_model_without_encoder(self, _, model_class, kwargs):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, latents, audio_values = self._save_full_model(tmp_dir)
            decoder, loading_info = model_class.from_pretrained(tmp_dir, output_loading_info=True, **kwargs)

        # no missing, unexpected or mismatched weights
        self.assertFalse(any(loading_info.values()), loading_info)

        # reproduces the full model's audio (the full class exposes `decode`, the half class is called directly)
        decoder.to(torch_device).eval()
        decode = decoder.decode if hasattr(decoder, "decode") else decoder
        with torch.no_grad():
            torch.testing.assert_close(decode(latents).audio_values, audio_values)

    def test_model_without_encoder_cannot_encode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_values, _, _ = self._save_full_model(tmp_dir)
            decoder_only = YuE2AcousticTokenizerModel.from_pretrained(tmp_dir, encoder_config=None)

        with self.assertRaises(ValueError):
            decoder_only.encode(input_values)

    @unittest.skip(reason="The acoustic tokenizer has no token embeddings.")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="The acoustic tokenizer is convolutional: it has no attention.")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The acoustic tokenizer is convolutional: it records no hidden states.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="The acoustic tokenizer is convolutional: it has no attention nor recorded hidden states.")
    def test_retain_grad_hidden_states_attentions(self):
        pass


@slow
@require_torch
@require_soundfile
class YuE2AcousticTokenizerIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    checkpoint = "m-a-p/YuE2-Vae"
    # adds `preprocessor_config.json`: https://huggingface.co/m-a-p/YuE2-Vae/discussions/1
    revision = "9a63959a7e34304bd31ef7bf9940ba03ee978c51"
    songs_checkpoint = "m-a-p/YuE2-3B"

    @classmethod
    def setUpClass(cls):
        cls.feature_extractor = AutoFeatureExtractor.from_pretrained(cls.checkpoint, revision=cls.revision)
        cls.model = AutoModel.from_pretrained(cls.checkpoint, revision=cls.revision, device_map=torch_device)

    def setUp(self):
        super().setUp()

        fixture_name = self._testMethodName.removeprefix("test_")
        path = (
            Path(__file__).parent.parent.parent
            / "fixtures/yue2_acoustic_tokenizer"
            / f"expected_results_{fixture_name}.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            expected = json.load(f)
        self.input_spec = expected["input"]
        self.songs_revision = expected["metadata"]["songs_revision"]
        self.expected = {
            key: torch.tensor(value) for key, value in expected.items() if key.startswith(("latents", "audio"))
        }

    def _load_clip(self, spec):
        """The excerpt described by `spec` of an example song of m-a-p/YuE2-3B: (2, num_samples), 48 kHz stereo."""
        import soundfile as sf

        path = hf_hub_download(self.songs_checkpoint, f"assets/audio/{spec['song']}", revision=self.songs_revision)
        audio, _ = sf.read(
            path,
            start=int(spec["start_seconds"] * 48000),
            frames=spec["num_samples"],
            dtype="float32",
            always_2d=True,
        )
        return audio.T.copy()

    def test_encode_decode(self):
        """
        reproducer: https://gist.github.com/ebezzam/d03a9cc1d2bfc31c39c8e3704f73e5bc#file-reproducer_codec-py
        """
        clip = self._load_clip(self.input_spec)
        inputs = self.feature_extractor(clip, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt")

        with torch.no_grad():
            latents = self.model.encode(inputs["input_values"].to(torch_device)).latents
            audio = self.model.decode(latents).audio_values

        latents_head, audio_head = self.expected["latents_head"], self.expected["audio_head"]
        torch.testing.assert_close(latents[0, : latents_head.shape[0]].cpu(), latents_head)
        torch.testing.assert_close(audio[0, :, : audio_head.shape[-1]].cpu(), audio_head)

    def test_encode_decode_batch(self):
        """
        reproducer: https://gist.github.com/ebezzam/d03a9cc1d2bfc31c39c8e3704f73e5bc#file-reproducer_codec_batch-py
        """
        clips = [self._load_clip(spec) for spec in self.input_spec]
        inputs = self.feature_extractor(clips, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt")

        with torch.no_grad():
            latents = self.model.encode(inputs["input_values"].to(torch_device)).latents
            audio = self.model.decode(latents).audio_values

        # the tail of the shorter excerpt lies in its padding, which the non-causal model's outputs depend on
        latents_head, latents_tail = self.expected["latents_head"], self.expected["latents_tail"]
        audio_head, audio_tail = self.expected["audio_head"], self.expected["audio_tail"]
        torch.testing.assert_close(latents[:, : latents_head.shape[1]].cpu(), latents_head)
        torch.testing.assert_close(latents[:, -latents_tail.shape[1] :].cpu(), latents_tail)
        torch.testing.assert_close(audio[..., : audio_head.shape[-1]].cpu(), audio_head)
        torch.testing.assert_close(audio[..., -audio_tail.shape[-1] :].cpu(), audio_tail)
