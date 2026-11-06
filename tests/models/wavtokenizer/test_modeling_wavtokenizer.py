# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch WavTokenizer model."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from transformers import AutoFeatureExtractor, WavTokenizerConfig
from transformers.audio_utils import load_audio
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_processing_common import url_to_local_path
from ...utils.test_audio_utils import compute_rmse


if is_torch_available():
    import torch

    from transformers import WavTokenizerEncoderModel, WavTokenizerFeatureExtractor, WavTokenizerModel


def randomize_codebook(model, seed=0):
    """`_init_weights` zero-inits the VQ codebook, so every frame would get code 0. Fill it with perturbed encoder
    frames of a random waveform so that codes depend on the input."""
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(seed)
        encoder_model = model.base_model
        codebook = encoder_model.quantizer.codebook.embed
        codebook_size = codebook.shape[0]
        waveform = torch.rand(1, 1, codebook_size * encoder_model.hop_length, generator=generator) * 2 - 1
        frames = encoder_model.encoder(waveform)[0].transpose(0, 1)[:codebook_size]
        codebook.copy_(frames + 0.1 * frames.std() * torch.randn(frames.shape, generator=generator))
    return model


@require_torch
class WavTokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=1,
        sample_rate=24000,
        num_filters=2,
        # four stages like the released checkpoints; the odd ratio exercises the asymmetric conv padding branch
        upsampling_ratios=(3, 2, 2, 2),
        hidden_size=32,
        codebook_size=64,
        codebook_dim=32,
        decoder_hidden_size=32,
        decoder_intermediate_size=64,
        decoder_num_layers=2,
        decoder_attention_num_groups=8,  # several channels per GroupNorm group, as in the released decoder
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.num_filters = num_filters
        self.upsampling_ratios = upsampling_ratios
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_intermediate_size = decoder_intermediate_size
        self.decoder_num_layers = decoder_num_layers
        self.decoder_attention_num_groups = decoder_attention_num_groups
        self.is_training = is_training

        self.hop_length = math.prod(upsampling_ratios)
        self.num_samples = self.hop_length * 25

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.num_samples], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return WavTokenizerConfig(
            sampling_rate=self.sample_rate,
            audio_channels=self.num_channels,
            num_filters=self.num_filters,
            upsampling_ratios=self.upsampling_ratios,
            hidden_size=self.hidden_size,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_intermediate_size=self.decoder_intermediate_size,
            decoder_num_layers=self.decoder_num_layers,
            decoder_attention_num_groups=self.decoder_attention_num_groups,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        """One code per started hop, each decoding to `hop_length` samples, and a reconstruction sliced back to the
        input length: for a single sample, an exact multiple of the hop and a length that is not."""
        model = WavTokenizerModel(config=config).to(torch_device).eval()
        hop = config.hop_length
        for num_samples in [1, hop, self.num_samples - 1]:
            with self.parent.subTest(num_samples=num_samples):
                input_values = inputs_dict["input_values"][..., :num_samples]
                num_codes = math.ceil(num_samples / hop)
                with torch.no_grad():
                    result = model(input_values)
                    decoded = model.decode(result.audio_codes).audio_values
                self.parent.assertEqual(result.audio_codes.shape, (self.batch_size, 1, num_codes))
                self.parent.assertEqual(result.audio_codes.dtype, torch.long)
                self.parent.assertEqual(decoded.shape, (self.batch_size, 1, num_codes * hop))
                self.parent.assertEqual(result.audio_values.shape, input_values.shape)


@require_torch
class WavTokenizerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (WavTokenizerEncoderModel, WavTokenizerModel) if is_torch_available() else ()
    is_encoder_decoder = True
    test_resize_embeddings = False
    test_torch_exportable = False  # data-dependent control flow in `_pad1d` (`if length <= max_pad`)
    pipeline_model_mapping = {"feature-extraction": WavTokenizerModel} if is_torch_available() else {}

    def setUp(self):
        self.model_tester = WavTokenizerModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=WavTokenizerConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_encoder_model_loads_full_checkpoint(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        full_model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            full_model.save_pretrained(tmp_dir)
            encoder_model, loading_info = WavTokenizerEncoderModel.from_pretrained(tmp_dir, output_loading_info=True)
        encoder_model = encoder_model.to(torch_device).eval()

        self.assertFalse(any(loading_info.values()), loading_info)
        # the encoder model's `forward` must also route `padding_mask` through to the codes mask
        input_values = inputs_dict["input_values"]
        padding_mask = torch.ones(input_values.shape[0], input_values.shape[-1], device=torch_device)
        padding_mask[0, -config.hop_length :] = 0
        with torch.no_grad():
            expected = full_model.encode(input_values, padding_mask=padding_mask)
            actual = encoder_model(input_values, padding_mask=padding_mask)
        torch.testing.assert_close(actual.audio_codes, expected.audio_codes, rtol=0, atol=0)
        torch.testing.assert_close(actual.audio_codes_mask, expected.audio_codes_mask, rtol=0, atol=0)

    def test_batched_forward_matches_single(self):
        """Samples in a batch must not influence each other, in the encoder or the decoder. The common batching test
        cannot see this: with the zero-initialized codebook, every output is independent of the input."""
        torch.manual_seed(0)
        config = self.model_tester.get_config()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        generator = torch.Generator().manual_seed(0)
        shape = (self.model_tester.batch_size, 1, self.model_tester.num_samples)
        input_values = (2 * torch.rand(shape, generator=generator) - 1).to(torch_device)
        with torch.no_grad():
            batched = model(input_values)
            singles = [model(sample.unsqueeze(0)) for sample in input_values]
        # guard against degenerate codes or a decoder that ignores them, which would make the comparison vacuous
        self.assertGreater(batched.audio_codes.unique().numel(), 1)
        self.assertFalse(torch.allclose(batched.audio_values[0], batched.audio_values[1]))
        torch.testing.assert_close(batched.audio_codes, torch.cat([single.audio_codes for single in singles]))
        torch.testing.assert_close(
            batched.audio_values, torch.cat([single.audio_values for single in singles]), rtol=1e-4, atol=1e-5
        )

    def test_ragged_batch_codes_mask(self):
        """padding_mask -> audio_codes_mask marks exactly ceil(valid_len / hop) codes per sample, aligned to the
        padding side: at the start for right padding and at the end for left padding."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = WavTokenizerModel(config).to(torch_device).eval()
        hop = config.hop_length
        lengths = [2 * hop, 5 * hop - 1, 9 * hop + 1]
        batch = [floats_tensor([length], scale=1.0).cpu().numpy() for length in lengths]
        for padding_side in ["right", "left"]:
            with self.subTest(padding_side=padding_side):
                feature_extractor = WavTokenizerFeatureExtractor(
                    sampling_rate=config.sampling_rate, hop_length=hop, padding_side=padding_side
                )
                inputs = feature_extractor(batch, sampling_rate=config.sampling_rate, return_tensors="pt").to(
                    torch_device
                )
                with torch.no_grad():
                    out = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"])

                num_codes = out.audio_codes.shape[-1]
                valid_codes = [feature_extractor.get_num_audio_codes(length) for length in lengths]
                expected = [[1] * valid + [0] * (num_codes - valid) for valid in valid_codes]
                if padding_side == "left":
                    expected = [row[::-1] for row in expected]
                self.assertEqual(out.audio_codes_mask[:, 0].tolist(), expected)

    @unittest.skip(reason="WavTokenizer has no input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="WavTokenizer outputs expose neither `hidden_states` nor `attentions`")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="WavTokenizer takes no `output_attentions`")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="WavTokenizer takes no `output_hidden_states`")
    def test_hidden_states_output(self):
        pass


@slow
@require_torch
class WavTokenizerIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    The expected values are the outputs of the original implementation (jishengpeng/WavTokenizer) with the released
    `wavtokenizer_large_unify_600_24k.ckpt`, on native 24 kHz speech (no resampling involved). They come from a batch
    of both clips; row 0 is the longest clip, so it is not padded and also serves as the single-clip reference.

    reproducer: https://gist.github.com/RaphaelKreft/ea70c883c9c89a60681ada25414d5873
    """

    checkpoint = "swiss-ai/wavtokenizer-large-unify-40token"
    audio_urls = [
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Carter_man.wav",
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Frank_man.wav",
    ]
    # the decoded audio matches the original up to 2.1e-7 on CPU
    ATOL = 1e-5
    # the codec error is a mean over the whole clip, so it only absorbs the decoded-audio tolerance above
    CODEC_ERROR_RTOL = 1e-4

    def setUp(self):
        super().setUp()
        with open(
            Path(__file__).parent.parent.parent / "fixtures/wavtokenizer/expected_results.json", encoding="utf-8"
        ) as file:
            self.expected = json.load(file)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.checkpoint)
        self.audios = [
            load_audio(url_to_local_path(url), sampling_rate=self.feature_extractor.sampling_rate)
            for url in self.audio_urls
        ]

    def test_integration(self):
        model = WavTokenizerModel.from_pretrained(self.checkpoint, device_map=torch_device)
        inputs = self.feature_extractor(
            self.audios[0], sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        ).to(torch_device)

        audio_codes = model.encode(**inputs).audio_codes
        self.assertEqual(audio_codes[0, 0].tolist(), self.expected["audio_codes"][0])

        audio_values = model.decode(audio_codes).audio_values
        expected_values = torch.tensor(self.expected["audio_values"][0])
        torch.testing.assert_close(
            audio_values[0, 0, : len(expected_values)].cpu(), expected_values, rtol=0, atol=self.ATOL
        )
        codec_error = compute_rmse(inputs["input_values"], audio_values).item()
        torch.testing.assert_close(codec_error, self.expected["codec_errors"][0], rtol=self.CODEC_ERROR_RTOL, atol=0)

        # `forward` equals `decode(encode())`, sliced back to the input length
        forward_values = model(**inputs).audio_values
        torch.testing.assert_close(forward_values, audio_values[..., : forward_values.shape[-1]], rtol=0, atol=0)

        # the encoder model alone, loaded from the same full checkpoint, gives the same codes
        encoder_model = WavTokenizerEncoderModel.from_pretrained(self.checkpoint, device_map=torch_device)
        torch.testing.assert_close(encoder_model(**inputs).audio_codes, audio_codes, rtol=0, atol=0)

    def test_batch_integration(self):
        """
        Both clips go through the feature extractor as one right-padded batch. The encoder does not mask the padding, so
        the trailing codes of the shorter clip depend on it, as in the original: compare the codes that
        `audio_codes_mask` marks as valid, and each row's codec error over its valid samples.
        """
        expected = self.expected
        model = WavTokenizerModel.from_pretrained(self.checkpoint, device_map=torch_device)
        inputs = self.feature_extractor(
            self.audios, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        ).to(torch_device)

        encoded = model.encode(**inputs)
        audio_values = model.decode(encoded.audio_codes).audio_values
        num_values = len(expected["audio_values"][0])
        for row in range(len(self.audios)):
            valid_codes = encoded.audio_codes[row, 0][encoded.audio_codes_mask[row, 0].bool()]
            self.assertEqual(valid_codes.tolist(), expected["audio_codes"][row])
            torch.testing.assert_close(
                audio_values[row, 0, :num_values].cpu(),
                torch.tensor(expected["audio_values"][row]),
                rtol=0,
                atol=self.ATOL,
            )
            num_samples = int(inputs["padding_mask"][row].sum())
            codec_error = compute_rmse(
                inputs["input_values"][row, ..., :num_samples], audio_values[row, ..., :num_samples]
            ).item()
            torch.testing.assert_close(codec_error, expected["codec_errors"][row], rtol=self.CODEC_ERROR_RTOL, atol=0)
