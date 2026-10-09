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

import math
import os
import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import WavTokenizerConfig
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import WavTokenizerEncoderModel, WavTokenizerFeatureExtractor, WavTokenizerModel


def randomize_codebook(model, seed=0):
    """`_init_weights` zero-inits the VQ codebook (all entries tie, argmin returns 0 everywhere).

    Fill it deterministically with perturbed encoder frames of a random waveform so the entries sit at the scale
    of the encoder output and encode tests exercise real, non-degenerate code assignment. A plain `randn` codebook
    is not enough: the randomly initialized encoder emits small activations, so argmin then picks the single
    smallest-norm entry for every frame."""
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
        """The reconstruction is sliced back to the input length, from a single hop (or less) up to many hops."""
        model = randomize_codebook(WavTokenizerModel(config=config)).to(torch_device).eval()
        hop = config.hop_length
        for num_samples in [1, hop - 1, hop, self.num_samples]:
            with self.parent.subTest(num_samples=num_samples):
                input_values = inputs_dict["input_values"][..., :num_samples]
                with torch.no_grad():
                    result = model(input_values)
                self.parent.assertEqual(result.audio_values.shape, input_values.shape)
                self.parent.assertEqual(result.audio_codes.shape, (self.batch_size, 1, math.ceil(num_samples / hop)))


@require_torch
class WavTokenizerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (WavTokenizerEncoderModel, WavTokenizerModel) if is_torch_available() else ()
    is_encoder_decoder = True
    test_resize_embeddings = False
    test_torch_exportable = False  # data-dependent control flow in `_pad1d` (`if length <= max_pad`)
    pipeline_model_mapping = {"feature-extraction": WavTokenizerModel} if is_torch_available() else {}

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = WavTokenizerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=WavTokenizerConfig,
            num_filters=8,
            hidden_size=32,
            codebook_dim=32,
            common_properties=[],
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_rejects_indivisible_decoder_groups(self):
        """The decoder GroupNorm layers need `decoder_hidden_size` to be a multiple of the group count."""
        config = self.model_tester.get_config()
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "must be divisible by"):
            WavTokenizerConfig(**{**config.to_dict(), "decoder_hidden_size": 40, "decoder_attention_num_groups": 16})

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

        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        self.assertFalse(loading_info["mismatched_keys"])
        # the encoder model's `forward` must also route `padding_mask` through to the codes mask
        input_values = inputs_dict["input_values"].to(torch_device)
        padding_mask = torch.ones(input_values.shape[0], input_values.shape[-1], device=torch_device)
        padding_mask[0, -config.hop_length :] = 0
        with torch.no_grad():
            expected = full_model.encode(input_values, padding_mask=padding_mask)
            actual = encoder_model(input_values, padding_mask=padding_mask)
        torch.testing.assert_close(actual.audio_codes, expected.audio_codes, rtol=0, atol=0)
        torch.testing.assert_close(actual.audio_codes_mask, expected.audio_codes_mask, rtol=0, atol=0)

    def test_encode_frame_count_matches_feature_extractor(self):
        """The feature-extractor-predicted code count must equal the encoder output for arbitrary lengths.
        Downstream models (apertus1p5) rely on this to expand audio placeholders to the exact code count."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        feature_extractor = WavTokenizerFeatureExtractor(
            sampling_rate=config.sampling_rate, hop_length=config.hop_length
        )
        hop = config.hop_length
        for num_samples in [1, hop - 1, hop, hop + 1, 3 * hop, 3 * hop + hop // 2, 100 * hop + 1]:
            with self.subTest(num_samples=num_samples):
                input_values = floats_tensor([1, 1, num_samples], scale=1.0).to(torch_device)
                with torch.no_grad():
                    audio_codes = model.encode(input_values).audio_codes
                # a single codebook of int64 ids, one per hop
                self.assertEqual(audio_codes.dtype, torch.int64)
                self.assertEqual(audio_codes.shape, (1, 1, feature_extractor.get_num_audio_codes(num_samples)))

    def test_decode_output_length(self):
        """Each code decodes to `hop_length` samples, down to a single code of a single sample (the GroupNorm
        edge case)."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = WavTokenizerModel(config).to(torch_device).eval()
        for batch_size, num_codes in [(1, 1), (2, 7)]:
            with self.subTest(batch_size=batch_size, num_codes=num_codes):
                audio_codes = ids_tensor([batch_size, 1, num_codes], config.codebook_size).to(torch_device)
                with torch.no_grad():
                    audio_values = model.decode(audio_codes).audio_values
                self.assertEqual(audio_values.shape, (batch_size, 1, num_codes * config.hop_length))

    def test_encode_batched_matches_single(self):
        """Same-length samples encoded in a batch must produce the same codes as encoded individually."""
        torch.manual_seed(0)
        config = self.model_tester.get_config()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        generator = torch.Generator().manual_seed(0)
        shape = (self.model_tester.batch_size, 1, self.model_tester.num_samples)
        input_values = (2 * torch.rand(shape, generator=generator) - 1).to(torch_device)
        with torch.no_grad():
            batched = model.encode(input_values).audio_codes
            singles = [model.encode(sample.unsqueeze(0)).audio_codes for sample in input_values]
        # guard against a degenerate codebook, which would make this and every other code comparison vacuous
        self.assertGreater(batched.unique().numel(), 1)
        self.assertTrue(torch.equal(batched, torch.cat(singles, dim=0)))

    def test_ragged_batch_codes_mask(self):
        """padding_mask -> audio_codes_mask marks exactly ceil(valid_len / hop) codes per sample, aligned to the
        padding side: at the start for right padding and at the end for left padding."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
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
                expected_masks = []
                for length in lengths:
                    valid_codes = feature_extractor.get_num_audio_codes(length)
                    padding = [0] * (num_codes - valid_codes)
                    valid = [1] * valid_codes
                    expected_masks.append(valid + padding if padding_side == "right" else padding + valid)
                self.assertEqual(out.audio_codes_mask[:, 0].tolist(), expected_masks)

    @unittest.skip("WavTokenizer does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("WavTokenizerModel does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="WavTokenizerModel does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="WavTokenizerModel does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass


@slow
@require_torch
class WavTokenizerIntegrationTest(unittest.TestCase):
    """Integration tests against the released checkpoint.

    `WAVTOKENIZER_HF_CHECKPOINT` overrides the default with another Hub repo id or a locally converted
    directory (the output of `convert_wavtokenizer_checkpoint.py`); `WAVTOKENIZER_CHECKPOINT_VARIANT` then
    selects which frozen golden codes to compare against, and defaults to the released checkpoint's.
    """

    DEFAULT_CHECKPOINT = "swiss-ai/wavtokenizer-large-unify-40token"

    # Golden codes for a 0.5 s, 440 Hz, -6 dBFS sine at 24 kHz (first 10 of 20 codes), frozen from the
    # converted `wavtokenizer_large_unify_600_24k.ckpt` and verified bit-exact against the original
    # implementation (2026-07-14).
    EXPECTED_FIRST_CODES: list[int] | None = [1323, 1442, 3524, 2056, 3229, 1723, 2785, 1389, 3144, 1723]

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = os.environ.get("WAVTOKENIZER_HF_CHECKPOINT", cls.DEFAULT_CHECKPOINT)
        # the golden codes belong to the default checkpoint, so only claim them when that is what is loaded
        default_variant = "large-unify-40" if cls.checkpoint == cls.DEFAULT_CHECKPOINT else None
        cls.checkpoint_variant = os.environ.get("WAVTOKENIZER_CHECKPOINT_VARIANT", default_variant)

    def _sine(self, seconds=0.5, freq=440.0, sampling_rate=24000):
        t = torch.arange(int(seconds * sampling_rate)) / sampling_rate
        return (0.5 * torch.sin(2 * torch.pi * freq * t))[None, None, :]

    def test_real_checkpoint_encode_decode(self):
        model = WavTokenizerModel.from_pretrained(self.checkpoint).to(torch_device).eval()
        encoder_model = WavTokenizerEncoderModel.from_pretrained(self.checkpoint).to(torch_device).eval()
        waveform = self._sine().to(torch_device)
        with torch.no_grad():
            codes = model.encode(waveform).audio_codes
            encoder_codes = encoder_model(waveform).audio_codes
            audio = model.decode(codes).audio_values

        num_expected = math.ceil(waveform.shape[-1] / model.config.hop_length)
        self.assertEqual(codes.shape, (1, 1, num_expected))
        self.assertEqual(audio.shape, (1, 1, num_expected * model.config.hop_length))
        self.assertEqual(codes.dtype, torch.long)
        self.assertGreaterEqual(codes.min().item(), 0)
        self.assertLess(codes.max().item(), model.config.codebook_size)
        torch.testing.assert_close(encoder_codes, codes, rtol=0, atol=0)

        if self.checkpoint_variant == "large-unify-40" and self.EXPECTED_FIRST_CODES is not None:
            self.assertEqual(codes[0, 0, :10].cpu().tolist(), self.EXPECTED_FIRST_CODES)
