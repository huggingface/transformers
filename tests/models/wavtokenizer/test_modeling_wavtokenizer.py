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

import inspect
import math
import os
import tempfile
import unittest

from transformers import WavTokenizerConfig
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import WavTokenizerEncoderModel, WavTokenizerFeatureExtractor, WavTokenizerModel


def randomize_codebook(model, seed=0):
    """`_init_weights` zero-inits the VQ codebook (all entries tie, argmin returns 0 everywhere).
    Randomize it deterministically so encode tests exercise real code assignment."""
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(seed)
        codebook = model.base_model.quantizer.codebook.embed
        codebook.copy_(torch.randn(codebook.shape, generator=generator))
    return model


@require_torch
class WavTokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=1,
        sample_rate=24000,
        num_filters=8,
        upsampling_ratios=(2, 2),
        hidden_size=32,
        codebook_size=64,
        codebook_dim=32,
        decoder_hidden_size=32,
        decoder_intermediate_size=64,
        decoder_num_layers=2,
        decoder_attention_num_groups=8,
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

        self.hop_length = 1
        for ratio in upsampling_ratios:
            self.hop_length *= ratio
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
        model = WavTokenizerModel(config=config).to(torch_device).eval()
        result = model(inputs_dict["input_values"])
        self.parent.assertEqual(
            result.audio_values.shape,
            (self.batch_size, self.num_channels, self.num_samples),
        )


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

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_encoder_model_matches_full_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        full_model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        encoder_model = full_model.encoder_model

        self.assertIsInstance(encoder_model, WavTokenizerEncoderModel)
        self.assertIs(full_model.base_model, encoder_model)

        padding_mask = torch.ones(inputs_dict["input_values"].shape[0], inputs_dict["input_values"].shape[-1])
        padding_mask[0, -self.model_tester.hop_length :] = 0
        input_values = inputs_dict["input_values"].to(torch_device)
        padding_mask = padding_mask.to(torch_device)
        with torch.no_grad():
            full_output = full_model.encode(input_values, padding_mask=padding_mask)
            encoder_output = encoder_model(input_values, padding_mask=padding_mask)

        torch.testing.assert_close(encoder_output.audio_codes, full_output.audio_codes, rtol=0, atol=0)
        torch.testing.assert_close(encoder_output.audio_codes_mask, full_output.audio_codes_mask, rtol=0, atol=0)
        self.assertFalse(hasattr(encoder_model, "backbone"))
        self.assertFalse(hasattr(encoder_model, "head"))

    def test_checkpoint_key_layout(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        full_model = WavTokenizerModel(config)
        encoder_model = WavTokenizerEncoderModel(config)

        full_keys = set(full_model.state_dict())
        self.assertTrue(any(key.startswith("encoder_model.encoder.") for key in full_keys))
        self.assertTrue(any(key.startswith("encoder_model.quantizer.") for key in full_keys))
        self.assertFalse(any(key.startswith(("encoder.", "quantizer.")) for key in full_keys))
        encoder_keys = set(encoder_model.state_dict())
        self.assertTrue(any(key.startswith("encoder.") for key in encoder_keys))
        self.assertTrue(any(key.startswith("quantizer.") for key in encoder_keys))
        self.assertFalse(any(key.startswith("encoder_model.") for key in encoder_keys))

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
        input_values = inputs_dict["input_values"].to(torch_device)
        with torch.no_grad():
            expected = full_model.encode(input_values).audio_codes
            actual = encoder_model(input_values).audio_codes
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_values", "padding_mask"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

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
            input_values = floats_tensor([1, 1, num_samples], scale=1.0).to(torch_device)
            with torch.no_grad():
                audio_codes = model.encode(input_values).audio_codes
            self.assertEqual(
                audio_codes.shape[-1],
                feature_extractor.get_num_audio_codes(num_samples),
                f"code count mismatch for num_samples={num_samples}",
            )

    def test_encode_codes_range_and_dtype(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        with torch.no_grad():
            audio_codes = model.encode(inputs_dict["input_values"].to(torch_device)).audio_codes
        self.assertEqual(audio_codes.dtype, torch.int64)
        self.assertEqual(audio_codes.shape[1], 1)  # single codebook
        self.assertGreaterEqual(audio_codes.min().item(), 0)
        self.assertLess(audio_codes.max().item(), config.codebook_size)
        # the randomized codebook must assign diverse codes to diverse embeddings (argmin is not degenerate)
        generator = torch.Generator(device="cpu").manual_seed(1)
        embeddings = torch.randn(1, config.hidden_size, 50, generator=generator).to(torch_device)
        quantizer_codes = model.encoder_model.quantizer.encode(embeddings)
        self.assertGreater(quantizer_codes.unique().numel(), 1)

    def test_decode_output_length(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = WavTokenizerModel(config).to(torch_device).eval()
        num_codes = 7
        audio_codes = torch.randint(0, config.codebook_size, (2, 1, num_codes), device=torch_device)
        with torch.no_grad():
            audio_values = model.decode(audio_codes).audio_values
        self.assertEqual(audio_values.shape, (2, 1, num_codes * config.hop_length))

    def test_released_checkpoint_geometries(self):
        """Both released hop geometries must construct, encode, and decode with config-derived lengths."""
        for upsampling_ratios, expected_hop in [([6, 5, 5, 4], 600), ([8, 5, 4, 2], 320)]:
            with self.subTest(upsampling_ratios=upsampling_ratios):
                config = WavTokenizerConfig(
                    num_filters=4,
                    upsampling_ratios=upsampling_ratios,
                    hidden_size=64,
                    codebook_size=64,
                    codebook_dim=64,
                    decoder_hidden_size=32,
                    decoder_intermediate_size=64,
                    decoder_num_layers=2,
                    decoder_attention_num_groups=8,
                )
                model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
                input_values = floats_tensor([1, 1, 2 * expected_hop + 1], scale=1.0).to(torch_device)
                with torch.no_grad():
                    codes = model.encode(input_values).audio_codes
                    decoded = model.decode(codes).audio_values
                self.assertEqual(config.hop_length, expected_hop)
                self.assertEqual(codes.shape, (1, 1, 3))
                self.assertEqual(decoded.shape, (1, 1, 3 * expected_hop))

    def test_decode_single_code(self):
        """A single code is valid with the production architecture's multiple channels per GroupNorm group."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = WavTokenizerModel(config).to(torch_device).eval()
        audio_codes = torch.zeros(1, 1, 1, dtype=torch.long, device=torch_device)
        self.assertGreater(config.decoder_hidden_size // config.decoder_attention_num_groups, 1)
        with torch.no_grad():
            audio_values = model.decode(audio_codes).audio_values
        self.assertEqual(audio_values.shape, (1, 1, config.hop_length))

    def test_forward_single_code_inputs(self):
        """Waveforms up to one hop encode to one code and can be reconstructed."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        for num_samples in [1, config.hop_length - 1, config.hop_length]:
            input_values = floats_tensor([1, 1, num_samples], scale=1.0).to(torch_device)
            with torch.no_grad():
                output = model(input_values)
            self.assertEqual(output.audio_codes.shape, (1, 1, 1))
            self.assertEqual(output.audio_values.shape, input_values.shape)

    def test_encode_deterministic(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        input_values = inputs_dict["input_values"].to(torch_device)
        with torch.no_grad():
            codes_1 = model.encode(input_values).audio_codes
            codes_2 = model.encode(input_values).audio_codes
        self.assertTrue(torch.equal(codes_1, codes_2))

    def test_encode_batched_matches_single(self):
        """Same-length samples encoded in a batch must produce the same codes as encoded individually."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        input_values = inputs_dict["input_values"].to(torch_device)
        with torch.no_grad():
            batched = model.encode(input_values).audio_codes
            singles = [model.encode(input_values[i : i + 1]).audio_codes for i in range(input_values.shape[0])]
        self.assertTrue(torch.equal(batched, torch.cat(singles, dim=0)))

    def test_ragged_batch_codes_mask(self):
        """padding_mask -> audio_codes_mask marks exactly ceil(valid_len / hop) codes per sample."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        feature_extractor = WavTokenizerFeatureExtractor(
            sampling_rate=config.sampling_rate, hop_length=config.hop_length
        )
        hop = config.hop_length
        lengths = [2 * hop, 5 * hop - 1, 9 * hop + 1]
        batch = [floats_tensor([length], scale=1.0).cpu().numpy() for length in lengths]
        inputs = feature_extractor(batch, sampling_rate=config.sampling_rate, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            out = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"])

        num_codes = out.audio_codes.shape[-1]
        expected_masks = []
        for length in lengths:
            valid_codes = feature_extractor.get_num_audio_codes(length)
            expected_masks.append([1] * valid_codes + [0] * (num_codes - valid_codes))
        self.assertEqual(out.audio_codes_mask[:, 0].tolist(), expected_masks)

    def test_left_padded_batch_codes_mask(self):
        """Left-padded inputs must mark valid audio codes at the end of each sequence."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = randomize_codebook(WavTokenizerModel(config)).to(torch_device).eval()
        feature_extractor = WavTokenizerFeatureExtractor(
            sampling_rate=config.sampling_rate, hop_length=config.hop_length, padding_side="left"
        )
        hop = config.hop_length
        lengths = [2 * hop, 5 * hop - 1, 9 * hop + 1]
        batch = [floats_tensor([length], scale=1.0).cpu().numpy() for length in lengths]
        inputs = feature_extractor(batch, sampling_rate=config.sampling_rate, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            out = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"])

        num_codes = out.audio_codes.shape[-1]
        expected_masks = []
        for length in lengths:
            valid_codes = feature_extractor.get_num_audio_codes(length)
            expected_masks.append([0] * (num_codes - valid_codes) + [1] * valid_codes)
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
