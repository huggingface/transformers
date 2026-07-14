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
import os
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

    from transformers import WavTokenizerFeatureExtractor, WavTokenizerModel


def randomize_codebook(model, seed=0):
    """`_init_weights` zero-inits the VQ codebook (all entries tie, argmin returns 0 everywhere).
    Randomize it deterministically so encode tests exercise real code assignment."""
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(seed)
        model.quantizer.codebook.embed.copy_(torch.randn(model.quantizer.codebook.embed.shape, generator=generator))
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
    all_model_classes = (WavTokenizerModel,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_resize_embeddings = False
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
        quantizer_codes = model.quantizer.encode(embeddings)
        self.assertGreater(quantizer_codes.unique().numel(), 1)

    def test_decode_output_length(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = WavTokenizerModel(config).to(torch_device).eval()
        num_codes = 7
        audio_codes = torch.randint(0, config.codebook_size, (2, 1, num_codes), device=torch_device)
        with torch.no_grad():
            audio_values = model.decode(audio_codes).audio_values
        self.assertEqual(audio_values.shape, (2, 1, num_codes * config.hop_length))

    def test_decode_single_code_raises(self):
        """Inputs of at most `hop_length` samples yield one code; the decoder's GroupNorm cannot process it."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = WavTokenizerModel(config).to(torch_device).eval()
        audio_codes = torch.zeros(1, 1, 1, dtype=torch.long, device=torch_device)
        with self.assertRaises(ValueError):
            model.decode(audio_codes)

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
        batch = [floats_tensor([length], scale=1.0).numpy() for length in lengths]
        inputs = feature_extractor(batch, sampling_rate=config.sampling_rate, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            out = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"])
        mask_sums = out.audio_codes_mask.sum(dim=-1).flatten().tolist()
        self.assertEqual(mask_sums, [feature_extractor.get_num_audio_codes(length) for length in lengths])

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
    """Integration tests against a converted real checkpoint.

    Set `WAVTOKENIZER_HF_CHECKPOINT` to a converted model dir (output of
    `convert_wavtokenizer_checkpoint.py`) or a Hub repo id. For full bit-parity verification against the
    ORIGINAL implementation, run `scripts/check_wavtokenizer_parity.py` instead (needs the original repo).
    """

    # Golden codes for a 0.5 s, 440 Hz, -6 dBFS sine at 24 kHz (first 10 of 20 codes), frozen from the
    # converted `wavtokenizer_large_unify_600_24k.ckpt` — verified bit-exact against the original
    # implementation by scripts/check_wavtokenizer_parity.py (2026-07-14).
    EXPECTED_FIRST_CODES: list[int] | None = [1323, 1442, 3524, 2056, 3229, 1723, 2785, 1389, 3144, 1723]

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = os.environ.get("WAVTOKENIZER_HF_CHECKPOINT")
        if cls.checkpoint is None:
            raise unittest.SkipTest("WAVTOKENIZER_HF_CHECKPOINT not set (converted checkpoint required)")

    def _sine(self, seconds=0.5, freq=440.0, sampling_rate=24000):
        t = torch.arange(int(seconds * sampling_rate)) / sampling_rate
        return (0.5 * torch.sin(2 * torch.pi * freq * t))[None, None, :]

    def test_real_checkpoint_encode_decode(self):
        model = WavTokenizerModel.from_pretrained(self.checkpoint).to(torch_device).eval()
        waveform = self._sine().to(torch_device)
        with torch.no_grad():
            codes = model.encode(waveform).audio_codes
            audio = model.decode(codes).audio_values

        num_expected = waveform.shape[-1] // model.config.hop_length
        self.assertEqual(codes.shape, (1, 1, num_expected))
        self.assertEqual(audio.shape, (1, 1, num_expected * model.config.hop_length))
        # a real codebook must produce diverse codes on a sine
        self.assertGreater(codes.unique().numel(), 1)

        if self.EXPECTED_FIRST_CODES is not None:
            self.assertEqual(codes[0, 0, :10].cpu().tolist(), self.EXPECTED_FIRST_CODES)
