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

import inspect
import json
import unittest
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, floats_tensor
from transformers import AutoFeatureExtractor, Xcodec2Config, Xcodec2Model
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import Xcodec2Model


@require_torch
class Xcodec2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=1,
        sample_rate=16000,
        num_mel_bins=80,
        stride=2,
        encoder_hidden_size=8,
        downsampling_ratios=(2, 2, 4),
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=8,
        quantization_levels=(4, 4, 4, 4),
        semantic_hidden_size=32,
        semantic_num_hidden_layers=17,
        semantic_num_attention_heads=4,
        semantic_intermediate_size=64,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.is_training = is_training
        self.hop_length = int(np.prod(downsampling_ratios))
        self.num_samples = self.hop_length * 80  # feature extractor will pad to multiple of hop_length
        self.num_channels = num_channels
        self.num_mel_bins = num_mel_bins
        self.stride = stride
        self.mel_hop_length = self.hop_length  # match acoustic encoder's downsampling ratio
        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.quantization_levels = quantization_levels
        self.semantic_hidden_size = semantic_hidden_size
        self.semantic_num_hidden_layers = semantic_num_hidden_layers
        self.semantic_num_attention_heads = semantic_num_attention_heads
        self.semantic_intermediate_size = semantic_intermediate_size

    def prepare_config_and_inputs(self):
        audio = floats_tensor([self.batch_size, self.num_channels, self.num_samples], scale=1.0)
        audio_spectrogram = floats_tensor(
            [self.batch_size, self.num_samples // self.mel_hop_length, self.num_mel_bins * self.stride], scale=1.0
        )
        config = self.get_config()
        inputs_dict = {"audio": audio, "audio_spectrogram": audio_spectrogram}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        semantic_model_config = {
            "model_type": "wav2vec2-bert",
            "hidden_size": self.semantic_hidden_size,
            "num_hidden_layers": self.semantic_num_hidden_layers,
            "num_attention_heads": self.semantic_num_attention_heads,
            "intermediate_size": self.semantic_intermediate_size,
            "feature_projection_input_dim": self.num_mel_bins * self.stride,
            "output_hidden_size": self.semantic_hidden_size,
        }

        return Xcodec2Config(
            encoder_hidden_size=self.encoder_hidden_size,
            downsampling_ratios=self.downsampling_ratios,
            hidden_size=self.hidden_size,
            semantic_model_config=semantic_model_config,
            sampling_rate=self.sample_rate,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_hidden_layers=self.num_hidden_layers,
            head_dim=self.head_dim,
            quantization_dim=self.hidden_size + self.semantic_hidden_size,
            quantization_levels=self.quantization_levels,
            audio_channels=self.num_channels,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = Xcodec2Model(config=config).to(torch_device).eval()
        audio = inputs_dict["audio"]
        audio_spectrogram = inputs_dict["audio_spectrogram"]
        result = model(audio, audio_spectrogram)
        self.parent.assertEqual(
            result.audio_values.shape,
            (self.batch_size, self.num_channels, self.num_samples),
        )


@require_torch
class Xcodec2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Xcodec2Model,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_resize_embeddings = False
    pipeline_model_mapping = {"feature-extraction": Xcodec2Model} if is_torch_available() else {}

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = Xcodec2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Xcodec2Config,
            encoder_hidden_size=8,
            hidden_size=32,
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

            expected_arg_names = ["audio", "audio_spectrogram", "padding_mask"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.text_encoder.gradient_checkpointing = True
            config.audio_encoder.gradient_checkpointing = True
            config.decoder.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    @unittest.skip("XCodec2 does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("Xcodec2Model does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Xcodec2Model does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Xcodec2Model does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass


# Copied from transformers.tests.encodec.test_modeling_encodec.normalize
def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


# Copied from transformers.tests.encodec.test_modeling_encodec.compute_rmse
def compute_rmse(arr1, arr2):
    arr1_np = arr1.cpu().numpy().squeeze()
    arr2_np = arr2.cpu().numpy().squeeze()
    max_length = min(arr1.shape[-1], arr2.shape[-1])
    arr1_np = arr1_np[..., :max_length]
    arr2_np = arr2_np[..., :max_length]
    arr1_normalized = normalize(arr1_np)
    arr2_normalized = normalize(arr2_np)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


@slow
@require_torch
class Xcodec2IntegrationTest(unittest.TestCase):
    """NOTE (ebezzam): PyPI model does not support batch inference."""

    def test_integration(self):
        """
        reproducer: https://gist.github.com/ebezzam/3b79481b5d48d8e35c4ecc582aee0cb3#file-reproducer_single-py
        """
        results_path = Path(__file__).parent.parent.parent / "fixtures/xcodec2/expected_results_single.json"
        with open(results_path, "r") as f:
            raw_data = json.load(f)
        exp_code = torch.tensor(raw_data["audio_codes"])
        exp_recon = torch.tensor(raw_data["recon_wav"])
        exp_codec_error = float(raw_data["codec_error"])

        model_id = "bezzam/xcodec2"
        model = Xcodec2Model.from_pretrained(model_id).to(torch_device).eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        audio = dataset[0]["audio"]["array"]
        inputs = feature_extractor(
            audio=audio,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            audio_codes = model.encode(inputs["audio"], inputs["audio_spectrogram"], return_dict=False)[0]
            self.assertTrue(torch.equal(audio_codes.squeeze().cpu().to(exp_code.dtype), exp_code))

            dec = model.decode(audio_codes=audio_codes).audio_values
            torch.testing.assert_close(dec.squeeze().cpu(), exp_recon, rtol=1e-3, atol=1e-3)

            # compare codec error
            codec_error = compute_rmse(inputs["audio"], dec).item()
            torch.testing.assert_close(codec_error, exp_codec_error, rtol=1e-5, atol=1e-5)

            # make sure forward and decode gives same result
            enc_dec = model(inputs["audio"], inputs["audio_spectrogram"]).audio_values
            self.assertTrue(torch.equal(dec[..., : enc_dec.shape[-1]], enc_dec))
