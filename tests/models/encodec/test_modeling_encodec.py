# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Encodec model. """

import inspect
import unittest
import numpy as np
from torch import exp

from datasets import Audio, load_dataset

from transformers import AutoProcessor, EncodecConfig
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)
from datasets import load_dataset, Audio

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import EncodecModel


def prepare_inputs_dict(
    config,
    input_ids=None,
    input_values=None,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if input_ids is not None:
        encoder_dict = {"input_ids": input_ids}
    else:
        encoder_dict = {"input_values": input_values}

    decoder_dict = {"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {}

    return {
        **encoder_dict,
        **decoder_dict,
    }


@require_torch
class EncodecModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        channels=2,  # 2 channels actually
        is_training=False,
        num_hidden_layers=4,
        intermediate_size=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.channels = channels
        self.is_training = is_training

        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return EncodecConfig()

    def create_and_check_model_forward(self, config, inputs_dict):
        model = EncodecModel(config=config).to(torch_device).eval()

        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))


@require_torch
class EncodecModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EncodecModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"automatic-speech-recognition": EncodecModel, "feature-extraction": EncodecModel}
        if is_torch_available()
        else {}
    )
    # use EnCodecForSpeechToText later on
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False

    input_name = "input_values"

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not have attention and does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = EncodecModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EncodecConfig, hidden_size=37, common_properties=[])

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

            expected_arg_names = ["input_values", "bandwidth"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    # this model has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # this model has no input embeddings
    def test_model_common_attributes(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
        pass

    @slow
    def test_torchscript_output_attentions(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    @slow
    def test_torchscript_output_hidden_state(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    @slow
    def test_torchscript_simple(self):
        # disabled because this model doesn't have decoder_input_ids
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv"]
                # TODO find the correct init values for lstm (or let them be pytorch)
                ignore_init = ["lstm"]
                if param.requires_grad:
                    if any([x in name for x in uniform_init_parms]):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif not any([x in name for x in ignore_init]):
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )



def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


def compute_rmse(arr1, arr2):
    arr1_normalized = normalize(arr1)
    arr2_normalized = normalize(arr2)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


@slow
@require_torch
class EncodecIntegrationTest(unittest.TestCase):

    def test_integration_24kHz(self):
        expected_rmse = {
            "1.5": 0.0025,
            "24.0": 0.0015,
        }
        expected_codesums = {
            "1.5": [371955],
            "24.0": [6644616],
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "Matthijs/encodec_24khz"

        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        input_values = processor(audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt").input_values.to(torch_device)

        for bandwidth, expected_rmse in expected_rmse.items():
            with torch.no_grad():
                # use max bandwith for best possible reconstruction
                audio_code = model.encode(input_values, bandwidth=float(bandwidth))

                audio_code_sums = [a[0][0].sum().cpu().item() for a in audio_code]

                # make sure audio encoded codes are correct
                self.assertListEqual(audio_code_sums, expected_codesums[bandwidth])

                input_values_dec = model.decode(audio_code)
                input_values_enc_dec = model(input_values, bandwidth=float(bandwidth))

            # make sure forward and decode gives same result
            self.assertTrue(torch.allclose(input_values_dec, input_values_enc_dec, atol=1e-3))

            # make sure shape matches
            self.assertTrue(input_values.shape == input_values_enc_dec.shape)

            arr = input_values[0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(arr, arr_enc_dec)
            self.assertTrue(rmse < expected_rmse)

    def test_integration_48kHz(self):
        expected_rmse = {
            "3.0": 0.001,
            "24.0": 0.0005,
        }
        expected_codesums = {
            "3.0": [144259, 146765, 156205, 176871, 102780],
            "24.0": [1567904, 1297170, 1310040, 1464657, 813925],
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "Matthijs/encodec_48khz"

        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        # transform mono to stereo
        audio_sample = np.array([audio_sample, audio_sample])

        input_values = processor(audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt").input_values.to(torch_device)

        for bandwidth, expected_rmse in expected_rmse.items():
            with torch.no_grad():
                # use max bandwith for best possible reconstruction
                audio_code = model.encode(input_values, bandwidth=float(bandwidth))

                audio_code_sums = [a[0][0].sum().cpu().item() for a in audio_code]

                # make sure audio encoded codes are correct
                self.assertListEqual(audio_code_sums, expected_codesums[bandwidth])

                input_values_dec = model.decode(audio_code)
                input_values_enc_dec = model(input_values, bandwidth=float(bandwidth))

            # make sure forward and decode gives same result
            self.assertTrue(torch.allclose(input_values_dec, input_values_enc_dec, atol=1e-3))

            # make sure shape matches
            self.assertTrue(input_values.shape == input_values_enc_dec.shape)

            arr = input_values[0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(arr, arr_enc_dec)
            self.assertTrue(rmse < expected_rmse)
