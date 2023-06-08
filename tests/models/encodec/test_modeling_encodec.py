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

from transformers import EncodecConfig, AutoProcessor
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)
from datasets import load_dataset, Audio

from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    random_attention_mask,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    random_attention_mask,
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

    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)

    return {
        **encoder_dict,
        **decoder_dict,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


@require_torch
class EncodecModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=81,
        hidden_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length, self.hidden_size], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        
        decoder_attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        inputs_dict = prepare_inputs_dict(
            config,
            input_values=input_values,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return EncodecConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = EncodecModel(config=config).to(torch_device).eval()

        input_values = inputs_dict["input_values"]
        attention_mask = inputs_dict["attention_mask"]
        decoder_input_values = inputs_dict["decoder_input_values"]

        result = model(input_values, attention_mask=attention_mask, decoder_input_values=decoder_input_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def _prepare_for_class(self, inputs_dict, model_class, return_labels = False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        inputs_dict.pop("decoder_input_values")
        return inputs_dict

@require_torch
class EncodecModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EncodecModel,) if is_torch_available() else ()
    # pipeline_model_mapping = (
    #     {"automatic-speech-recognition": EncodecForSpeechToText, "feature-extraction": EncodecModel}
    #     if is_torch_available()
    #     else {}
    # )
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

    def setUp(self):
        self.model_tester = EncodecModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EncodecConfig, hidden_size=37, common_properties = [])

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

            expected_arg_names = [
                "input_values",
                "attention_mask",
                "decoder_attention_mask",
            ]
            expected_arg_names.extend(
                ["head_mask", "decoder_head_mask", "cross_attn_head_mask", "encoder_outputs"]
                if "head_mask" and "decoder_head_mask" and "cross_attn_head_mask" in arg_names
                else ["encoder_outputs"]
            )
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
                uniform_init_parms = [
                    "conv"
                ]
                # TODO find the correct init values for lstm (or let them be pytorch)
                ignore_init = [
                    "lstm"
                ]
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
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "Matthijs/encodec_24khz"

        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]

        input_values = processor(audio=audio_sample["audio"]["array"], return_tensors="pt").input_values.to(torch_device)

        for bandwith, expected_rmse in expected_rmse.items():
            with torch.no_grad():
                # use max bandwith for best possible reconstruction
                input_values_enc_dec = model(input_values, bandwidth=float(bandwith))

            self.assertTrue(input_values.shape == input_values_enc_dec.shape)

            arr = input_values[0, 0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0, 0].cpu().numpy()

            rmse = compute_rmse(arr, arr_enc_dec)

            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            self.assertTrue(rmse < expected_rmse)

    def test_integration_48kHz(self):
        expected_rmse = {
            "3.0": 0.001,
            "24.0": 0.0005,
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "Matthijs/encodec_48khz"

        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        # transform mono to stereo
        audio_sample = np.array([audio_sample, audio_sample])

        input_values = processor(audio=audio_sample, return_tensors="pt").input_values.to(torch_device)

        for bandwith, expected_rmse in expected_rmse.items():
            with torch.no_grad():
                # use max bandwith for best possible reconstruction
                input_values_enc_dec = model(input_values, bandwidth=float(bandwith))

            self.assertTrue(input_values.shape == input_values_enc_dec.shape)

            arr = input_values[0].cpu().numpy()
            arr_enc_dec = input_values_enc_dec[0].cpu().numpy()

            rmse = compute_rmse(arr, arr_enc_dec)

            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            self.assertTrue(rmse < expected_rmse)
