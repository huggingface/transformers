# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Xcodec model."""

import inspect
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset
from parameterized import parameterized

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from transformers import AutoFeatureExtractor, XcodecConfig
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import XcodecModel


@require_torch
class XcodecModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        num_channels=1,
        sample_rate=16000,
        codebook_size=1024,
        num_samples=400,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.codebook_size = codebook_size
        self.is_training = is_training
        self.num_samples = num_samples

    def prepare_config_and_inputs(self):
        config = self.get_config()
        inputs_dict = {
            "input_values": floats_tensor([self.batch_size, self.num_channels, self.num_samples], scale=1.0)
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config, inputs_dict = self.prepare_config_and_inputs()
        codes_length = math.ceil(self.num_samples / config.hop_length)
        inputs_dict["audio_codes"] = ids_tensor(
            [self.batch_size, config.num_quantizers, codes_length], config.codebook_size
        )
        return config, inputs_dict

    def get_config(self):
        return XcodecConfig(
            sample_rate=self.sample_rate,
            audio_channels=self.num_channels,
            codebook_size=self.codebook_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = XcodecModel(config=config).to(torch_device).eval()
        result = model(input_values=inputs_dict["input_values"])
        self.parent.assertEqual(result.audio_values.shape, (self.batch_size, self.num_channels, self.num_samples))


@require_torch
class XcodecModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (XcodecModel,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False
    test_can_init_all_missing_weights = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = XcodecModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=XcodecConfig, common_properties=[], has_text_modality=False
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

            expected_arg_names = ["input_values", "audio_codes", "bandwidth", "return_dict"]
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

    @unittest.skip("XcodecModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have the usual `attention` logic")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodec.XcodecModelTest._create_and_check_torchscript
    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            main_input_name = model_class.main_input_name

            try:
                main_input = inputs[main_input_name]
                model(main_input)
                traced_model = torch.jit.trace(model, main_input)
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                if layer_name in loaded_model_state_dict:
                    p2 = loaded_model_state_dict[layer_name]
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()

    @unittest.skip(reason="The XcodecModel does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The XcodecModel does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodecEncodecModelTest.test_determinism
    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # outputs are not tensors but list (since each sequence don't have the same frame_length)
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    # Copied from transformers.tests.encodec.test_modeling_encodecEncodecModelTest.test_model_outputs_equivalence
    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs)

                self.assertTrue(isinstance(tuple_output, tuple))
                self.assertTrue(isinstance(dict_output, dict))

                for tuple_value, dict_value in zip(tuple_output, dict_output.values()):
                    self.assertTrue(
                        torch.allclose(
                            set_nan_tensor_to_zero(tuple_value), set_nan_tensor_to_zero(dict_value), atol=1e-5
                        ),
                        msg=(
                            "Tuple and dict output are not equal. Difference:"
                            f" {torch.max(torch.abs(tuple_value - dict_value))}. Tuple has `nan`:"
                            f" {torch.isnan(tuple_value).any()} and `inf`: {torch.isinf(tuple_value)}. Dict has"
                            f" `nan`: {torch.isnan(dict_value).any()} and `inf`: {torch.isinf(dict_value)}."
                        ),
                    )

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # skipping the parametrizations original0 tensor
                if name == "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0":
                    continue

                uniform_init_parms = ["conv"]

                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of {model_class.__name__} seems not properly initialized",
                        )

    @unittest.skip(reason="The XcodecModel does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
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


"""
Integration tests for XCodec

Code for reproducing expected outputs can be found here:
https://gist.github.com/ebezzam/cdaf8c223e59e7677b2ea6bc2dc8230b

One reason for higher tolerances is because of different implementation of `Snake1d` within Transformer version DAC
See here: https://github.com/huggingface/transformers/pull/39793#issue-3277407384

"""

RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/xcodec/integration_tests.json"

with open(RESULTS_PATH, "r") as f:
    raw_data = json.load(f)

# convert dicts into tuples ordered to match test args
EXPECTED_OUTPUTS_JSON = [
    (
        f"{d['repo_id']}_{d['bandwidth']}",
        d["repo_id"],
        d["bandwidth"],
        d["codes"],
        d["decoded"],
        d["codec_error"],
        d["codec_tol"],
        d["dec_tol"],
    )
    for d in raw_data
]


@slow
@require_torch
class XcodecIntegrationTest(unittest.TestCase):
    @parameterized.expand(EXPECTED_OUTPUTS_JSON)
    def test_integration(
        self, test_name, repo_id, bandwidth, exp_codes, exp_decoded, exp_codec_err, codec_tol, dec_tol
    ):
        # load model
        model = XcodecModel.from_pretrained(repo_id).to(torch_device).eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

        # load audio example
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column(
            "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
        )
        audio_array = librispeech_dummy[0]["audio"]["array"]
        inputs = feature_extractor(
            raw_audio=audio_array, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
        ).to(torch_device)
        x = inputs["input_values"]

        with torch.no_grad():
            ENC_TOL = 0
            audio_codes = model.encode(x, bandwidth=bandwidth, return_dict=False)
            if exp_codes is not None:
                exp_codes = torch.tensor(exp_codes).to(torch_device)
                torch.testing.assert_close(
                    audio_codes[..., : exp_codes.shape[-1]],
                    exp_codes,
                    rtol=ENC_TOL,
                    atol=ENC_TOL,
                )

            # dec_tol = 1e-5    # increased to 1e-4 for passing on 4 kbps
            input_values_dec = model.decode(audio_codes).audio_values
            if exp_decoded is not None:
                exp_decoded = torch.tensor(exp_decoded).to(torch_device)
                torch.testing.assert_close(
                    input_values_dec[..., : exp_decoded.shape[-1]],
                    exp_decoded,
                    rtol=dec_tol,
                    atol=dec_tol,
                )

            # compute codec error
            codec_err = compute_rmse(input_values_dec, x)
            torch.testing.assert_close(codec_err, exp_codec_err, rtol=codec_tol, atol=codec_tol)

            # make sure forward and decode gives same result
            audio_values_enc_dec = model(x, bandwidth=bandwidth).audio_values
            torch.testing.assert_close(input_values_dec, audio_values_enc_dec, rtol=1e-6, atol=1e-6)
