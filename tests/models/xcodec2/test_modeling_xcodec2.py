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
"""Testing suite for the PyTorch Xcodec2 model."""

import inspect
import math
import os
import tempfile
import unittest

import numpy as np
from datasets import Audio, load_dataset
from pytest import mark

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from transformers import AutoFeatureExtractor, Xcodec2Config
from transformers.testing_utils import (
    is_flaky,
    is_torch_available,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
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
        batch_size=4,
        num_channels=1,
        sample_rate=16000,
        num_samples=400,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.is_training = is_training
        self.num_samples = num_samples
        self.num_channels = num_channels

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.num_samples], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
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
        return Xcodec2Config(
            sample_rate=self.sample_rate,
            audio_channels=self.num_channels,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = Xcodec2Model(config=config).to(torch_device).eval()
        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(
            result.audio_values.shape,
            (self.batch_size, self.num_channels, self.num_samples + 320 - (self.num_samples % 320)),
        )


@require_torch
class Xcodec2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Xcodec2Model,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False

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
            # self, config_class=Xcodec2Config, common_properties=[], has_text_modality=False
            self,
            config_class=Xcodec2Config,
            encoder_hidden_size=37,
            decoder_hidden_size=37,
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

            expected_arg_names = ["input_values", "audio_codes", "return_dict"]
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

    # Copied from transformers.tests.encodec.test_modeling_encodec.EncodecModelTest._create_and_check_torchscript
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

    @unittest.skip(reason="The Xcodec2Model does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The Xcodec2Model does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodec.EncodecModelTest.test_determinism
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

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                dummy_input = inputs_dict[model.main_input_name][:1]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                outputs = model(dummy_input)
                outputs_fa = model_fa(dummy_input)

                logits = outputs[1]
                logits_fa = outputs_fa[1]

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

    @unittest.skip(reason="The Xcodec2Model does not support right padding")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="The Xcodec2Model does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass


# Copied from transformers.tests.encodec.test_modeling_encodec.normalize
def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


# Copied from transformers.tests.encodec.test_modeling_encodec.compute_rmse
def compute_rmse(arr1, arr2):
    arr1_normalized = normalize(arr1)
    arr2_normalized = normalize(arr2)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


# @slow
@require_torch
class Xcodec2IntegrationTest(unittest.TestCase):
    def test_integration(self):
        expected_rmse = 0.07212554663419724
        expected_codes = [
            23060,
            39948,
            47126,
            60684,
            63493,
            23094,
            7188,
            48137,
            40247,
            32029,
            26934,
            64776,
            36662,
            48482,
            43026,
            44407,
            48533,
            59698,
            44402,
            44582,
            48758,
            60983,
            48721,
            44594,
            32421,
            61110,
            60695,
            59682,
            50697,
            61486,
            58159,
            20799,
            20511,
            4943,
            37853,
            44425,
            25045,
            27974,
            16597,
            38606,
            57455,
            62431,
            41275,
            38862,
            16586,
            24227,
            53819,
            32553,
            5735,
            41797,
            24338,
            49829,
            61057,
            53234,
            49366,
            53734,
            55891,
            53170,
            41938,
            9735,
            44557,
            40202,
            8133,
            42662,
            60915,
            48627,
            61294,
            37758,
            28207,
            48758,
            11221,
            9165,
            36689,
            19266,
            31489,
            50650,
            49335,
            27868,
            32481,
            8528,
            5977,
            11934,
            6299,
            56591,
            35035,
            55311,
            14363,
            64869,
            36154,
            20111,
            49807,
            452,
            6850,
            17940,
            1411,
            53789,
            19487,
            7567,
            3031,
            6996,
            43797,
            50230,
            54258,
            51481,
            41382,
            37716,
            7113,
            8132,
            44238,
            60443,
            18051,
            44934,
            55580,
            41639,
            54809,
            57375,
            8783,
            52557,
            8361,
            37477,
            24015,
            40394,
            52683,
            52931,
            8070,
            58906,
            52202,
            39603,
            55701,
            41099,
            53526,
            39831,
            49345,
            57991,
            54352,
            41425,
            41372,
            45380,
            12876,
            29589,
            62014,
            48686,
            22822,
            30341,
            6668,
            9512,
            12572,
            32052,
            64885,
            60773,
            48823,
            65078,
            44325,
            47991,
            60422,
            64869,
            28197,
            44662,
            47398,
            45028,
            44722,
            64827,
            23415,
            14793,
            64585,
            46247,
            14021,
            25045,
            61807,
            53165,
            29092,
            41856,
            59434,
            58079,
            63851,
            36958,
            5081,
            39232,
            25314,
            51550,
            45675,
            9160,
            56024,
            26769,
            33688,
            25554,
            54125,
            34395,
            25999,
            17348,
            977,
            45396,
            63034,
            48446,
            2993,
            34692,
            54249,
            52306,
            33474,
            51282,
            51922,
            49377,
            52883,
            50898,
            49286,
            51814,
            52894,
            52722,
            26776,
            41696,
            59582,
            45695,
            23143,
            12992,
            36120,
            47144,
            39944,
            64818,
            60708,
            44729,
            44851,
            64853,
            27954,
            23923,
            65277,
        ]

        librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "bezzam/xcodec2"
        model = Xcodec2Model.from_pretrained(model_id).to(torch_device).eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        librispeech = librispeech.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        audio = librispeech[-1]["audio"]["array"]

        inputs = feature_extractor(
            raw_audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            audio_codes = model.encode(inputs["input_values"], return_dict=False)
            codes = audio_codes.squeeze(0).squeeze(0).tolist()

            self.assertEqual(codes, expected_codes)

            input_values_dec = model.decode(audio_codes, return_dict=False)
            input_values_enc_dec = model(inputs["input_values"])[1]

        self.assertTrue(torch.allclose(input_values_dec, input_values_enc_dec, atol=1e-6))

        arr = inputs["input_values"][0].cpu().numpy()
        arr_enc_dec = input_values_enc_dec[0].cpu().numpy()
        arr_enc_dec_truncated = arr_enc_dec[:, : arr.shape[1]]
        rmse = np.sqrt(((arr - arr_enc_dec_truncated) ** 2).mean())
        self.assertTrue(np.abs(rmse - expected_rmse) < 1e-6)
