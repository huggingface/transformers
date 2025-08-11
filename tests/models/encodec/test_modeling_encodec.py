# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Encodec model."""

import copy
import inspect
import os
import tempfile
import unittest

import numpy as np
from datasets import Audio, load_dataset
from parameterized import parameterized

from transformers import AutoProcessor, EncodecConfig
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import EncodecFeatureExtractor, EncodecModel


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

    return {**encoder_dict, **decoder_dict}


@require_torch
class EncodecModelTester:
    def __init__(
        self,
        parent,
        # `batch_size` needs to be an even number if the model has some outputs with batch dim != 0.
        batch_size=12,
        num_channels=2,
        is_training=False,
        intermediate_size=40,
        hidden_size=32,
        num_filters=8,
        num_residual_layers=1,
        upsampling_ratios=[8, 4],
        num_lstm_layers=1,
        codebook_size=64,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.upsampling_ratios = upsampling_ratios
        self.num_lstm_layers = num_lstm_layers
        self.codebook_size = codebook_size

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config, inputs_dict = self.prepare_config_and_inputs()
        inputs_dict["audio_codes"] = ids_tensor([1, self.batch_size, 1, self.num_channels], self.codebook_size).type(
            torch.int32
        )
        inputs_dict["audio_scales"] = [None]

        return config, inputs_dict

    def prepare_config_and_inputs_for_normalization(self):
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        config.normalize = True

        processor = EncodecFeatureExtractor(feature_size=config.audio_channels, sampling_rate=config.sampling_rate)
        input_values = input_values.tolist()
        inputs_dict = processor(
            input_values, sampling_rate=config.sampling_rate, padding=True, return_tensors="pt"
        ).to(torch_device)

        return config, inputs_dict

    def get_config(self):
        return EncodecConfig(
            audio_channels=self.num_channels,
            chunk_in_sec=None,
            hidden_size=self.hidden_size,
            num_filters=self.num_filters,
            num_residual_layers=self.num_residual_layers,
            upsampling_ratios=self.upsampling_ratios,
            num_lstm_layers=self.num_lstm_layers,
            codebook_size=self.codebook_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = EncodecModel(config=config).to(torch_device).eval()
        result = model(**inputs_dict)
        self.parent.assertEqual(
            result.audio_values.shape, (self.batch_size, self.num_channels, self.intermediate_size)
        )


@require_torch
class EncodecModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EncodecModel,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    pipeline_model_mapping = {"feature-extraction": EncodecModel} if is_torch_available() else {}

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
        self.config_tester = ConfigTester(
            self, config_class=EncodecConfig, hidden_size=37, common_properties=[], has_text_modality=False
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

            expected_arg_names = ["input_values", "padding_mask", "bandwidth"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip(reason="The EncodecModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The EncodecModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `attention` logic"
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `attention` logic"
    )
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_torchscript_output_hidden_state(self):
        pass

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
            for key in loaded_model_state_dict:
                if key not in model_state_dict:
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

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `attention` logic"
    )
    def test_attention_outputs(self):
        pass

    def test_feed_forward_chunking(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        # original_config.norm_type = "time_group_norm"
        for model_class in self.all_model_classes:
            torch.manual_seed(0)
            config = copy.deepcopy(original_config)
            config.chunk_length_s = None
            config.overlap = None
            config.sampling_rate = 20

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs["input_values"] = inputs["input_values"].repeat(1, 1, 10)

            hidden_states_no_chunk = model(**inputs)[1]

            torch.manual_seed(0)
            config.chunk_length_s = 2
            config.overlap = 0
            config.sampling_rate = 20

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_with_chunk = model(**inputs)[1]
            torch.testing.assert_close(hidden_states_no_chunk, hidden_states_with_chunk, rtol=1e-1, atol=1e-2)

    @unittest.skip(
        reason="The EncodecModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_hidden_states_output(self):
        pass

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

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def assert_nested_tensors_close(a, b):
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
                for i, (x, y) in enumerate(zip(a, b)):
                    assert_nested_tensors_close(x, y)
            elif torch.is_tensor(a) and torch.is_tensor(b):
                a_clean = set_nan_tensor_to_zero(a)
                b_clean = set_nan_tensor_to_zero(b)
                assert torch.allclose(a_clean, b_clean, atol=1e-5), (
                    "Tuple and dict output are not equal. Difference:"
                    f" Max diff: {torch.max(torch.abs(a_clean - b_clean))}. "
                    f"Tuple has nan: {torch.isnan(a).any()} and inf: {torch.isinf(a)}. "
                    f"Dict has nan: {torch.isnan(b).any()} and inf: {torch.isinf(b)}."
                )
            else:
                raise ValueError(f"Mismatch between {a} vs {b}")

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs)

            self.assertTrue(isinstance(tuple_output, tuple))
            self.assertTrue(isinstance(dict_output, dict))
            # cast dict_output.values() to list as it is a odict_values object
            assert_nested_tensors_close(tuple_output, list(dict_output.values()))

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
                uniform_init_parms = ["conv"]
                ignore_init = ["lstm"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif not any(x in name for x in ignore_init):
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_identity_shortcut(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.use_conv_shortcut = False
        self.model_tester.create_and_check_model_forward(config, inputs_dict)

    def test_model_forward_with_normalization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_normalization()
        self.model_tester.create_and_check_model_forward(config, inputs_dict)


def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


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
Integration tests for the Encodec model.

Code for expected output can be found below:
- test_integration: https://gist.github.com/ebezzam/2a34e249e729881130d1f5a42229d31f#file-test_encodec-py
- test_batch: https://gist.github.com/ebezzam/2a34e249e729881130d1f5a42229d31f#file-test_encodec_batch-py

"""

# fmt: off
# first key is model_id from hub, second key is bandwidth
# -- test_integration
EXPECTED_ENCODER_CODES = {
    "facebook/encodec_24khz": {
        "1.5": torch.tensor([[[  62,  835,  835,  835,  835,  835,  835,  835,  408,  408],
         [1007, 1007, 1007,  544,  424,  424, 1007,  424,  302,  424]]]),
        "3.0": torch.tensor(
            [
                [
                    [62, 835, 835, 835, 835, 835, 835, 835, 408, 408],
                    [1007, 1007, 1007, 544, 424, 424, 1007, 424, 302, 424],
                    [786, 678, 821, 786, 36, 36, 786, 212, 937, 937],
                    [741, 741, 741, 993, 741, 1018, 993, 919, 741, 741],
                ],
            ]
        ),
        "6.0": torch.tensor(
            [
                [
                    [62, 835, 835, 835, 835, 835, 835, 835, 408, 408],
                    [1007, 1007, 1007, 544, 424, 424, 1007, 424, 302, 424],
                    [786, 678, 821, 786, 36, 36, 786, 212, 937, 937],
                    [741, 741, 741, 993, 741, 1018, 993, 919, 741, 741],
                    [528, 446, 198, 190, 446, 622, 646, 448, 646, 448],
                    [1011, 140, 185, 986, 683, 986, 435, 41, 140, 939],
                    [896, 772, 562, 772, 485, 528, 896, 853, 562, 772],
                    [899, 975, 468, 468, 468, 701, 1013, 828, 518, 899],

                ],
            ]
        ),
        "12.0": torch.tensor(
            [
                [
                    [62, 835, 835, 835, 835, 835, 835, 835, 408, 408],
                    [1007, 1007, 1007, 544, 424, 424, 1007, 424, 302, 424],
                    [786, 678, 821, 786, 36, 36, 786, 212, 937, 937],
                    [741, 741, 741, 993, 741, 1018, 993, 919, 741, 741],
                    [528, 446, 198, 190, 446, 622, 646, 448, 646, 448],
                    [1011, 140, 185, 986, 683, 986, 435, 41, 140, 939],
                    [896, 772, 562, 772, 485, 528, 896, 853, 562, 772],
                    [899, 975, 468, 468, 468, 701, 1013, 828, 518, 899],
                    [827, 807, 938, 320, 699, 470, 909, 628, 301, 827],
                    [963, 801, 630, 477, 717, 354, 205, 359, 874, 744],
                    [1000, 1000, 388, 1000, 408, 740, 568, 364, 709, 843],
                    [413, 835, 382, 840, 742, 1019, 375, 962, 835, 742],
                    [971, 410, 998, 485, 798, 410, 351, 485, 485, 920],
                    [848, 694, 662, 784, 848, 427, 1022, 848, 920, 694],
                    [420, 911, 889, 911, 993, 776, 948, 477, 911, 911],
                    [587, 755, 834, 962, 860, 425, 982, 982, 425, 461],
                ],
            ]
        ),
        "24.0": torch.tensor(
            [
                [
                    [62, 835, 835, 835, 835, 835, 835, 835, 408, 408],
                    [1007, 1007, 1007, 544, 424, 424, 1007, 424, 302, 424],
                    [786, 678, 821, 786, 36, 36, 786, 212, 937, 937],
                    [741, 741, 741, 993, 741, 1018, 993, 919, 741, 741],
                    [528, 446, 198, 190, 446, 622, 646, 448, 646, 448],
                    [1011, 140, 185, 986, 683, 986, 435, 41, 140, 939],
                    [896, 772, 562, 772, 485, 528, 896, 853, 562, 772],
                    [899, 975, 468, 468, 468, 701, 1013, 828, 518, 899],
                    [827, 807, 938, 320, 699, 470, 909, 628, 301, 827],
                    [963, 801, 630, 477, 717, 354, 205, 359, 874, 744],
                    [1000, 1000, 388, 1000, 408, 740, 568, 364, 709, 843],
                    [413, 835, 382, 840, 742, 1019, 375, 962, 835, 742],
                    [971, 410, 998, 485, 798, 410, 351, 485, 485, 920],
                    [848, 694, 662, 784, 848, 427, 1022, 848, 920, 694],
                    [420, 911, 889, 911, 993, 776, 948, 477, 911, 911],
                    [587, 755, 834, 962, 860, 425, 982, 982, 425, 461],
                    [270, 160, 26, 131, 597, 506, 670, 637, 248, 160],
                    [ 15, 215, 134, 69, 215, 155, 1012, 1009, 260, 417],
                    [580, 561, 686, 896, 497, 637, 580, 245, 896, 264],
                    [511, 239, 560, 691, 571, 627, 571, 571, 258, 619],
                    [591, 942, 591, 251, 250, 250, 857, 486, 295, 295],
                    [565, 546, 654, 301, 301, 623, 639, 568, 565, 282],
                    [539, 317, 639, 539, 651, 539, 538, 640, 615, 615],
                    [637, 556, 637, 582, 640, 515, 515, 632, 254, 613],
                    [305, 643, 500, 550, 522, 500, 550, 561, 522, 305],
                    [954, 456, 584, 755, 505, 782, 661, 671, 497, 505],
                    [577, 464, 637, 647, 552, 552, 624, 647, 624, 647],
                    [728, 748, 931, 608, 538, 1015, 294, 294, 666, 538],
                    [602, 535, 666, 665, 655, 979, 574, 535, 571, 781],
                    [321, 620, 557, 566, 511, 910, 672, 623, 853, 674],
                    [621, 556, 947, 474, 610, 752, 1002, 597, 474, 474],
                    [605, 948, 657, 588, 485, 633, 459, 968, 939, 325],
                ],
            ]
        ),
        },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([[[214, 214, 214, 214, 214, 118, 214, 214, 214, 214],
            [989, 989, 611,  77,  77, 989, 976, 976, 976,  77]]]),
        "6.0": torch.tensor([[[ 214,  214,  214,  214,  214,  118,  214,  214,  214,  214],
            [ 989,  989,  611,   77,   77,  989,  976,  976,  976,   77],
            [ 977, 1009,  538,  925,  925,  977, 1022, 1022, 1022,  925],
            [ 376, 1012, 1023,  725,  725, 1023,  376,  962,  376,  847]]]),
        "12.0": torch.tensor([[[ 214,  214,  214,  214,  214,  118,  214,  214,  214,  214],
            [ 989,  989,  611,   77,   77,  989,  976,  976,  976,   77],
            [ 977, 1009,  538,  925,  925,  977, 1022, 1022, 1022,  925],
            [ 376, 1012, 1023,  725,  725, 1023,  376,  962,  376,  847],
            [ 979, 1012,  323,  695, 1018, 1023,  979, 1023,  979,  650],
            [ 945,  762,  528,  865,  824,  945,  945,  945,  957,  957],
            [ 904,  973, 1014,  681,  582, 1014, 1014, 1014, 1014,  681],
            [ 229,  392,  796,  392,  977, 1017,  250, 1017,  250, 1017]]]),
        "24.0": torch.tensor([[[ 214,  214,  214,  214,  214,  118,  214,  214,  214,  214],
            [ 989,  989,  611,   77,   77,  989,  976,  976,  976,   77],
            [ 977, 1009,  538,  925,  925,  977, 1022, 1022, 1022,  925],
            [ 376, 1012, 1023,  725,  725, 1023,  376,  962,  376,  847],
            [ 979, 1012,  323,  695, 1018, 1023,  979, 1023,  979,  650],
            [ 945,  762,  528,  865,  824,  945,  945,  945,  957,  957],
            [ 904,  973, 1014,  681,  582, 1014, 1014, 1014, 1014,  681],
            [ 229,  392,  796,  392,  977, 1017,  250, 1017,  250, 1017],
            [ 902,  436,  935, 1011, 1023, 1023, 1023,  154, 1023,  392],
            [ 982,  878,  961,  832,  629,  431,  919,  629,  919,  792],
            [ 727,  727,  401,  727,  979,  587,  727,  487,  413,  201],
            [ 928,  924,  965,  934,  840,  480,  924,  920,  924,  486],
            [  10,  625,  712,  552,  712,  259,  394,  131,  726,  516],
            [ 882, 1022,   32,  524,  267,  861,  974,  882,  108,  521],
            [ 304,  841,  306,  415,   69,  376,  928,  510,  381,  104],
            [   0,    0,    0,  484,   83,    0,  307,  262,    0,    0]]])
    }
}
EXPECTED_ENCODER_SCALES = {
    "facebook/encodec_24khz": {
        "1.5": None,
        "3.0": None,
        "6.0": None,
        "12.0": None,
        "24.0": None
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([5.365404e-02, 8.153407e-02, 6.266369e-02, 6.688326e-02, 5.458422e-02,
        4.483359e-02, 1.000000e-08]),
        "6.0": torch.tensor([5.365404e-02, 8.153407e-02, 6.266369e-02, 6.688326e-02, 5.458422e-02,
        4.483359e-02, 1.000000e-08]),
        "12.0": torch.tensor([5.365404e-02, 8.153407e-02, 6.266369e-02, 6.688326e-02, 5.458422e-02,
        4.483359e-02, 1.000000e-08]),
        "24.0": torch.tensor([5.365404e-02, 8.153407e-02, 6.266369e-02, 6.688326e-02, 5.458422e-02,
        4.483359e-02, 1.000000e-08])
    }
}
EXPECTED_DECODER_OUTPUTS = {
    "facebook/encodec_24khz": {
        "1.5": torch.tensor(
            [[ 0.0003, -0.0002, -0.0000, -0.0004, 0.0004, 0.0003, -0.0000, 0.0001, 0.0005, 0.0001, -0.0015, -0.0007, -0.0002, -0.0018, -0.0003, 0.0013, 0.0011, 0.0008, 0.0008, 0.0008, 0.0008, 0.0002, -0.0003, -0.0004, -0.0006, -0.0009, -0.0010, -0.0012, -0.0011, -0.0006, -0.0006, -0.0005, 0.0000, 0.0001, 0.0003, 0.0002, -0.0001, -0.0002, -0.0008, -0.0012, -0.0011, -0.0012, -0.0013, -0.0003, 0.0002, 0.0006, 0.0006, 0.0006, 0.0009, 0.0010]]
        ),
        "3.0": torch.tensor(
            [[ 0.0003, -0.0002, -0.0000, -0.0004, 0.0004, 0.0003, -0.0000, 0.0001, 0.0006, 0.0002, -0.0015, -0.0008, -0.0002, -0.0018, -0.0003, 0.0013, 0.0011, 0.0008, 0.0008, 0.0008, 0.0008, 0.0002, -0.0003, -0.0004, -0.0005, -0.0008, -0.0010, -0.0012, -0.0011, -0.0006, -0.0006, -0.0005, -0.0000, 0.0001, 0.0003, 0.0002, -0.0001, -0.0002, -0.0008, -0.0013, -0.0011, -0.0013, -0.0014, -0.0003, 0.0002, 0.0006, 0.0006, 0.0006, 0.0009, 0.0010]]
        ),
        "6.0": torch.tensor(
            [[ 0.0004, -0.0001, 0.0001, -0.0003, 0.0004, 0.0003, 0.0000, 0.0001, 0.0007, 0.0002, -0.0013, -0.0007, -0.0002, -0.0015, -0.0001, 0.0014, 0.0014, 0.0011, 0.0010, 0.0010, 0.0009, 0.0004, 0.0000, 0.0000, 0.0000, -0.0000, -0.0001, -0.0004, -0.0004, -0.0001, -0.0002, -0.0002, 0.0002, 0.0005, 0.0009, 0.0010, 0.0008, 0.0007, 0.0002, -0.0003, -0.0004, -0.0008, -0.0008, 0.0000, 0.0006, 0.0010, 0.0012, 0.0012, 0.0013, 0.0014]]
        ),
        "12.0": torch.tensor(
            [[ 0.0004, -0.0001, 0.0001, -0.0004, 0.0003, 0.0002, -0.0000, 0.0001, 0.0006, 0.0002, -0.0013, -0.0006, -0.0001, -0.0014, 0.0001, 0.0018, 0.0018, 0.0014, 0.0012, 0.0013, 0.0011, 0.0006, 0.0000, 0.0000, -0.0000, -0.0001, -0.0001, -0.0004, -0.0004, -0.0000, -0.0000, -0.0000, 0.0005, 0.0007, 0.0011, 0.0011, 0.0009, 0.0007, 0.0002, -0.0003, -0.0004, -0.0007, -0.0007, 0.0002, 0.0009, 0.0013, 0.0015, 0.0014, 0.0015, 0.0016]]
        ),
        "24.0": torch.tensor(
            [[ 0.0005, 0.0001, 0.0004, -0.0001, 0.0003, 0.0002, 0.0000, 0.0001, 0.0007, 0.0005, -0.0011, -0.0005, -0.0001, -0.0018, -0.0000, 0.0021, 0.0019, 0.0013, 0.0011, 0.0012, 0.0012, 0.0006, -0.0000, -0.0001, -0.0000, -0.0000, -0.0001, -0.0004, -0.0004, -0.0000, -0.0001, -0.0002, 0.0003, 0.0004, 0.0008, 0.0007, 0.0006, 0.0007, 0.0001, -0.0004, -0.0003, -0.0006, -0.0008, 0.0004, 0.0011, 0.0015, 0.0016, 0.0015, 0.0016, 0.0018]]
        )
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor(
            [
                [0.0034, 0.0028, 0.0037, 0.0041, 0.0029, 0.0022, 0.0021, 0.0020, 0.0021, 0.0023, 0.0021, 0.0018, 0.0019, 0.0020, 0.0020, 0.0020, 0.0021, 0.0023, 0.0025, 0.0022, 0.0017, 0.0015, 0.0017, 0.0020, 0.0024, 0.0031, 0.0039, 0.0045, 0.0046, 0.0042, 0.0034, 0.0027, 0.0023, 0.0022, 0.0023, 0.0024, 0.0022, 0.0023, 0.0024, 0.0027, 0.0027, 0.0027, 0.0025, 0.0024, 0.0024, 0.0026, 0.0028, 0.0027, 0.0024, 0.0022],
                [ -0.0031, -0.0027, -0.0018, -0.0017, -0.0024, -0.0029, -0.0030, -0.0026, -0.0021, -0.0018, -0.0018, -0.0019, -0.0017, -0.0014, -0.0012, -0.0010, -0.0008, -0.0004, -0.0001, -0.0004, -0.0012, -0.0015, -0.0014, -0.0013, -0.0011, -0.0005, 0.0002, 0.0007, 0.0008, 0.0004, -0.0003, -0.0010, -0.0012, -0.0011, -0.0009, -0.0009, -0.0009, -0.0008, -0.0006, -0.0005, -0.0005, -0.0005, -0.0006, -0.0008, -0.0008, -0.0006, -0.0005, -0.0007, -0.0010, -0.0012],
            ]
        ),
        "6.0": torch.tensor(
            [
                [0.0052, 0.0049, 0.0057, 0.0058, 0.0048, 0.0043, 0.0042, 0.0041, 0.0041, 0.0042, 0.0040, 0.0038, 0.0038, 0.0038, 0.0037, 0.0037, 0.0037, 0.0037, 0.0038, 0.0037, 0.0035, 0.0034, 0.0036, 0.0039, 0.0043, 0.0047, 0.0053, 0.0057, 0.0057, 0.0055, 0.0050, 0.0046, 0.0043, 0.0041, 0.0042, 0.0042, 0.0041, 0.0041, 0.0042, 0.0043, 0.0043, 0.0043, 0.0041, 0.0040, 0.0040, 0.0041, 0.0042, 0.0042, 0.0040, 0.0039],
                [ 0.0001, 0.0006, 0.0013, 0.0011, 0.0005, 0.0001, -0.0001, 0.0001, 0.0003, 0.0005, 0.0005, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010, 0.0013, 0.0015, 0.0014, 0.0010, 0.0008, 0.0010, 0.0012, 0.0015, 0.0019, 0.0023, 0.0026, 0.0026, 0.0024, 0.0020, 0.0016, 0.0013, 0.0013, 0.0014, 0.0015, 0.0015, 0.0016, 0.0017, 0.0017, 0.0017, 0.0016, 0.0015, 0.0013, 0.0013, 0.0013, 0.0013, 0.0012, 0.0010, 0.0009],
            ]
        ),
        "12.0": torch.tensor(
            [
                [0.0014, 0.0012, 0.0021, 0.0024, 0.0017, 0.0013, 0.0012, 0.0011, 0.0011, 0.0012, 0.0011, 0.0010, 0.0009, 0.0009, 0.0008, 0.0008, 0.0009, 0.0010, 0.0012, 0.0012, 0.0009, 0.0008, 0.0010, 0.0013, 0.0017, 0.0024, 0.0031, 0.0036, 0.0036, 0.0033, 0.0028, 0.0023, 0.0020, 0.0020, 0.0022, 0.0022, 0.0022, 0.0022, 0.0023, 0.0024, 0.0024, 0.0023, 0.0021, 0.0021, 0.0021, 0.0023, 0.0024, 0.0024, 0.0022, 0.0021],
                [ -0.0034, -0.0029, -0.0020, -0.0020, -0.0024, -0.0027, -0.0030, -0.0030, -0.0028, -0.0025, -0.0025, -0.0025, -0.0025, -0.0025, -0.0023, -0.0022, -0.0020, -0.0017, -0.0013, -0.0014, -0.0017, -0.0019, -0.0018, -0.0015, -0.0011, -0.0006, 0.0000, 0.0005, 0.0005, 0.0002, -0.0003, -0.0008, -0.0010, -0.0009, -0.0007, -0.0006, -0.0006, -0.0005, -0.0005, -0.0005, -0.0005, -0.0007, -0.0008, -0.0009, -0.0009, -0.0008, -0.0007, -0.0008, -0.0010, -0.0011],
            ]
        ),
        "24.0": torch.tensor(
            [
                [ 0.0010, 0.0008, 0.0018, 0.0021, 0.0014, 0.0011, 0.0009, 0.0007, 0.0006, 0.0006, 0.0005, 0.0003, 0.0003, 0.0002, 0.0001, 0.0001, 0.0001, 0.0002, 0.0002, 0.0001, -0.0002, -0.0004, -0.0003, 0.0000, 0.0005, 0.0011, 0.0018, 0.0022, 0.0022, 0.0018, 0.0012, 0.0007, 0.0004, 0.0003, 0.0004, 0.0006, 0.0006, 0.0007, 0.0007, 0.0009, 0.0008, 0.0007, 0.0005, 0.0004, 0.0004, 0.0006, 0.0007, 0.0007, 0.0005, 0.0004],
                [-0.0039, -0.0035, -0.0027, -0.0026, -0.0028, -0.0031, -0.0035, -0.0035, -0.0034, -0.0033, -0.0032, -0.0032, -0.0031, -0.0031, -0.0029, -0.0028, -0.0026, -0.0024, -0.0021, -0.0021, -0.0024, -0.0025, -0.0024, -0.0021, -0.0017, -0.0011, -0.0006, -0.0002, -0.0002, -0.0004, -0.0009, -0.0013, -0.0015, -0.0015, -0.0014, -0.0013, -0.0012, -0.0011, -0.0010, -0.0010, -0.0011, -0.0012, -0.0014, -0.0015, -0.0015, -0.0014, -0.0013, -0.0014, -0.0016, -0.0017],
            ]
        )
    }
}
EXPECTED_CODEC_ERROR = {
    "facebook/encodec_24khz": {
        "1.5": 0.0022229827009141445,
        "3.0": 0.001862662611529231,
        "6.0": 0.0015231302240863442,
        "12.0": 0.0013,
        "24.0": 0.0012,
    },
    "facebook/encodec_48khz": {
        "3.0": 0.000840399123262614,
        "6.0": 0.0006692984024994075,
        "12.0": 0.0005328940460458398,
        "24.0": 0.0004473362350836396,
    }
}
# -- test_batch
EXPECTED_ENCODER_CODES_BATCH = {
    "facebook/encodec_24khz": {
        "1.5": torch.tensor(
            [
                [
                    [62, 106, 475, 475, 404, 404, 475, 404, 404, 475, 475, 404, 475, 475, 475, 835, 475, 475, 835, 835,
                     106, 106, 738, 106, 738, 106, 408, 408, 738, 408, 408, 408, 738, 408, 408, 408, 408, 738, 408,
                     1017, 604, 64, 303, 394, 5, 570, 991, 570, 969, 814],
                    [424, 969, 913, 1007, 544, 1007, 1007, 1007, 969, 1007, 729, 1007, 961, 1007, 1007, 961, 969, 1007,
                     1007, 424, 518, 1007, 544, 1007, 518, 913, 424, 424, 544, 424, 518, 518, 518, 302, 424, 424, 424,
                     544, 424, 114, 200, 787, 931, 343, 434, 315, 487, 872, 769, 463],

                ],
                [
                    [835, 835, 835, 835, 835, 835, 835, 835, 835, 835, 835, 835, 408, 835, 738, 408, 408, 408, 408, 408,
                     408, 738, 408, 408, 408, 408, 408, 408, 408, 408, 738, 408, 408, 408, 408, 408, 408, 408, 408, 408,
                     339, 834, 819, 875, 957, 670, 811, 670, 237, 53],
                    [857, 857, 544, 518, 937, 518, 913, 913, 518, 913, 518, 913, 518, 518, 544, 424, 424, 518, 424, 424,
                     424, 544, 424, 424, 424, 518, 424, 518, 518, 937, 544, 424, 518, 302, 518, 424, 424, 518, 424, 424,
                     913, 857, 841, 363, 463, 78, 176, 645, 255, 571],

                ],

            ]

        ),
        "3.0": torch.tensor(
            [
                [
                    [62, 106, 475, 475, 404, 404, 475, 404, 404, 475],
                    [424, 969, 913, 1007, 544, 1007, 1007, 1007, 969, 1007],
                    [212, 832, 212, 36, 36, 36, 767, 653, 982, 1016],
                    [956, 741, 838, 1019, 739, 780, 838, 1019, 1014, 1019],

                ],
                [
                    [835, 835, 835, 835, 835, 835, 835, 835, 835, 835],
                    [857, 857, 544, 518, 937, 518, 913, 913, 518, 913],
                    [705, 989, 934, 989, 678, 934, 934, 786, 934, 786],
                    [366, 1018, 398, 398, 398, 398, 673, 741, 398, 741],

                ],
            ]
        ),
        "6.0": torch.tensor(
            [
                [
                    [62, 106, 475, 475, 404, 404, 475, 404, 404, 475],
                    [424, 969, 913, 1007, 544, 1007, 1007, 1007, 969, 1007],
                    [212, 832, 212, 36, 36, 36, 767, 653, 982, 1016],
                    [956, 741, 838, 1019, 739, 780, 838, 1019, 1014, 1019],
                    [712, 862, 712, 448, 528, 646, 446, 373, 694, 373],
                    [939, 881, 939, 19, 334, 881, 1005, 763, 632, 781],
                    [853, 464, 772, 782, 782, 983, 890, 874, 983, 782],
                    [899, 475, 173, 701, 701, 947, 468, 1019, 882, 518],

                ],
                [
                    [835, 835, 835, 835, 835, 835, 835, 835, 835, 835],
                    [857, 857, 544, 518, 937, 518, 913, 913, 518, 913],
                    [705, 989, 934, 989, 678, 934, 934, 786, 934, 786],
                    [366, 1018, 398, 398, 398, 398, 673, 741, 398, 741],
                    [373, 373, 375, 373, 373, 222, 862, 373, 190, 373],
                    [293, 949, 435, 435, 435, 293, 949, 881, 632, 986],
                    [800, 528, 528, 853, 782, 485, 772, 900, 528, 853],
                    [916, 237, 828, 701, 518, 835, 948, 315, 948, 315],

                ],
            ]
        ),
        "12.0": torch.tensor(
            [
                [
                    [62, 106, 475, 475, 404, 404, 475, 404, 404, 475],
                    [424, 969, 913, 1007, 544, 1007, 1007, 1007, 969, 1007],
                    [212, 832, 212, 36, 36, 36, 767, 653, 982, 1016],
                    [956, 741, 838, 1019, 739, 780, 838, 1019, 1014, 1019],
                    [712, 862, 712, 448, 528, 646, 446, 373, 694, 373],
                    [939, 881, 939, 19, 334, 881, 1005, 763, 632, 781],
                    [853, 464, 772, 782, 782, 983, 890, 874, 983, 782],
                    [899, 475, 173, 701, 701, 947, 468, 1019, 882, 518],
                    [817, 470, 588, 675, 675, 588, 960, 927, 909, 466],
                    [953, 776, 717, 630, 359, 717, 861, 630, 861, 359],
                    [623, 740, 1000, 388, 420, 388, 740, 818, 958, 743],
                    [413, 835, 742, 249, 892, 352, 190, 498, 866, 890],
                    [817, 351, 804, 751, 938, 535, 434, 879, 351, 971],
                    [792, 495, 935, 848, 792, 795, 942, 935, 723, 531],
                    [622, 681, 477, 713, 752, 871, 713, 514, 993, 777],
                    [928, 799, 962, 1005, 860, 439, 312, 922, 982, 922],
                ],
                [
                    [835, 835, 835, 835, 835, 835, 835, 835, 835, 835],
                    [857, 857, 544, 518, 937, 518, 913, 913, 518, 913],
                    [705, 989, 934, 989, 678, 934, 934, 786, 934, 786],
                    [366, 1018, 398, 398, 398, 398, 673, 741, 398, 741],
                    [373, 373, 375, 373, 373, 222, 862, 373, 190, 373],
                    [293, 949, 435, 435, 435, 293, 949, 881, 632, 986],
                    [800, 528, 528, 853, 782, 485, 772, 900, 528, 853],
                    [916, 237, 828, 701, 518, 835, 948, 315, 948, 315],
                    [420, 628, 918, 628, 628, 628, 248, 628, 909, 811],
                    [736, 717, 994, 974, 477, 874, 963, 979, 355, 979],
                    [1002, 1002, 894, 875, 388, 709, 534, 408, 881, 709],
                    [735, 828, 763, 742, 640, 835, 828, 375, 840, 375],
                    [898, 938, 556, 658, 410, 951, 486, 658, 877, 877],
                    [ 0, 797, 428, 694, 428, 920, 1022, 1022, 809, 797],
                    [622, 421, 422, 776, 911, 911, 958, 421, 776, 421],
                    [1005, 312, 922, 755, 834, 461, 461, 702, 597, 974],

                ],
            ]
        ),
        "24.0": torch.tensor(
            [
                [
                    [62, 106, 475, 475, 404, 404, 475, 404, 404, 475],
                    [424, 969, 913, 1007, 544, 1007, 1007, 1007, 969, 1007],
                    [212, 832, 212, 36, 36, 36, 767, 653, 982, 1016],
                    [956, 741, 838, 1019, 739, 780, 838, 1019, 1014, 1019],
                    [712, 862, 712, 448, 528, 646, 446, 373, 694, 373],
                    [939, 881, 939, 19, 334, 881, 1005, 763, 632, 781],
                    [853, 464, 772, 782, 782, 983, 890, 874, 983, 782],
                    [899, 475, 173, 701, 701, 947, 468, 1019, 882, 518],
                    [817, 470, 588, 675, 675, 588, 960, 927, 909, 466],
                    [953, 776, 717, 630, 359, 717, 861, 630, 861, 359],
                    [623, 740, 1000, 388, 420, 388, 740, 818, 958, 743],
                    [413, 835, 742, 249, 892, 352, 190, 498, 866, 890],
                    [817, 351, 804, 751, 938, 535, 434, 879, 351, 971],
                    [792, 495, 935, 848, 792, 795, 942, 935, 723, 531],
                    [622, 681, 477, 713, 752, 871, 713, 514, 993, 777],
                    [928, 799, 962, 1005, 860, 439, 312, 922, 982, 922],
                    [939, 637, 861, 506, 861, 61, 475, 264, 1019, 260],
                    [166, 215, 69, 69, 890, 69, 284, 828, 396, 180],
                    [561, 896, 841, 144, 580, 659, 886, 514, 686, 451],
                    [691, 691, 239, 735, 62, 287, 383, 972, 550, 505],
                    [451, 811, 238, 251, 250, 841, 734, 329, 551, 846],
                    [313, 601, 494, 763, 811, 565, 748, 441, 601, 480],
                    [653, 242, 630, 572, 701, 973, 632, 374, 561, 521],
                    [984, 987, 419, 454, 386, 507, 532, 636, 515, 671],
                    [647, 550, 515, 292, 876, 1011, 719, 549, 691, 911],
                    [683, 536, 656, 603, 698, 867, 987, 857, 886, 491],
                    [444, 937, 826, 555, 585, 710, 466, 852, 655, 591],
                    [658, 952, 903, 508, 739, 596, 420, 721, 464, 306],
                    [665, 334, 765, 532, 618, 278, 836, 838, 517, 597],
                    [613, 674, 596, 904, 987, 977, 938, 615, 672, 776],
                    [689, 386, 749, 658, 250, 869, 957, 806, 750, 659],
                    [652, 509, 910, 826, 566, 622, 951, 696, 900, 895],
                ],
                [
                    [835, 835, 835, 835, 835, 835, 835, 835, 835, 835],
                    [857, 857, 544, 518, 937, 518, 913, 913, 518, 913],
                    [705, 989, 934, 989, 678, 934, 934, 786, 934, 786],
                    [366, 1018, 398, 398, 398, 398, 673, 741, 398, 741],
                    [373, 373, 375, 373, 373, 222, 862, 373, 190, 373],
                    [293, 949, 435, 435, 435, 293, 949, 881, 632, 986],
                    [800, 528, 528, 853, 782, 485, 772, 900, 528, 853],
                    [916, 237, 828, 701, 518, 835, 948, 315, 948, 315],
                    [420, 628, 918, 628, 628, 628, 248, 628, 909, 811],
                    [736, 717, 994, 974, 477, 874, 963, 979, 355, 979],
                    [1002, 1002, 894, 875, 388, 709, 534, 408, 881, 709],
                    [735, 828, 763, 742, 640, 835, 828, 375, 840, 375],
                    [898, 938, 556, 658, 410, 951, 486, 658, 877, 877],
                    [ 0, 797, 428, 694, 428, 920, 1022, 1022, 809, 797],
                    [622, 421, 422, 776, 911, 911, 958, 421, 776, 421],
                    [1005, 312, 922, 755, 834, 461, 461, 702, 597, 974],
                    [248, 248, 637, 248, 977, 506, 546, 270, 670, 506],
                    [547, 447, 15, 134, 1009, 215, 134, 396, 260, 160],
                    [635, 497, 686, 765, 264, 497, 244, 675, 624, 656],
                    [864, 571, 616, 511, 588, 781, 525, 258, 674, 503],
                    [449, 757, 857, 451, 658, 486, 299, 299, 251, 596],
                    [809, 628, 255, 568, 623, 301, 639, 546, 617, 623],
                    [551, 497, 908, 539, 661, 710, 640, 539, 646, 315],
                    [689, 507, 875, 515, 613, 637, 527, 515, 662, 637],
                    [983, 686, 456, 768, 601, 561, 768, 653, 500, 688],
                    [493, 566, 664, 782, 683, 683, 721, 603, 323, 497],
                    [1015, 552, 411, 423, 607, 646, 687, 1018, 689, 607],
                    [516, 293, 471, 294, 293, 294, 608, 538, 803, 717],
                    [974, 994, 952, 637, 637, 927, 535, 571, 602, 535],
                    [776, 789, 476, 944, 652, 959, 589, 679, 321, 623],
                    [776, 931, 720, 1009, 676, 731, 386, 676, 701, 676],
                    [684, 543, 716, 392, 661, 517, 792, 588, 922, 676],
                ],
            ]
        )
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([[[790, 790, 790, 214, 214, 214, 799, 214, 214, 214],
            [989, 989,  77, 546, 989, 546, 989, 160, 546, 989]],
            [[214, 214, 214, 214, 214, 214, 214, 214, 214, 214],
            [289, 289, 989, 764, 289, 289, 882, 882, 882, 882]]]),
        "6.0": torch.tensor([[[ 790,  790,  790,  214,  214,  214,  799,  214,  214,  214],
            [ 989,  989,   77,  546,  989,  546,  989,  160,  546,  989],
            [ 977,  977,  977,  977,  538,  977,  977,  960,  977,  977],
            [ 376,  376,  962,  962,  607,  962,  963,  896,  962,  376]],
            [[ 214,  214,  214,  214,  214,  214,  214,  214,  214,  214],
            [ 289,  289,  989,  764,  289,  289,  882,  882,  882,  882],
            [1022, 1022,  471,  925,  821,  821,  267,  925,  925,  267],
            [ 979,  992,  914,  921,    0,    0, 1023,  963,  963, 1023]]]),
        "12.0": torch.tensor([[[ 790,  790,  790,  214,  214,  214,  799,  214,  214,  214],
            [ 989,  989,   77,  546,  989,  546,  989,  160,  546,  989],
            [ 977,  977,  977,  977,  538,  977,  977,  960,  977,  977],
            [ 376,  376,  962,  962,  607,  962,  963,  896,  962,  376],
            [ 979,  979,  979, 1012,  979, 1012,  921,    0, 1002,  695],
            [ 824, 1018,  762,  957,  824,  762,  762, 1007,  957,  336],
            [ 681,  973,  973,  452,  211,  681,  802,  679,  547,  884],
            [ 950, 1017, 1016, 1017,  986, 1017,  229,  607, 1017,  689]],

            [[ 214,  214,  214,  214,  214,  214,  214,  214,  214,  214],
            [ 289,  289,  989,  764,  289,  289,  882,  882,  882,  882],
            [1022, 1022,  471,  925,  821,  821,  267,  925,  925,  267],
            [ 979,  992,  914,  921,    0,    0, 1023,  963,  963, 1023],
            [ 403,  940,  976, 1018,  677, 1002,  979,  677,  677,  677],
            [1018,  794,  762,  444,  485,  485,  974,  548,  548, 1018],
            [ 679,  243,  679, 1005, 1005,  973, 1014, 1005, 1005, 1014],
            [ 810,   13, 1017,  537,  522,  702,  202, 1017, 1017,   15]]]),
        "24.0": torch.tensor(
            [
                [
                    [790, 790, 790, 214, 214, 214, 799, 214, 214, 214],
                    [989, 989, 77, 546, 989, 546, 989, 160, 546, 989],
                    [977, 977, 977, 977, 538, 977, 977, 960, 977, 977],
                    [376, 376, 962, 962, 607, 962, 963, 896, 962, 376],
                    [979, 979, 979, 1012, 979, 1012, 921, 0, 1002, 695],
                    [824, 1018, 762, 957, 824, 762, 762, 1007, 957, 336],
                    [681, 973, 973, 452, 211, 681, 802, 679, 547, 884],
                    [950, 1017, 1016, 1017, 986, 1017, 229, 607, 1017, 689],
                    [1004, 1011, 669, 1023, 1023, 1023, 905, 297, 810, 970],
                    [982, 681, 982, 629, 662, 919, 878, 476, 629, 982],
                    [727, 727, 959, 959, 979, 959, 530, 959, 337, 961],
                    [924, 456, 924, 486, 924, 959, 102, 924, 805, 924],
                    [649, 542, 993, 993, 949, 787, 56, 886, 949, 405],
                    [864, 1022, 1022, 1022, 460, 753, 805, 309, 1022, 32],
                    [953, 0, 0, 180, 352, 10, 581, 516, 322, 452],
                    [300, 0, 1020, 307, 0, 543, 924, 627, 258, 262],
                ],
                [
                    [214, 214, 214, 214, 214, 214, 214, 214, 214, 214],
                    [289, 289, 989, 764, 289, 289, 882, 882, 882, 882],
                    [1022, 1022, 471, 925, 821, 821, 267, 925, 925, 267],
                    [979, 992, 914, 921, 0, 0, 1023, 963, 963, 1023],
                    [403, 940, 976, 1018, 677, 1002, 979, 677, 677, 677],
                    [1018, 794, 762, 444, 485, 485, 974, 548, 548, 1018],
                    [679, 243, 679, 1005, 1005, 973, 1014, 1005, 1005, 1014],
                    [810, 13, 1017, 537, 522, 702, 202, 1017, 1017, 15],
                    [728, 252, 970, 984, 971, 950, 673, 902, 1011, 810],
                    [332, 1014, 476, 854, 1014, 861, 332, 411, 411, 408],
                    [959, 727, 611, 979, 611, 727, 999, 497, 821, 0],
                    [995, 698, 924, 688, 102, 510, 924, 970, 344, 961],
                    [ 81, 516, 847, 924, 10, 240, 1005, 726, 993, 378],
                    [467, 496, 484, 496, 456, 1022, 337, 600, 456, 1022],
                    [789, 65, 937, 976, 159, 953, 343, 764, 179, 159],
                    [ 10, 790, 483, 10, 1020, 352, 848, 333, 83, 848],
                ],
            ]
        )
    }
}
EXPECTED_ENCODER_SCALES_BATCH = {
    "facebook/encodec_24khz": {
        "1.5": None,
        "3.0": None,
        "6.0": None,
        "12.0": None,
        "24.0": None
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([[[1.027247e-01],
            [7.877284e-02]],
            [[1.014922e-01],
            [8.696266e-02]],
            [[6.308002e-02],
            [7.748771e-02]],
            [[6.899278e-02],
            [1.045912e-01]],
            [[6.440169e-02],
            [8.843135e-02]],
            [[4.139878e-02],
            [1.000000e-08]],
            [[5.848629e-02],
            [1.000000e-08]],
            [[2.329416e-04],
            [1.000000e-08]],
            [[1.000000e-08],
            [1.000000e-08]]]),
        "6.0": torch.tensor([[[1.027247e-01],
            [7.877284e-02]],
            [[1.014922e-01],
            [8.696266e-02]],
            [[6.308002e-02],
            [7.748771e-02]],
            [[6.899278e-02],
            [1.045912e-01]],
            [[6.440169e-02],
            [8.843135e-02]],
            [[4.139878e-02],
            [1.000000e-08]],
            [[5.848629e-02],
            [1.000000e-08]],
            [[2.329416e-04],
            [1.000000e-08]],
            [[1.000000e-08],
            [1.000000e-08]]]),
        "12.0": torch.tensor([[[1.027247e-01],
            [7.877284e-02]],
            [[1.014922e-01],
            [8.696266e-02]],
            [[6.308002e-02],
            [7.748771e-02]],
            [[6.899278e-02],
            [1.045912e-01]],
            [[6.440169e-02],
            [8.843135e-02]],
            [[4.139878e-02],
            [1.000000e-08]],
            [[5.848629e-02],
            [1.000000e-08]],
            [[2.329416e-04],
            [1.000000e-08]],
            [[1.000000e-08],
            [1.000000e-08]]]),
        "24.0": torch.tensor([[[1.027247e-01],
            [7.877284e-02]],
            [[1.014922e-01],
            [8.696266e-02]],
            [[6.308002e-02],
            [7.748771e-02]],
            [[6.899278e-02],
            [1.045912e-01]],
            [[6.440169e-02],
            [8.843135e-02]],
            [[4.139878e-02],
            [1.000000e-08]],
            [[5.848629e-02],
            [1.000000e-08]],
            [[2.329416e-04],
            [1.000000e-08]],
            [[1.000000e-08],
            [1.000000e-08]]])
    }
}
EXPECTED_DECODER_OUTPUTS_BATCH = {
    "facebook/encodec_24khz": {
        "1.5": torch.tensor(
            [
                [[ 0.0010, 0.0004, 0.0005, 0.0002, 0.0005, -0.0001, -0.0003, -0.0001, 0.0003, 0.0001, -0.0014, -0.0009, -0.0007, -0.0023, -0.0009, 0.0008, 0.0007, 0.0003, 0.0001, 0.0001, 0.0003, -0.0001, -0.0003, -0.0004, -0.0005, -0.0007, -0.0009, -0.0011, -0.0010, -0.0006, -0.0007, -0.0007, -0.0005, -0.0005, -0.0003, -0.0002, -0.0002, -0.0001, -0.0005, -0.0008, -0.0005, -0.0007, -0.0009, -0.0002, 0.0003, 0.0005, 0.0004, 0.0001, 0.0003, 0.0004]],
                [[ -0.0001, -0.0000, 0.0003, 0.0001, 0.0005, 0.0001, -0.0006, -0.0002, 0.0002, 0.0002, -0.0031, -0.0004, 0.0006, -0.0066, -0.0032, 0.0044, 0.0025, -0.0019, -0.0017, 0.0001, 0.0019, -0.0010, -0.0014, -0.0009, -0.0007, -0.0009, -0.0019, -0.0024, -0.0019, -0.0001, -0.0017, -0.0022, -0.0004, 0.0005, -0.0014, -0.0023, 0.0002, 0.0015, -0.0022, -0.0033, 0.0024, 0.0009, -0.0041, 0.0000, 0.0030, 0.0020, -0.0015, -0.0018, 0.0014, 0.0007]],
            ]
        ),
        "3.0": torch.tensor(
            [
                [[ 0.0013, 0.0007, 0.0009, 0.0005, 0.0006, 0.0002, -0.0001, 0.0000, 0.0005, 0.0003, -0.0012, -0.0006, -0.0003, -0.0019, -0.0003, 0.0015, 0.0013, 0.0009, 0.0008, 0.0007, 0.0008, 0.0004, 0.0001, -0.0000, -0.0001, -0.0002, -0.0003, -0.0004, -0.0004, 0.0001, -0.0000, -0.0000, 0.0003, 0.0003, 0.0005, 0.0005, 0.0004, 0.0005, 0.0001, -0.0003, -0.0002, -0.0004, -0.0006, 0.0003, 0.0009, 0.0012, 0.0013, 0.0012, 0.0014, 0.0015]],
                [[ 0.0000, -0.0003, 0.0005, 0.0004, 0.0011, 0.0013, 0.0002, 0.0005, 0.0002, 0.0006, -0.0025, -0.0005, 0.0004, -0.0069, -0.0027, 0.0038, 0.0013, -0.0015, -0.0005, 0.0003, 0.0014, -0.0006, -0.0002, -0.0010, -0.0008, -0.0001, -0.0006, -0.0012, -0.0016, 0.0010, 0.0001, -0.0010, -0.0002, 0.0013, -0.0002, -0.0017, 0.0005, 0.0019, -0.0019, -0.0035, 0.0022, -0.0001, -0.0040, 0.0012, 0.0015, 0.0012, 0.0001, -0.0010, 0.0005, 0.0004]],
            ]
        ),
        "6.0": torch.tensor(
            [
                [[ 0.0010, 0.0005, 0.0007, 0.0001, 0.0003, -0.0000, -0.0002, -0.0001, 0.0003, 0.0001, -0.0014, -0.0007, -0.0004, -0.0019, -0.0004, 0.0013, 0.0012, 0.0008, 0.0007, 0.0007, 0.0008, 0.0003, 0.0001, 0.0001, -0.0000, -0.0001, -0.0001, -0.0002, -0.0001, 0.0002, 0.0002, 0.0001, 0.0005, 0.0005, 0.0008, 0.0008, 0.0007, 0.0008, 0.0004, 0.0001, 0.0002, -0.0001, -0.0002, 0.0006, 0.0012, 0.0015, 0.0016, 0.0014, 0.0016, 0.0017]],
                [[ -0.0005, -0.0001, 0.0003, 0.0001, 0.0010, 0.0012, 0.0002, 0.0004, 0.0012, 0.0003, -0.0023, -0.0003, -0.0005, -0.0063, -0.0026, 0.0040, 0.0024, -0.0018, -0.0005, 0.0016, 0.0004, -0.0008, 0.0009, 0.0002, -0.0015, -0.0003, 0.0004, -0.0011, -0.0013, 0.0012, 0.0001, -0.0019, 0.0007, 0.0021, -0.0009, -0.0016, 0.0015, 0.0013, -0.0022, -0.0015, 0.0016, -0.0014, -0.0033, 0.0017, 0.0025, -0.0004, -0.0005, 0.0010, 0.0005, 0.0001]],
            ]
        ),
        "12.0": torch.tensor(
            [
                [[ 0.0003, 0.0002, 0.0004, -0.0004, -0.0003, -0.0007, -0.0008, -0.0006, -0.0001, -0.0002, -0.0016, -0.0009, -0.0004, -0.0021, -0.0003, 0.0015, 0.0016, 0.0012, 0.0011, 0.0010, 0.0010, 0.0005, 0.0002, 0.0001, 0.0000, -0.0001, -0.0002, -0.0004, -0.0004, 0.0000, -0.0000, -0.0002, 0.0001, 0.0001, 0.0004, 0.0003, 0.0002, 0.0004, -0.0001, -0.0005, -0.0004, -0.0006, -0.0007, 0.0003, 0.0009, 0.0013, 0.0015, 0.0015, 0.0017, 0.0018]],
                [[ -0.0008, -0.0003, 0.0003, -0.0001, 0.0008, 0.0013, 0.0004, 0.0008, 0.0015, 0.0006, -0.0021, -0.0001, -0.0003, -0.0062, -0.0022, 0.0043, 0.0028, -0.0013, -0.0002, 0.0017, 0.0010, -0.0001, 0.0008, 0.0001, -0.0010, 0.0003, 0.0008, -0.0006, -0.0007, 0.0012, 0.0003, -0.0013, 0.0007, 0.0019, -0.0002, -0.0013, 0.0011, 0.0016, -0.0016, -0.0017, 0.0014, -0.0006, -0.0029, 0.0011, 0.0028, 0.0006, -0.0004, 0.0005, 0.0008, 0.0003]],
            ]
        ),
        "24.0": torch.tensor(
            [
                [[ 0.0009, 0.0004, 0.0007, 0.0002, 0.0004, -0.0001, -0.0003, -0.0002, 0.0002, 0.0001, -0.0015, -0.0009, -0.0006, -0.0024, -0.0005, 0.0016, 0.0014, 0.0010, 0.0009, 0.0008, 0.0008, 0.0004, 0.0001, 0.0000, -0.0001, -0.0002, -0.0003, -0.0006, -0.0006, -0.0003, -0.0005, -0.0006, -0.0003, -0.0004, -0.0001, -0.0002, -0.0003, -0.0001, -0.0006, -0.0011, -0.0008, -0.0010, -0.0012, -0.0000, 0.0007, 0.0011, 0.0012, 0.0011, 0.0013, 0.0014]],
                [[ -0.0009, -0.0004, 0.0001, -0.0003, 0.0007, 0.0012, 0.0003, 0.0006, 0.0017, 0.0008, -0.0020, 0.0001, -0.0002, -0.0064, -0.0023, 0.0047, 0.0029, -0.0016, -0.0004, 0.0019, 0.0010, -0.0002, 0.0007, -0.0001, -0.0013, 0.0005, 0.0012, -0.0007, -0.0008, 0.0013, -0.0001, -0.0022, 0.0004, 0.0020, -0.0004, -0.0014, 0.0017, 0.0020, -0.0018, -0.0016, 0.0015, -0.0015, -0.0036, 0.0014, 0.0030, 0.0004, 0.0002, 0.0015, 0.0011, 0.0007]],
            ]
        )
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([[[ 0.005083,  0.004669,  0.005723,  0.005600,  0.004231,  0.003830,
            0.003684,  0.003349,  0.003032,  0.003055,  0.002768,  0.002370,
            0.002384,  0.002450,  0.002391,  0.002363,  0.002357,  0.002435,
            0.002568,  0.002463,  0.002137,  0.002092,  0.002440,  0.002772,
            0.003035,  0.003473,  0.003963,  0.004288,  0.004315,  0.004087,
            0.003618,  0.003166,  0.002874,  0.002775,  0.002820,  0.002758,
            0.002565,  0.002498,  0.002583,  0.002671,  0.002656,  0.002613,
            0.002433,  0.002236,  0.002215,  0.002302,  0.002287,  0.002113,
            0.001909,  0.001767],
            [-0.003928, -0.002733, -0.001330, -0.001914, -0.002927, -0.003272,
            -0.003677, -0.003615, -0.003341, -0.002907, -0.002764, -0.002742,
            -0.002593, -0.002308, -0.002024, -0.001856, -0.001672, -0.001256,
            -0.000929, -0.001217, -0.001864, -0.002118, -0.002025, -0.001932,
            -0.001816, -0.001572, -0.001214, -0.000885, -0.000829, -0.000976,
            -0.001417, -0.001874, -0.002030, -0.001952, -0.001858, -0.001863,
            -0.001895, -0.001843, -0.001801, -0.001792, -0.001812, -0.001865,
            -0.002008, -0.002120, -0.002132, -0.002093, -0.002170, -0.002370,
            -0.002587, -0.002749]],
            [[ 0.004229,  0.003422,  0.005044,  0.006059,  0.005242,  0.004623,
            0.004231,  0.004050,  0.004314,  0.004701,  0.004559,  0.004105,
            0.003874,  0.003713,  0.003355,  0.003055,  0.003235,  0.003927,
            0.004500,  0.004195,  0.003328,  0.002804,  0.002628,  0.002456,
            0.002693,  0.003883,  0.005604,  0.006791,  0.006702,  0.005427,
            0.003622,  0.002328,  0.002173,  0.002871,  0.003505,  0.003410,
            0.002851,  0.002511,  0.002534,  0.002685,  0.002714,  0.002538,
            0.002110,  0.001697,  0.001786,  0.002415,  0.002940,  0.002856,
            0.002348,  0.001883],
            [-0.003444, -0.002916, -0.000590,  0.000157, -0.000702, -0.001472,
            -0.002032, -0.001891, -0.001283, -0.000670, -0.000590, -0.000875,
            -0.001090, -0.001095, -0.001172, -0.001287, -0.000907,  0.000111,
            0.000858,  0.000471, -0.000532, -0.001127, -0.001463, -0.001853,
            -0.001762, -0.000666,  0.000964,  0.002054,  0.001914,  0.000743,
            -0.000876, -0.001990, -0.001951, -0.001042, -0.000229, -0.000171,
            -0.000558, -0.000752, -0.000704, -0.000609, -0.000594, -0.000723,
            -0.001085, -0.001455, -0.001374, -0.000795, -0.000350, -0.000480,
            -0.000993, -0.001432]]]),
        "6.0": torch.tensor([[[ 5.892794e-03,  5.767163e-03,  7.065284e-03,  7.068626e-03,
            5.825328e-03,  5.601424e-03,  5.582351e-03,  5.209565e-03,
            4.829186e-03,  4.809568e-03,  4.663883e-03,  4.402087e-03,
            4.337528e-03,  4.311915e-03,  4.236566e-03,  4.209972e-03,
            4.179818e-03,  4.196202e-03,  4.309553e-03,  4.267083e-03,
            4.052189e-03,  4.068719e-03,  4.381632e-03,  4.692366e-03,
            4.998885e-03,  5.466312e-03,  5.895300e-03,  6.115717e-03,
            6.055626e-03,  5.773376e-03,  5.316667e-03,  4.826934e-03,
            4.450697e-03,  4.315911e-03,  4.310716e-03,  4.202125e-03,
            4.008702e-03,  3.957694e-03,  4.017603e-03,  4.060654e-03,
            4.036821e-03,  3.923071e-03,  3.659022e-03,  3.427053e-03,
            3.387271e-03,  3.462438e-03,  3.434755e-03,  3.247944e-03,
            3.009581e-03,  2.800536e-03],
            [-1.867314e-03, -6.082351e-04,  9.374358e-04,  5.555808e-04,
            -3.020080e-04, -5.281629e-04, -9.364292e-04, -1.057594e-03,
            -9.703087e-04, -6.292185e-04, -4.193477e-04, -3.605868e-04,
            -2.948678e-04, -1.198237e-04,  4.924605e-05,  1.602105e-04,
            3.162385e-04,  6.700790e-04,  9.868707e-04,  8.484383e-04,
            4.327767e-04,  3.108105e-04,  4.244343e-04,  5.422112e-04,
            7.239584e-04,  1.008546e-03,  1.265120e-03,  1.447669e-03,
            1.436084e-03,  1.271058e-03,  8.684017e-04,  4.149990e-04,
            2.143449e-04,  2.508474e-04,  3.018488e-04,  2.782424e-04,
            2.369677e-04,  3.040710e-04,  3.242530e-04,  2.599912e-04,
            2.211208e-04,  1.311762e-04, -9.807519e-05, -2.752687e-04,
            -3.114068e-04, -2.832832e-04, -3.900219e-04, -6.142824e-04,
            -8.507833e-04, -1.055882e-03]],
            [[ 3.971702e-04, -2.164055e-04,  1.562327e-03,  2.695718e-03,
            2.374928e-03,  2.145125e-03,  1.870762e-03,  1.852614e-03,
            2.074345e-03,  2.312302e-03,  2.222824e-03,  1.876336e-03,
            1.609606e-03,  1.420574e-03,  1.193270e-03,  9.592943e-04,
            1.132237e-03,  1.776782e-03,  2.258269e-03,  1.945908e-03,
            9.930646e-04,  1.733529e-04, -2.533881e-04, -3.138177e-04,
            3.226010e-04,  1.859203e-03,  3.879325e-03,  5.267750e-03,
            5.101699e-03,  3.609465e-03,  1.653315e-03,  2.709297e-04,
            -3.190451e-05,  5.129501e-04,  1.224789e-03,  1.397457e-03,
            1.110794e-03,  8.736057e-04,  8.860155e-04,  1.055910e-03,
            1.100855e-03,  8.834896e-04,  3.825913e-04, -3.267327e-05,
            6.586456e-05,  7.147206e-04,  1.394876e-03,  1.535393e-03,
            1.192172e-03,  7.061819e-04],
            [-6.897163e-03, -6.407891e-03, -4.015491e-03, -3.082125e-03,
            -3.434983e-03, -3.885052e-03, -4.456392e-03, -4.296550e-03,
            -3.861045e-03, -3.553474e-03, -3.547473e-03, -3.800863e-03,
            -4.123025e-03, -4.237277e-03, -4.244958e-03, -4.263899e-03,
            -3.808572e-03, -2.811858e-03, -2.147519e-03, -2.516703e-03,
            -3.550721e-03, -4.353373e-03, -4.846224e-03, -4.960613e-03,
            -4.273535e-03, -2.714785e-03, -7.043980e-04,  6.689885e-04,
            5.069164e-04, -9.122533e-04, -2.816979e-03, -4.124952e-03,
            -4.235019e-03, -3.491365e-03, -2.676077e-03, -2.381226e-03,
            -2.492559e-03, -2.634424e-03, -2.632524e-03, -2.528266e-03,
            -2.536691e-03, -2.746170e-03, -3.187869e-03, -3.553530e-03,
            -3.462211e-03, -2.862707e-03, -2.273719e-03, -2.201617e-03,
            -2.565818e-03, -3.044683e-03]]]),
        "12.0": torch.tensor([[[ 2.237194e-03,  2.508208e-03,  3.986347e-03,  4.020395e-03,
            2.889890e-03,  2.733388e-03,  2.684146e-03,  2.251372e-03,
            1.787451e-03,  1.720550e-03,  1.689184e-03,  1.495478e-03,
            1.321027e-03,  1.185375e-03,  1.098422e-03,  1.055453e-03,
            9.591801e-04,  9.328910e-04,  1.026154e-03,  1.031992e-03,
            9.155220e-04,  9.732856e-04,  1.282264e-03,  1.624059e-03,
            1.920021e-03,  2.333685e-03,  2.730524e-03,  2.919153e-03,
            2.856711e-03,  2.632692e-03,  2.256703e-03,  1.901129e-03,
            1.684760e-03,  1.638201e-03,  1.644909e-03,  1.569378e-03,
            1.448412e-03,  1.478291e-03,  1.580583e-03,  1.633777e-03,
            1.597190e-03,  1.475462e-03,  1.242885e-03,  1.065243e-03,
            1.052842e-03,  1.103825e-03,  1.059115e-03,  9.251673e-04,
            7.235570e-04,  5.053390e-04],
            [-4.534880e-03, -3.111026e-03, -1.486247e-03, -1.739966e-03,
            -2.399862e-03, -2.583335e-03, -3.157276e-03, -3.517166e-03,
            -3.598212e-03, -3.303007e-03, -3.037215e-03, -2.982930e-03,
            -3.026671e-03, -2.958387e-03, -2.836909e-03, -2.775315e-03,
            -2.719575e-03, -2.431532e-03, -2.090512e-03, -2.095603e-03,
            -2.366266e-03, -2.404480e-03, -2.235661e-03, -2.063206e-03,
            -1.888533e-03, -1.640449e-03, -1.407782e-03, -1.250053e-03,
            -1.275359e-03, -1.373277e-03, -1.601508e-03, -1.838720e-03,
            -1.876643e-03, -1.736149e-03, -1.622051e-03, -1.578928e-03,
            -1.564748e-03, -1.455850e-03, -1.391748e-03, -1.418254e-03,
            -1.462577e-03, -1.554713e-03, -1.730076e-03, -1.829485e-03,
            -1.816249e-03, -1.772218e-03, -1.855736e-03, -2.013720e-03,
            -2.196174e-03, -2.378810e-03]],
            [[ 8.993230e-04,  6.808847e-04,  2.595528e-03,  3.586462e-03,
            3.023965e-03,  2.479527e-03,  1.868662e-03,  1.565682e-03,
            1.563900e-03,  1.666364e-03,  1.715061e-03,  1.609638e-03,
            1.294764e-03,  8.647116e-04,  5.122397e-04,  2.899101e-04,
            3.817413e-04,  8.303743e-04,  1.253686e-03,  1.179640e-03,
            6.591807e-04,  1.167982e-04, -3.405492e-04, -5.258832e-04,
            -4.165239e-05,  1.393227e-03,  3.473584e-03,  4.953051e-03,
            4.779391e-03,  3.182305e-03,  1.140233e-03, -2.133392e-04,
            -4.233644e-04,  2.426380e-04,  1.126914e-03,  1.557022e-03,
            1.490265e-03,  1.264647e-03,  1.170405e-03,  1.237709e-03,
            1.112253e-03,  6.990263e-04,  1.700171e-04, -1.761244e-04,
            1.852706e-05,  8.140961e-04,  1.621285e-03,  1.813497e-03,
            1.394625e-03,  7.860070e-04],
            [-4.677887e-03, -3.966209e-03, -1.634288e-03, -8.592710e-04,
            -1.395248e-03, -2.189968e-03, -3.198638e-03, -3.410639e-03,
            -3.241918e-03, -3.051681e-03, -2.845973e-03, -2.786646e-03,
            -3.078280e-03, -3.367662e-03, -3.450923e-03, -3.427895e-03,
            -3.058358e-03, -2.258006e-03, -1.607386e-03, -1.647450e-03,
            -2.164357e-03, -2.647080e-03, -3.110953e-03, -3.304542e-03,
            -2.798792e-03, -1.407999e-03,  5.630683e-04,  1.961336e-03,
            1.813856e-03,  3.529640e-04, -1.526076e-03, -2.695498e-03,
            -2.702039e-03, -1.889018e-03, -9.337939e-04, -3.885011e-04,
            -2.970786e-04, -4.415356e-04, -5.492531e-04, -5.430978e-04,
            -7.051138e-04, -1.102020e-03, -1.577104e-03, -1.846151e-03,
            -1.623901e-03, -8.853760e-04, -1.772702e-04, -4.866864e-05,
            -4.633263e-04, -1.017192e-03]]]),
        "24.0": torch.tensor(
            [
                [
                    [0.0004, 0.0008, 0.0024, 0.0024, 0.0013, 0.0013, 0.0013, 0.0009, 0.0005, 0.0005, 0.0006, 0.0005, 0.0005, 0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0003, 0.0004, 0.0003, 0.0004, 0.0008, 0.0012, 0.0015, 0.0018, 0.0021, 0.0022, 0.0021, 0.0019, 0.0016, 0.0014, 0.0012, 0.0011, 0.0012, 0.0012, 0.0012, 0.0012, 0.0013, 0.0014, 0.0014, 0.0013, 0.0011, 0.0009, 0.0009, 0.0010, 0.0010, 0.0010, 0.0009, 0.0007],
                    [ -0.0055, -0.0040, -0.0024, -0.0026, -0.0031, -0.0031, -0.0036, -0.0039, -0.0039, -0.0035, -0.0031, -0.0029, -0.0028, -0.0027, -0.0026, -0.0024, -0.0023, -0.0020, -0.0017, -0.0016, -0.0017, -0.0017, -0.0015, -0.0012, -0.0010, -0.0008, -0.0006, -0.0004, -0.0004, -0.0005, -0.0006, -0.0007, -0.0006, -0.0004, -0.0002, -0.0001, 0.0001, 0.0002, 0.0003, 0.0004, 0.0004, 0.0003, 0.0001, 0.0001, 0.0000, 0.0001, 0.0000, -0.0001, -0.0002, -0.0004],
                ],
                [
                    [-0.0024, -0.0029, -0.0009, 0.0002, -0.0002, -0.0007, -0.0012, -0.0013, -0.0012, -0.0011, -0.0011, -0.0012, -0.0016, -0.0021, -0.0024, -0.0026, -0.0024, -0.0018, -0.0013, -0.0015, -0.0022, -0.0029, -0.0035, -0.0038, -0.0031, -0.0015, 0.0008, 0.0025, 0.0023, 0.0006, -0.0016, -0.0030, -0.0032, -0.0024, -0.0015, -0.0010, -0.0009, -0.0011, -0.0010, -0.0009, -0.0010, -0.0014, -0.0020, -0.0023, -0.0020, -0.0011, -0.0001, 0.0001, -0.0003, -0.0009],
                    [-0.0086, -0.0081, -0.0059, -0.0050, -0.0053, -0.0061, -0.0071, -0.0071, -0.0069, -0.0067, -0.0066, -0.0066, -0.0070, -0.0073, -0.0074, -0.0073, -0.0069, -0.0060, -0.0053, -0.0055, -0.0061, -0.0067, -0.0072, -0.0074, -0.0067, -0.0052, -0.0031, -0.0015, -0.0016, -0.0029, -0.0048, -0.0059, -0.0059, -0.0051, -0.0041, -0.0036, -0.0034, -0.0034, -0.0034, -0.0033, -0.0035, -0.0039, -0.0043, -0.0046, -0.0043, -0.0035, -0.0027, -0.0025, -0.0029, -0.0034],
                ],
            ]
        )
    }
}
# ---- error over whole batch
EXPECTED_CODEC_ERROR_BATCH = {
    "facebook/encodec_24khz": {
        "1.5": 0.0011174238752573729,
        "3.0": 0.0009308119188062847,
        "6.0": 0.0008,
        "12.0": 0.0006830253987573087,
        "24.0": 0.000642190920189023,
    },
    "facebook/encodec_48khz": {
        "3.0": 0.00039895583176985383,
        "6.0": 0.0003249854489695281,
        "12.0": 0.0002540576097089797,
        "24.0": 0.00021899679268244654,
    }
}
# fmt: on


@slow
@require_torch
class EncodecIntegrationTest(unittest.TestCase):
    @parameterized.expand(
        [
            (f"{os.path.basename(model_id)}_{bandwidth.replace('.', 'p')}", model_id, bandwidth)
            for model_id, v in EXPECTED_ENCODER_CODES.items()
            for bandwidth in v
        ]
    )
    def test_integration(self, name, model_id, bandwidth):
        # load model
        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        # load audio
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_array = librispeech_dummy[0]["audio"]["array"]
        if model.config.audio_channels > 1:
            audio_array = np.array([audio_array] * model.config.audio_channels)
        inputs = processor(
            raw_audio=audio_array,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        model = model.eval()
        with torch.no_grad():
            # Compare encoder outputs with expected values
            encoded_frames = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth))
            codes = torch.cat([encoded[0] for encoded in encoded_frames["audio_codes"]], dim=-1).unsqueeze(0)
            torch.testing.assert_close(
                codes[..., : EXPECTED_ENCODER_CODES[model_id][bandwidth].shape[-1]],
                EXPECTED_ENCODER_CODES[model_id][bandwidth].to(torch_device),
                rtol=1e-4,
                atol=1e-4,
            )
            if EXPECTED_ENCODER_SCALES[model_id][bandwidth] is not None:
                scales = torch.tensor([encoded[0].squeeze() for encoded in encoded_frames["audio_scales"]])
                torch.testing.assert_close(scales, EXPECTED_ENCODER_SCALES[model_id][bandwidth], rtol=1e-4, atol=1e-4)

            # Compare decoder outputs with expected values
            decoded_frames = model.decode(
                encoded_frames["audio_codes"],
                encoded_frames["audio_scales"],
                inputs["padding_mask"],
                last_frame_pad_length=encoded_frames["last_frame_pad_length"],
            )
            torch.testing.assert_close(
                decoded_frames["audio_values"][0][..., : EXPECTED_DECODER_OUTPUTS[model_id][bandwidth].shape[-1]],
                EXPECTED_DECODER_OUTPUTS[model_id][bandwidth].to(torch_device),
                rtol=1e-4,
                atol=1e-4,
            )

            # Compare codec error with expected values
            codec_error = compute_rmse(decoded_frames["audio_values"], inputs["input_values"])
            torch.testing.assert_close(codec_error, EXPECTED_CODEC_ERROR[model_id][bandwidth], rtol=1e-4, atol=1e-4)

            # make sure forward and enc-dec give same result
            full_enc = model(inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth))
            torch.testing.assert_close(
                full_enc["audio_values"],
                decoded_frames["audio_values"],
                rtol=1e-4,
                atol=1e-4,
            )

    @parameterized.expand(
        [
            (f"{os.path.basename(model_id)}_{bandwidth.replace('.', 'p')}", model_id, bandwidth)
            for model_id, v in EXPECTED_ENCODER_CODES_BATCH.items()
            for bandwidth in v
        ]
    )
    def test_batch(self, name, model_id, bandwidth):
        # load model
        model = EncodecModel.from_pretrained(model_id).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_id)

        # load audio
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        n_channels = model.config.audio_channels
        if n_channels == 1:
            audio_samples = [audio_sample["array"] for audio_sample in librispeech_dummy[-2:]["audio"]]
        else:
            audio_samples = []
            for _sample in librispeech_dummy[-2:]["audio"]:
                # concatenate mono channels to target number of channels
                audio_array = np.concatenate([_sample["array"][np.newaxis]] * n_channels, axis=0)
                audio_samples.append(audio_array)
        inputs = processor(
            raw_audio=audio_samples,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # apply model
        model = model.eval()
        with torch.no_grad():
            # Compare encoder outputs with expected values
            encoded_frames = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth))
            codes = encoded_frames["audio_codes"].permute(1, 2, 0, 3)
            codes = codes.reshape(codes.size(0), codes.size(1), -1)
            torch.testing.assert_close(
                codes[..., : EXPECTED_ENCODER_CODES_BATCH[model_id][bandwidth].shape[-1]],
                EXPECTED_ENCODER_CODES_BATCH[model_id][bandwidth].to(torch_device),
                rtol=1e-4,
                atol=1e-4,
            )
            if EXPECTED_ENCODER_SCALES_BATCH[model_id][bandwidth] is not None:
                scales = torch.stack(encoded_frames["audio_scales"])
                torch.testing.assert_close(
                    scales, EXPECTED_ENCODER_SCALES_BATCH[model_id][bandwidth].to(torch_device), rtol=1e-4, atol=1e-4
                )

            # Compare decoder outputs with expected values
            decoded_frames = model.decode(
                encoded_frames["audio_codes"],
                encoded_frames["audio_scales"],
                inputs["padding_mask"],
                last_frame_pad_length=encoded_frames["last_frame_pad_length"],
            )
            torch.testing.assert_close(
                decoded_frames["audio_values"][..., : EXPECTED_DECODER_OUTPUTS_BATCH[model_id][bandwidth].shape[-1]],
                EXPECTED_DECODER_OUTPUTS_BATCH[model_id][bandwidth].to(torch_device),
                rtol=1e-4,
                atol=1e-4,
            )

            # Compare codec error with expected values
            codec_error = compute_rmse(decoded_frames["audio_values"], inputs["input_values"])
            torch.testing.assert_close(
                codec_error, EXPECTED_CODEC_ERROR_BATCH[model_id][bandwidth], rtol=1e-4, atol=1e-4
            )

            # make sure forward and enc-dec give same result
            input_values_dec = model(inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth))
            torch.testing.assert_close(
                input_values_dec["audio_values"], decoded_frames["audio_values"], rtol=1e-4, atol=1e-4
            )
