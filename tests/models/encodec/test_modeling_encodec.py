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
        "3.0": torch.tensor([[[  62,  835,  835,  835,  835,  835,  835,  835,  408,  408],
            [1007, 1007, 1007,  544,  424,  424, 1007,  424,  302,  424],
            [ 786,  678,  821,  786,   36,   36,  786,  212,  937,  937],
            [ 741,  741,  741,  993,  741, 1018,  741,  919,  741,  741]]]),
        "6.0": torch.tensor([[[  62,  835,  835,  835,  835,  835,  835,  835,  408,  408],
            [1007, 1007, 1007,  544,  424,  424, 1007,  424,  302,  424],
            [ 786,  678,  821,  786,   36,   36,  786,  212,  937,  937],
            [ 741,  741,  741,  993,  741, 1018,  741,  919,  741,  741],
            [ 528,  446,  198,  190,  446,  622,  446,  448,  646,  448],
            [1011,  140,  185,  986,  683,  986,  334,   41,  140,  939],
            [ 896,  772,  562,  772,  485,  528,  570,  853,  562,  772],
            [ 899,  975,  468,  468,  468,  701,  975,  828,  518,  899]]]),
        "12.0": torch.tensor([[[  62,  835,  835,  835,  835,  835,  835,  835,  408,  408],
            [1007, 1007, 1007,  544,  424,  424, 1007,  424,  302,  424],
            [ 786,  678,  821,  786,   36,   36,  786,  212,  937,  937],
            [ 741,  741,  741,  993,  741, 1018,  741,  919,  741,  741],
            [ 528,  446,  198,  190,  446,  622,  446,  448,  646,  448],
            [1011,  140,  185,  986,  683,  986,  334,   41,  140,  939],
            [ 896,  772,  562,  772,  485,  528,  570,  853,  562,  772],
            [ 899,  975,  468,  468,  468,  701,  975,  828,  518,  899],
            [ 827,  807,  938,  320,  699,  470,  766,  628,  811,  827],
            [ 963,  801,  630,  477,  717,  354,  205,  359,  703,  744],
            [1000, 1000,  388, 1000,  408,  740,  875,  364,  875,  843],
            [ 413,  835,  382,  840,  742, 1019,  742,  962,  835,  742],
            [ 971,  410,  998,  485,  798,  410,  410,  485,  828,  920],
            [ 848,  694,  662,  784,  848,  427,  696,  848,  495,  784],
            [ 420,  911,  889,  911,  993,  776,  458,  477,  911,  911],
            [ 587,  755,  834,  962,  860,  425,  732,  982,  587,  962]]]),
        "24.0": torch.tensor([[[  62,  835,  835,  835,  835,  835,  835,  835,  408,  408],
            [1007, 1007, 1007,  544,  424,  424, 1007,  424,  302,  424],
            [ 786,  678,  821,  786,   36,   36,  786,  212,  937,  937],
            [ 741,  741,  741,  993,  741, 1018,  741,  919,  741,  741],
            [ 528,  446,  198,  190,  446,  622,  446,  448,  646,  448],
            [1011,  140,  185,  986,  683,  986,  334,   41,  140,  939],
            [ 896,  772,  562,  772,  485,  528,  570,  853,  562,  772],
            [ 899,  975,  468,  468,  468,  701,  975,  828,  518,  899],
            [ 827,  807,  938,  320,  699,  470,  766,  628,  811,  827],
            [ 963,  801,  630,  477,  717,  354,  205,  359,  703,  744],
            [1000, 1000,  388, 1000,  408,  740,  875,  364,  875,  843],
            [ 413,  835,  382,  840,  742, 1019,  742,  962,  835,  742],
            [ 971,  410,  998,  485,  798,  410,  410,  485,  828,  920],
            [ 848,  694,  662,  784,  848,  427,  696,  848,  495,  784],
            [ 420,  911,  889,  911,  993,  776,  458,  477,  911,  911],
            [ 587,  755,  834,  962,  860,  425,  732,  982,  587,  962],
            [ 270,  160,   26,  131,  597,  506,  506,  637,  248,  160],
            [  15,  215,  134,   69,  215,  155,  215, 1009,  447,  417],
            [ 580,  561,  686,  896,  497,  637,  896,  245,  896,  264],
            [ 511,  239,  560,  691,  571,  627,  691,  571,  879,  879],
            [ 591,  857,  591,  251,  250,  250,  632,  477,  486,  295],
            [ 565,  431,  654,  301,  301,  623,  623,  282,  549,  565],
            [ 539,  317,  639,  539,  651,  539,  538,  640,  539,  646],
            [ 637,  874,  637,  582,  640,  515,  507,  632,  613,  613],
            [ 601,  905,  500,  550,  522,  500,  636,  647,  624,  561],
            [ 603,  683,  584,  566,  505,  782,  687,  671,  505,  661],
            [ 577,  687,  637,  647,  552,  552,  646,  647,  689,  647],
            [ 256,  293,  931,  606,  538, 1015,  538,  294,  538,  570],
            [ 620, 1020,  666,  619,  655,  979,  946,  535,  717,  781],
            [ 387,  728,  557,  652,  511,  910,  634,  315,  623,  634],
            [ 659,  696,  947,  500,  610,  752,  659,  701,  474,  610],
            [ 567,  684,  657,  467,  485,  633,  611,  693,  946,  282]]]),
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
            [  10,  625,  712,  254,  712,  259,  394,  131,  726,  516],
            [ 882, 1022,   32,  882,  267,  861,  974,  456,  108,  521],
            [ 304,  841,  306,  354,   69,  376,  928,  510,  381,  104],
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
        "3.0": torch.tensor([5.364821e-02, 8.147693e-02, 6.264076e-02, 6.684893e-02, 5.461470e-02,
    4.484973e-02, 1.000000e-08]),
        "6.0": torch.tensor([5.364821e-02, 8.147693e-02, 6.264076e-02, 6.684893e-02, 5.461470e-02,
    4.484973e-02, 1.000000e-08]),
        "12.0": torch.tensor([5.364821e-02, 8.147693e-02, 6.264076e-02, 6.684893e-02, 5.461470e-02,
    4.484973e-02, 1.000000e-08]),
        "24.0": torch.tensor([5.364821e-02, 8.147693e-02, 6.264076e-02, 6.684893e-02, 5.461470e-02,
    4.484973e-02, 1.000000e-08])
    }
}
EXPECTED_DECODER_OUTPUTS = {
    "facebook/encodec_24khz": {
        "1.5": torch.tensor([[ 3.1715e-04, -2.1567e-04, -3.4972e-05, -4.3113e-04,  3.8687e-04,
            3.1857e-04, -1.2257e-05,  5.2728e-05,  5.3636e-04,  1.0837e-04,
            -1.5145e-03, -7.3853e-04, -1.8533e-04, -1.7607e-03, -3.0082e-04,
            1.2471e-03,  1.0653e-03,  8.0556e-04,  7.6602e-04,  7.8297e-04,
            7.8308e-04,  1.8988e-04, -3.0641e-04, -4.0059e-04, -5.5953e-04,
            -8.6804e-04, -1.0133e-03, -1.2140e-03, -1.0755e-03, -6.1971e-04,
            -5.8099e-04, -4.6168e-04, -1.2555e-06,  1.3419e-04,  3.0151e-04,
            2.0594e-04, -7.0424e-05, -2.1520e-04, -7.6370e-04, -1.2239e-03,
            -1.0429e-03, -1.2351e-03, -1.3447e-03, -3.1351e-04,  1.6296e-04,
            5.5809e-04,  5.7635e-04,  6.1265e-04,  9.1805e-04,  1.0196e-03]]),
        "3.0": torch.tensor([[ 3.6024e-04, -1.7961e-04, -4.6106e-07, -4.0335e-04,  4.0227e-04,
            3.1581e-04, -1.1605e-05,  6.2232e-05,  5.5693e-04,  1.5903e-04,
            -1.5058e-03, -7.6550e-04, -1.9891e-04, -1.8228e-03, -3.1951e-04,
            1.3154e-03,  1.1226e-03,  8.5452e-04,  8.0753e-04,  8.0904e-04,
            8.0481e-04,  1.9883e-04, -3.1064e-04, -3.9557e-04, -5.4534e-04,
            -8.5365e-04, -1.0148e-03, -1.2358e-03, -1.1014e-03, -6.3906e-04,
            -6.1239e-04, -4.8200e-04, -1.1463e-05,  1.2033e-04,  3.2778e-04,
            2.3195e-04, -5.9551e-05, -1.9393e-04, -7.8600e-04, -1.3011e-03,
            -1.0863e-03, -1.2658e-03, -1.3777e-03, -2.8378e-04,  2.0423e-04,
            6.0651e-04,  6.2392e-04,  6.3013e-04,  9.3731e-04,  1.0421e-03]]),
        "6.0": torch.tensor([[ 4.2074e-04, -1.2565e-04,  1.0722e-04, -3.2633e-04,  4.0398e-04,
            2.6664e-04,  2.1648e-06,  1.0934e-04,  6.4041e-04,  2.2555e-04,
            -1.2660e-03, -6.5737e-04, -2.1216e-04, -1.5314e-03, -5.0612e-05,
            1.4057e-03,  1.3726e-03,  1.1284e-03,  1.0424e-03,  1.0011e-03,
            8.7659e-04,  3.8591e-04,  1.8163e-05,  3.9260e-05,  1.2949e-05,
            -4.2129e-05, -1.1284e-04, -3.8112e-04, -4.0966e-04, -1.3800e-04,
            -1.6168e-04, -1.9312e-04,  2.5365e-04,  4.7200e-04,  8.9443e-04,
            9.6140e-04,  8.0938e-04,  7.1872e-04,  1.9855e-04, -2.9236e-04,
            -4.3676e-04, -7.4314e-04, -7.6657e-04,  5.5808e-05,  6.6191e-04,
            1.0533e-03,  1.2512e-03,  1.1840e-03,  1.3011e-03,  1.3873e-03]]),
        "12.0": torch.tensor([[ 3.2841e-04, -1.2950e-04,  1.1491e-04, -4.1131e-04,  2.7256e-04,
            1.5712e-04, -4.0078e-05,  4.2538e-05,  5.4797e-04,  1.5380e-04,
            -1.3412e-03, -6.5399e-04, -1.2879e-04, -1.4738e-03,  1.0960e-04,
            1.7911e-03,  1.7757e-03,  1.3782e-03,  1.2334e-03,  1.2549e-03,
            1.1401e-03,  5.4041e-04,  1.6667e-05, -1.7691e-05, -3.9906e-05,
            -8.3014e-05, -1.3624e-04, -3.9517e-04, -3.6955e-04, -4.9966e-05,
            -4.2282e-05, -2.2129e-05,  4.6914e-04,  6.9300e-04,  1.0839e-03,
            1.0737e-03,  8.6604e-04,  7.5363e-04,  2.0341e-04, -2.9842e-04,
            -4.1852e-04, -7.0207e-04, -6.9245e-04,  1.8132e-04,  8.7860e-04,
            1.3013e-03,  1.4980e-03,  1.4017e-03,  1.5343e-03,  1.6416e-03]]),
        "24.0": torch.tensor([[ 6.3314e-04,  1.6935e-04,  4.8219e-04, -1.8313e-05,  4.4245e-04,
            2.7923e-04,  1.4140e-04,  1.6600e-04,  7.3208e-04,  5.7477e-04,
            -9.4150e-04, -1.9938e-04,  1.4342e-04, -1.6950e-03,  1.1838e-04,
            2.2838e-03,  2.0600e-03,  1.2086e-03,  9.7601e-04,  1.1968e-03,
            1.3207e-03,  6.0015e-04, -1.0179e-04, -9.8698e-05, -6.2593e-05,
            -1.0316e-04, -2.5080e-04, -5.8962e-04, -4.8105e-04, -7.9790e-05,
            -1.2886e-04, -1.6267e-04,  2.5516e-04,  3.7714e-04,  7.3679e-04,
            6.5363e-04,  5.3034e-04,  6.3213e-04,  3.3862e-05, -5.0317e-04,
            -3.4814e-04, -5.8166e-04, -7.1848e-04,  3.0287e-04,  1.1706e-03,
            1.5412e-03,  1.6507e-03,  1.4969e-03,  1.7539e-03,  2.0165e-03]])
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([[ 0.003255,  0.002593,  0.003511,  0.003965,  0.002821,  0.002096,
            0.001996,  0.001961,  0.002057,  0.002266,  0.002059,  0.001769,
            0.001856,  0.002004,  0.001959,  0.001958,  0.002084,  0.002295,
            0.002439,  0.002221,  0.001727,  0.001540,  0.001785,  0.002110,
            0.002501,  0.003215,  0.004056,  0.004626,  0.004685,  0.004270,
            0.003513,  0.002788,  0.002362,  0.002300,  0.002414,  0.002401,
            0.002256,  0.002272,  0.002457,  0.002661,  0.002729,  0.002691,
            0.002520,  0.002346,  0.002372,  0.002597,  0.002748,  0.002620,
            0.002302,  0.002101],
            [-0.003156, -0.002705, -0.001824, -0.001764, -0.002487, -0.002942,
            -0.003004, -0.002661, -0.002201, -0.001832, -0.001872, -0.001954,
            -0.001789, -0.001489, -0.001276, -0.001129, -0.000910, -0.000480,
            -0.000186, -0.000556, -0.001308, -0.001632, -0.001554, -0.001415,
            -0.001140, -0.000536,  0.000204,  0.000739,  0.000806,  0.000445,
            -0.000258, -0.000924, -0.001230, -0.001147, -0.000954, -0.000900,
            -0.000930, -0.000821, -0.000646, -0.000516, -0.000504, -0.000552,
            -0.000687, -0.000855, -0.000867, -0.000671, -0.000595, -0.000801,
            -0.001121, -0.001288]]),
        "6.0": torch.tensor([[5.273837e-03, 5.029356e-03, 5.818919e-03, 5.877095e-03, 4.892784e-03,
            4.424623e-03, 4.366699e-03, 4.308038e-03, 4.231700e-03, 4.329772e-03,
            4.204455e-03, 3.996513e-03, 3.987412e-03, 3.972975e-03, 3.908725e-03,
            3.879502e-03, 3.881384e-03, 3.901682e-03, 3.939928e-03, 3.889010e-03,
            3.655977e-03, 3.570635e-03, 3.820161e-03, 4.169275e-03, 4.518270e-03,
            5.015073e-03, 5.585828e-03, 5.983434e-03, 6.020361e-03, 5.740213e-03,
            5.275079e-03, 4.855449e-03, 4.524189e-03, 4.358025e-03, 4.380845e-03,
            4.385425e-03, 4.310483e-03, 4.275807e-03, 4.368290e-03, 4.516815e-03,
            4.524615e-03, 4.450295e-03, 4.300606e-03, 4.195956e-03, 4.200535e-03,
            4.266551e-03, 4.339241e-03, 4.321544e-03, 4.173155e-03, 3.998725e-03],
            [1.453261e-04, 6.929478e-04, 1.376755e-03, 1.170405e-03, 5.611599e-04,
            2.139929e-04, 3.400779e-05, 2.110353e-04, 4.127354e-04, 6.001534e-04,
            6.037725e-04, 5.884786e-04, 6.209539e-04, 7.169978e-04, 8.618810e-04,
            9.602793e-04, 1.068815e-03, 1.298671e-03, 1.503815e-03, 1.386970e-03,
            1.000985e-03, 8.528330e-04, 1.007717e-03, 1.272305e-03, 1.584020e-03,
            1.958798e-03, 2.394634e-03, 2.756161e-03, 2.800428e-03, 2.557749e-03,
            2.118449e-03, 1.715322e-03, 1.465794e-03, 1.417461e-03, 1.499242e-03,
            1.543587e-03, 1.563998e-03, 1.640117e-03, 1.725809e-03, 1.785000e-03,
            1.758363e-03, 1.679130e-03, 1.540396e-03, 1.428279e-03, 1.349456e-03,
            1.350701e-03, 1.344747e-03, 1.228058e-03, 1.051439e-03, 8.998821e-04]]),
        "12.0": torch.tensor([[ 0.001343,  0.001114,  0.002068,  0.002333,  0.001609,  0.001302,
            0.001195,  0.001101,  0.001100,  0.001212,  0.001112,  0.000941,
            0.000908,  0.000859,  0.000806,  0.000821,  0.000870,  0.000996,
            0.001155,  0.001139,  0.000910,  0.000824,  0.001023,  0.001350,
            0.001782,  0.002428,  0.003155,  0.003647,  0.003675,  0.003337,
            0.002826,  0.002384,  0.002108,  0.002061,  0.002197,  0.002291,
            0.002255,  0.002242,  0.002320,  0.002416,  0.002400,  0.002305,
            0.002168,  0.002094,  0.002142,  0.002305,  0.002460,  0.002446,
            0.002266,  0.002099],
            [-0.003327, -0.002871, -0.001998, -0.001949, -0.002365, -0.002659,
            -0.002960, -0.002893, -0.002690, -0.002491, -0.002458, -0.002481,
            -0.002498, -0.002452, -0.002306, -0.002163, -0.001999, -0.001643,
            -0.001301, -0.001364, -0.001719, -0.001851, -0.001729, -0.001471,
            -0.001066, -0.000507,  0.000105,  0.000538,  0.000563,  0.000252,
            -0.000264, -0.000696, -0.000883, -0.000840, -0.000685, -0.000554,
            -0.000500, -0.000460, -0.000416, -0.000402, -0.000497, -0.000626,
            -0.000770, -0.000871, -0.000864, -0.000743, -0.000676, -0.000764,
            -0.000963, -0.001103]]),
        "24.0": torch.tensor([[ 1.010626e-03,  7.813113e-04,  1.791859e-03,  2.098512e-03,
            1.443315e-03,  1.131619e-03,  9.572321e-04,  7.757131e-04,
            5.984887e-04,  5.843973e-04,  4.991977e-04,  3.642855e-04,
            3.081888e-04,  2.148986e-04,  1.306767e-04,  9.979090e-05,
            1.165471e-04,  1.778778e-04,  2.110642e-04,  1.012480e-04,
            -1.902406e-04, -3.529328e-04, -2.062108e-04,  1.239474e-04,
            5.703444e-04,  1.169234e-03,  1.804531e-03,  2.212374e-03,
            2.161223e-03,  1.734669e-03,  1.171784e-03,  7.166255e-04,
            4.258847e-04,  3.626751e-04,  5.071349e-04,  6.565793e-04,
            7.122927e-04,  7.401685e-04,  8.060430e-04,  8.942910e-04,
            8.606060e-04,  7.286463e-04,  5.690599e-04,  4.778978e-04,
            5.111226e-04,  6.298538e-04,  7.323848e-04,  7.390758e-04,
            6.107587e-04,  4.459959e-04],
            [-3.571097e-03, -3.193419e-03, -2.389309e-03, -2.272637e-03,
            -2.525185e-03, -2.782251e-03, -3.181019e-03, -3.243267e-03,
            -3.204876e-03, -3.106045e-03, -3.054639e-03, -3.002484e-03,
            -2.944369e-03, -2.860032e-03, -2.721580e-03, -2.604716e-03,
            -2.478817e-03, -2.216445e-03, -1.960302e-03, -1.987081e-03,
            -2.266678e-03, -2.383426e-03, -2.242301e-03, -1.928755e-03,
            -1.491635e-03, -9.579324e-04, -4.238072e-04, -4.774341e-05,
            -1.462011e-05, -2.922945e-04, -7.516183e-04, -1.156422e-03,
            -1.393303e-03, -1.409855e-03, -1.269845e-03, -1.115955e-03,
            -1.014855e-03, -9.461480e-04, -8.898404e-04, -8.677132e-04,
            -9.717353e-04, -1.130508e-03, -1.274382e-03, -1.349356e-03,
            -1.346857e-03, -1.286301e-03, -1.257888e-03, -1.309370e-03,
            -1.460626e-03, -1.610748e-03]])
    }
}
EXPECTED_CODEC_ERROR = {
    "facebook/encodec_24khz": {
        "1.5": 0.0022192629985511303,
        "3.0": 0.001855933922342956,
        "6.0": 0.0014942693524062634,
        "12.0": 0.0012968705268576741,
        "24.0": 0.0012268663849681616,
    },
    "facebook/encodec_48khz": {
        "3.0": 0.00084255903493613,
        "6.0": 0.0006662720115855336,
        "12.0": 0.0005296851741150022,
        "24.0": 0.0004447767569217831,
    }
}
# -- test_batch
EXPECTED_ENCODER_CODES_BATCH = {
    "facebook/encodec_24khz": {
        "1.5": torch.tensor([[[  62,  106,  475,  475,  404,  404,  475,  404,  404,  475],
            [ 424,  969,  913, 1007,  544, 1007, 1007, 1007,  969, 1007]],

            [[ 835,  835,  835,  835,  835,  835,  835,  835,  835,  835],
            [ 857,  857,  544,  518,  937,  518,  913,  913,  518,  913]]]),
        "3.0": torch.tensor([[[  62,  106,  475,  475,  404,  404,  475,  404,  404,  475],
            [ 424,  969,  913, 1007,  544, 1007, 1007, 1007,  969, 1007],
            [ 212,  832,  212,   36,   36,   36,  767,  653,  982, 1016],
            [ 956,  741,  838, 1019,  739,  780,  838, 1019, 1014, 1019]],

            [[ 835,  835,  835,  835,  835,  835,  835,  835,  835,  835],
            [ 857,  857,  544,  518,  937,  518,  913,  913,  518,  913],
            [ 705,  989,  934,  989,  678,  934,  934,  786,  934,  786],
            [ 366, 1018,  398,  398,  398,  398,  673,  741,  398,  741]]]),
        "6.0": torch.tensor([[[  62,  106,  475,  475,  404,  404,  475,  404,  404,  475],
            [ 424,  969,  913, 1007,  544, 1007, 1007, 1007,  969, 1007],
            [ 212,  832,  212,   36,   36,   36,  767,  653,  982, 1016],
            [ 956,  741,  838, 1019,  739,  780,  838, 1019, 1014, 1019],
            [ 712,  862,  712,  448,  528,  646,  446,  373,  694,  373],
            [ 939,  881,  939,   19,  334,  881, 1005,  763,  632,  781],
            [ 853,  464,  772,  782,  782,  983,  890,  874,  983,  782],
            [ 899,  475,  173,  701,  701,  947,  468, 1019,  882,  518]],

            [[ 835,  835,  835,  835,  835,  835,  835,  835,  835,  835],
            [ 857,  857,  544,  518,  937,  518,  913,  913,  518,  913],
            [ 705,  989,  934,  989,  678,  934,  934,  786,  934,  786],
            [ 366, 1018,  398,  398,  398,  398,  673,  741,  398,  741],
            [ 373,  373,  375,  373,  373,  222,  862,  373,  190,  373],
            [ 293,  949,  435,  435,  435,  293,  949,  881,  632,  986],
            [ 800,  528,  528,  853,  782,  485,  772,  900,  528,  853],
            [ 916,  237,  828,  701,  518,  835,  948,  315,  948,  315]]]),
        "12.0": torch.tensor([[[  62,  106,  475,  475,  404,  404,  475,  404,  404,  475],
            [ 424,  969,  913, 1007,  544, 1007, 1007, 1007,  969, 1007],
            [ 212,  832,  212,   36,   36,   36,  767,  653,  982, 1016],
            [ 956,  741,  838, 1019,  739,  780,  838, 1019, 1014, 1019],
            [ 712,  862,  712,  448,  528,  646,  446,  373,  694,  373],
            [ 939,  881,  939,   19,  334,  881, 1005,  763,  632,  781],
            [ 853,  464,  772,  782,  782,  983,  890,  874,  983,  782],
            [ 899,  475,  173,  701,  701,  947,  468, 1019,  882,  518],
            [ 817,  470,  588,  675,  675,  588,  960,  927,  909,  466],
            [ 953,  776,  717,  630,  359,  717,  861,  630,  861,  359],
            [ 623,  740, 1000,  388,  420,  388,  740,  818,  958,  743],
            [ 413,  835,  742,  249,  890,  352, 1006,  498,  866,  890],
            [ 817,  351,  804,  751,  192,  535,  552,  879,  351,  971],
            [ 792,  495,  935,  848,  792,  795,  989,  935,  723,  531],
            [ 622,  681,  477,  713,  477,  871,  713,  514,  993,  777],
            [ 928,  799,  732, 1005,  928,  439,  732,  922,  982,  922]],

            [[ 835,  835,  835,  835,  835,  835,  835,  835,  835,  835],
            [ 857,  857,  544,  518,  937,  518,  913,  913,  518,  913],
            [ 705,  989,  934,  989,  678,  934,  934,  786,  934,  786],
            [ 366, 1018,  398,  398,  398,  398,  673,  741,  398,  741],
            [ 373,  373,  375,  373,  373,  222,  862,  373,  190,  373],
            [ 293,  949,  435,  435,  435,  293,  949,  881,  632,  986],
            [ 800,  528,  528,  853,  782,  485,  772,  900,  528,  853],
            [ 916,  237,  828,  701,  518,  835,  948,  315,  948,  315],
            [ 420,  628,  918,  628,  628,  628,  248,  628,  909,  811],
            [ 736,  717,  994,  974,  477,  874,  963,  979,  355,  205],
            [1002, 1002,  894,  875,  388,  709,  534,  408,  881,  709],
            [ 735,  828,  763,  742,  640,  835,  828,  375,  840,  375],
            [ 898,  938,  556,  658,  410,  951,  486,  658,  877,  877],
            [   0,  797,  428,  669,  428,  920, 1022, 1022,  809,  450],
            [ 622,  421,  422,  911,  911,  911,  958,  421,  776,  421],
            [1005,  312,  922,  755,  834,  461,  461,  702,  597,  907]]]),
        "24.0": torch.tensor([[[  62,  106,  475,  475,  404,  404,  475,  404,  404,  475],
            [ 424,  969,  913, 1007,  544, 1007, 1007, 1007,  969, 1007],
            [ 212,  832,  212,   36,   36,   36,  767,  653,  982, 1016],
            [ 956,  741,  838, 1019,  739,  780,  838, 1019, 1014, 1019],
            [ 712,  862,  712,  448,  528,  646,  446,  373,  694,  373],
            [ 939,  881,  939,   19,  334,  881, 1005,  763,  632,  781],
            [ 853,  464,  772,  782,  782,  983,  890,  874,  983,  782],
            [ 899,  475,  173,  701,  701,  947,  468, 1019,  882,  518],
            [ 817,  470,  588,  675,  675,  588,  960,  927,  909,  466],
            [ 953,  776,  717,  630,  359,  717,  861,  630,  861,  359],
            [ 623,  740, 1000,  388,  420,  388,  740,  818,  958,  743],
            [ 413,  835,  742,  249,  890,  352, 1006,  498,  866,  890],
            [ 817,  351,  804,  751,  192,  535,  552,  879,  351,  971],
            [ 792,  495,  935,  848,  792,  795,  989,  935,  723,  531],
            [ 622,  681,  477,  713,  477,  871,  713,  514,  993,  777],
            [ 928,  799,  732, 1005,  928,  439,  732,  922,  982,  922],
            [ 939,  637,  861,  506,  861,   61,  993,  264, 1019,  260],
            [ 166,  215,   69,   69,  215,   69, 1016,  828,  396,  180],
            [ 561,  896,  686,  144,  538,  659,  216,  514,  686,  451],
            [ 691,  691,  627,  735,  674,  287,  277,  972,  550,  505],
            [ 451,  811,  596,  251,  473,  841,  567,  329,  551,  846],
            [ 313,  601,  654,  763, 1019,  565,  599,  441,  601,  480],
            [ 653,  242,  292,  572,  685,  973,  623,  374,  561,  521],
            [ 984,  987,  618,  454,  888,  507,  741,  636,  515,  671],
            [ 647,  550,  356,  292,  513, 1011,  208,  549,  691,  911],
            [ 683,  536,  431,  603,  508,  867,  996,  857,  886,  491],
            [ 444,  937,  674,  555,  954,  710,  899,  852,  655,  591],
            [ 658,  952,  835,  508,  616,  596,  321,  721,  464,  306],
            [ 665,  334,  691,  532,  651,  278,  874,  838,  517,  597],
            [ 613,  674, 1000,  904,  716,  977,  995,  739,  672,  776],
            [ 689,  386,  841,  658,  386,  869,  938,  806,  750,  659],
            [ 652,  509,  647,  826,  704,  622,  221,  477,  900,  895]],

            [[ 835,  835,  835,  835,  835,  835,  835,  835,  835,  835],
            [ 857,  857,  544,  518,  937,  518,  913,  913,  518,  913],
            [ 705,  989,  934,  989,  678,  934,  934,  786,  934,  786],
            [ 366, 1018,  398,  398,  398,  398,  673,  741,  398,  741],
            [ 373,  373,  375,  373,  373,  222,  862,  373,  190,  373],
            [ 293,  949,  435,  435,  435,  293,  949,  881,  632,  986],
            [ 800,  528,  528,  853,  782,  485,  772,  900,  528,  853],
            [ 916,  237,  828,  701,  518,  835,  948,  315,  948,  315],
            [ 420,  628,  918,  628,  628,  628,  248,  628,  909,  811],
            [ 736,  717,  994,  974,  477,  874,  963,  979,  355,  205],
            [1002, 1002,  894,  875,  388,  709,  534,  408,  881,  709],
            [ 735,  828,  763,  742,  640,  835,  828,  375,  840,  375],
            [ 898,  938,  556,  658,  410,  951,  486,  658,  877,  877],
            [   0,  797,  428,  669,  428,  920, 1022, 1022,  809,  450],
            [ 622,  421,  422,  911,  911,  911,  958,  421,  776,  421],
            [1005,  312,  922,  755,  834,  461,  461,  702,  597,  907],
            [ 248,  248,   48,  546,  977,  506,  546,  270,  670,  670],
            [ 547,  447,   10,  160, 1009,  215,  134,  396,  260,   15],
            [ 635,  497,  580,  497,  245,  497,  244,  675,  624,  656],
            [ 864,  571,  616,  482,  588,  781,  525,  258,  674,  503],
            [ 449,  757,  857,  451,  519,  486,  299,  299,  251,  596],
            [ 809,  255,  255,  255,  639,  301,  639,  546,  617,  639],
            [ 551,  497,  908,  640,  661,  710,  640,  539,  646,  317],
            [ 689,  507,  254,  662,  522,  637,  527,  515,  662,  507],
            [ 983,  686,  500,  927,  653,  561,  768,  653,  891,  688],
            [ 493,  566,  664,  782,  603,  683,  497,  603,  270,  721],
            [ 978,  552,  982,  766,  607,  646,  687, 1018,  764,  620],
            [ 330,  293,  711,  658,  293,  294,  608,  606,  658,  627],
            [ 954,  994,  960,  951,  908,  927,  535,  571,  557,  620],
            [ 768,  259,  911, 1006,  607,  959,  589,  718,  715,  566],
            [ 420,  931,  210,  791,  676,  731,  894,  676,  917,  894],
            [ 675,  960,  705,  997, 1017,  517,  676,  588,  503,  816]]])
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
            [1022, 1022,  471,  284,  821,  821,  267,  925,  925,  267],
            [ 979,  992,  914,  939,    0,    0, 1023,  963,  963, 1023]]]),
        "12.0": torch.tensor([[[ 790,  790,  790,  214,  214,  214,  799,  214,  214,  214],
            [ 989,  989,   77,  546,  989,  546,  989,  160,  546,  989],
            [ 977,  977,  977,  977,  538,  977,  977,  960,  977,  977],
            [ 376,  376,  962,  962,  607,  962,  963,  896,  962,  376],
            [ 979,  979,  979, 1012,  979, 1012,  921,    0,  979,  695],
            [ 824, 1018,  762,  957,  824,  762,  762, 1007,  957,  336],
            [ 681,  973,  973,  452,  211,  681,  973,  679,  904,  884],
            [ 950, 1017, 1016, 1017, 1017, 1017,  229,  607,  229,  689]],

            [[ 214,  214,  214,  214,  214,  214,  214,  214,  214,  214],
            [ 289,  289,  989,  764,  289,  289,  882,  882,  882,  882],
            [1022, 1022,  471,  284,  821,  821,  267,  925,  925,  267],
            [ 979,  992,  914,  939,    0,    0, 1023,  963,  963, 1023],
            [ 403,  940,  976,  136,  677, 1002,  979,  677,  677,  677],
            [1018,  794,  762, 1018,  485,  485,  974,  548,  548,  282],
            [ 679,  243,  679,  788, 1005, 1005, 1014, 1005, 1005, 1014],
            [ 810,   13, 1017,  810,  522,  702,  202, 1017, 1017,  694]]]),
        "24.0": torch.tensor([[[ 790,  790,  790,  214,  214,  214,  799,  214,  214,  214],
            [ 989,  989,   77,  546,  989,  546,  989,  160,  546,  989],
            [ 977,  977,  977,  977,  538,  977,  977,  960,  977,  977],
            [ 376,  376,  962,  962,  607,  962,  963,  896,  962,  376],
            [ 979,  979,  979, 1012,  979, 1012,  921,    0,  979,  695],
            [ 824, 1018,  762,  957,  824,  762,  762, 1007,  957,  336],
            [ 681,  973,  973,  452,  211,  681,  973,  679,  904,  884],
            [ 950, 1017, 1016, 1017, 1017, 1017,  229,  607,  229,  689],
            [1023, 1011,  669, 1023,  996, 1023, 1011,  297,  902,  970],
            [ 982,  681,  982,  629,  662,  982,  878,  476,  629,  982],
            [ 727,  727,  959,  959,  987,  959,  530,  959,  732,  961],
            [ 886,  456,  924,  486,  388,  959,  920,  924,  388,  924],
            [ 516, 1005,  712,  993,  949,  131,   56,  886,  712,  405],
            [1022,  460, 1022, 1022, 1022, 1022,  882,  309,  864,   32],
            [  10,   65,  169,  164,    0,    0,    0,  516,  257,  452],
            [ 307,  761,  175,  428,    0, 1020,  352,  627,  484,  262]],

            [[ 214,  214,  214,  214,  214,  214,  214,  214,  214,  214],
            [ 289,  289,  989,  764,  289,  289,  882,  882,  882,  882],
            [1022, 1022,  471,  284,  821,  821,  267,  925,  925,  267],
            [ 979,  992,  914,  939,    0,    0, 1023,  963,  963, 1023],
            [ 403,  940,  976,  136,  677, 1002,  979,  677,  677,  677],
            [1018,  794,  762, 1018,  485,  485,  974,  548,  548,  282],
            [ 679,  243,  679,  788, 1005, 1005, 1014, 1005, 1005, 1014],
            [ 810,   13, 1017,  810,  522,  702,  202, 1017, 1017,  694],
            [ 728,  252,  970, 1005,  971,  297,  673,  902, 1011,  996],
            [ 332, 1014,  476,    0, 1014,  878,  332,  411,  411,  205],
            [ 959,  727,  611,  165,  611,  303,  999,  497,  821,  727],
            [ 995,  698,  924,  843,  102,   30,  178,  970,  344,  831],
            [  81,  516,  847,  378,   10,  394,  712,  726,  993,  604],
            [ 467,  496,  484,  773,  456,  524,  337,  600,  456,  676],
            [ 789,   65,  937,  607,  159,  803,  333,  764,  179,  953],
            [ 975,  790,  483,  955, 1020,  848,  307,  333,   83,  649]]])
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
        "3.0": torch.tensor([[[1.027163e-01],[7.874525e-02]],
            [[1.014897e-01],[8.692046e-02]],
            [[6.306949e-02],[7.737400e-02]],
            [[6.879590e-02],[1.045877e-01]],
            [[6.438798e-02],[8.843125e-02]],
            [[4.138255e-02],[1.000000e-08]],
            [[5.843059e-02],[1.000000e-08]],
            [[2.328752e-04],[1.000000e-08]],
            [[1.000000e-08],[1.000000e-08]]]),
        "6.0": torch.tensor([[[1.027163e-01],[7.874525e-02]],
            [[1.014897e-01],[8.692046e-02]],
            [[6.306949e-02],[7.737400e-02]],
            [[6.879590e-02],[1.045877e-01]],
            [[6.438798e-02],[8.843125e-02]],
            [[4.138255e-02],[1.000000e-08]],
            [[5.843059e-02],[1.000000e-08]],
            [[2.328752e-04],[1.000000e-08]],
            [[1.000000e-08],[1.000000e-08]]]),
        "12.0": torch.tensor([[[1.027163e-01],[7.874525e-02]],
            [[1.014897e-01],[8.692046e-02]],
            [[6.306949e-02],[7.737400e-02]],
            [[6.879590e-02],[1.045877e-01]],
            [[6.438798e-02],[8.843125e-02]],
            [[4.138255e-02],[1.000000e-08]],
            [[5.843059e-02],[1.000000e-08]],
            [[2.328752e-04],[1.000000e-08]],
            [[1.000000e-08],[1.000000e-08]]]),
        "24.0": torch.tensor([[[1.027163e-01],[7.874525e-02]],
            [[1.014897e-01],[8.692046e-02]],
            [[6.306949e-02],[7.737400e-02]],
            [[6.879590e-02],[1.045877e-01]],
            [[6.438798e-02],[8.843125e-02]],
            [[4.138255e-02],[1.000000e-08]],
            [[5.843059e-02],[1.000000e-08]],
            [[2.328752e-04],[1.000000e-08]],
            [[1.000000e-08],[1.000000e-08]]])
    }
}
EXPECTED_DECODER_OUTPUTS_BATCH = {
    "facebook/encodec_24khz": {
        "1.5": torch.tensor([[[ 1.013392e-03,  4.239874e-04,  5.438067e-04,  2.441398e-04,
            4.788681e-04, -4.591281e-05, -2.652992e-04, -1.301691e-04,
            3.209875e-04,  1.224446e-04, -1.422606e-03, -9.020185e-04,
            -7.137851e-04, -2.338690e-03, -8.717416e-04,  7.857495e-04,
            7.357287e-04,  3.073364e-04,  1.401616e-04,  1.028691e-04,
            3.130747e-04, -3.795157e-05, -3.154487e-04, -3.871178e-04,
            -4.727610e-04]],
            [[-8.695186e-05, -3.661285e-05,  2.726965e-04,  8.320069e-05,
            5.073360e-04,  5.860149e-05, -6.264808e-04, -2.175142e-04,
            2.345634e-04,  2.121252e-04, -3.128372e-03, -4.368990e-04,
            5.641552e-04, -6.625581e-03, -3.175025e-03,  4.339077e-03,
            2.495302e-03, -1.864744e-03, -1.646358e-03,  9.006890e-05,
            1.901837e-03, -1.024531e-03, -1.394920e-03, -9.337442e-04,
            -6.891684e-04]]]),
        "3.0": torch.tensor([[[ 1.347399e-03,  7.201232e-04,  9.491952e-04,  4.557268e-04,
            6.211160e-04,  1.896985e-04, -4.435031e-05,  3.495693e-05,
            4.662518e-04,  3.166875e-04, -1.171891e-03, -5.976104e-04,
            -3.084179e-04, -1.955440e-03, -2.947041e-04,  1.493250e-03,
            1.333031e-03,  9.091700e-04,  7.741846e-04,  7.233927e-04,
            8.087571e-04,  3.518451e-04,  1.029574e-04, -7.226015e-06,
            -7.873680e-05]],
            [[ 2.099830e-05, -3.115222e-04,  4.737607e-04,  3.889582e-04,
            1.137543e-03,  1.321617e-03,  2.444885e-04,  4.939229e-04,
            2.091017e-04,  5.864516e-04, -2.476613e-03, -5.323743e-04,
            3.971122e-04, -6.959726e-03, -2.731542e-03,  3.839387e-03,
            1.281597e-03, -1.542277e-03, -4.809136e-04,  2.538285e-04,
            1.453182e-03, -6.334702e-04, -1.837280e-04, -1.012873e-03,
            -8.318835e-04]]]),
        "6.0": torch.tensor([[[ 9.865819e-04,  4.606829e-04,  7.131533e-04,  1.038504e-04,
            3.535512e-04,  1.169508e-06, -2.204090e-04, -1.218239e-04,
            3.323262e-04,  1.046079e-04, -1.349166e-03, -7.417354e-04,
            -3.896481e-04, -1.930452e-03, -3.437125e-04,  1.291438e-03,
            1.173518e-03,  8.286917e-04,  7.394444e-04,  7.335080e-04,
            7.550231e-04,  3.377336e-04,  9.265670e-05,  6.742500e-05,
            -3.093854e-06]],
            [[-4.863184e-04, -3.645968e-05,  3.537351e-04,  7.219508e-05,
            1.014986e-03,  1.212869e-03,  1.633720e-04,  3.999271e-04,
            1.189967e-03,  3.040747e-04, -2.280689e-03, -3.016855e-04,
            -4.569764e-04, -6.265447e-03, -2.627017e-03,  4.000908e-03,
            2.413808e-03, -1.844914e-03, -5.087608e-04,  1.617260e-03,
            4.162213e-04, -7.854021e-04,  9.556522e-04,  1.707535e-04,
            -1.516761e-03]]]),
        "12.0": torch.tensor([[[ 3.194872e-04,  1.803518e-04,  3.800326e-04, -4.404867e-04,
            -3.628117e-04, -7.472747e-04, -7.856790e-04, -6.655603e-04,
            -1.496278e-04, -1.996732e-04, -1.604301e-03, -8.457428e-04,
            -3.662954e-04, -1.999104e-03, -3.349860e-04,  1.466900e-03,
            1.484257e-03,  1.153394e-03,  1.061683e-03,  9.438619e-04,
            9.520103e-04,  4.728826e-04,  1.411439e-04,  1.391396e-04,
            2.010784e-06]],
            [[-7.489597e-04, -2.746669e-04,  3.350459e-04, -7.545383e-05,
            8.780524e-04,  1.273136e-03,  4.482581e-04,  7.704421e-04,
            1.529676e-03,  6.392814e-04, -2.118281e-03, -9.380205e-05,
            -3.059052e-04, -6.205492e-03, -2.208594e-03,  4.336085e-03,
            2.778761e-03, -1.297656e-03, -1.472477e-04,  1.728590e-03,
            9.723459e-04, -1.275589e-04,  8.126144e-04,  9.759124e-05,
            -1.029218e-03]]]),
        "24.0": torch.tensor([[[ 8.963001e-04,  4.680050e-04,  7.523689e-04,  2.133644e-04,
            4.220959e-04, -6.198639e-05, -3.262536e-04, -2.266234e-04,
            2.323251e-04,  8.989772e-05, -1.493751e-03, -9.475549e-04,
            -5.909826e-04, -2.402702e-03, -5.504806e-04,  1.536033e-03,
            1.408144e-03,  9.456231e-04,  8.457940e-04,  7.749633e-04,
            8.450603e-04,  3.983123e-04,  9.087895e-05,  4.164633e-05,
            -7.317864e-05]],
            [[-1.071962e-03, -5.034541e-04,  4.564493e-05, -3.994362e-04,
            6.052152e-04,  1.037086e-03,  1.989063e-04,  5.453927e-04,
            1.578666e-03,  5.307236e-04, -2.187066e-03,  5.432010e-05,
            -1.815173e-04, -6.337838e-03, -2.263878e-03,  4.604226e-03,
            2.813407e-03, -1.453131e-03, -3.590481e-05,  1.999856e-03,
            9.529511e-04, -2.681161e-04,  7.286667e-04, -3.204506e-06,
            -1.175861e-03]]])
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([[[ 4.910652e-03,  4.487927e-03,  5.474150e-03,  5.329455e-03,
            3.999488e-03,  3.597105e-03,  3.452409e-03,  3.185376e-03,
            2.891660e-03,  2.878466e-03,  2.609050e-03,  2.248081e-03,
            2.244160e-03,  2.293820e-03,  2.243023e-03,  2.228363e-03,
            2.223562e-03,  2.303203e-03,  2.421144e-03,  2.304706e-03,
            2.000508e-03,  1.998602e-03,  2.356016e-03,  2.645040e-03,
            2.864246e-03],
            [-3.927794e-03, -2.710658e-03, -1.336009e-03, -1.954665e-03,
            -2.990181e-03, -3.344528e-03, -3.706116e-03, -3.591381e-03,
            -3.321486e-03, -2.881436e-03, -2.700662e-03, -2.670048e-03,
            -2.535211e-03, -2.241722e-03, -1.974841e-03, -1.831735e-03,
            -1.636737e-03, -1.205729e-03, -9.191858e-04, -1.243288e-03,
            -1.850551e-03, -2.049224e-03, -1.971533e-03, -1.932652e-03,
            -1.833131e-03]],
            [[ 4.092955e-03,  3.261657e-03,  4.857938e-03,  5.871172e-03,
            5.011545e-03,  4.332938e-03,  3.948282e-03,  3.794592e-03,
            4.025058e-03,  4.381745e-03,  4.237822e-03,  3.783854e-03,
            3.537434e-03,  3.367567e-03,  3.014043e-03,  2.707134e-03,
            2.879554e-03,  3.582661e-03,  4.177222e-03,  3.887933e-03,
            3.008300e-03,  2.474976e-03,  2.324393e-03,  2.196758e-03,
            2.465154e-03],
            [-3.633212e-03, -3.065406e-03, -7.602383e-04, -6.224975e-05,
            -9.681851e-04, -1.771872e-03, -2.319312e-03, -2.141163e-03,
            -1.538356e-03, -9.446472e-04, -8.622363e-04, -1.128861e-03,
            -1.331049e-03, -1.314968e-03, -1.382660e-03, -1.511173e-03,
            -1.126274e-03, -1.081851e-04,  6.227733e-04,  2.399606e-04,
            -7.697257e-04, -1.378626e-03, -1.699449e-03, -2.044841e-03,
            -1.916049e-03]]]),
        "6.0": torch.tensor([[[ 5.278078e-03,  5.178319e-03,  6.527592e-03,  6.524404e-03,
            5.352218e-03,  5.227673e-03,  5.201800e-03,  4.819684e-03,
            4.499243e-03,  4.510947e-03,  4.369515e-03,  4.122742e-03,
            4.086687e-03,  4.084477e-03,  3.984717e-03,  3.922353e-03,
            3.900286e-03,  3.968338e-03,  4.098464e-03,  4.027730e-03,
            3.801033e-03,  3.811361e-03,  4.116764e-03,  4.462680e-03,
            4.801529e-03],
            [-2.227204e-03, -9.760207e-04,  6.298354e-04,  2.527072e-04,
            -5.946775e-04, -7.455797e-04, -1.164058e-03, -1.375168e-03,
            -1.288798e-03, -8.764245e-04, -6.270527e-04, -6.031074e-04,
            -5.788715e-04, -4.047145e-04, -2.409496e-04, -1.292536e-04,
            2.172117e-05,  3.444548e-04,  6.672975e-04,  5.716619e-04,
            1.596151e-04,  2.851594e-05,  1.768746e-04,  3.491421e-04,
            5.490248e-04]],
            [[ 9.416717e-04,  4.200584e-04,  2.254361e-03,  3.346253e-03,
            2.877208e-03,  2.499807e-03,  2.145845e-03,  2.095558e-03,
            2.327676e-03,  2.582574e-03,  2.501125e-03,  2.180276e-03,
            1.885313e-03,  1.592768e-03,  1.284752e-03,  1.084729e-03,
            1.332648e-03,  2.025678e-03,  2.562569e-03,  2.296569e-03,
            1.287939e-03,  3.246501e-04, -2.085320e-04, -2.626389e-04,
            4.613599e-04],
            [-6.250596e-03, -5.674047e-03, -3.235426e-03, -2.371760e-03,
            -2.883253e-03, -3.488926e-03, -4.171863e-03, -4.061838e-03,
            -3.607824e-03, -3.273506e-03, -3.221979e-03, -3.405106e-03,
            -3.751244e-03, -3.948259e-03, -3.978872e-03, -3.962256e-03,
            -3.468205e-03, -2.401381e-03, -1.653940e-03, -1.983325e-03,
            -3.063045e-03, -4.022270e-03, -4.668531e-03, -4.783047e-03,
            -3.972892e-03]]]),
        "12.0": torch.tensor([[[ 0.001894,  0.002112,  0.003574,  0.003577,  0.002552,  0.002490,
            0.002458,  0.002061,  0.001559,  0.001435,  0.001413,  0.001251,
            0.001084,  0.000945,  0.000889,  0.000848,  0.000719,  0.000697,
            0.000807,  0.000783,  0.000646,  0.000734,  0.001056,  0.001404,
            0.001720],
            [-0.004676, -0.003411, -0.001793, -0.001982, -0.002535, -0.002670,
            -0.003232, -0.003541, -0.003656, -0.003438, -0.003184, -0.003101,
            -0.003140, -0.003074, -0.002915, -0.002845, -0.002812, -0.002550,
            -0.002220, -0.002200, -0.002455, -0.002496, -0.002321, -0.002124,
            -0.001930]],
            [[ 0.001492,  0.001270,  0.003372,  0.004547,  0.003978,  0.003330,
            0.002642,  0.002266,  0.002267,  0.002438,  0.002546,  0.002467,
            0.002092,  0.001520,  0.001045,  0.000781,  0.000897,  0.001359,
            0.001735,  0.001660,  0.001229,  0.000770,  0.000323,  0.000107,
            0.000593],
            [-0.004268, -0.003596, -0.001144, -0.000199, -0.000673, -0.001569,
            -0.002739, -0.003079, -0.002903, -0.002609, -0.002280, -0.002121,
            -0.002430, -0.002854, -0.003052, -0.003041, -0.002616, -0.001789,
            -0.001148, -0.001156, -0.001605, -0.002012, -0.002455, -0.002695,
            -0.002188]]]),
        "24.0": torch.tensor([[[ 9.885861e-04,  1.248861e-03,  2.756448e-03,  2.822373e-03,
            1.902522e-03,  1.918571e-03,  1.906093e-03,  1.558219e-03,
            1.150225e-03,  1.113512e-03,  1.136557e-03,  1.002904e-03,
            8.083017e-04,  5.770588e-04,  4.331270e-04,  3.562384e-04,
            2.459183e-04,  1.715542e-04,  2.108640e-04,  2.620750e-04,
            1.999366e-04,  2.666551e-04,  6.229684e-04,  1.045243e-03,
            1.361705e-03],
            [-5.809102e-03, -4.616351e-03, -3.071268e-03, -3.211289e-03,
            -3.672507e-03, -3.799221e-03, -4.405306e-03, -4.681841e-03,
            -4.679744e-03, -4.372290e-03, -4.075142e-03, -3.938845e-03,
            -3.955660e-03, -3.922716e-03, -3.801933e-03, -3.718506e-03,
            -3.641769e-03, -3.387401e-03, -3.051993e-03, -2.944843e-03,
            -3.084793e-03, -3.016482e-03, -2.709389e-03, -2.360094e-03,
            -2.032678e-03]],
            [[-1.842556e-03, -2.225490e-03, -1.820538e-04,  1.039590e-03,
            5.808301e-04, -4.512940e-05, -7.407555e-04, -9.319350e-04,
            -8.306241e-04, -7.037186e-04, -5.580072e-04, -6.179014e-04,
            -1.077674e-03, -1.647270e-03, -2.029547e-03, -2.243858e-03,
            -2.021034e-03, -1.430934e-03, -9.589257e-04, -1.106581e-03,
            -1.781156e-03, -2.478808e-03, -3.164497e-03, -3.399065e-03,
            -2.717207e-03],
            [-7.580369e-03, -7.114318e-03, -4.795671e-03, -3.767334e-03,
            -4.121532e-03, -5.008015e-03, -6.179525e-03, -6.350174e-03,
            -6.095477e-03, -5.900454e-03, -5.687315e-03, -5.642931e-03,
            -6.038548e-03, -6.441440e-03, -6.575500e-03, -6.540283e-03,
            -6.044587e-03, -5.139905e-03, -4.480643e-03, -4.599496e-03,
            -5.195050e-03, -5.713775e-03, -6.295441e-03, -6.446271e-03,
            -5.684488e-03]]])
    }
}
# ---- error over whole batch
EXPECTED_CODEC_ERROR_BATCH = {
    "facebook/encodec_24khz": {
        "1.5": 0.0011287896195426583,
        "3.0": 0.0009268419235013425,
        "6.0": 0.0007567914435639977,
        "12.0": 0.0006700338562950492,
        "24.0": 0.0006302036927081645,
    },
    "facebook/encodec_48khz": {
        "3.0": 0.000394877337384969,
        "6.0": 0.0003207701665814966,
        "12.0": 0.00025246545556001365,
        "24.0": 0.0002160599105991423,
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
                rtol=1e-6,
                atol=1e-6,
            )
            if EXPECTED_ENCODER_SCALES[model_id][bandwidth] is not None:
                scales = torch.tensor([encoded[0].squeeze() for encoded in encoded_frames["audio_scales"]])
                torch.testing.assert_close(scales, EXPECTED_ENCODER_SCALES[model_id][bandwidth], rtol=1e-6, atol=1e-6)

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
                rtol=1e-6,
                atol=1e-6,
            )

            # Compare codec error with expected values
            codec_error = compute_rmse(decoded_frames["audio_values"], inputs["input_values"])
            torch.testing.assert_close(codec_error, EXPECTED_CODEC_ERROR[model_id][bandwidth], rtol=1e-6, atol=1e-6)

            # make sure forward and enc-dec give same result
            full_enc = model(inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth))
            torch.testing.assert_close(
                full_enc["audio_values"],
                decoded_frames["audio_values"],
                rtol=1e-6,
                atol=1e-6,
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
                rtol=1e-6,
                atol=1e-6,
            )
            if EXPECTED_ENCODER_SCALES_BATCH[model_id][bandwidth] is not None:
                scales = torch.stack(encoded_frames["audio_scales"])
                torch.testing.assert_close(
                    scales, EXPECTED_ENCODER_SCALES_BATCH[model_id][bandwidth].to(torch_device), rtol=1e-6, atol=1e-6
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
                rtol=1e-6,
                atol=1e-6,
            )

            # Compare codec error with expected values
            codec_error = compute_rmse(decoded_frames["audio_values"], inputs["input_values"])
            torch.testing.assert_close(
                codec_error, EXPECTED_CODEC_ERROR_BATCH[model_id][bandwidth], rtol=1e-6, atol=1e-6
            )

            # make sure forward and enc-dec give same result
            input_values_dec = model(inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth))
            torch.testing.assert_close(
                input_values_dec["audio_values"], decoded_frames["audio_values"], rtol=1e-6, atol=1e-6
            )
