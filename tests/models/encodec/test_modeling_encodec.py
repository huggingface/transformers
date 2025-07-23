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
         [ 741,  741,  741,  993,  741, 1018,  993,  919,  741,  741]]]),
        "6.0": torch.tensor([[[  62,  835,  835,  835,  835,  835,  835,  835,  408,  408],
            [1007, 1007, 1007,  544,  424,  424, 1007,  424,  302,  424],
            [ 786,  678,  821,  786,   36,   36,  786,  212,  937,  937],
            [ 741,  741,  741,  993,  741, 1018,  993,  919,  741,  741],
            [ 528,  446,  198,  190,  446,  622,  646,  448,  646,  448],
            [1011,  140,  185,  986,  683,  986,  435,   41,  140,  939],
            [ 896,  772,  562,  772,  485,  528,  896,  853,  562,  772],
            [ 899,  975,  468,  468,  468,  701, 1013,  828,  518,  899]]]),
        "12.0": torch.tensor([[[  62,  835,  835,  835,  835,  835,  835,  835,  408,  408],
            [1007, 1007, 1007,  544,  424,  424, 1007,  424,  302,  424],
            [ 786,  678,  821,  786,   36,   36,  786,  212,  937,  937],
            [ 741,  741,  741,  993,  741, 1018,  993,  919,  741,  741],
            [ 528,  446,  198,  190,  446,  622,  646,  448,  646,  448],
            [1011,  140,  185,  986,  683,  986,  435,   41,  140,  939],
            [ 896,  772,  562,  772,  485,  528,  896,  853,  562,  772],
            [ 899,  975,  468,  468,  468,  701, 1013,  828,  518,  899],
            [ 827,  807,  938,  320,  699,  470,  909,  628,  301,  827],
            [ 963,  801,  630,  477,  717,  354,  205,  359,  874,  744],
            [1000, 1000,  388, 1000,  408,  740,  568,  364,  709,  843],
            [ 413,  835,  382,  840,  742, 1019,  375,  962,  835,  742],
            [ 971,  410,  998,  485,  798,  410,  351,  485,  485,  920],
            [ 848,  694,  662,  784,  848,  427, 1022,  848,  920,  694],
            [ 420,  911,  889,  911,  993,  776,  948,  477,  911,  911],
            [ 587,  755,  834,  962,  860,  425,  982,  982,  425,  461]]]),
        "24.0": torch.tensor([[[  62,  835,  835,  835,  835,  835,  835,  835,  408,  408],
            [1007, 1007, 1007,  544,  424,  424, 1007,  424,  302,  424],
            [ 786,  678,  821,  786,   36,   36,  786,  212,  937,  937],
            [ 741,  741,  741,  993,  741, 1018,  993,  919,  741,  741],
            [ 528,  446,  198,  190,  446,  622,  646,  448,  646,  448],
            [1011,  140,  185,  986,  683,  986,  435,   41,  140,  939],
            [ 896,  772,  562,  772,  485,  528,  896,  853,  562,  772],
            [ 899,  975,  468,  468,  468,  701, 1013,  828,  518,  899],
            [ 827,  807,  938,  320,  699,  470,  909,  628,  301,  827],
            [ 963,  801,  630,  477,  717,  354,  205,  359,  874,  744],
            [1000, 1000,  388, 1000,  408,  740,  568,  364,  709,  843],
            [ 413,  835,  382,  840,  742, 1019,  375,  962,  835,  742],
            [ 971,  410,  998,  485,  798,  410,  351,  485,  485,  920],
            [ 848,  694,  662,  784,  848,  427, 1022,  848,  920,  694],
            [ 420,  911,  889,  911,  993,  776,  948,  477,  911,  911],
            [ 587,  755,  834,  962,  860,  425,  982,  982,  425,  461],
            [ 270,  160,   26,  131,  597,  506,  670,  637,  248,  160],
            [  15,  215,  134,   69,  215,  155, 1012, 1009,  260,  417],
            [ 580,  561,  686,  896,  497,  637,  580,  245,  896,  264],
            [ 511,  239,  560,  691,  571,  627,  571,  571,  258,  619],
            [ 591,  942,  591,  251,  250,  250,  857,  486,  295,  295],
            [ 565,  546,  654,  301,  301,  623,  639,  568,  565,  282],
            [ 539,  317,  639,  539,  651,  539,  538,  640,  615,  615],
            [ 637,  556,  637,  582,  640,  515,  515,  632,  254,  613],
            [ 305,  643,  500,  550,  522,  500,  550,  561,  522,  305],
            [ 954,  456,  584,  755,  505,  782,  661,  671,  497,  505],
            [ 577,  464,  637,  647,  552,  552,  624,  647,  624,  647],
            [ 728,  748,  931,  608,  538, 1015,  294,  294,  666,  538],
            [ 602,  535,  666,  665,  655,  979,  574,  535,  571,  781],
            [ 321,  620,  557,  566,  511,  910,  672,  623,  853,  674],
            [ 621,  556,  947,  474,  610,  752, 1002,  597,  474,  474],
            [ 605,  948,  657,  588,  485,  633,  459,  968,  939,  325]]]),
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
        "1.5": torch.tensor([[ 3.087018e-04, -2.249996e-04, -2.755085e-05, -4.305589e-04,
            3.778310e-04,  3.234166e-04, -1.467389e-05,  6.210566e-05,
            5.410432e-04,  1.136818e-04, -1.511903e-03, -7.401218e-04,
            -1.899542e-04, -1.767058e-03, -2.996303e-04,  1.261118e-03,
            1.065084e-03,  8.070581e-04,  7.651932e-04,  7.807905e-04,
            7.747693e-04,  1.948763e-04, -3.002863e-04, -3.910004e-04,
            -5.593142e-04, -8.633587e-04, -1.014841e-03, -1.215550e-03,
            -1.080051e-03, -6.163150e-04, -5.803605e-04, -4.559063e-04,
            6.545106e-07,  1.374525e-04,  3.101048e-04,  2.086539e-04,
            -7.083790e-05, -2.040481e-04, -7.651921e-04, -1.227450e-03,
            -1.051491e-03, -1.242532e-03, -1.349094e-03, -3.129471e-04,
            1.561179e-04,  5.577153e-04,  5.725902e-04,  6.134030e-04,
            9.167758e-04,  1.023911e-03]]),
        "3.0": torch.tensor([[ 3.427579e-04, -1.965427e-04, -9.465658e-06, -3.997829e-04,
            4.041354e-04,  3.245588e-04, -4.020725e-06,  7.692889e-05,
            5.687681e-04,  1.576325e-04, -1.524248e-03, -7.779750e-04,
            -2.097847e-04, -1.842512e-03, -3.374760e-04,  1.318993e-03,
            1.115316e-03,  8.486996e-04,  8.107351e-04,  8.106462e-04,
            8.139778e-04,  2.142329e-04, -2.974489e-04, -3.783157e-04,
            -5.386065e-04, -8.332478e-04, -9.924322e-04, -1.217482e-03,
            -1.090740e-03, -6.342730e-04, -6.249548e-04, -4.910673e-04,
            -1.881851e-05,  1.271031e-04,  3.347668e-04,  2.426834e-04,
            -5.276939e-05, -1.842646e-04, -7.858569e-04, -1.302414e-03,
            -1.087313e-03, -1.271453e-03, -1.391595e-03, -2.859407e-04,
            2.002510e-04,  6.105778e-04,  6.221389e-04,  6.397478e-04,
            9.393271e-04,  1.037851e-03]]),
        "6.0": torch.tensor([[ 4.097951e-04, -1.358893e-04,  1.122614e-04, -3.205390e-04,
            4.156321e-04,  2.945793e-04,  2.515914e-05,  1.316031e-04,
            6.587585e-04,  2.420154e-04, -1.271795e-03, -6.608871e-04,
            -2.096211e-04, -1.542059e-03, -5.626467e-05,  1.425945e-03,
            1.386978e-03,  1.142271e-03,  1.047366e-03,  1.007879e-03,
            8.801602e-04,  3.939698e-04,  2.170181e-05,  4.447053e-05,
            9.107151e-06, -4.322665e-05, -1.146311e-04, -3.770065e-04,
            -4.023792e-04, -1.354858e-04, -1.741600e-04, -1.991981e-04,
            2.435511e-04,  4.627375e-04,  8.875711e-04,  9.537265e-04,
            7.995916e-04,  7.257143e-04,  1.987337e-04, -3.021287e-04,
            -4.460105e-04, -7.526089e-04, -7.929441e-04,  4.802318e-05,
            6.481988e-04,  1.042258e-03,  1.237199e-03,  1.169973e-03,
            1.283981e-03,  1.369294e-03]]),
        "12.0": torch.tensor([[ 3.599113e-04, -1.228984e-04,  1.194533e-04, -3.809562e-04,
            3.096270e-04,  1.871234e-04, -3.093132e-05,  6.259106e-05,
            5.706630e-04,  1.785299e-04, -1.312468e-03, -6.263725e-04,
            -1.081272e-04, -1.436411e-03,  1.275754e-04,  1.786048e-03,
            1.766336e-03,  1.385626e-03,  1.240537e-03,  1.264289e-03,
            1.145659e-03,  5.549915e-04,  3.496132e-05,  6.686400e-06,
            -2.233174e-05, -6.928211e-05, -1.322472e-04, -3.865736e-04,
            -3.617364e-04, -3.343959e-05, -3.247391e-05, -2.688728e-05,
            4.550286e-04,  6.845109e-04,  1.077411e-03,  1.075274e-03,
            8.641503e-04,  7.475881e-04,  1.910333e-04, -3.040591e-04,
            -4.346113e-04, -7.088160e-04, -7.091159e-04,  1.574968e-04,
            8.514467e-04,  1.279237e-03,  1.475426e-03,  1.390552e-03,
            1.519793e-03,  1.624769e-03]]),
        "24.0": torch.tensor([[ 5.389549e-04,  1.001411e-04,  4.115460e-04, -1.345069e-04,
            3.394131e-04,  1.673254e-04,  2.863718e-05,  9.096695e-05,
            6.653586e-04,  5.024463e-04, -1.092146e-03, -4.650788e-04,
            -8.546474e-05, -1.841910e-03, -1.083735e-05,  2.069108e-03,
            1.900907e-03,  1.263875e-03,  1.062679e-03,  1.153082e-03,
            1.226268e-03,  6.062322e-04, -2.111010e-05, -5.016541e-05,
            -4.448114e-05, -1.264126e-05, -1.051225e-04, -4.422228e-04,
            -3.979425e-04, -1.405734e-05, -1.041874e-04, -1.577306e-04,
            2.517499e-04,  4.010100e-04,  7.958416e-04,  7.111662e-04,
            5.579393e-04,  7.072217e-04,  1.398614e-04, -4.379500e-04,
            -3.199176e-04, -6.377858e-04, -7.612635e-04,  3.631268e-04,
            1.148239e-03,  1.488227e-03,  1.624454e-03,  1.462807e-03,
            1.621545e-03,  1.848965e-03]])
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([[ 3.408650e-03,  2.755654e-03,  3.668536e-03,  4.094988e-03,
            2.945655e-03,  2.198583e-03,  2.073594e-03,  2.048190e-03,
            2.145090e-03,  2.341735e-03,  2.114077e-03,  1.813927e-03,
            1.900145e-03,  2.026745e-03,  1.971920e-03,  1.962152e-03,
            2.080119e-03,  2.298388e-03,  2.455704e-03,  2.230631e-03,
            1.708692e-03,  1.503234e-03,  1.738413e-03,  2.039039e-03,
            2.406925e-03,  3.108143e-03,  3.943520e-03,  4.517261e-03,
            4.575449e-03,  4.160416e-03,  3.415633e-03,  2.706856e-03,
            2.288600e-03,  2.223748e-03,  2.349732e-03,  2.364075e-03,
            2.246506e-03,  2.261108e-03,  2.447826e-03,  2.658853e-03,
            2.710755e-03,  2.660485e-03,  2.507466e-03,  2.355561e-03,
            2.392722e-03,  2.631321e-03,  2.798146e-03,  2.672679e-03,
            2.362285e-03,  2.175428e-03],
            [-3.079133e-03, -2.654349e-03, -1.781143e-03, -1.715236e-03,
            -2.423057e-03, -2.909977e-03, -2.983763e-03, -2.604161e-03,
            -2.149914e-03, -1.805579e-03, -1.833625e-03, -1.881540e-03,
            -1.706100e-03, -1.408323e-03, -1.174338e-03, -1.028600e-03,
            -8.162585e-04, -3.740802e-04, -6.475294e-05, -4.338631e-04,
            -1.185309e-03, -1.511731e-03, -1.448557e-03, -1.330467e-03,
            -1.057644e-03, -4.734192e-04,  2.269606e-04,  7.330708e-04,
            7.882260e-04,  4.163630e-04, -2.965240e-04, -9.526202e-04,
            -1.233442e-03, -1.141584e-03, -9.377991e-04, -8.633148e-04,
            -8.694985e-04, -7.642351e-04, -5.878785e-04, -4.593362e-04,
            -4.585982e-04, -4.998497e-04, -6.303269e-04, -8.036116e-04,
            -8.215446e-04, -6.297485e-04, -5.400242e-04, -7.223252e-04,
            -1.034525e-03, -1.202643e-03]]),
        "6.0": torch.tensor([[ 5.181137e-03,  4.922385e-03,  5.716439e-03,  5.774433e-03,
            4.772815e-03,  4.282302e-03,  4.199335e-03,  4.138452e-03,
            4.081835e-03,  4.168777e-03,  4.016222e-03,  3.810926e-03,
            3.831653e-03,  3.820020e-03,  3.743140e-03,  3.703121e-03,
            3.690324e-03,  3.711614e-03,  3.762511e-03,  3.703800e-03,
            3.451020e-03,  3.354752e-03,  3.599911e-03,  3.928193e-03,
            4.257949e-03,  4.745825e-03,  5.311278e-03,  5.701936e-03,
            5.735849e-03,  5.468391e-03,  5.014659e-03,  4.591195e-03,
            4.272762e-03,  4.129300e-03,  4.167245e-03,  4.172151e-03,
            4.091898e-03,  4.069445e-03,  4.174400e-03,  4.318385e-03,
            4.334324e-03,  4.272258e-03,  4.132325e-03,  4.022794e-03,
            4.024054e-03,  4.107836e-03,  4.191909e-03,  4.159877e-03,
            4.011335e-03,  3.859342e-03],
            [ 9.999929e-05,  6.048981e-04,  1.280482e-03,  1.105714e-03,
            5.083668e-04,  1.425198e-04, -5.647287e-05,  1.154560e-04,
            3.294387e-04,  5.224237e-04,  5.086779e-04,  4.936489e-04,
            5.691826e-04,  6.867505e-04,  8.367290e-04,  9.330270e-04,
            1.028508e-03,  1.271363e-03,  1.496933e-03,  1.370267e-03,
            9.752752e-04,  8.336182e-04,  9.773751e-04,  1.198026e-03,
            1.486629e-03,  1.862197e-03,  2.283163e-03,  2.615628e-03,
            2.646522e-03,  2.407925e-03,  1.968294e-03,  1.563834e-03,
            1.338653e-03,  1.316061e-03,  1.408976e-03,  1.464309e-03,
            1.489681e-03,  1.557224e-03,  1.656536e-03,  1.723670e-03,
            1.684144e-03,  1.610477e-03,  1.473058e-03,  1.340209e-03,
            1.279023e-03,  1.304318e-03,  1.304162e-03,  1.197890e-03,
            1.023026e-03,  8.714336e-04]]),
        "12.0": torch.tensor([[ 1.415910e-03,  1.192794e-03,  2.146655e-03,  2.428372e-03,
            1.663597e-03,  1.314939e-03,  1.212386e-03,  1.102170e-03,
            1.089469e-03,  1.229083e-03,  1.144423e-03,  9.564958e-04,
            9.269248e-04,  8.943039e-04,  8.352196e-04,  8.377288e-04,
            8.848651e-04,  1.012009e-03,  1.170346e-03,  1.152839e-03,
            9.146292e-04,  8.161103e-04,  1.015097e-03,  1.332060e-03,
            1.746635e-03,  2.391235e-03,  3.111152e-03,  3.597358e-03,
            3.639205e-03,  3.295199e-03,  2.764174e-03,  2.318388e-03,
            2.047770e-03,  2.009262e-03,  2.152099e-03,  2.243418e-03,
            2.209484e-03,  2.197980e-03,  2.274900e-03,  2.381738e-03,
            2.377144e-03,  2.281965e-03,  2.142197e-03,  2.069176e-03,
            2.128260e-03,  2.296847e-03,  2.445204e-03,  2.423759e-03,
            2.233535e-03,  2.056958e-03],
            [-3.388673e-03, -2.910895e-03, -2.036629e-03, -1.975390e-03,
            -2.404006e-03, -2.719029e-03, -3.015102e-03, -2.955779e-03,
            -2.758377e-03, -2.540904e-03, -2.499833e-03, -2.532349e-03,
            -2.540440e-03, -2.477246e-03, -2.336786e-03, -2.200043e-03,
            -2.028422e-03, -1.660484e-03, -1.311706e-03, -1.368339e-03,
            -1.728671e-03, -1.874492e-03, -1.757278e-03, -1.517199e-03,
            -1.133040e-03, -5.816742e-04,  1.887305e-05,  4.575543e-04,
            4.940191e-04,  1.725325e-04, -3.440334e-04, -7.704243e-04,
            -9.601419e-04, -9.033157e-04, -7.289852e-04, -6.062163e-04,
            -5.671192e-04, -5.318156e-04, -4.846281e-04, -4.621183e-04,
            -5.413577e-04, -6.716363e-04, -8.260311e-04, -9.177739e-04,
            -8.950318e-04, -7.743726e-04, -7.041333e-04, -7.864889e-04,
            -9.855746e-04, -1.135576e-03]]),
        "24.0": torch.tensor([[ 1.018651e-03,  7.552321e-04,  1.753446e-03,  2.060588e-03,
            1.388925e-03,  1.058750e-03,  8.910452e-04,  7.373589e-04,
            5.907872e-04,  5.859354e-04,  4.967420e-04,  3.487321e-04,
            2.728861e-04,  1.710933e-04,  9.102832e-05,  6.730608e-05,
            9.563086e-05,  1.588390e-04,  1.925910e-04,  7.877406e-05,
            -2.257030e-04, -4.092292e-04, -2.810485e-04,  4.973972e-05,
            5.090538e-04,  1.122208e-03,  1.782423e-03,  2.205568e-03,
            2.165871e-03,  1.761347e-03,  1.197651e-03,  7.130997e-04,
            3.948150e-04,  3.065997e-04,  4.365446e-04,  5.816780e-04,
            6.346264e-04,  6.625293e-04,  7.463146e-04,  8.522305e-04,
            8.166951e-04,  6.683207e-04,  4.954571e-04,  4.100801e-04,
            4.461716e-04,  5.606022e-04,  6.683783e-04,  6.590774e-04,
            5.163411e-04,  3.634602e-04],
            [-3.862556e-03, -3.509443e-03, -2.713567e-03, -2.569903e-03,
            -2.813273e-03, -3.077238e-03, -3.452139e-03, -3.475926e-03,
            -3.390719e-03, -3.262727e-03, -3.226191e-03, -3.196019e-03,
            -3.138484e-03, -3.056201e-03, -2.923863e-03, -2.801908e-03,
            -2.645475e-03, -2.359281e-03, -2.102200e-03, -2.129314e-03,
            -2.405624e-03, -2.532886e-03, -2.405624e-03, -2.096412e-03,
            -1.651015e-03, -1.128547e-03, -6.046548e-04, -2.096784e-04,
            -1.587007e-04, -4.258386e-04, -8.811383e-04, -1.290479e-03,
            -1.523965e-03, -1.543541e-03, -1.412241e-03, -1.263308e-03,
            -1.158103e-03, -1.078569e-03, -1.013150e-03, -9.844207e-04,
            -1.074704e-03, -1.232233e-03, -1.392531e-03, -1.471757e-03,
            -1.462431e-03, -1.392031e-03, -1.349437e-03, -1.406376e-03,
            -1.565751e-03, -1.714973e-03]])
    }
}
EXPECTED_CODEC_ERROR = {
    "facebook/encodec_24khz": {
        "1.5": 0.0022227917797863483,
        "3.0": 0.0018621447961777449,
        "6.0": 0.001523723010905087,
        "12.0": 0.0013096692273393273,
        "24.0": 0.001233122544363141,
    },
    "facebook/encodec_48khz": {
        "3.0": 0.0008403955143876374,
        "6.0": 0.000669293396640569,
        "12.0": 0.0005329190171323717,
        "24.0": 0.0004473400767892599,
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
            [ 413,  835,  742,  249,  892,  352,  190,  498,  866,  890],
            [ 817,  351,  804,  751,  938,  535,  434,  879,  351,  971],
            [ 792,  495,  935,  848,  792,  795,  942,  935,  723,  531],
            [ 622,  681,  477,  713,  752,  871,  713,  514,  993,  777],
            [ 928,  799,  962, 1005,  860,  439,  312,  922,  982,  922]],

            [[ 835,  835,  835,  835,  835,  835,  835,  835,  835,  835],
            [ 857,  857,  544,  518,  937,  518,  913,  913,  518,  913],
            [ 705,  989,  934,  989,  678,  934,  934,  786,  934,  786],
            [ 366, 1018,  398,  398,  398,  398,  673,  741,  398,  741],
            [ 373,  373,  375,  373,  373,  222,  862,  373,  190,  373],
            [ 293,  949,  435,  435,  435,  293,  949,  881,  632,  986],
            [ 800,  528,  528,  853,  782,  485,  772,  900,  528,  853],
            [ 916,  237,  828,  701,  518,  835,  948,  315,  948,  315],
            [ 420,  628,  918,  628,  628,  628,  248,  628,  909,  811],
            [ 736,  717,  994,  974,  477,  874,  963,  979,  355,  979],
            [1002, 1002,  894,  875,  388,  709,  534,  408,  881,  709],
            [ 735,  828,  763,  742,  640,  835,  828,  375,  840,  375],
            [ 898,  938,  556,  658,  410,  951,  486,  658,  877,  877],
            [   0,  797,  428,  694,  428,  920, 1022, 1022,  809,  797],
            [ 622,  421,  422,  776,  911,  911,  958,  421,  776,  421],
            [1005,  312,  922,  755,  834,  461,  461,  702,  597,  974]]]),
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
            [ 413,  835,  742,  249,  892,  352,  190,  498,  866,  890],
            [ 817,  351,  804,  751,  938,  535,  434,  879,  351,  971],
            [ 792,  495,  935,  848,  792,  795,  942,  935,  723,  531],
            [ 622,  681,  477,  713,  752,  871,  713,  514,  993,  777],
            [ 928,  799,  962, 1005,  860,  439,  312,  922,  982,  922],
            [ 939,  637,  861,  506,  861,   61,  475,  264, 1019,  260],
            [ 166,  215,   69,   69,  890,   69,  284,  828,  396,  180],
            [ 561,  896,  841,  144,  580,  659,  886,  514,  686,  451],
            [ 691,  691,  239,  735,   62,  287,  383,  972,  550,  505],
            [ 451,  811,  238,  251,  250,  841,  734,  329,  551,  846],
            [ 313,  601,  494,  763,  811,  565,  748,  441,  601,  480],
            [ 653,  242,  630,  572,  701,  973,  632,  374,  561,  521],
            [ 984,  987,  419,  454,  386,  507,  532,  636,  515,  671],
            [ 647,  550,  515,  292,  876, 1011,  719,  549,  691,  911],
            [ 683,  536,  656,  603,  698,  867,  987,  857,  886,  491],
            [ 444,  937,  826,  555,  585,  710,  466,  852,  655,  591],
            [ 658,  952,  903,  508,  739,  596,  420,  721,  464,  306],
            [ 665,  334,  765,  532,  618,  278,  836,  838,  517,  597],
            [ 613,  674,  596,  904,  987,  977,  938,  615,  672,  776],
            [ 689,  386,  749,  658,  250,  869,  957,  806,  750,  659],
            [ 652,  509,  910,  826,  566,  622,  951,  696,  900,  895]],

            [[ 835,  835,  835,  835,  835,  835,  835,  835,  835,  835],
            [ 857,  857,  544,  518,  937,  518,  913,  913,  518,  913],
            [ 705,  989,  934,  989,  678,  934,  934,  786,  934,  786],
            [ 366, 1018,  398,  398,  398,  398,  673,  741,  398,  741],
            [ 373,  373,  375,  373,  373,  222,  862,  373,  190,  373],
            [ 293,  949,  435,  435,  435,  293,  949,  881,  632,  986],
            [ 800,  528,  528,  853,  782,  485,  772,  900,  528,  853],
            [ 916,  237,  828,  701,  518,  835,  948,  315,  948,  315],
            [ 420,  628,  918,  628,  628,  628,  248,  628,  909,  811],
            [ 736,  717,  994,  974,  477,  874,  963,  979,  355,  979],
            [1002, 1002,  894,  875,  388,  709,  534,  408,  881,  709],
            [ 735,  828,  763,  742,  640,  835,  828,  375,  840,  375],
            [ 898,  938,  556,  658,  410,  951,  486,  658,  877,  877],
            [   0,  797,  428,  694,  428,  920, 1022, 1022,  809,  797],
            [ 622,  421,  422,  776,  911,  911,  958,  421,  776,  421],
            [1005,  312,  922,  755,  834,  461,  461,  702,  597,  974],
            [ 248,  248,  637,  248,  977,  506,  546,  270,  670,  506],
            [ 547,  447,   15,  134, 1009,  215,  134,  396,  260,  160],
            [ 635,  497,  686,  765,  264,  497,  244,  675,  624,  656],
            [ 864,  571,  616,  511,  588,  781,  525,  258,  674,  503],
            [ 449,  757,  857,  451,  658,  486,  299,  299,  251,  596],
            [ 809,  628,  255,  568,  623,  301,  639,  546,  617,  623],
            [ 551,  497,  908,  539,  661,  710,  640,  539,  646,  315],
            [ 689,  507,  875,  515,  613,  637,  527,  515,  662,  637],
            [ 983,  686,  456,  768,  601,  561,  768,  653,  500,  688],
            [ 493,  566,  664,  782,  683,  683,  721,  603,  323,  497],
            [1015,  552,  411,  423,  607,  646,  687, 1018,  689,  607],
            [ 516,  293,  471,  294,  293,  294,  608,  538,  803,  717],
            [ 974,  994,  952,  637,  637,  927,  535,  571,  602,  535],
            [ 776,  789,  476,  944,  652,  959,  589,  679,  321,  623],
            [ 776,  931,  720, 1009,  676,  731,  386,  676,  701,  676],
            [ 684,  543,  716,  392,  661,  517,  792,  588,  922,  676]]])
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
        "24.0": torch.tensor([[[ 790,  790,  790,  214,  214,  214,  799,  214,  214,  214],
            [ 989,  989,   77,  546,  989,  546,  989,  160,  546,  989],
            [ 977,  977,  977,  977,  538,  977,  977,  960,  977,  977],
            [ 376,  376,  962,  962,  607,  962,  963,  896,  962,  376],
            [ 979,  979,  979, 1012,  979, 1012,  921,    0, 1002,  695],
            [ 824, 1018,  762,  957,  824,  762,  762, 1007,  957,  336],
            [ 681,  973,  973,  452,  211,  681,  802,  679,  547,  884],
            [ 950, 1017, 1016, 1017,  986, 1017,  229,  607, 1017,  689],
            [1004, 1011,  669, 1023, 1023, 1023,  905,  297,  810,  970],
            [ 982,  681,  982,  629,  662,  919,  878,  476,  629,  982],
            [ 727,  727,  959,  959,  979,  959,  530,  959,  337,  961],
            [ 924,  456,  924,  486,  924,  959,  102,  924,  805,  924],
            [ 649,  542,  993,  993,  949,  787,   56,  886,  949,  405],
            [ 864, 1022, 1022, 1022,  460,  753,  805,  309, 1022,   32],
            [ 953,    0,    0,  180,  352,   10,  581,  516,  322,  452],
            [ 300,    0, 1020,  307,    0,  543,  924,  627,  258,  262]],
            [[ 214,  214,  214,  214,  214,  214,  214,  214,  214,  214],
            [ 289,  289,  989,  764,  289,  289,  882,  882,  882,  882],
            [1022, 1022,  471,  925,  821,  821,  267,  925,  925,  267],
            [ 979,  992,  914,  921,    0,    0, 1023,  963,  963, 1023],
            [ 403,  940,  976, 1018,  677, 1002,  979,  677,  677,  677],
            [1018,  794,  762,  444,  485,  485,  974,  548,  548, 1018],
            [ 679,  243,  679, 1005, 1005,  973, 1014, 1005, 1005, 1014],
            [ 810,   13, 1017,  537,  522,  702,  202, 1017, 1017,   15],
            [ 728,  252,  970,  984,  971,  950,  673,  902, 1011,  810],
            [ 332, 1014,  476,  854, 1014,  861,  332,  411,  411,  408],
            [ 959,  727,  611,  979,  611,  727,  999,  497,  821,    0],
            [ 995,  698,  924,  688,  102,  510,  924,  970,  344,  961],
            [  81,  516,  847,  924,   10,  240, 1005,  726,  993,  378],
            [ 467,  496,  484,  496,  456, 1022,  337,  600,  456, 1022],
            [ 789,   65,  937,  976,  159,  953,  343,  764,  179,  159],
            [  10,  790,  483,   10, 1020,  352,  848,  333,   83,  848]]])
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
        "1.5": torch.tensor([[[ 1.016760e-03,  4.296955e-04,  5.491953e-04,  2.432936e-04,
            4.727909e-04, -5.819977e-05, -2.841300e-04, -1.349407e-04,
            3.247863e-04,  1.316009e-04, -1.419757e-03, -9.023010e-04,
            -7.104225e-04, -2.328745e-03, -8.700579e-04,  7.946090e-04,
            7.428678e-04,  3.130919e-04,  1.344616e-04,  1.046253e-04,
            3.083577e-04, -5.001850e-05, -3.287713e-04, -4.010783e-04,
            -4.882467e-04, -7.169820e-04, -8.809665e-04, -1.057924e-03,
            -9.677098e-04, -6.003812e-04, -6.542843e-04, -7.203204e-04,
            -4.899080e-04, -4.978263e-04, -2.667716e-04, -2.295148e-04,
            -2.456166e-04, -1.191492e-04, -4.553242e-04, -7.707328e-04,
            -5.192623e-04, -6.851763e-04, -8.513132e-04, -1.535531e-04,
            3.076633e-04,  5.190631e-04,  3.738496e-04,  9.542546e-05,
            2.819930e-04,  4.286929e-04]],

            [[-8.746407e-05, -3.769387e-05,  2.712351e-04,  8.664969e-05,
            5.081955e-04,  6.161898e-05, -6.286080e-04, -2.123654e-04,
            2.401308e-04,  2.092494e-04, -3.127429e-03, -4.338862e-04,
            5.606076e-04, -6.612622e-03, -3.159188e-03,  4.351259e-03,
            2.488110e-03, -1.871437e-03, -1.655042e-03,  9.219015e-05,
            1.904125e-03, -1.025013e-03, -1.411492e-03, -9.499654e-04,
            -6.975870e-04, -9.307091e-04, -1.928556e-03, -2.357107e-03,
            -1.920695e-03, -6.232093e-05, -1.725447e-03, -2.188495e-03,
            -3.747486e-04,  4.702547e-04, -1.422658e-03, -2.347665e-03,
            1.765973e-04,  1.453091e-03, -2.163914e-03, -3.296894e-03,
            2.419690e-03,  9.068951e-04, -4.141090e-03,  4.795498e-05,
            2.958802e-03,  2.040136e-03, -1.471912e-03, -1.753003e-03,
            1.443812e-03,  7.041964e-04]]]),
        "3.0": torch.tensor([[[ 1.343659e-03,  7.125742e-04,  9.489519e-04,  4.540573e-04,
            6.155492e-04,  1.943384e-04, -5.050822e-05,  2.990257e-05,
            4.607496e-04,  3.118639e-04, -1.180453e-03, -6.103680e-04,
            -3.182798e-04, -1.949246e-03, -2.894997e-04,  1.500616e-03,
            1.335346e-03,  9.064308e-04,  7.724696e-04,  7.328121e-04,
            8.118699e-04,  3.537435e-04,  9.363241e-05, -1.012128e-05,
            -8.166106e-05, -2.414797e-04, -3.100030e-04, -4.069041e-04,
            -3.605697e-04,  5.427224e-05, -1.685640e-05, -4.699508e-05,
            2.533230e-04,  2.761237e-04,  5.017862e-04,  4.520432e-04,
            3.627467e-04,  4.912826e-04,  8.745489e-05, -3.357749e-04,
            -1.565991e-04, -4.362466e-04, -6.411030e-04,  2.668832e-04,
            8.712247e-04,  1.223074e-03,  1.269870e-03,  1.166783e-03,
            1.400485e-03,  1.511375e-03]],

            [[ 2.844271e-05, -3.037703e-04,  4.771219e-04,  3.885964e-04,
            1.135865e-03,  1.319824e-03,  2.363302e-04,  4.848799e-04,
            1.998402e-04,  5.864529e-04, -2.471300e-03, -5.230539e-04,
            4.068493e-04, -6.936637e-03, -2.727145e-03,  3.842823e-03,
            1.274725e-03, -1.548761e-03, -4.859250e-04,  2.506756e-04,
            1.443204e-03, -6.444526e-04, -1.950119e-04, -1.014257e-03,
            -8.308233e-04, -7.412067e-05, -5.526430e-04, -1.210564e-03,
            -1.557695e-03,  9.681146e-04,  9.815478e-05, -1.002733e-03,
            -1.818334e-04,  1.307675e-03, -2.413917e-04, -1.725320e-03,
            5.425749e-04,  1.920893e-03, -1.853338e-03, -3.483505e-03,
            2.241354e-03, -1.207485e-04, -4.033331e-03,  1.198035e-03,
            1.464398e-03,  1.233750e-03,  7.617568e-05, -1.014426e-03,
            5.306019e-04,  3.885927e-04]]]),
        "6.0": torch.tensor([[[ 9.856705e-04,  4.613276e-04,  7.139955e-04,  1.031105e-04,
            3.460704e-04, -4.380017e-06, -2.268782e-04, -1.285234e-04,
            3.194800e-04,  9.699859e-05, -1.351965e-03, -7.373966e-04,
            -3.947380e-04, -1.935044e-03, -3.514756e-04,  1.285362e-03,
            1.167942e-03,  8.359596e-04,  7.403764e-04,  7.356673e-04,
            7.585916e-04,  3.403491e-04,  8.870137e-05,  6.197098e-05,
            -2.209028e-05, -1.437895e-04, -1.402956e-04, -2.174404e-04,
            -1.496818e-04,  2.266621e-04,  2.013011e-04,  1.491743e-04,
            4.613827e-04,  5.055442e-04,  7.740391e-04,  7.579576e-04,
            6.915248e-04,  8.011281e-04,  4.407558e-04,  1.073418e-04,
            1.985691e-04, -6.035986e-05, -1.726040e-04,  6.478655e-04,
            1.200661e-03,  1.507104e-03,  1.567813e-03,  1.441455e-03,
            1.614193e-03,  1.743808e-03]],
            [[-4.968375e-04, -5.699885e-05,  3.453821e-04,  7.873229e-05,
            1.017531e-03,  1.212661e-03,  1.562901e-04,  4.107448e-04,
            1.192567e-03,  3.004201e-04, -2.290045e-03, -3.077699e-04,
            -4.507170e-04, -6.253246e-03, -2.617962e-03,  4.004277e-03,
            2.403168e-03, -1.848922e-03, -5.089801e-04,  1.618322e-03,
            4.110660e-04, -7.860775e-04,  9.414512e-04,  1.662138e-04,
            -1.518620e-03, -3.415730e-04,  4.394002e-04, -1.111124e-03,
            -1.285446e-03,  1.159592e-03,  7.222511e-05, -1.946597e-03,
            6.775646e-04,  2.066761e-03, -8.791457e-04, -1.622458e-03,
            1.464378e-03,  1.342919e-03, -2.202417e-03, -1.495352e-03,
            1.611857e-03, -1.379079e-03, -3.261228e-03,  1.695663e-03,
            2.461790e-03, -3.538040e-04, -4.772630e-04,  1.014796e-03,
            5.161863e-04,  1.249900e-04]]]),
        "12.0": torch.tensor([[[ 3.240381e-04,  1.859961e-04,  4.124675e-04, -3.920239e-04,
            -3.369707e-04, -7.265419e-04, -7.682350e-04, -6.377182e-04,
            -1.415013e-04, -1.757126e-04, -1.610253e-03, -8.593510e-04,
            -3.860577e-04, -2.062391e-03, -3.374436e-04,  1.539979e-03,
            1.551710e-03,  1.200164e-03,  1.096843e-03,  9.670343e-04,
            9.829033e-04,  5.022055e-04,  1.522593e-04,  1.353110e-04,
            2.122422e-06, -1.286293e-04, -1.765887e-04, -3.780451e-04,
            -3.723379e-04,  3.546702e-05, -3.932872e-05, -1.706433e-04,
            8.779503e-05,  8.855003e-05,  3.635873e-04,  3.277722e-04,
            2.322989e-04,  3.700842e-04, -7.182171e-05, -5.442704e-04,
            -3.658136e-04, -6.142394e-04, -7.353026e-04,  2.630944e-04,
            9.309459e-04,  1.315633e-03,  1.491559e-03,  1.497613e-03,
            1.694125e-03,  1.813141e-03]],

            [[-7.617325e-04, -3.001209e-04,  3.099851e-04, -1.082436e-04,
            8.482186e-04,  1.253113e-03,  4.189500e-04,  7.541728e-04,
            1.505941e-03,  6.196215e-04, -2.139770e-03, -1.105258e-04,
            -3.170513e-04, -6.202026e-03, -2.209310e-03,  4.314546e-03,
            2.760602e-03, -1.288410e-03, -1.531337e-04,  1.722789e-03,
            9.883647e-04, -1.085126e-04,  8.099665e-04,  9.746556e-05,
            -1.027339e-03,  2.923012e-04,  8.302956e-04, -6.267453e-04,
            -7.253869e-04,  1.217150e-03,  2.920751e-04, -1.339112e-03,
            6.993820e-04,  1.940263e-03, -1.675738e-04, -1.279397e-03,
            1.078182e-03,  1.614507e-03, -1.626562e-03, -1.731943e-03,
            1.403012e-03, -6.133486e-04, -2.918293e-03,  1.127128e-03,
            2.817717e-03,  6.479989e-04, -3.635103e-04,  4.945297e-04,
            8.428321e-04,  3.297553e-04]]]),
        "24.0": torch.tensor([[[ 8.708999e-04,  4.444173e-04,  7.339944e-04,  2.035806e-04,
            4.156487e-04, -6.760976e-05, -3.325132e-04, -2.232262e-04,
            2.375321e-04,  1.050107e-04, -1.486843e-03, -9.337315e-04,
            -5.663714e-04, -2.380878e-03, -5.389228e-04,  1.558222e-03,
            1.423162e-03,  9.680114e-04,  8.546007e-04,  7.627727e-04,
            8.313138e-04,  3.810106e-04,  6.985760e-05,  3.564874e-05,
            -7.827355e-05, -1.881752e-04, -3.348238e-04, -6.103034e-04,
            -6.327576e-04, -3.001292e-04, -4.550446e-04, -5.744925e-04,
            -3.339901e-04, -3.840872e-04, -1.238777e-04, -1.864854e-04,
            -3.128771e-04, -1.207665e-04, -5.940771e-04, -1.108224e-03,
            -7.864439e-04, -1.035711e-03, -1.192722e-03, -1.562952e-05,
            7.172663e-04,  1.061155e-03,  1.187272e-03,  1.095990e-03,
            1.259618e-03,  1.404953e-03]],

            [[-9.067027e-04, -3.832980e-04,  1.422571e-04, -3.004262e-04,
            7.383430e-04,  1.176006e-03,  2.596731e-04,  6.081390e-04,
            1.724371e-03,  7.576933e-04, -2.022029e-03,  9.830501e-05,
            -2.197362e-04, -6.369531e-03, -2.287629e-03,  4.669123e-03,
            2.874129e-03, -1.597645e-03, -3.653977e-04,  1.883242e-03,
            1.038559e-03, -2.297463e-04,  7.287907e-04, -8.251559e-05,
            -1.254569e-03,  4.980841e-04,  1.203353e-03, -6.938266e-04,
            -8.305379e-04,  1.328052e-03, -1.471515e-04, -2.209868e-03,
            3.674791e-04,  2.045512e-03, -4.252343e-04, -1.377982e-03,
            1.706603e-03,  2.012259e-03, -1.817411e-03, -1.572970e-03,
            1.465687e-03, -1.492544e-03, -3.609665e-03,  1.358004e-03,
            2.979322e-03,  4.409417e-04,  2.396049e-04,  1.490476e-03,
            1.113037e-03,  6.844322e-04]]])
    },
    "facebook/encodec_48khz": {
        "3.0": torch.tensor([[[ 0.005063,  0.004641,  0.005683,  0.005589,  0.004229,  0.003780,
            0.003613,  0.003319,  0.003032,  0.003042,  0.002764,  0.002386,
            0.002384,  0.002437,  0.002380,  0.002356,  0.002361,  0.002442,
            0.002563,  0.002465,  0.002160,  0.002118,  0.002446,  0.002754,
            0.003003,  0.003448,  0.003948,  0.004274,  0.004306,  0.004102,
            0.003643,  0.003157,  0.002855,  0.002794,  0.002833,  0.002737,
            0.002564,  0.002532,  0.002613,  0.002696,  0.002685,  0.002613,
            0.002414,  0.002242,  0.002244,  0.002330,  0.002317,  0.002157,
            0.001941,  0.001775],
            [-0.003951, -0.002748, -0.001346, -0.001927, -0.002943, -0.003300,
            -0.003706, -0.003621, -0.003342, -0.002916, -0.002751, -0.002713,
            -0.002591, -0.002326, -0.002043, -0.001868, -0.001671, -0.001249,
            -0.000940, -0.001247, -0.001875, -0.002094, -0.002009, -0.001953,
            -0.001842, -0.001582, -0.001226, -0.000902, -0.000839, -0.000971,
            -0.001396, -0.001865, -0.002031, -0.001934, -0.001823, -0.001839,
            -0.001888, -0.001819, -0.001756, -0.001766, -0.001803, -0.001851,
            -0.002000, -0.002128, -0.002137, -0.002082, -0.002155, -0.002349,
            -0.002562, -0.002729]],
            [[ 0.004238,  0.003433,  0.005066,  0.006078,  0.005237,  0.004610,
            0.004241,  0.004077,  0.004343,  0.004727,  0.004579,  0.004118,
            0.003882,  0.003719,  0.003362,  0.003054,  0.003231,  0.003942,
            0.004530,  0.004210,  0.003322,  0.002788,  0.002621,  0.002467,
            0.002705,  0.003893,  0.005632,  0.006823,  0.006707,  0.005422,
            0.003636,  0.002346,  0.002180,  0.002876,  0.003517,  0.003425,
            0.002865,  0.002527,  0.002543,  0.002694,  0.002744,  0.002578,
            0.002143,  0.001730,  0.001820,  0.002435,  0.002959,  0.002889,
            0.002375,  0.001900],
            [-0.003469, -0.002934, -0.000603,  0.000135, -0.000737, -0.001507,
            -0.002055, -0.001900, -0.001285, -0.000679, -0.000604, -0.000882,
            -0.001093, -0.001105, -0.001188, -0.001303, -0.000920,  0.000109,
            0.000864,  0.000477, -0.000532, -0.001138, -0.001477, -0.001861,
            -0.001764, -0.000659,  0.000975,  0.002060,  0.001925,  0.000768,
            -0.000857, -0.001979, -0.001929, -0.001025, -0.000233, -0.000165,
            -0.000534, -0.000740, -0.000716, -0.000621, -0.000593, -0.000722,
            -0.001092, -0.001459, -0.001380, -0.000809, -0.000362, -0.000493,
            -0.001013, -0.001448]]]),
        "6.0": torch.tensor([[[ 5.869901e-03,  5.760327e-03,  7.069177e-03,  7.066526e-03,
            5.822343e-03,  5.575222e-03,  5.537111e-03,  5.188591e-03,
            4.825683e-03,  4.805411e-03,  4.657399e-03,  4.389170e-03,
            4.339578e-03,  4.335582e-03,  4.256999e-03,  4.213983e-03,
            4.180849e-03,  4.199632e-03,  4.306081e-03,  4.261285e-03,
            4.049223e-03,  4.062556e-03,  4.386880e-03,  4.731768e-03,
            5.043522e-03,  5.495222e-03,  5.928938e-03,  6.144925e-03,
            6.089266e-03,  5.817766e-03,  5.332279e-03,  4.829353e-03,
            4.483666e-03,  4.350332e-03,  4.302521e-03,  4.173910e-03,
            4.008351e-03,  3.973074e-03,  4.025426e-03,  4.078578e-03,
            4.055981e-03,  3.923228e-03,  3.659830e-03,  3.450820e-03,
            3.412818e-03,  3.447949e-03,  3.392654e-03,  3.220664e-03,
            2.995196e-03,  2.787384e-03],
            [-1.871272e-03, -6.005588e-04,  9.470949e-04,  5.537971e-04,
            -3.004853e-04, -5.247117e-04, -9.447127e-04, -1.075132e-03,
            -9.855141e-04, -6.314514e-04, -4.126531e-04, -3.645656e-04,
            -3.196475e-04, -1.544012e-04,  2.836789e-05,  1.540148e-04,
            3.032683e-04,  6.487701e-04,  9.752917e-04,  8.422130e-04,
            4.215928e-04,  2.998439e-04,  4.248715e-04,  5.703363e-04,
            7.572219e-04,  1.020319e-03,  1.278826e-03,  1.471286e-03,
            1.452998e-03,  1.273981e-03,  8.653461e-04,  4.241794e-04,
            2.350350e-04,  2.694613e-04,  3.120844e-04,  2.902264e-04,
            2.605360e-04,  3.195161e-04,  3.351446e-04,  2.883321e-04,
            2.407181e-04,  1.270566e-04, -9.068521e-05, -2.492648e-04,
            -3.023433e-04, -3.102121e-04, -4.303945e-04, -6.429633e-04,
            -8.705048e-04, -1.072327e-03]],
            [[ 3.559958e-04, -2.624307e-04,  1.521017e-03,  2.670331e-03,
            2.357603e-03,  2.144770e-03,  1.881193e-03,  1.857387e-03,
            2.086283e-03,  2.318392e-03,  2.215614e-03,  1.891491e-03,
            1.632142e-03,  1.410138e-03,  1.171329e-03,  9.479795e-04,
            1.109657e-03,  1.752274e-03,  2.251860e-03,  1.941987e-03,
            9.938907e-04,  1.825662e-04, -2.526554e-04, -3.093578e-04,
            3.407836e-04,  1.877459e-03,  3.897025e-03,  5.279080e-03,
            5.114052e-03,  3.642317e-03,  1.673742e-03,  2.648601e-04,
            -3.748844e-05,  5.081998e-04,  1.213369e-03,  1.391713e-03,
            1.115478e-03,  8.749667e-04,  8.865817e-04,  1.070473e-03,
            1.118169e-03,  8.939956e-04,  3.953168e-04, -1.637264e-05,
            7.792079e-05,  7.118615e-04,  1.394239e-03,  1.556494e-03,
            1.197250e-03,  6.937113e-04],
            [-6.903003e-03, -6.428347e-03, -4.033376e-03, -3.075367e-03,
            -3.423146e-03, -3.885631e-03, -4.464363e-03, -4.296352e-03,
            -3.856031e-03, -3.574003e-03, -3.573838e-03, -3.798211e-03,
            -4.127680e-03, -4.263531e-03, -4.259319e-03, -4.271819e-03,
            -3.825634e-03, -2.822138e-03, -2.154582e-03, -2.530225e-03,
            -3.553435e-03, -4.354145e-03, -4.862584e-03, -4.980015e-03,
            -4.288009e-03, -2.715500e-03, -6.886562e-04,  6.734396e-04,
            4.995088e-04, -9.112117e-04, -2.820514e-03, -4.133820e-03,
            -4.240293e-03, -3.491622e-03, -2.665361e-03, -2.367060e-03,
            -2.484381e-03, -2.633339e-03, -2.637386e-03, -2.526013e-03,
            -2.522185e-03, -2.728689e-03, -3.177021e-03, -3.553903e-03,
            -3.470586e-03, -2.875333e-03, -2.275428e-03, -2.189715e-03,
            -2.566542e-03, -3.050586e-03]]]),
        "12.0": torch.tensor([[[ 2.257938e-03,  2.542818e-03,  4.031566e-03,  4.037765e-03,
            2.882440e-03,  2.731046e-03,  2.686418e-03,  2.248694e-03,
            1.782076e-03,  1.710185e-03,  1.660699e-03,  1.473885e-03,
            1.318177e-03,  1.163520e-03,  1.057188e-03,  1.017599e-03,
            9.272197e-04,  9.115945e-04,  1.020609e-03,  1.023146e-03,
            8.969481e-04,  9.699261e-04,  1.303349e-03,  1.653811e-03,
            1.944527e-03,  2.349225e-03,  2.750804e-03,  2.942657e-03,
            2.868483e-03,  2.634068e-03,  2.247135e-03,  1.884673e-03,
            1.672699e-03,  1.635104e-03,  1.645545e-03,  1.588700e-03,
            1.494296e-03,  1.506079e-03,  1.587649e-03,  1.656168e-03,
            1.623322e-03,  1.478925e-03,  1.226927e-03,  1.050771e-03,
            1.045225e-03,  1.104862e-03,  1.075605e-03,  9.365730e-04,
            7.415658e-04,  5.584497e-04],
            [-4.527394e-03, -3.100233e-03, -1.465354e-03, -1.727746e-03,
            -2.408700e-03, -2.596933e-03, -3.152253e-03, -3.492440e-03,
            -3.576341e-03, -3.296936e-03, -3.042614e-03, -2.980997e-03,
            -3.021639e-03, -2.972345e-03, -2.852018e-03, -2.770843e-03,
            -2.709008e-03, -2.426040e-03, -2.089489e-03, -2.107811e-03,
            -2.383181e-03, -2.413462e-03, -2.237480e-03, -2.050810e-03,
            -1.866539e-03, -1.630285e-03, -1.412170e-03, -1.256945e-03,
            -1.271450e-03, -1.369129e-03, -1.617016e-03, -1.857888e-03,
            -1.892968e-03, -1.759882e-03, -1.645662e-03, -1.583826e-03,
            -1.554817e-03, -1.463936e-03, -1.404100e-03, -1.407008e-03,
            -1.448195e-03, -1.557180e-03, -1.741051e-03, -1.840148e-03,
            -1.835931e-03, -1.797252e-03, -1.861087e-03, -2.016203e-03,
            -2.210470e-03, -2.383690e-03]],
            [[ 8.669291e-04,  6.633290e-04,  2.590281e-03,  3.575038e-03,
            2.981230e-03,  2.433491e-03,  1.848123e-03,  1.549588e-03,
            1.548465e-03,  1.659816e-03,  1.708845e-03,  1.604346e-03,
            1.290162e-03,  8.546511e-04,  5.015506e-04,  2.808765e-04,
            3.726322e-04,  8.240933e-04,  1.248888e-03,  1.172699e-03,
            6.499548e-04,  1.113696e-04, -3.471897e-04, -5.421294e-04,
            -4.821626e-05,  1.395895e-03,  3.469223e-03,  4.944056e-03,
            4.770631e-03,  3.181130e-03,  1.132303e-03, -2.475958e-04,
            -4.528951e-04,  2.381504e-04,  1.123149e-03,  1.544491e-03,
            1.482745e-03,  1.275527e-03,  1.189198e-03,  1.247984e-03,
            1.120710e-03,  6.996147e-04,  1.587822e-04, -1.691869e-04,
            3.564649e-05,  8.121991e-04,  1.616788e-03,  1.810330e-03,
            1.384631e-03,  7.841352e-04],
            [-4.692973e-03, -3.978275e-03, -1.646651e-03, -8.647070e-04,
            -1.400311e-03, -2.197095e-03, -3.207461e-03, -3.427803e-03,
            -3.253629e-03, -3.052853e-03, -2.855265e-03, -2.804205e-03,
            -3.093277e-03, -3.380454e-03, -3.461935e-03, -3.438796e-03,
            -3.061399e-03, -2.246781e-03, -1.599439e-03, -1.653356e-03,
            -2.173241e-03, -2.649075e-03, -3.118944e-03, -3.329998e-03,
            -2.818193e-03, -1.424192e-03,  5.292109e-04,  1.918952e-03,
            1.780283e-03,  3.458853e-04, -1.528895e-03, -2.733524e-03,
            -2.741933e-03, -1.891067e-03, -9.265941e-04, -4.087975e-04,
            -3.315209e-04, -4.550755e-04, -5.497474e-04, -5.594483e-04,
            -7.312594e-04, -1.121874e-03, -1.587182e-03, -1.849515e-03,
            -1.635191e-03, -9.038674e-04, -1.878504e-04, -5.893710e-05,
            -4.832710e-04, -1.038469e-03]]]),
        "24.0": torch.tensor([[[ 3.502094e-04,  8.196753e-04,  2.401087e-03,  2.422293e-03,
            1.338031e-03,  1.258599e-03,  1.292651e-03,  9.482148e-04,
            5.277835e-04,  5.052023e-04,  5.743213e-04,  5.232239e-04,
            4.526575e-04,  3.370009e-04,  2.665430e-04,  2.586179e-04,
            2.041550e-04,  1.859163e-04,  2.845646e-04,  3.607040e-04,
            3.329840e-04,  4.471931e-04,  8.043678e-04,  1.180795e-03,
            1.457668e-03,  1.783830e-03,  2.094105e-03,  2.228326e-03,
            2.132790e-03,  1.913564e-03,  1.610200e-03,  1.358332e-03,
            1.192049e-03,  1.148841e-03,  1.189805e-03,  1.212025e-03,
            1.190095e-03,  1.226319e-03,  1.314654e-03,  1.413193e-03,
            1.424014e-03,  1.310239e-03,  1.091048e-03,  9.493366e-04,
            9.498795e-04,  1.007852e-03,  1.030397e-03,  9.944630e-04,
            8.802176e-04,  7.208823e-04],
            [-5.462438e-03, -4.020801e-03, -2.404662e-03, -2.576407e-03,
            -3.050965e-03, -3.128292e-03, -3.627163e-03, -3.860958e-03,
            -3.854608e-03, -3.490870e-03, -3.086853e-03, -2.865939e-03,
            -2.808679e-03, -2.715059e-03, -2.550788e-03, -2.424257e-03,
            -2.319678e-03, -2.023905e-03, -1.659949e-03, -1.575328e-03,
            -1.736223e-03, -1.707521e-03, -1.471874e-03, -1.201030e-03,
            -9.769592e-04, -7.719753e-04, -5.889812e-04, -4.402417e-04,
            -4.338916e-04, -4.694163e-04, -5.855702e-04, -6.758154e-04,
            -6.211676e-04, -4.390080e-04, -2.437853e-04, -5.821476e-05,
            8.254158e-05,  2.359983e-04,  3.430442e-04,  3.914508e-04,
            3.906525e-04,  3.074106e-04,  1.429227e-04,  5.456448e-05,
            4.066665e-05,  5.068180e-05,  4.706475e-06, -9.239689e-05,
            -2.490468e-04, -4.258359e-04]],
            [[-2.421390e-03, -2.857222e-03, -9.442655e-04,  2.274845e-04,
            -1.641855e-04, -6.657653e-04, -1.223122e-03, -1.344017e-03,
            -1.229039e-03, -1.115404e-03, -1.063133e-03, -1.195672e-03,
            -1.624271e-03, -2.127357e-03, -2.431253e-03, -2.615102e-03,
            -2.412377e-03, -1.804257e-03, -1.285638e-03, -1.450913e-03,
            -2.196640e-03, -2.903191e-03, -3.542131e-03, -3.777154e-03,
            -3.127120e-03, -1.535154e-03,  8.080448e-04,  2.468609e-03,
            2.258267e-03,  5.741441e-04, -1.588355e-03, -2.994006e-03,
            -3.179719e-03, -2.434129e-03, -1.456228e-03, -9.759070e-04,
            -9.239108e-04, -1.052614e-03, -1.031848e-03, -8.511216e-04,
            -9.775782e-04, -1.422395e-03, -1.985779e-03, -2.280825e-03,
            -2.013009e-03, -1.129731e-03, -1.453102e-04,  1.190267e-04,
            -2.841220e-04, -8.684628e-04],
            [-8.572406e-03, -8.119672e-03, -5.915692e-03, -4.957645e-03,
            -5.292121e-03, -6.062876e-03, -7.052404e-03, -7.133138e-03,
            -6.883462e-03, -6.730509e-03, -6.616289e-03, -6.639611e-03,
            -7.018533e-03, -7.345675e-03, -7.391906e-03, -7.347103e-03,
            -6.888600e-03, -5.992718e-03, -5.318657e-03, -5.460210e-03,
            -6.124684e-03, -6.657714e-03, -7.195413e-03, -7.371689e-03,
            -6.692436e-03, -5.190619e-03, -3.053391e-03, -1.504106e-03,
            -1.559952e-03, -2.917909e-03, -4.754724e-03, -5.903687e-03,
            -5.908769e-03, -5.078458e-03, -4.095469e-03, -3.558868e-03,
            -3.385065e-03, -3.422534e-03, -3.395833e-03, -3.304028e-03,
            -3.479672e-03, -3.875955e-03, -4.340887e-03, -4.557735e-03,
            -4.294242e-03, -3.504779e-03, -2.662931e-03, -2.477755e-03,
            -2.877857e-03, -3.420144e-03]]])
    }
}
# ---- error over whole batch
EXPECTED_CODEC_ERROR_BATCH = {
    "facebook/encodec_24khz": {
        "1.5": 0.0011174300452694297,
        "3.0": 0.0009308874723501503,
        "6.0": 0.0007695500389672816,
        "12.0": 0.0006829536287114024,
        "24.0": 0.0006418082630261779,
    },
    "facebook/encodec_48khz": {
        "3.0": 0.0003983891801908612,
        "6.0": 0.0003246906562708318,
        "12.0": 0.00025439003366045654,
        "24.0": 0.000219063411350362,
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
