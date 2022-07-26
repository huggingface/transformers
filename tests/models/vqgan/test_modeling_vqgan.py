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
""" Testing suite for the PyTorch VQGAN model. """


import inspect
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

from transformers import VQGANConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import VQGANModel
    from transformers.models.vqgan.modeling_vqgan import VQGAN_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image

    from transformers import AutoFeatureExtractor


class VQGANModelTester:
    def __init__(
        self,
        parent,
        hidden_channels=64,
        num_channels=3,
        num_res_blocks=2,
        resolution=64,
        z_channels=64,
        channel_mult=(1, 2, 4),
        attn_resolutions=(8,),
        num_embeddings=128,
        quantized_embed_dim=64,
        dropout=0.0,
        resample_with_conv=True,
        scope=None,
        is_training=True,
    ):
        self.parent = parent
        self.hidden_channels = hidden_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.z_channels = z_channels
        self.channel_mult = channel_mult
        self.attn_resolutions = attn_resolutions
        self.num_embeddings = num_embeddings
        self.quantized_embed_dim = quantized_embed_dim
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.scope = scope
        self.batch_size = 4
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor((self.batch_size, self.num_channels, self.resolution, self.resolution))
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return VQGANConfig(
            hidden_channels=self.hidden_channels,
            num_channels=self.num_channels,
            num_res_blocks=self.num_res_blocks,
            resolution=self.resolution,
            z_channels=self.z_channels,
            channel_mult=self.channel_mult,
            attn_resolutions=self.attn_resolutions,
            num_embeddings=self.num_embeddings,
            quantized_embed_dim=self.quantized_embed_dim,
            dropout=self.dropout,
            resample_with_conv=self.resample_with_conv,
        )

    def create_and_check_model(self, config, pixel_values):
        model = VQGANModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        result = model(pixel_values, return_loss=True)
        self.parent.assertEqual(
            result.reconstructed_pixel_values.shape,
            (self.batch_size, self.num_channels, self.resolution, self.resolution),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VQGANModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (VQGANModel,) if is_torch_available() else ()

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = VQGANModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VQGANConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    @unittest.skip(reason="VQGAN does not output hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="VQGAN does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="VQGAN does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VQGAN does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="VQGAN does not output hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                    self.assertTrue(
                        torch.all(module.weight == 1),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                    self.assertTrue(
                        torch.all(module.bias == 0),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = VQGANModel(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs_dict)

        out_2 = outputs[0].cpu().numpy()
        out_2[np.isnan(out_2)] = 0

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = VQGANModel.from_pretrained(tmpdirname)
            model.to(torch_device)
            with torch.no_grad():
                after_outputs = model(**inputs_dict)

            # Make sure we don't have nans
            out_1 = after_outputs[0].cpu().numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(VQGANModel, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(VQGANModel.main_input_name, observed_main_input_name)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in VQGAN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = VQGANModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class VQGANModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return (
            AutoFeatureExtractor.from_pretrained(VQGAN_PRETRAINED_MODEL_ARCHIVE_LIST[0])
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_masked_lm(self):
        model = VQGANModel.from_pretrained("valhalla/vqgan_imagenet_f16_16384")

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 3, 256, 256))
        self.assertEqual(outputs.reconstructed_pixel_values.shape, expected_shape)

        expected_slice = torch.tensor([0.7532, 0.7453, 0.7901, 0.8110, 0.7729, 0.6804, 1.0291, 0.7947, 0.7244]).to(
            torch_device
        )

        self.assertTrue(torch.allclose(outputs.reconstructed_pixel_values[0, 0, -3:, -3:], expected_slice, atol=1e-3))
