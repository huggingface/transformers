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

import numpy as np

from transformers import VQGANConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import VQGANModel
    from transformers.models.vqgan.modeling_vqgan import VQGAN_PRETRAINED_MODEL_ARCHIVE_LIST


class VQGANModelTester:
    def __init__(
        self,
        parent,
        ch = 64,
        out_ch = 3,
        in_channels = 3,
        num_res_blocks = 2,
        resolution = 64,
        z_channels = 64,
        ch_mult = (1, 2, 4),
        attn_resolutions = (8,),
        n_embed = 128,
        embed_dim = 64,
        dropout = 0.0,
        double_z = False,
        resamp_with_conv = True,
        give_pre_end = False,
        scope=None,
    ):
        self.parent = parent
        self.ch = ch
        self.out_ch = out_ch
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.z_channels = z_channels
        self.ch_mult = ch_mult
        self.attn_resolutions = attn_resolutions
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.double_z = double_z
        self.resamp_with_conv = resamp_with_conv
        self.give_pre_end = give_pre_end
        self.scope = scope
        self.batch_size = 4

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor((self.batch_size, self.in_channels, self.resolution, self.resolution))
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return VQGANConfig(
            ch=self.ch,
            out_ch=self.out_ch,
            in_channels=self.in_channels,
            num_res_blocks=self.num_res_blocks,
            resolution=self.resolution,
            z_channels=self.z_channels,
            ch_mult=self.ch_mult,
            attn_resolutions=self.attn_resolutions,
            n_embed=self.n_embed,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
            double_z=self.double_z,
            resamp_with_conv=self.resamp_with_conv,
            give_pre_end=self.give_pre_end,
        )

    def create_and_check_model(self, config, pixel_values):
        model = VQGANModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        result = model(pixel_values, return_loss=True)
        self.parent.assertEqual(
            result.reconstructed_pixel_values.shape, 
            (self.batch_size, self.out_ch, self.resolution, self.resolution)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VQGANModelTest(unittest.TestCase):

    all_model_classes = (VQGANModel,) if is_torch_available() else ()

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

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)
    
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

    @slow
    def test_model_from_pretrained(self):
        for model_name in VQGAN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = VQGANModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# TODO (patil-suraj): Fix this test
@require_torch
class VQGANModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = VQGANModel.from_pretrained("vqgan-imagenet-f16-1024")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        # TODO Replace vocab size
        vocab_size = 32000

        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        # TODO Replace values below with what was printed above.
        expected_slice = torch.tensor(
            [[[-0.0483, 0.1188, -0.0313], [-0.0606, 0.1435, 0.0199], [-0.0235, 0.1519, 0.0175]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
