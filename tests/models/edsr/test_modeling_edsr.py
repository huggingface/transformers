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
""" Testing suite for the PyTorch EDSR model. """
import inspect
import unittest

from transformers import EDSRConfig, EDSRModel
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    from torch import nn

    from transformers import EDSRForImageSuperResolution, EDSRModel
    from transformers.models.edsr.modeling_edsr import EDSR_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image


class EDSRModelTester:
    def __init__(
        self,
        upscale=2,
        num_channels=3,
        batch_size=8,
        image_size=128,
        hidden_act="relu",
        num_res_block=16,
        num_feature_maps=64,
        res_scale=1,
        shift_mean=True,
        self_ensemble=True,
        **kwargs,
    ):
        self.upscale = upscale
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.hidden_act = hidden_act
        self.num_res_block = num_res_block
        self.num_feature_maps = num_feature_maps
        self.res_scale = res_scale
        self.shift_mean = shift_mean
        self.self_ensemble = self_ensemble
        self.image_size = image_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return EDSRConfig(
            upscale=self.upscale,
            hidden_act=self.hidden_act,
            num_res_block=self.num_res_block,
            num_feature_maps=self.num_feature_maps,
            res_scale=self.res_scale,
            shift_mean=self.shift_mean,
            self_ensemble=self.self_ensemble,
        )

    def create_and_check_model(self, config, pixel_values):
        model = EDSRModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.embed_dim, self.image_size, self.image_size)
        )

    def create_and_check_for_image_super_resolution(self, config, pixel_values):
        model = EDSRForImageSuperResolution(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_image_size = self.image_size * self.upscale

        self.parent.assertEqual(
            result.reconstruction.shape, (self.batch_size, self.num_channels, expected_image_size, expected_image_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class EDSRModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (EDSRModel, EDSRForImageSuperResolution) if is_torch_available() else ()

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = EDSRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EDSRConfig, embed_dim=37)

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

    def test_model_for_image_super_resolution(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_super_resolution(*config_and_inputs)

    @unittest.skip(reason="EDSR does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EDSR does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="EDSR does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        for model_name in EDSR_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = EDSRModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    # overwriting because of `logit_scale` parameter
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "logit_scale" in name:
                    continue
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


@require_vision
@require_torch
class EDSRModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_super_resolution_head(self):
        # TODO update to appropriate organization
        model = EDSRForImageSuperResolution.from_pretrained("edsr-base-x2").to(torch_device)
        processor = self.default_feature_extractor

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.3947, -0.4306, 0.0026]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
