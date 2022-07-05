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
""" Testing suite for the PyTorch Van model. """


import inspect
import math
import unittest

from transformers import VanConfig
from transformers.testing_utils import require_scipy, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_scipy_available, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_scipy_available():
    from scipy import stats

if is_torch_available():
    import torch
    from torch import nn

    from transformers import VanForImageClassification, VanModel
    from transformers.models.van.modeling_van import VAN_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import AutoFeatureExtractor


class VanModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=224,
        num_channels=3,
        hidden_sizes=[16, 32, 64, 128],
        depths=[1, 1, 1, 1],
        is_training=True,
        use_labels=True,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.type_sequence_label_size = num_labels

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return VanConfig(
            num_channels=self.num_channels,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            num_labels=self.num_labels,
            is_decoder=False,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = VanModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # expected last hidden states: B, C, H // 32, W // 32
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.hidden_sizes[-1], self.image_size // 32, self.image_size // 32),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        model = VanForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VanModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Van does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VanModel, VanForImageClassification) if is_torch_available() else ()

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = VanModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VanConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    @unittest.skip(reason="Van does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Van does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Van does not support input and output embeddings")
    def test_model_common_attributes(self):
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

    @require_scipy
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)

        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    self.assertTrue(
                        torch.all(module.weight == 1),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                    self.assertTrue(
                        torch.all(module.bias == 0),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                elif isinstance(module, nn.Conv2d):
                    fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                    fan_out //= module.groups
                    std = math.sqrt(2.0 / fan_out)
                    # divide by std -> mean = 0, std = 1
                    data = module.weight.data.cpu().flatten().numpy() / std
                    test = stats.anderson(data)
                    self.assertTrue(test.statistic > 0.05)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_stages = len(self.model_tester.hidden_sizes)
            # van has no embeddings
            self.assertEqual(len(hidden_states), expected_num_stages)

            # Van's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 4, self.model_tester.image_size // 4],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in VAN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = VanModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class VanModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return AutoFeatureExtractor.from_pretrained(VAN_PRETRAINED_MODEL_ARCHIVE_LIST[0])

    @slow
    def test_inference_image_classification_head(self):
        model = VanForImageClassification.from_pretrained(VAN_PRETRAINED_MODEL_ARCHIVE_LIST[0]).to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([0.1029, -0.0904, -0.6365]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
