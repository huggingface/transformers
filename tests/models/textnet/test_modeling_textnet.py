# coding=utf-8
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
""" Testing suite for the PyTorch TextNet model. """
import inspect
import unittest

from transformers import TextNetConfig
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        TextNetBackbone,
        TextNetForImageClassification,
        TextNetModel,
        is_torch_available,
    )
    from transformers.models.textnet.modeling_textnet import TEXTNET_PRETRAINED_MODEL_ARCHIVE_LIST


class TextNetModelTester:
    def __init__(
        self,
        parent,
        kernel_size=3,
        stride=2,
        in_channels=3,
        out_channels=64,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
        conv_layer_kernel_sizes=[[[3, 3]], [[3, 1]], [[1, 3]], [[3, 3]]],
        conv_layer_strides=[
            [
                1,
            ],
            [
                2,
            ],
            [
                2,
            ],
            [
                2,
            ],
        ],
        out_features=["stage1", "stage2", "stage3", "stage4"],
        out_indices=[1, 2, 3, 4],
        batch_size=3,
        num_channels=3,
        image_size=32,
        is_training=True,
        use_labels=True,
        hidden_act="relu",
        num_labels=3,
        hidden_sizes=[64, 64, 64, 64, 64],
    ):
        self.parent = parent
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order
        self.conv_layer_kernel_sizes = conv_layer_kernel_sizes
        self.conv_layer_strides = conv_layer_strides

        self.out_features = out_features
        self.out_indices = out_indices

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.hidden_sizes = hidden_sizes

        self.num_stages = 5

    def get_config(self):
        return TextNetConfig(
            kernel_size=self.kernel_size,
            stride=self.stride,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            use_bn=self.use_bn,
            act_func=self.act_func,
            dropout_rate=self.dropout_rate,
            ops_order=self.ops_order,
            conv_layer_kernel_sizes=self.conv_layer_kernel_sizes,
            conv_layer_strides=self.conv_layer_strides,
            out_features=self.out_features,
            out_indices=self.out_indices,
            hidden_sizes=self.hidden_sizes,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = TextNetModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.hidden_sizes[-1], 2, 2),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = TextNetForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = TextNetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[1], 16, 16]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, config.hidden_sizes[1:])

        # verify backbone works with out_features=None
        config.out_features = None
        model = TextNetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, 64, 2, 2])

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_sizes[-1]])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class TextNetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TextNetModel, TextNetForImageClassification, TextNetBackbone) if is_torch_available() else ()

    pipeline_model_mapping = (
        {"feature-extraction": TextNetModel, "image-classification": TextNetForImageClassification}
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = TextNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TextNetConfig, hidden_size=37)

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

    @unittest.skip(reason="TextNet does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="TextNet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TextNet does not support input and output embeddings")
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

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_stages = self.model_tester.num_stages - 1
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 2, self.model_tester.image_size // 2],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        layers_type = ["preactivation", "bottleneck"]
        for model_class in self.all_model_classes:
            for layer_type in layers_type:
                config.layer_type = layer_type
                inputs_dict["output_hidden_states"] = True
                check_hidden_states_output(inputs_dict, config, model_class)

                # check that output_hidden_states also work using config
                del inputs_dict["output_hidden_states"]
                config.output_hidden_states = True

                check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="TextNet does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TEXTNET_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TextNetModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class TextNetBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (TextNetBackbone,) if is_torch_available() else ()
    config_class = TextNetConfig

    has_attentions = False

    def setUp(self):
        self.model_tester = TextNetModelTester(self)
