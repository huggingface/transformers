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

import torch.nn as nn

from transformers import (
    TextNetBackbone,
    TextNetConfig,
    is_torch_available,
)
from transformers.models.textnet.modeling_textnet import TextNetForImageClassification
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        TextNetModel,
    )


class TextNetModelTester:
    def __init__(
        self,
        parent,
        backbone_kernel_size=3,
        backbone_stride=2,
        backbone_dilation=1,
        backbone_groups=1,
        backbone_bias=False,
        backbone_has_shuffle=False,
        backbone_in_channels=3,
        backbone_out_channels=64,
        backbone_use_bn=True,
        backbone_act_func="relu",
        backbone_dropout_rate=0,
        backbone_ops_order="weight_bn_act",
        backbone_stage1_in_channels=[64],
        backbone_stage1_out_channels=[64],
        backbone_stage1_kernel_size=[[3, 3]],
        backbone_stage1_stride=[1],
        backbone_stage1_dilation=[1],
        backbone_stage1_groups=[1],
        backbone_stage2_in_channels=[64],
        backbone_stage2_out_channels=[128],
        backbone_stage2_kernel_size=[[3, 1]],
        backbone_stage2_stride=[2],
        backbone_stage2_dilation=[1],
        backbone_stage2_groups=[1],
        backbone_stage3_in_channels=[128],
        backbone_stage3_out_channels=[256],
        backbone_stage3_kernel_size=[[1, 3]],
        backbone_stage3_stride=[2],
        backbone_stage3_dilation=[1],
        backbone_stage3_groups=[1],
        backbone_stage4_in_channels=[256],
        backbone_stage4_out_channels=[512],
        backbone_stage4_kernel_size=[[3, 3]],
        backbone_stage4_stride=[2],
        backbone_stage4_dilation=[1],
        backbone_stage4_groups=[1],
        out_features=["stage1", "stage2", "stage3", "stage4"],
        out_indices=[1, 2, 3, 4],
        batch_size=3,
        num_channels=3,
        image_size=32,
        is_training=True,
        use_labels=True,
        hidden_act="relu",
        num_labels=3,
        hidden_sizes=[64, 64, 128, 256, 512],
    ):
        self.parent = parent
        self.backbone_kernel_size = backbone_kernel_size
        self.backbone_stride = backbone_stride
        self.backbone_dilation = backbone_dilation
        self.backbone_groups = backbone_groups
        self.backbone_bias = backbone_bias
        self.backbone_has_shuffle = backbone_has_shuffle
        self.backbone_in_channels = backbone_in_channels
        self.backbone_out_channels = backbone_out_channels
        self.backbone_use_bn = backbone_use_bn
        self.backbone_act_func = backbone_act_func
        self.backbone_dropout_rate = backbone_dropout_rate
        self.backbone_ops_order = backbone_ops_order

        self.backbone_stage1_in_channels = backbone_stage1_in_channels
        self.backbone_stage1_out_channels = backbone_stage1_out_channels
        self.backbone_stage1_kernel_size = backbone_stage1_kernel_size
        self.backbone_stage1_stride = backbone_stage1_stride
        self.backbone_stage1_dilation = backbone_stage1_dilation
        self.backbone_stage1_groups = backbone_stage1_groups

        self.backbone_stage2_in_channels = backbone_stage2_in_channels
        self.backbone_stage2_out_channels = backbone_stage2_out_channels
        self.backbone_stage2_kernel_size = backbone_stage2_kernel_size
        self.backbone_stage2_stride = backbone_stage2_stride
        self.backbone_stage2_dilation = backbone_stage2_dilation
        self.backbone_stage2_groups = backbone_stage2_groups

        self.backbone_stage3_in_channels = backbone_stage3_in_channels
        self.backbone_stage3_out_channels = backbone_stage3_out_channels
        self.backbone_stage3_kernel_size = backbone_stage3_kernel_size
        self.backbone_stage3_stride = backbone_stage3_stride
        self.backbone_stage3_dilation = backbone_stage3_dilation
        self.backbone_stage3_groups = backbone_stage3_groups

        self.backbone_stage4_in_channels = backbone_stage4_in_channels
        self.backbone_stage4_out_channels = backbone_stage4_out_channels
        self.backbone_stage4_kernel_size = backbone_stage4_kernel_size
        self.backbone_stage4_stride = backbone_stage4_stride
        self.backbone_stage4_dilation = backbone_stage4_dilation
        self.backbone_stage4_groups = backbone_stage4_groups

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
            backbone_kernel_size=self.backbone_kernel_size,
            backbone_stride=self.backbone_stride,
            backbone_dilation=self.backbone_dilation,
            backbone_groups=self.backbone_groups,
            backbone_bias=self.backbone_bias,
            backbone_has_shuffle=self.backbone_has_shuffle,
            backbone_in_channels=self.backbone_in_channels,
            backbone_out_channels=self.backbone_out_channels,
            backbone_use_bn=self.backbone_use_bn,
            backbone_act_func=self.backbone_act_func,
            backbone_dropout_rate=self.backbone_dropout_rate,
            backbone_ops_order=self.backbone_ops_order,
            backbone_stage1_in_channels=self.backbone_stage1_in_channels,
            backbone_stage1_out_channels=self.backbone_stage1_out_channels,
            backbone_stage1_kernel_size=self.backbone_stage1_kernel_size,
            backbone_stage1_stride=self.backbone_stage1_stride,
            backbone_stage1_dilation=self.backbone_stage1_dilation,
            backbone_stage1_groups=self.backbone_stage1_groups,
            backbone_stage2_in_channels=self.backbone_stage2_in_channels,
            backbone_stage2_out_channels=self.backbone_stage2_out_channels,
            backbone_stage2_kernel_size=self.backbone_stage2_kernel_size,
            backbone_stage2_stride=self.backbone_stage2_stride,
            backbone_stage2_dilation=self.backbone_stage2_dilation,
            backbone_stage2_groups=self.backbone_stage2_groups,
            backbone_stage3_in_channels=self.backbone_stage3_in_channels,
            backbone_stage3_out_channels=self.backbone_stage3_out_channels,
            backbone_stage3_kernel_size=self.backbone_stage3_kernel_size,
            backbone_stage3_stride=self.backbone_stage3_stride,
            backbone_stage3_dilation=self.backbone_stage3_dilation,
            backbone_stage3_groups=self.backbone_stage3_groups,
            backbone_stage4_in_channels=self.backbone_stage4_in_channels,
            backbone_stage4_out_channels=self.backbone_stage4_out_channels,
            backbone_stage4_kernel_size=self.backbone_stage4_kernel_size,
            backbone_stage4_stride=self.backbone_stage4_stride,
            backbone_stage4_dilation=self.backbone_stage4_dilation,
            backbone_stage4_groups=self.backbone_stage4_groups,
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
            list(result.feature_maps[0].shape), [self.batch_size, self.backbone_stage1_out_channels[-1], 16, 16]
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
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, 512, 2, 2])

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
    # fx_compatible = False
    # test_pruning = False
    # test_resize_embeddings = False
    # test_head_masking = False
    # has_attentions = False

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

    @unittest.skip(reason="Bit does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Bit does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Bit does not support input and output embeddings")
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

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    self.assertTrue(
                        torch.all(module.weight == 1),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                    self.assertTrue(
                        torch.all(module.bias == 0),
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

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

            # Bit's feature maps are of shape (batch_size, num_channels, height, width)
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

    def test_model_is_small(self):
        # Just a consistency check to make sure we are not running tests on 80M parameter models.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            num_params = model.num_parameters()
            assert (
                num_params < 3000000
            ), f"{model_class} is too big for the common tests ({num_params})! It should have 1M max."

    @unittest.skip(reason="Bit does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    # def test_for_image_classification(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    # @slow
    # def test_model_from_pretrained(self):
    #     for model_name in BIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
    #         model = BitModel.from_pretrained(model_name)
    #         self.assertIsNotNone(model)


@require_torch
class BitBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (TextNetBackbone,) if is_torch_available() else ()
    config_class = TextNetConfig

    has_attentions = False

    def setUp(self):
        self.model_tester = TextNetModelTester(self)
