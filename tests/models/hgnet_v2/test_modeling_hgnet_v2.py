# coding=utf-8
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

import unittest

import torch
from torch import nn

from transformers import HGNetV2Config
from transformers.testing_utils import require_torch, torch_device
from transformers.utils.import_utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    from transformers import HGNetV2Backbone, HGNetV2ForImageClassification


class HGNetV2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_size=32,
        num_channels=3,
        embeddings_size=10,
        hidden_sizes=[64, 128, 256, 512],
        stage_in_channels=[16, 64, 128, 256],
        stage_mid_channels=[16, 32, 64, 128],
        stage_out_channels=[64, 128, 256, 512],
        stage_num_blocks=[1, 1, 2, 1],
        stage_downsample=[False, True, True, True],
        stage_light_block=[False, False, True, True],
        stage_kernel_size=[3, 3, 5, 5],
        stage_numb_of_layers=[3, 3, 3, 3],
        stem_channels=[3, 16, 16],
        depths=[1, 1, 2, 1],
        is_training=True,
        use_labels=True,
        hidden_act="relu",
        num_labels=3,
        scope=None,
        out_features=["stage2", "stage3", "stage4"],
        out_indices=[2, 3, 4],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.embeddings_size = embeddings_size
        self.hidden_sizes = hidden_sizes
        self.stage_in_channels = stage_in_channels
        self.stage_mid_channels = stage_mid_channels
        self.stage_out_channels = stage_out_channels
        self.stage_num_blocks = stage_num_blocks
        self.stage_downsample = stage_downsample
        self.stage_light_block = stage_light_block
        self.stage_kernel_size = stage_kernel_size
        self.stage_numb_of_layers = stage_numb_of_layers
        self.stem_channels = stem_channels
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.scope = scope
        self.num_stages = len(hidden_sizes)
        self.out_features = out_features
        self.out_indices = out_indices

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return HGNetV2Config(
            num_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
            hidden_sizes=self.hidden_sizes,
            stage_in_channels=self.stage_in_channels,
            stage_mid_channels=self.stage_mid_channels,
            stage_out_channels=self.stage_out_channels,
            stage_num_blocks=self.stage_num_blocks,
            stage_downsample=self.stage_downsample,
            stage_light_block=self.stage_light_block,
            stage_kernel_size=self.stage_kernel_size,
            stage_numb_of_layers=self.stage_numb_of_layers,
            stem_channels=self.stem_channels,
            depths=self.depths,
            hidden_act=self.hidden_act,
            num_labels=self.num_labels,
            out_features=self.out_features,
            out_indices=self.out_indices,
        )

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = HGNetV2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[1], 4, 4])

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, config.hidden_sizes[1:])

        # verify backbone works with out_features=None
        config.out_features = None
        model = HGNetV2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[-1], 1, 1])

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_sizes[-1]])

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = HGNetV2ForImageClassification(config)
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
class RTDetrResNetBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (HGNetV2Backbone,) if is_torch_available() else ()
    has_attentions = False
    config_class = HGNetV2Config

    def setUp(self):
        self.model_tester = HGNetV2ModelTester(self)


@require_torch
class HGNetV2ForImageClassificationTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some tests of test_modeling_common.py, as TextNet does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (HGNetV2ForImageClassification, HGNetV2Backbone) if is_torch_available() else ()
    pipeline_model_mapping = {"image-classification": HGNetV2ForImageClassification} if is_torch_available() else {}

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torch_exportable = True
    has_attentions = False

    def setUp(self):
        self.model_tester = HGNetV2ModelTester(self)

    @unittest.skip(reason="Does not work on the tiny model.")
    def test_model_parallelism(self):
        super().test_model_parallelism()

    @unittest.skip(reason="HGNetV2 does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="HGNetV2 does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="HGNetV2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="HGNetV2 does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="HGNetV2 does not have a model")
    def test_model(self):
        pass

    @unittest.skip(reason="Not relevant for the model")
    def test_can_init_all_missing_weights(self):
        pass

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

            self.assertEqual(len(hidden_states), self.model_tester.num_stages + 1)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 4, self.model_tester.image_size // 4],
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

    @unittest.skip(reason="Retain_grad is not supposed to be tested")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="TextNet does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @unittest.skip(reason="HGNetV2 does not use model")
    def test_model_from_pretrained(self):
        pass
