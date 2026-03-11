# coding = utf-8
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
"""Testing suite for the PP-LCNetV3 backbone."""

import unittest

from transformers import (
    PPLCNetV3Backbone,
    PPLCNetV3Config,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
)

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import floats_tensor


if is_torch_available():
    pass

if is_vision_available():
    pass


class PPLCNetV3ModelTester:
    def __init__(
        self,
        batch_size=3,
        image_size=128,
        num_channels=3,
        num_stages=5,
        is_training=False,
        scale=1.0,
        reduction=4,
        dropout_prob=0.2,
        class_expand=1280,
        use_last_convolution=True,
        hidden_act="hardswish",
        num_labels=4,
        out_features=["stage2", "stage3", "stage4"],
        out_indices=[2, 3, 4],
        stem_channels=16,
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.num_stages = num_stages
        self.scale = scale
        self.reduction = reduction
        self.dropout_prob = dropout_prob
        self.class_expand = class_expand
        self.use_last_convolution = use_last_convolution
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.out_features = out_features
        self.out_indices = out_indices
        self.stem_channels = stem_channels
        self.block_configs = [
            [[3, 16, 32, 1, False]],
            [[3, 32, 32, 2, False], [3, 32, 32, 1, False]],
            [[3, 32, 32, 2, False], [3, 32, 32, 1, False]],
            [
                [3, 32, 32, 2, False],
                [5, 32, 32, 1, False],
                [5, 32, 32, 1, False],
                [5, 32, 32, 1, False],
                [5, 32, 32, 1, False],
            ],
            [[5, 32, 32, 2, True], [5, 32, 32, 1, True], [5, 32, 32, 1, False], [5, 32, 32, 1, False]],
        ]

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> PPLCNetV3Config:
        config = PPLCNetV3Config(
            scale=self.scale,
            reduction=self.reduction,
            dropout_prob=self.dropout_prob,
            class_expand=self.class_expand,
            use_last_conv=self.use_last_convolution,
            hidden_act=self.hidden_act,
            out_features=self.out_features,
            out_indices=self.out_indices,
            block_configs=self.block_configs,
        )

        return config


@require_torch
class PPLCNetBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (PPLCNetV3Backbone,) if is_torch_available() else ()
    has_attentions = False
    config_class = PPLCNetV3Config

    def setUp(self):
        self.model_tester = PPLCNetV3ModelTester()
        self.config_tester = ConfigTester(
            self,
            config_class=PPLCNetV3Config,
            has_text_modality=False,
            common_properties=[],
        )
