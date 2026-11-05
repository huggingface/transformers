# coding = utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PP-LCNetV4 backbone."""

import unittest

from transformers import (
    PPLCNetV4Backbone,
    PPLCNetV4Config,
)
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
)

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import floats_tensor


class PPLCNetV4ModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_size=128,
        num_channels=3,
        is_training=False,
        hidden_act="relu",
        stem_type="small",
        num_labels=4,
        out_features=["stage2", "stage3", "stage4"],
        out_indices=[2, 3, 4],
        stem_channels=(3, 16, 16),
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.stem_type = stem_type
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.out_features = out_features
        self.out_indices = out_indices
        self.stem_channels = stem_channels
        self.block_configs = [
            [[3, 16, 16, 1, False]],
            [[3, 16, 16, 2, False], [3, 16, 16, 1, False]],
            [[3, 16, 16, 2, False], [3, 16, 16, 1, False]],
            [
                [3, 16, 16, 2, False],
                [5, 16, 16, 1, False],
                [5, 16, 16, 1, False],
            ],
        ]

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> PPLCNetV4Config:
        config = PPLCNetV4Config(
            num_channels=self.num_channels,
            stem_type=self.stem_type,
            stem_channels=self.stem_channels,
            hidden_act=self.hidden_act,
            out_features=self.out_features,
            out_indices=self.out_indices,
            block_configs=self.block_configs,
        )

        return config


@require_torch
class PPLCNetBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (PPLCNetV4Backbone,) if is_torch_available() else ()
    has_attentions = False
    config_class = PPLCNetV4Config

    def setUp(self):
        self.model_tester = PPLCNetV4ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PPLCNetV4Config,
            has_text_modality=False,
            common_properties=[],
        )
