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

import unittest

from transformers import RTDetrResNetConfig
from transformers.testing_utils import require_torch, torch_device
from transformers.utils.import_utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_modeling_common import floats_tensor, ids_tensor


if is_torch_available():
    from transformers import RTDetrResNetBackbone


class RTDetrResNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_size=32,
        num_channels=3,
        embeddings_size=10,
        hidden_sizes=[10, 20, 30, 40],
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
        return RTDetrResNetConfig(
            num_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            hidden_act=self.hidden_act,
            num_labels=self.num_labels,
            out_features=self.out_features,
            out_indices=self.out_indices,
        )

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = RTDetrResNetBackbone(config=config)
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
        model = RTDetrResNetBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[-1], 1, 1])

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_sizes[-1]])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class RTDetrResNetBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (RTDetrResNetBackbone,) if is_torch_available() else ()
    has_attentions = False
    config_class = RTDetrResNetConfig

    def setUp(self):
        self.model_tester = RTDetrResNetModelTester(self)
