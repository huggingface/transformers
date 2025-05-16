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
"""Testing suite for the PyTorch EoMT model."""

import unittest

from transformers import EoMTConfig, EoMTForUniversalSegmentation
from transformers.testing_utils import (
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


if is_vision_available():
    pass


class EoMTForUniversalSegmentationTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        is_training=True,
        image_size=30,
        patch_size=2,
        num_queries=10,
        num_labels=4,
        hidden_dim=64,
        num_attention_heads=4,
        num_hidden_layers=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_queries = num_queries
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.mask_feature_size = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, 3, self.image_size, self.image_size]).to(
            torch_device
        )

        mask_labels = (
            torch.rand([self.batch_size, self.num_labels, self.image_size, self.image_size], device=torch_device) > 0.5
        ).float()
        class_labels = (torch.rand((self.batch_size, self.num_labels), device=torch_device) > 0.5).long()

        config = self.get_config()
        return config, pixel_values, mask_labels, class_labels

    def get_config(self):
        config = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_dim,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
        }
        return EoMTConfig(**config)

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, mask_labels, class_labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class EoMTForUniversalSegmentationTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (EoMTForUniversalSegmentation,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = EoMTForUniversalSegmentationTester(self)
        self.config_tester = ConfigTester(self, config_class=EoMTConfig, has_text_modality=False)
