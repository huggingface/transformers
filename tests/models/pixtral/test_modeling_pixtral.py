# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Pixtral model."""

import unittest

from transformers import (
    PixtralVisionConfig,
    PixtralVisionModel,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch


class PixtralVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in Pixtral, the seq length equals the number of patches * batch_size because the patches are flattened
        self.seq_length = (image_size // patch_size) ** 2 * batch_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        image_sizes = torch.tensor(
            [[self.image_size, self.image_size]] * self.batch_size, dtype=torch.long, device=torch_device
        )
        config = self.get_config()

        return config, pixel_values, image_sizes

    def get_config(self):
        return PixtralVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, image_sizes = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values, "image_sizes": image_sizes}
        return config, inputs_dict


@require_torch
class PixtralVisionModelModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `PixtralVisionModel`.
    """

    all_model_classes = (PixtralVisionModel,) if is_torch_available() else ()
    additional_model_inputs = ["image_sizes"]
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PixtralVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PixtralVisionConfig, has_text_modality=False)

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (torch.nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, torch.nn.Linear))
