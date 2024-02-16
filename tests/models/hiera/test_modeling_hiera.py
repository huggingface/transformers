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
""" Testing suite for the PyTorch Hiera model. """

import unittest

from transformers import HieraConfig
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    from transformers import HieraModel
    # Assuming HIERA_PRETRAINED_MODEL_ARCHIVE_LIST is defined somewhere for your model
    from transformers.models.hiera.configuration_hiera import HIERA_PRETRAINED_MODEL_ARCHIVE_LIST


class HieraModelTester:
    # Define this tester to initialize Hiera model and its configurations for testing
    def __init__(
        self,
        parent,
        batch_size=8,
        num_channels=3,
        image_size=224,
        # Add other model-specific parameters here
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        # Initialize other necessary attributes here

    def prepare_config_and_inputs(self):
        # Prepare configuration and inputs for testing your model
        pixel_values = torch.rand((self.batch_size, self.num_channels, self.image_size, self.image_size), device=torch_device)

        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return HieraConfig(
            # Define necessary configuration parameters here
        )

    def create_and_check_model(self, config, pixel_values):
        model = HieraModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values=pixel_values)
        # Perform checks here, e.g., output shapes, etc.
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.num_attention_heads, self.seq_length, self.hidden_size))


@require_torch
class HieraModelTest(unittest.TestCase):

    def setUp(self):
        self.model_tester = HieraModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in HIERA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = HieraModel.from_pretrained(model_name)
            self.assertIsNotNone(model)