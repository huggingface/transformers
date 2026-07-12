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
"""Testing suite for the PyTorch LingBot-Vision model."""

import unittest

from transformers import AutoBackbone, AutoModel, AutoModelForImageClassification, LingbotVisionConfig
from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import floats_tensor, ids_tensor


if is_torch_available():
    from transformers import LingbotVisionBackbone, LingbotVisionForImageClassification, LingbotVisionModel


class LingbotVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_size=16,
        patch_size=4,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=2.0,
        num_storage_tokens=2,
        num_labels=7,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.num_storage_tokens = num_storage_tokens
        self.num_labels = num_labels
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_length = self.num_patches + 1 + num_storage_tokens

    def get_config(self):
        return LingbotVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_ratio=self.mlp_ratio,
            num_storage_tokens=self.num_storage_tokens,
            rope_dtype="fp32",
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = ids_tensor([self.batch_size], self.num_labels)
        return self.get_config(), pixel_values, labels


@require_torch
class LingbotVisionModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = LingbotVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LingbotVisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, pixel_values, _ = self.model_tester.prepare_config_and_inputs()
        model = LingbotVisionModel(config).to(torch_device)
        model.eval()

        result = model(pixel_values)

        self.assertEqual(
            result.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size),
        )
        self.assertEqual(result.pooler_output.shape, (self.model_tester.batch_size, self.model_tester.hidden_size))

    def test_model_with_output_hidden_states_and_attentions(self):
        config, pixel_values, _ = self.model_tester.prepare_config_and_inputs()
        model = LingbotVisionModel(config).to(torch_device)
        model.eval()

        result = model(pixel_values, output_hidden_states=True, output_attentions=True)

        self.assertEqual(len(result.hidden_states), self.model_tester.num_hidden_layers + 1)
        self.assertEqual(len(result.attentions), self.model_tester.num_hidden_layers)
        self.assertEqual(
            result.attentions[0].shape,
            (
                self.model_tester.batch_size,
                self.model_tester.num_attention_heads,
                self.model_tester.seq_length,
                self.model_tester.seq_length,
            ),
        )

    def test_backbone(self):
        config, pixel_values, _ = self.model_tester.prepare_config_and_inputs()
        model = LingbotVisionBackbone(config).to(torch_device)
        model.eval()

        result = model(pixel_values)

        expected_size = self.model_tester.image_size // self.model_tester.patch_size
        self.assertEqual(len(result.feature_maps), len(config.out_features))
        self.assertEqual(
            result.feature_maps[0].shape,
            (self.model_tester.batch_size, self.model_tester.hidden_size, expected_size, expected_size),
        )

    def test_image_classification(self):
        config, pixel_values, labels = self.model_tester.prepare_config_and_inputs()
        model = LingbotVisionForImageClassification(config).to(torch_device)
        model.eval()

        result = model(pixel_values, labels=labels)

        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))
        self.assertIsNotNone(result.loss)

    def test_auto_classes(self):
        config = self.model_tester.get_config()

        self.assertIsInstance(AutoModel.from_config(config), LingbotVisionModel)
        self.assertIsInstance(AutoBackbone.from_config(config), LingbotVisionBackbone)
        self.assertIsInstance(AutoModelForImageClassification.from_config(config), LingbotVisionForImageClassification)
