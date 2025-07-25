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
"""Testing suite for the PyTorch MLCD model."""

import unittest

import requests
from PIL import Image

from transformers import (
    AutoProcessor,
    MLCDVisionConfig,
    MLCDVisionModel,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor


if is_torch_available():
    import torch


class MLCDVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
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
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in MLCD, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return MLCDVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = MLCDVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class MLCDVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `MLCDVisionModel`.
    """

    all_model_classes = (MLCDVisionModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    test_resize_embeddings = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = MLCDVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MLCDVisionConfig, has_text_modality=False)

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (torch.nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, torch.nn.Linear))

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad and "class_pos_emb" not in name:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


@require_torch
class MLCDVisionModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "DeepGlint-AI/mlcd-vit-bigG-patch14-448"
        model = MLCDVisionModel.from_pretrained(model_name).to(torch_device)
        processor = AutoProcessor.from_pretrained(model_name)

        # process single image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(images=image, return_tensors="pt")

        # move inputs to the same device as the model
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        last_hidden_state = outputs.last_hidden_state
        last_attention = outputs.attentions[-1]

        # verify the shapes of last_hidden_state and last_attention
        self.assertEqual(
            last_hidden_state.shape,
            torch.Size([1, 1025, 1664]),
        )
        self.assertEqual(
            last_attention.shape,
            torch.Size([1, 16, 1025, 1025]),
        )

        # verify the values of last_hidden_state and last_attention
        # fmt: off
        expected_partial_5x5_last_hidden_state = torch.tensor(
            [
                [-0.8978, -0.1181,  0.4769,  0.4761, -0.5779],
                [ 0.2640, -2.6150,  0.4853,  0.5743, -1.1003],
                [ 0.3314, -0.3328, -0.4305, -0.1874, -0.7701],
                [-1.5174, -1.0238, -1.1854,  0.1749, -0.8786],
                [ 0.2323, -0.8346, -0.9680, -0.2951,  0.0867],
            ]
        ).to(torch_device)

        expected_partial_5x5_last_attention = torch.tensor(
            [
                [2.0930e-01, 6.3073e-05, 1.4717e-03, 2.6881e-05, 3.0513e-03],
                [1.5828e-04, 2.1056e-03, 4.6784e-04, 1.8276e-03, 5.3233e-04],
                [5.7824e-04, 1.1446e-03, 1.3854e-03, 1.1775e-03, 1.2750e-03],
                [9.6343e-05, 1.6365e-03, 2.9066e-04, 3.1089e-03, 2.0607e-04],
                [6.2688e-04, 1.1656e-03, 1.5030e-03, 8.2819e-04, 2.6992e-03],
            ]
        ).to(torch_device)
        # fmt: on

        torch.testing.assert_close(
            last_hidden_state[0, :5, :5], expected_partial_5x5_last_hidden_state, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            last_attention[0, 0, :5, :5], expected_partial_5x5_last_attention, rtol=1e-4, atol=1e-4
        )
