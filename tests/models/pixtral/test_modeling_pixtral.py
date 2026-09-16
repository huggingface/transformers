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
    logging,
)
from transformers.testing_utils import (
    CaptureLogger,
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

    test_resize_embeddings = False
    test_torch_exportable = False  # data-dependent vision placeholder mask

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

    def test_vision_axial_rope(self):
        # override -> the freqs are `//2` of head dim for this model

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        rope_class = None
        base_model = PixtralVisionModel(config)
        for name, module in base_model.named_modules():
            if hasattr(module, "compute_axial_rope_parameters"):
                rope_class = type(module)
                vision_config = module.config
                break

        if rope_class is None:
            self.skipTest("Couldn't infer RoPE layer for this model class.")

        # First make sure that validation on default config raises no rope-related warnings
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("", cl.out)
        logger.warning_once.cache_clear()

        # Axial rope type expects only `rope_theta`, otherwise raises warning
        vision_config.rope_parameters["factor"] = 0.25
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("Unrecognized keys in `rope_parameters` for 'rope_type'='axial': {'factor'}\n", cl.out)
        del vision_config.rope_parameters["factor"]
        logger.warning_once.cache_clear()

        inv_freq, attention_scale = rope_class.compute_axial_rope_parameters(config=vision_config)
        rope_module = rope_class(vision_config).to(device=torch_device)

        self.assertTrue(hasattr(rope_module, "inv_freq"))
        self.assertTrue(hasattr(rope_module, "attention_scaling"))
        self.assertEqual(attention_scale, 1.0)  # attention scale is always 1
        torch.testing.assert_close(inv_freq, rope_module.inv_freq.cpu())

        # create 2D position IDs for a single grid of one row and 10 cols `size=(10, 2)`
        position_ids = torch.stack(
            [
                torch.arange(10, dtype=torch.long, device=torch_device),
                torch.zeros(10, dtype=torch.long, device=torch_device),
            ]
        ).transpose(0, 1)
        # and an empty hidden states used only to infer device/dtype
        hidden_states = torch.empty(1, dtype=torch.float32, device=torch_device)
        cos, sin = rope_module(hidden_states, position_ids)
        self.assertEqual(cos.shape[-1], inv_freq.shape[-1] * 2)  # the freq are `//2` of head dim
