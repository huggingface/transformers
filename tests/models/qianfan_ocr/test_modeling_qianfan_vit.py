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
"""Testing suite for the PyTorch QianfanViT model."""

import unittest

from transformers import QianfanViTConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import QianfanViTModel


class QianfanViTModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=64,
        patch_size=4,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_mean_pooling=True,
        use_absolute_position_embeddings=True,
        drop_path_rate=0.0,
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
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_mean_pooling = use_mean_pooling
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.drop_path_rate = drop_path_rate

        # seq_length = num_patches + 1 (CLS token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1
        self.num_masks = num_patches // 2

    def get_config(self):
        return QianfanViTConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            use_mean_pooling=self.use_mean_pooling,
            use_absolute_position_embeddings=self.use_absolute_position_embeddings,
            drop_path_rate=self.drop_path_rate,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_model(self, config, pixel_values):
        model = QianfanViTModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )


@require_torch
class QianfanViTModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (QianfanViTModel,) if is_torch_available() else ()
    test_resize_embeddings = False
    test_head_masking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = QianfanViTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=QianfanViTConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, pixel_values)

    @unittest.skip(reason="QianfanViTModel has no text modality and no input_ids")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="QianfanViTModel has no text modality and no input_ids")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="FlashAttention only supports fp16 and bf16")
    def test_flash_attn_2_fp32_ln(self):
        pass
