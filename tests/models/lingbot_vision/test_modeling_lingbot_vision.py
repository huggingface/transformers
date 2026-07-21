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
from functools import cached_property

from transformers import AutoBackbone, AutoModel, LingbotVisionConfig
from transformers.testing_utils import Expectations, require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import LingbotVisionBackbone, LingbotVisionModel


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor, LingbotVisionImageProcessor


# HF-format checkpoint of the flagship ViT-g/16 backbone (produced by
# `convert_lingbot_vision_to_hf.py`). Used only by the `@slow` integration tests.
LINGBOT_GIANT = "IMvision12/lingbot-vision-vit-giant-hf"


class LingbotVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=16,
        patch_size=4,
        num_channels=3,
        is_training=False,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=2.0,
        num_storage_tokens=2,
        ffn_layer="swiglu",
        norm_layer="layernorm",
        layer_scale_init_value=1e-5,
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
        self.mlp_ratio = mlp_ratio
        self.num_storage_tokens = num_storage_tokens
        self.ffn_layer = ffn_layer
        self.norm_layer = norm_layer
        self.layer_scale_init_value = layer_scale_init_value
        self.initializer_range = initializer_range
        self.scope = scope

        self.num_patches = (image_size // patch_size) ** 2
        self.seq_length = self.num_patches + 1 + self.num_storage_tokens
        self.mask_length = self.num_patches
        self.num_masks = max(1, self.num_patches // 2)

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
            ffn_layer=self.ffn_layer,
            norm_layer=self.norm_layer,
            layer_scale_init_value=self.layer_scale_init_value,
            initializer_range=self.initializer_range,
            rope_dtype="fp32",
            out_indices=[1, 2],
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        return self.get_config(), pixel_values

    def create_and_check_model(self, config, pixel_values):
        model = LingbotVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_backbone(self, config, pixel_values):
        config.out_features = ["stage1", "stage2"]
        config.reshape_hidden_states = True

        model = LingbotVisionBackbone(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(pixel_values)

        expected_size = self.image_size // self.patch_size
        self.parent.assertEqual(len(outputs.feature_maps), 2)
        for feature_map in outputs.feature_maps:
            self.parent.assertEqual(
                feature_map.shape, (self.batch_size, self.hidden_size, expected_size, expected_size)
            )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class LingbotVisionModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    LingBot-Vision is a vision backbone: it does not use input_ids / inputs_embeds / attention_mask, so the
    corresponding common tests are overwritten or skipped.
    """

    all_model_classes = (LingbotVisionModel, LingbotVisionBackbone) if is_torch_available() else ()
    pipeline_model_mapping = {"image-feature-extraction": LingbotVisionModel} if is_torch_available() else {}

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = LingbotVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=LingbotVisionConfig, has_text_modality=False, hidden_size=32
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_auto_classes(self):
        config = self.model_tester.get_config()
        self.assertIsInstance(AutoModel.from_config(config), LingbotVisionModel)
        self.assertIsInstance(AutoBackbone.from_config(config), LingbotVisionBackbone)

    @require_vision
    def test_image_processor_defaults(self):
        image_processor = LingbotVisionImageProcessor()
        self.assertEqual(image_processor.size.height, 512)
        self.assertEqual(image_processor.size.width, 512)
        self.assertEqual(image_processor.image_mean, (0.485, 0.456, 0.406))
        self.assertEqual(image_processor.image_std, (0.229, 0.224, 0.225))

    @unittest.skip(reason="LingBot-Vision does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="LingBot-Vision does not use inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="LingBot-Vision does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = LingbotVisionModel.from_pretrained(LINGBOT_GIANT)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class LingbotVisionModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained(LINGBOT_GIANT) if is_vision_available() else None

    @slow
    def test_inference_no_head(self):
        model = LingbotVisionModel.from_pretrained(LINGBOT_GIANT).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        # seq length = num patches + 1 (CLS) + num storage tokens
        _, _, height, width = inputs["pixel_values"].shape
        num_patches = (height // model.config.patch_size) * (width // model.config.patch_size)
        expected_seq_length = num_patches + 1 + model.config.num_storage_tokens
        expected_shape = torch.Size((1, expected_seq_length, model.config.hidden_size))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_pooler = torch.tensor(
            Expectations({(None, None): [-0.3808, 0.8251, 0.3735, 1.1332, 1.7006]}).get_expectation(),
            device=torch_device,
        )
        torch.testing.assert_close(outputs.pooler_output[0, :5], expected_pooler, rtol=1e-4, atol=1e-4)

        first_patch_token = outputs.last_hidden_state[:, model.config.num_storage_tokens + 1 :]
        expected_patch = torch.tensor(
            Expectations({(None, None): [0.5824, 0.2652, -0.3335, -0.0979, 0.2420]}).get_expectation(),
            device=torch_device,
        )
        torch.testing.assert_close(first_patch_token[0, 0, :5], expected_patch, rtol=1e-4, atol=1e-4)
