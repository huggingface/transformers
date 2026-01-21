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
"""Testing suite for the PyTorch DINOv3 model."""

import unittest
from functools import cached_property

from transformers import DINOv3ViTConfig
from transformers.testing_utils import (
    require_torch,
    require_torch_large_accelerator,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import DINOv3ViTBackbone, DINOv3ViTForImageClassification, DINOv3ViTModel


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class DINOv3ViTModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=False,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_register_tokens=2,
        mask_ratio=0.5,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_register_tokens = num_register_tokens
        self.scope = scope

        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1 + self.num_register_tokens
        self.mask_ratio = mask_ratio
        self.num_masks = int(mask_ratio * self.seq_length)
        self.mask_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DINOv3ViTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            num_register_tokens=self.num_register_tokens,
            stage_names=["embeddings"] + [f"stage{i}" for i in range(1, self.num_hidden_layers + 1)],
            out_indices=[0, 1],
            reshape_hidden_states=True,
        )

    def create_and_check_backbone(self, config, pixel_values, labels):
        config.out_features = ["stage1", "stage2"]
        config.reshape_hidden_states = True

        model = DINOv3ViTBackbone(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(pixel_values)

        self.parent.assertEqual(len(outputs.feature_maps), 2)
        for fm in outputs.feature_maps:
            b, c, h, w = fm.shape
            self.parent.assertEqual(b, self.batch_size)
            self.parent.assertEqual(c, self.hidden_size)
            self.parent.assertGreater(h, 0)
            self.parent.assertGreater(w, 0)

    def test_output_hidden_states(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**inputs_dict, output_hidden_states=True)

            self.assertIsNotNone(outputs.hidden_states)
            expected_num_hidden_states = config.num_hidden_layers + 1
            self.assertEqual(len(outputs.hidden_states), expected_num_hidden_states)

            for hidden_state in outputs.hidden_states:
                expected_shape = (
                    self.model_tester.batch_size,
                    self.model_tester.seq_length,
                    self.model_tester.hidden_size,
                )
                self.assertEqual(hidden_state.shape, expected_shape)

    def create_and_check_model(self, config, pixel_values, labels):
        model = DINOv3ViTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        torch_device_override = "cpu"  # Required, or else VRAM is not enough.
        config.device_map = torch_device_override
        model = DINOv3ViTForImageClassification(config)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1

        model = DINOv3ViTForImageClassification(config)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size]).to(torch_device_override)
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class Dinov3ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Dinov3 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (DINOv3ViTModel, DINOv3ViTBackbone, DINOv3ViTForImageClassification) if is_torch_available() else ()
    )
    pipeline_model_mapping = (
        {
            "image-feature-extraction": DINOv3ViTModel,
        }
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False
    test_torch_exportable = True
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = DINOv3ViTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DINOv3ViTConfig, has_text_modality=False, hidden_size=37)

    def test_backbone(self):
        config, pixel_values, labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(config, pixel_values, labels)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Dinov3 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Dinov3 does not support feedforward chunking yet")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        model = DINOv3ViTModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_model_for_image_classification_from_pretrained(self):
        model_name = "dimidagd/dinov3-vit7b16-pretrain-lvd1689m-imagenet1k-lc"
        model = DINOv3ViTForImageClassification.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class DINOv3ViTModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
            if is_vision_available()
            else None
        )

    @require_torch_large_accelerator
    @slow
    def test_inference_lc_head_imagenet(self):
        torch_device_override = "cpu"
        model = DINOv3ViTForImageClassification.from_pretrained(
            "dimidagd/dinov3-vit7b16-pretrain-lvd1689m-imagenet1k-lc", device_map=torch_device_override
        )

        ground_truth_class_imagenet1 = "tabby, tabby cat"
        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device_override)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Verify logits
        expected_logits = torch.tensor([-1.0708860159, -0.7589257956, -1.1738269329, -0.9263097048, -1.0259437561]).to(
            torch_device_override
        )

        torch.testing.assert_close(outputs.logits[0, : len(expected_logits)], expected_logits, rtol=1e-4, atol=1e-4)

        # Test correct class prediction
        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_class_str = model.config.id2label[predicted_class_idx]

        self.assertEqual(predicted_class_str, ground_truth_class_imagenet1)

    @slow
    def test_inference_no_head(self):
        model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the last hidden states
        # in DINOv3 with Registers, the seq length equals the number of patches + 1 + num_register_tokens (we add 1 for the [CLS] token)
        _, _, height, width = inputs["pixel_values"].shape
        num_patches = (height // model.config.patch_size) * (width // model.config.patch_size)
        expected_seq_length = num_patches + 1 + model.config.num_register_tokens
        expected_shape = torch.Size((1, expected_seq_length, model.config.hidden_size))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        last_layer_cls_token = outputs.pooler_output
        expected_slice = torch.tensor([0.4637, -0.4160, 0.4086, -0.1265, -0.2865], device=torch_device)
        torch.testing.assert_close(last_layer_cls_token[0, :5], expected_slice, rtol=1e-4, atol=1e-4)

        last_layer_patch_tokens = outputs.last_hidden_state[:, model.config.num_register_tokens + 1 :]
        expected_slice = torch.tensor([-0.0386, -0.2509, -0.0161, -0.4556, 0.5716], device=torch_device)
        torch.testing.assert_close(last_layer_patch_tokens[0, 0, :5], expected_slice, rtol=1e-4, atol=1e-4)
