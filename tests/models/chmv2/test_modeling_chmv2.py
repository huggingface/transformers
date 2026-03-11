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
"""Testing suite for the PyTorch CHMv2 model."""

import unittest

import requests

from transformers import CHMv2Config
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import CHMv2ForDepthEstimation
    from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig

if is_vision_available():
    from PIL import Image

    from transformers import CHMv2ImageProcessorFast


class CHMv2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=32,
        patch_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        out_indices=(1, 2),
        reassemble_hidden_size=32,
        reassemble_factors=(4, 2),
        post_process_channels=(16, 16),
        fusion_hidden_size=16,
        head_hidden_size=16,
        number_output_channels=4,
        readout_type="project",
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.out_indices = out_indices
        self.reassemble_hidden_size = reassemble_hidden_size
        self.reassemble_factors = reassemble_factors
        self.post_process_channels = post_process_channels
        self.fusion_hidden_size = fusion_hidden_size
        self.head_hidden_size = head_hidden_size
        self.number_output_channels = number_output_channels
        self.readout_type = readout_type
        self.is_training = is_training
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def get_config(self):
        backbone_config = DINOv3ViTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_register_tokens=0,
            key_bias=True,
            out_indices=list(self.out_indices),
            apply_layernorm=True,
            reshape_hidden_states=True,
            layer_norm_eps=1e-6,
            return_class_token=True,
        )
        return CHMv2Config(
            backbone_config=backbone_config,
            patch_size=self.patch_size,
            reassemble_hidden_size=self.reassemble_hidden_size,
            reassemble_factors=list(self.reassemble_factors),
            post_process_channels=list(self.post_process_channels),
            fusion_hidden_size=self.fusion_hidden_size,
            head_hidden_size=self.head_hidden_size,
            number_output_channels=self.number_output_channels,
            readout_type=self.readout_type,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_for_depth_estimation(self, config, pixel_values):
        model = CHMv2ForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))


@require_torch
class CHMv2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CHMv2ForDepthEstimation,) if is_torch_available() else ()
    pipeline_model_mapping = {"depth-estimation": CHMv2ForDepthEstimation} if is_torch_available() else {}

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = CHMv2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CHMv2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CHMv2 does not have a base model and hence no token input_embeddings (nn.Embedding)")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        """CHMv2 uses patch (convolutional) embeddings, not token embeddings."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            # Patch embeddings are nn.Module (Conv2d), not nn.Embedding
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_for_depth_estimation(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(config, pixel_values)

    @unittest.skip(reason="CHMv2 does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="CHMv2 does not support training yet")
    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        pass


@require_torch
@require_vision
@slow
class CHMv2IntegrationTest(unittest.TestCase):
    def test_inference_depth_estimation(self):
        processor = CHMv2ImageProcessorFast.from_pretrained(
            "facebook/dinov3-vitl16-chmv2-dpt-head", revision="refs/pr/1"
        )
        model = CHMv2ForDepthEstimation.from_pretrained(
            "facebook/dinov3-vitl16-chmv2-dpt-head", revision="refs/pr/1"
        ).to(torch_device)

        img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/chmv2_example.tif"
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

        inputs = processor(images=raw_image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape = torch.Size([1, 448, 448])
        self.assertEqual(outputs.predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.1028, 0.0562, 0.0575], [0.4136, 0.5476, 0.4333], [1.8045, 2.3640, 1.6928]]
        ).to(torch_device)
        print(outputs.predicted_depth[0, :3, :3])
        print(expected_slice)
        torch.testing.assert_close(outputs.predicted_depth[0, :3, :3], expected_slice, atol=5e-3, rtol=5e-3)

        # post-processing: without target_sizes keeps the model's native output resolution
        depth = processor.post_process_depth_estimation(outputs)[0]["predicted_depth"]
        self.assertEqual(depth.shape, torch.Size([448, 448]))

        # post-processing: with target_sizes resizes to the original image dimensions
        depth_resized = processor.post_process_depth_estimation(
            outputs, target_sizes=[(raw_image.height, raw_image.width)]
        )[0]["predicted_depth"]
        self.assertEqual(depth_resized.shape, torch.Size([raw_image.height, raw_image.width]))
