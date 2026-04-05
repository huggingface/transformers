# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch SAM 3.1 video model."""

import gc
import tempfile
import unittest

from transformers.testing_utils import (
    backend_empty_cache,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
    require_torch,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import Sam3_1VideoConfig, Sam3_1VideoModel, Sam3_1ViTModel


class Sam3_1VideoModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        num_channels=3,
        image_size=112,
        patch_size=14,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_multimask_outputs=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_multimask_outputs = num_multimask_outputs

    def get_config(self, image_size=None):
        image_size = self.image_size if image_size is None else image_size
        patch_grid_size = image_size // self.patch_size

        return Sam3_1VideoConfig(
            vision_config={
                "model_type": "sam3_vision_model",
                "backbone_config": {
                    "model_type": "sam3_vit_model",
                    "hidden_size": self.hidden_size,
                    "num_hidden_layers": self.num_hidden_layers,
                    "num_attention_heads": self.num_attention_heads,
                    "intermediate_size": self.intermediate_size,
                    "image_size": image_size,
                    "patch_size": self.patch_size,
                    "window_size": patch_grid_size,
                    "global_attn_indexes": list(range(self.num_hidden_layers)),
                },
                "fpn_hidden_size": self.hidden_size,
                "scale_factors": [4.0, 2.0, 1.0],
            },
            prompt_encoder_config={
                "hidden_size": self.hidden_size,
                "image_size": image_size,
                "patch_size": self.patch_size,
            },
            mask_decoder_config={
                "hidden_size": self.hidden_size,
                "mlp_dim": self.intermediate_size,
                "num_hidden_layers": 2,
                "num_attention_heads": self.num_attention_heads,
                "iou_head_hidden_dim": self.hidden_size,
                "num_multimask_outputs": self.num_multimask_outputs,
            },
            image_size=image_size,
            num_maskmem=3,
            memory_attention_hidden_size=self.hidden_size,
            memory_attention_num_layers=2,
            memory_attention_num_attention_heads=1,
            memory_attention_feed_forward_hidden_size=self.intermediate_size,
            memory_encoder_hidden_size=self.hidden_size,
            memory_encoder_output_channels=self.hidden_size // 2,
            mask_downsampler_embed_dim=self.hidden_size,
            memory_fuser_embed_dim=self.hidden_size,
            memory_fuser_intermediate_dim=self.intermediate_size,
        )

    def prepare_config_and_inputs(self, image_size=None):
        image_size = self.image_size if image_size is None else image_size
        config = self.get_config(image_size=image_size)
        pixel_values = torch.randn(self.batch_size, self.num_channels, image_size, image_size)
        input_points = torch.tensor(
            [[[[0.15 * image_size, 0.30 * image_size], [0.60 * image_size, 0.70 * image_size]]]],
            dtype=torch.float32,
        )
        input_labels = torch.tensor([[[1, 0]]], dtype=torch.int64)
        input_boxes = torch.tensor(
            [[[0.10 * image_size, 0.15 * image_size, 0.85 * image_size, 0.90 * image_size]]],
            dtype=torch.float32,
        )
        return config, pixel_values, input_points, input_labels, input_boxes


@require_torch
class Sam3_1VideoModelTest(unittest.TestCase):
    all_model_classes = (Sam3_1VideoModel, Sam3_1ViTModel) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Sam3_1VideoModelTester(self)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_config_round_trip(self):
        config = self.model_tester.get_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            reloaded_config = Sam3_1VideoConfig.from_pretrained(tmp_dir)

        self.assertEqual(reloaded_config.image_size, config.image_size)
        self.assertEqual(reloaded_config.image_mean, config.image_mean)
        self.assertEqual(reloaded_config.image_std, config.image_std)
        self.assertEqual(
            reloaded_config.vision_config.backbone_feature_sizes, config.vision_config.backbone_feature_sizes
        )

    def test_model_forward_with_point_prompts_and_propagation_head(self):
        config, pixel_values, input_points, input_labels, _ = self.model_tester.prepare_config_and_inputs()
        model = Sam3_1VideoModel(config).to(torch_device)
        model.eval()

        low_res_size = 4 * (config.image_size // config.vision_config.backbone_config.patch_size)

        with torch.no_grad():
            outputs = model(
                pixel_values.to(torch_device),
                input_points=input_points.to(torch_device),
                input_labels=input_labels.to(torch_device),
                run_propagation_head=True,
            )

        self.assertEqual(outputs.interactive_pred_masks.shape, (1, 1, low_res_size, low_res_size))
        self.assertEqual(outputs.interactive_high_res_masks.shape, (1, 1, config.image_size, config.image_size))
        self.assertEqual(outputs.interactive_iou_scores.shape, (1, 1, config.num_multimask_outputs))
        self.assertEqual(outputs.interactive_object_pointer.shape, (1, 1, self.model_tester.hidden_size))
        self.assertEqual(
            outputs.propagation_masks.shape,
            (1, 1, config.num_multimask_outputs, low_res_size, low_res_size),
        )
        self.assertEqual(len(outputs.backbone_outputs.interactive["backbone_fpn"]), 3)
        self.assertEqual(
            outputs.backbone_outputs.interactive["vision_features"].shape, (1, self.model_tester.hidden_size, 8, 8)
        )
        self.assertEqual(
            outputs.backbone_outputs.interactive["backbone_fpn"][0].shape[-2:], (low_res_size, low_res_size)
        )

    def test_model_forward_with_box_prompts(self):
        config, pixel_values, _, _, input_boxes = self.model_tester.prepare_config_and_inputs()
        model = Sam3_1VideoModel(config).to(torch_device)
        model.eval()

        low_res_size = 4 * (config.image_size // config.vision_config.backbone_config.patch_size)

        with torch.no_grad():
            outputs = model(
                pixel_values.to(torch_device),
                input_boxes=input_boxes.to(torch_device),
                multimask_output=False,
            )

        self.assertEqual(outputs.interactive_pred_masks.shape, (1, 1, low_res_size, low_res_size))
        self.assertEqual(outputs.interactive_iou_scores.shape, (1, 1, 1))
        self.assertIsNone(outputs.propagation_masks)

    def test_model_save_and_reload(self):
        config, pixel_values, input_points, input_labels, _ = self.model_tester.prepare_config_and_inputs()
        model = Sam3_1VideoModel(config).eval()

        with torch.no_grad():
            original_outputs = model(
                pixel_values,
                input_points=input_points,
                input_labels=input_labels,
                run_propagation_head=True,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            reloaded_model = Sam3_1VideoModel.from_pretrained(tmp_dir).eval()

            with torch.no_grad():
                reloaded_outputs = reloaded_model(
                    pixel_values,
                    input_points=input_points,
                    input_labels=input_labels,
                    run_propagation_head=True,
                )

        torch.testing.assert_close(original_outputs.interactive_pred_masks, reloaded_outputs.interactive_pred_masks)
        torch.testing.assert_close(original_outputs.propagation_masks, reloaded_outputs.propagation_masks)

    def test_custom_image_size(self):
        custom_image_size = 140
        config, pixel_values, input_points, input_labels, _ = self.model_tester.prepare_config_and_inputs(
            image_size=custom_image_size
        )
        model = Sam3_1VideoModel(config).to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(
                pixel_values.to(torch_device),
                input_points=input_points.to(torch_device),
                input_labels=input_labels.to(torch_device),
                run_propagation_head=True,
            )

        self.assertEqual(config.vision_config.backbone_feature_sizes, [[40, 40], [20, 20], [10, 10]])
        self.assertEqual(outputs.interactive_high_res_masks.shape, (1, 1, custom_image_size, custom_image_size))
        self.assertEqual(
            outputs.backbone_outputs.interactive["vision_features"].shape, (1, self.model_tester.hidden_size, 10, 10)
        )

    def test_requires_prompt_inputs(self):
        config, pixel_values, _, _, _ = self.model_tester.prepare_config_and_inputs()
        model = Sam3_1VideoModel(config)

        with self.assertRaisesRegex(ValueError, "At least one prompt input is required"):
            model(pixel_values)

    def test_vit_backbone_forward(self):
        config, pixel_values, _, _, _ = self.model_tester.prepare_config_and_inputs()
        model = Sam3_1ViTModel(config.vision_config.backbone_config).to(torch_device)
        model.eval()

        patch_grid_size = config.image_size // config.vision_config.backbone_config.patch_size

        with torch.no_grad():
            outputs = model(pixel_values.to(torch_device))

        self.assertEqual(
            outputs.last_hidden_state.shape,
            (self.model_tester.batch_size, patch_grid_size * patch_grid_size, self.model_tester.hidden_size),
        )

    def test_inference_with_supported_low_precision_dtypes(self):
        supported_dtypes = []
        if is_torch_fp16_available_on_device(torch_device):
            supported_dtypes.append(torch.float16)
        if is_torch_bf16_available_on_device(torch_device):
            supported_dtypes.append(torch.bfloat16)

        if not supported_dtypes:
            self.skipTest(f"No low-precision dtype support on {torch_device}.")

        config, pixel_values, input_points, input_labels, _ = self.model_tester.prepare_config_and_inputs()

        for dtype in supported_dtypes:
            model = Sam3_1VideoModel(config).to(torch_device, dtype=dtype)
            model.eval()
            with torch.no_grad():
                outputs = model(
                    pixel_values.to(torch_device, dtype=dtype),
                    input_points=input_points.to(torch_device),
                    input_labels=input_labels.to(torch_device),
                    multimask_output=False,
                )
            self.assertEqual(outputs.interactive_pred_masks.dtype, dtype)
