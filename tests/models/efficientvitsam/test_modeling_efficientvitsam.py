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
"""Testing suite for the PyTorch EfficientViT-SAM model."""

import unittest

import numpy as np
import pytest

from transformers import (
    EfficientViTSamConfig,
    EfficientViTSamMaskDecoderConfig,
    EfficientViTSamPromptEncoderConfig,
    EfficientViTSamVisionConfig,
)
from transformers.testing_utils import Expectations, require_torch, require_torchvision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        EfficientViTSamMaskDecoder,
        EfficientViTSamModel,
        EfficientViTSamPromptEncoder,
        EfficientViTSamVisionModel,
    )


if is_vision_available():
    pass


class EfficientViTSamVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=32,
        width_list=None,
        depth_list=None,
        block_list=None,
        expand_list=None,
        fewer_norm_list=None,
        in_channels=3,
        qkv_dim=16,
        norm="bn2d",
        act_func="gelu",
        fid_list=None,
        in_channel_list=None,
        head_width=32,
        head_depth=2,
        expand_ratio=1.0,
        middle_op="fmb",
        out_dim=32,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.width_list = width_list if width_list is not None else [8, 16, 24, 32, 64]
        self.depth_list = depth_list if depth_list is not None else [1, 1, 1, 1, 1]
        self.block_list = block_list if block_list is not None else ["res", "fmb", "fmb", "mb", "att"]
        self.expand_list = expand_list if expand_list is not None else [1.0, 1.0, 1.0, 1.0, 2.0]
        self.fewer_norm_list = fewer_norm_list if fewer_norm_list is not None else [False, False, False, False, True]
        self.in_channels = in_channels
        self.qkv_dim = qkv_dim
        self.norm = norm
        self.act_func = act_func
        self.fid_list = fid_list if fid_list is not None else ["stage4", "stage3", "stage2"]
        self.in_channel_list = in_channel_list if in_channel_list is not None else [64, 32, 24]
        self.head_width = head_width
        self.head_depth = head_depth
        self.expand_ratio = expand_ratio
        self.middle_op = middle_op
        self.out_dim = out_dim
        self.is_training = is_training

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_model(self, config, pixel_values):
        model = EfficientViTSamVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.out_dim, 64, 64))

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.in_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return EfficientViTSamVisionConfig(
            width_list=self.width_list,
            depth_list=self.depth_list,
            block_list=self.block_list,
            expand_list=self.expand_list,
            fewer_norm_list=self.fewer_norm_list,
            in_channels=self.in_channels,
            qkv_dim=self.qkv_dim,
            norm=self.norm,
            act_func=self.act_func,
            fid_list=self.fid_list,
            in_channel_list=self.in_channel_list,
            head_width=self.head_width,
            head_depth=self.head_depth,
            expand_ratio=self.expand_ratio,
            middle_op=self.middle_op,
            out_dim=self.out_dim,
            image_size=self.image_size,
            num_pos_feats=self.out_dim // 2,
            scale=float(self.out_dim // 2),
        )


class SamPromptEncoderTester:
    def __init__(
        self,
        hidden_size=32,
        input_image_size=128,
        patch_size=2,
        mask_input_channels=4,
        num_point_embeddings=4,
        hidden_act="gelu",
    ):
        self.hidden_size = hidden_size
        self.input_image_size = input_image_size
        self.patch_size = patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act

    def get_config(self):
        return EfficientViTSamPromptEncoderConfig(
            image_size=self.input_image_size,
            patch_size=self.patch_size,
            mask_input_channels=self.mask_input_channels,
            hidden_size=self.hidden_size,
            num_point_embeddings=self.num_point_embeddings,
            hidden_act=self.hidden_act,
        )


class SamMaskDecoderTester:
    def __init__(
        self,
        hidden_size=32,
        hidden_act="relu",
        mlp_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=32,
        layer_norm_eps=1e-6,
    ):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_dim = mlp_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.layer_norm_eps = layer_norm_eps

    def get_config(self):
        return EfficientViTSamMaskDecoderConfig(
            hidden_size=self.hidden_size,
            hidden_act=self.hidden_act,
            mlp_dim=self.mlp_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            attention_downsample_rate=self.attention_downsample_rate,
            num_multimask_outputs=self.num_multimask_outputs,
            iou_head_depth=self.iou_head_depth,
            iou_head_hidden_dim=self.iou_head_hidden_dim,
            layer_norm_eps=self.layer_norm_eps,
        )


class EfficientViTSamModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.vision_tester = EfficientViTSamVisionModelTester(parent)
        self.prompt_encoder_tester = SamPromptEncoderTester(hidden_size=32)
        self.mask_decoder_tester = SamMaskDecoderTester(hidden_size=32)

        self.batch_size = self.vision_tester.batch_size
        self.is_training = True

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor(
            [
                self.vision_tester.batch_size,
                self.vision_tester.in_channels,
                self.vision_tester.image_size,
                self.vision_tester.image_size,
            ]
        )
        return config, pixel_values

    def get_config(self):
        return EfficientViTSamConfig(
            vision_config=self.vision_tester.get_config(),
            prompt_encoder_config=self.prompt_encoder_tester.get_config(),
            mask_decoder_config=self.mask_decoder_tester.get_config(),
        )

    def create_and_check_model(self, config, pixel_values):
        model = EfficientViTSamModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(result.iou_scores.shape, (self.vision_tester.batch_size, 1, 3))
        self.parent.assertEqual(result.pred_masks.shape[:3], (self.vision_tester.batch_size, 1, 3))

    def create_and_check_get_image_features(self, config, pixel_values):
        model = EfficientViTSamModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model.get_image_embeddings(pixel_values)
        self.parent.assertEqual(result.shape, (self.vision_tester.batch_size, self.vision_tester.out_dim, 64, 64))

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_prompt_inputs(self):
        input_points = torch.tensor(
            [
                [[[4.0, 8.0], [12.0, 16.0]], [[20.0, 24.0], [28.0, 30.0]]],
                [[[6.0, 10.0], [14.0, 18.0]], [[22.0, 26.0], [27.0, 29.0]]],
            ],
            device=torch_device,
        )
        input_labels = torch.tensor([[[1, 0], [1, -10]], [[0, 1], [1, 1]]], device=torch_device)
        input_boxes = torch.tensor(
            [
                [[2.0, 4.0, 16.0, 20.0], [10.0, 12.0, 28.0, 30.0]],
                [[3.0, 5.0, 17.0, 21.0], [11.0, 13.0, 29.0, 31.0]],
            ],
            device=torch_device,
        )
        return input_points, input_labels, input_boxes


@require_torch
class EfficientViTSamModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (EfficientViTSamModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": EfficientViTSamModel, "mask-generation": EfficientViTSamModel}
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False
    _is_composite = True

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def setUp(self):
        self.model_tester = EfficientViTSamModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EfficientViTSamConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_sub_configs_are_efficientvitsam_configs(self):
        config = self.model_tester.get_config()

        self.assertIsInstance(config.vision_config, EfficientViTSamVisionConfig)
        self.assertIsInstance(config.prompt_encoder_config, EfficientViTSamPromptEncoderConfig)
        self.assertIsInstance(config.mask_decoder_config, EfficientViTSamMaskDecoderConfig)

        config = EfficientViTSamConfig.from_dict(config.to_dict())
        self.assertIsInstance(config.vision_config, EfficientViTSamVisionConfig)
        self.assertIsInstance(config.prompt_encoder_config, EfficientViTSamPromptEncoderConfig)
        self.assertIsInstance(config.mask_decoder_config, EfficientViTSamMaskDecoderConfig)

    @unittest.skip(reason="EfficientViT-SAM does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM does not use input_ids")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM does not support get_input_embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM does not support dynamic attention setting")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM does not support dynamic attention setting")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM does not output standard attention shapes")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM does not output backbone hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM does not output backbone hidden states")
    def test_image_hidden_states(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM does not output backbone hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_model(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, pixel_values)

    def test_get_image_features(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_get_image_features(*config_and_inputs)

    def test_precomputed_image_embeddings_match_pixel_values(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(pixel_values)
            outputs_from_pixels = model(pixel_values=pixel_values)
            outputs_from_embeddings = model(image_embeddings=image_embeddings)

        torch.testing.assert_close(outputs_from_embeddings.iou_scores, outputs_from_pixels.iou_scores)
        torch.testing.assert_close(outputs_from_embeddings.pred_masks, outputs_from_pixels.pred_masks)

    def test_model_runs_on_available_devices(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = EfficientViTSamModel(config).eval()
        reference_outputs = None

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(torch.device("mps"))

        for device in devices:
            with self.subTest(device=device):
                model.to(device)
                with torch.no_grad():
                    outputs = model(pixel_values.to(device))
                self.assertEqual(outputs.pred_masks.device.type, device.type)
                if device.index is not None:
                    self.assertEqual(outputs.pred_masks.device.index, device.index)
                self.assertFalse(torch.isnan(outputs.pred_masks).any())
                if reference_outputs is None:
                    reference_outputs = outputs.pred_masks.cpu()
                else:
                    torch.testing.assert_close(outputs.pred_masks.cpu(), reference_outputs, rtol=1e-3, atol=1e-3)

    def test_model_supports_reduced_precision(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        dtypes = [torch.float32]
        if torch.cuda.is_available():
            dtypes.extend([torch.float16, torch.bfloat16])
        elif torch.device(torch_device).type == "mps":
            dtypes.append(torch.float16)
        else:
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                model = EfficientViTSamModel(config).to(device=torch_device, dtype=dtype).eval()
                with torch.no_grad():
                    outputs = model(pixel_values.to(device=torch_device, dtype=dtype))
                self.assertEqual(outputs.pred_masks.dtype, dtype)
                self.assertTrue(torch.isfinite(outputs.pred_masks).all())

    def test_config_roundtrip_preserves_all_sub_configs(self):
        config = self.model_tester.get_config()
        reloaded = EfficientViTSamConfig.from_dict(config.to_dict())

        self.assertEqual(reloaded.to_dict(), config.to_dict())
        self.assertEqual(reloaded.vision_config.to_dict(), config.vision_config.to_dict())
        self.assertEqual(reloaded.prompt_encoder_config.to_dict(), config.prompt_encoder_config.to_dict())
        self.assertEqual(reloaded.mask_decoder_config.to_dict(), config.mask_decoder_config.to_dict())
        for config_class, sub_config in (
            (EfficientViTSamVisionConfig, config.vision_config),
            (EfficientViTSamPromptEncoderConfig, config.prompt_encoder_config),
            (EfficientViTSamMaskDecoderConfig, config.mask_decoder_config),
        ):
            self.assertEqual(config_class.from_dict(sub_config.to_dict()).to_dict(), sub_config.to_dict())

    @require_torchvision
    @unittest.skipUnless(is_vision_available(), "Vision dependencies are required for processor tests")
    def test_processor_resizes_images_and_normalizes_prompts(self):
        from transformers.models.efficientvitsam.image_processing_efficientvitsam import EfficientViTSamImageProcessor
        from transformers.models.efficientvitsam.processing_efficientvitsam import EfficientViTSamProcessor

        image_processor = EfficientViTSamImageProcessor(
            size={"longest_edge": 32}, pad_size={"height": 32, "width": 32}
        )
        processor = EfficientViTSamProcessor(image_processor=image_processor)
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        inputs = processor(
            images=image,
            input_points=[[[20.0, 10.0]]],
            input_labels=[[1]],
            input_boxes=[[[10.0, 5.0, 30.0, 15.0]]],
            return_tensors="pt",
        )

        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 32, 32))
        torch.testing.assert_close(inputs["input_points"], torch.tensor([[[[16.0, 8.0]]]], dtype=torch.float64))
        torch.testing.assert_close(
            inputs["input_boxes"], torch.tensor([[[8.0, 4.0, 24.0, 12.0]]], dtype=torch.float64)
        )
        self.assertEqual(inputs["input_labels"].tolist(), [[[1]]])

    @require_torchvision
    @unittest.skipUnless(is_vision_available(), "Vision dependencies are required for pipeline tests")
    def test_mask_generation_pipeline(self):
        from transformers import pipeline
        from transformers.models.efficientvitsam.image_processing_efficientvitsam import EfficientViTSamImageProcessor

        config, _ = self.model_tester.prepare_config_and_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()
        image_processor = EfficientViTSamImageProcessor(
            size={"longest_edge": 32}, pad_size={"height": 32, "width": 32}
        )
        generator = pipeline("mask-generation", model=model, image_processor=image_processor, device=torch_device)

        from PIL import Image

        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        output = generator(image, points_per_crop=2, points_per_batch=4)
        self.assertIn("masks", output)
        self.assertIn("scores", output)

    def test_point_and_box_prompts_support_multiple_masks_per_image(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        input_points, input_labels, input_boxes = self.model_tester.prepare_prompt_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
            )

        self.assertEqual(outputs.iou_scores.shape, (self.model_tester.batch_size, 2, 3))
        self.assertEqual(outputs.pred_masks.shape[:3], (self.model_tester.batch_size, 2, 3))

    def test_point_prompts_default_to_foreground_labels(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        input_points, _, _ = self.model_tester.prepare_prompt_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        with torch.no_grad():
            outputs_without_labels = model(pixel_values=pixel_values, input_points=input_points)
            outputs_with_labels = model(
                pixel_values=pixel_values,
                input_points=input_points,
                input_labels=torch.ones(input_points.shape[:-1], device=torch_device, dtype=torch.int),
            )

        torch.testing.assert_close(outputs_without_labels.iou_scores, outputs_with_labels.iou_scores)
        torch.testing.assert_close(outputs_without_labels.pred_masks, outputs_with_labels.pred_masks)

    def test_multimask_output_returns_single_best_mask(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        input_points, input_labels, _ = self.model_tester.prepare_prompt_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_points=input_points,
                input_labels=input_labels,
                multimask_output=False,
            )

        self.assertEqual(outputs.iou_scores.shape, (self.model_tester.batch_size, 2, 1))
        self.assertEqual(outputs.pred_masks.shape[:3], (self.model_tester.batch_size, 2, 1))

    def test_forward_rejects_invalid_prompt_and_image_inputs(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        input_points, _, input_boxes = self.model_tester.prepare_prompt_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        with self.assertRaisesRegex(ValueError, "Either pixel_values or image_embeddings"):
            model()
        with self.assertRaisesRegex(ValueError, "Only one of pixel_values and image_embeddings"):
            model(pixel_values=pixel_values, image_embeddings=model.get_image_embeddings(pixel_values))
        with self.assertRaisesRegex(ValueError, "input_points must be a 4D tensor"):
            model(pixel_values=pixel_values, input_points=input_points[:, 0])
        with self.assertRaisesRegex(ValueError, "input_boxes must be a 3D tensor"):
            model(pixel_values=pixel_values, input_boxes=input_boxes.unsqueeze(1))
        with self.assertRaisesRegex(ValueError, "as many bounding boxes as input points"):
            model(pixel_values=pixel_values, input_points=input_points, input_boxes=input_boxes[:, :1])

    @pytest.mark.torch_compile_test
    def test_prompt_inference_can_be_compiled(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is unavailable")

        config = self.model_tester.get_config()
        prompt_encoder = EfficientViTSamPromptEncoder(config).to(torch_device).eval()
        input_points, input_labels, input_boxes = self.model_tester.prepare_prompt_inputs()
        compiled_prompt_encoder = torch.compile(prompt_encoder, backend="eager", dynamic=False)

        with torch.no_grad():
            eager_sparse, eager_dense = prompt_encoder(input_points, input_labels, input_boxes, None)
            compiled_sparse, compiled_dense = compiled_prompt_encoder(input_points, input_labels, input_boxes, None)

        torch.testing.assert_close(compiled_sparse, eager_sparse)
        torch.testing.assert_close(compiled_dense, eager_dense)

    def test_batched_outputs_match_single_sample_outputs(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        input_points, input_labels, input_boxes = self.model_tester.prepare_prompt_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        with torch.no_grad():
            batched_outputs = model(
                pixel_values=pixel_values,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
            )
            single_outputs = [
                model(
                    pixel_values=pixel_values[i : i + 1],
                    input_points=input_points[i : i + 1],
                    input_labels=input_labels[i : i + 1],
                    input_boxes=input_boxes[i : i + 1],
                )
                for i in range(pixel_values.shape[0])
            ]

        for i, outputs in enumerate(single_outputs):
            torch.testing.assert_close(batched_outputs.iou_scores[i : i + 1], outputs.iou_scores, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(batched_outputs.pred_masks[i : i + 1], outputs.pred_masks, rtol=1e-4, atol=1e-5)

    def test_points_only_single_point_per_object(self):
        """Single foreground point per object — simplest point prompt."""
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        # shape: (batch=2, point_batch=1, num_points=1, 2)
        input_points = torch.tensor([[[[4.0, 8.0]]], [[[6.0, 10.0]]]], device=torch_device)
        input_labels = torch.ones((2, 1, 1), dtype=torch.int, device=torch_device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_points=input_points, input_labels=input_labels)

        self.assertEqual(outputs.iou_scores.shape, (self.model_tester.batch_size, 1, 3))
        self.assertEqual(outputs.pred_masks.shape[:3], (self.model_tester.batch_size, 1, 3))

    def test_points_only_multi_point_per_object(self):
        """Multiple points per object (foreground + background labels)."""
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        # (batch=2, point_batch=1, num_points=3, 2) — mix of fg/bg labels
        input_points = torch.tensor(
            [[[[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]]], [[[6.0, 10.0], [14.0, 18.0], [22.0, 26.0]]]],
            device=torch_device,
        )
        # label 1 = foreground, 0 = background, -1 = not-a-point padding
        input_labels = torch.tensor([[[1, 0, 1]], [[1, 1, 0]]], dtype=torch.int, device=torch_device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_points=input_points, input_labels=input_labels)

        self.assertEqual(outputs.iou_scores.shape, (self.model_tester.batch_size, 1, 3))
        self.assertEqual(outputs.pred_masks.shape[:3], (self.model_tester.batch_size, 1, 3))

    def test_points_only_negative_foreground_label_arrays(self):
        """Explicit negative/foreground label arrays per point prompt group."""
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        input_points = torch.tensor(
            [[[[4.0, 8.0], [12.0, 16.0]]], [[[6.0, 10.0], [14.0, 18.0]]]],
            device=torch_device,
        )
        # -1 labels mark background / not-a-point, 1 marks foreground
        input_labels = torch.tensor([[[-1, 1]], [[1, -1]]], dtype=torch.int, device=torch_device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_points=input_points, input_labels=input_labels)

        self.assertEqual(outputs.iou_scores.shape, (self.model_tester.batch_size, 1, 3))
        self.assertEqual(outputs.pred_masks.shape[:3], (self.model_tester.batch_size, 1, 3))

    def test_boxes_only_single_box_per_image(self):
        """Single bounding box per image in the batch."""
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        # (batch=2, nb_boxes=1, 4)
        input_boxes = torch.tensor(
            [[[2.0, 4.0, 16.0, 20.0]], [[3.0, 5.0, 17.0, 21.0]]],
            device=torch_device,
        )

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes)

        self.assertEqual(outputs.iou_scores.shape, (self.model_tester.batch_size, 1, 3))
        self.assertEqual(outputs.pred_masks.shape[:3], (self.model_tester.batch_size, 1, 3))

    def test_boxes_only_multi_box_batched(self):
        """Multiple bounding boxes per image — batched multi-box inference."""
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        # (batch=2, nb_boxes=2, 4)
        input_boxes = torch.tensor(
            [
                [[2.0, 4.0, 16.0, 20.0], [10.0, 12.0, 28.0, 30.0]],
                [[3.0, 5.0, 17.0, 21.0], [11.0, 13.0, 29.0, 31.0]],
            ],
            device=torch_device,
        )

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_boxes=input_boxes)

        self.assertEqual(outputs.iou_scores.shape, (self.model_tester.batch_size, 2, 3))
        self.assertEqual(outputs.pred_masks.shape[:3], (self.model_tester.batch_size, 2, 3))

    def test_mixed_point_box_mask_prompts(self):
        """All three prompt types (points + boxes + masks) provided simultaneously."""
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        input_points, input_labels, input_boxes = self.model_tester.prepare_prompt_inputs()
        model = EfficientViTSamModel(config).to(torch_device).eval()

        # Low-resolution mask prompt: (batch, 1, img_embed_size*4, img_embed_size*4)
        img_embed_size = config.prompt_encoder_config.image_embedding_size
        input_masks = torch.randn(
            self.model_tester.batch_size, 1, img_embed_size * 4, img_embed_size * 4, device=torch_device
        )

        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                input_masks=input_masks,
            )

        self.assertEqual(outputs.iou_scores.shape, (self.model_tester.batch_size, 2, 3))
        self.assertEqual(outputs.pred_masks.shape[:3], (self.model_tester.batch_size, 2, 3))

    @slow
    def test_inference_numerical_parity(self):
        from huggingface_hub import hf_hub_download
        from PIL import Image

        from transformers.models.efficientvitsam.convert_efficientvitsam_to_hf import get_config, replace_keys
        from transformers.models.efficientvitsam.image_processing_efficientvitsam import EfficientViTSamImageProcessor
        from transformers.models.efficientvitsam.processing_efficientvitsam import EfficientViTSamProcessor

        expected_outputs = {
            "efficientvit-sam-l0": ([0.4383514, 0.6686006, 0.7585137], [-0.4104123, 0.3950768, 0.9315740]),
            "efficientvit-sam-l1": ([0.5030453, 0.8049473, 0.8187212], [-2.2492948, -1.4918246, -1.4300048]),
            "efficientvit-sam-l2": ([0.5292551, 0.7983652, 0.8195354], [-1.6958143, -0.7988551, -0.8736166]),
            "efficientvit-sam-xl0": ([0.2872823, 0.5676863, 0.8392619], [-9.0085440, -6.0312724, -8.8787479]),
            "efficientvit-sam-xl1": ([0.2453989, 0.2956744, 0.6442841], [-15.1235638, -10.6101160, -14.4371719]),
        }

        for model_name, (scores, masks) in expected_outputs.items():
            with self.subTest(model_name=model_name):
                checkpoint_path = hf_hub_download("mit-han-lab/efficientvit-sam", model_name.replace("-", "_") + ".pt")
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                state_dict = state_dict.get("state_dict", state_dict)

                model = EfficientViTSamModel(get_config(model_name))
                model.load_state_dict(replace_keys(state_dict, model_name), strict=True)
                model.to(torch_device).eval()

                image_size = 1024 if "xl" in model_name else 512
                processor = EfficientViTSamProcessor(
                    EfficientViTSamImageProcessor(
                        size={"longest_edge": image_size}, pad_size={"height": image_size, "width": image_size}
                    )
                )
                inputs = processor(
                    images=Image.new("RGB", (64, 64), color=(127, 63, 31)),
                    input_points=[[[32, 32]]],
                    return_tensors="pt",
                ).to(torch_device)
                with torch.no_grad():
                    outputs = model(**inputs)

                expected_scores = Expectations({(None, None): scores})
                expected_masks = Expectations({(None, None): masks})
                torch.testing.assert_close(
                    outputs.iou_scores.flatten(),
                    torch.tensor(expected_scores.get_expectation(), device=torch_device),
                    rtol=2e-4,
                    atol=2e-4,
                )
                try:
                    torch.testing.assert_close(
                        outputs.pred_masks.flatten()[:3],
                        torch.tensor(expected_masks.get_expectation(), device=torch_device),
                        rtol=2e-4,
                        atol=2e-4,
                    )
                except AssertionError as error:
                    raise AssertionError(
                        f"{model_name}: actual={outputs.pred_masks.flatten()[:3].tolist()}, {error}"
                    ) from error


@require_torch
class EfficientViTSamVisionModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (EfficientViTSamVisionModel,) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = EfficientViTSamVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EfficientViTSamVisionConfig, has_text_modality=False)

    def test_config(self):
        config = self.model_tester.get_config()
        reloaded_config = EfficientViTSamVisionConfig.from_dict(config.to_dict())

        self.assertEqual(reloaded_config.to_dict(), config.to_dict())
        self.assertEqual(reloaded_config.width_list, self.model_tester.width_list)
        self.assertEqual(reloaded_config.in_channel_list, self.model_tester.in_channel_list)

    @unittest.skip(reason="EfficientViT-SAM's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM's vision encoder does not support get_input_embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="EfficientViT-SAM's vision encoder does not output attention weights")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM's vision encoder does not output hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="EfficientViT-SAM's vision encoder does not output hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_vision_encoder_stage_and_block_structures(self):
        config = self.model_tester.get_config()
        config.width_list = [4, 8, 12, 16, 32]
        config.depth_list = [1, 1, 1, 1, 1]
        config.block_list = ["res", "fmb", "fmb", "mb", "att"]
        config.expand_list = [1.0, 1.0, 1.0, 1.0, 2.0]
        config.fewer_norm_list = [False, False, False, False, True]
        config.in_channel_list = [32, 16, 12]

        model = EfficientViTSamVisionModel(config=config).to(torch_device).eval()
        pixel_values = floats_tensor([1, 3, config.image_size, config.image_size]).to(torch_device)
        with torch.no_grad():
            output = model(pixel_values)
        self.assertEqual(output.last_hidden_state.shape, (1, config.out_dim, 64, 64))

    def test_vision_encoder_feature_extraction_dimensions(self):
        config = self.model_tester.get_config()
        model = EfficientViTSamVisionModel(config=config).to(torch_device).eval()
        for batch_size in [1, 2]:
            for img_size in [32, 64]:
                pixel_values = floats_tensor([batch_size, 3, img_size, img_size]).to(torch_device)
                with torch.no_grad():
                    output = model(pixel_values)
                self.assertEqual(output.last_hidden_state.shape, (batch_size, config.out_dim, 64, 64))

    def test_vision_encoder_norm_and_act_flexibility(self):
        for norm in ["bn2d", "ln2d"]:
            for act_func in ["gelu", "relu"]:
                config = self.model_tester.get_config()
                config.norm = norm
                config.act_func = act_func
                model = EfficientViTSamVisionModel(config=config).to(torch_device).eval()
                pixel_values = floats_tensor([1, 3, config.image_size, config.image_size]).to(torch_device)
                with torch.no_grad():
                    output = model(pixel_values)
                self.assertEqual(output.last_hidden_state.shape, (1, config.out_dim, 64, 64))


@require_torch
class EfficientViTSamPromptEncoderTest(unittest.TestCase):
    def setUp(self):
        self.vision_tester = EfficientViTSamVisionModelTester(self)
        self.prompt_encoder_tester = SamPromptEncoderTester(hidden_size=32)
        self.config = EfficientViTSamConfig(
            vision_config=self.vision_tester.get_config(),
            prompt_encoder_config=self.prompt_encoder_tester.get_config(),
        )
        self.prompt_encoder = EfficientViTSamPromptEncoder(self.config).to(torch_device).eval()

    def test_sparse_embedding_generation(self):
        batch_size = 2
        num_queries = 2
        num_points = 3
        hidden_size = self.config.prompt_encoder_config.hidden_size

        # 1. Points only (with pad=True when input_boxes is None)
        input_points = torch.randint(
            0, 100, (batch_size, num_queries, num_points, 2), dtype=torch.float, device=torch_device
        )
        input_labels = torch.ones((batch_size, num_queries, num_points), dtype=torch.int, device=torch_device)
        sparse_embeds, _ = self.prompt_encoder(
            input_points=input_points, input_labels=input_labels, input_boxes=None, input_masks=None
        )
        self.assertEqual(sparse_embeds.shape, (batch_size, num_queries, num_points + 1, hidden_size))

        # 2. Boxes only
        input_boxes = torch.tensor(
            [[[2.0, 4.0, 16.0, 20.0], [10.0, 12.0, 28.0, 30.0]], [[3.0, 5.0, 17.0, 21.0], [11.0, 13.0, 29.0, 31.0]]],
            device=torch_device,
        )
        sparse_embeds_box, _ = self.prompt_encoder(
            input_points=None, input_labels=None, input_boxes=input_boxes, input_masks=None
        )
        self.assertEqual(sparse_embeds_box.shape, (batch_size, num_queries, 2, hidden_size))

        # 3. Combined Points + Boxes (pad=False when input_boxes is provided)
        sparse_embeds_combined, _ = self.prompt_encoder(
            input_points=input_points, input_labels=input_labels, input_boxes=input_boxes, input_masks=None
        )
        self.assertEqual(sparse_embeds_combined.shape, (batch_size, num_queries, num_points + 2, hidden_size))

    def test_dense_mask_inputs(self):
        batch_size = 2
        hidden_size = self.config.prompt_encoder_config.hidden_size
        img_embed_size = self.config.prompt_encoder_config.image_embedding_size
        mask_size = 4 * img_embed_size
        input_masks = torch.randn(batch_size, 1, mask_size, mask_size, device=torch_device)
        _, dense_embeds = self.prompt_encoder(
            input_points=None, input_labels=None, input_boxes=None, input_masks=input_masks
        )
        self.assertEqual(dense_embeds.shape, (batch_size, hidden_size, img_embed_size, img_embed_size))

    def test_no_prompt_fallback(self):
        hidden_size = self.config.prompt_encoder_config.hidden_size
        img_embed_size = self.config.prompt_encoder_config.image_embedding_size
        sparse_embeds, dense_embeds = self.prompt_encoder(
            input_points=None, input_labels=None, input_boxes=None, input_masks=None
        )
        self.assertIsNone(sparse_embeds)
        self.assertEqual(dense_embeds.shape, (1, hidden_size, img_embed_size, img_embed_size))
        self.assertFalse(torch.isnan(dense_embeds).any())
        self.assertFalse(torch.isinf(dense_embeds).any())


@require_torch
class EfficientViTSamMaskDecoderTest(unittest.TestCase):
    def setUp(self):
        self.mask_decoder_tester = SamMaskDecoderTester(hidden_size=32)
        self.config = self.mask_decoder_tester.get_config()
        self.mask_decoder = EfficientViTSamMaskDecoder(self.config).to(torch_device).eval()

    def test_two_way_transformer_attention(self):
        batch_size = 2
        hidden_size = self.config.hidden_size
        h, w = 16, 16
        image_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)
        image_pos_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)
        dense_prompt_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)

        for n_tokens in [1, 4, 8]:
            point_batch_size = 2
            sparse_prompt_embeddings = torch.randn(
                batch_size, point_batch_size, n_tokens, hidden_size, device=torch_device
            )
            with torch.no_grad():
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_positional_embeddings=image_pos_embeddings,
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                    multimask_output=True,
                )
            self.assertEqual(
                low_res_masks.shape, (batch_size, point_batch_size, self.config.num_multimask_outputs, 4 * h, 4 * w)
            )
            self.assertEqual(iou_predictions.shape, (batch_size, point_batch_size, self.config.num_multimask_outputs))

    def test_output_shape_invariants(self):
        batch_size = 2
        point_batch_size = 3
        n_tokens = 5
        hidden_size = self.config.hidden_size
        h, w = 8, 8

        image_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)
        image_pos_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)
        sparse_prompt_embeddings = torch.randn(
            batch_size, point_batch_size, n_tokens, hidden_size, device=torch_device
        )
        dense_prompt_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)

        with torch.no_grad():
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_pos_embeddings,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=True,
            )

        self.assertEqual(iou_predictions.shape, (batch_size, point_batch_size, self.config.num_multimask_outputs))
        self.assertEqual(
            low_res_masks.shape, (batch_size, point_batch_size, self.config.num_multimask_outputs, 4 * h, 4 * w)
        )

    def test_single_vs_multimask_outputs(self):
        batch_size = 1
        point_batch_size = 2
        n_tokens = 2
        hidden_size = self.config.hidden_size
        h, w = 16, 16

        image_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)
        image_pos_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)
        sparse_prompt_embeddings = torch.randn(
            batch_size, point_batch_size, n_tokens, hidden_size, device=torch_device
        )
        dense_prompt_embeddings = torch.randn(batch_size, hidden_size, h, w, device=torch_device)

        with torch.no_grad():
            multi_masks, multi_iou = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_pos_embeddings,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=True,
            )
            single_masks, single_iou = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_pos_embeddings,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=False,
            )

        self.assertEqual(multi_masks.shape[2], 3)
        self.assertEqual(multi_iou.shape[2], 3)

        self.assertEqual(single_masks.shape[2], 1)
        self.assertEqual(single_iou.shape[2], 1)
