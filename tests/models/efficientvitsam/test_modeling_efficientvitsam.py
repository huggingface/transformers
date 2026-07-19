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

from transformers import (
    EfficientViTSamConfig,
    EfficientViTSamMaskDecoderConfig,
    EfficientViTSamPromptEncoderConfig,
    EfficientViTSamVisionConfig,
)
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        EfficientViTSamModel,
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
        with self.assertRaisesRegex(ValueError, "input_points must be a 3D tensor"):
            model(pixel_values=pixel_values, input_boxes=input_boxes.unsqueeze(1))
        with self.assertRaisesRegex(ValueError, "as many bounding boxes as input points"):
            model(pixel_values=pixel_values, input_points=input_points, input_boxes=input_boxes[:, :1])

    @slow
    def test_inference_l0(self):
        model = EfficientViTSamModel.from_pretrained("./test_l0_hf")
        model.to(torch_device)
        model.eval()

        pixel_values = torch.zeros(1, 3, 512, 512).to(torch_device)
        with torch.no_grad():
            outputs = model(pixel_values)

        self.assertEqual(outputs.iou_scores.shape, (1, 1, 3))


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
