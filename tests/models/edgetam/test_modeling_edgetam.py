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
"""Testing suite for the PyTorch EDGETAM model."""

import gc
import unittest

import requests

from transformers import (
    EdgeTamConfig,
    EdgeTamMaskDecoderConfig,
    EdgeTamPromptEncoderConfig,
    EdgeTamVisionConfig,
    Sam2Processor,
    pipeline,
)
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available
from transformers.video_utils import load_video

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import AutoConfig, EdgeTamModel, Sam2Processor


if is_vision_available():
    from PIL import Image


class EdgeTamPromptEncoderTester:
    def __init__(
        self,
        hidden_size=32,
        input_image_size=128,
        patch_size=16,
        mask_input_channels=8,
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
        return EdgeTamPromptEncoderConfig(
            image_size=self.input_image_size,
            patch_size=self.patch_size,
            mask_input_channels=self.mask_input_channels,
            hidden_size=self.hidden_size,
            num_point_embeddings=self.num_point_embeddings,
            hidden_act=self.hidden_act,
        )

    def prepare_config_and_inputs(self):
        dummy_points = floats_tensor([self.batch_size, 3, 2])
        config = self.get_config()

        return config, dummy_points


class EdgeTamMaskDecoderTester:
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

    def get_config(self):
        return EdgeTamMaskDecoderConfig(
            hidden_size=self.hidden_size,
            hidden_act=self.hidden_act,
            mlp_dim=self.mlp_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            attention_downsample_rate=self.attention_downsample_rate,
            num_multimask_outputs=self.num_multimask_outputs,
            iou_head_depth=self.iou_head_depth,
            iou_head_hidden_dim=self.iou_head_hidden_dim,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        dummy_inputs = {
            "image_embedding": floats_tensor([self.batch_size, self.hidden_size]),
        }

        return config, dummy_inputs


class EdgeTamModelTester:
    def __init__(
        self,
        parent,
        num_channels=3,
        image_size=128,
        hidden_size=12,
        patch_kernel_size=7,
        patch_stride=4,
        patch_padding=3,
        dim_mul=2.0,
        backbone_channel_list=[96, 48, 24, 12],
        backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]],
        fpn_hidden_size=32,
        memory_encoder_hidden_size=32,
        batch_size=2,
        is_training=False,
    ):
        self.parent = parent
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.dim_mul = dim_mul
        self.backbone_channel_list = backbone_channel_list
        self.backbone_feature_sizes = backbone_feature_sizes
        self.fpn_hidden_size = fpn_hidden_size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.memory_encoder_hidden_size = memory_encoder_hidden_size

        self.prompt_encoder_tester = EdgeTamPromptEncoderTester()
        self.mask_decoder_tester = EdgeTamMaskDecoderTester()

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        vision_config = EdgeTamVisionConfig(
            backbone_config=AutoConfig.from_pretrained(
                "timm/repvit_m1.dist_in1k",
                model_args={
                    "in_chans": 3,
                    "features_only": True,
                    "out_indices": (0, 1, 2, 3),
                    "embed_dim": self.backbone_channel_list[::-1],
                },
            ),
            backbone_channel_list=self.backbone_channel_list,
            backbone_feature_sizes=self.backbone_feature_sizes,
            fpn_hidden_size=self.fpn_hidden_size,
        )

        prompt_encoder_config = self.prompt_encoder_tester.get_config()

        mask_decoder_config = self.mask_decoder_tester.get_config()

        return EdgeTamConfig(
            vision_config=vision_config,
            prompt_encoder_config=prompt_encoder_config,
            mask_decoder_config=mask_decoder_config,
            memory_attention_hidden_size=self.hidden_size,
            memory_encoder_hidden_size=self.memory_encoder_hidden_size,
            image_size=self.image_size,
            mask_downsampler_embed_dim=32,
            memory_fuser_embed_dim=32,
            memory_attention_num_layers=1,
            memory_attention_feed_forward_hidden_size=32,
        )

    def create_and_check_model(self, config, pixel_values):
        model = EdgeTamModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(result.iou_scores.shape, (self.batch_size, 1, 3))
        self.parent.assertEqual(result.pred_masks.shape[:3], (self.batch_size, 1, 3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class EdgeTamModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SAM's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (EdgeTamModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": EdgeTamModel, "mask-generation": EdgeTamModel} if is_torch_available() else {}
    )

    test_resize_embeddings = False
    _is_composite = True

    def setUp(self):
        self.model_tester = EdgeTamModelTester(self)
        common_properties = ["initializer_range"]
        self.config_tester = ConfigTester(
            self, config_class=EdgeTamConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Timm model does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Can't get or set embeddings for Timm model")
    def test_model_get_set_embeddings(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Override as diffence slightly higher than the threshold
    # def test_batching_equivalence(self, atol=5e-4, rtol=5e-4):
    #     super().test_batching_equivalence(atol=atol, rtol=rtol)

    @unittest.skip(reason="TimmWrapperModel does not support an attention implementation")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    @unittest.skip(reason="vision_hidden_states from TimmWrapperModel")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="TIMM's attention implementation is self configured and won't raise ValueError on global attention implementation."
    )
    def test_flash_attn_2_can_dispatch_composite_models(self):
        pass

    @unittest.skip("TimmWrapperModel cannot be tested with meta device")
    def test_can_be_initialized_on_meta(self):
        pass

    @unittest.skip("TimmWrapperModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    ## Skip flash attention releated tests below
    ## correct configuration:
    ## from_pretrained(model_id, attn_implementation={"text_config": "flash_attention_2", "vision_config": "eager"}
    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_eager_matches_fa3_generate(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip("SDPA test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_eager_matches_sdpa_generate_with_dynamic_cache(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_3_inference_equivalence_right_padding(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_kernels_inference_equivalence(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_kernels_mps_inference_equivalence(self):
        pass

    @unittest.skip("SDPA test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip("EdgeTAM does not have language_model, vision_tower, multi_modal_projector.")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_generate_compilation_all_outputs(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "yonigozlan/EdgeTAM-hf"
        model = EdgeTamModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_sdpa_can_compile_dynamic(self):
        self.skipTest(reason="EDGETAM model can't be compiled dynamic yet")


def prepare_image():
    img_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_groceries_image():
    img_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/groceries.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_dog_img():
    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_video():
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    raw_video, _ = load_video(video_url)
    return raw_video


@slow
class EdgeTamModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model = EdgeTamModel.from_pretrained("yonigozlan/EdgeTAM-hf").to(torch.float32)
        self.processor = Sam2Processor.from_pretrained("yonigozlan/EdgeTAM-hf")
        self.model.to(torch_device)
        self.model.eval()

    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        backend_empty_cache(torch_device)

    def test_inference_mask_generation_one_point_multimask(self):
        raw_image = prepare_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        inputs = self.processor(
            images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        self.assertEqual(outputs.iou_scores.shape, (1, 1, 3))
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 3, 256, 256))
        sorted_indices = torch.argsort(outputs.iou_scores.squeeze(), descending=True)
        scores = outputs.iou_scores.squeeze()[sorted_indices]
        masks_logits = outputs.pred_masks.squeeze()[sorted_indices][0, :3, :3]
        torch.testing.assert_close(
            scores, torch.tensor([0.7621, 0.4859, 0.0461]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-19.5483, -22.3549, -26.0962], [-18.1821, -23.4761, -24.2262], [-20.3549, -24.5518, -22.7232]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_one_point_no_multimask(self):
        raw_image = prepare_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        inputs = self.processor(
            images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        self.assertEqual(outputs.iou_scores.shape, (1, 1, 1))
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.pred_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.7621]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-19.5483, -22.3549, -26.0962], [-18.1821, -23.4761, -24.2262], [-20.3549, -24.5518, -22.7232]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_batched_images_multi_points(self):
        raw_image1 = prepare_image()
        raw_image2 = prepare_dog_img()
        input_points = [[[[500, 375]]], [[[770, 200], [730, 120]]]]
        input_labels = [[[1]], [[1, 0]]]

        inputs = self.processor(
            images=[raw_image1, raw_image2], input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        self.assertEqual(outputs.iou_scores.shape, (2, 1, 3))
        self.assertEqual(outputs.pred_masks.shape, (2, 1, 3, 256, 256))

        sorted_indices = torch.argsort(outputs.iou_scores[0].squeeze(), descending=True)
        scores1 = outputs.iou_scores[0].squeeze()[sorted_indices]
        masks_logits1 = outputs.pred_masks[0].squeeze()[sorted_indices][0, :3, :3]
        sorted_indices = torch.argsort(outputs.iou_scores[1].squeeze(), descending=True)
        scores2 = outputs.iou_scores[1].squeeze()[sorted_indices]
        masks_logits2 = outputs.pred_masks[1].squeeze()[sorted_indices][0, :3, :3]
        torch.testing.assert_close(
            scores1, torch.tensor([0.7490, 0.4685, 0.0463]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            masks_logits1,
            torch.tensor(
                [[-19.1423, -21.6488, -25.6816], [-17.8018, -22.6512, -23.5699], [-19.9140, -23.6919, -22.3147]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        torch.testing.assert_close(
            scores2, torch.tensor([0.7225, 0.6515, 0.6350]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            masks_logits2,
            torch.tensor([[-8.8259, -7.7961, -9.3665], [-8.2648, -8.7771, -9.1390], [-9.5951, -8.3995, -9.0599]]).to(
                torch_device
            ),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_batched_images_batched_points_multi_points(self):
        raw_image1 = prepare_image()
        raw_image2 = prepare_groceries_image()
        input_points = [[[[500, 375]], [[650, 750]]], [[[400, 300]], [[630, 300], [550, 300]]]]
        input_labels = [[[1], [1]], [[1], [1, 1]]]
        inputs = self.processor(
            images=[raw_image1, raw_image2], input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        self.assertEqual(outputs.iou_scores.shape, (2, 2, 1))
        self.assertEqual(outputs.pred_masks.shape, (2, 2, 1, 256, 256))
        torch.testing.assert_close(
            outputs.iou_scores,
            torch.tensor([[[0.7490], [0.9397]], [[0.7952], [0.8723]]]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[:, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-19.1423, -21.6488], [-17.8018, -22.6512]]], [[[-7.1591, -9.8201], [-7.4133, -9.2781]]]],
                    [[[[-16.7645, -15.2790], [-16.1805, -16.2937]]], [[[-8.5934, -8.4215], [-8.1873, -8.3722]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_batched_images_batched_boxes(self):
        raw_image1 = prepare_image()
        raw_image2 = prepare_groceries_image()
        input_boxes = [
            [[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750]],
            [[450, 170, 520, 350], [350, 190, 450, 350], [500, 170, 580, 350], [580, 170, 640, 350]],
        ]
        inputs = self.processor(images=[raw_image1, raw_image2], input_boxes=input_boxes, return_tensors="pt").to(
            torch_device
        )
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        self.assertEqual(outputs.iou_scores.shape, (2, 4, 1))
        self.assertEqual(outputs.pred_masks.shape, (2, 4, 1, 256, 256))
        torch.testing.assert_close(
            outputs.iou_scores,
            torch.tensor([[[0.9773], [0.9415], [0.9683], [0.8792]], [[0.9721], [0.9852], [0.9812], [0.9760]]]).to(
                torch_device
            ),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[:, :, :, :2, :2],
            torch.tensor(
                [
                    [
                        [[[-12.6412, -12.0553], [-11.8415, -13.1696]]],
                        [[[-16.0378, -19.9641], [-15.4939, -19.0260]]],
                        [[[-18.8254, -23.6185], [-17.7889, -23.2116]]],
                        [[[-25.7024, -29.8722], [-22.9264, -30.0557]]],
                    ],
                    [
                        [[[-19.0264, -17.0396], [-16.9458, -16.3287]]],
                        [[[-20.9671, -19.2132], [-18.5827, -18.0511]]],
                        [[[-22.4642, -19.7389], [-19.4541, -19.4717]]],
                        [[[-21.9226, -18.6297], [-18.9272, -18.8151]]],
                    ],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_from_existing_points_and_mask(self):
        raw_image = prepare_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]
        original_inputs = self.processor(
            images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)
        with torch.no_grad():
            outputs = self.model(**original_inputs)

        # best mask to use as input for new points
        mask_input = outputs.pred_masks[:, :, torch.argmax(outputs.iou_scores)]

        new_input_points = [[[[500, 375], [1125, 625]]]]
        new_input_labels = [[[1, 1]]]
        inputs = self.processor(
            input_points=new_input_points,
            input_labels=new_input_labels,
            original_sizes=original_inputs["original_sizes"],
            return_tensors="pt",
        ).to(torch_device)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                input_masks=mask_input,
                image_embeddings=outputs.image_embeddings,
                multimask_output=False,
            )

        self.assertEqual(outputs.iou_scores.shape, (1, 1, 1))
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.pred_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.9431]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor([[-4.1968, -4.9034, -6.0680], [-4.4053, -5.1200, -5.8580], [-4.3920, -5.5096, -5.8166]]).to(
                torch_device
            ),
            atol=1e-4,
            rtol=1e-4,
        )

        # with negative point
        new_input_points = [[[[500, 375], [1125, 625]]]]
        new_input_labels = [[[1, 0]]]
        inputs = self.processor(
            input_points=new_input_points,
            input_labels=new_input_labels,
            original_sizes=original_inputs["original_sizes"],
            return_tensors="pt",
        ).to(torch_device)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                input_masks=mask_input,
                image_embeddings=outputs.image_embeddings,
                multimask_output=False,
            )
        self.assertEqual(outputs.iou_scores.shape, (1, 1, 1))
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.pred_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.9695]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-14.3212, -15.4295, -17.4482], [-13.2246, -15.9468, -17.1341], [-15.1678, -16.4498, -14.7385]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_dummy_pipeline_generation(self):
        generator = pipeline("mask-generation", model="yonigozlan/EdgeTAM-hf", device=torch_device)
        raw_image = prepare_image()

        _ = generator(raw_image, points_per_batch=64)
