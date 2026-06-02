# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Sapiens2 model."""

import unittest
from functools import cached_property

from transformers import Sapiens2Config, Sapiens2HeadConfig
from transformers.testing_utils import Expectations, require_cv2, require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        Sapiens2Backbone,
        Sapiens2ForImageMatting,
        Sapiens2ForNormalEstimation,
        Sapiens2ForPointmapEstimation,
        Sapiens2ForPoseEstimation,
        Sapiens2ForSemanticSegmentation,
        Sapiens2ImageProcessor,
        Sapiens2Model,
    )
    from transformers.image_utils import load_image_as_tensor
    from transformers.models.sapiens2.modeling_sapiens2 import (
        Sapiens2ImageMattingOutput,
        Sapiens2PointmapEstimatorOutput,
    )


class Sapiens2ModelTester:
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
        return Sapiens2Config(
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
            stage_names=["stem"] + [f"stage{i}" for i in range(1, self.num_hidden_layers + 1)],
            out_indices=[0, 1],
            reshape_hidden_states=True,
            num_labels=4,
            flip_pairs=[[1, 2], [3, 4]],
            # Head config sized to satisfy all model conversion patterns in test_reverse_loading_mapping
            head_config=Sapiens2HeadConfig(
                upsample_out_channels=[8, 4, 4, 4],
                upsample_kernel_sizes=[4, 4, 4, 4],
                conv_out_channels=[4, 4, 4],
                conv_kernel_sizes=[1, 1, 1],
                scale_conv_out_channels=[8, 4, 4],
                scale_conv_kernel_sizes=[1, 1, 1],
                scale_final_hidden_sizes=[8, 4],
            ),
        )

    def create_and_check_backbone(self, config, pixel_values, labels):
        config.out_features = ["stage1", "stage2"]
        config.reshape_hidden_states = True

        model = Sapiens2Backbone(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(pixel_values)

        self.parent.assertEqual(len(outputs.feature_maps), 2)
        for fm in outputs.feature_maps:
            b, c, h, w = fm.shape
            self.parent.assertEqual(b, self.batch_size)
            self.parent.assertEqual(c, self.hidden_size)
            self.parent.assertEqual(h, self.image_size // self.patch_size)
            self.parent.assertEqual(w, self.image_size // self.patch_size)

    def create_and_check_model(self, config, pixel_values, labels):
        model = Sapiens2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, labels):
        model = Sapiens2ForSemanticSegmentation(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # patch_height = image_size // patch_size = 30 // 2 = 15
        # 4 deconv layers with stride=2: 15 * 2^4 = 240
        patch_height = self.image_size // self.patch_size
        expected_h = patch_height * (2 ** len(config.head_config.upsample_out_channels))
        self.parent.assertEqual(
            result.logits.shape,
            (self.batch_size, config.num_labels, expected_h, expected_h),
        )

    def create_and_check_for_pose_estimation(self, config, pixel_values, labels):
        model = Sapiens2ForPoseEstimation(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        patch_height = self.image_size // self.patch_size
        expected_h = patch_height * (2 ** len(config.head_config.upsample_out_channels))
        self.parent.assertEqual(
            result.heatmaps.shape,
            (self.batch_size, config.num_labels, expected_h, expected_h),
        )

    def create_and_check_for_normal_estimation(self, config, pixel_values, labels):
        model = Sapiens2ForNormalEstimation(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # PixelShuffle: Conv2d(padding=(ks-1)//2) then shuffle(2) — size per layer: (h + 2p - ks + 1) * 2
        expected_h = config.image_size // self.patch_size
        for ks in config.head_config.upsample_kernel_sizes:
            padding = (ks - 1) // 2
            expected_h = (expected_h + 2 * padding - ks + 1) * 2
        self.parent.assertEqual(
            result.normals.shape,
            (self.batch_size, config.num_labels, expected_h, expected_h),
        )
        self.parent.assertIsNone(result.loss)
        with self.parent.assertRaises(NotImplementedError):
            model(pixel_values, labels=torch.randn_like(result.normals))

    def create_and_check_for_matting(self, config, pixel_values, labels):
        model = Sapiens2ForImageMatting(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        expected_h = config.image_size // self.patch_size
        for ks in config.head_config.upsample_kernel_sizes:
            padding = (ks - 1) // 2
            expected_h = (expected_h + 2 * padding - ks + 1) * 2
        self.parent.assertEqual(result.foregrounds.shape, (self.batch_size, 3, expected_h, expected_h))
        self.parent.assertEqual(result.alphas.shape, (self.batch_size, 1, expected_h, expected_h))
        # outputs are sigmoid-activated
        self.parent.assertGreaterEqual(result.foregrounds.min().item(), 0.0)
        self.parent.assertLessEqual(result.foregrounds.max().item(), 1.0)
        self.parent.assertGreaterEqual(result.alphas.min().item(), 0.0)
        self.parent.assertLessEqual(result.alphas.max().item(), 1.0)
        self.parent.assertIsNone(result.loss)
        with self.parent.assertRaises(NotImplementedError):
            model(pixel_values, labels=torch.randn(self.batch_size, 4, expected_h, expected_h))

    def create_and_check_for_pointmap_estimation(self, config, pixel_values, labels):
        model = Sapiens2ForPointmapEstimation(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # PixelShuffle: Conv2d(padding=(ks-1)//2) then shuffle(2) — size per layer: (h + 2p - ks + 1) * 2
        expected_h = config.image_size // self.patch_size
        for ks in config.head_config.upsample_kernel_sizes:
            padding = (ks - 1) // 2
            expected_h = (expected_h + 2 * padding - ks + 1) * 2
        self.parent.assertEqual(
            result.pointmaps.shape,
            (self.batch_size, config.num_labels, expected_h, expected_h),
        )
        self.parent.assertEqual(result.scales.shape, (self.batch_size, 1))
        self.parent.assertIsNone(result.loss)
        with self.parent.assertRaises(NotImplementedError):
            model(pixel_values, labels=torch.randn_like(result.pointmaps))

    def prepare_config_and_inputs_for_semantic_segmentation(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = ids_tensor([self.batch_size, self.image_size, self.image_size], config.num_labels)
        return config, pixel_values, labels

    def prepare_config_and_inputs_for_pointmap_estimation(self):
        config = self.get_config()
        config.head_config.use_pixel_shuffle = True
        pixel_values = floats_tensor([self.batch_size, self.num_channels, config.image_size, config.image_size])
        labels = None
        return config, pixel_values, labels

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        # Use pixel-shuffle so all model classes (including Normal/Pointmap/Matting) instantiate
        # decode_head.input_conv, satisfying the conversion patterns checked by test_reverse_loading_mapping.
        config.head_config.use_pixel_shuffle = True
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class Sapiens2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Sapiens2 does not use input_ids,
    inputs_embeds, attention_mask and seq_length.
    """

    all_model_classes = (
        (
            Sapiens2Model,
            Sapiens2Backbone,
            Sapiens2ForImageMatting,
            Sapiens2ForNormalEstimation,
            Sapiens2ForPointmapEstimation,
            Sapiens2ForPoseEstimation,
            Sapiens2ForSemanticSegmentation,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "image-feature-extraction": Sapiens2Model,
        }
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = Sapiens2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Sapiens2Config, has_text_modality=False, hidden_size=32)
        # The decoder heads contain ConvTranspose2d layers which are non-deterministic on CUDA.
        # This non-deterministic behavior is amplified by the InstanceNorm2d layers and results in up
        # to 6e-3 output differences with identical head inputs. We set cudnn.deterministic = True
        # for test stability.
        self._original_cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

    def tearDown(self):
        torch.backends.cudnn.deterministic = self._original_cudnn_deterministic

    def test_backbone(self):
        config, pixel_values, labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(config, pixel_values, labels)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Sapiens2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_semantic_segmentation()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    def test_for_pose_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_semantic_segmentation()
        self.model_tester.create_and_check_for_pose_estimation(*config_and_inputs)

    def test_for_pointmap_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_pointmap_estimation()
        self.model_tester.create_and_check_for_pointmap_estimation(*config_and_inputs)

    def test_for_normal_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_pointmap_estimation()
        self.model_tester.create_and_check_for_normal_estimation(*config_and_inputs)

    def test_for_matting(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_pointmap_estimation()
        self.model_tester.create_and_check_for_matting(*config_and_inputs)

    def test_batching_equivalence(self, atol=1e-4, rtol=1e-4):
        # InstanceNorm2d in the decoder heads computes per-instance statistics; different batch
        # sizes can trigger different parallelisation paths on CPU, producing O(1e-5) FP differences.
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    @unittest.skip(reason="Sapiens2 does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass


def prepare_img():
    image = load_image_as_tensor("./tests/fixtures/tests_samples/COCO/000000004016.png")
    return image


@require_torch
@require_vision
class Sapiens2ModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        # The decoder heads contain ConvTranspose2d layers which are non-deterministic on CUDA.
        # This non-deterministic behavior is amplified by the InstanceNorm2d layers and results in up
        # to 6e-3 output differences with identical head inputs. We set cudnn.deterministic = True
        # for test stability.
        self._original_cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

    def tearDown(self):
        torch.backends.cudnn.deterministic = self._original_cudnn_deterministic

    @cached_property
    def default_image_processor(self):
        return Sapiens2ImageProcessor.from_pretrained("facebook/sapiens2-pretrain-0.4b")

    @slow
    def test_inference_no_head(self):
        model = Sapiens2Model.from_pretrained("facebook/sapiens2-pretrain-0.4b").eval().to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the last hidden states
        # seq length = num_patches + 1 (CLS token) + num_register_tokens
        _, _, height, width = inputs["pixel_values"].shape
        num_patches = (height // model.config.patch_size) * (width // model.config.patch_size)
        expected_seq_length = num_patches + 1 + model.config.num_register_tokens
        expected_shape = torch.Size((1, expected_seq_length, model.config.hidden_size))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        last_layer_cls_token = outputs.pooler_output
        EXPECTED_CLS_SLICE = Expectations({("cuda", None): [-0.09233, -0.00107, -0.12215, 0.07374, -0.03773]})
        expected_cls_slice = torch.tensor(EXPECTED_CLS_SLICE.get_expectation(), device=torch_device)
        torch.testing.assert_close(last_layer_cls_token[0, :5], expected_cls_slice, rtol=1e-3, atol=1e-3)

        last_layer_register_tokens = outputs.last_hidden_state[:, 1 : model.config.num_register_tokens + 1]
        EXPECTED_REGISTER_SLICE = Expectations({("cuda", None): [0.08412, 0.04387, 0.05709, -0.04962, 0.03715]})
        expected_register_slice = torch.tensor(EXPECTED_REGISTER_SLICE.get_expectation(), device=torch_device)
        torch.testing.assert_close(last_layer_register_tokens[0, 0, :5], expected_register_slice, rtol=1e-3, atol=1e-3)

        last_layer_patch_tokens = outputs.last_hidden_state[:, model.config.num_register_tokens + 1 :]
        EXPECTED_PATCH_SLICE = Expectations({("cuda", None): [0.14232, -0.11947, -0.05910, -0.09457, -0.11410]})
        expected_patch_slice = torch.tensor(EXPECTED_PATCH_SLICE.get_expectation(), device=torch_device)
        torch.testing.assert_close(last_layer_patch_tokens[0, 0, :5], expected_patch_slice, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_semantic_segmentation(self):
        model = Sapiens2ForSemanticSegmentation.from_pretrained("facebook/sapiens2-seg-0.4b").eval().to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits shape: segmentation head upsamples back to the original image resolution
        _, _, height, width = inputs["pixel_values"].shape
        expected_shape = torch.Size((1, model.config.num_labels, height, width))
        self.assertEqual(logits.shape, expected_shape)

        EXPECTED_LOGITS_SLICE = Expectations(
            {("cuda", None): [[3.45260, 5.55483, 6.57901], [5.71913, 7.21420, 8.11209], [6.82645, 7.98208, 8.31385]]}
        )
        expected_logits_slice = torch.tensor(EXPECTED_LOGITS_SLICE.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0, 0, :3, :3], expected_logits_slice, rtol=1e-3, atol=1e-3)

        # verify post-processing without resizing: output shape matches model input resolution
        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs)
        self.assertEqual(len(segmentation), 1)
        self.assertEqual(segmentation[0].shape, torch.Size([height, width]))

        # verify post-processing with target_sizes
        target_size = (height // 2, width // 2)
        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[target_size])
        self.assertEqual(len(segmentation), 1)
        self.assertEqual(segmentation[0].shape, torch.Size(target_size))

        EXPECTED_CLASS_IDS = Expectations({("cuda", None): [[4, 3, 3], [3, 3, 3], [3, 3, 3]]})
        expected_class_ids = torch.tensor(EXPECTED_CLASS_IDS.get_expectation(), device=torch_device)
        torch.testing.assert_close(segmentation[0][50:53, 50:53], expected_class_ids)

    @require_cv2
    @slow
    def test_inference_pose_estimation(self):
        model = Sapiens2ForPoseEstimation.from_pretrained("facebook/sapiens2-pose-0.4b").eval().to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()

        image_height, image_width = image.shape[-2:]

        # person bbox in COCO format (x, y, w, h)
        boxes = [[[2.7080630e02, 5.7221174e-01, 2.9409006e02, 3.7946970e02]]]
        inputs = image_processor(image, boxes=boxes, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        heatmaps = outputs.heatmaps
        self.assertEqual(heatmaps.shape, torch.Size([1, model.config.num_labels, 256, 192]))
        EXPECTED_HEATMAPS = Expectations(
            {("cuda", None): [[0.26140, 0.24656, 0.21673], [0.33708, 0.31597, 0.28028], [0.41624, 0.39270, 0.35014]]}
        )
        expected_heatmaps = torch.tensor(EXPECTED_HEATMAPS.get_expectation(), device=torch_device)
        torch.testing.assert_close(heatmaps[0, 0, 70:73, 70:73], expected_heatmaps, rtol=1e-2, atol=1e-2)

        results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)
        person = results[0][0]

        keypoints = person["keypoints"]
        EXPECTED_KEYPOINTS = Expectations(
            {("cuda", None): [[364.33920111, 97.92528764], [373.25104943, 80.97749201], [353.21072316, 83.38954486]]}
        )
        expected_keypoints = torch.tensor(EXPECTED_KEYPOINTS.get_expectation(), device=torch_device)
        torch.testing.assert_close(keypoints[:3], expected_keypoints, rtol=1e-2, atol=1e-2)

        scores = person["scores"]
        EXPECTED_SCORES = Expectations({("cuda", None): [1.0007433, 0.9987416, 1.0015154]})
        expected_scores = torch.tensor(EXPECTED_SCORES.get_expectation(), device=torch_device)
        torch.testing.assert_close(scores[:3], expected_scores, rtol=1e-2, atol=1e-2)

        bbox = person["bbox"]
        expected_bbox_xywh = torch.tensor(boxes[0][0], device=torch_device)
        expected_bbox_xyxy = torch.tensor(
            [
                expected_bbox_xywh[0],
                expected_bbox_xywh[1],
                expected_bbox_xywh[0] + expected_bbox_xywh[2],
                expected_bbox_xywh[1] + expected_bbox_xywh[3],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(bbox, expected_bbox_xyxy, rtol=1e-3, atol=1e-3)

        # target_sizes without source_sizes must raise
        with self.assertRaises(ValueError):
            image_processor.post_process_pose_estimation(outputs, boxes=boxes, target_sizes=[(432, 640)])

        # source_sizes + target_sizes: keypoints and bbox scaled by target/source
        target_height, target_width = image_height * 2, image_width * 2
        results_scaled = image_processor.post_process_pose_estimation(
            outputs,
            boxes=boxes,
            source_sizes=[(image_height, image_width)],
            target_sizes=[(target_height, target_width)],
        )
        torch.testing.assert_close(results_scaled[0][0]["keypoints"], keypoints * 2.0)
        torch.testing.assert_close(results_scaled[0][0]["bbox"], expected_bbox_xyxy * 2.0)

        # Test flipping
        flipped_inputs = {"pixel_values": inputs["pixel_values"].flip(-1)}
        flip_pairs = torch.tensor(model.config.flip_pairs)

        with torch.no_grad():
            flipped_outputs = model(**flipped_inputs, flip_pairs=flip_pairs)

        flipped_heatmaps = flipped_outputs.heatmaps
        EXPECTED_FLIPPED_HEATMAPS = Expectations(
            {("cuda", None): [[0.27348, 0.25426, 0.22496], [0.34877, 0.32563, 0.28418], [0.43967, 0.40607, 0.35721]]}
        )
        expected_flipped_heatmaps = torch.tensor(EXPECTED_FLIPPED_HEATMAPS.get_expectation(), device=torch_device)
        torch.testing.assert_close(
            flipped_heatmaps[0, 0, 70:73, 70:73], expected_flipped_heatmaps, rtol=1e-2, atol=1e-2
        )

        final_results = image_processor.post_process_pose_estimation(
            outputs, outputs_flipped=flipped_outputs, boxes=boxes
        )
        self.assertEqual(len(final_results), 1)
        self.assertEqual(len(final_results[0]), 1)

        final_person = final_results[0][0]
        final_keypoints = final_person["keypoints"]
        EXPECTED_FINAL_KEYPOINTS = Expectations(
            {("cuda", None): [[364.14644305, 97.99268751], [373.66756367, 81.19966519], [353.4574526, 83.647911]]}
        )
        expected_final_keypoints = torch.tensor(EXPECTED_FINAL_KEYPOINTS.get_expectation(), device=torch_device)
        torch.testing.assert_close(final_keypoints[:3], expected_final_keypoints, rtol=1e-2, atol=1e-2)

        final_scores = final_person["scores"]
        EXPECTED_FINAL_SCORES = Expectations({("cuda", None): [1.0064079, 0.98746514, 0.99821794]})
        expected_final_scores = torch.tensor(EXPECTED_FINAL_SCORES.get_expectation(), device=torch_device)
        torch.testing.assert_close(final_scores[:3], expected_final_scores, rtol=1e-2, atol=1e-2)

        final_bbox = final_person["bbox"]
        torch.testing.assert_close(final_bbox, expected_bbox_xyxy, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_normal_estimation(self):
        model = Sapiens2ForNormalEstimation.from_pretrained("facebook/sapiens2-normal-0.4b").eval().to(torch_device)

        image_processor = Sapiens2ImageProcessor.from_pretrained("facebook/sapiens2-normal-0.4b")
        image = prepare_img()
        image_height, image_width = image.shape[-2:]
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        _, _, height, width = inputs["pixel_values"].shape
        self.assertEqual(outputs.normals.shape, torch.Size([1, 3, height, width]))

        # We can get closer to expected values by using cv2 resize instead of torchvision.
        EXPECTED_NORMALS = Expectations(
            {("cuda", None): [[0.9577, 1.8808, 0.9826], [1.6904, 1.7351, 1.9120], [2.4828, 1.9887, 2.5168]]}
        )
        expected_normals = torch.tensor(EXPECTED_NORMALS.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.normals[0, 0, :3, :3], expected_normals, rtol=1e-2, atol=1e-2)

        result = image_processor.post_process_normal_estimation(outputs, source_sizes=[(image_height, image_width)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["normals"].shape, torch.Size([3, 432, 640]))

        EXPECTED_POSTPROCESSED_NORMALS = Expectations(
            {("cuda", None): [[-0.8266, -0.7899, -0.7512], [-0.8227, -0.7843, -0.7440], [-0.8098, -0.7721, -0.7318]]}
        )
        expected_postprocessed_normals = torch.tensor(
            EXPECTED_POSTPROCESSED_NORMALS.get_expectation(), device=torch_device
        )
        torch.testing.assert_close(
            result[0]["normals"][0, :3, :3], expected_postprocessed_normals, rtol=1e-2, atol=1e-2
        )

    @slow
    def test_inference_pointmap_estimation(self):
        model = (
            Sapiens2ForPointmapEstimation.from_pretrained("facebook/sapiens2-pointmap-0.4b").eval().to(torch_device)
        )

        image_processor = Sapiens2ImageProcessor.from_pretrained("facebook/sapiens2-pointmap-0.4b")
        image = prepare_img()
        image_height, image_width = image.shape[-2:]
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsInstance(outputs, Sapiens2PointmapEstimatorOutput)
        _, _, height, width = inputs["pixel_values"].shape
        self.assertEqual(outputs.pointmaps.shape, torch.Size([1, 3, height, width]))
        self.assertEqual(outputs.scales.shape, torch.Size([1, 1]))

        EXPECTED_SCALE = Expectations({("cuda", None): [[0.9931]]})
        expected_scale = torch.tensor(EXPECTED_SCALE.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.scales, expected_scale, rtol=1e-3, atol=1e-3)

        EXPECTED_POINTMAP = Expectations(
            {("cuda", None): [[-0.0096, -0.0567, -0.0460], [-0.0657, -0.0583, -0.0688], [-0.1035, -0.0363, -0.0659]]}
        )
        expected_pointmap = torch.tensor(EXPECTED_POINTMAP.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.pointmaps[0, 0, :3, :3], expected_pointmap, rtol=1e-2, atol=1e-2)

        result = image_processor.post_process_pointmap_estimation(outputs, source_sizes=[(image_height, image_width)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pointmap"].shape, torch.Size([3, image_height, image_width]))

        # Head and post-processing are exactly identical to original code but differences from backbone
        # get amplified after scaling and resizing so we need to relax the tolerance here.
        EXPECTED_POSTPROCESSED_POINTMAP = Expectations(
            {("cuda", None): [[0.0771, 0.1335, 0.3025], [-0.1179, 0.2904, 0.7140], [0.0337, 0.3037, 0.4390]]}
        )
        expected_postprocessed_pointmap = torch.tensor(
            EXPECTED_POSTPROCESSED_POINTMAP.get_expectation(), device=torch_device
        )
        torch.testing.assert_close(
            result[0]["pointmap"][0, :3, :3], expected_postprocessed_pointmap, rtol=1e-2, atol=1e-2
        )

    @slow
    def test_inference_matting(self):
        model = Sapiens2ForImageMatting.from_pretrained("facebook/sapiens2-matting-1b").eval().to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        image_height, image_width = image.shape[-2:]
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsInstance(outputs, Sapiens2ImageMattingOutput)
        _, _, height, width = inputs["pixel_values"].shape
        self.assertEqual(outputs.foregrounds.shape, torch.Size([1, 3, height, width]))
        self.assertEqual(outputs.alphas.shape, torch.Size([1, 1, height, width]))

        # Difference due to cv2 vs torchvision pre-processing. Model outputs are equal on same tensor input.
        EXPECTED_FOREGROUNDS = Expectations(
            {("cuda", None): [[0.1432, 0.2051, 0.3043], [0.1889, 0.2681, 0.3509], [0.2511, 0.3076, 0.4047]]}
        )
        expected_foregrounds = torch.tensor(EXPECTED_FOREGROUNDS.get_expectation(), device=torch_device)
        torch.testing.assert_close(
            outputs.foregrounds[0, 0, 100:103, 100:103], expected_foregrounds, rtol=1e-2, atol=1e-2
        )

        background = torch.tensor([177, 64, 0], device=torch_device).view(3, 1, 1)
        result = image_processor.post_process_image_matting(
            outputs, target_sizes=[(image_height, image_width)], backgrounds=background
        )
        self.assertEqual(len(result), 1)

        alpha = result[0]["alpha"]
        foreground = result[0]["foreground"]
        composite = result[0]["composite"]
        self.assertEqual(alpha.shape, (1, image_height, image_width))
        self.assertEqual(foreground.shape, (3, image_height, image_width))
        self.assertEqual(composite.shape, (3, image_height, image_width))

        EXPECTED_ALPHA = Expectations(
            {
                ("cuda", None): [
                    [0.99995, 0.9999123, 0.9997628],
                    [0.99991906, 0.9997431, 0.99754137],
                    [0.9997362, 0.99711365, 0.9444071],
                ]
            }
        )
        expected_alpha = torch.tensor(EXPECTED_ALPHA.get_expectation(), device=torch_device)
        torch.testing.assert_close(alpha[0, 300:303, 300:303], expected_alpha, rtol=1e-3, atol=1e-3)

        EXPECTED_FOREGROUND = Expectations(
            {
                ("cuda", None): [
                    [0.7175647, 0.6906685, 0.65860075],
                    [0.7162684, 0.6867891, 0.64463294],
                    [0.6924842, 0.67141336, 0.5356377],
                ]
            }
        )
        expected_foreground = torch.tensor(EXPECTED_FOREGROUND.get_expectation(), device=torch_device)
        torch.testing.assert_close(foreground[0, 300:303, 300:303], expected_foreground, rtol=1e-2, atol=1e-2)

        EXPECTED_COMPOSITE = Expectations({("cuda", None): [[182, 176, 167], [182, 175, 164], [176, 171, 136]]})
        expected_composite = torch.tensor(EXPECTED_COMPOSITE.get_expectation(), dtype=torch.uint8, device=torch_device)
        torch.testing.assert_close(composite[0, 300:303, 300:303], expected_composite, rtol=0, atol=1)


@require_torch
class Sapiens2BackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (Sapiens2Backbone,) if is_torch_available() else ()
    config_class = Sapiens2Config

    def setUp(self):
        self.model_tester = Sapiens2ModelTester(self)
