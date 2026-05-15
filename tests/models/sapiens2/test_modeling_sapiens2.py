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

from transformers import Sapiens2Config, Sapiens2ImageProcessor
from transformers.image_utils import load_image_as_tensor
from transformers.testing_utils import require_cv2, require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        Sapiens2Backbone,
        Sapiens2ForPoseEstimation,
        Sapiens2ForSemanticSegmentation,
        Sapiens2Model,
    )
    from transformers.modeling_outputs import SemanticSegmenterOutput


if is_vision_available():
    pass


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
        )

    # TODO(guarin): Check if multiple get_config methods is the best approach here.
    def get_config_for_semantic_segmentation(self):
        config = self.get_config()
        config.num_labels = 3
        config.head_upsample_out_channels = [32, 16, 8, 8]
        config.head_upsample_kernel_sizes = [4, 4, 4, 4]
        config.head_conv_out_channels = [8, 8, 8]
        config.head_conv_kernel_sizes = [1, 1, 1]
        return config

    def get_config_for_pose_estimation(self):
        config = self.get_config()
        config.num_labels = 3
        config.head_upsample_out_channels = [32, 16]
        config.head_upsample_kernel_sizes = [4, 4]
        config.head_conv_out_channels = [16, 16, 16]
        config.head_conv_kernel_sizes = [1, 1, 1]
        return config

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
            self.parent.assertGreater(h, 0)
            self.parent.assertGreater(w, 0)

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
        expected_h = patch_height * (2 ** len(config.head_upsample_out_channels))
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
        # patch_height = image_size // patch_size = 30 // 2 = 15
        # 2 deconv layers with stride=2: 15 * 2^2 = 60
        patch_height = self.image_size // self.patch_size
        expected_h = patch_height * (2 ** len(config.head_upsample_out_channels))
        self.parent.assertEqual(
            result.heatmaps.shape,
            (self.batch_size, config.num_labels, expected_h, expected_h),
        )

    def prepare_config_and_inputs_for_semantic_segmentation(self):
        config = self.get_config_for_semantic_segmentation()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = ids_tensor([self.batch_size, self.image_size, self.image_size], config.num_labels)
        return config, pixel_values, labels

    def prepare_config_and_inputs_for_pose_estimation(self):
        config = self.get_config_for_pose_estimation()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        labels = ids_tensor([self.batch_size, self.image_size, self.image_size], config.num_labels)
        return config, pixel_values, labels

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config_for_semantic_segmentation()
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
        (Sapiens2Model, Sapiens2Backbone, Sapiens2ForSemanticSegmentation, Sapiens2ForPoseEstimation)
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
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_pose_estimation()
        self.model_tester.create_and_check_for_pose_estimation(*config_and_inputs)

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

    @unittest.skip(reason="Sapiens2 does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_post_process_semantic_segmentation(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        image_processor = Sapiens2ImageProcessor()

        batch_size = self.model_tester.batch_size
        height = width = self.model_tester.image_size
        outputs = SemanticSegmenterOutput(logits=torch.randn(batch_size, config.num_labels, height, width))

        # without target_sizes: spatial dims match logits
        segmentation = image_processor.post_process_semantic_segmentation(outputs)
        self.assertEqual(len(segmentation), batch_size)
        self.assertEqual(segmentation[0].shape, torch.Size([height, width]))

        # with target_sizes: output is resized to requested size
        target_sizes = [(height * 2, width * 2)] * batch_size
        segmentation = image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        self.assertEqual(len(segmentation), batch_size)
        self.assertEqual(segmentation[0].shape, torch.Size([height * 2, width * 2]))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(100, 100)])

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/sapiens2-pretrain-0.4b"
        # TODO(guarin): remove config. transformers_weights required for now because original checkpoints are called
        # "sapiens2_0.4b_pretrain.safetensors" instead of "model.safetensors"
        config = Sapiens2Config()
        config.transformers_weights = "sapiens2_0.4b_pretrain.safetensors"
        model = Sapiens2Model.from_pretrained(model_name, config=config)
        self.assertIsNotNone(model)


def prepare_img():
    image = load_image_as_tensor("./tests/fixtures/tests_samples/COCO/000000004016.png")
    return image


@require_torch
@require_vision
class Sapiens2ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        # TODO(guarin): switch to AutoImageProcessor once it works properly
        # return AutoImageProcessor.from_pretrained("facebook/sapiens2-pretrain-0.4b") if is_vision_available() else None
        return Sapiens2ImageProcessor() if is_vision_available() else None

    @slow
    def test_inference_no_head(self):
        # TODO(guarin): remove config. transformers_weights required for now because original checkpoints are called
        # "sapiens2_0.4b_pretrain.safetensors" instead of "model.safetensors"
        config = Sapiens2Config()
        config.transformers_weights = "sapiens2_0.4b_pretrain.safetensors"
        model = Sapiens2Model.from_pretrained("facebook/sapiens2-pretrain-0.4b", config=config).eval().to(torch_device)

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
        expected_slice = torch.tensor([-0.09233, -0.00107, -0.12215, 0.07374, -0.03773], device=torch_device)
        torch.testing.assert_close(last_layer_cls_token[0, :5], expected_slice, rtol=1e-3, atol=1e-3)

        last_layer_register_tokens = outputs.last_hidden_state[:, 1 : model.config.num_register_tokens + 1]
        expected_slice = torch.tensor([0.08412, 0.04387, 0.05709, -0.04962, 0.03715], device=torch_device)
        torch.testing.assert_close(last_layer_register_tokens[0, 0, :5], expected_slice, rtol=1e-3, atol=1e-3)

        last_layer_patch_tokens = outputs.last_hidden_state[:, model.config.num_register_tokens + 1 :]
        expected_slice = torch.tensor([0.14232, -0.11947, -0.05910, -0.09457, -0.11410], device=torch_device)
        torch.testing.assert_close(last_layer_patch_tokens[0, 0, :5], expected_slice, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_semantic_segmentation(self):
        # TODO(guarin): remove config. transformers_weights required for now because original checkpoints are called
        # "sapiens2_0.4b_seg.safetensors" instead of "model.safetensors"
        config = Sapiens2Config(
            num_labels=29,
            head_upsample_out_channels=[512, 256, 128, 64],
            head_upsample_kernel_sizes=[4, 4, 4, 4],
            head_conv_out_channels=[64, 64, 64],
            head_conv_kernel_sizes=[1, 1, 1],
        )
        config.transformers_weights = "sapiens2_0.4b_seg.safetensors"
        model = (
            Sapiens2ForSemanticSegmentation.from_pretrained("facebook/sapiens2-seg-0.4b", config=config)
            .eval()
            .to(torch_device)
        )

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

        expected_slice = torch.tensor(
            [[3.45260, 5.55483, 6.57901], [5.71913, 7.21420, 8.11209], [6.82645, 7.98208, 8.31385]],
            device=torch_device,
        )
        torch.testing.assert_close(logits[0, 0, :3, :3], expected_slice, rtol=1e-3, atol=1e-3)

        # verify post-processing without resizing: output shape matches model input resolution
        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs)
        self.assertEqual(len(segmentation), 1)
        self.assertEqual(segmentation[0].shape, torch.Size([height, width]))

        # verify post-processing with target_sizes
        target_size = (height // 2, width // 2)
        segmentation = image_processor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[target_size])
        self.assertEqual(len(segmentation), 1)
        self.assertEqual(segmentation[0].shape, torch.Size(target_size))

        expected_class_ids = torch.tensor([[4, 3, 3], [3, 3, 3], [3, 3, 3]], device=torch_device)
        torch.testing.assert_close(segmentation[0][50:53, 50:53], expected_class_ids)

    @require_cv2
    @slow
    def test_inference_pose_estimation(self):
        # TODO(guarin): remove config. transformers_weights required for now because original checkpoints are called
        # "sapiens2_0.4b_pose.safetensors" instead of "model.safetensors"
        config = Sapiens2Config(
            num_labels=308,
            head_upsample_out_channels=[1024, 768],
            head_upsample_kernel_sizes=[4, 4],
            head_conv_out_channels=[512, 512, 256],
            head_conv_kernel_sizes=[1, 1, 1],
        )
        config.transformers_weights = "sapiens2_0.4b_pose.safetensors"
        model = (
            Sapiens2ForPoseEstimation.from_pretrained("facebook/sapiens2-pose-0.4b", config=config)
            .eval()
            .to(torch_device)
        )

        image_processor = self.default_image_processor
        image = prepare_img()
        _, img_height, img_width = image.shape
        # Person bbox in xyxy format covering the full COCO test image
        boxes = [[[0, 0, img_width, img_height]]]
        inputs = image_processor(image, boxes=boxes, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(outputs.heatmaps.shape, torch.Size([1, model.config.num_labels, 192, 256]))

        results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)
        person = results[0][0]
        # All keypoint x-coordinates should be within the image width
        self.assertTrue(person["keypoints"][:, 0].min().item() >= 0)
        self.assertTrue(person["keypoints"][:, 0].max().item() <= img_width)
        # All keypoint y-coordinates should be within the image height
        self.assertTrue(person["keypoints"][:, 1].min().item() >= 0)
        self.assertTrue(person["keypoints"][:, 1].max().item() <= img_height)


@require_torch
class Sapiens2BackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (Sapiens2Backbone,) if is_torch_available() else ()
    config_class = Sapiens2Config

    def setUp(self):
        self.model_tester = Sapiens2ModelTester(self)
