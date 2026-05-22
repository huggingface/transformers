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
        Sapiens2ForMatting,
        Sapiens2ForNormalEstimation,
        Sapiens2ForPointmapEstimation,
        Sapiens2ForPoseEstimation,
        Sapiens2ForSemanticSegmentation,
        Sapiens2Model,
    )
    from transformers.modeling_outputs import SemanticSegmenterOutput
    from transformers.models.sapiens2.modeling_sapiens2 import (
        Sapiens2MattingOutput,
        Sapiens2NormalEstimatorOutput,
        Sapiens2PointmapEstimatorOutput,
        Sapiens2PoseEstimatorOutput,
    )


REVISION = "refs/pr/1"


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
            # Head config sized to satisfy all model conversion patterns in test_reverse_loading_mapping
            head_upsample_out_channels=[8, 4, 4, 4],
            head_upsample_kernel_sizes=[4, 4, 4, 4],
            head_conv_out_channels=[4, 4, 4],
            head_conv_kernel_sizes=[1, 1, 1],
            head_scale_conv_out_channels=[8, 4, 4],
            head_scale_conv_kernel_sizes=[1, 1, 1],
            head_scale_final_hidden_sizes=[8],
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

    def create_and_check_for_normal_estimation(self, config, pixel_values, labels):
        model = Sapiens2ForNormalEstimation(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # PixelShuffle: Conv2d(padding=(ks-1)//2) then shuffle(2) — size per layer: (h + 2p - ks + 1) * 2
        expected_h = config.image_size // self.patch_size
        for ks in config.head_upsample_kernel_sizes:
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
        model = Sapiens2ForMatting(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        expected_h = config.image_size // self.patch_size
        for ks in config.head_upsample_kernel_sizes:
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
        for ks in config.head_upsample_kernel_sizes:
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
        pixel_values = floats_tensor([self.batch_size, self.num_channels, config.image_size, config.image_size])
        labels = None
        return config, pixel_values, labels

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
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
            Sapiens2ForMatting,
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

    def test_batching_equivalence(self, atol=1e-4, rtol=1e-4):
        # InstanceNorm2d in the decoder heads computes per-instance statistics; different batch
        # sizes can trigger different parallelisation paths on CPU, producing O(1e-5) FP differences.
        super().test_batching_equivalence(atol=atol, rtol=rtol)

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

    def test_post_process_normal_estimation(self):
        image_processor = Sapiens2ImageProcessor()
        batch_size = 2
        num_labels = 3
        height = width = 16
        outputs = Sapiens2NormalEstimatorOutput(normals=torch.randn(batch_size, num_labels, height, width))

        # without target_sizes: spatial dims match normals, values are L2-normalized
        result = image_processor.post_process_normal_estimation(outputs)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0].shape, torch.Size([num_labels, height, width]))
        norms = result[0].norm(p=2, dim=0)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

        # with target_sizes: output is resized before normalization
        target_sizes = [(height * 2, width * 2)] * batch_size
        result = image_processor.post_process_normal_estimation(outputs, target_sizes=target_sizes)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0].shape, torch.Size([num_labels, height * 2, width * 2]))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_normal_estimation(outputs, target_sizes=[(100, 100)])

    def test_post_process_pointmap(self):
        image_processor = Sapiens2ImageProcessor()
        batch_size = 2
        num_labels = 3
        height = width = 16
        outputs = Sapiens2PointmapEstimatorOutput(pointmaps=torch.randn(batch_size, num_labels, height, width))

        # without target_sizes: spatial dims match pointmap
        result = image_processor.post_process_pointmap(outputs)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0].shape, torch.Size([num_labels, height, width]))

        # with target_sizes: output is resized to requested size
        target_sizes = [(height * 2, width * 2)] * batch_size
        result = image_processor.post_process_pointmap(outputs, target_sizes=target_sizes)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0].shape, torch.Size([num_labels, height * 2, width * 2]))

        # with scales: scale division is applied
        scale = torch.tensor([[2.0], [0.5]])
        outputs_with_scale = Sapiens2PointmapEstimatorOutput(
            pointmaps=torch.ones(batch_size, num_labels, height, width), scales=scale
        )
        result = image_processor.post_process_pointmap(outputs_with_scale)
        torch.testing.assert_close(result[0], torch.full((num_labels, height, width), 0.5))
        torch.testing.assert_close(result[1], torch.full((num_labels, height, width), 2.0))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_pointmap(outputs, target_sizes=[(100, 100)])

    def test_post_process_matting(self):
        image_processor = Sapiens2ImageProcessor()
        batch_size = 2
        height = width = 16
        outputs = Sapiens2MattingOutput(
            foregrounds=torch.rand(batch_size, 3, height, width),
            alphas=torch.rand(batch_size, 1, height, width),
        )

        # without target_sizes: spatial dims unchanged
        result = image_processor.post_process_matting(outputs)
        self.assertEqual(len(result), batch_size)
        self.assertEqual(result[0]["foreground"].shape, torch.Size([3, height, width]))
        self.assertEqual(result[0]["alpha"].shape, torch.Size([1, height, width]))
        # values stay in [0, 1]
        self.assertGreaterEqual(result[0]["alpha"].min().item(), 0.0)
        self.assertLessEqual(result[0]["alpha"].max().item(), 1.0)

        # with target_sizes: output is resized
        target_sizes = [(height * 2, width * 2)] * batch_size
        result = image_processor.post_process_matting(outputs, target_sizes=target_sizes)
        self.assertEqual(result[0]["foreground"].shape, torch.Size([3, height * 2, width * 2]))

        # mismatched batch size raises ValueError
        with self.assertRaises(ValueError):
            image_processor.post_process_matting(outputs, target_sizes=[(100, 100)])

    @slow
    def test_model_from_pretrained(self):
        model = Sapiens2Model.from_pretrained("facebook/sapiens2-pretrain-0.4b", revision=REVISION)
        self.assertIsNotNone(model)


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
        return (
            Sapiens2ImageProcessor.from_pretrained("facebook/sapiens2-pretrain-0.4b", revision=REVISION)
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_no_head(self):
        model = (
            Sapiens2Model.from_pretrained("facebook/sapiens2-pretrain-0.4b", revision=REVISION).eval().to(torch_device)
        )

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
        model = (
            Sapiens2ForSemanticSegmentation.from_pretrained("facebook/sapiens2-seg-0.4b", revision=REVISION)
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
        model = (
            Sapiens2ForPoseEstimation.from_pretrained("facebook/sapiens2-pose-0.4b", revision=REVISION)
            .eval()
            .to(torch_device)
        )

        image_processor = self.default_image_processor
        image = prepare_img()

        # person bbox in COCO format (x, y, w, h)
        boxes = [[[2.7080630e02, 5.7221174e-01, 2.9409006e02, 3.7946970e02]]]
        inputs = image_processor(image, boxes=boxes, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        heatmaps = outputs.heatmaps
        self.assertEqual(heatmaps.shape, torch.Size([1, model.config.num_labels, 256, 192]))
        expected_heatmaps = torch.tensor(
            [
                [0.26140, 0.24656, 0.21673],
                [0.33708, 0.31597, 0.28028],
                [0.41624, 0.39270, 0.35014],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(heatmaps[0, 0, 70:73, 70:73], expected_heatmaps, rtol=1e-3, atol=1e-3)

        results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 1)
        person = results[0][0]

        keypoints = person["keypoints"]
        expected_keypoints = torch.tensor(
            [
                [364.33920111, 97.92528764],
                [373.25104943, 80.97749201],
                [353.21072316, 83.38954486],
            ]
        )
        torch.testing.assert_close(keypoints[:3], expected_keypoints, rtol=1e-3, atol=1e-3)

        scores = person["scores"]
        expected_scores = torch.tensor([1.0007433, 0.9987416, 1.0015154])
        torch.testing.assert_close(scores[:3], expected_scores, rtol=1e-3, atol=1e-3)

        bbox = person["bbox"]
        # Padded, aspect-ratio-corrected xyxy box derived from center ± scale/2
        expected_bbox = torch.tensor([234.04503, -54.76801, 601.65761, 435.38211])
        torch.testing.assert_close(bbox, expected_bbox, rtol=1e-3, atol=1e-3)

        # Passing normalised boxes + target_sizes must produce identical results
        img_w, img_h = 640, 432
        norm_boxes = [[[b / s for b, s in zip(boxes[0][0], [img_w, img_h, img_w, img_h])]]]
        results_norm = image_processor.post_process_pose_estimation(
            outputs, boxes=norm_boxes, target_sizes=[(img_w, img_h)]
        )
        torch.testing.assert_close(results_norm[0][0]["keypoints"], keypoints)
        torch.testing.assert_close(results_norm[0][0]["bbox"], bbox)

        # Test flipping
        flipped_inputs = {"pixel_values": inputs["pixel_values"].flip(-1)}
        flip_pairs = torch.tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
                [13, 14],
                [15, 18],
                [16, 19],
                [17, 20],
                [21, 42],
                [22, 43],
                [23, 44],
                [24, 45],
                [25, 46],
                [26, 47],
                [27, 48],
                [28, 49],
                [29, 50],
                [30, 51],
                [31, 52],
                [32, 53],
                [33, 54],
                [34, 55],
                [35, 56],
                [36, 57],
                [37, 58],
                [38, 59],
                [39, 60],
                [40, 61],
                [41, 62],
                [63, 64],
                [65, 66],
                [67, 68],
                [78, 87],
                [79, 88],
                [80, 89],
                [81, 90],
                [82, 91],
                [83, 93],
                [84, 92],
                [85, 95],
                [86, 94],
                [96, 120],
                [97, 121],
                [98, 122],
                [99, 123],
                [100, 124],
                [101, 125],
                [102, 126],
                [103, 127],
                [104, 128],
                [105, 129],
                [106, 130],
                [107, 131],
                [108, 132],
                [109, 133],
                [110, 134],
                [111, 135],
                [112, 136],
                [113, 137],
                [114, 138],
                [115, 139],
                [116, 140],
                [117, 141],
                [118, 142],
                [119, 143],
                [144, 161],
                [145, 162],
                [146, 163],
                [147, 164],
                [148, 165],
                [149, 166],
                [150, 167],
                [151, 168],
                [152, 169],
                [153, 170],
                [154, 171],
                [155, 172],
                [156, 173],
                [157, 174],
                [158, 175],
                [159, 176],
                [160, 177],
                [180, 181],
                [182, 185],
                [183, 186],
                [184, 187],
                [188, 189],
                [192, 193],
                [194, 195],
                [196, 199],
                [197, 198],
                [200, 203],
                [201, 202],
                [204, 205],
                [208, 209],
                [210, 211],
                [212, 215],
                [213, 214],
                [216, 219],
                [217, 218],
                [220, 246],
                [221, 247],
                [222, 248],
                [223, 249],
                [224, 250],
                [225, 251],
                [226, 252],
                [227, 253],
                [228, 254],
                [229, 255],
                [230, 256],
                [231, 257],
                [232, 258],
                [233, 259],
                [234, 260],
                [235, 261],
                [236, 262],
                [237, 263],
                [238, 264],
                [239, 265],
                [240, 266],
                [241, 267],
                [242, 268],
                [243, 269],
                [244, 270],
                [245, 271],
                [272, 281],
                [273, 286],
                [274, 285],
                [275, 284],
                [276, 283],
                [277, 282],
                [278, 289],
                [279, 288],
                [280, 287],
                [290, 299],
                [291, 304],
                [292, 303],
                [293, 302],
                [294, 301],
                [295, 300],
                [296, 307],
                [297, 306],
                [298, 305],
            ]
        )

        with torch.no_grad():
            flipped_outputs = model(**flipped_inputs, flip_pairs=flip_pairs)

        flipped_heatmaps = flipped_outputs.heatmaps
        expected_flipped_heatmaps = torch.tensor(
            [
                [0.27348, 0.25426, 0.22496],
                [0.34877, 0.32563, 0.28418],
                [0.43967, 0.40607, 0.35721],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(
            flipped_heatmaps[0, 0, 70:73, 70:73], expected_flipped_heatmaps, rtol=1e-3, atol=1e-3
        )

        final_heatmaps = (heatmaps + flipped_heatmaps) / 2.0
        final_outputs = Sapiens2PoseEstimatorOutput(heatmaps=final_heatmaps)
        final_results = image_processor.post_process_pose_estimation(final_outputs, boxes=boxes)
        self.assertEqual(len(final_results), 1)
        self.assertEqual(len(final_results[0]), 1)

        final_person = final_results[0][0]
        final_keypoints = final_person["keypoints"]
        expected_final_keypoints = torch.tensor(
            [[364.14644305, 97.99268751], [373.66756367, 81.19966519], [353.4574526, 83.647911]],
        )
        torch.testing.assert_close(final_keypoints[:3], expected_final_keypoints, rtol=1e-3, atol=1e-3)

        final_scores = final_person["scores"]
        expected_final_scores = torch.tensor([1.0064079, 0.98746514, 0.99821794])
        torch.testing.assert_close(final_scores[:3], expected_final_scores, rtol=1e-3, atol=1e-3)

        final_bbox = final_person["bbox"]
        torch.testing.assert_close(final_bbox, expected_bbox, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_normal_estimation(self):
        model = (
            Sapiens2ForNormalEstimation.from_pretrained("facebook/sapiens2-normal-0.4b", revision=REVISION)
            .eval()
            .to(torch_device)
        )

        image_processor = Sapiens2ImageProcessor.from_pretrained("facebook/sapiens2-normal-0.4b", revision=REVISION)
        image = prepare_img()
        image_height, image_width = image.shape[-2:]
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        _, _, height, width = inputs["pixel_values"].shape
        self.assertEqual(outputs.normals.shape, torch.Size([1, 3, height, width]))

        # We can get closer to expected values by using cv2 resize instead of torchvision.
        expected_normals = torch.tensor(
            [[0.9577, 1.8808, 0.9826], [1.6904, 1.7351, 1.9120], [2.4828, 1.9887, 2.5168]],
            device=torch_device,
        )
        torch.testing.assert_close(outputs.normals[0, 0, :3, :3], expected_normals, rtol=1e-2, atol=1e-2)

        result = image_processor.post_process_normal_estimation(outputs, source_sizes=[(image_height, image_width)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, torch.Size([3, 432, 640]))

        expected_postprocessed_normals = torch.tensor(
            [[-0.8266, -0.7899, -0.7512], [-0.8227, -0.7843, -0.7440], [-0.8098, -0.7721, -0.7318]],
            device=torch_device,
        )
        torch.testing.assert_close(result[0][0, :3, :3], expected_postprocessed_normals, rtol=1e-2, atol=1e-2)

    @slow
    def test_inference_pointmap_estimation(self):
        model = (
            Sapiens2ForPointmapEstimation.from_pretrained("facebook/sapiens2-pointmap-0.4b", revision=REVISION)
            .eval()
            .to(torch_device)
        )

        image_processor = Sapiens2ImageProcessor.from_pretrained("facebook/sapiens2-pointmap-0.4b", revision=REVISION)
        image = prepare_img()
        image_height, image_width = image.shape[-2:]
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsInstance(outputs, Sapiens2PointmapEstimatorOutput)
        _, _, height, width = inputs["pixel_values"].shape
        self.assertEqual(outputs.pointmaps.shape, torch.Size([1, 3, height, width]))
        self.assertEqual(outputs.scales.shape, torch.Size([1, 1]))

        expected_scale = torch.tensor([[0.9931]], device=torch_device)
        torch.testing.assert_close(outputs.scales, expected_scale, rtol=1e-3, atol=1e-3)

        expected_pointmap = torch.tensor(
            [[-0.0096, -0.0567, -0.0460], [-0.0657, -0.0583, -0.0688], [-0.1035, -0.0363, -0.0659]],
            device=torch_device,
        )
        torch.testing.assert_close(outputs.pointmaps[0, 0, :3, :3], expected_pointmap, rtol=1e-3, atol=1e-3)

        result = image_processor.post_process_pointmap(outputs, source_sizes=[(image_height, image_width)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, torch.Size([3, image_height, image_width]))

        # Head and post-processing are exactly identical to original code but differences from backbone
        # get amplified after scaling and resizing so we need to relax the tolerance here.
        expected_postprocessed_pointmap = torch.tensor(
            [[0.0771, 0.1335, 0.3025], [-0.1179, 0.2904, 0.7140], [0.0337, 0.3037, 0.4390]],
            device=torch_device,
        )
        torch.testing.assert_close(result[0][0, :3, :3], expected_postprocessed_pointmap, rtol=1e-2, atol=1e-2)

    @slow
    def test_inference_matting(self):
        model = (
            Sapiens2ForMatting.from_pretrained("facebook/sapiens2-matting-1b", revision=REVISION)
            .eval()
            .to(torch_device)
        )

        image_processor = self.default_image_processor
        image = prepare_img()
        image_height, image_width = image.shape[-2:]
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsInstance(outputs, Sapiens2MattingOutput)
        _, _, height, width = inputs["pixel_values"].shape
        self.assertEqual(outputs.foregrounds.shape, torch.Size([1, 3, height, width]))
        self.assertEqual(outputs.alphas.shape, torch.Size([1, 1, height, width]))

        # Difference due to cv2 vs torchvision pre-processing. Model outputs are equal on same tensor input.
        expected_foregrounds = torch.tensor(
            [
                [0.1432, 0.2051, 0.3043],
                [0.1889, 0.2681, 0.3509],
                [0.2511, 0.3076, 0.4047],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(
            outputs.foregrounds[0, 0, 100:103, 100:103], expected_foregrounds, rtol=1e-2, atol=1e-2
        )

        background = torch.tensor([177, 64, 0], device=torch_device).view(3, 1, 1)
        result = image_processor.post_process_matting(
            outputs, target_sizes=[(image_height, image_width)], backgrounds=background
        )
        self.assertEqual(len(result), 1)

        alpha = result[0]["alpha"]
        foreground = result[0]["foreground"]
        composite = result[0]["composite"]
        self.assertEqual(alpha.shape, (1, image_height, image_width))
        self.assertEqual(foreground.shape, (3, image_height, image_width))
        self.assertEqual(composite.shape, (3, image_height, image_width))

        expected_alpha = torch.tensor(
            [
                [0.99995, 0.9999123, 0.9997628],
                [0.99991906, 0.9997431, 0.99754137],
                [0.9997362, 0.99711365, 0.9444071],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(torch.tensor(alpha[0, 300:303, 300:303]), expected_alpha, rtol=1e-3, atol=1e-3)

        expected_foreground = torch.tensor(
            [
                [0.7175647, 0.6906685, 0.65860075],
                [0.7162684, 0.6867891, 0.64463294],
                [0.6924842, 0.67141336, 0.5356377],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(
            torch.tensor(foreground[0, 300:303, 300:303]), expected_foreground, rtol=1e-2, atol=1e-2
        )

        expected_composite = torch.tensor(
            [[182, 176, 167], [182, 175, 164], [176, 171, 136]],
            dtype=torch.uint8,
            device=torch_device,
        )
        torch.testing.assert_close(torch.tensor(composite[0, 300:303, 300:303]), expected_composite, rtol=0, atol=1)


@require_torch
class Sapiens2BackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (Sapiens2Backbone,) if is_torch_available() else ()
    config_class = Sapiens2Config

    def setUp(self):
        self.model_tester = Sapiens2ModelTester(self)
