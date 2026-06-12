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
"""Testing suite for the PyTorch TIPSv2-DPT model."""

import unittest
from functools import cached_property

from transformers import Tipsv2DptConfig
from transformers.testing_utils import Expectations, require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        Tipsv2DptForDepthEstimation,
        Tipsv2DptForNormalEstimation,
        Tipsv2DptForSemanticSegmentation,
        Tipsv2DptImageProcessor,
        Tipsv2DptModel,
    )
    from transformers.image_utils import load_image_as_tensor
    from transformers.models.tipsv2_dpt.modeling_tipsv2_dpt import Tipsv2DptNormalEstimatorOutput, Tipsv2DptOutput


class Tipsv2DptModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        image_size=28,
        patch_size=14,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_ratio=2,
        num_register_tokens=1,
        neck_hidden_sizes=(4, 8),
        fusion_hidden_size=8,
        reassemble_factors=(4, 2),
        num_depth_bins=32,
        min_depth=0.001,
        max_depth=10.0,
        num_labels=5,
        is_training=False,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.reassemble_factors = reassemble_factors
        self.num_depth_bins = num_depth_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_labels = num_labels
        self.is_training = is_training
        self.scope = scope

        self.patch_height = image_size // patch_size
        self.patch_width = image_size // patch_size
        self.seq_length = 1 + num_register_tokens + self.patch_height * self.patch_width

        # output spatial resolution: 2 fusion layers, each doubling resolution
        # starting from (patch_h * factor_min) upsampled 2 times by 2
        # deepest reassemble factor is 2 → 4x4, then 2× doublings → 16×16
        self.decoder_height = int(self.patch_height * min(reassemble_factors)) * 2 ** len(neck_hidden_sizes)
        self.decoder_width = int(self.patch_width * min(reassemble_factors)) * 2 ** len(neck_hidden_sizes)

    def get_config(self):
        from transformers import Tipsv2VisionConfig

        backbone_config = Tipsv2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_ratio=self.mlp_ratio,
            num_register_tokens=self.num_register_tokens,
            use_swiglu_ffn=False,
            out_indices=list(range(1, self.num_hidden_layers + 1))[-4:],
            apply_layernorm=True,
            reshape_hidden_states=False,
        )
        return Tipsv2DptConfig(
            backbone_config=backbone_config,
            neck_hidden_sizes=self.neck_hidden_sizes,
            fusion_hidden_size=self.fusion_hidden_size,
            reassemble_factors=self.reassemble_factors,
            num_depth_bins=self.num_depth_bins,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        return config, {"pixel_values": pixel_values}

    def create_and_check_model(self, config, pixel_values):
        model = Tipsv2DptModel(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(pixel_values)
        self.parent.assertIsInstance(outputs, Tipsv2DptOutput)
        self.parent.assertEqual(
            outputs.predicted_depth.shape,
            (self.batch_size, self.decoder_height, self.decoder_width),
        )
        self.parent.assertEqual(
            outputs.normals.shape,
            (self.batch_size, 3, self.decoder_height, self.decoder_width),
        )
        self.parent.assertEqual(
            outputs.logits.shape,
            (self.batch_size, self.num_labels, self.decoder_height, self.decoder_width),
        )

    def create_and_check_for_depth_estimation(self, config, pixel_values):
        model = Tipsv2DptForDepthEstimation(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(pixel_values)
        self.parent.assertEqual(
            outputs.predicted_depth.shape,
            (self.batch_size, self.decoder_height, self.decoder_width),
        )

    def create_and_check_for_normal_estimation(self, config, pixel_values):
        model = Tipsv2DptForNormalEstimation(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(pixel_values)
        self.parent.assertEqual(
            outputs.normals.shape,
            (self.batch_size, 3, self.decoder_height, self.decoder_width),
        )

    def create_and_check_for_semantic_segmentation(self, config, pixel_values):
        model = Tipsv2DptForSemanticSegmentation(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(pixel_values)
        self.parent.assertEqual(
            outputs.logits.shape,
            (self.batch_size, self.num_labels, self.decoder_height, self.decoder_width),
        )

    def create_and_check_for_semantic_segmentation_with_loss(self, config, pixel_values):
        model = Tipsv2DptForSemanticSegmentation(config).to(torch_device).eval()
        labels = ids_tensor([self.batch_size, self.decoder_height, self.decoder_width], self.num_labels).to(
            torch_device
        )
        with torch.no_grad():
            outputs = model(pixel_values, labels=labels)
        self.parent.assertIsNotNone(outputs.loss)
        self.parent.assertEqual(outputs.loss.dim(), 0)


@require_torch
class Tipsv2DptModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Tipsv2DptModel,
            Tipsv2DptForDepthEstimation,
            Tipsv2DptForNormalEstimation,
            Tipsv2DptForSemanticSegmentation,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "depth-estimation": Tipsv2DptForDepthEstimation,
        }
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False
    has_attentions = False

    def setUp(self):
        self.model_tester = Tipsv2DptModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Tipsv2DptConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, pixel_values)

    def test_for_depth_estimation(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(config, pixel_values)

    def test_for_normal_estimation(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_normal_estimation(config, pixel_values)

    def test_for_semantic_segmentation(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(config, pixel_values)

    def test_for_semantic_segmentation_with_loss(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation_with_loss(config, pixel_values)

    @unittest.skip(reason="TIPSv2-DPT does not use input_ids or inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TIPSv2-DPT does not use input_ids or inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @unittest.skip(reason="TIPSv2-DPT does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass


def prepare_img():
    image = load_image_as_tensor(
        url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
        )
    )
    return image


@require_torch
@require_vision
class Tipsv2DptModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        # TODO: switch to Auto
        return Tipsv2DptImageProcessor()

    @slow
    def test_inference_model(self):
        # TODO: switch to google repo before merge
        model = Tipsv2DptModel.from_pretrained("guarin/tipsv2-b14-dpt", device_map=torch_device).eval()

        image = prepare_img()
        image_processor = self.default_image_processor
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsInstance(outputs, Tipsv2DptOutput)
        _, _, height, width = inputs["pixel_values"].shape
        expected_height, expected_width = 256, 256
        self.assertEqual(outputs.predicted_depth.shape, torch.Size([1, expected_height, expected_width]))
        self.assertEqual(outputs.normals.shape, torch.Size([1, 3, expected_height, expected_width]))
        self.assertEqual(
            outputs.logits.shape, torch.Size([1, model.config.num_labels, expected_height, expected_width])
        )

        EXPECTED_DEPTH = Expectations(
            {
                ("cuda", None): [
                    [1.49565, 1.48132, 1.45840],
                    [1.46976, 1.45724, 1.43721],
                    [1.42834, 1.41871, 1.40331],
                ],
            }
        )

        expected_depth = torch.tensor(EXPECTED_DEPTH.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.predicted_depth[0, :3, :3], expected_depth, rtol=1e-3, atol=1e-3)

        EXPECTED_NORMALS = Expectations(
            {
                ("cuda", None): [
                    [0.10756, 0.12425, 0.15097],
                    [0.11787, 0.13472, 0.16168],
                    [0.13436, 0.15146, 0.17882],
                ],
            }
        )

        expected_normals = torch.tensor(EXPECTED_NORMALS.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.normals[0, 0, :3, :3], expected_normals, rtol=1e-3, atol=1e-3)

        EXPECTED_SEG_LOGITS = Expectations(
            {
                ("cuda", None): [
                    [2.76751, 3.22963, 3.96903],
                    [3.30514, 3.87242, 4.78006],
                    [4.16536, 4.90088, 6.07772],
                ],
            }
        )

        expected_seg_logits = torch.tensor(EXPECTED_SEG_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.logits[0, 0, :3, :3], expected_seg_logits, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_depth_estimation(self):
        model = Tipsv2DptForDepthEstimation.from_pretrained("guarin/tipsv2-b14-dpt", device_map=torch_device).eval()

        image = prepare_img()
        image_processor = self.default_image_processor
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        _, _, height, width = inputs["pixel_values"].shape
        expected_height, expected_width = 256, 256
        self.assertEqual(outputs.predicted_depth.shape, torch.Size([1, expected_height, expected_width]))

        EXPECTED_DEPTH = Expectations(
            {
                ("cuda", None): [
                    [1.49565, 1.48132, 1.45840],
                    [1.46976, 1.45724, 1.43721],
                    [1.42834, 1.41871, 1.40331],
                ],
            }
        )

        expected_depth = torch.tensor(EXPECTED_DEPTH.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.predicted_depth[0, :3, :3], expected_depth, rtol=1e-3, atol=1e-3)

        result = image_processor.post_process_depth_estimation(outputs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["predicted_depth"].shape, torch.Size([expected_height, expected_width]))

        target_size = (height // 2, width // 2)
        result = image_processor.post_process_depth_estimation(outputs, target_sizes=[target_size])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["predicted_depth"].shape, torch.Size(target_size))

    @slow
    def test_inference_normal_estimation(self):
        model = Tipsv2DptForNormalEstimation.from_pretrained("guarin/tipsv2-b14-dpt", device_map=torch_device).eval()

        image = prepare_img()
        image_processor = self.default_image_processor
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        _, _, height, width = inputs["pixel_values"].shape
        expected_height, expected_width = 256, 256
        self.assertIsInstance(outputs, Tipsv2DptNormalEstimatorOutput)
        self.assertEqual(outputs.normals.shape, torch.Size([1, 3, expected_height, expected_width]))

        EXPECTED_NORMALS = Expectations(
            {
                ("cuda", None): [
                    [0.10756, 0.12425, 0.15097],
                    [0.11787, 0.13472, 0.16168],
                    [0.13436, 0.15146, 0.17882],
                ],
            }
        )

        expected_normals = torch.tensor(EXPECTED_NORMALS.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.normals[0, 0, :3, :3], expected_normals, rtol=1e-3, atol=1e-3)

        result = image_processor.post_process_normal_estimation(outputs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["normals"].shape, torch.Size([3, expected_height, expected_width]))
        norms = result[0]["normals"].norm(p=2, dim=0)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

        target_size = (height // 2, width // 2)
        result = image_processor.post_process_normal_estimation(outputs, target_sizes=[target_size])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["normals"].shape, torch.Size([3, *target_size]))
        norms = result[0]["normals"].norm(p=2, dim=0)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_semantic_segmentation(self):
        model = Tipsv2DptForSemanticSegmentation.from_pretrained(
            "guarin/tipsv2-b14-dpt", device_map=torch_device
        ).eval()

        image = prepare_img()
        image_processor = self.default_image_processor
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        _, _, height, width = inputs["pixel_values"].shape
        expected_height, expected_width = 256, 256
        self.assertEqual(
            outputs.logits.shape, torch.Size([1, model.config.num_labels, expected_height, expected_width])
        )

        EXPECTED_SEG_LOGITS = Expectations(
            {
                ("cuda", None): [
                    [2.76751, 3.22963, 3.96903],
                    [3.30514, 3.87242, 4.78006],
                    [4.16536, 4.90088, 6.07772],
                ],
            }
        )

        expected_seg_logits = torch.tensor(EXPECTED_SEG_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(outputs.logits[0, 0, :3, :3], expected_seg_logits, rtol=1e-3, atol=1e-3)

        segmentation = image_processor.post_process_semantic_segmentation(outputs)
        self.assertEqual(len(segmentation), 1)
        self.assertEqual(segmentation[0].shape, torch.Size([expected_height, expected_width]))

        EXPECTED_SEG_LABELS = Expectations(
            {
                ("cuda", None): [[23, 23, 23], [23, 23, 23], [23, 23, 23]],
            }
        )
        expected_seg_labels = torch.tensor(EXPECTED_SEG_LABELS.get_expectation(), device=torch_device)
        torch.testing.assert_close(segmentation[0][:3, :3], expected_seg_labels)

        target_size = (height // 2, width // 2)
        segmentation = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])
        self.assertEqual(len(segmentation), 1)
        self.assertEqual(segmentation[0].shape, torch.Size(target_size))
