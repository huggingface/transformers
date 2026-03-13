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
"""Testing suite for the PyTorch VidEoMT model."""

import unittest

import numpy as np

from transformers import VideomtConfig, VideomtForUniversalSegmentation
from transformers.testing_utils import (
    require_torch,
    require_torch_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn


if is_vision_available():
    from PIL import Image

    from transformers import AutoVideoProcessor


class VideomtForUniversalSegmentationTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_frames=1,
        image_size=40,
        patch_size=2,
        num_queries=5,
        num_register_tokens=19,
        num_labels=4,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_register_tokens = num_register_tokens
        self.is_training = is_training

        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1 + self.num_register_tokens

    def get_config(self):
        config = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_register_tokens": self.num_register_tokens,
            "num_queries": self.num_queries,
            "num_blocks": 1,
            "rope_parameters": {"rope_theta": 100.0},
        }
        return VideomtConfig(**config)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_frames, 3, self.image_size, self.image_size]).to(
            torch_device
        )

        config = self.get_config()
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VideomtForUniversalSegmentationTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (VideomtForUniversalSegmentation,) if is_torch_available() else ()
    pipeline_model_mapping = {}
    is_encoder_decoder = False
    test_missing_keys = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = VideomtForUniversalSegmentationTester(self)
        self.config_tester = ConfigTester(self, config_class=VideomtConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="VideoMT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), nn.Module)
            output_embeddings = model.get_output_embeddings()
            self.assertTrue(output_embeddings is None or isinstance(output_embeddings, nn.Linear))

    @unittest.skip(reason="VideoMT is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="VideoMT does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    def test_image_inputs_raise(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = VideomtForUniversalSegmentation(config).to(torch_device)
        model.eval()

        with self.assertRaisesRegex(ValueError, "only supports 5D video inputs"):
            model(inputs_dict["pixel_values"][:, 0])

    def test_pixel_values_videos_alias(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = VideomtForUniversalSegmentation(config).to(torch_device)
        model.eval()

        with torch.inference_mode():
            outputs = model(pixel_values_videos=inputs_dict["pixel_values"])

        expected_batch = inputs_dict["pixel_values"].shape[0] * inputs_dict["pixel_values"].shape[1]
        self.assertEqual(outputs.class_queries_logits.shape[0], expected_batch)
        self.assertEqual(outputs.masks_queries_logits.shape[0], expected_batch)


@slow
@require_torch
@require_vision
class VideomtForUniversalSegmentationIntegrationTest(unittest.TestCase):
    instance_model_id = "tue-mps/videomt-dinov2-small-ytvis2019"

    def prepare_video(self, num_frames=2):
        frame = np.array(Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png").convert("RGB"))
        return [frame.copy() for _ in range(num_frames)]

    def prepare_model_and_inputs(self, model_id, num_frames=2, dtype=None):
        model_kwargs = {"device_map": "auto"}
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        model = VideomtForUniversalSegmentation.from_pretrained(model_id, **model_kwargs)
        processor = AutoVideoProcessor.from_pretrained(model_id)
        video_frames = self.prepare_video(num_frames=num_frames)
        inputs = processor(videos=[video_frames], return_tensors="pt").to(model.device)
        return model, processor, video_frames, inputs

    def run_inference(self, model_id, num_frames=2, dtype=None):
        model, processor, video_frames, inputs = self.prepare_model_and_inputs(
            model_id, num_frames=num_frames, dtype=dtype
        )

        with torch.inference_mode():
            outputs = model(**inputs)

        self.assert_common_video_outputs(outputs, model, len(video_frames))
        return model, processor, video_frames, outputs

    def assert_common_video_outputs(self, outputs, model, num_frames):
        expected_mask_size = (
            (model.config.image_size // model.config.patch_size) * (2**model.config.num_upscale_blocks),
            (model.config.image_size // model.config.patch_size) * (2**model.config.num_upscale_blocks),
        )

        self.assertEqual(
            outputs.class_queries_logits.shape, (num_frames, model.config.num_queries, model.config.num_labels + 1)
        )
        self.assertEqual(
            outputs.masks_queries_logits.shape, (num_frames, model.config.num_queries, *expected_mask_size)
        )
        self.assertTrue(torch.isfinite(outputs.class_queries_logits.float()).all())
        self.assertTrue(torch.isfinite(outputs.masks_queries_logits.float()).all())

    def test_instance_segmentation_inference(self):
        _, processor, video_frames, outputs = self.run_inference(self.instance_model_id)

        target_sizes = [frame.shape[:2] for frame in video_frames]
        results = processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)

        self.assertEqual(len(results), len(video_frames))
        for frame, result in zip(video_frames, results):
            self.assertEqual(result["segmentation"].shape, frame.shape[:2])
            self.assertGreaterEqual(len(result["segments_info"]), 1)
            for info in result["segments_info"]:
                self.assertIn("label_id", info)
                self.assertIn("score", info)
                self.assertTrue(0.0 <= info["score"] <= 1.0)

    def test_semantic_segmentation_inference(self):
        _, processor, video_frames, outputs = self.run_inference(self.instance_model_id)

        target_sizes = [frame.shape[:2] for frame in video_frames]
        semantic_results = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        self.assertEqual(len(semantic_results), len(video_frames))
        for frame, seg_map in zip(video_frames, semantic_results):
            self.assertEqual(seg_map.shape, frame.shape[:2])
            self.assertFalse(torch.is_floating_point(seg_map))
            self.assertGreaterEqual(seg_map.min().item(), 0)
            self.assertLess(seg_map.max().item(), outputs.class_queries_logits.shape[-1] - 1)

    def test_panoptic_segmentation_inference(self):
        _, processor, video_frames, outputs = self.run_inference(self.instance_model_id)

        target_sizes = [frame.shape[:2] for frame in video_frames]
        panoptic_results = processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)

        self.assertEqual(len(panoptic_results), len(video_frames))
        for frame, result in zip(video_frames, panoptic_results):
            self.assertEqual(result["segmentation"].shape, frame.shape[:2])
            self.assertIsInstance(result["segments_info"], list)
            for info in result["segments_info"]:
                self.assertIn("label_id", info)
                self.assertIn("score", info)
                self.assertTrue(0.0 <= info["score"] <= 1.0)

    @require_torch_gpu
    def test_instance_segmentation_inference_bf16(self):
        _, _, _, outputs = self.run_inference(self.instance_model_id, dtype=torch.bfloat16)

        self.assertEqual(outputs.class_queries_logits.dtype, torch.bfloat16)
        self.assertEqual(outputs.masks_queries_logits.dtype, torch.bfloat16)
