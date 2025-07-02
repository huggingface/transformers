# coding=utf-8
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
"""Testing suite for the PyTorch V-JEPA2 model."""

import unittest

import numpy as np

from transformers import VJEPA2Config
from transformers.testing_utils import (
    is_flaky,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_video_processing_common import (
    prepare_video_inputs,
)


if is_torch_available():
    import torch
    from torch import nn

    from transformers import VJEPA2ForVideoClassification, VJEPA2Model


if is_vision_available():
    from PIL import Image

    from transformers import AutoVideoProcessor

VJEPA_HF_MODEL = "facebook/vjepa2-vitl-fpc64-256"


class VJEPA2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=16,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_frames=2,
        mlp_ratio=1,
        pred_hidden_size=32,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=2,
        pred_num_mask_tokens=10,
        is_training=False,
        attn_implementation="sdpa",
        mask_ratio=0.5,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_frames = num_frames
        self.mlp_ratio = mlp_ratio
        self.pred_hidden_size = pred_hidden_size
        self.pred_num_attention_heads = pred_num_attention_heads
        self.pred_num_hidden_layers = pred_num_hidden_layers
        self.pred_num_mask_tokens = pred_num_mask_tokens
        self.attn_implementation = attn_implementation
        self.is_training = is_training
        self.mask_ratio = mask_ratio

        num_patches = ((image_size // patch_size) ** 2) * (num_frames // 2)
        self.seq_length = num_patches
        self.num_masks = int(self.mask_ratio * self.seq_length)
        self.mask_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                self.num_frames,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )

        config = self.get_config()

        return config, pixel_values_videos

    def get_config(self):
        return VJEPA2Config(
            crop_size=self.image_size,
            frames_per_clip=self.num_frames,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            mlp_ratio=self.mlp_ratio,
            pred_hidden_size=self.pred_hidden_size,
            pred_num_attention_heads=self.pred_num_attention_heads,
            pred_num_hidden_layers=self.pred_num_hidden_layers,
            pred_num_mask_tokens=self.pred_num_mask_tokens,
        )

    def create_and_check_model(self, config, pixel_values_videos):
        model = VJEPA2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values_videos)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values_videos,
        ) = config_and_inputs
        inputs_dict = {"pixel_values_videos": pixel_values_videos}
        return config, inputs_dict


@require_torch
class VJEPA2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VJEPA2 does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    test_torch_exportable = True

    all_model_classes = (VJEPA2Model, VJEPA2ForVideoClassification) if is_torch_available() else ()

    fx_compatible = True

    pipeline_model_mapping = {}

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = VJEPA2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VJEPA2Config, has_text_modality=False, hidden_size=37)

    @is_flaky(max_attempts=3, description="`torch.nn.init.trunc_normal_` is flaky.")
    def test_initialization(self):
        super().test_initialization()

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="VJEPA2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
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

    @unittest.skip(reason="VJEPA2 does not support feedforward chunking yet")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = VJEPA2Model.from_pretrained(VJEPA_HF_MODEL)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


def prepare_random_video(image_size=256):
    videos = prepare_video_inputs(
        batch_size=1,
        num_frames=16,
        num_channels=3,
        min_resolution=image_size,
        max_resolution=image_size,
        equal_resolution=True,
        return_tensors="torch",
    )
    return videos


@require_torch
@require_vision
class VJEPA2ModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_video_processor(self):
        return AutoVideoProcessor.from_pretrained(VJEPA_HF_MODEL) if is_vision_available() else None

    @slow
    def test_inference_image(self):
        model = VJEPA2Model.from_pretrained(VJEPA_HF_MODEL).to(torch_device)

        video_processor = self.default_video_processor
        image = prepare_img()
        inputs = video_processor(torch.Tensor(np.array(image)), return_tensors="pt").to(torch_device)
        pixel_values_videos = inputs.pixel_values_videos
        pixel_values_videos = pixel_values_videos.repeat(1, model.config.frames_per_clip, 1, 1, 1)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values_videos)

        # verify the last hidden states
        expected_shape = torch.Size((1, 8192, 1024))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.0061, -1.8365, 2.7343], [-2.5938, -2.7181, -0.1663], [-1.7993, -2.2430, -1.1388]],
            device=torch_device,
        )
        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=8e-2, atol=8e-2)

    @slow
    def test_inference_video(self):
        model = VJEPA2Model.from_pretrained(VJEPA_HF_MODEL).to(torch_device)

        video_processor = self.default_video_processor
        video = prepare_random_video()
        inputs = video_processor(video, return_tensors="pt").to(torch_device)
        pixel_values_videos = inputs.pixel_values_videos

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values_videos)

        # verify the last hidden states
        expected_shape = torch.Size((1, 2048, 1024))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

    @slow
    def test_predictor_outputs(self):
        model = VJEPA2Model.from_pretrained(VJEPA_HF_MODEL).to(torch_device)

        video_processor = self.default_video_processor
        video = prepare_random_video()
        inputs = video_processor(video, return_tensors="pt").to(torch_device)
        pixel_values_videos = inputs.pixel_values_videos

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values_videos)

        # verify the last hidden states
        expected_shape = torch.Size((1, 2048, 1024))
        self.assertEqual(outputs.predictor_output.last_hidden_state.shape, expected_shape)

    @slow
    def test_predictor_full_mask(self):
        model = VJEPA2Model.from_pretrained(VJEPA_HF_MODEL).to(torch_device)

        video_processor = self.default_video_processor
        video = prepare_random_video()
        inputs = video_processor(video, return_tensors="pt").to(torch_device)
        pixel_values_videos = inputs.pixel_values_videos

        # forward pass
        with torch.no_grad():
            context_mask = [torch.arange(2048, device=pixel_values_videos.device).unsqueeze(0)]
            predictor_mask = context_mask
            outputs = model(pixel_values_videos, context_mask=context_mask, target_mask=predictor_mask)

        # verify the last hidden states
        expected_shape = torch.Size((1, 2048, 1024))
        self.assertEqual(outputs.predictor_output.last_hidden_state.shape, expected_shape)

    @slow
    def test_predictor_partial_mask(self):
        model = VJEPA2Model.from_pretrained(VJEPA_HF_MODEL).to(torch_device)

        video_processor = self.default_video_processor
        video = prepare_random_video()
        inputs = video_processor(video, return_tensors="pt").to(torch_device)
        pixel_values_videos = inputs.pixel_values_videos

        num_patches = 2048
        num_masks = 100
        # forward pass
        with torch.no_grad():
            pos_ids = torch.arange(num_patches, device=pixel_values_videos.device)
            context_mask = [pos_ids[0 : num_patches - num_masks].unsqueeze(0)]
            predictor_mask = [pos_ids[num_patches - num_masks :].unsqueeze(0)]
            outputs = model(pixel_values_videos, context_mask=context_mask, target_mask=predictor_mask)

        # verify the last hidden states
        expected_shape = torch.Size((1, num_masks, 1024))
        self.assertEqual(outputs.predictor_output.last_hidden_state.shape, expected_shape)

    @slow
    def test_video_classification(self):
        checkpoint = "facebook/vjepa2-vitl-fpc16-256-ssv2"

        model = VJEPA2ForVideoClassification.from_pretrained(checkpoint).to(torch_device)
        video_processor = AutoVideoProcessor.from_pretrained(checkpoint)

        sample_video = np.ones((16, 3, 256, 256))
        inputs = video_processor(sample_video, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(outputs.logits.shape, (1, 174))

        expected_logits = torch.tensor([0.8814, -0.1195, -0.6389], device=torch_device)
        resulted_logits = outputs.logits[0, 100:103]
        torch.testing.assert_close(resulted_logits, expected_logits, rtol=1e-2, atol=1e-2)
