# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch VideoMask2Former model. """

import inspect
import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from tests.test_modeling_common import floats_tensor
from transformers import Mask2FormerConfig, is_torch_available, is_torchvision_available, is_vision_available
from transformers.testing_utils import (
    require_torch,
    require_torch_multi_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import VideoMask2FormerForVideoSegmentation, VideoMask2FormerModel

    if is_vision_available():
        from transformers import VideoMask2FormerImageProcessor

if is_torchvision_available():
    import torchvision

if is_vision_available():
    pass


class VideoVideoMask2FormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        is_training=True,
        use_auxiliary_loss=False,
        num_queries=10,
        num_channels=3,
        min_size=480,
        max_size=640,
        num_labels=4,
        hidden_dim=64,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_auxiliary_loss = use_auxiliary_loss
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.mask_feature_size = hidden_dim

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.min_size, self.max_size]).to(
            torch_device
        )

        pixel_mask = torch.ones([self.batch_size, self.min_size, self.max_size], device=torch_device)

        mask_labels = [
            (
                torch.zeros([self.num_labels, self.batch_size, self.min_size, self.max_size], device=torch_device)
                > 0.5
            ).float()
        ]
        class_labels = [(torch.zeros((self.num_labels), device=torch_device)).long()]

        config = self.get_config()
        return config, pixel_values, pixel_mask, mask_labels, class_labels

    def get_config(self):
        config = Mask2FormerConfig(
            hidden_size=self.hidden_dim,
        )
        config.num_queries = self.num_queries
        config.num_labels = self.num_labels

        config.backbone_config.depths = [1, 1, 1, 1]
        config.backbone_config.num_channels = self.num_channels

        config.encoder_feedforward_dim = 64
        config.dim_feedforward = 128
        config.hidden_dim = self.hidden_dim
        config.mask_feature_size = self.hidden_dim
        config.feature_size = self.hidden_dim
        return config

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    def check_output_hidden_state(self, output, config):
        encoder_hidden_states = output.encoder_hidden_states
        pixel_decoder_hidden_states = output.pixel_decoder_hidden_states
        transformer_decoder_hidden_states = output.transformer_decoder_hidden_states

        self.parent.assertTrue(len(encoder_hidden_states), len(config.backbone_config.depths))
        self.parent.assertTrue(len(pixel_decoder_hidden_states), len(config.backbone_config.depths))
        self.parent.assertTrue(len(transformer_decoder_hidden_states), config.decoder_layers)

    def create_and_check_video_mask2former_model(self, config, pixel_values, pixel_mask, output_hidden_states=False):
        with torch.no_grad():
            model = VideoMask2FormerModel(config=config)
            model.to(torch_device)
            model.eval()

            output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            output = model(pixel_values, output_hidden_states=True)

        self.parent.assertEqual(
            output.transformer_decoder_last_hidden_state.shape,
            (1, self.num_queries, self.hidden_dim),
        )

        # let's ensure the other two hidden state exists
        self.parent.assertTrue(output.pixel_decoder_last_hidden_state is not None)
        self.parent.assertTrue(output.encoder_last_hidden_state is not None)

        if output_hidden_states:
            self.check_output_hidden_state(output, config)

    def create_and_check_video_mask2former_segmentation_head_model(
        self, config, pixel_values, pixel_mask, mask_labels, class_labels
    ):
        model = VideoMask2FormerForVideoSegmentation(config=config)
        model.to(torch_device)
        model.eval()

        def comm_check_on_output(result):
            # let's still check that all the required stuff is there
            self.parent.assertTrue(result.transformer_decoder_last_hidden_state is not None)
            self.parent.assertTrue(result.pixel_decoder_last_hidden_state is not None)
            self.parent.assertTrue(result.encoder_last_hidden_state is not None)
            # okay, now we need to check the logits shape
            # due to the encoder compression, masks have a //4 spatial size
            self.parent.assertEqual(
                result.masks_queries_logits.shape,
                (self.num_queries, self.batch_size, self.min_size // 4, self.max_size // 4),
            )
            # + 1 for null class
            self.parent.assertEqual(result.class_queries_logits.shape, (1, self.num_queries, self.num_labels + 1))

        with torch.no_grad():
            result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            result = model(pixel_values)

            comm_check_on_output(result)

            result = model(
                pixel_values=pixel_values, pixel_mask=pixel_mask, mask_labels=mask_labels, class_labels=class_labels
            )

        comm_check_on_output(result)

        self.parent.assertTrue(result.loss is not None)
        self.parent.assertEqual(result.loss.shape, torch.Size([1]))


@require_torch
class VideoMask2FormerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (VideoMask2FormerModel, VideoMask2FormerForVideoSegmentation) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": VideoMask2FormerModel} if is_torch_available() else {}

    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = VideoVideoMask2FormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Mask2FormerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_video_mask2former_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_video_mask2former_model(config, **inputs, output_hidden_states=False)

    def test_video_mask2former_segmentation_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_video_mask2former_segmentation_head_model(*config_and_inputs)

    @unittest.skip(reason="Video Mask2Former does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Video Mask2Former does not have a get_input_embeddings method")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Mask2Former is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="Video Mask2Former does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(
        reason="Video Mask2Former has some layers using `add_module` which doesn't work well with `nn.DataParallel`"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ["shivi/video-mask2former-swin-tiny-youtubevis-2021-instance"]:
            model = VideoMask2FormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_model_with_labels(self):
        size = (self.model_tester.min_size, self.model_tester.max_size)

        inputs = {
            "pixel_values": torch.randn((2, 3, *size), device=torch_device),
            "mask_labels": [torch.zeros((10, 2, *size), device=torch_device)],
            "class_labels": [torch.zeros(10, device=torch_device).long()],
        }
        config = self.model_tester.get_config()

        model = VideoMask2FormerForVideoSegmentation(config).to(torch_device)
        outputs = model(**inputs)
        self.assertTrue(outputs.loss is not None)

    def test_hidden_states_output(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_video_mask2former_model(config, **inputs, output_hidden_states=True)

    def test_attention_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(**inputs, output_attentions=True)
            self.assertTrue(outputs.attentions is not None)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        model_class = self.all_model_classes[1]
        config, pixel_values, pixel_mask, mask_labels, class_labels = self.model_tester.prepare_config_and_inputs()

        model = model_class(config)
        model.to(torch_device)
        model.train()

        loss = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels).loss
        loss.backward()

    def test_retain_grad_hidden_states_attentions(self):
        model_class = self.all_model_classes[1]
        config, pixel_values, pixel_mask, mask_labels, class_labels = self.model_tester.prepare_config_and_inputs()
        config.output_hidden_states = True
        config.output_attentions = True

        model = model_class(config).to(torch_device)
        model.train()

        outputs = model(pixel_values, mask_labels=mask_labels, class_labels=class_labels)

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_hidden_states.retain_grad()

        pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states[0]
        pixel_decoder_hidden_states.retain_grad()

        transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states[0]
        transformer_decoder_hidden_states.retain_grad()

        attentions = outputs.attentions[0]
        attentions.retain_grad()

        outputs.loss.backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(pixel_decoder_hidden_states.grad)
        self.assertIsNotNone(transformer_decoder_hidden_states.grad)
        self.assertIsNotNone(attentions.grad)


TOLERANCE = 1e-4


# We will verify our results on a video of cars
def prepare_video():
    filepath = hf_hub_download(repo_id="shivi/video-demo", filename="cars.mp4", repo_type="dataset")
    video = torchvision.io.read_video(filepath)[0]
    return video


@require_vision
@slow
class VideoMask2FormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def model_checkpoints(self):
        return "shivi/video-mask2former-swin-tiny-youtubevis-2021-instance"

    @cached_property
    def default_image_processor(self):
        return (
            VideoMask2FormerImageProcessor.from_pretrained(self.model_checkpoints) if is_vision_available() else None
        )

    def test_video_mask2former_inference_no_head(self):
        # load model and processor
        model = VideoMask2FormerModel.from_pretrained(self.model_checkpoints).to(torch_device)
        image_processor = self.default_image_processor

        video = prepare_video()
        video_input = image_processor(images=list(video[:5]), return_tensors="pt").pixel_values

        video_shape = video_input.shape

        # check size is divisible by 32
        self.assertTrue((video_shape[-1] % 32) == 0 and (video_shape[-2] % 32) == 0)

        # check video size
        self.assertEqual(video_shape, (5, 3, 480, 640))

        with torch.no_grad():
            outputs = model(video_input)

        # check if we are getting expected hidden states from backbone, pixel_decoder and transformer_decoder
        expected_slice_hidden_state = torch.tensor(
            [[-0.4360, 0.7583, 0.9673], [-0.1426, -0.0267, 0.5829], [-0.2385, 0.0773, -0.1495]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.encoder_last_hidden_state[0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[-0.3904, -0.2753, -0.2319], [-0.3824, -0.1098, -0.2226], [-0.3643, -0.0688, -0.1076]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.pixel_decoder_last_hidden_state[0, 0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

        expected_slice_hidden_state = torch.tensor(
            [[-1.7580, -0.5369, -0.9012], [-1.1428, -0.7896, -0.2773], [-1.3240, -0.3889, -0.1932]]
        ).to(torch_device)
        self.assertTrue(
            torch.allclose(
                outputs.transformer_decoder_last_hidden_state[0, :3, :3], expected_slice_hidden_state, atol=TOLERANCE
            )
        )

    def test_video_mask2former_segmentation_head_model(self):
        # load model and processor
        model = VideoMask2FormerForVideoSegmentation.from_pretrained(self.model_checkpoints).to(torch_device)
        image_processor = self.default_image_processor

        video = prepare_video()
        video_input = image_processor(images=list(video[:5]), return_tensors="pt").pixel_values

        video_shape = video_input.shape

        # check size is divisible by 32
        self.assertTrue((video_shape[-1] % 32) == 0 and (video_shape[-2] % 32) == 0)

        # check video size
        self.assertEqual(video_shape, (5, 3, 480, 640))

        with torch.no_grad():
            outputs = model(video_input)

        masks_queries_logits = outputs.masks_queries_logits

        self.assertEqual(
            masks_queries_logits.shape,
            (model.config.num_queries, video_shape[0], video_shape[-2] // 4, video_shape[-1] // 4),
        )
        expected_slice = [
            [-63.1859, -76.4797, -69.0339],
            [-62.5494, -68.8616, -81.0631],
            [-61.4500, -68.9583, -73.0472],
        ]
        expected_slice = torch.tensor(expected_slice).to(torch_device)

        self.assertTrue(torch.allclose(masks_queries_logits[0, 0, :3, :3], expected_slice, atol=TOLERANCE))

        # class_queries_logits
        class_queries_logits = outputs.class_queries_logits
        self.assertEqual(class_queries_logits.shape, (1, model.config.num_queries, model.config.num_labels + 1))

        expected_slice = torch.tensor(
            [
                [-3.0352, -3.6200, -5.5203],
                [-1.9340, -2.5809, -4.5011],
                [-2.3295, -2.9547, -4.4537],
            ]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_slice, atol=TOLERANCE))

    def test_with_segmentation_maps_and_loss(self):
        model = VideoMask2FormerForVideoSegmentation.from_pretrained(self.model_checkpoints).to(torch_device).eval()
        image_processor = self.default_image_processor

        video_inputs = image_processor(
            [np.zeros((3, 800, 1333)), np.zeros((3, 800, 1333)), np.zeros((3, 800, 1333)), np.zeros((3, 800, 1333))],
            segmentation_maps=[
                np.zeros((480, 640)).astype(np.float32),
                np.zeros((480, 640)).astype(np.float32),
                np.zeros((480, 640)).astype(np.float32),
                np.zeros((480, 640)).astype(np.float32),
            ],
            do_sampling=True,
            return_tensors="pt",
        )

        video_inputs["pixel_values"] = video_inputs["pixel_values"].to(torch_device)
        video_inputs["mask_labels"] = [el.to(torch_device) for el in video_inputs["mask_labels"]]
        video_inputs["class_labels"] = [el.to(torch_device) for el in video_inputs["class_labels"]]

        with torch.no_grad():
            outputs = model(**video_inputs)

        self.assertTrue(outputs.loss is not None)
