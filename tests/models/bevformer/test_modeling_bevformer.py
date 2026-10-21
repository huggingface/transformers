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
"""Testing suite for the PyTorch BEVFormer model."""

import unittest

from transformers import BevformerConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import BevformerFor3DObjectDetection, BevformerModel


class BevformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        is_training=False,
        num_cameras=2,
        image_height=32,
        image_width=32,
        bev_height=4,
        bev_width=4,
        hidden_size=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_queries=4,
        num_classes=10,
        num_points_in_pillar=2,
        num_points_in_bev=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_cameras = num_cameras
        self.image_height = image_height
        self.image_width = image_width
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_points_in_pillar = num_points_in_pillar
        self.num_points_in_bev = num_points_in_bev

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_cameras, 3, self.image_height, self.image_width])
        # lidar2img identity matrices scaled
        lidar2img = torch.eye(4, device=torch_device).view(1, 1, 4, 4).repeat(self.batch_size, self.num_cameras, 1, 1)
        can_bus = torch.zeros(self.batch_size, 18, device=torch_device)

        config = self.get_config()
        return config, pixel_values, lidar2img, can_bus

    def get_config(self):
        from transformers import ResNetConfig

        backbone_config = ResNetConfig(
            num_channels=3,
            embeddings_size=16,
            hidden_sizes=[16, 32, 64, 128],
            depths=[1, 1, 1, 1],
            out_features=["stage4"],
        )
        return BevformerConfig(
            backbone_config=backbone_config,
            num_cameras=self.num_cameras,
            bev_height=self.bev_height,
            bev_width=self.bev_width,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            num_queries=self.num_queries,
            num_classes=self.num_classes,
            num_points_in_pillar=self.num_points_in_pillar,
            num_points_in_bev=self.num_points_in_bev,
            is_training=self.is_training,
        )

    def create_and_check_model(self, config, pixel_values, lidar2img, can_bus):
        model = BevformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values=pixel_values, lidar2img=lidar2img, can_bus=can_bus)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.bev_height * self.bev_width, self.hidden_size),
        )

    def create_and_check_for_3d_object_detection(self, config, pixel_values, lidar2img, can_bus):
        model = BevformerFor3DObjectDetection(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values=pixel_values, lidar2img=lidar2img, can_bus=can_bus)
        self.parent.assertEqual(
            result.logits.shape,
            (self.batch_size, self.num_decoder_layers, self.num_queries, self.num_classes),
        )
        self.parent.assertEqual(
            result.pred_boxes.shape,
            (self.batch_size, self.num_decoder_layers, self.num_queries, 10),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, lidar2img, can_bus = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "lidar2img": lidar2img,
            "can_bus": can_bus,
        }
        return config, inputs_dict


@require_torch
class BevformerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BevformerModel, BevformerFor3DObjectDetection) if is_torch_available() else ()
    is_encoder_decoder = False
    has_attentions = False
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_missing_keys = False

    @unittest.skip(reason="BEVFormer does not output intermediate hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="BEVFormer does not output intermediate hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="BEVFormer does not support standard input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="BEVFormer does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    def setUp(self):
        self.model_tester = BevformerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=BevformerConfig,
            has_text_modality=False,
            common_properties=["hidden_size", "num_encoder_layers", "num_decoder_layers"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_3d_object_detection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_3d_object_detection(*config_and_inputs)

    def test_forward_with_prev_bev(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = BevformerFor3DObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            out1 = model(**inputs)
            bev_embed = out1.bev_embed
            inputs["prev_bev"] = bev_embed
            out2 = model(**inputs)

        self.assertEqual(
            out2.bev_embed.shape,
            (
                self.model_tester.batch_size,
                self.model_tester.bev_height * self.model_tester.bev_width,
                self.model_tester.hidden_size,
            ),
        )
        self.assertFalse(torch.allclose(out1.logits, out2.logits))
