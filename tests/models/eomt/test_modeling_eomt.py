# coding=utf-8
# Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EoMT model."""

import unittest

from tests.test_modeling_common import floats_tensor
from transformers import EoMTConfig, is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_torch,
    require_torch_multi_gpu,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin
from ...test_pipeline_mixin import PipelineTesterMixin


CHECKPOINT_IDS = [
    "tue-mps/coco_panoptic_eomt_large_640",
    "tue-mps/coco_panoptic_eomt_large_1280",
    "tue-mps/coco_panoptic_eomt_giant_640",
    "tue-mps/coco_panoptic_eomt_giant_1280",
    "tue-mps/ade20k_panoptic_eomt_large_640",
    "tue-mps/ade20k_panoptic_eomt_large_1280",
    "tue-mps/ade20k_panoptic_eomt_giant_640",
    "tue-mps/ade20k_panoptic_eomt_giant_1280",
    "tue-mps/cityscapes_semantic_eomt_large_1024",
    "tue-mps/ade20k_semantic_eomt_large_512",
    "tue-mps/coco_instance_eomt_large_640",
    "tue-mps/coco_instance_eomt_large_1280",
]


if is_torch_available():
    import torch

    from transformers import EoMTForUniversalSegmentation
    from transformers.models.eomt.modeling_eomt import EoMTNetwork

    if is_vision_available():
        pass

if is_vision_available():
    pass


class EoMTModelTester:
    is_training = False  # training code not yet implemented

    def __init__(
        self,
        parent,
        batch_size=2,
        num_queries=200,
        num_channels=3,
        image_size=640,
        num_labels=130,
        num_blocks=3,
        num_prefix_tokens=5,
        embed_dim=384,
        patch_size=16,
        backbone="vit_small_patch14_reg4_dinov2",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_blocks = num_blocks
        self.num_prefix_tokens = num_prefix_tokens
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.backbone = backbone

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size]).to(
            torch_device
        )

        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        config = EoMTConfig(
            backbone=self.backbone,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_blocks=self.num_blocks,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        config.use_pretrained_backbone = False  # pre-trained backbone causes batch equivalence test to fail
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def check_output_hidden_state(self, output, config):
        self.parent.assertTrue(len(output.hidden_states), config.num_hidden_layers)

    def create_and_check_eomt_model(self, config, pixel_values, output_hidden_states=False):
        with torch.no_grad():
            model = EoMTNetwork(config=config)
            model.to(torch_device)
            model.eval()

            output = model(pixel_values, output_hidden_states=output_hidden_states)

        expected_num_tokens = self.num_queries + self.num_prefix_tokens + (self.image_size // self.patch_size) ** 2
        self.parent.assertEqual(
            output.last_hidden_state.shape,
            (self.batch_size, expected_num_tokens, self.embed_dim),
        )

        if output_hidden_states:
            self.check_output_hidden_state(output, config)

    def create_and_check_eomt_universal_segmentation_head_model(self, config, pixel_values):
        model = EoMTForUniversalSegmentation(config=config)
        model.to(torch_device)
        model.eval()

        def comm_check_on_output(result):
            # we need to check the logits shape
            # due to the encoder compression, masks have a //4 spatial size
            self.parent.assertEqual(
                result.masks_queries_logits.shape,
                (self.batch_size, self.num_queries, self.image_size // 4, self.image_size // 4),
            )
            # + 1 for null class
            self.parent.assertEqual(
                result.class_queries_logits.shape, (self.batch_size, self.num_queries, self.num_labels + 1)
            )

        with torch.no_grad():
            result = model(pixel_values)

        comm_check_on_output(result)


@require_torch
class EoMTModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_model_mapping = {"image-segmentation": EoMTForUniversalSegmentation} if is_torch_available() else {}
    all_model_classes = (EoMTForUniversalSegmentation,) if is_torch_available() else ()

    is_encoder_decoder = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torch_exportable = True
    has_attentions = False

    @unittest.skip(reason="EoMT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EoMT does not have a get_input_embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="EoMT is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="EoMT does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(reason="EoMT has some layers using `add_module` which doesn't work well with `nn.DataParallel`")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(
        reason="Skipping weight initialization test because custom weight init is not specified in the config"
    )
    def test_initialization(self):
        pass

    @unittest.skip(reason="Need to use a timm model and there is no tiny model available.")
    def test_model_is_small(self):
        pass

    def setUp(self):
        self.model_tester = EoMTModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=EoMTConfig, has_text_modality=False, common_properties=["num_hidden_layers"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_eomt_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_eomt_model(config, **inputs)

    def test_eomt_universal_segmentation_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_eomt_universal_segmentation_head_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in CHECKPOINT_IDS:
            model, loading_info = EoMTForUniversalSegmentation.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            for info_type, info in loading_info.items():
                if info_type == "unexpected_keys":
                    ignore_keys = {
                        "criterion.empty_weight",
                        "network.attn_mask_probs",
                        "network.encoder.pixel_mean",
                        "network.encoder.pixel_std",
                    }
                    info = [k for k in info if k not in ignore_keys]
                self.assertEqual(len(info), 0, msg=f"{info_type} in {model_name}: {info}")

    def test_hidden_states_output(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_eomt_model(config, **inputs, output_hidden_states=True)
