# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch MaskFormer model. """


import unittest

import pytest

from parameterized import parameterized
from tests.test_modeling_common import floats_tensor
from transformers import MaskFormerConfig, is_torch_available, is_vision_available
from transformers.file_utils import cached_property
from transformers.models.maskformer.configuration_maskformer import MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.models.maskformer.MaskFormer.mask_former.mask_former_model import MaskFormer
from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerForInstanceSegmentationOutput,
    MaskFormerOutput,
)
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.trainer_callback import TrainerState

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import MaskFormerForInstanceSegmentation, MaskFormerModel
    from transformers.models.maskformer import MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image

    from transformers import MaskFormerFeatureExtractor


class MaskFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        is_training=True,
        use_auxilary_loss=False,
        num_queries=100,
        num_channels=3,
        min_size=384,
        max_size=640,
        num_labels=150,
        mask_feature_size=256,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_auxilary_loss = use_auxilary_loss
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.num_labels = num_labels
        self.mask_feature_size = mask_feature_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.min_size, self.max_size])

        pixel_mask = torch.ones([self.batch_size, self.min_size, self.max_size], device=torch_device)

        mask_labels = (
            torch.rand([self.batch_size, self.num_labels, self.min_size, self.max_size], device=torch_device) > 0.5
        ).float()
        class_labels = (torch.rand((self.batch_size, self.num_labels), device=torch_device) > 0.5).long()

        config = self.get_config()
        return config, pixel_values, pixel_mask, mask_labels, class_labels

    def get_config(self):
        return MaskFormerConfig(
            num_queries=self.num_queries,
            num_channels=self.num_channels,
            num_labels=self.num_labels,
            mask_feature_size=self.mask_feature_size,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    def check_output_hidden_state(self, output: MaskFormerOutput, config: MaskFormerConfig):
        encoder_hidden_states = output.encoder_hidden_states
        pixel_decoder_hidden_states = output.pixel_decoder_hidden_states
        transformer_decoder_hidden_states = output.transformer_decoder_hidden_states

        self.parent.assertTrue(len(encoder_hidden_states), len(config.backbone.depths))
        self.parent.assertTrue(len(pixel_decoder_hidden_states), len(config.backbone.depths))
        self.parent.assertTrue(len(transformer_decoder_hidden_states), config.transformer_decoder.decoder_layers)

    def create_and_check_maskformer_model(
        self, config, pixel_values, pixel_mask, output_hidden_states=False, **kwargs
    ):
        model = MaskFormerModel(config=config)
        model.to(torch_device)
        model.eval()

        output: MaskFormerOutput = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        output: MaskFormerOutput = model(pixel_values, output_hidden_states=True)
        # the correct shape of output.transformer_decoder_hidden_states ensure the correcteness of the
        # encoder and pixel decoder
        self.parent.assertEqual(
            output.transformer_decoder_last_hidden_state.shape,
            (self.batch_size, self.num_queries, self.mask_feature_size),
        )
        # let's ensure the other two hidden state exists
        self.parent.assertTrue(output.pixel_decoder_last_hidden_state is not None)
        self.parent.assertTrue(output.encoder_last_hidden_state is not None)

        if output_hidden_states:
            self.check_output_hidden_state(output, config)

    def create_and_check_maskformer_instance_segmentation_head_model(
        self, config, pixel_values, pixel_mask, mask_labels, class_labels
    ):
        model = MaskFormerForInstanceSegmentation(config=config)
        model.to(torch_device)
        model.eval()

        def comm_check_on_output(result):
            # let's still check that all the required stuff is there
            self.parent.assertTrue(result.transformer_decoder_hidden_states is not None)
            self.parent.assertTrue(result.pixel_decoder_last_hidden_state is not None)
            self.parent.assertTrue(result.encoder_last_hidden_state is not None)
            # okay, now we need to check the logits shape
            # due to the encoder compression, masks have a //4 spatial size
            self.parent.assertEqual(
                result.masks_queries_logits.shape,
                (self.batch_size, self.num_queries, self.min_size // 4, self.max_size // 4),
            )
            # + 1 for null class
            self.parent.assertEqual(
                result.class_queries_logits.shape, (self.batch_size, self.num_queries, self.num_labels + 1)
            )

        result: MaskFormerForInstanceSegmentationOutput = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        comm_check_on_output(result)

        result = model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, mask_labels=mask_labels, class_labels=class_labels
        )

        comm_check_on_output(result)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertTrue(result.loss.shape is not None)


@require_torch
class MaskFormerModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            MaskFormerModel,
            MaskFormerForInstanceSegmentation,
        )
        if is_torch_available()
        else ()
    )
    is_encoder_decoder = False
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = MaskFormerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MaskFormerConfig, has_text_modality=False)

    def test_config(self):
        # TO ASK, maskformer doesn't have a hidden_size
        self.config_tester.run_common_tests()

    def test_maskformer_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_maskformer_model(config, **inputs, output_hidden_states=False)

    def test_maskformer_instance_segmentation_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_maskformer_instance_segmentation_head_model(*config_and_inputs)

    @unittest.skip(reason="MaskFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MaskFormer does not have a get_input_embeddings method")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="MaskFormer is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="MaskFormer does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = MaskFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_with_labels(self):
        self.parent.assertTrue(False)

    def test_outputs_hidden_states(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        self.model_tester.create_and_check_maskformer_model(config, **inputs, output_hidden_states=True)

    def test_attention_outputs(self):
        # TODO, what should I output from the model?
        self.assertFalse(True)


TOLERANCE = 1e-4


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@slow
class MaskFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return (
            MaskFormerFeatureExtractor.from_pretrained(MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP[0])
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_no_head(self):
        self.assertTrue(False)
