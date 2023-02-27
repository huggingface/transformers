# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Seaformer model. """


import inspect
import unittest

from transformers import SeaformerConfig, is_torch_available, is_vision_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        MODEL_MAPPING,
        SeaformerForSemanticSegmentation,
        SeaformerModel,
    )
    from transformers.models.seaformer.modeling_seaformer import SEAFORMER_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import SeaformerImageProcessor


class SeaformerConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))
        for block_cfg in config.mv2_blocks_cfgs:
            for layer_cfg in block_cfg:
                self.parent.assertTrue(layer_cfg[-1] in [1,2])

class SeaformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        image_size=512,
        num_channels=3,
        num_encoder_blocks=3,
        depths=[3, 3, 3],
        num_labels = 150,
        channels = [32, 64, 128, 192, 256, 320],
        mv2_blocks_cfgs = [
                [   [3, 3, 32, 1],  
                    [3, 4, 64, 2], 
                    [3, 4, 64, 1]],  
                [
                    [5, 4, 128, 2],  
                    [5, 4, 128, 1]],  
                [
                    [3, 4, 192, 2],  
                    [3, 4, 192, 1]],
                [
                    [5, 4, 256, 2]],  
                [
                    [3, 6, 320, 2]]
            ],
        drop_path_rate = 0.1,
        emb_dims = [192, 256, 320],
        key_dims = [16, 20, 24],
        num_attention_heads=8,
        mlp_ratios=[2,4,6],
        attn_ratios = 2,
        in_channels = [128, 192, 256, 320],
        in_index = [0, 1, 2, 3],
        decoder_channels = 192,
        embed_dims = [128, 160, 192],
        is_depthwise = True,
        align_corners = False,
        semantic_loss_ignore_index=255,
        hidden_sizes = [128],
        hidden_act = 'relu',
        is_training=True,
        use_labels=True
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.channels = channels
        self.mv2_blocks_cfgs = mv2_blocks_cfgs
        self.drop_path_rate = drop_path_rate
        self.emb_dims = emb_dims
        self.key_dims = key_dims
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.mlp_ratios = mlp_ratios
        self.attn_ratios = attn_ratios
        self.in_channels = in_channels
        self.in_index = in_index
        self.decoder_channels = decoder_channels
        self.embed_dims = embed_dims
        self.is_depthwise = is_depthwise
        self.align_corners = align_corners
        self.semantic_loss_ignore_index=semantic_loss_ignore_index

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return SeaformerConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            num_encoder_blocks=self.num_encoder_blocks,
            depths=self.depths,
            hidden_sizes=self.hidden_sizes,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = SeaformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        expected_height = expected_width = self.image_size // (self.downsampling_rates[-1] * 2)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.hidden_sizes[-1], expected_height, expected_width)
        )

    def create_and_check_for_image_segmentation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = SeaformerForSemanticSegmentation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size // 8, self.image_size // 8)
        )
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size // 8, self.image_size // 8)
        )
        self.parent.assertGreater(result.loss, 0.0)

    def create_and_check_for_binary_image_segmentation(self, config, pixel_values, labels):
        config.num_labels = 1
        model = SeaformerForSemanticSegmentation(config=config)
        model.to(torch_device)
        model.eval()
        labels = torch.randint(0, 1, (self.batch_size, self.image_size, self.image_size)).to(torch_device)
        result = model(pixel_values, labels=labels)
        self.parent.assertGreater(result.loss, 0.0)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SeaformerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            SeaformerModel,
            SeaformerForSemanticSegmentation,
        )
        if is_torch_available()
        else ()
    )

    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = SeaformerModelTester(self)
        self.config_tester = SeaformerConfigTester(self, config_class=SeaformerConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_binary_image_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_binary_image_segmentation(*config_and_inputs)

    def test_for_image_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_segmentation(*config_and_inputs)

    @unittest.skip("Seaformer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("Seaformer does not have get_input_embeddings method and get_output_embeddings methods")
    def test_model_common_attributes(self):
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

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            print('attentions', attentions)

            expected_num_attentions = sum(self.model_tester.depths)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertEqual(len(attentions), expected_num_attentions)

            # verify the first attentions (first block, first layer)
            expected_seq_len = (self.model_tester.image_size // 4) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (4 * self.model_tester.sr_ratios[0])) ** 2
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads[0], expected_seq_len, expected_reduced_seq_len],
            )

            # verify the last attentions (last block, last layer)
            expected_seq_len = (self.model_tester.image_size // 32) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (32 * self.model_tester.sr_ratios[-1])) ** 2
            self.assertListEqual(
                list(attentions[-1].shape[-3:]),
                [self.model_tester.num_attention_heads[-1], expected_seq_len, expected_reduced_seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)
            # verify the first attentions (first block, first layer)
            expected_seq_len = (self.model_tester.image_size // 4) ** 2
            expected_reduced_seq_len = (self.model_tester.image_size // (4 * self.model_tester.sr_ratios[0])) ** 2
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads[0], expected_seq_len, expected_reduced_seq_len],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            # expected_num_layers = self.model_tester.num_encoder_blocks
            in_index = self.model_tester.in_index
            self.assertEqual(len(hidden_states), len(in_index))

            # verify the first hidden states (first block)
            self.assertListEqual(
                list(hidden_states[0].shape[-3:]),
                [
                    self.model_tester.hidden_sizes[0],
                    self.model_tester.image_size // 8,
                    self.model_tester.image_size // 8,
                ],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            if model_class in get_values(MODEL_MAPPING):
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @slow
    def test_model_from_pretrained(self):
        for model_name in SEAFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SeaformerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
class SeaformerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_segmentation_ade(self):
        # only resize + normalize
        feature_extractor = SeaformerImageProcessor(
            image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
        )
        model = SeaformerForSemanticSegmentation.from_pretrained("Inderpreet01/seaformer-semantic-segmentation-large").to(
            torch_device
        )

        image = prepare_img()
        encoded_inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoded_inputs.pixel_values.to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values)

        expected_shape = torch.Size((1, model.config.num_labels, 64, 64))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [   
                [[ -2.0818,  -4.6320,  -5.9963], [ -3.2360,  -7.2340,  -8.7455], [ -2.9308,  -8.1080, -9.9713]],
                [[ -5.4941,  -7.2591,  -8.4649], [ -6.2536,  -8.9669, -10.4255], [ -6.1386,  -9.4373, -11.4133]],
                [[ -9.2548, -11.4705, -13.2432], [-10.3784, -13.9842, -16.0520], [-10.4125, -14.8483, -17.2390]],
            ]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_post_processing_semantic_segmentation(self):
        # only resize + normalize
        feature_extractor = SeaformerImageProcessor(
            image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
        )
        model = SeaformerForSemanticSegmentation.from_pretrained("Inderpreet01/seaformer-semantic-segmentation-large").to(
            torch_device
        )

        image = prepare_img()
        encoded_inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoded_inputs.pixel_values.to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values)

        outputs.logits = outputs.logits.detach().cpu()

        segmentation = feature_extractor.post_process_semantic_segmentation(outputs=outputs, target_sizes=[(500, 300)])
        expected_shape = torch.Size((500, 300))
        self.assertEqual(segmentation[0].shape, expected_shape)

        segmentation = feature_extractor.post_process_semantic_segmentation(outputs=outputs)
        expected_shape = torch.Size((64, 64))
        self.assertEqual(segmentation[0].shape, expected_shape)
