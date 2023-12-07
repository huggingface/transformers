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
""" Testing suite for the PyTorch SegGpt model. """


import inspect
import unittest

from transformers import SegGptConfig
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import SegGptModel
    from transformers.models.seggpt.modeling_seggpt import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import SegGptImageProcessor


class SegGptModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        mlp_ratio=2.0,
        merge_index=0,
        encoder_output_indicies=[1],
        pretrain_img_size=10,
        decoder_hidden_size=10,
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
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.mlp_ratio = mlp_ratio
        self.merge_index = merge_index
        self.encoder_output_indicies = encoder_output_indicies
        self.pretrain_img_size = pretrain_img_size
        self.decoder_hidden_size = decoder_hidden_size

        # in SegGpt, the seq length equals the number of patches (we don't use the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size // 2, self.image_size])
        prompt_pixel_values = floats_tensor(
            [self.batch_size, self.num_channels, self.image_size // 2, self.image_size]
        )
        prompt_masks = floats_tensor([self.batch_size, self.num_channels, self.image_size // 2, self.image_size])

        labels = None
        if self.use_labels:
            labels = floats_tensor([self.batch_size, self.num_channels, self.image_size // 2, self.image_size])

        config = self.get_config()

        return config, pixel_values, prompt_pixel_values, prompt_masks, labels

    def get_config(self):
        return SegGptConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            mlp_ratio=self.mlp_ratio,
            merge_index=self.merge_index,
            encoder_output_indicies=self.encoder_output_indicies,
            pretrain_img_size=self.pretrain_img_size,
            decoder_hidden_size=self.decoder_hidden_size,
        )

    def create_and_check_model(self, config, pixel_values, prompt_pixel_values, prompt_masks, labels):
        model = SegGptModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, prompt_pixel_values, prompt_masks)
        self.parent.assertEqual(
            result.pred_masks.shape, (self.batch_size, self.num_channels, self.image_size, self.image_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            prompt_pixel_values,
            prompt_masks,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "prompt_pixel_values": prompt_pixel_values,
            "prompt_masks": prompt_masks,
        }
        return config, inputs_dict


@require_torch
class SegGptModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SegGpt does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (SegGptModel,) if is_torch_available() else ()
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False
    pipeline_model_mapping = (
        {"feature-extraction": SegGptModel, "mask-generation": SegGptModel} if is_torch_available() else {}
    )

    def setUp(self):
        self.model_tester = SegGptModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SegGptConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SegGpt does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values", "prompt_pixel_values", "prompt_masks"]
            self.assertListEqual(arg_names[:3], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            patch_height = patch_width = config.image_size // config.patch_size

            self.assertListEqual(
                list(hidden_states[0].shape[-3:]),
                [patch_height, patch_width, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @slow
    def test_model_from_pretrained(self):
        for model_name in SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SegGptModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png").convert("RGB")
    return image


"./tests/fixtures/tests_samples/COCO/000000039769.png"


@require_torch
@require_vision
class SegGptModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            SegGptImageProcessor.from_pretrained("EduardoPacheco/seggpt-vit-large") if is_vision_available() else None
        )

    @slow
    def test_inference(self):
        # SegGpt models have an `interpolate_pos_encoding` argument in their forward method,
        # allowing to interpolate the pre-trained position embeddings in order to use
        # the model on higher resolutions. The DINO model by Facebook AI leverages this
        # to visualize self-attention on higher resolution images.
        model = SegGptModel.from_pretrained("EduardoPacheco/seggpt-vit-large").to(torch_device)

        image_processor = SegGptImageProcessor.from_pretrained("EduardoPacheco/seggpt-vit-large", size=480)
        image = prepare_img()
        inputs = image_processor(images=image, prompt_images=image, prompt_masks=image, return_tensors="pt")

        # Verify pixel values
        expected_pixel_values = torch.tensor(
            [
                [[0.2967, 0.3652, 0.3138], [0.2624, 0.2796, 0.4679]],
                [[-1.5980, -1.6155, -1.7031], [-1.6155, -1.6331, -1.5630]],
                [[-0.7761, -0.5844, -0.6367], [-0.8633, -0.9678, -0.5321]],
            ]
        )

        expected_prompt_pixel_values = torch.tensor(
            [
                [[0.2967, 0.3652, 0.3138], [0.2624, 0.2796, 0.4679]],
                [[-1.5980, -1.6155, -1.7031], [-1.6155, -1.6331, -1.5630]],
                [[-0.7761, -0.5844, -0.6367], [-0.8633, -0.9678, -0.5321]],
            ]
        )

        expected_prompt_masks = torch.tensor(
            [
                [[0.2796, 0.3823, 0.3138], [0.2453, 0.2624, 0.4851]],
                [[-1.5980, -1.6155, -1.7031], [-1.6506, -1.6856, -1.5280]],
                [[-0.8284, -0.5321, -0.6715], [-0.8110, -0.9678, -0.4798]],
            ]
        )

        self.assertTrue(torch.allclose(inputs.pixel_values[0, :, :2, :3], expected_pixel_values, atol=1e-4))
        self.assertTrue(
            torch.allclose(inputs.prompt_pixel_values[0, :, :2, :3], expected_prompt_pixel_values, atol=1e-4)
        )
        self.assertTrue(torch.allclose(inputs.prompt_masks[0, :, :2, :3], expected_prompt_masks, atol=1e-4))

        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 3, 896, 448))
        self.assertEqual(outputs.pred_masks.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [[0.2796, 0.3823, 0.3138], [0.2453, 0.2624, 0.4851]],
                [[-1.5980, -1.6155, -1.7031], [-1.6506, -1.6856, -1.5280]],
                [[-0.8284, -0.5321, -0.6715], [-0.8110, -0.9678, -0.4798]],
            ]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.pred_masks[0, :, :2, :3], expected_slice, atol=1e-4))

        expected_post_process = torch.tensor(
            [
                [[162.2683, 162.2683, 161.4149], [162.2683, 162.2683, 161.4149]],
                [[21.2490, 21.2490, 21.4580], [21.2490, 21.2490, 21.4580]],
                [[69.7492, 69.7492, 69.3204], [69.7492, 69.7492, 69.3204]],
            ]
        ).to(torch_device)

        result = image_processor.post_process_masks(outputs.pred_masks, (image.size[::-1]))[0]

        self.assertTrue(torch.allclose(result[0, :, :2, :3], expected_post_process, atol=1e-4))
