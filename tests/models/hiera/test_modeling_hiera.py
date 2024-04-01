# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Hiera model. """


import math
import unittest

from transformers import HieraConfig
from transformers.testing_utils import (
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import HieraForImageClassification, HieraForMaskedImageModeling, HieraModel
    from transformers.models.hiera.modeling_hiera import HIERA_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class HieraModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        input_size=(32, 32),
        mlp_ratio=1.0,
        num_channels=3,
        depths=[1, 1, 1, 1],
        initial_num_heads=1,
        num_head_multiplier=1.0,
        embed_dim_multiplier=1.0,
        is_training=True,
        use_labels=True,
        embed_dim=32,
        hidden_act="gelu",
        initializer_range=0.02,
        scope=None,
        type_sequence_label_size=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.input_size = input_size
        self.mlp_ratio = mlp_ratio
        self.num_channels = num_channels
        self.depths = depths
        self.initial_num_heads = initial_num_heads
        self.num_head_multiplier = num_head_multiplier
        self.embed_dim_multiplier = embed_dim_multiplier
        self.is_training = is_training
        self.use_labels = use_labels
        self.embed_dim = embed_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.scope = scope
        self.type_sequence_label_size = type_sequence_label_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.input_size[0], self.input_size[1]])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return HieraConfig(
            embed_dim=self.embed_dim,
            input_size=self.input_size,
            mlp_ratio=self.mlp_ratio,
            num_channels=self.num_channels,
            depths=self.depths,
            initial_num_heads=self.initial_num_heads,
            num_head_multiplier=self.num_head_multiplier,
            embed_dim_multiplier=self.embed_dim_multiplier,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = HieraModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_seq_len = math.prod([i // s for i, s in zip(config.input_size, config.patch_stride)]) * math.prod(
            config.query_stride
        ) ** (-len(config.depths))
        expected_dim = int(config.embed_dim * config.embed_dim_multiplier ** (len(config.depths) - 1))

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, expected_dim))

    def create_and_check_for_masked_image_modeling(self, config, pixel_values, labels):
        model = HieraForMaskedImageModeling(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.reconstruction.shape, (self.batch_size, self.num_channels, self.image_size, self.image_size)
        )

        # test greyscale images
        config.num_channels = 1
        model = HieraForMaskedImageModeling(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.input_size[0], self.input_size[0]])
        result = model(pixel_values)
        self.parent.assertEqual(result.reconstruction.shape, (self.batch_size, 1, self.image_size, self.image_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = HieraForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = HieraForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class HieraModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Hiera does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            HieraModel,
            HieraForImageClassification,
            HieraForMaskedImageModeling,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-feature-extraction": HieraModel, "image-classification": HieraForImageClassification}
        if is_torch_available()
        else {}
    )
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = HieraModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HieraConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Hiera does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_image_modeling(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_image_modeling(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in HIERA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = HieraModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class HieraModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            AutoImageProcessor.from_pretrained("EduardoPacheco/hiera-tiny-224-in1k") if is_vision_available() else None
        )

    @slow
    def test_inference_image_classification_head(self):
        model = HieraForImageClassification.from_pretrained("EduardoPacheco/hiera-tiny-224-in1k").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        expected_pixel_values = torch.tensor(
            [
                [[0.2967, 0.4679, 0.4508], [0.3309, 0.4337, 0.3309], [0.3309, 0.3823, 0.3309]],
                [[-1.5455, -1.4930, -1.5455], [-1.5280, -1.4755, -1.5980], [-1.5630, -1.5280, -1.4755]],
                [[-0.6367, -0.4973, -0.5321], [-0.7936, -0.6715, -0.6715], [-0.8284, -0.7413, -0.5670]],
            ]
        ).to(torch_device)

        self.assertTrue(torch.allclose(inputs.pixel_values[0, :3, :3, :3], expected_pixel_values, atol=1e-4))

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([[0.8028, 0.2409, -0.2254, -0.3712, -0.2848]]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :5], expected_slice, atol=1e-4))

    @slow
    def test_inference_interpolate_pos_encoding(self):
        # Hiera models have an `interpolate_pos_encoding` argument in their forward method,
        # allowing to interpolate the pre-trained position embeddings in order to use
        # the model on higher resolutions. The DINO model by Facebook AI leverages this
        # to visualize self-attention on higher resolution images.
        model = HieraModel.from_pretrained("facebook/dino-hieras8").to(torch_device)

        image_processor = AutoImageProcessor.from_pretrained("facebook/dino-hieras8", size=480)
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values, interpolate_pos_encoding=True)

        # verify the logits
        expected_shape = torch.Size((1, 3601, 384))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[4.2340, 4.3906, -6.6692], [4.5463, 1.8928, -6.7257], [4.4429, 0.8496, -5.8585]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

    @slow
    @require_accelerate
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        r"""
        A small test to make sure that inference work in half precision without any problem.
        """
        model = HieraModel.from_pretrained(
            "EduardoPacheco/hiera-tiny-224", torch_dtype=torch.float16, device_map="auto"
        )
        image_processor = self.default_image_processor

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        # forward pass to make sure inference works in fp16
        with torch.no_grad():
            _ = model(pixel_values)
