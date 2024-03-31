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

import unittest
from typing import Tuple

import requests

from transformers import HieraConfig
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_modeling_common import ModelTesterMixin


if is_torch_available():
    import torch
    from PIL import Image

    from transformers import BeitImageProcessor, HieraModel
    from transformers.models.hiera.modeling_hiera import HIERA_PRETRAINED_MODEL_ARCHIVE_LIST
import math


class HieraModelTester:
    all_model_classes = (HieraModel,) if is_torch_available() else ()

    def __init__(
        self,
        parent,
        input_size: Tuple[int, ...] = (32, 32),
        in_chans: int = 3,
        embedding_dimension: int = 32,  # initial embedding input_dim
        number_of_heads: int = 1,  # initial number of number_of_heads
        num_classes: int = 3,
        stages: Tuple[int, ...] = (1, 1, 1, 1),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_position_embeddings: bool = False,
        is_training=True,
    ):
        self.parent = parent
        self.input_size = input_size
        self.in_chans = in_chans
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.num_classes = num_classes
        self.stages = stages
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.mask_unit_size = mask_unit_size
        self.mask_unit_attn = mask_unit_attn
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.patch_kernel = patch_kernel
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.head_dropout = head_dropout
        self.head_init_scale = head_init_scale
        self.sep_position_embeddings = sep_position_embeddings
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        # Prepare configuration and inputs for testing your model
        pixel_values = torch.rand((1, self.in_chans, self.input_size[0], self.input_size[1]))

        config = self.get_config()

        return config, pixel_values.to(torch_device)

    def get_config(self):
        return HieraConfig(
            input_size=self.input_size,
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.number_of_heads,
            stages=self.stages,
            num_classes=self.num_classes,
        )

    def create_and_check_model(self, config, pixel_values):
        batch_size = 1
        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(torch_device)
            num_patches = (
                int(
                    ((self.input_size[0] - self.patch_kernel[0] + 2 * self.patch_padding[0]) / self.patch_stride[0])
                    + 1
                )
                ** 2
            )
            flat_q_stride = math.prod(self.q_stride)
            embedding_dimension = self.embedding_dimension
            indermediate_shapes = []
            for _ in self.stages:
                indermediate_shapes.append(
                    (batch_size, int(math.sqrt(num_patches)), int(math.sqrt(num_patches)), embedding_dimension)
                )
                num_patches = num_patches / flat_q_stride
                embedding_dimension = embedding_dimension * 2
            model.eval()
            with torch.no_grad():
                result = model(pixel_values=pixel_values.to(torch_device))

            for idx, x in enumerate(result.intermediates):
                self.parent.assertEqual(x.shape, indermediate_shapes[idx], "Invalid Intermediate shape")

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class HieraModelTest(unittest.TestCase, ModelTesterMixin):
    all_model_classes = (HieraModel,) if is_torch_available() else ()

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_attention_outputs = False
    test_model_outputs_equivalence = False
    test_config = False
    test_hidden_states_output = False
    test_initialization = False
    test_retain_grad_hidden_states_attentions = False

    def setUp(self):
        self.model_tester = HieraModelTester(self)
        # self.config_tester = ConfigTester(self, config_class=HieraConfig, has_text_modality=False, hidden_size=32)

    def test_config(self):
        pass
        # self.config_tester.run_common_tests()

    @unittest.skip(reason="Hiera does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model(self):
        for model_name in HIERA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config_and_inputs = self.model_tester.prepare_config_and_inputs()
            self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in HIERA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = HieraModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # for model_class in self.all_model_classes:
        #     model = model_class(config)
        #     self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
        #     x = model.get_output_embeddings()
        #     self.assertTrue(x is None or isinstance(x, nn.Linear))


def prepare_img():
    image = Image.open("/home/ubuntu/home/hiera/transformers/tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@slow
class HieraModelIntegrationTest(unittest.TestCase):
    def test_forward(self):
        model = HieraModel.from_pretrained("namangarg110/hiera_base_224")
        model.to(torch_device)

        url = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        image_processor = BeitImageProcessor.from_pretrained("namangarg110/hiera_image_processor", size=224)

        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)
        expected_slice = torch.tensor([0.1825, 0.8655, 0.5779, 1.1550, 1.1025, 0.6381, 1.0288, -0.0624, 0.1455])
        # If you also want intermediate feature maps
        out = model(pixel_values)
        out.last_hidden_state.argmax(dim=-1).item()
        self.assertTrue(torch.allclose(out.last_hidden_state[0, :9], expected_slice, atol=1e-4))


if __name__ == "__main__":
    test = HieraModelIntegrationTest()
    test.test_forward()
