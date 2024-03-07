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

from transformers import HieraConfig
from transformers.testing_utils import (
    require_torch,
    slow,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import HieraModel
    from transformers.models.hiera.modeling_hiera import HIERA_PRETRAINED_MODEL_ARCHIVE_LIST, HieraBlock

import math


class HieraModelTester:
    all_model_classes = (HieraModel,) if is_torch_available() else ()

    def __init__(
        self,
        parent,
        input_size: Tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        embedding_dimension: int = 96,  # initial embedding input_dim
        number_of_heads: int = 1,  # initial number of number_of_heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
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

    def prepare_config_and_inputs(self, checkpoint_url):
        # Prepare configuration and inputs for testing your model
        pixel_values = torch.rand((1, self.in_chans, self.input_size[0], self.input_size[1]))

        config = self.get_config(checkpoint_url=checkpoint_url)

        return config, pixel_values

    def get_config(self, checkpoint_url):
        if "hiera_tiny_224" in checkpoint_url:
            config = HieraConfig(
                embedding_dimension=96,
                number_of_heads=1,
                stages=(1, 2, 7, 2),
            )

        elif "hiera_small_224" in checkpoint_url:
            config = HieraConfig(
                embedding_dimension=96,
                number_of_heads=1,
                stages=(1, 2, 11, 2),
            )

        elif "hiera_base_224" in checkpoint_url:
            config = HieraConfig(
                embedding_dimension=96,
                number_of_heads=1,
                stages=(2, 3, 16, 3),
            )

        elif "hiera_base_plus_224" in checkpoint_url:
            config = HieraConfig(
                embedding_dimension=112,
                number_of_heads=2,
                stages=(2, 3, 16, 3),
            )

        elif "hiera_large_224" in checkpoint_url:
            config = HieraConfig(
                embedding_dimension=144,
                number_of_heads=2,
                stages=(2, 6, 36, 4),
            )

        elif "hiera_huge_224" in checkpoint_url:
            config = HieraConfig(embedding_dimension=256, number_of_heads=4, stages=(2, 6, 36, 4))

        elif "hiera_base_16x224" in checkpoint_url:
            config = HieraConfig(
                num_classes=self.num_classes,
                input_size=(16, 224, 224),
                q_stride=(1, 2, 2),
                mask_unit_size=(1, 8, 8),
                patch_kernel=(3, 7, 7),
                patch_stride=(2, 4, 4),
                patch_padding=(1, 3, 3),
                sep_position_embeddings=True,
            )

        elif "hiera_base_plus_16x224" in checkpoint_url:
            config = HieraConfig(embedding_dimension=112, number_of_heads=2, stages=(2, 3, 16, 3))

        elif "hiera_large_16x224" in checkpoint_url:
            config = HieraConfig(
                embedding_dimension=144,
                number_of_heads=2,
                stages=(2, 6, 36, 4),
            )

        elif "hiera_huge_16x224" in checkpoint_url:
            config = HieraConfig(embedding_dimension=256, number_of_heads=4, stages=(2, 6, 36, 4))
        else:
            raise RuntimeError(f"Invalid checkpoint url ({checkpoint_url})")

        return config

    def create_and_check_model(self, config, pixel_values):
        batch_size = 1
        for model_class in self.all_model_classes:
            model = model_class(config=config)
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
                result = model(pixel_values=pixel_values)

            for idx, x in enumerate(result.intermediates):
                self.parent.assertEqual(x.shape, indermediate_shapes[idx], "Invalid Intermediate shape")


@require_torch
class HieraModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = HieraModelTester(self)

    def test_model(self):
        for model_name in HIERA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config_and_inputs = self.model_tester.prepare_config_and_inputs(model_name)
            self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in HIERA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = HieraModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
@slow
class HieraModelIntegrationTest(unittest.TestCase):
    def test_forward(self):
        torch_device = "cpu"
        input_size = 224
        batch_size = 1
        patch_kernel = (7, 7)
        patch_padding = (3, 3)
        patch_stride = (4, 4)
        q_stride = (2, 2)
        flat_q_stride = math.prod(q_stride)
        stages = (2, 3, 16, 3)
        embedding_dimension = 96
        model = HieraModel.from_pretrained("namangarg110/hiera_base_224")
        model.to(torch_device)

        random_tensor = torch.rand(batch_size, 3, input_size, input_size)
        num_patches = int(((input_size - patch_kernel[0] + 2 * patch_padding[0]) / patch_stride[0]) + 1) ** 2

        indermediate_shapes = []
        for _ in stages:
            indermediate_shapes.append(
                (batch_size, int(math.sqrt(num_patches)), int(math.sqrt(num_patches)), embedding_dimension)
            )
            num_patches = num_patches / flat_q_stride
            embedding_dimension = embedding_dimension * 2
        out = model(random_tensor)

        out.last_hidden_state.argmax(dim=-1).item()

        out = model(random_tensor, output_intermediates=True)
        for idx, x in enumerate(out.intermediates):
            self.assertEqual(x.shape, indermediate_shapes[idx], "Invalid Intermediate shape")
