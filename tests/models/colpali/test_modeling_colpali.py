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
"""Testing suite for the PyTorch ColPali model."""

import unittest
from typing import cast

import torch
from PIL import Image

from tests.test_modeling_common import ModelTesterMixin
from transformers import (
    is_torch_available,
    is_vision_available,
)
from transformers.models.colpali import ColPaliForRetrieval, ColPaliProcessor
from transformers.models.colpali.modeling_colpali import ColPaliModelOutput
from transformers.testing_utils import require_torch, require_vision, slow


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


@require_torch
class ColPaliForRetrievalTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `ColPaliForRetrieval`.
    """

    all_model_classes = (ColPaliForRetrieval,) if is_torch_available() else ()
    fx_compatible = False
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False

    @classmethod
    def setUpClass(cls):
        cls.model_name = "vidore/colpali-v1.2-hf"

        # TODO: replace with randomly initialized model
        cls.model = cast(
            ColPaliForRetrieval,
            ColPaliForRetrieval.from_pretrained(
                cls.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ),
        )

        cls.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(cls.model_name))
        cls.device = cls.model.device

    @slow
    @require_vision
    def test_colpali_forward_images(self):
        # Create a batch of dummy images
        images = [
            Image.new("RGB", (32, 32), color="white"),
            Image.new("RGB", (16, 16), color="black"),
        ]

        # Process the image
        batch_images = ColPaliForRetrievalTest.processor.process_images(images).to(ColPaliForRetrievalTest.device)

        # Forward pass
        with torch.no_grad():
            outputs = ColPaliForRetrievalTest.model(**batch_images, return_dict=True)

        # Assertions
        self.assertIsInstance(outputs, ColPaliModelOutput)
        self.assertIsInstance(outputs.embeddings, torch.Tensor)
        self.assertEqual(outputs.embeddings.dim(), 3)

        batch_size, n_query_tokens, embedding_dim = outputs.embeddings.shape
        self.assertEqual(batch_size, len(images))
        self.assertEqual(embedding_dim, ColPaliForRetrievalTest.model.embedding_dim)

    @slow
    def test_colpali_forward_queries(self):
        queries = [
            "Is attention really all you need?",
            "Are Benjamin, Antoine, Merve, and Jo best friends?",
        ]

        # Process the queries
        batch_queries = ColPaliForRetrievalTest.processor.process_queries(queries).to(ColPaliForRetrievalTest.device)

        # Forward pass
        with torch.no_grad():
            outputs = ColPaliForRetrievalTest.model(**batch_queries, return_dict=True)

        # Assertions
        self.assertIsInstance(outputs, ColPaliModelOutput)
        self.assertIsInstance(outputs.embeddings, torch.Tensor)
        self.assertEqual(outputs.embeddings.dim(), 3)

        batch_size, n_query_tokens, embedding_dim = outputs.embeddings.shape
        self.assertEqual(batch_size, len(queries))
        self.assertEqual(embedding_dim, ColPaliForRetrievalTest.model.embedding_dim)
