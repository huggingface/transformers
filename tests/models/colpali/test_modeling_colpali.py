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

from typing import Generator, cast

import pytest
import torch
from PIL import Image

from transformers.models.colpali import ColPaliForRetrieval, ColPaliProcessor
from transformers.models.colpali.processing_colpali import get_torch_device


@pytest.fixture(scope="module")
def colpali_model_path() -> str:
    return "vidore/colpali-v1.2"


@pytest.fixture(scope="module")
def colpali_from_pretrained(colpali_model_path: str) -> Generator[ColPaliForRetrieval, None, None]:
    device = get_torch_device("auto")
    print(f"Device used: {device}")

    yield cast(
        ColPaliForRetrieval,
        ColPaliForRetrieval.from_pretrained(
            colpali_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        ),
    )


@pytest.fixture(scope="module")
def processor() -> Generator[ColPaliProcessor, None, None]:
    yield cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("google/paligemma-3b-mix-448"))


@pytest.mark.slow
def test_load_colpali_from_pretrained(colpali_from_pretrained: ColPaliForRetrieval):
    assert isinstance(colpali_from_pretrained, ColPaliForRetrieval)


@pytest.mark.slow
def test_colpali_forward_images(
    colpali_from_pretrained: ColPaliForRetrieval,
    processor: ColPaliProcessor,
):
    # Create a batch of dummy images
    images = [
        Image.new("RGB", (32, 32), color="white"),
        Image.new("RGB", (16, 16), color="black"),
    ]

    # Process the image
    batch_images = processor.process_images(images).to(colpali_from_pretrained.device)

    # Forward pass
    with torch.no_grad():
        outputs = colpali_from_pretrained(**batch_images)

    # Assertions
    assert isinstance(outputs, torch.Tensor)
    assert outputs.dim() == 3
    batch_size, n_visual_tokens, emb_dim = outputs.shape
    assert batch_size == len(images)
    assert emb_dim == colpali_from_pretrained.dim


@pytest.mark.slow
def test_colpali_forward_queries(
    colpali_from_pretrained: ColPaliForRetrieval,
    processor: ColPaliProcessor,
):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_queries = processor.process_queries(queries).to(colpali_from_pretrained.device)

    # Forward pass
    with torch.no_grad():
        outputs = colpali_from_pretrained(**batch_queries)

    # Assertions
    assert isinstance(outputs, torch.Tensor)
    assert outputs.dim() == 3
    batch_size, n_query_tokens, emb_dim = outputs.shape
    assert batch_size == len(queries)
    assert emb_dim == colpali_from_pretrained.dim
