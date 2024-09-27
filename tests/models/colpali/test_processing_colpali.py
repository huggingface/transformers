from typing import Generator, cast

import pytest
import torch
from colpali_engine.models import ColPaliProcessor
from PIL import Image


@pytest.fixture(scope="module")
def colpali_processor_path() -> str:
    return "google/paligemma-3b-mix-448"


@pytest.fixture(scope="module")
def processor_from_pretrained(colpali_processor_path: str) -> Generator[ColPaliProcessor, None, None]:
    yield cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(colpali_processor_path))


def test_load_processor_from_pretrained(processor_from_pretrained: ColPaliProcessor):
    assert isinstance(processor_from_pretrained, ColPaliProcessor)


def test_process_images(processor_from_pretrained: ColPaliProcessor):
    # Create a dummy image
    image = Image.new("RGB", (16, 16), color="black")
    images = [image]

    # Process the image
    batch_feature = processor_from_pretrained.process_images(images)

    # Assertions
    assert "pixel_values" in batch_feature
    assert batch_feature["pixel_values"].shape == torch.Size([1, 3, 448, 448])


def test_process_queries(processor_from_pretrained: ColPaliProcessor):
    queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]

    # Process the queries
    batch_encoding = processor_from_pretrained.process_queries(queries)

    # Assertions
    assert "input_ids" in batch_encoding
    assert isinstance(batch_encoding["input_ids"], torch.Tensor)
    assert cast(torch.Tensor, batch_encoding["input_ids"]).shape[0] == len(queries)
