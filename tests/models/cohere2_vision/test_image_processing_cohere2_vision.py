import numpy as np
import pytest
from PIL import Image

from transformers import BaseImageProcessor
from transformers.models.cohere2_vision.image_processing_cohere2_vision import Cohere2VisionImageProcessor


MAX_CROPS = 12


@pytest.fixture
def image_processor():
    return Cohere2VisionImageProcessor(
        max_patches=MAX_CROPS,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        img_size=512,
        patch_size=16,
        downsample_factor=2,
    )


def test_inheritance(image_processor):
    assert isinstance(image_processor, BaseImageProcessor)


def test_preprocess_raises_error(image_processor):
    img = Image.new("RGB", (100, 200), color=(255, 0, 0))
    with pytest.raises(NotImplementedError):
        image_processor.preprocess(img)

    with pytest.raises(NotImplementedError):
        image_processor(images=img)


@pytest.mark.parametrize(
    ["hw", "expected_resolution"],
    [
        [(512, 512), (2, 2)],
        [(1000, 200), (3, 1)],
        [(200, 1000), (1, 3)],
    ],
)
def test_select_tiling(hw: tuple[int, int], expected_resolution: tuple[int, int], image_processor):
    resolutions = image_processor.get_all_possible_resolutions(MAX_CROPS)
    h, w = hw
    base_image_size = 384
    selected_resolution = image_processor.select_tiling(h, w, base_image_size, resolutions)
    assert np.all(selected_resolution == np.array(expected_resolution))


@pytest.mark.parametrize(["hw", "crops", "crops_size"], [[(512, 512), 5, 384], [(1000, 1000), 10, 384]])
def test_scale_to_optimal_aspect_ratio(hw: tuple[int, int], crops: int, crops_size: int, image_processor):
    all_resolutions = image_processor.get_all_possible_resolutions(MAX_CROPS)
    h, w = hw
    image = Image.new("RGB", (h, w))
    tiles = image_processor.scale_to_optimal_aspect_ratio(image, all_resolutions, crops_size)
    assert len(tiles) == crops
    assert all(crop.size == (crops_size, crops_size) for crop in tiles)


def test_image_encoder(image_processor):
    image = Image.new("RGB", (1000, 1000))
    _, encoded_images, _ = image_processor.process_image(image)
    assert len(encoded_images) == 5


def test_normalization(image_processor):
    random_array = np.random.randint(-128, 128, (512, 512, 3), dtype=np.int8).astype(float)
    (normalized_image_processor,) = image_processor.normalize_patches([random_array])
    normalized_expected_code = random_array / (255 / 2) - 1

    assert np.array_equal(normalized_expected_code, normalized_image_processor)
