# Copyright 2024 The HuggingFace Inc. team.
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
"""PIL Image processor class for Mllama."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode, get_image_size
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    make_nested_list_of_images,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from .image_processing_mllama import (
    MllamaImageProcessorKwargs,
    _validate_mllama_preprocess_arguments,
    convert_to_rgb,
    get_all_supported_aspect_ratios,
    get_image_size_fit_to_canvas,
    get_optimal_tiled_canvas,
)


def split_to_tiles_np(image: np.ndarray, num_tiles_height: int, num_tiles_width: int) -> np.ndarray:
    """Split an image into tiles (numpy version)."""
    num_channels, height, width = image.shape
    tile_height = height // num_tiles_height
    tile_width = width // num_tiles_width

    image = image.reshape(num_channels, num_tiles_height, tile_height, num_tiles_width, tile_width)
    image = image.transpose(1, 3, 0, 2, 4)
    image = image.reshape(num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)

    return np.ascontiguousarray(image)


def build_aspect_ratio_mask_np(aspect_ratios: list[list[tuple[int, int]]], max_image_tiles: int) -> np.ndarray:
    """
    Build aspect ratio mask (numpy version).

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of lists containing aspect ratios for each image in the batch.
            Each aspect ratio is represented as a tuple of (width, height) in terms of number of tiles.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.

    Returns:
        `np.ndarray`: A 3D array of shape (batch_size, max_num_images, max_image_tiles).
            The mask contains 1s for valid tiles and 0s for padding.
    """
    batch_size = len(aspect_ratios)
    max_num_images = max(len(row) for row in aspect_ratios)

    aspect_ratio_mask = np.zeros((batch_size, max_num_images, max_image_tiles), dtype=np.int64)
    # Set the first tile to 1 for all aspect ratios
    # because in original implementation aspect ratios are padded with (1, 1),
    # but original code examples are not built to handle batches, so we might remove it later
    aspect_ratio_mask[:, :, 0] = 1

    # Set the aspect ratio mask for the rest of the tiles
    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_w, num_tiles_h) in enumerate(sample_aspect_ratios):
            aspect_ratio_mask[i, j, : num_tiles_w * num_tiles_h] = 1

    return aspect_ratio_mask


def pack_images(
    batch_images: list[list[np.ndarray]],
    max_image_tiles: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Stack nested image tiles into a padded array.
    Each list in the input represents a batch sample, and each image within a list is expected to be
    pre-split into tiles. The resulting array will have a shape of
    (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).

    Args:
        batch_images (`List[List[np.ndarray]]`):
            A list of lists of image tiles. Each inner list represents
            a batch sample containing multiple images, where each image is pre-split into tiles.
            The shape of each tile array is (num_tiles, channels, tile_height, tile_width).
        max_image_tiles (int):
            The maximum number of tiles any image was potentially split.

    Returns:
        `Tuple[np.ndarray, List[List[int]]]`: A tuple containing:
            - stacked_images (`np.ndarray`):
                An array of stacked images with shape
                (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).
            - all_num_tiles (`List[List[int]]`):
                A list of lists containing the number of tiles
                for each image in each batch sample.
    """
    # Determine output shape
    batch_size = len(batch_images)
    max_num_images = max(len(images) for images in batch_images)
    shapes = [image.shape for images in batch_images for image in images]
    _, channels, tile_height, tile_width = shapes[0]

    # Initialize the stacked images array with zeros
    stacked_images = np.zeros(
        (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width),
        dtype=np.float32,
    )

    # Fill the stacked images array with the tiled images from the batch
    all_num_tiles = []
    for i, images in enumerate(batch_images):
        num_sample_tiles = []
        for j, image in enumerate(images):
            num_tiles = image.shape[0]
            stacked_images[i, j, :num_tiles] = image
            num_sample_tiles.append(num_tiles)
        all_num_tiles.append(num_sample_tiles)

    return stacked_images, all_num_tiles


def convert_aspect_ratios_to_ids_np(aspect_ratios: list[list[tuple[int, int]]], max_image_tiles: int) -> np.ndarray:
    """
    Convert aspect ratio tuples to ids (numpy version).

    For batch padding we use 0, because there might be different number of images in each batch.
    The aspect ratio ids start from 1, with 1 corresponding to the first supported aspect ratio.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios for each image in the batch.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.

    Returns:
        `np.ndarray`:
            The aspect ratios ids as an array with shape (batch_size, max_num_images).
            Each id corresponds to the index of the aspect ratio in the list of supported aspect ratios,
            offset by 1 (so 0 can be used for padding).
    """
    batch_size = len(aspect_ratios)
    max_num_images = max(len(row) for row in aspect_ratios)
    supported_aspect_ratios = get_all_supported_aspect_ratios(max_image_tiles)

    aspect_ratios_ids = np.zeros((batch_size, max_num_images), dtype=np.int64)
    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_h, num_tiles_w) in enumerate(sample_aspect_ratios):
            aspect_ratios_ids[i, j] = supported_aspect_ratios.index((num_tiles_h, num_tiles_w)) + 1
    return aspect_ratios_ids


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class MllamaImageProcessorPil(PilBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    max_image_tiles = 4
    valid_kwargs = MllamaImageProcessorKwargs
    model_input_names = ["pixel_values", "num_tiles", "aspect_ratio_ids", "aspect_ratio_mask"]

    def __init__(self, **kwargs: Unpack[MllamaImageProcessorKwargs]):
        super().__init__(**kwargs)
        _validate_mllama_preprocess_arguments(self.do_resize, self.size, self.do_pad, self.max_image_tiles)

    def _validate_preprocess_kwargs(self, **kwargs):
        super()._validate_preprocess_kwargs(**kwargs)
        _validate_mllama_preprocess_arguments(self.do_resize, self.size, self.do_pad, self.max_image_tiles)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[MllamaImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        """Prepare a nested images structure for processing."""
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        """Converts an image to RGB format."""
        return convert_to_rgb(image)

    def pad(
        self,
        image: np.ndarray,
        size: dict[str, int],
        aspect_ratio: tuple[int, int],
    ) -> np.ndarray:
        """
        Pad an image to the `size` x `aspect_ratio`. For example, if size is {height: 224, width: 224} and aspect ratio is
        (1, 2), the image will be padded to 224x448.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`Dict[str, int]`):
                Size of the output image.
            aspect_ratio (`Tuple[int, int]`):
                The aspect ratio of the image.

        Returns:
            `np.ndarray`: The padded image.
        """
        image_height, image_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        num_tiles_height, num_tiles_width = aspect_ratio
        padded_height = num_tiles_height * size["height"]
        padded_width = num_tiles_width * size["width"]
        # Spatial padding: ((height_before, height_after), (width_before, width_after))
        # np_pad will add channel dimension padding for channels_first
        padding = ((0, padded_height - image_height), (0, padded_width - image_width))

        image = np_pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=0,
            data_format="channels_first",
            input_data_format="channels_first",
        )
        return image

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        max_image_tiles: int,
        resample: PILImageResampling | None = None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Resizes an image to fit within a tiled canvas while maintaining its aspect ratio.
        The optimal canvas size is calculated based on the maximum number of tiles and the tile size.

        The function first determines the best tile arrangement for the image, then resizes the image
        to fit within this canvas. The resized image and the number of tiles along the height and width
        dimensions are returned.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            max_image_tiles (`int`):
                The maximum number of tiles to split the image into.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.

        Returns:
            `Union[np.ndarray, Tuple[int, int]]`: The resized image and a tuple containing the number of tiles
            along the height and width dimensions.
        """
        image_height, image_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        tile_size = size.height if hasattr(size, "height") else size["height"]

        canvas_height, canvas_width = get_optimal_tiled_canvas(
            image_height=image_height,
            image_width=image_width,
            max_image_tiles=max_image_tiles,
            tile_size=tile_size,
        )
        num_tiles_height = canvas_height // tile_size
        num_tiles_width = canvas_width // tile_size

        new_height, new_width = get_image_size_fit_to_canvas(
            image_height=image_height,
            image_width=image_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            tile_size=tile_size,
        )

        image = super().resize(image, SizeDict(height=new_height, width=new_width), resample=resample)
        return image, (num_tiles_height, num_tiles_width)

    def _preprocess(
        self,
        images: list[list[np.ndarray]],
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_image_tiles: int | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        batch_images = []
        batch_aspect_ratios = []

        size_dict = size if isinstance(size, dict) else {"height": size.height, "width": size.width}

        for sample_images in images:
            sample_tiles = []
            sample_aspect_ratios = []

            for image in sample_images:
                image, aspect_ratio = self.resize(image, size=size, max_image_tiles=max_image_tiles, resample=resample)
                image = self.pad(image, size=size_dict, aspect_ratio=aspect_ratio)

                # Rescale and normalize
                if do_rescale:
                    image = self.rescale(image, rescale_factor)
                if do_normalize:
                    image = self.normalize(image, image_mean, image_std)

                num_tiles_height, num_tiles_width = aspect_ratio
                tiles = split_to_tiles_np(image, num_tiles_height, num_tiles_width)
                sample_tiles.append(tiles)
                sample_aspect_ratios.append(aspect_ratio)

            batch_images.append(sample_tiles)
            batch_aspect_ratios.append(sample_aspect_ratios)

        stacked_images, num_tiles = pack_images(batch_images, max_image_tiles)
        aspect_ratio_ids = convert_aspect_ratios_to_ids_np(batch_aspect_ratios, max_image_tiles)
        aspect_ratio_mask = build_aspect_ratio_mask_np(batch_aspect_ratios, max_image_tiles)

        encoded_inputs = BatchFeature(
            data={
                "pixel_values": stacked_images,
                "aspect_ratio_ids": aspect_ratio_ids,
                "aspect_ratio_mask": aspect_ratio_mask,
            },
            tensor_type=return_tensors,
        )
        encoded_inputs["num_tiles"] = num_tiles

        return encoded_inputs


__all__ = ["MllamaImageProcessorPil"]
