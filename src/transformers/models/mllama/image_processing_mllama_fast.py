# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
from typing import Dict, List, Optional, Tuple, Union

import PIL
from PIL import Image

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    BatchFeature,
    ChannelDimension,
    DefaultFastImageProcessorKwargs,
    ImageInput,
    SizeDict,
    TensorType,
    Unpack,
    group_images_by_shape,
    reorder_images,
    validate_fast_preprocess_arguments,
)
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...utils import (
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
)
from .image_processing_mllama import (
    get_all_supported_aspect_ratios,
    get_image_size_fit_to_canvas,
    get_optimal_tiled_canvas,
)


if is_vision_available():
    from ...image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


def _validate_size(size: SizeDict) -> None:
    if not (size.height and size.width):
        raise ValueError(f"Argument `size` must be a dictionary with keys 'height' and 'width'. Got: {size}")
    if size.height != size.width:
        raise ValueError(f"Argument `size` must have the same height and width, got {size}")


def _validate_mllama_preprocess_arguments(do_resize, size, do_pad, max_image_tiles):
    if not do_pad:
        raise ValueError("MllamaImageProcessor doesn't support `do_pad=False` mode.")
    if not do_resize:
        raise ValueError("MllamaImageProcessor doesn't support `do_resize=False` mode.")
    if max_image_tiles is None or max_image_tiles <= 0:
        raise ValueError(f"MllamaImageProcessor `max_image_tiles` must be a positive integer, got {max_image_tiles}.")
    _validate_size(size)


def split_to_tiles(image: "torch.Tensor", num_tiles_height: int, num_tiles_width: int) -> "torch.Tensor":
    """
    Split an image into a specified number of tiles along its width and height dimensions.

    Args:
        image (`torch.Tensor`):
            Input image with shape (num_channels, height, width).
        num_tiles_height (`int`):
            Number of tiles to split the image into along its height.
        num_tiles_width (`int`):
            Number of tiles to split the image into along its width.

    Returns:
        `torch.Tensor`:
            Array of image tiles with shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width).
    """
    batch_size, num_channels, height, width = image.shape
    tile_height = height // num_tiles_height
    tile_width = width // num_tiles_width

    image = image.reshape(batch_size, num_channels, num_tiles_height, tile_height, num_tiles_width, tile_width)

    # Permute to (batch_size, num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
    image = image.permute(0, 2, 4, 1, 3, 5)

    # Reshape into the desired output shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
    image = image.reshape(batch_size, num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)

    return image.contiguous()


def build_aspect_ratio_mask(aspect_ratios: List[Tuple[int, int]], max_image_tiles: int) -> "torch.Tensor":
    """
    Builds a mask for the aspect ratios of the images.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of lists containing aspect ratios for each image in the batch.
            Each aspect ratio is represented as a tuple of (width, height) in terms of number of tiles.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.

    Returns:
        `np.ndarray`: A 3D numpy array of shape (batch_size, max_num_images, max_image_tiles).
            The mask contains 1s for valid tiles and 0s for padding.
    """
    batch_size = len(aspect_ratios)
    max_num_images = 1

    aspect_ratio_mask = torch.zeros((batch_size, max_num_images, max_image_tiles), dtype=torch.long)

    # Set the first tile to 1 for all aspect ratios
    # because in original implementation aspect ratios are padded with (1, 1),
    # but original code examples are not built to handle batches, so we might remove it later
    aspect_ratio_mask[:, :, 0] = 1

    # Set the aspect ratio mask for the rest of the tiles
    for i, (num_tiles_w, num_tiles_h) in enumerate(aspect_ratios):
        aspect_ratio_mask[i, 0, : num_tiles_w * num_tiles_h] = 1

    return aspect_ratio_mask


def pack_images(
    batch_images: List[List["torch.Tensor"]],
    max_image_tiles: int,
) -> Tuple["torch.Tensor", List[List[int]]]:
    """
    Stack a list of lists of images with variable lengths into a numpy array, applying zero padding as needed.
    Each list in the input represents a batch sample, and each image within a list is expected to be
    pre-split into tiles. The resulting array will have a shape of
    (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).

    Args:
        batch_images (`List[List[np.ndarray]]`):
            A list of lists of image tiles. Each inner list represents
            a batch sample containing multiple images, where each image is pre-split into tiles.
            The shape of each tile array is (num_tiles, channels, tile_height, tile_width).
        max_image_tiles (int):
            The maximum number of tiles any image was potantially split.

    Returns:
        `Tuple[np.ndarray, List[List[int]]]`: A tuple containing:
            - stacked_images (`np.ndarray`):
                A numpy array of stacked images with shape
                (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).
            - all_num_tiles (`List[List[int]]`):
                A list of lists containing the number of tiles
                for each image in each batch sample.
    """

    # Determine output shape
    batch_size = len(batch_images)
    max_num_images = 1
    shapes = [image.shape for image in batch_images]
    _, channels, tile_height, tile_width = shapes[0]

    # Initialize the stacked images array with zeros
    stacked_images = torch.zeros(
        (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width),
        dtype=torch.float32,
    )

    # Fill the stacked images array with the tiled images from the batch
    all_num_tiles = []
    for i, image in enumerate(batch_images):
        num_sample_tiles = []
        num_tiles = image.shape[0]
        stacked_images[i, 0, :num_tiles] = image
        num_sample_tiles.append(num_tiles)
        all_num_tiles.append(num_sample_tiles)

    return stacked_images, all_num_tiles


def pack_aspect_ratios(aspect_ratios: List[Tuple[int, int]], pad_value: int = 1) -> "torch.Tensor":
    """
    Stack a list of aspect ratios into a numpy array.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios.
        pad_value (`int`, *optional*, defaults to 1):
            The value to pad the aspect ratios with.

    Returns:
        `np.ndarray`:
            The aspect ratios stacked into a numpy array with shape (batch_size, max_num_images, 2).
    """
    batch_size = len(aspect_ratios)
    max_num_images = 1

    aspect_ratios_stacked = torch.full((batch_size, max_num_images, 2), pad_value, dtype=torch.long)
    for i, row in enumerate(aspect_ratios):
        aspect_ratios_stacked[i, 0] = torch.tensor(row)

    return aspect_ratios_stacked


def convert_aspect_ratios_to_ids(aspect_ratios: List[List[Tuple[int, int]]], max_image_tiles: int) -> "torch.Tensor":
    """
    Convert aspect ratio tuples to unique ids.

    For batch padding we use 0, because there might be different number of images in each batch.
    The aspect ratio ids start from 1, with 1 corresponding to the first supported aspect ratio.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios for each image in the batch.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.

    Returns:
        `np.ndarray`:
            The aspect ratios ids as a numpy array with shape (batch_size, max_num_images).
            Each id corresponds to the index of the aspect ratio in the list of supported aspect ratios,
            offset by 1 (so 0 can be used for padding).
    """

    batch_size = len(aspect_ratios)
    max_num_images = 1
    supported_aspect_ratios = get_all_supported_aspect_ratios(max_image_tiles)

    aspect_ratios_ids = torch.zeros((batch_size, max_num_images), dtype=torch.long)
    for i, (num_tiles_h, num_tiles_w) in enumerate(aspect_ratios):
        aspect_ratios_ids[i, 0] = supported_aspect_ratios.index((num_tiles_h, num_tiles_w)) + 1
    return aspect_ratios_ids


# Copied from transformers.models.idefics2.image_processing_idefics2.convert_to_rgb
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """
    if not isinstance(image, PIL.Image.Image):
        return image

    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class MllamaFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_pad: Optional[bool]
    max_image_tiles: Optional[int]


@add_start_docstrings(
    "Constructs a fast Mllama image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether or not to pad the images to the largest height and width in the batch.
        max_image_tiles (`int`, *optional*, defaults to 4):
            The maximum number of tiles to split the image into.
    """,
)
class MllamaImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
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
    valid_kwargs = MllamaFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[MllamaFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            do_pad (`bool`, *optional*, defaults to `True`):
                Whether or not to pad the images to the largest height and width in the batch.
            max_image_tiles (`int`, *optional*, defaults to 4):
                The maximum number of tiles to split the image into.
        """,
    )
    def preprocess(self, images: ImageInput, **kwargs: Unpack[MllamaFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _validate_preprocess_kwargs(
        self,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, tuple[float]]] = None,
        image_std: Optional[Union[float, tuple[float]]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[SizeDict] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[SizeDict] = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        do_pad: Optional[bool] = None,
        max_image_tiles: Optional[int] = None,
        **kwargs,
    ):
        """
        validate the kwargs for the preprocess method.
        """
        validate_fast_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )

        _validate_mllama_preprocess_arguments(
            do_resize=do_resize, size=size, do_pad=do_pad, max_image_tiles=max_image_tiles
        )

    def pad(
        self,
        image: "torch.Tensor",
        size: Dict[str, int],
        aspect_ratio: Tuple[int, int],
    ) -> "torch.Tensor":
        """
        Pad an image to the `size` x `aspect_ratio`. For example, if size is {height: 224, width: 224} and aspect ratio is
        (1, 2), the image will be padded to 224x448.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            aspect_ratio (`Tuple[int, int]`):
                The aspect ratio of the image.

        Returns:
            `torch.Tensor`: The padded image.
        """

        image_height, image_width = image.shape[-2:]
        num_tiles_height, num_tiles_width = aspect_ratio
        padded_height = num_tiles_height * size.height
        padded_width = num_tiles_width * size.width
        pad_size = (0, 0, padded_width - image_width, padded_height - image_height)

        image = F.pad(
            image,
            pad_size,
            fill=0,
        )

        return image

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        max_image_tiles: int,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
    ) -> Union["torch.Tensor", Tuple[int, int]]:
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
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.

        Returns:
            `Union[np.ndarray, Tuple[int, int]]`: The resized image and a tuple containing the number of tiles
            along the height and width dimensions.
        """

        image_height, image_width = image.shape[-2:]
        tile_size = size.height

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

        image = F.resize(image, (new_height, new_width), interpolation=interpolation, antialias=antialias)

        return image, (num_tiles_height, num_tiles_width)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_pad: Optional[bool],
        max_image_tiles: Optional[int],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        aspect_ratio_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # do_resize=False is not supported, validated
            stacked_images, aspect_ratio = self.resize(
                image=stacked_images, size=size, interpolation=interpolation, max_image_tiles=max_image_tiles
            )
            # do_pad=False is not supported, validated
            stacked_images = self.pad(
                image=stacked_images,
                size=size,
                aspect_ratio=aspect_ratio,
            )
            num_tiles_height, num_tiles_width = aspect_ratio
            aspect_ratio_grouped[shape] = [(num_tiles_height, num_tiles_width)] * len(stacked_images)
            resized_images_grouped[shape] = stacked_images

        aspect_ratios = reorder_images(aspect_ratio_grouped, grouped_images_index)
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        aspect_ratio_dict = {}
        for aspect_ratio, image in zip(aspect_ratios, resized_images):
            aspect_ratio_dict[image.shape[-2:]] = aspect_ratio

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            num_tiles_height, num_tiles_width = aspect_ratio_dict[shape]
            stacked_images = split_to_tiles(stacked_images, num_tiles_height, num_tiles_width)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images, num_tiles = pack_images(processed_images, max_image_tiles)

        aspect_ratio_ids = convert_aspect_ratios_to_ids(aspect_ratios, max_image_tiles=max_image_tiles)
        aspect_ratio_mask = build_aspect_ratio_mask(aspect_ratios, max_image_tiles=max_image_tiles)

        # processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        encoded_inputs = BatchFeature(
            data={
                "pixel_values": processed_images,
                "aspect_ratio_ids": aspect_ratio_ids,
                "aspect_ratio_mask": aspect_ratio_mask,
            },
            tensor_type=return_tensors,
        )
        encoded_inputs["num_tiles"] = num_tiles

        return encoded_inputs


__all__ = ["MllamaImageProcessorFast"]
