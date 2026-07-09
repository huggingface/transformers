# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PIL Image processor class for Idefics3."""

import math
from typing import TYPE_CHECKING

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode, pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    make_nested_list_of_images,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


if TYPE_CHECKING:
    pass


def _make_pixel_mask(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Make pixel mask: 1=valid, 0=padding. Images are CHW."""
    h, w = image.shape[-2:]
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:h, :w] = 1
    return mask


# Adapted from transformers.models.idefics3.image_processing_idefics3.MAX_IMAGE_SIZE
MAX_IMAGE_SIZE = 4096  # 4k resolution as absolute maximum


# Adapted from transformers.models.idefics3.image_processing_idefics3.Idefics3ImageProcessorKwargs
class Idefics3ImageProcessorKwargs(ImagesKwargs, total=False):
    """
    do_image_splitting (`bool`, *optional*, defaults to `True`):
        Whether to split the image into sub-images concatenated with the original image. They are split into patches
        such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
    max_image_size (`Dict`, *optional*, defaults to `{"longest_edge": 364}`):
        Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key "longest_edge".
    return_row_col_info (`bool`, *optional*, defaults to `False`):
        Whether to return the row and column information of the images.
    """

    do_image_splitting: bool
    max_image_size: dict[str, int]
    return_row_col_info: bool


# Adapted from transformers.models.idefics3.image_processing_idefics3._resize_output_size_rescale_to_max_len
def _resize_output_size_rescale_to_max_len(
    height: int, width: int, min_len: int | None = 1, max_len: int | None = None
) -> tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    Args:
        height (`int`):
            Height of the input image.
        width (`int`):
            Width of the input image.
        min_len (`int`, *optional*, defaults to 1):
            Minimum size of the output image.
        max_len (`int`, *optional*, defaults to the maximum size of the image):
            Maximum size of the output image.
    Returns:
        The output size of the image after resizing.
    """
    max_len = max(height, width) if max_len is None else max_len
    aspect_ratio = width / height

    if width >= height:
        width = max_len
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    elif height > width:
        height = max_len
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1

    # Avoid resizing to a size smaller than min_len
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


# Adapted from transformers.models.idefics3.image_processing_idefics3._resize_output_size_scale_below_upper_bound
def _resize_output_size_scale_below_upper_bound(
    height: int, width: int, max_len: dict[str, int] | None = None
) -> tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    Args:
        height (`int`):
            Height of the input image.
        width (`int`):
            Width of the input image.
        max_len (`Dict[str, int]`, *optional*, defaults to the maximum size of the image):
            Defines the maximum dimensions of the image.
    Returns:
        The output size of the image after resizing.
    """
    max_len = max(height, width) if max_len is None else max_len

    aspect_ratio = width / height
    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)

    # Avoid resizing to a size smaller than 1
    height = max(height, 1)
    width = max(width, 1)
    return height, width


def get_max_height_width(images_list: list[list[np.ndarray]]) -> tuple[int, int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    image_sizes = []
    for images in images_list:
        for image in images:
            image_sizes.append(image.shape[-2:])

    max_height = max(size[0] for size in image_sizes)
    max_width = max(size[1] for size in image_sizes)
    return (max_height, max_width)


def get_num_channels(images_list: list[list[np.ndarray]]) -> int:
    """
    Get the number of channels across all images in a batch. Handle empty sublists like in [[], [image]].
    """
    for images in images_list:
        if images:
            return images[0].shape[0]

    raise ValueError("No images found in the batch.")


def get_resize_output_image_size(
    image: np.ndarray,
    resolution_max_side: int,
) -> tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    Args:
        image (`np.ndarray`):
            Image to resize.
        resolution_max_side (`int`):
            The longest edge of the image will be resized to this value. The shortest edge will be resized to keep the
            input aspect ratio.
    Returns:
        The output size of the image after resizing.
    """
    height, width = image.shape[-2:]

    # Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
    height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=resolution_max_side)
    # Find the output size when scaling the image to be below the MAX_IMAGE_SIZE
    height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)
    return height, width


@auto_docstring
class Idefics3ImageProcessorPil(PilBackend):
    resample = PILImageResampling.LANCZOS
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"longest_edge": 4 * 364}
    max_image_size = {"longest_edge": 364}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_image_splitting = True
    do_pad = True
    return_row_col_info = False
    valid_kwargs = Idefics3ImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_attention_mask"]

    def __init__(self, **kwargs: Unpack[Idefics3ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Idefics3ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        **kwargs,
    ) -> np.ndarray:
        if size.longest_edge:
            new_size = get_resize_output_image_size(image, resolution_max_side=size.longest_edge)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError("size must be a dictionary with key 'longest_edge' or 'height' and 'width'.")
        return super().resize(image, SizeDict(height=new_size[0], width=new_size[1]), resample=resample, **kwargs)

    def split_images(
        self,
        image: np.ndarray,
        max_image_size: dict[str, int],
        resample: "PILImageResampling | None" = None,
    ):
        """Split an image into patches (mirrors TorchvisionBackend.split_images). Images are always CHW."""
        num_channels, height, width = image.shape
        max_height = max_width = max_image_size["longest_edge"]

        if height > max_height or width > max_width:
            num_splits_h = (height - max_height) // max_height + 1
            num_splits_w = (width - max_width) // max_width + 1

            frames = []
            for r in range(num_splits_h):
                for c in range(num_splits_w):
                    start_y = r * max_height
                    start_x = c * max_width
                    end_y = start_y + max_height
                    end_x = start_x + max_width
                    crop = image[:, start_y:end_y, start_x:end_x]
                    frames.append(crop)

            global_image_height, global_image_width = max_height, max_width
            image = self.resize(
                image, SizeDict(height=global_image_height, width=global_image_width), resample=resample
            )
            frames.append(image)
        else:
            num_splits_h, num_splits_w = 0, 0
            frames = [image]

        return frames, num_splits_h, num_splits_w

    def resize_for_vision_encoder(
        self,
        image: np.ndarray,
        vision_encoder_max_size: int,
        resample: "PILImageResampling | None" = None,
    ):
        """Resize images to be multiples of vision_encoder_max_size. Images are always CHW."""
        height, width = image.shape[-2:]
        aspect_ratio = width / height
        if width >= height:
            width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
            height = int(width / aspect_ratio)
            height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
        elif height > width:
            height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
            width = int(height * aspect_ratio)
            width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
        new_size = SizeDict(height=height, width=width)
        return self.resize(image, size=new_size, resample=resample)

    def pad(
        self,
        image: np.ndarray,
        padded_size: tuple[int, int],
        fill: int = 0,
        return_pixel_mask: bool = True,
    ):
        """Pad image to padded_size. Mirrors TorchvisionBackend.pad. Images are always CHW."""
        original_size = image.shape[-2:]
        padding_bottom = padded_size[0] - original_size[0]
        padding_right = padded_size[1] - original_size[1]

        if padding_bottom < 0 or padding_right < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded size: {padded_size}, original size: {original_size}."
            )

        pixel_mask = _make_pixel_mask(image, output_size=padded_size) if return_pixel_mask else None

        if original_size != padded_size:
            padding = ((0, padding_bottom), (0, padding_right))
            image = pad(
                image,
                padding,
                mode=PaddingMode.CONSTANT,
                constant_values=fill,
                data_format="channels_first",
                input_data_format="channels_first",
            )

        return image, pixel_mask

    def _preprocess(
        self,
        images: list[list[np.ndarray]],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        do_image_splitting: bool | None,
        max_image_size: dict[str, int] | None,
        return_row_col_info: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """Process a batch of images. Mirrors TorchvisionBackend._preprocess with per-image loops instead of batching."""
        # Resize
        if do_resize:
            images = [
                [self.resize(image=img, size=size, resample=resample) for img in batch_images]
                for batch_images in images
            ]

        # Image splitting
        if do_image_splitting:
            images = [
                [
                    self.resize_for_vision_encoder(image, max_image_size["longest_edge"], resample=resample)
                    for image in batch_images
                ]
                for batch_images in images
            ]
            images_split_arrays = []
            images_rows = []
            images_cols = []
            for batch_images in images:
                split_image_arrays = []
                image_rows = []
                image_cols = []
                for image in batch_images:
                    split_image_array, rows, cols = self.split_images(
                        image, max_image_size=max_image_size, resample=resample
                    )
                    split_image_arrays.extend(split_image_array)
                    image_rows.append(rows)
                    image_cols.append(cols)
                images_split_arrays.append(split_image_arrays)
                images_rows.append(image_rows)
                images_cols.append(image_cols)
            images = images_split_arrays
            rows = images_rows
            cols = images_cols
        else:
            images = [
                [
                    self.resize(
                        image=image,
                        size=SizeDict(height=max_image_size["longest_edge"], width=max_image_size["longest_edge"]),
                        resample=resample,
                    )
                    for image in batch_images
                ]
                for batch_images in images
            ]
            rows = [[0] * len(batch_images) for batch_images in images]
            cols = [[0] * len(batch_images) for batch_images in images]

        # Rescale and normalize
        if do_rescale:
            images = [[self.rescale(img, rescale_factor) for img in batch_images] for batch_images in images]
        if do_normalize:
            images = [[self.normalize(img, image_mean, image_std) for img in batch_images] for batch_images in images]

        # Pad
        if do_pad:
            max_num_images = max(len(images_) for images_ in images)
            max_height, max_width = get_max_height_width(images)
            num_channels = get_num_channels(images)

            padded_images_list = [
                [np.zeros((num_channels, max_height, max_width), dtype=np.float32) for _ in range(max_num_images)]
                for _ in range(len(images))
            ]
            pixel_attention_masks = [
                [np.zeros((max_height, max_width), dtype=np.int64) for _ in range(max_num_images)]
                for _ in range(len(images))
            ]

            for i, batch_images in enumerate(images):
                for j, image in enumerate(batch_images):
                    padded_images_list[i][j], pixel_attention_masks[i][j] = self.pad(image, (max_height, max_width))
            images = padded_images_list

        if do_pad:
            data = {
                "pixel_values": np.array(images),
                "pixel_attention_mask": np.array(pixel_attention_masks),
            }
        elif return_tensors == "pt":
            data = {"pixel_values": np.asarray(images)}
        else:
            data = {"pixel_values": images}

        encoding = BatchFeature(data=data, tensor_type=return_tensors)
        if return_row_col_info:
            encoding["rows"] = rows
            encoding["cols"] = cols

        return encoding

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_valid_processor_keys", None)
        encoder_dict.pop("return_row_col_info", None)
        return encoder_dict

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict):
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of patches per image.
        """
        do_image_splitting = images_kwargs.get("do_image_splitting", self.do_image_splitting)
        max_image_size = images_kwargs.get("max_image_size", self.max_image_size)
        size = images_kwargs.get("size", self.size)

        num_patches = num_rows = num_cols = 0
        if do_image_splitting:
            height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=size["longest_edge"])
            height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)
            aspect_ratio = width / height

            if width >= height:
                resized_width = math.ceil(width / max_image_size["longest_edge"]) * max_image_size["longest_edge"]
                resized_height = int(width / aspect_ratio)
                resized_height = math.ceil(height / max_image_size["longest_edge"]) * max_image_size["longest_edge"]
            elif height > width:
                resized_height = math.ceil(height / max_image_size["longest_edge"]) * max_image_size["longest_edge"]
                resized_width = int(height * aspect_ratio)
                resized_width = math.ceil(width / max_image_size["longest_edge"]) * max_image_size["longest_edge"]

            max_height = max_width = max_image_size["longest_edge"]
            if resized_height > max_height or resized_width > max_width:
                num_rows = math.ceil(resized_height / max_height)
                num_cols = math.ceil(resized_width / max_width)
                num_patches = num_rows * num_cols + 1

        return num_patches, num_rows, num_cols


__all__ = ["Idefics3ImageProcessorPil"]
