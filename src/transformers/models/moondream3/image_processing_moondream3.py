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
"""Image processor class for Moondream3."""

import math
from typing import Optional, Union

import torch
import numpy as np

from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_utils import (
    ImageInput,
    make_flat_list_of_images,
    valid_images,
    validate_kwargs,
)
from transformers.processing_utils import ImagesKwargs
from transformers.utils import TensorType, logging
from transformers.utils.import_utils import requires_backends


logger = logging.get_logger(__name__)


import PIL


class Moondream3ImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`Union[dict[str, int], int]` *optional*, defaults to `{"height": 16, "width": 16}`):
        Size of the patches in the model, used to calculate the output image size. Can be overridden by `patch_size` in the `preprocess` method.
    """

    pass


def select_tiling(
    height: int, width: int, crop_size: int, max_crops: int
) -> tuple[int, int]:
    """
    Determine the optimal number of tiles to cover an image with overlapping crops.
    """
    if height <= crop_size or width <= crop_size:
        return (1, 1)

    # Minimum required tiles in each dimension
    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)

    # If minimum required tiles exceed max_crops, return proportional distribution
    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))

    # Perfect aspect-ratio tiles that satisfy max_crops
    h_tiles = math.floor(math.sqrt(max_crops * height / width))
    w_tiles = math.floor(math.sqrt(max_crops * width / height))

    # Ensure we meet minimum tile requirements
    h_tiles = max(h_tiles, min_h)
    w_tiles = max(w_tiles, min_w)

    # If we exceeded max_crops, scale down the larger dimension
    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)

    return (max(1, h_tiles), max(1, w_tiles))


def overlap_crop_image(
    image: np.ndarray,
    overlap_margin: int,
    max_crops: int,
    base_size: tuple[int, int] = (378, 378),
    patch_size: int = 14,
):
    """
    Process an image using an overlap-and-resize cropping strategy with margin handling.

    This function takes an input image and creates multiple overlapping crops with
    consistent margins. It produces:
    1. A single global crop resized to base_size
    2. Multiple overlapping local crops that maintain high resolution details
    3. A patch ordering matrix that tracks correspondence between crops

    The overlap strategy ensures:
    - Smooth transitions between adjacent crops
    - No loss of information at crop boundaries
    - Proper handling of features that cross crop boundaries
    - Consistent patch indexing across the full image

    Args:
        image (np.ndarray): Input image as numpy array with shape (H,W,C)
        base_size (tuple[int,int]): Target size for crops, default (378,378)
        patch_size (int): Size of patches in pixels, default 14
        overlap_margin (int): Margin size in patch units, default 4
        max_crops (int): Maximum number of crops allowed, default 12

    Returns:
        OverlapCropOutput: Dictionary containing:
            - crops: A numpy array containing the global crop of the full image (index 0)
                followed by the overlapping cropped regions (indices 1+)
            - tiling: Tuple of (height,width) tile counts
    """
    original_h, original_w = image.shape[:2]

    # Convert margin from patch units to pixels
    margin_pixels = patch_size * overlap_margin
    total_margin_pixels = margin_pixels * 2  # Both sides

    # Calculate crop parameters
    crop_patches = base_size[0] // patch_size  # patches per crop dimension
    crop_window_patches = crop_patches - (2 * overlap_margin)  # usable patches
    crop_window_size = crop_window_patches * patch_size  # usable size in pixels

    # Determine tiling
    tiling = select_tiling(
        original_h - total_margin_pixels,
        original_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    # Pre-allocate crops.
    n_crops = tiling[0] * tiling[1] + 1  # 1 = global crop
    crops = np.zeros(
        (n_crops, base_size[0], base_size[1], image.shape[2]), dtype=np.uint8
    )

    # Resize image to fit tiling
    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    pil_img = PIL.Image.fromarray(image)
    resized = pil_img.resize(
        (int(target_size[1]), int(target_size[0])),
        resample=PIL.Image.Resampling.LANCZOS,
    )
    image = np.asarray(resized)

    # Create global crop
    global_pil = pil_img.resize(
        (int(base_size[1]), int(base_size[0])), resample=PIL.Image.Resampling.LANCZOS
    )
    crops[0] = np.asarray(global_pil)

    for i in range(tiling[0]):
        for j in range(tiling[1]):
            # Calculate crop coordinates
            y0 = i * crop_window_size
            x0 = j * crop_window_size

            # Extract crop with padding if needed
            y_end = min(y0 + base_size[0], image.shape[0])
            x_end = min(x0 + base_size[1], image.shape[1])

            crop_region = image[y0:y_end, x0:x_end]
            crops[
                1 + i * tiling[1] + j, : crop_region.shape[0], : crop_region.shape[1]
            ] = crop_region

    return {"crops": crops, "tiling": tiling}


def prepare_crops(image, max_crops=12, overlap_margin=4):
    if isinstance(image, PIL.Image.Image):
        np_image = np.array(image.convert("RGB"))
    elif isinstance(image, torch.Tensor):
        np_image = image.cpu().detach().numpy()
    else:
        np_image = image
    overlap_crops = overlap_crop_image(
        np_image, max_crops=max_crops, overlap_margin=overlap_margin
    )
    all_crops = overlap_crops["crops"]
    all_crops = np.transpose(all_crops, (0, 3, 1, 2))
    all_crops = all_crops = (
        torch.from_numpy(all_crops)
        .to(device="cpu", dtype=torch.bfloat16)
        .div_(255.0)
        .sub_(0.5)
        .div_(0.5)
    )
    return all_crops.tolist(), overlap_crops["tiling"]


class Moondream3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Moondream3 image processor.
    """

    model_input_names = ["pixel_values", "image_sizes"]
    valid_kwargs = Moondream3ImageProcessorKwargs

    def __init__(
        self,
        max_crops: int = 12,
        overlap_margin: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_crops = max_crops
        self.overlap_margin = overlap_margin
        self._valid_processor_keys = [
            "max_crops",
            "overlap_margin",
        ]

    def preprocess(
        self,
        images: ImageInput,
        max_crops: Optional[int] = None,
        overlap_margin: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            max_crops (`bool`, *optional*, defaults to `self.max_crops`):
            overlap_margin (`dict[str, int]`, *optional*, defaults to `self.overlap_margin`):
        """
        overlap_margin = (
            overlap_margin if overlap_margin is not None else self.overlap_margin
        )
        max_crops = max_crops if max_crops is not None else self.max_crops

        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self._valid_processor_keys,
        )

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)

        if not valid_images(images[0]):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor"
            )

        batch_images = []
        batch_tiling = []
        for image in images:
            pixel_values, tiling = prepare_crops(
                image, max_crops=max_crops, overlap_margin=overlap_margin
            )
            batch_images.append(pixel_values)
            batch_tiling.append(tiling)

        return BatchFeature(
            data={"pixel_values": batch_images, "tiling": batch_tiling},
            tensor_type=return_tensors,
        )


__all__ = ["Moondream3ImageProcessor"]
