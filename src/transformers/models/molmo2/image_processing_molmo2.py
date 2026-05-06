# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Image processor class for Molmo2"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, logging


logger = logging.get_logger(__name__)


# Copied from transformers.models.cohere2_vision.image_processing_cohere2_vision.get_all_supported_aspect_ratios
def get_all_supported_aspect_ratios(max_image_tiles: int) -> list[tuple[int, int]]:
    """
    Computes all allowed aspect ratios for a given maximum number of input tiles.

    This function calculates all possible arrangements of tiles that can be formed
    within the constraint of the maximum number of tiles. Each arrangement is
    represented by its aspect ratio (width/height) and the corresponding tile configuration.

    Args:
        max_image_tiles (`int`):
            The maximum number of tiles allowed.

    Returns:
        `list[tuple[int, int]]`: A list of tuples, each tuple representing a valid (width, height)
        configuration in terms of number of tiles.

    Example:
        >>> get_all_supported_aspect_ratios(4)
        [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)]

    """
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles:
                aspect_ratios.append((width, height))
    return aspect_ratios


# Copied from transformers.models.cohere2_vision.image_processing_cohere2_vision.get_optimal_tiled_canvas
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]:
    possible_resolutions = get_all_supported_aspect_ratios(max_image_tiles)
    possible_resolutions = sorted(possible_resolutions, key=lambda x: x[0] * x[1])
    image_height, image_width = original_image_size
    patch_size_height, patch_size_width = target_tile_size  # (height == width)

    candidate_resolutions = np.array(possible_resolutions) * patch_size_height
    # tiles following (width, height) order to align with aspect ratio convention
    tile_size = np.stack([image_width, image_height])
    required_scales = candidate_resolutions / tile_size
    required_scale = np.min(required_scales, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        best_grid = possible_resolutions[np.argmax(required_scale)]
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        best_grid = possible_resolutions[np.argmin(required_scale)]
    return best_grid  # (width, height)


def batch_pixels_to_patches(array: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = array.reshape(n_crops, h_patches, patch_size, w_patches, patch_size)
        array = array.permute(0, 1, 3, 2, 4)
        array = array.reshape(n_crops, h_patches * w_patches, patch_size * patch_size)
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        array = array.reshape(n_crops, h_patches, patch_size, w_patches, patch_size, c)
        array = array.permute(0, 1, 3, 2, 4, 5)
        array = array.reshape(n_crops, h_patches * w_patches, patch_size * patch_size * c)
        return array


def arange_for_pooling(
    idx_arr: torch.Tensor,
    pool_h: int,
    pool_w: int,
) -> torch.Tensor:
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = F.pad(
        idx_arr,
        (w_pad // 2, (w_pad + 1) // 2, h_pad // 2, (h_pad + 1) // 2),
        mode="constant",
        value=-1,
    )
    h, w = idx_arr.shape[0] // pool_h, idx_arr.shape[1] // pool_w
    return idx_arr.reshape(h, pool_h, w, pool_w).permute(0, 2, 1, 3).reshape(h, w, pool_h * pool_w)


class Molmo2ImagesKwargs(ImagesKwargs, total=False):
    """
    max_crops (`int`, *optional*, defaults to 8):
        Maximum number of crops to use per image.
    overlap_margins (`list[int]`, *optional*, defaults to `[4, 4]`):
        Overlap margins (in patches) for overlapping crop extraction.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    pooling_size (`list[int]`, *optional*, defaults to `[2, 2]`):
        The pooling size of the vision adapter.
    """

    max_crops: int | None
    overlap_margins: list[int] | None
    patch_size: int | None
    pooling_size: list[int] | None


@auto_docstring
class Molmo2ImageProcessor(TorchvisionBackend):
    valid_kwargs = Molmo2ImagesKwargs
    model_input_names = ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 378, "width": 378}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    max_crops = 8
    overlap_margins = [4, 4]
    patch_size = 14
    pooling_size = [2, 2]

    def __init__(self, **kwargs: Unpack[Molmo2ImagesKwargs]):
        super().__init__(**kwargs)

    def _build_resized_image(
        self,
        image_chw: torch.Tensor,
        base_image_input_size: list[int],
        resample: PILImageResampling,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chw_resized = self.resize(
            image_chw,
            size=SizeDict(height=base_image_input_size[0], width=base_image_input_size[1]),
            resample=resample,
            antialias=False,
        )
        chw_float = self.rescale(chw_resized.float(), scale=1.0 / 255.0)
        chw_normalized = self.normalize(chw_float, mean=image_mean, std=image_std)
        resized = chw_normalized.permute(1, 2, 0).unsqueeze(0)  # → [1, H, W, 3]
        crop_patch_w = base_image_input_size[1] // image_patch_size
        crop_patch_h = base_image_input_size[0] // image_patch_size
        resize_idx = torch.arange(crop_patch_w * crop_patch_h, dtype=torch.int32).reshape(crop_patch_h, crop_patch_w)
        return resized, resize_idx

    def _build_overlapping_crops(
        self,
        image_chw: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        base_image_input_size: list[int],
        resample: PILImageResampling,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose an image into a set of overlapping crops

        :return crop_arr: [n_crops, h, w, 3] The crops
        :return patch_idx: [overlap_patch_h, overlap_patch_w] For each patch in the resized image
                            the crops were extracted from, what patch in `crop_arr` it corresponds to
        """
        _, original_image_h, original_image_w = image_chw.shape
        crop_size = base_image_input_size[0]
        if base_image_input_size[0] != base_image_input_size[1]:
            raise ValueError(f"Expected square base_image_input_size, got {base_image_input_size}")

        left_margin, right_margin = overlap_margins
        total_margin_pixels = image_patch_size * (right_margin + left_margin)  # pixels removed per dim
        crop_patches = base_image_input_size[0] // image_patch_size  # patches per crop dim
        crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
        crop_window_size = crop_window_patches * image_patch_size
        crop_patch_w = base_image_input_size[1] // image_patch_size
        crop_patch_h = base_image_input_size[0] // image_patch_size

        # Decide how to tile the image, to account for the overlap margins we compute the tiling
        # as if we had an image without the margins and were using a crop size without the margins
        effective_image_size = (original_image_h - total_margin_pixels, original_image_w - total_margin_pixels)
        tiling_w, tiling_h = get_optimal_tiled_canvas(
            original_image_size=effective_image_size,
            target_tile_size=(crop_window_size, crop_window_size),
            min_image_tiles=1,
            max_image_tiles=max_crops,
        )

        src_h = tiling_h * crop_window_size + total_margin_pixels
        src_w = tiling_w * crop_window_size + total_margin_pixels
        chw_resized = self.resize(
            image_chw,
            size=SizeDict(height=src_h, width=src_w),
            resample=resample,
            antialias=False,
        )
        chw_float = self.rescale(chw_resized.float(), scale=1.0 / 255.0)
        chw_normalized = self.normalize(chw_float, mean=image_mean, std=image_std)
        src = chw_normalized.permute(1, 2, 0)  # → HWC

        # Now we have to split the image into crops, and track what patches came from
        # where in `patch_idx_arr`
        n_crops = tiling_h * tiling_w
        crop_arr = torch.empty((n_crops, crop_size, crop_size, 3), dtype=src.dtype)
        patch_idx_arr = torch.empty((n_crops, crop_patch_h, crop_patch_w), dtype=torch.int32)
        on_crop = 0
        for i in range(tiling_h):
            # Slide over `src` by `crop_window_size` steps, but extract crops of size `crops_size`
            # which results in overlapping crop windows
            y0 = i * crop_window_size
            for j in range(tiling_w):
                x0 = j * crop_window_size
                crop_arr[on_crop] = src[y0 : y0 + crop_size, x0 : x0 + crop_size]
                patch_idx = torch.arange(crop_patch_w * crop_patch_h, dtype=torch.int32).reshape(
                    crop_patch_h, crop_patch_w
                )
                patch_idx += on_crop * crop_patch_h * crop_patch_w

                # Mask out idx that are in the overlap region
                if i != 0:
                    patch_idx[:left_margin, :] = -1
                if j != 0:
                    patch_idx[:, :left_margin] = -1
                if i != tiling_h - 1:
                    patch_idx[-right_margin:, :] = -1
                if j != tiling_w - 1:
                    patch_idx[:, -right_margin:] = -1
                patch_idx_arr[on_crop] = patch_idx
                on_crop += 1

        # `patch_idx_arr` is ordered crop-by-crop, here we transpose `patch_idx_arr`
        # so it is ordered left-to-right order
        patch_idx_arr = patch_idx_arr.reshape(tiling_h, tiling_w, crop_patch_h, crop_patch_w)
        patch_idx_arr = patch_idx_arr.permute(0, 2, 1, 3)
        patch_idx_arr = patch_idx_arr.reshape(-1)

        # Now get the parts not in the overlap region, so it should map each patch in `src`
        # to the correct patch it should come from in `crop_arr`
        patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
            src.shape[0] // image_patch_size,
            src.shape[1] // image_patch_size,
        )
        return crop_arr, patch_idx_arr

    def _image_to_patches_and_grids(
        self,
        image_chw: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        base_image_input_size: list[int],
        resample: PILImageResampling,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
        image_pooling_w: int,
        image_pooling_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return image_grids, the shape of each (low-res, high-res) image after pooling
        :return crops, the image crops to processes with the ViT
        :return pooled_patch_idx, for each patch_id tokens in `image_tokens`, the indices of the
                                    patches in `crops` to pool for that token, masked with -1
        """
        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        pooling_w = image_pooling_w
        pooling_h = image_pooling_h
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        crop_arr, patch_idx_arr = self._build_overlapping_crops(
            image_chw,
            max_crops,
            overlap_margins,
            base_image_input_size,
            resample,
            image_mean,
            image_std,
            image_patch_size,
        )
        pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
        h, w = pooling_idx.shape[:2]
        pooling_idx = pooling_idx.reshape([-1, pooling_h * pooling_w])

        # Finally do the same for the global image
        resized, resize_idx = self._build_resized_image(
            image_chw,
            base_image_input_size,
            resample,
            image_mean,
            image_std,
            image_patch_size,
        )
        crop_arr = torch.cat([resized, crop_arr], 0)

        resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
        resized_h, resized_w = resize_idx.shape[:2]
        resize_idx = resize_idx.reshape([-1, pooling_h * pooling_w])

        # Global image goes first, so the order of patches in previous crops gets increased
        pooling_idx = torch.where(pooling_idx >= 0, pooling_idx + crop_patch_h * crop_patch_w, -1)
        pooling_idx = torch.cat([resize_idx, pooling_idx])
        image_grid = torch.tensor([[resized_h, resized_w, h, w]], dtype=torch.int64)

        return image_grid, batch_pixels_to_patches(crop_arr, image_patch_size), pooling_idx

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Molmo2ImagesKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        do_convert_rgb: bool,
        max_crops: int,
        overlap_margins: list[int],
        patch_size: int,
        pooling_size: list[int],
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        base_image_input_size = [size.height, size.width]
        image_pooling_h, image_pooling_w = pooling_size

        batch_grids = []
        batch_crops = []
        batch_pooled_patches_idx = []
        batch_num_crops = []

        for image in images:
            image_grid, crops, pooled_idx = self._image_to_patches_and_grids(
                image,
                max_crops,
                overlap_margins,
                base_image_input_size,
                resample,
                image_mean,
                image_std,
                patch_size,
                image_pooling_w,
                image_pooling_h,
            )
            batch_grids.append(image_grid)
            batch_crops.append(crops)
            batch_pooled_patches_idx.append(pooled_idx)
            batch_num_crops.append(crops.shape[0])

        return BatchFeature(
            data={
                "pixel_values": torch.cat(batch_crops, 0),
                "image_token_pooling": torch.cat(batch_pooled_patches_idx, 0),
                "image_grids": torch.cat(batch_grids, 0),
                "image_num_crops": torch.tensor(batch_num_crops, dtype=torch.int64),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """
        Returns the number of IMAGE_PATCH_TOKEN tokens inserted into the sequence for an image of
        the given size. Used by vLLM to compute the number of placeholders without running the full
        image processor.

        Args:
            height (`int`): Image height in pixels.
            width (`int`): Image width in pixels.
            images_kwargs (`dict`, *optional*): Overrides for processor defaults (e.g. `max_crops`, `patch_size`).

        Returns:
            `int`: Total number of patch tokens (low-res + high-res, before special tokens).
        """
        if images_kwargs is None:
            images_kwargs = {}
        max_crops = images_kwargs.get("max_crops", self.max_crops)
        overlap_margins = images_kwargs.get("overlap_margins", self.overlap_margins)
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        pooling_size = images_kwargs.get("pooling_size", self.pooling_size)
        size = images_kwargs.get("size", self.size)

        base_h = size["height"] if isinstance(size, dict) else size.height
        base_w = size["width"] if isinstance(size, dict) else size.width
        left_margin, right_margin = overlap_margins
        pooling_h, pooling_w = pooling_size

        total_margin_pixels = patch_size * (left_margin + right_margin)
        crop_patches = base_h // patch_size
        crop_window_patches = crop_patches - (left_margin + right_margin)
        crop_window_size = crop_window_patches * patch_size

        effective_h = height - total_margin_pixels
        effective_w = width - total_margin_pixels
        tiling_w, tiling_h = get_optimal_tiled_canvas(
            original_image_size=(effective_h, effective_w),
            target_tile_size=(crop_window_size, crop_window_size),
            min_image_tiles=1,
            max_image_tiles=max_crops,
        )

        high_res_h = (tiling_h * crop_window_patches + left_margin + right_margin)
        high_res_w = (tiling_w * crop_window_patches + left_margin + right_margin)
        h = math.ceil(high_res_h / pooling_h)
        w = math.ceil(high_res_w / pooling_w)

        crop_patch_h = base_h // patch_size
        crop_patch_w = base_w // patch_size
        resized_h = math.ceil(crop_patch_h / pooling_h)
        resized_w = math.ceil(crop_patch_w / pooling_w)

        return resized_h * resized_w + h * w


__all__ = ["Molmo2ImageProcessor"]
