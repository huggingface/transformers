# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

import math

import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import divide_to_patches, group_images_by_shape, reorder_images
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


class MiniCPMV4_6ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    max_slice_nums (`int`, *optional*, defaults to 9):
        Maximum number of slices when splitting a high-resolution image.
    scale_resolution (`int`, *optional*, defaults to 448):
        Target resolution for individual slices.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    slice_mode (`bool`, *optional*, defaults to `True`):
        Whether to split images into multiple slices for higher resolution.
    downsample_mode (`str`, *optional*, defaults to `"16x"`):
        Visual token downsampling mode. `"16x"` applies full merge; `"4x"` keeps
        4x more tokens.
    use_image_id (`bool`, *optional*, defaults to `True`):
        Whether to prepend an image-id tag (``<image_id>N</image_id>``) before
        each image placeholder. Consumed by the Processor for placeholder
        generation, not by the image processing pipeline itself.
    """

    max_slice_nums: int
    scale_resolution: int
    patch_size: int
    slice_mode: bool
    downsample_mode: str
    use_image_id: bool


@auto_docstring
class MiniCPMV4_6ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    do_resize = False
    do_rescale = True
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    max_slice_nums = 9
    scale_resolution = 448
    patch_size = 14
    slice_mode = True
    downsample_mode = "16x"
    use_image_id = True
    valid_kwargs = MiniCPMV4_6ImageProcessorKwargs
    model_input_names = ["pixel_values", "target_sizes"]

    # ------------------------------------------------------------------
    # Slicing geometry helpers (pure math, no image data)
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_divide(length: int, divisor: int) -> int:
        return max(round(length / divisor) * divisor, divisor)

    @classmethod
    def _find_best_resize(
        cls, original_size: tuple[int, int], scale_resolution: int, patch_size: int, allow_upscale: bool = False
    ) -> tuple[int, int]:
        width, height = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # factor 4 = two successive 2×2 spatial merges (ViT insert merger + downsample MLP)
        best_width = cls._ensure_divide(width, patch_size * 4)
        best_height = cls._ensure_divide(height, patch_size * 4)
        return (best_width, best_height)

    @classmethod
    def _get_refine_size(
        cls,
        original_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        width, height = original_size
        grid_x, grid_y = grid
        refine_width = cls._ensure_divide(width, grid_x)
        refine_height = cls._ensure_divide(height, grid_y)
        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y
        best_grid_size = cls._find_best_resize(
            (grid_width, grid_height), scale_resolution, patch_size, allow_upscale=allow_upscale
        )
        return (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    @staticmethod
    def _get_sliced_grid(
        image_size: tuple[int, int], max_slice_nums: int, scale_resolution: int, never_split: bool = False
    ) -> list[int] | None:
        original_width, original_height = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or never_split:
            return None

        best_grid = [1, 1]
        min_error = float("inf")
        for num_slices in [multiple - 1, multiple, multiple + 1]:
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows == 0:
                    num_cols = num_slices // num_rows
                    error = abs(log_ratio - math.log(num_rows / num_cols))
                    if error < min_error:
                        best_grid = [num_rows, num_cols]
                        min_error = error
        return best_grid

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_by_patch(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
        """Reshape ``[C, H, W]`` into NaViT patchified format ``[C, patch_size, H*W/patch_size]``."""
        num_channels = image.shape[0]
        patches = torch.nn.functional.unfold(
            image.unsqueeze(0), (patch_size, patch_size), stride=(patch_size, patch_size)
        )
        patches = patches.reshape(num_channels, patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(num_channels, patch_size, -1)
        return patches

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[MiniCPMV4_6ImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
        slice_mode: bool,
        downsample_mode: str,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        token_divisor = 4 if downsample_mode == "4x" else 16
        disable_grouping = kwargs.get("disable_grouping")

        per_image_pixel_values: list[list[torch.Tensor]] = []
        per_image_target_sizes: list[list[list[int]]] = []
        all_grids: list[list[int]] = []
        all_source_visual_tokens: list[int] = []
        all_patch_visual_tokens: list[int] = []

        for image in images:
            _, height, width = image.shape
            original_size = (width, height)

            if slice_mode:
                best_grid = self._get_sliced_grid(original_size, max_slice_nums, scale_resolution)
            else:
                best_grid = None

            # Always resize source image
            source_wh = self._find_best_resize(
                original_size, scale_resolution, patch_size, allow_upscale=(best_grid is None)
            )
            source_img = self.resize(image, size=SizeDict(height=source_wh[1], width=source_wh[0]), resample=resample)

            source_width, source_height = source_wh
            source_visual_tokens = source_height * source_width // (patch_size * patch_size * token_divisor)

            # Collect all patches for this image: [source, *slices]
            image_patches = [source_img]
            patch_visual_tokens = 0
            patch_height = patch_width = 0
            if best_grid is not None:
                refine_wh = self._get_refine_size(
                    original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
                )
                refine_img = self.resize(
                    image, size=SizeDict(height=refine_wh[1], width=refine_wh[0]), resample=resample
                )
                _, refine_height, refine_width = refine_img.shape
                grid_x, grid_y = best_grid
                slice_patches = divide_to_patches(refine_img, (refine_height // grid_y, refine_width // grid_x))
                patch_height, patch_width = slice_patches[0].shape[1], slice_patches[0].shape[2]
                patch_visual_tokens = patch_height * patch_width // (patch_size * patch_size * token_divisor)
                image_patches.extend(slice_patches)

            # Group patches by shape and batch rescale + normalize
            grouped_patches, grouped_index = group_images_by_shape(image_patches, disable_grouping=disable_grouping)
            processed_grouped = {}
            for shape, stacked in grouped_patches.items():
                stacked = self.rescale_and_normalize(
                    stacked.float(), do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                processed_grouped[shape] = stacked
            processed_patches = reorder_images(processed_grouped, grouped_index)

            image_pv = [self._reshape_by_patch(processed_patches[0], patch_size)]
            image_ts = [[source_height // patch_size, source_width // patch_size]]
            for processed_slice in processed_patches[1:]:
                image_pv.append(self._reshape_by_patch(processed_slice, patch_size))
                image_ts.append([patch_height // patch_size, patch_width // patch_size])

            per_image_pixel_values.append(image_pv)
            per_image_target_sizes.append(image_ts)
            all_grids.append(best_grid if best_grid is not None else [0, 0])
            all_source_visual_tokens.append(source_visual_tokens)
            all_patch_visual_tokens.append(patch_visual_tokens)

        return BatchFeature(
            data={
                "pixel_values": per_image_pixel_values,
                "target_sizes": per_image_target_sizes,
                "grids": all_grids,
                "source_image_visual_tokens": all_source_visual_tokens,
                "patch_visual_tokens": all_patch_visual_tokens,
            },
        )


__all__ = ["MiniCPMV4_6ImageProcessor"]
