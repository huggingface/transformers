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
"""PIL Image processor class for MiniCPM-V 4.6."""

import math

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import divide_to_patches
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


def ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


class MiniCPMV4_6ImageProcessorPilKwargs(ImagesKwargs, total=False):
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
class MiniCPMV4_6ImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    do_resize = True
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
    valid_kwargs = MiniCPMV4_6ImageProcessorPilKwargs
    model_input_names = ["pixel_values", "target_sizes"]

    def __init__(self, **kwargs: Unpack[MiniCPMV4_6ImageProcessorPilKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, **kwargs):
        # Drop `do_resize`, model resizes based on auto-inferred size at run-time
        kwargs.pop("do_resize")
        super()._validate_preprocess_kwargs(**kwargs)

    def find_best_resize(
        self,
        image_size: tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = image_size
        if (height * width > scale_resolution * scale_resolution) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # factor 4 = two successive 2×2 spatial merges (ViT insert merger + downsample MLP)
        best_width = ensure_divide(width, patch_size * 4)
        best_height = ensure_divide(height, patch_size * 4)
        return best_height, best_width

    def get_refine_size(
        self,
        image_size: tuple[int, int],
        grid: list[int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        height, width = image_size
        grid_y, grid_x = grid
        refine_width = ensure_divide(width, grid_x)
        refine_height = ensure_divide(height, grid_y)

        best_height, best_width = self.find_best_resize(
            image_size=(refine_height / grid_y, refine_width / grid_x),
            scale_resolution=scale_resolution,
            patch_size=patch_size,
            allow_upscale=allow_upscale,
        )
        return best_height * grid_y, best_width * grid_x

    def get_sliced_grid(
        self,
        image_size: tuple[int, int],
        max_slice_nums: int,
        scale_resolution: int,
    ) -> list[int] | None:
        original_height, original_width = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
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
                        best_grid = [num_cols, num_rows]
                        min_error = error
        return best_grid

    def reshape_by_patch(self, image: np.ndarray, patch_size: int) -> np.ndarray:
        """Reshape ``[C, H, W]`` into NaViT patchified format ``[C, patch_size, H*W/patch_size]``."""
        num_channels, height, width = image.shape
        num_patches_h, num_patches_w = height // patch_size, width // patch_size
        patches = image.reshape(num_channels, num_patches_h, patch_size, num_patches_w, patch_size)
        patches = patches.transpose(0, 2, 1, 3, 4)
        return patches.reshape(num_channels, patch_size, -1)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[MiniCPMV4_6ImageProcessorPilKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
        slice_mode: bool,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        per_image_pixel_values: list[list[np.ndarray]] = []
        per_image_target_sizes: list[list[list[int]]] = []
        all_grids: list[list[int]] = []

        for image in images:
            image_size = image.shape[-2:]
            best_grid = None

            if slice_mode:
                best_grid = self.get_sliced_grid(image_size, max_slice_nums, scale_resolution)

            source_img = image
            source_height, source_width = image_size
            if do_resize:
                source_height, source_width = self.find_best_resize(
                    image_size, scale_resolution, patch_size, allow_upscale=(best_grid is None)
                )
                source_img = self.resize(
                    image, size=SizeDict(height=source_height, width=source_width), resample=resample
                )

            if do_rescale:
                source_img = self.rescale(source_img, rescale_factor)
            if do_normalize:
                source_img = self.normalize(source_img, image_mean, image_std)

            image_pv = [self.reshape_by_patch(source_img, patch_size)]
            image_ts = [[source_height // patch_size, source_width // patch_size]]

            if best_grid is not None:
                refine_img = image
                refine_h, refine_w = image_size
                if do_resize:
                    refine_h, refine_w = self.get_refine_size(
                        image_size, best_grid, scale_resolution, patch_size, allow_upscale=True
                    )
                    refine_img = self.resize(image, size=SizeDict(height=refine_h, width=refine_w), resample=resample)

                refine_height, refine_width = refine_img.shape[-2:]
                grid_y, grid_x = best_grid
                patch_height, patch_width = refine_height // grid_y, refine_width // grid_x
                slice_patches = divide_to_patches(refine_img, (refine_height // grid_y, refine_width // grid_x))

                for patch_arr in slice_patches:
                    if do_rescale:
                        patch_arr = self.rescale(patch_arr, rescale_factor)
                    if do_normalize:
                        patch_arr = self.normalize(patch_arr, image_mean, image_std)
                    image_pv.append(self.reshape_by_patch(patch_arr, patch_size))
                    image_ts.append([patch_height // patch_size, patch_width // patch_size])

            per_image_pixel_values.append(image_pv)
            per_image_target_sizes.append(image_ts)
            all_grids.append(best_grid if best_grid is not None else [0, 0])

        all_pv = [pv for sublist in per_image_pixel_values for pv in sublist]
        pixel_values = np.concatenate(all_pv, axis=-1)[np.newaxis, ...]

        all_ts = [ts for sublist in per_image_target_sizes for ts in sublist]
        target_sizes = np.array(all_ts, dtype=np.int32)

        num_patches_per_image = [len(sublist) for sublist in per_image_pixel_values]

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "target_sizes": target_sizes,
                "grids": all_grids,
                "num_patches_per_image": num_patches_per_image,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids", "num_patches_per_image"],
        )


__all__ = ["MiniCPMV4_6ImageProcessorPil"]
