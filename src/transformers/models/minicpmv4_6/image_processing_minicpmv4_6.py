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


def ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


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
    valid_kwargs = MiniCPMV4_6ImageProcessorKwargs
    model_input_names = ["pixel_values", "target_sizes"]

    def __init__(self, **kwargs: Unpack[MiniCPMV4_6ImageProcessorKwargs]):
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

    def reshape_by_patch(self, image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
        """Reshape ``[C, H, W]`` into NaViT patchified format ``[C, patch_size, H*W/patch_size]``."""
        num_channels = image.shape[0]
        patches = torch.nn.functional.unfold(
            image.unsqueeze(0), (patch_size, patch_size), stride=(patch_size, patch_size)
        )
        patches = patches.reshape(num_channels, patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(num_channels, patch_size, -1)
        return patches

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
        return_tensors: str | TensorType | None = None,
        disable_grouping: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        per_image_pixel_values: list[list[torch.Tensor]] = []
        per_image_target_sizes: list[list[list[int]]] = []
        all_grids: list[list[int]] = []

        for image in images:
            image_size = image.shape[-2:]
            best_grid = None

            if slice_mode:
                best_grid = self.get_sliced_grid(image_size, max_slice_nums, scale_resolution)

            image_patches = [image]
            if do_resize:
                # Always resize source image
                source_h, source_w = self.find_best_resize(
                    image_size, scale_resolution, patch_size, allow_upscale=(best_grid is None)
                )
                source_img = self.resize(image, size=SizeDict(height=source_h, width=source_w), resample=resample)

                # Collect all patches for this image: [source, *slices]
                image_patches = [source_img]
                patch_height = patch_width = 0
                if best_grid is not None:
                    refine_h, refine_w = self.get_refine_size(
                        image_size, best_grid, scale_resolution, patch_size, allow_upscale=True
                    )
                    refine_img = self.resize(image, size=SizeDict(height=refine_h, width=refine_w), resample=resample)
                    refine_height, refine_width = refine_img.shape[-2:]
                    grid_y, grid_x = best_grid
                    patch_height, patch_width = refine_height // grid_y, refine_width // grid_x
                    slice_patches = divide_to_patches(refine_img, (patch_height, patch_width))
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

            image_pv = [self.reshape_by_patch(processed_patches[0], patch_size)]
            image_ts = [[source_h // patch_size, source_w // patch_size]]
            for processed_slice in processed_patches[1:]:
                image_pv.append(self.reshape_by_patch(processed_slice, patch_size))
                image_ts.append([patch_height // patch_size, patch_width // patch_size])

            per_image_pixel_values.append(image_pv)
            per_image_target_sizes.append(image_ts)
            all_grids.append(best_grid if best_grid is not None else [0, 0])

        all_pv = [pv for sublist in per_image_pixel_values for pv in sublist]
        pixel_values = torch.cat(all_pv, dim=-1).unsqueeze(0)

        all_ts = [ts for sublist in per_image_target_sizes for ts in sublist]
        target_sizes = torch.tensor(all_ts, dtype=torch.int32)

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


__all__ = ["MiniCPMV4_6ImageProcessor"]
