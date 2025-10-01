# coding=utf-8
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

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)
from .image_processing_ovis2 import get_min_tile_covering_grid, get_optimal_tiled_canvas


class Ovis2ImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
        crop_to_patches (`bool`, *optional*, defaults to `False`):
            Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
            `preprocess` method.
        min_patches (`int`, *optional*, defaults to 1):
            The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
            set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
        max_patches (`int`, *optional*, defaults to 12):
            The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
            set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
        use_covering_area_grid (`bool`, *optional*, defaults to `True`):
            Whether to use the covering area grid to determine the number of patches. Only has an effect if
            `crop_to_patches` is set to `True`. Can be overridden by the `use_covering_area_grid` parameter in the
            `preprocess` method.
    """

    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]
    use_covering_area_grid: Optional[bool]


@auto_docstring
class Ovis2ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    default_to_square = None
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    crop_to_patches = False
    min_patches = 1
    max_patches = 12
    use_covering_area_grid = True
    valid_kwargs = Ovis2ImageProcessorKwargs

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Ovis2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def crop_image_to_patches(
        self,
        images: "torch.Tensor",
        min_patches: int,
        max_patches: int,
        use_covering_area_grid: bool = True,
        covering_threshold: float = 0.9,
        patch_size: Optional[Union[tuple, int, dict]] = None,
        interpolation: Optional["F.InterpolationMode"] = None,
    ):
        """
        Crop the images to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target patch size and the minimum and maximum number of patches.
        The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

        Args:
            images (`torch.Tensor`):
                The images to be cropped.
            min_patches (`int`):
                The minimum number of patches to be extracted from the image.
            max_patches (`int`):
                The maximum number of patches to be extracted from the image.
            use_covering_area_grid (`bool`, *optional*, defaults to `True`):
                Whether to use the original OVIS2 approach: compute the minimal number of tiles that cover at least 90%
                of the image area. If `False`, the closest aspect ratio to the target is used.
            covering_threshold (`float`, *optional*, defaults to `0.9`):
                The threshold for the covering area. Only has an effect if `use_covering_area_grid` is set to `True`.
            patch_size (`int`, `Tuple[int, int]`, `dict`, *optional*):
                The size of the output patches.
                The format of the image data. If `None`, the format is inferred from the input image.
            interpolation (`InterpolationMode`):
                Resampling filter to use if resizing the image.

        Returns:
            List[`PIL.Image.Image`] or List[np.ndarray]: The list of cropped images.
        """
        num_image = images.shape[0]
        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = images.shape[-2:]

        if use_covering_area_grid:
            # Use the original OVIS2 approach: compute the minimal number of tiles that cover at least 90% of the image area
            num_columns, num_rows = get_min_tile_covering_grid(
                (original_height, original_width),
                target_patch_size=patch_size_height,  # square patch size
                max_image_tiles=max_patches,
                covering_threshold=covering_threshold,
            )
        else:
            # find the closest aspect ratio to the target
            num_columns, num_rows = get_optimal_tiled_canvas(
                (original_height, original_width), (patch_size_height, patch_size_width), min_patches, max_patches
            )

        # calculate the target width and height
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = self.resize(
            images, SizeDict(height=target_height, width=target_width), interpolation=interpolation
        )
        # split the image into patches
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            # split the image
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            processed_images.append(patch_image)

        if len(processed_images) != 1:
            thumbnail_img = self.resize(images, patch_size, interpolation=interpolation)
            processed_images.insert(0, thumbnail_img)

        processed_images = torch.stack(processed_images, dim=0).transpose(0, 1).contiguous()
        grid = [[num_rows, num_columns] for _ in range(num_image)]

        return processed_images, grid

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        crop_to_patches: bool,
        min_patches: int,
        max_patches: int,
        use_covering_area_grid: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        if crop_to_patches and max_patches > 1:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
            processed_images_grouped = {}
            grids = {}
            for shape, stacked_images in grouped_images.items():
                stacked_images, grid = self.crop_image_to_patches(
                    stacked_images,
                    min_patches,
                    max_patches,
                    patch_size=size,
                    use_covering_area_grid=use_covering_area_grid,
                    interpolation=interpolation,
                )
                processed_images_grouped[shape] = stacked_images
                grids[shape] = grid
            images = reorder_images(processed_images_grouped, grouped_images_index)
            images = [image for images_list in images for image in images_list]
            grids = reorder_images(grids, grouped_images_index)
        else:
            grids = [[1, 1] for _ in range(len(images))]

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(data={"pixel_values": processed_images, "grids": grids}, tensor_type=return_tensors)


__all__ = ["Ovis2ImageProcessorFast"]
