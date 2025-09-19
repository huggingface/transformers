# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for PerceptionLM."""

import math
from functools import reduce
from typing import Optional, Union

import numpy as np
import torch
from torchvision.transforms import functional as F

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    get_image_size,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring


class PerceptionLMFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    vision_input_type (`str`, *optional*, defaults to `"thumb+tile"`):
        Vision processing strategy. `"thumb+tile"` uses both thumbnails and multiple tiles for
        multi-scale processing, otherwise uses single tile for lower memory usage.
    tile_size (`int`, *optional*, defaults to `448`):
        Height and width dimension (in pixels) of each tile used for image processing.
    max_num_tiles (`int`, *optional*, defaults to `36`):
        Maximum number of tiles an image can be split into based on its aspect ratio.
    """

    vision_input_type: str = "thumb+tile"
    tile_size: int = 448
    max_num_tiles: int = 36


@auto_docstring
class PerceptionLMImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    size = {"width": 448, "height": 448}  # for backward compatibility in tests
    valid_kwargs = PerceptionLMFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[PerceptionLMFastImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images, **kwargs: Unpack[PerceptionLMFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    @staticmethod
    def _factors(n: int):
        """Return all factors of a number."""
        return set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )

    def _find_supported_aspect_ratios(self):
        """
        This function computes all the allowed aspect ratios for a fixed
        number of input chunks. The order of returned items matters for the result of `_fit_image_to_canvas` function.
        If tie exists in `_fit_image_to_canvas`, the latter in `_find_supported_aspect_ratios` wins.

        For example, with `num_tiles=5`, it will return:
        {
            0.2: [(1, 5)],
            5.0: [(5, 1)],
            0.25: [(1, 4)],
            1.0: [(2, 2), (1, 1)],
            4.0: [(4, 1)],
            0.3333333333333333: [(1, 3)],
            3.0: [(3, 1)],
            0.5: [(1, 2)],
            2.0: [(2, 1)]
        }
        """
        asp_dict = {}
        for chunk_size in range(self.max_num_tiles, 0, -1):
            _factors = sorted(self._factors(chunk_size))
            _asp_ratios = [(x, chunk_size // x) for x in _factors]
            for ratio in _asp_ratios:
                k = ratio[0] / ratio[1]
                if k not in asp_dict:
                    asp_dict[k] = [ratio]
                else:
                    asp_dict[k].append(ratio)
        return asp_dict

    def _get_image_height_width(
        self, image_width: int, image_height: int, target_width: int, target_height: int
    ) -> tuple[int, int]:
        """
        Given image width, height and target width, height for the canvas, return the dimensions of how the image would be resized
        with aspect ratio preservation.
        """
        scale = image_width / image_height

        if scale > 1.0:
            # Width is larger than height

            # Rescaling factor is the minimum of the two scaling factors. Else one side would be outside of the canvas.
            rescaling_factor = min(target_width / image_width, target_height / image_height)

            # Set new width to target width and height to the rescaled height.
            new_w = rescaling_factor * image_width
            new_h = math.floor(new_w / scale)

        else:
            # Height is larger than width

            # Rescaling factor is the minimum of the two scaling factors. Else one side would be outside of the canvas.
            rescaling_factor = min(target_width / image_width, target_height / image_height)

            # Set new height to target height and width to the rescaled width.
            new_h = rescaling_factor * image_height
            new_w = math.floor(new_h * scale)

        return new_w, new_h

    def _fit_image_to_canvas(self, img_width: int, img_height: int, tile_size: int):
        """
        Given an image width, height and target number of chunks this function will see if the image
        can be fit into any of the canvases that can be build from arranging the tiles in a grid.
        If the image can be fit onto several canvases, it will return the canvas where the shorter edge
        of the image will be largest.
        """
        # Initialize the optimal canvas to None. If no canvas is found where image fits, function returns None.
        optimal_canvas = None
        optimal_image_width_height = None

        scale = img_width / img_height

        # Gather all potential supported image resolutions and iterate through them to find best match
        potential_arrangements = [
            item for sublist in self._find_supported_aspect_ratios().values() for item in sublist
        ]
        for n_w, n_h in potential_arrangements:
            # Compute the canvas size
            canvas_width, canvas_height = n_w * tile_size, n_h * tile_size

            # Check if image can fit into the canvas without downsampling
            if canvas_width >= img_width and canvas_height >= img_height:
                # If we did not find a good canvas yet, we will use the current one
                if optimal_canvas is None:
                    # Set optimal canvas and determine the actual image height and width in the canvas with aspect ratio preserving resampling
                    optimal_canvas = (n_w, n_h)
                    optimal_image_width_height = self._get_image_height_width(
                        image_width=img_width,
                        image_height=img_height,
                        target_width=n_w * tile_size,
                        target_height=n_h * tile_size,
                    )
                else:
                    # If we already found an optimal canvas before, we will check if the shorter edge of the image will be larger than the current optimal canvas.
                    # This means we can potentially upsample the image resolution which is beneficial to performance.
                    image_width_height = self._get_image_height_width(
                        image_width=img_width,
                        image_height=img_height,
                        target_width=n_w * tile_size,
                        target_height=n_h * tile_size,
                    )
                    # Llama3V dynamic tiling. Prioritize biggest canvas.
                    if (scale < 1.0 and (image_width_height[0] >= optimal_image_width_height[0])) or (
                        scale >= 1.0 and (image_width_height[1] >= optimal_image_width_height[1])
                    ):
                        optimal_canvas = (n_w, n_h)
                        optimal_image_width_height = image_width_height
        return optimal_canvas

    def _find_closest_aspect_ratio(self, img_width: int, img_height: int, tile_size: int) -> tuple:
        """
        Given an image width, height and target number of chunks
        this function will find the closest supported aspect ratio.
        """
        target_aspect_ratio = img_width / img_height
        asp_dict = self._find_supported_aspect_ratios()
        closest_aspect_ratio = None
        if target_aspect_ratio >= 1:
            closest_aspect_ratio = min(
                [k for k in asp_dict if k <= target_aspect_ratio],
                key=lambda x: abs(x - target_aspect_ratio),
            )
            tiles_given_aspect_ratio = asp_dict[closest_aspect_ratio]
            # select largest width
            return max(tiles_given_aspect_ratio, key=lambda x: x[0])
        else:
            closest_aspect_ratio = min(
                [k for k in asp_dict if k > target_aspect_ratio],
                key=lambda x: abs(1 / x - 1 / target_aspect_ratio),
            )
            tiles_given_aspect_ratio = asp_dict[closest_aspect_ratio]
            # select largest height
            return max(tiles_given_aspect_ratio, key=lambda x: x[1])

    def _split(self, image: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        # Split image into number of required tiles (width x height)
        batch_size, num_channels, height, width = image.size()
        image = image.view(batch_size, num_channels, nch, height // nch, ncw, width // ncw)
        # Permute dimensions to reorder the axes
        image = image.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
        image = image.view(batch_size, ncw * nch, num_channels, height // nch, width // ncw)
        return image

    def resize(
        self,
        image: np.ndarray,
        tile_size: int,
        max_num_tiles: int,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        height, width = get_image_size(image, channel_dim=input_data_format)
        if max_num_tiles > 1:
            aspect_ratio = self._fit_image_to_canvas(img_width=width, img_height=height, tile_size=tile_size)
            if aspect_ratio is None:
                # If we did not find a canvas, we have to find the closest aspect ratio and downsample the image
                aspect_ratio = self._find_closest_aspect_ratio(img_width=width, img_height=height, tile_size=tile_size)
        else:
            aspect_ratio = (1, 1)
        new_width, new_height = aspect_ratio[0] * tile_size, aspect_ratio[1] * tile_size
        image = F.resize(image, (new_height, new_width), interpolation=resample)
        return image, aspect_ratio

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        do_rescale: Optional[bool],
        rescale_factor: Optional[Union[int, float]],
        do_normalize: Optional[bool],
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        vision_input_type: str,
        tile_size: int,
        max_num_tiles: int,
        return_tensors: Optional[Union[str, TensorType]],
        disable_grouping: bool,
        **kwargs: Unpack[PerceptionLMFastImageProcessorKwargs],
    ) -> BatchFeature:
        # Group images by size for batched transformation
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                if vision_input_type == "thumb+tile":
                    thumbnails, _ = self.resize(stacked_images, tile_size, max_num_tiles=1)
                    images_for_tiling, (tiles_w, tiles_h) = self.resize(
                        stacked_images, tile_size, max_num_tiles=max_num_tiles
                    )
                    image_tiles = self._split(images_for_tiling, tiles_w, tiles_h)
                    stacked_images = torch.cat([thumbnails.unsqueeze(1), image_tiles], dim=1)
                else:  # vanilla single tile for low memory devices
                    stacked_images, _ = self.resize(stacked_images, tile_size, max_num_tiles=1)

            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = [p[None] if p.ndim == 3 else p for p in processed_images]  # add tiles dimension if needed
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["PerceptionLMImageProcessorFast"]
