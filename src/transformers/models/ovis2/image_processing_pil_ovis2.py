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
"""PIL Image processor class for OVIS2."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    infer_channel_dimension_format,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from .image_processing_ovis2 import (
    Ovis2ImageProcessorKwargs,
    get_min_tile_covering_grid,
    get_optimal_tiled_canvas,
)


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class Ovis2ImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    crop_to_patches = False
    min_patches = 1
    max_patches = 12
    use_covering_area_grid = True
    valid_kwargs = Ovis2ImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Ovis2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Ovis2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def crop_image_to_patches(
        self,
        image: np.ndarray,
        min_patches: int,
        max_patches: int,
        use_covering_area_grid: bool = True,
        covering_threshold: float = 0.9,
        patch_size: SizeDict | None = None,
        resample: "PILImageResampling | int | None" = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        Mirrors TorchvisionBackend.crop_image_to_patches.
        """
        # Normalize to CHW when called directly (e.g. from tests); _preprocess already receives CHW
        input_data_format = infer_channel_dimension_format(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = image.shape[-2:]

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
        resized_image = self.resize(image, SizeDict(height=target_height, width=target_width), resample=resample)
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
            patch_image = resized_image[:, box[1] : box[3], box[0] : box[2]]
            processed_images.append(patch_image)

        if len(processed_images) != 1:
            thumbnail_img = self.resize(image, patch_size, resample=resample)
            processed_images.insert(0, thumbnail_img)

        return processed_images, [num_rows, num_columns]

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = False,
        min_patches: int = 1,
        max_patches: int = 12,
        use_covering_area_grid: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if crop_to_patches and max_patches > 1:
            # Crop to patches first
            processed_images = []
            grids = []
            for image in images:
                patches, grid = self.crop_image_to_patches(
                    image,
                    min_patches,
                    max_patches,
                    patch_size=size,
                    use_covering_area_grid=use_covering_area_grid,
                    resample=resample,
                )
                processed_images.extend(patches)
                grids.append(grid)
            images = processed_images
        else:
            grids = [[1, 1] for _ in range(len(images))]

        # Process all images (including patches if any) through the standard pipeline
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images, "grids": grids}, tensor_type=return_tensors)


__all__ = ["Ovis2ImageProcessorPil"]
