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
"""Image processor class for LLaVa-NeXT."""

from typing import Optional, Union

import numpy as np

from ...image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    PythonBackend,
    TorchVisionBackend,
    divide_to_patches,
    get_patch_output_size,
    group_images_by_shape,
    reorder_images,
    select_best_resolution,
)
from ...image_transforms import (
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs
from ...utils import TensorType, auto_docstring, is_torchvision_available, logging


if is_torchvision_available():
    import torch
    from torchvision.transforms.v2 import functional as F

logger = logging.get_logger(__name__)


class LlavaNextImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    image_grid_pinpoints (`list[list[int]]`, *optional*):
        A list of possible resolutions to use for processing high resolution images. The best resolution is selected
        based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
        method.
    """

    image_grid_pinpoints: list[list[int]]


class LlavaNextTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for LLaVA-NeXT with patch processing support."""

    def _get_padding_size(self, original_resolution: tuple, target_resolution: tuple):
        """Get padding size for patching (returns list format for F.pad)."""
        original_height, original_width = original_resolution
        target_height, target_width = target_resolution
        paste_x, r_x = divmod(target_width - original_width, 2)
        paste_y, r_y = divmod(target_height - original_height, 2)
        return [paste_x, paste_y, paste_x + r_x, paste_y + r_y]

    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        resample: Union["PILImageResampling", "F.InterpolationMode", int, None],
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """Resizes an image to a target resolution while maintaining aspect ratio."""
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)
        resized_image = self.resize(
            image=image,
            size=SizeDict(height=new_height, width=new_width),
            resample=resample,
        )

        return resized_image

    def _pad_for_patching(
        self, image: "torch.Tensor", target_resolution: tuple, input_data_format: ChannelDimension
    ) -> "torch.Tensor":
        """Pad an image to a target resolution while maintaining aspect ratio."""
        new_resolution = get_patch_output_size(image, target_resolution, input_data_format)
        padding = self._get_padding_size(new_resolution, target_resolution)

        padded_image = F.pad(image, padding=padding)

        return padded_image

    def _get_image_patches(
        self,
        image: "torch.Tensor",
        grid_pinpoints: list[list[int]],
        size: tuple,
        patch_size: int,
        resample: Union["PILImageResampling", "F.InterpolationMode", int, None],
    ) -> list["torch.Tensor"]:
        """Process an image with variable resolutions by dividing it into patches."""
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=ChannelDimension.FIRST
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=ChannelDimension.FIRST)
        patches = divide_to_patches(padded_image, patch_size=patch_size)
        # Resize original image using backend's resize method (handles resample conversion)
        # size is a tuple (height, width), convert to SizeDict
        size_height, size_width = size
        resized_original_image = self.resize(
            image=image,
            size=SizeDict(height=size_height, width=size_width),
            resample=resample,
        )

        image_patches = [resized_original_image] + patches

        return image_patches

    def _pad_for_batching(
        self,
        pixel_values: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        """Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches."""
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
            for image in pixel_values
        ]

        return pixel_values

    def preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        image_grid_pinpoints: list[list[int]],
        resample: Union["PILImageResampling", "F.InterpolationMode", int, None],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for LLaVA-NeXT with patch processing."""
        processed_images = []
        image_sizes = []

        # Backend's resize method handles resample conversion, so we can pass it directly
        # Determine the size tuple
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)

        # Determine the patch size
        if crop_size and crop_size.height:
            patch_size = crop_size.height
        elif size and size.height:
            patch_size = size.height
        else:
            patch_size = size.shortest_edge

        for image in images:
            image_patches = self._get_image_patches(
                image,
                image_grid_pinpoints,
                size=size_tuple,
                patch_size=patch_size,
                resample=resample,
            )

            # Group images by size for batched processing
            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(
                image_patches, disable_grouping=disable_grouping
            )
            for shape, stacked_image_patches in grouped_image_patches.items():
                if do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        resample=resample,
                    )
                if do_center_crop:
                    stacked_image_patches = self.center_crop(stacked_image_patches, crop_size)
                # Fused rescale and normalize
                # Convert lists to tuples for lru_cache compatibility
                image_mean_tuple = tuple(image_mean) if isinstance(image_mean, list) else image_mean
                image_std_tuple = tuple(image_std) if isinstance(image_std, list) else image_std
                stacked_image_patches = self._rescale_and_normalize(
                    stacked_image_patches, do_rescale, rescale_factor, do_normalize, image_mean_tuple, image_std_tuple
                )
                processed_image_patches_grouped[shape] = stacked_image_patches
            processed_image_patches = reorder_images(processed_image_patches_grouped, grouped_image_patches_index)
            processed_image_patches = torch.stack(processed_image_patches, dim=0)
            processed_images.append(processed_image_patches)
            image_sizes.append(get_image_size(image, ChannelDimension.FIRST))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)

        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


class LlavaNextPythonBackend(PythonBackend):
    """Python backend for LLaVA-NeXT with patch processing support."""

    def _get_padding_size(self, original_resolution: tuple, target_resolution: tuple):
        """Get padding size for patching (returns tuple format for np.pad)."""
        original_height, original_width = original_resolution
        target_height, target_width = target_resolution
        paste_x, r_x = divmod(target_width - original_width, 2)
        paste_y, r_y = divmod(target_height - original_height, 2)
        return (paste_y, paste_y + r_y), (paste_x, paste_x + r_x)

    def _resize_for_patching(
        self,
        image: np.ndarray,
        target_resolution: tuple,
        resample: PILImageResampling,
        input_data_format: ChannelDimension,
    ) -> np.ndarray:
        """Resizes an image to a target resolution while maintaining aspect ratio."""
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)
        resized_image = self.resize(image=image, size=SizeDict(height=new_height, width=new_width), resample=resample)

        return resized_image

    def _pad_for_patching(
        self, image: np.ndarray, target_resolution: tuple, input_data_format: ChannelDimension
    ) -> np.ndarray:
        """Pad an image to a target resolution while maintaining aspect ratio."""
        new_resolution = get_patch_output_size(image, target_resolution, input_data_format)
        padding_hw = self._get_padding_size(new_resolution, target_resolution)

        # For channels_first format (C, H, W), add (0, 0) for channel dimension
        # padding_hw is ((before_h, after_h), (before_w, after_w))
        # np.pad expects ((before_C, after_C), (before_H, after_H), (before_W, after_W))
        padding = ((0, 0), padding_hw[0], padding_hw[1])

        # Use np.pad directly for patching padding
        padded_image = np.pad(image, padding, mode="constant", constant_values=0)

        return padded_image

    def get_image_patches(
        self,
        image: np.ndarray,
        grid_pinpoints: list[list[int]],
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> list[np.ndarray]:
        """Process an image with variable resolutions by dividing it into patches."""
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)

        patches = divide_to_patches(padded_image, patch_size=patch_size)

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in patches
        ]

        size_height, size_width = size
        resized_original_image = self.resize(
            image=image,
            size=SizeDict(height=size_height, width=size_width),
            resample=resample,
        )

        image_patches = [resized_original_image] + patches

        return image_patches

    def _pad_for_batching(
        self,
        pixel_values: list[np.ndarray],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> list[np.ndarray]:
        """Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches."""
        max_patch = max(len(x) for x in pixel_values)
        # Use np.pad directly for patch dimension padding
        padded_values = []
        for image in pixel_values:
            # Padding format: ((before_dim0, after_dim0), (before_dim1, after_dim1), ...)
            padding = ((0, max_patch - image.shape[0]), (0, 0), (0, 0), (0, 0))
            padded_image = np.pad(image, padding, mode="constant", constant_values=0)
            padded_values.append(padded_image)

        return padded_values

    def preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        image_grid_pinpoints: list[list[int]],
        resample: Union["PILImageResampling", "F.InterpolationMode", int, None],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for LLaVA-NeXT with patch processing."""
        processed_images = []
        image_sizes = []

        # Backend's resize method handles resample conversion, so we can pass it directly
        # Determine the size tuple
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)

        # Determine the patch size
        if crop_size and crop_size.height:
            patch_size = crop_size.height
        elif size and size.height:
            patch_size = size.height
        else:
            patch_size = size.shortest_edge

        # Backend always uses channels_first format
        input_data_format_backend = ChannelDimension.FIRST

        for image in images:
            # convert image into a list of patches
            # we intentionally use the same data format as the input data format
            image_patches = self.get_image_patches(
                image,
                image_grid_pinpoints,
                size=size_tuple,
                patch_size=patch_size,
                resample=resample,
                data_format=input_data_format_backend,
                input_data_format=input_data_format_backend,
            )

            # preprocess patches
            pixel_values = []
            for patch in image_patches:
                if do_resize:
                    patch = self.resize(image=patch, size=size, resample=resample)

                if do_center_crop:
                    patch = self.center_crop(image=patch, size=crop_size)

                if do_rescale:
                    patch = self.rescale(image=patch, scale=rescale_factor)

                if do_normalize:
                    patch = self.normalize(image=patch, mean=image_mean, std=image_std)

                pixel_values.append(patch)

            pixel_values = np.array(pixel_values)
            processed_images.append(pixel_values)
            image_sizes.append(get_image_size(image, channel_dim=ChannelDimension.FIRST))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)

        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


@auto_docstring
class LlavaNextImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_sizes"]
    valid_kwargs = LlavaNextImageProcessorKwargs

    _backend_classes = {
        "torchvision": LlavaNextTorchVisionBackend,
        "python": LlavaNextPythonBackend,
    }

    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]


__all__ = ["LlavaNextImageProcessor"]
