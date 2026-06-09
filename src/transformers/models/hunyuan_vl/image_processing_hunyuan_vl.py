# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for HunYuanVL."""

import math

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import convert_to_rgb
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    make_list_of_images,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_torchvision_available


def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 512 * 512,
    max_pixels: int = 2048 * 2048,
) -> tuple[int, int]:
    """Rescale ``(height, width)`` to a patch-aligned size while keeping the original aspect ratio."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class HunYuanVLImageProcessor(BaseImageProcessor):
    r"""
    Constructs a slow HunYuanVL image processor that dynamically resizes images to a patch-aligned grid and produces
    flat per-patch pixel features together with the matching ``image_grid_thw`` tensor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input image.
        size (`dict[str, int]`, *optional*):
            Pixel-count bounds expressed as `{"shortest_edge": min_pixels, "longest_edge": max_pixels}`. Defaults to
            `{"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}`.
        resample (`PILImageResampling`, *optional*, defaults to `BICUBIC`):
            Resampling filter used by `PIL.Image.resize`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale pixel values.
        rescale_factor (`int | float`, *optional*, defaults to `1 / 255`):
            Rescaling factor applied to pixel values.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize pixel values.
        image_mean (`float | list[float]`, *optional*):
            Mean used during normalization. Defaults to `OPENAI_CLIP_MEAN`.
        image_std (`float | list[float]`, *optional*):
            Standard deviation used during normalization. Defaults to `OPENAI_CLIP_STD`.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert images to RGB before further processing.
        min_pixels (`int`, *optional*):
            Convenience override for `size["shortest_edge"]`.
        max_pixels (`int`, *optional*):
            Convenience override for `size["longest_edge"]`.
        patch_size (`int`, *optional*, defaults to 16):
            Spatial patch size used by the vision tower.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            Temporal patch size. The open-source variant is image-only and keeps this at `1`.
        merge_size (`int`, *optional*, defaults to 2):
            Spatial merge factor applied by the patch merger.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    do_resize = True
    resample = PILImageResampling.BICUBIC
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    spatial_patch_size = 1
    size = {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}

    def __init__(
        self,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool | None = None,
        rescale_factor: int | float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        size = dict(self.size) if size is None else dict(size)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels

        self.size = size
        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]

        self.do_resize = self.do_resize if do_resize is None else do_resize
        self.resample = self.resample if resample is None else resample
        self.do_rescale = self.do_rescale if do_rescale is None else do_rescale
        self.rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        self.do_normalize = self.do_normalize if do_normalize is None else do_normalize
        self.image_mean = self.image_mean if image_mean is None else image_mean
        self.image_std = self.image_std if image_std is None else image_std
        self.do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb
        self.patch_size = self.patch_size if patch_size is None else patch_size
        self.temporal_patch_size = self.temporal_patch_size if temporal_patch_size is None else temporal_patch_size
        self.merge_size = self.merge_size if merge_size is None else merge_size

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: bool,
        size: dict[str, int],
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        do_convert_rgb: bool,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        width, height = images[0].width, images[0].height
        resized_width, resized_height = width, height
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = image.resize((resized_width, resized_height))

            if do_normalize:
                if not is_torchvision_available():
                    raise ImportError(
                        "HunYuanVLImageProcessor requires torchvision when do_normalize=True. "
                        "Install torchvision or use HunYuanVLImageProcessorPil instead."
                    )
                import torchvision.transforms as transforms

                image = transforms.Compose([transforms.ToTensor(), transforms.Normalize(image_mean, image_std)])(image)
            processed_images.append(image)

        patches = np.array(processed_images)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            1,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.transpose(0, 2, 3, 5, 6, 1, 4, 7)
        flatten_patches = patches.reshape(1 * grid_h * grid_w, channel * patch_size * patch_size)

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        do_convert_rgb: bool | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension | None = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
    ) -> BatchFeature:
        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        elif min_pixels is not None and max_pixels is not None:
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        resample = resample if resample is not None else self.resample
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, "
                "tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        pixel_values, vision_grid_thws = [], []
        for image in images:
            patches, image_grid_thw = self._preprocess(
                image,
                do_resize=do_resize,
                size=size,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
                do_convert_rgb=do_convert_rgb,
            )
            pixel_values.extend(patches)
            vision_grid_thws.append(image_grid_thw)

        return BatchFeature(
            data={
                "pixel_values": np.array(pixel_values),
                "image_grid_thw": np.array(vision_grid_thws),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> tuple[int, int]:
        """Return the `(grid_h, grid_w)` patch counts that the processor would produce for an image of `height x width`."""
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs.get("min_pixels", self.size["shortest_edge"])
        max_pixels = images_kwargs.get("max_pixels", self.size["longest_edge"])
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return resized_height // patch_size, resized_width // patch_size


__all__ = ["HunYuanVLImageProcessor"]
