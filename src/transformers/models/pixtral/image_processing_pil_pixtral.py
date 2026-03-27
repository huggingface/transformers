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
"""Image processor class for Pixtral."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import PaddingMode, pad
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from .image_processing_pixtral import PixtralImageProcessorKwargs, get_resize_output_image_size


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class PixtralImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    patch_size = {"height": 16, "width": 16}
    size = {"longest_edge": 1024}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = PixtralImageProcessorKwargs

    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(self, **kwargs: Unpack[PixtralImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[PixtralImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        patch_size: SizeDict,
        resample: "PILImageResampling | int | None" = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The longest edge is resized to size["longest_edge"], with aspect ratio preserved.
        Output dimensions are aligned to patch_size.
        """
        if size.longest_edge:
            size_tuple = (size.longest_edge, size.longest_edge)
        elif size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            raise ValueError("size must contain either 'longest_edge' or 'height' and 'width'.")

        if patch_size.height and patch_size.width:
            patch_size_tuple = (patch_size.height, patch_size.width)
        else:
            raise ValueError("patch_size must contain 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image, size=size_tuple, patch_size=patch_size_tuple, input_data_format=ChannelDimension.FIRST
        )
        return super().resize(
            image, size=SizeDict(height=output_size[0], width=output_size[1]), resample=resample, **kwargs
        )

    def _pad_for_batching(
        self,
        pixel_values: list[np.ndarray],
        image_sizes: list[tuple[int, int]],
    ) -> np.ndarray:
        """Pad images to form a batch of same shape."""
        max_shape = (max(s[0] for s in image_sizes), max(s[1] for s in image_sizes))
        padded = []
        for img, size in zip(pixel_values, image_sizes):
            pad_h = max_shape[0] - size[0]
            pad_w = max_shape[1] - size[1]
            padded_img = pad(
                img,
                padding=((0, pad_h), (0, pad_w)),
                mode=PaddingMode.CONSTANT,
                constant_values=0,
                input_data_format=ChannelDimension.FIRST,
            )
            padded.append(padded_img)
        return np.stack(padded)

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
        patch_size: dict[str, int] | SizeDict | None = None,
        **kwargs,
    ) -> BatchFeature:
        patch_size = get_size_dict(patch_size or self.patch_size, default_to_square=True)
        patch_size_sd = SizeDict(**patch_size)

        processed_images = []
        batch_image_sizes = []

        for image in images:
            if do_resize:
                image = self.resize(image, size=size, patch_size=patch_size_sd, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            processed_images.append(image)
            batch_image_sizes.append(get_image_size(image, channel_dim=ChannelDimension.FIRST))

        padded_images = self._pad_for_batching(
            pixel_values=processed_images,
            image_sizes=batch_image_sizes,
        )

        return BatchFeature(
            data={"pixel_values": padded_images, "image_sizes": batch_image_sizes}, tensor_type=return_tensors
        )


__all__ = ["PixtralImageProcessorPil"]
