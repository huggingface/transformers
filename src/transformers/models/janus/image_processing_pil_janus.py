# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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
"""PIL Image processor class for Janus."""

from collections.abc import Iterable

import numpy as np
import PIL

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import resize as np_resize
from ...image_transforms import to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    to_numpy_array,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)
from ...utils.import_utils import requires
from .image_processing_janus import JanusImageProcessorKwargs


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class JanusImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    min_size = 14
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    valid_kwargs = JanusImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[JanusImageProcessorKwargs]):
        super().__init__(**kwargs)
        image_mean = getattr(self, "image_mean", None)
        if image_mean is None:
            background_color = (127, 127, 127)
        else:
            background_color = tuple(int(x * 255) for x in image_mean)
        self.background_color = tuple(background_color)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[JanusImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        min_size: int,
        resample: PILImageResampling | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize so largest side becomes size, with min_size floor."""
        if size.height is None or size.width is None or size.height != size.width:
            raise ValueError(
                f"Output height and width must be the same. Got height={size.height} and width={size.width}"
            )
        target_size = size.height

        height, width = image.shape[-2:]
        max_size = max(height, width)

        delta = target_size / max_size
        new_height = max(round(height * delta), min_size)
        new_width = max(round(width * delta), min_size)

        return np_resize(
            image,
            size=(new_height, new_width),
            resample=resample or self.resample,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: int | tuple[int, int, int] = 0,
    ) -> np.ndarray:
        """Pad an image to a square based on the longest edge."""
        height, width = image.shape[-2:]
        num_channels = image.shape[0]

        if height == width:
            return image

        max_dim = max(height, width)

        if isinstance(background_color, int):
            background_color = [background_color]
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        padded_image = np.zeros((num_channels, max_dim, max_dim), dtype=image.dtype)
        for i, color in enumerate(background_color):
            padded_image[i, :, :] = color

        if width > height:
            start = (max_dim - height) // 2
            padded_image[:, start : start + height, :] = image
        else:
            start = (max_dim - width) // 2
            padded_image[:, :, start : start + width] = image

        return padded_image

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        min_size: int,
        return_tensors: str | TensorType | None,
        do_pad: bool = True,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, min_size=min_size, resample=resample)
            if do_pad:
                image = self.pad_to_square(image, background_color=self.background_color)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def postprocess(
        self,
        images: ImageInput,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        input_data_format: str | None = None,
        return_tensors: str | None = None,
    ):
        """Applies post-processing to the decoded image tokens by reversing transformations applied during preprocessing."""
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = 1.0 / self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_flat_list_of_images(images)  # Ensures input is a list

        if isinstance(images[0], PIL.Image.Image):
            return images if len(images) > 1 else images[0]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])  # Determine format dynamically

        pixel_values = []

        for image in images:
            image = to_numpy_array(image)  # Ensure NumPy format

            if do_normalize:
                image = self.unnormalize(
                    image=image, image_mean=image_mean, image_std=image_std, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)
                image = image.clip(0, 255).astype(np.uint8)

            if do_normalize and do_rescale and return_tensors == "PIL.Image.Image":
                image = to_channel_dimension_format(image, ChannelDimension.LAST, input_channel_dim=input_data_format)
                image = PIL.Image.fromarray(image)

            pixel_values.append(image)

        data = {"pixel_values": pixel_values}
        return_tensors = return_tensors if return_tensors != "PIL.Image.Image" else None

        return BatchFeature(data=data, tensor_type=return_tensors)

    def unnormalize(
        self,
        image: np.ndarray,
        image_mean: float | Iterable[float],
        image_std: float | Iterable[float],
        input_data_format: str | ChannelDimension | None = None,
    ) -> np.ndarray:
        """
        Unnormalizes `image` using the mean and standard deviation specified by `mean` and `std`.
        image = (image * image_std) + image_mean
        Args:
            image (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)` or `(num_channels, image_size, image_size)`):
                Batch of pixel values to postprocess.
            image_mean (`float` or `Iterable[float]`):
                The mean to use for unnormalization.
            image_std (`float` or `Iterable[float]`):
                The standard deviation to use for unnormalization.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        num_channels = 3

        if isinstance(image_mean, Iterable):
            if len(image_mean) != num_channels:
                raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(image_mean)}")
        else:
            image_mean = [image_mean] * num_channels

        if isinstance(image_std, Iterable):
            if len(image_std) != num_channels:
                raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(image_std)}")
        else:
            image_std = [image_std] * num_channels

        rev_image_mean = tuple(-mean / std for mean, std in zip(image_mean, image_std))
        rev_image_std = tuple(1 / std for std in image_std)
        image = self.normalize(
            image=image, mean=rev_image_mean, std=rev_image_std, input_data_format=input_data_format
        )
        return image


__all__ = ["JanusImageProcessorPil"]
