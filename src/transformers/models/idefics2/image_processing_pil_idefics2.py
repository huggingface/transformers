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
"""PIL Image processor class for Idefics2."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode, pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    make_nested_list_of_images,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from .image_processing_idefics2 import (
    Idefics2ImageProcessorKwargs,
    convert_to_rgb,
    get_max_height_width,
    get_resize_output_image_size,
)


def _make_pixel_mask(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Make pixel mask: 1=valid, 0=padding. Images are CHW."""
    h, w = image.shape[-2:]
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:h, :w] = 1
    return mask


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class Idefics2ImageProcessorPil(PilBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    do_convert_rgb = True
    do_image_splitting = False
    size = {"shortest_edge": 378, "longest_edge": 980}
    default_to_square = False
    model_input_names = ["pixel_values", "pixel_attention_mask"]
    valid_kwargs = Idefics2ImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Idefics2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Idefics2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        """Convert an image to RGB format."""
        return convert_to_rgb(image)

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, expected_ndims=expected_ndims)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        **kwargs,
    ) -> np.ndarray:
        """Resize using Idefics2 shortest_edge/longest_edge logic. Images are always CHW."""
        if size.shortest_edge and size.longest_edge:
            new_size = get_resize_output_image_size(image, size)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError("Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys.")
        return super().resize(image, SizeDict(height=new_size[0], width=new_size[1]), resample=resample, **kwargs)

    def split_images(self, image: np.ndarray) -> list[np.ndarray]:
        """Split an image into 4 equal sub-images, and concatenate that sequence with the original image. Images are CHW."""
        height, width = image.shape[-2:]
        mid_width = width // 2
        mid_height = height // 2

        return [
            image[:, :mid_height, :mid_width],
            image[:, :mid_height, mid_width:],
            image[:, mid_height:, :mid_width],
            image[:, mid_height:, mid_width:],
            image,
        ]

    def pad(
        self,
        image: np.ndarray,
        padded_size: tuple[int, int],
        fill: int = 0,
        return_pixel_mask: bool = True,
    ):
        """Pad image to padded_size. Mirrors TorchvisionBackend.pad. Images are always CHW."""
        original_size = image.shape[-2:]
        padding_bottom = padded_size[0] - original_size[0]
        padding_right = padded_size[1] - original_size[1]

        if padding_bottom < 0 or padding_right < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded size: {padded_size}, original size: {original_size}."
            )

        pixel_mask = _make_pixel_mask(image, output_size=padded_size) if return_pixel_mask else None

        if original_size != padded_size:
            padding = ((0, padding_bottom), (0, padding_right))
            image = pad(
                image,
                padding,
                mode=PaddingMode.CONSTANT,
                constant_values=fill,
                data_format="channels_first",
                input_data_format="channels_first",
            )

        return image, pixel_mask

    def _preprocess(
        self,
        images: list[list[np.ndarray]],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        do_image_splitting: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """Process a batch of images. Mirrors TorchvisionBackend._preprocess with per-image loops."""
        if do_image_splitting:
            new_images = []
            for batch_images in images:
                new_batch = []
                for image in batch_images:
                    new_batch.extend(self.split_images(image))
                new_images.append(new_batch)
            images = new_images

        if do_resize:
            images = [
                [self.resize(image=img, size=size, resample=resample) for img in batch_images]
                for batch_images in images
            ]

        if do_rescale:
            images = [[self.rescale(img, rescale_factor) for img in batch_images] for batch_images in images]
        if do_normalize:
            images = [[self.normalize(img, image_mean, image_std) for img in batch_images] for batch_images in images]

        if do_pad:
            max_num_images = max(len(images_) for images_ in images)
            max_height, max_width = get_max_height_width(images)
            num_channels = images[0][0].shape[0]

            padded_images_list = [
                [np.zeros((num_channels, max_height, max_width), dtype=np.float32) for _ in range(max_num_images)]
                for _ in range(len(images))
            ]
            pixel_attention_masks = [
                [np.zeros((max_height, max_width), dtype=np.int64) for _ in range(max_num_images)]
                for _ in range(len(images))
            ]

            for i, batch_images in enumerate(images):
                for j, image in enumerate(batch_images):
                    padded_images_list[i][j], pixel_attention_masks[i][j] = self.pad(image, (max_height, max_width))
            images = padded_images_list

        if do_pad:
            data = {
                "pixel_values": np.array(images),
                "pixel_attention_mask": np.array(pixel_attention_masks),
            }
        elif return_tensors == "pt":
            data = {"pixel_values": np.asarray(images)}
        else:
            data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Idefics2ImageProcessorPil"]
