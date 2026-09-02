# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Idefics."""

from collections.abc import Callable

from ...image_processing_backends import PilBackend
from ...image_utils import ImageInput, PILImageResampling, make_flat_list_of_images
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring, is_torch_available


# Adapted from transformers.models.idefics.image_processing_idefics.IDEFICS_STANDARD_MEAN
IDEFICS_STANDARD_MEAN = [0.48145466, 0.4578275, 0.40821073]

# Adapted from transformers.models.idefics.image_processing_idefics.IDEFICS_STANDARD_STD
IDEFICS_STANDARD_STD = [0.26862954, 0.26130258, 0.27577711]


# Adapted from transformers.models.idefics.image_processing_idefics.IdeficsImageProcessorKwargs
class IdeficsImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    transform (`Callable`, *optional*, defaults to `None`):
        A custom transform function that accepts a single image can be passed for training. For example,
        `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
        assumed - and then a preset of inference-specific transforms will be applied to the images.
    image_size (`int`, *optional*, defaults to `self.image_size`):
        Resize to image size. This is a backward-compatible alias for `size`. When provided, it overrides
        `size` and sets it to `{"height": image_size, "width": image_size}`.
    image_num_channels (`int`, *optional*, defaults to `3`):
        The number of channels of the image.
    """

    transform: Callable | None
    image_size: int
    image_num_channels: int


@auto_docstring
class IdeficsImageProcessorPil(PilBackend):
    valid_kwargs = IdeficsImageProcessorKwargs
    resample = PILImageResampling.BICUBIC
    image_mean = IDEFICS_STANDARD_MEAN
    image_std = IDEFICS_STANDARD_STD
    size = {"height": 224, "width": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    image_num_channels = 3

    def __init__(self, **kwargs: Unpack[IdeficsImageProcessorKwargs]):
        image_size = kwargs.pop("image_size", None)
        if image_size is not None:
            kwargs["size"] = {"height": image_size, "width": image_size}
        super().__init__(**kwargs)
        self.image_size = self.size.height

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[IdeficsImageProcessorKwargs],
    ):
        r"""
        transform (`Callable`, *optional*, defaults to `None`):
            A custom transform function that accepts a single image can be passed for training. If `None`,
            inference-mode transforms are applied.
        """
        transform = kwargs.pop("transform", None)
        if transform is not None:
            if not is_torch_available():
                raise ImportError("To pass in `transform` torch must be installed")
            import torch

            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)
            images = [transform(x) for x in images]
            return torch.stack(images)
        return super().preprocess(images, **kwargs).pixel_values


__all__ = ["IdeficsImageProcessorPil"]
