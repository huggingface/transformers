# coding=utf-8
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

from typing import Callable, Dict, Optional

from PIL import Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available


IDEFICS_STANDARD_MEAN = (0.48145466, 0.4578275, 0.40821073)
IDEFICS_STANDARD_STD = (0.26862954, 0.26130258, 0.27577711)


def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class IdeficsImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Idefics image processor.

    Args:
        image_size (`int`, *optional*, defaults to `224`):
            Resize to image size
    """

    model_input_names = ["pixel_values"]

    def __init__(self, do_resize: bool = True, image_size: int = 224, **kwargs) -> None:
        super().__init__(**kwargs)

        self.image_size = image_size

    def preprocess(
        self,
        images: ImageInput,
        image_size: Optional[Dict[str, int]] = None,
        transform: Callable = None,
        **kwargs,
    ) -> TensorType.PYTORCH:
        """
        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            image_size (`int`, *optional*, defaults to `self.image_size`):
                Controls the size of the image in `resize` transform in inference.
            transform (`Callable`, *optional*, defaults to `None`):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
                assumed - and then a preset of inference-specific transforms will be applied to the images

        Returns:
            a PyTorch tensor of the processed images

        """
        image_size = image_size if image_size is not None else self.image_size
        size = (image_size, image_size)

        if len(images) == 0:
            return []

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # For training a user needs to pass their own set of transforms as a Callable
        # For reference this is what was used in the original IDEFICS training
        # transform = transforms.Compose([
        #     convert_to_rgb,
        #     transforms.RandomResizedCrop((size, size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=image_mean, std=image_std),
        # ])

        if transform is not None:
            if not is_torch_available():
                raise ImportError("To pass in transforms torch must be installed")
            import torch

            images = [transform(x) for x in images]
            return torch.stack(images)

        images = [convert_to_rgb(x) for x in images]
        # further transforms expect numpy arrays
        images = [to_numpy_array(x) for x in images]
        images = [resize(x, size, resample=PILImageResampling.BICUBIC) for x in images]
        images = [self.rescale(image=image, scale=1 / 255) for image in images]
        images = [self.normalize(x, mean=IDEFICS_STANDARD_MEAN, std=IDEFICS_STANDARD_STD) for x in images]
        images = [to_channel_dimension_format(x, ChannelDimension.FIRST) for x in images]
        # this converts to torch tensors - switch to convert_to_tensors once it becomes available
        images = BatchFeature(data={"pixel_values": images}, tensor_type=TensorType.PYTORCH)["pixel_values"]

        return images
