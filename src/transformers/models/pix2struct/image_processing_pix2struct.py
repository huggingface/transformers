# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for PIX2STRUCT."""
import io
import math
from typing import Dict, Optional, Union

import numpy as np
from huggingface_hub import hf_hub_download

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import convert_to_rgb, normalize, to_channel_dimension_format, to_pil_image
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    get_image_size,
    is_batched,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, is_vision_available, logging
from ...utils.import_utils import requires_backends


if is_vision_available():
    import textwrap

    import PIL
    from PIL import Image, ImageDraw, ImageFont

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)
DEFAULT_FONT_PATH = "ybelkada/fonts"


# adapted from: https://discuss.pytorch.org/t/tf-image-extract-patches-in-pytorch/171409/2
def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
    `patch_width`, `num_channels`x `patch_height` x `patch_width`)

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """
    requires_backends(torch_extract_patches, ["torch"])

    image_tensor = image_tensor.unsqueeze(0)
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches.unsqueeze(0)


class Pix2StructImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Pix2Struct image processor.

    Args:
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image. According to Pix2Struct paper and code, the patch size is 16x16.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. According to Pix2Struct paper and code, the image is normalized with its own mean and standard
            deviation.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        max_patches (`int`, *optional*, defaults to 2048):
            The maximum number of patches to extract from the image as per the [Pix2Struct
            paper](https://arxiv.org/pdf/2210.03347.pdf).
    """

    model_input_names = ["flattened_patches"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_normalize: bool = True,
        patch_size: Dict[str, int] = None,
        max_patches: int = 2048,
        is_vqa: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size if patch_size is not None else {"height": 16, "width": 16}
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.max_patches = max_patches
        self.is_vqa = is_vqa

    def extract_flattened_patches(self, image: np.ndarray, max_patches: int, patch_size: dict, **kwargs) -> np.ndarray:
        """
        Extract flattened patches from an image.

        Args:
            image (`np.ndarray`):
                Image to extract flattened patches from.
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`dict`):
                Dictionary containing the patch height and width.
        """
        requires_backends(self.extract_flattened_patches, "torch")

        # convert to torch if
        image = to_channel_dimension_format(image, ChannelDimension.FIRST)
        image = torch.from_numpy(image)

        patch_height, patch_width = patch_size["height"], patch_size["width"]
        image_height, image_width = get_image_size(image)

        # maximize scale s.t.
        scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        # [1, rows, columns, patch_height * patch_width * image_channels]
        patches = torch_extract_patches(image, patch_height, patch_width)

        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

        # [rows * columns, 1]
        row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

        # Offset by 1 so the ids do not contain zeros, which represent padding.
        row_ids += 1
        col_ids += 1

        # Prepare additional patch features.
        # [rows * columns, 1]
        row_ids = row_ids.to(torch.float32)
        col_ids = col_ids.to(torch.float32)

        # [rows * columns, 2 + patch_height * patch_width * image_channels]
        result = torch.cat([row_ids, col_ids, patches], -1)

        # [max_patches, 2 + patch_height * patch_width * image_channels]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        result = to_numpy_array(result)

        return result

    def normalize(
        self, image: np.ndarray, data_format: Optional[Union[str, ChannelDimension]] = None, **kwargs
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        The image std is to mimic the tensorflow implementation of the `per_image_standardization`:
        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

        Args:
            image (`np.ndarray`):
                Image to normalize.
            data_format (`str`):
                The data format of the image. Can be either "ChannelDimension.channels_first" or
                "ChannelDimension.channels_last".
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32)

        # take mean across the whole `image`
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

        return normalize(image, mean=mean, std=adjusted_stddev, data_format=data_format, **kwargs)

    def render_text(
        self,
        text: str,
        text_size: int = 36,
        text_color: str = "black",
        background_color: str = "white",
        left_padding: int = 5,
        right_padding: int = 5,
        top_padding: int = 5,
        bottom_padding: int = 5,
        font_bytes: Optional[bytes] = None,
        font_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Render text. This script is entirely adapted from the original script that can be found here:
        https://github.com/google-research/pix2struct/blob/main/pix2struct/preprocessing/preprocessing_utils.py

        Args:
            text (`str`):
                Text to render.
            text_size (`int`):
                Size of the text.
            text_color (`str`):
                Color of the text.
            background_color (`str`):
                Color of the background.
            left_padding (`int`):
                Padding on the left.
            right_padding (`int`):
                Padding on the right.
            top_padding (`int`):
                Padding on the top.
            bottom_padding (`int`):
                Padding on the bottom.
            font_bytes (`bytes`):
                Bytes of the font to use. If `None`, the default font will be used.
            font_path (`str`):
                Path to the font to use. If `None`, the default font will be used.
        """
        requires_backends(self.render_text, "vision")
        # Add new lines so that each line is no more than 80 characters.

        wrapper = textwrap.TextWrapper(width=80)
        lines = wrapper.wrap(text=text)
        wrapped_text = "\n".join(lines)

        if font_bytes is not None and font_path is None:
            font = io.BytesIO(font_bytes)
        elif font_path is not None:
            font = font_path
        else:
            font = hf_hub_download(DEFAULT_FONT_PATH, "Arial.TTF")
        font = ImageFont.truetype(font, encoding="UTF-8", size=text_size)

        # Use a temporary canvas to determine the width and height in pixels when
        # rendering the text.
        temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))
        _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

        # Create the actual image with a bit of padding around the text.
        image_width = text_width + left_padding + right_padding
        image_height = text_height + top_padding + bottom_padding
        image = Image.new("RGB", (image_width, image_height), background_color)
        draw = ImageDraw.Draw(image)
        draw.text(xy=(left_padding, top_padding), text=wrapped_text, fill=text_color, font=font)
        return image

    def render_header(self, image: ImageInput, header: str, **kwargs):
        """
        Renders a header on a PIL image and returns a new PIL image.

        Args:
            image (`PIL.Image`):
                The image to render the header on.
            header (`str`):
                The header text.

        Returns:
            `PIL.Image`: The image with the header rendered.
        """
        requires_backends(self.render_header, "vision")

        # Convert to PIL image if necessary
        image = to_pil_image(image)

        header_image = self.render_text(header, **kwargs)
        new_width = max(header_image.width, image.width)

        new_height = int(image.height * (new_width / image.width))
        new_header_height = int(header_image.height * (new_width / header_image.width))

        new_image = Image.new("RGB", (new_width, new_height + new_header_height), "white")
        new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
        new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

        # Convert back to the original framework if necessary
        image = to_numpy_array(image)

        return new_image

    def preprocess(
        self,
        images: ImageInput,
        max_patches: Optional[int] = None,
        header_text: Optional[str] = None,
        patch_size: Optional[Dict[str, int]] = None,
        do_normalize: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: bool = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images. The processor first computes the maximum possible number of
        aspect-ration preserving patches of size `patch_size` that can be extracted from the image. It then padds the
        image with zeros to make the image respect the constraint of `max_patches`. Before extracting the patches the
        images are standardized following the tensorflow implementation of `per_image_standardization`
        (https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization).


        Args:
            images (`ImageInput`):
                Image to preprocess.
            max_patches (`int`, *optional*):
                Maximum number of patches to extract.
            header_text (`str`, *optional*):
                Text to render as a header.
            patch_size (`dict`, *optional*):
                Dictionary containing the patch height and width.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format considered by the normalization function:
                    - `ChannelDimension.FIRST` or `'channels_first'`: The channel dimension is the first dimension.
                    - `ChannelDimension.LAST` or `'channels_last'`: The channel dimension is the last dimension.
        """
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_size = patch_size if patch_size is not None else self.patch_size
        max_patches = max_patches if max_patches is not None else self.max_patches
        is_vqa = self.is_vqa

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        if is_vqa:
            if header_text is None:
                raise ValueError("A header text must be provided for VQA models.")
            font_bytes = kwargs.pop("font_bytes", None)
            font_path = kwargs.pop("font_path", None)
            images = [
                self.render_header(image, header_text, font_bytes=font_bytes, font_path=font_path) for image in images
            ]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, data_format=data_format) for image in images]

        # convert to torch tensor and permute
        images = [
            self.extract_flattened_patches(image=image, max_patches=max_patches, patch_size=patch_size)
            for image in images
        ]

        # create attention mask in numpy
        attention_masks = [(image.sum(axis=-1) != 0).astype(np.float32) for image in images]

        encoded_outputs = BatchFeature(
            data={"flattened_patches": images, "attention_mask": attention_masks}, tensor_type=return_tensors
        )

        return encoded_outputs
