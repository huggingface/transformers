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
"""Image processor class for Pix2Struct."""

import math

import numpy as np
import torch
from PIL import Image

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import to_channel_dimension_format, to_pil_image
from ...image_utils import ChannelDimension, ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, requires_backends
from ...utils.import_utils import requires
from .image_processing_pix2struct import Pix2StructImageProcessorKwargs, render_text, torch_extract_patches


@auto_docstring
@requires(backends=("vision", "torch", "torchvision"))
class Pix2StructImageProcessorPil(PilBackend):
    rescale_factor = None
    do_normalize = True
    do_convert_rgb = True
    patch_size = {"height": 16, "width": 16}
    max_patches = 2048
    is_vqa = False
    valid_kwargs = Pix2StructImageProcessorKwargs
    model_input_names = ["flattened_patches", "attention_mask"]

    def _standardize_kwargs(self, patch_size: dict[str, int] | SizeDict | None = None, **kwargs) -> dict:
        """
        Process custom Pix2Struct kwargs, specifically converting patch_size to SizeDict.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        if patch_size is not None and not isinstance(patch_size, SizeDict):
            kwargs["patch_size"] = SizeDict(**get_size_dict(size=patch_size, param_name="patch_size"))
        else:
            kwargs["patch_size"] = patch_size

        return kwargs

    def _validate_preprocess_kwargs(self, **kwargs):
        """
        Skip standard validation as Pix2Struct uses custom preprocessing.
        """
        # Pix2Struct doesn't use standard resize/rescale/normalize parameters
        # so we skip the default validation
        pass

    def render_header(
        self, image: np.ndarray, header: str, font_bytes: bytes | None = None, font_path: str | None = None
    ) -> np.ndarray:
        """
        Render header text on image using numpy arrays.

        Args:
            image (`np.ndarray`):
                Image array in channel-first format (C, H, W).
            header (`str`):
                Header text to render.
            font_bytes (`bytes`, *optional*):
                Font bytes to use for rendering.
            font_path (`str`, *optional*):
                Path to font file to use for rendering.

        Returns:
            `np.ndarray`: Image with header in channel-first format (C, H, W).
        """
        # Convert numpy array to PIL

        image_pil = to_pil_image(image, input_data_format=ChannelDimension.FIRST)

        # Render header text as PIL image
        header_image = render_text(header, font_bytes=font_bytes, font_path=font_path)

        # Calculate new dimensions
        new_width = max(header_image.width, image_pil.width)
        new_height = int(image_pil.height * (new_width / image_pil.width))
        new_header_height = int(header_image.height * (new_width / header_image.width))

        # Create new image and paste header and original image
        new_image = Image.new("RGB", (new_width, new_height + new_header_height), "white")
        new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
        new_image.paste(image_pil.resize((new_width, new_height)), (0, new_header_height))

        # Convert back to numpy array (channel-first)

        result = np.array(new_image).astype(np.uint8)
        result = to_channel_dimension_format(result, ChannelDimension.FIRST, input_channel_dim=ChannelDimension.LAST)

        return result

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using per-image mean and standard deviation.

        Args:
            image (`np.ndarray`):
                Image array of shape (C, H, W).

        Returns:
            `np.ndarray`: Normalized image of shape (C, H, W).
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32)

        # Compute mean and std
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

        return (image - mean) / adjusted_stddev

    def extract_flattened_patches(self, image: np.ndarray, max_patches: int, patch_size: SizeDict) -> np.ndarray:
        """
        Extract flattened patches from an image. Uses torch for patch extraction.

        Args:
            image (`np.ndarray`):
                Image array of shape (channels, height, width).
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`SizeDict`):
                Dictionary containing patch height and width.

        Returns:
            `np.ndarray`: Flattened patches with row/column IDs of shape (max_patches, patch_dim).
        """
        requires_backends(self, "torch")
        # Convert to torch for patch extraction (pix2struct requires torch for unfold)
        image_torch = torch.from_numpy(image)
        patch_height, patch_width = patch_size.height, patch_size.width
        channels, image_height, image_width = image_torch.shape

        # Calculate scale to maximize patches while respecting max_patches
        scale = (max_patches * (patch_height / image_height) * (patch_width / image_width)) ** 0.5
        num_feasible_rows = max(min(int(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(int(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        # Resize image
        image_torch = image_torch.unsqueeze(0)  # Add batch dimension
        image_torch = torch.nn.functional.interpolate(
            image_torch.float(),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image_torch = image_torch.squeeze(0)

        # Extract patches: [1, rows, columns, patch_height * patch_width * channels]
        patches = torch_extract_patches(image_torch.unsqueeze(0), patch_height, patch_width)

        rows, columns, depth = patches.shape[1], patches.shape[2], patches.shape[3]

        # Reshape to [rows * columns, depth]
        patches = patches.squeeze(0).reshape(rows * columns, depth)

        # Create row and column IDs
        row_ids = torch.arange(rows).reshape(rows, 1).repeat(1, columns).reshape(rows * columns, 1)
        col_ids = torch.arange(columns).reshape(1, columns).repeat(rows, 1).reshape(rows * columns, 1)

        # Offset by 1 so IDs don't contain zeros (which represent padding)
        row_ids = (row_ids + 1).float()
        col_ids = (col_ids + 1).float()

        # Concatenate row_ids, col_ids, and patches: [rows * columns, 2 + depth]
        result = torch.cat([row_ids, col_ids, patches], dim=-1)

        # Pad to max_patches: [max_patches, 2 + depth]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        # Convert back to numpy
        return result.numpy()

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        header_text: str | list[str] | None = None,
        **kwargs: Unpack[Pix2StructImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        header_text (`Union[str, list[str]]`, *optional*):
            Text to render as a header. Only has an effect if `image_processor.is_vqa` is `True`.
        """
        return super().preprocess(images, header_text=header_text, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        header_text: str | list[str] | None = None,
        do_convert_rgb: bool = True,
        input_data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs: Unpack[Pix2StructImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess images for Pix2Struct.
        """
        # Prepare images (converts to numpy arrays)
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format
        )

        # Handle VQA mode with header rendering
        is_vqa = kwargs.get("is_vqa", self.is_vqa)
        if is_vqa:
            if header_text is None:
                raise ValueError("A header text must be provided for VQA models.")

            font_bytes = kwargs.pop("font_bytes", None)
            font_path = kwargs.pop("font_path", None)

            if isinstance(header_text, str):
                header_text = [header_text] * len(images)

            # Render headers
            images = [
                self.render_header(image, header_text[i], font_bytes=font_bytes, font_path=font_path)
                for i, image in enumerate(images)
            ]

        return self._preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_normalize: bool,
        max_patches: int,
        patch_size: SizeDict,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images to extract flattened patches.
        """
        flattened_patches = []
        attention_masks = []

        for image in images:
            # Normalize image with per-image mean and std
            if do_normalize:
                image = self.normalize(image)

            patches = self.extract_flattened_patches(image=image, max_patches=max_patches, patch_size=patch_size)
            mask = (patches.sum(axis=-1) != 0).astype(np.float32)

            flattened_patches.append(patches)
            attention_masks.append(mask)

        # Stack if return_tensors is set
        if return_tensors:
            requires_backends(self, "torch")
            flattened_patches = torch.stack([torch.from_numpy(p) for p in flattened_patches], dim=0)
            attention_masks = torch.stack([torch.from_numpy(m) for m in attention_masks], dim=0)

        return BatchFeature(
            data={"flattened_patches": flattened_patches, "attention_mask": attention_masks},
            tensor_type=return_tensors,
        )


__all__ = ["Pix2StructImageProcessorPil"]
