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
"""Fast Image processor class for Pix2Struct."""

from typing import Optional, Union

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ChannelDimension, ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from .image_processing_pix2struct import Pix2StructImageProcessorKwargs, render_text


# Disable as it causes issues with torch.compile
@torch.compiler.disable
def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Extract patches from image tensor. Returns tensor of shape (batch, rows, columns, patch_height*patch_width*channels).

    Args:
        image_tensor (`torch.Tensor`):
            Image tensor of shape (batch, channels, height, width).
        patch_height (`int`):
            Height of patches to extract.
        patch_width (`int`):
            Width of patches to extract.
    """
    batch_size, channels, height, width = image_tensor.shape
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(batch_size, channels, patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        batch_size, height // patch_height, width // patch_width, channels * patch_height * patch_width
    )
    return patches


@auto_docstring
class Pix2StructImageProcessorFast(BaseImageProcessorFast):
    rescale_factor = None
    do_normalize = True
    do_convert_rgb = True
    patch_size = {"height": 16, "width": 16}
    max_patches = 2048
    is_vqa = False
    valid_kwargs = Pix2StructImageProcessorKwargs
    model_input_names = ["flattened_patches", "attention_mask"]

    def _further_process_kwargs(
        self,
        patch_size: Optional[dict[str, int]] = None,
        **kwargs,
    ) -> dict:
        """
        Process custom Pix2Struct kwargs, specifically converting patch_size to SizeDict.
        """
        # Call super to handle standard kwargs processing (like converting patch_size to SizeDict)
        kwargs = super()._further_process_kwargs(**kwargs)
        kwargs["patch_size"] = SizeDict(**get_size_dict(size=patch_size, param_name="patch_size"))

        return kwargs

    def _validate_preprocess_kwargs(self, **kwargs):
        """
        Skip standard validation as Pix2Struct uses custom preprocessing.
        """
        # Pix2Struct doesn't use standard resize/rescale/normalize parameters
        # so we skip the default validation
        pass

    def render_header(
        self,
        image: torch.Tensor,
        header: str,
        font_bytes: Optional[bytes] = None,
        font_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Render header text on image using torch tensors.

        Args:
            image (`torch.Tensor`):
                Image tensor in channel-first format (C, H, W).
            header (`str`):
                Header text to render.
            font_bytes (`bytes`, *optional*):
                Font bytes to use for rendering.
            font_path (`str`, *optional*):
                Path to font file to use for rendering.

        Returns:
            `torch.Tensor`: Image with header in channel-first format (C, H, W).
        """
        device = image.device
        dtype = image.dtype

        # Convert tensor to PIL (channel-first to channel-last for PIL)
        if image.dtype == torch.uint8:
            image_pil = F.to_pil_image(image)
        else:
            # If float, convert to uint8 first
            image_uint8 = (image * 255).clamp(0, 255).to(torch.uint8)
            image_pil = F.to_pil_image(image_uint8)

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

        # Convert back to tensor (channel-first)
        result = F.pil_to_tensor(new_image).to(device)

        # Convert back to original dtype if needed
        if dtype != torch.uint8:
            result = result.float() / 255.0

        return result

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize batched images using per-image mean and standard deviation.

        Args:
            images (`torch.Tensor`):
                Batched float image tensor of shape (B, C, H, W).

        Returns:
            `torch.Tensor`: Normalized images of shape (B, C, H, W).
        """
        # Compute mean and std per image along spatial and channel dimensions
        mean = images.mean(dim=(1, 2, 3), keepdim=True)  # Shape: (B, 1, 1, 1)
        std = images.std(dim=(1, 2, 3), keepdim=True)  # Shape: (B, 1, 1, 1)

        num_elements_per_image = images.shape[1] * images.shape[2] * images.shape[3]
        min_std = 1.0 / num_elements_per_image**0.5
        adjusted_stddev = torch.maximum(std, torch.tensor(min_std, device=std.device))

        return (images - mean) / adjusted_stddev

    def extract_flattened_patches(
        self,
        images: torch.Tensor,
        max_patches: int,
        patch_size: SizeDict,
    ) -> torch.Tensor:
        """
        Extract flattened patches from a batch of images.

        Args:
            images (`torch.Tensor`):
                Batched images tensor of shape (batch, channels, height, width).
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`dict[str, int]`):
                Dictionary containing patch height and width.

        Returns:
            `torch.Tensor`: Batched flattened patches with row/column IDs of shape (batch, max_patches, patch_dim).
        """
        patch_height, patch_width = patch_size.height, patch_size.width
        batch_size, channels, image_height, image_width = images.shape

        # Calculate scale to maximize patches while respecting max_patches
        scale = (max_patches * (patch_height / image_height) * (patch_width / image_width)) ** 0.5
        num_feasible_rows = max(min(int(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(int(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        # Resize images (batched) using parent class method
        resize_size = SizeDict(height=resized_height, width=resized_width)
        images = self.resize(
            image=images, size=resize_size, interpolation=F.InterpolationMode.BILINEAR, antialias=True
        )

        # Extract patches: [batch, rows, columns, patch_height * patch_width * channels]
        patches = torch_extract_patches(images, patch_height, patch_width)

        batch_size, rows, columns, depth = patches.shape

        # Reshape to [batch, rows * columns, depth]
        patches = patches.reshape(batch_size, rows * columns, depth)

        # Create row and column IDs
        row_ids = (
            torch.arange(rows, device=images.device).reshape(rows, 1).repeat(1, columns).reshape(1, rows * columns, 1)
        )
        col_ids = (
            torch.arange(columns, device=images.device)
            .reshape(1, columns)
            .repeat(rows, 1)
            .reshape(1, rows * columns, 1)
        )

        # Expand to batch size
        row_ids = row_ids.expand(batch_size, -1, -1)
        col_ids = col_ids.expand(batch_size, -1, -1)

        # Offset by 1 so IDs don't contain zeros (which represent padding)
        row_ids = (row_ids + 1).float()
        col_ids = (col_ids + 1).float()

        # Concatenate row_ids, col_ids, and patches: [batch, rows * columns, 2 + depth]
        result = torch.cat([row_ids, col_ids, patches], dim=-1)

        # Pad to max_patches: [batch, max_patches, 2 + depth]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        return result

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        header_text: Optional[Union[str, list[str]]] = None,
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
        header_text: Optional[Union[str, list[str]]] = None,
        do_convert_rgb: bool = True,
        input_data_format: ChannelDimension = ChannelDimension.FIRST,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Unpack[Pix2StructImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess images for Pix2Struct.
        """
        # Prepare images (converts to torch tensors)
        images = self._prepare_image_like_inputs(
            images=images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
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

            # Render headers using torch-native method
            images = [
                self.render_header(image, header_text[i], font_bytes=font_bytes, font_path=font_path)
                for i, image in enumerate(images)
            ]

        return self._preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_normalize: bool,
        max_patches: int,
        patch_size: SizeDict,
        return_tensors: Optional[Union[str, TensorType]],
        disable_grouping: bool,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images to extract flattened patches.
        """
        # Group images by shape first for efficient batch processing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

        flattened_patches_grouped = {}
        attention_masks_grouped = {}

        for shape, stacked_images in grouped_images.items():
            # Convert to float if needed (for resize and other operations)
            if stacked_images.dtype == torch.uint8:
                stacked_images = stacked_images.float()

            # Normalize batched images with per-image mean and std
            if do_normalize:
                stacked_images = self.normalize(stacked_images)

            patches = self.extract_flattened_patches(
                images=stacked_images, max_patches=max_patches, patch_size=patch_size
            )
            masks = (patches.sum(dim=-1) != 0).float()

            flattened_patches_grouped[shape] = patches
            attention_masks_grouped[shape] = masks

        flattened_patches = reorder_images(flattened_patches_grouped, grouped_images_index)
        attention_masks = reorder_images(attention_masks_grouped, grouped_images_index)

        # Stack if return_tensors is set
        if return_tensors:
            flattened_patches = torch.stack(flattened_patches, dim=0)
            attention_masks = torch.stack(attention_masks, dim=0)

        return BatchFeature(
            data={"flattened_patches": flattened_patches, "attention_mask": attention_masks},
            tensor_type=return_tensors,
        )


__all__ = ["Pix2StructImageProcessorFast"]
