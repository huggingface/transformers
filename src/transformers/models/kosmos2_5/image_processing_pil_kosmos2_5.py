# Copyright 2025 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Kosmos2_5."""

import math

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_utils import ChannelDimension, ImageInput, SizeDict, get_image_size
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available, requires_backends
from ...utils.import_utils import requires


if is_torch_available():
    import torch


# Adapted from transformers.models.kosmos2_5.image_processing_kosmos2_5.Kosmos2_5ImageProcessorKwargs
class Kosmos2_5ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
        The patch size to use for the image. According to Kosmos2_5 paper and code, the patch size is 16x16.
    max_patches (`int`, *optional*, defaults to 4096):
        The maximum number of patches to extract from the image as per the
        [KOSMOS 2.5 paper](https://huggingface.co/papers/2309.11419).
    """

    patch_size: SizeDict | None
    max_patches: int


# Adapted from transformers.models.kosmos2_5.image_processing_kosmos2_5.torch_extract_patches
# Similar to transformers.models.pix2struct.image_processing_pix2struct.torch_extract_patches but dealing with a batch of images directly.
def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utility function to extract patches from a given tensor representing a batch of images. Returns a tensor of shape
    (batch_size, `rows`, `columns`, `num_channels` x `patch_height` x `patch_width`).

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(0),
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches


@auto_docstring
@requires(backends=("torch",))
class Kosmos2_5ImageProcessorPil(PilBackend):
    do_normalize = True
    do_convert_rgb = True
    patch_size = {"height": 16, "width": 16}
    max_patches = 4096
    rescale_factor = None
    valid_kwargs = Kosmos2_5ImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Kosmos2_5ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Kosmos2_5ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def normalize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Normalize an image using per-image mean and standard deviation.

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

    def extract_flattened_patches(
        self, image: np.ndarray, max_patches: int, patch_size: SizeDict
    ) -> tuple[np.ndarray, int, int, int, int]:
        """
        Extract flattened patches from an image. Uses torch for patch extraction.

        Args:
            image (`np.ndarray`):
                Image array of shape (channels, height, width).
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`SizeDict`):
                Dictionary containing the patch height and width.

        Returns:
            tuple: A tuple containing:
                - result (`np.ndarray`): A sequence of `max_patches` flattened patches.
                - resized_width (`int`): Width after resizing.
                - resized_height (`int`): Height after resizing.
                - rows (`int`): Number of patch rows.
                - columns (`int`): Number of patch columns.
        """
        requires_backends(self, ["torch"])

        # Convert to torch tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension

        patch_height, patch_width = patch_size.height, patch_size.width
        image_height, image_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        # maximize scale s.t.
        scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        image_tensor = torch.nn.functional.interpolate(
            image_tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False, antialias=True
        )

        # [1, rows, columns, patch_height * patch_width * image_channels]
        patches = torch_extract_patches(image_tensor, patch_height, patch_width)

        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

        # [rows * columns, 1]
        row_ids = (
            torch.arange(rows, device=patches.device)
            .reshape([rows, 1])
            .repeat(1, columns)
            .reshape([rows * columns, 1])
        )
        col_ids = (
            torch.arange(columns, device=patches.device)
            .reshape([1, columns])
            .repeat(rows, 1)
            .reshape([rows * columns, 1])
        )

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

        result_np = result.cpu().numpy()

        return result_np, resized_width, resized_height, rows, columns

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_normalize: bool,
        max_patches: int,
        patch_size: SizeDict,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if kwargs.get("data_format") is not None:
            raise ValueError("data_format is not an accepted input as the outputs are ")

        flattened_patches, width, height, rows, cols, attention_masks = [], [], [], [], [], []

        # Process images one by one
        for image in images:
            if do_normalize:
                image = self.normalize(image, **kwargs)

            patches, resized_width, resized_height, n_rows, n_columns = self.extract_flattened_patches(
                image=image, max_patches=max_patches, patch_size=patch_size
            )
            flattened_patches.append(patches)
            width.append(resized_width)
            height.append(resized_height)
            rows.append(n_rows)
            cols.append(n_columns)
            # create attention mask
            attention_masks.append((patches.sum(axis=-1) != 0).astype(np.float32))

        encoded_outputs = BatchFeature(
            data={
                "flattened_patches": flattened_patches,
                "attention_mask": attention_masks,
                "width": width,
                "height": height,
                "rows": rows,
                "cols": cols,
            },
            tensor_type=return_tensors,
        )

        return encoded_outputs

    def _validate_preprocess_kwargs(self, **kwargs):
        """
        Skip standard validation as Kosmos2_5 uses custom preprocessing with per-image normalization.
        """
        pass

    def _standardize_kwargs(self, patch_size: dict[str, int] | SizeDict | None = None, **kwargs) -> dict:
        """
        Process Kosmos2_5-specific kwargs before validation.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        if patch_size is not None and not isinstance(patch_size, SizeDict):
            patch_size = SizeDict(**get_size_dict(patch_size, param_name="patch_size"))
        kwargs["patch_size"] = patch_size
        return kwargs


__all__ = ["Kosmos2_5ImageProcessorPil"]
