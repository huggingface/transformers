# coding=utf-8
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
"""Fast Image processor class for Kosmos2_5."""

import math
from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import ChannelDimension, ImageInput, get_image_size
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, is_torch_available


if is_torch_available():
    import torch


# Similar to transformers.models.pix2struct.image_processing_pix2struct.torch_extract_patches but dealing with a batch of images directly.
def torch_extract_patches(image_tensor, patch_height, patch_width):
    """
    Utiliy function to extract patches from a given tensor representing a batch of images. Returns a tensor of shape
    (batch_size, `rows`, `columns`, `num_channels` x `patch_height` x `patch_width`).

    Args:
        image_tensor (torch.Tensor):
            The image tensor to extract patches from.
        patch_height (int):
            The height of the patches to extract.
        patch_width (int):
            The width of the patches to extract.
    """
    image_tensor = image_tensor
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(0),
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
    )
    return patches


class Kosmos2_5FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
        The patch size to use for the image. According to Kosmos2_5 paper and code, the patch size is 16x16.
    max_patches (`int`, *optional*, defaults to 4096):
        The maximum number of patches to extract from the image as per the
        [KOSMOS 2.5 paper](https://huggingface.co/papers/2309.11419).
    """

    patch_size: Optional[dict[str, int]]
    max_patches: Optional[int]


@auto_docstring
class Kosmos2_5ImageProcessorFast(BaseImageProcessorFast):
    # To be checked against the slow image processor
    # None values left after checking can be removed
    do_normalize = True
    do_convert_rgb = True
    patch_size = {"height": 16, "width": 16}
    max_patches = 4096
    rescale_factor = None
    valid_kwargs = Kosmos2_5FastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Kosmos2_5FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Kosmos2_5FastImageProcessorKwargs]) -> BatchFeature:
        r"""
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 16, "width": 16}`):
            The patch size to use for the image. According to Kosmos2_5 paper and code, the patch size is 16x16.
        max_patches (`int`, *optional*, defaults to 4096):
            The maximum number of patches to extract from the image as per the
            [KOSMOS 2.5 paper](https://huggingface.co/papers/2309.11419).
        """
        # return super().preprocess(images, **kwargs)
        # TODO: revert once the issue is fixed: https://huggingface.slack.com/archives/C02TXKQQLE5/p1743411133979019
        return super().preprocess(images, image_mean=0.0, image_std=0.0, **kwargs)

    def normalize(
        self,
        image: "torch.Tensor",
        **kwargs,
    ) -> "torch.Tensor":
        """
        Normalize an image. image = (image - image_mean) / image_std.

        The image std is to mimic the tensorflow implementation of the `per_image_standardization`:
        https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

        Args:
            image (`torch.Tensor`):
                Image to normalize.
        """
        # Q: should we keep this?
        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32)

        # take mean across the whole `image` except the batch dim (= 0).
        dim = list(range(1, image.ndim))
        mean = torch.mean(image, dim=dim)
        std = torch.std(image, dim=dim)
        # num_elements in a single image
        num_elements = torch.tensor(torch.numel(image[0]))
        adjusted_stddev = torch.max(std, 1.0 / torch.sqrt(num_elements))

        # change `image` from [batch_size, n_channels, width, height] to [n_channels, batch_size, width, height]
        image = torch.transpose(image, 0, 1)

        # 'torchvision.transforms.Normalize` works on the usual channel dimension (dim=1) which is the batch
        # dimension before we use `transpose`.
        image = super().normalize(
            image,
            mean=mean,
            std=adjusted_stddev,
            **kwargs,
        )
        # back to [batch_size, n_channels, width, height]
        normalized_image = torch.transpose(image, 0, 1)

        return normalized_image

    def extract_flattened_patches(
        self,
        image: "torch.Tensor",
        max_patches: int,
        patch_size: dict,
        # TODO: correct this return type, and the docstring
    ) -> "torch.Tensor":
        """
        Extract flattened patches from an image.

        Args:
            image (`np.ndarray`):
                Image to extract flattened patches from.
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`dict`):
                Dictionary containing the patch height and width.

        Returns:
            result (`np.ndarray`):
                A sequence of `max_patches` flattened patches.
        """
        patch_height, patch_width = patch_size["height"], patch_size["width"]
        image_height, image_width = get_image_size(image, ChannelDimension.FIRST)

        # maximize scale s.t.
        scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        image = torch.nn.functional.interpolate(
            image,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # [batch_size, rows, columns, patch_height * patch_width * image_channels]
        patches = torch_extract_patches(image, patch_height, patch_width)

        patches_shape = patches.shape
        batch_size = patches_shape[0]
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # [batch_size, rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([batch_size, rows * columns, depth])

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
        # [batch_size, rows * columns, 1]
        row_ids = row_ids.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.float32)
        col_ids = col_ids.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.float32)

        # [rows * columns, 2 + patch_height * patch_width * image_channels]
        result = torch.cat([row_ids, col_ids, patches], -1)

        # [batch_size, max_patches, 2 + patch_height * patch_width * image_channels]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        return result, resized_width, resized_height, rows, columns

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_normalize: bool,
        max_patches: int,
        patch_size: dict[str, int],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Q: should we have this?
        if kwargs.get("data_format") is not None:
            raise ValueError("data_format is not an accepted input as the outputs are ")

        width, height, rows, cols, attention_masks = [], [], [], [], []
        obj_idx_to_new_index_map = {}
        current_index = -1

        # Group images by size for batched resizing
        processed_image_patches_grouped = {}
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        for shape, stacked_images in grouped_images.items():
            # TODO: if it's possible to do in batch mode
            if do_normalize:
                stacked_images = self.normalize(stacked_images, **kwargs)

            # TODO: we need this to be in batch from
            # convert to torch tensor and permute
            patches, resized_width, resized_height, n_rows, n_columns = self.extract_flattened_patches(
                image=stacked_images,
                max_patches=max_patches,
                patch_size=patch_size,
            )
            n_of_stacked_images = stacked_images.size()[0]
            width.extend([resized_width] * n_of_stacked_images)
            height.extend([resized_height] * n_of_stacked_images)
            rows.extend([n_rows] * n_of_stacked_images)
            cols.extend([n_columns] * n_of_stacked_images)
            # create attention mask in numpy
            attention_masks.extend(list((patches.sum(axis=-1) != 0).to(dtype=torch.float32)))
            processed_image_patches_grouped[shape] = list(patches)
            for x in processed_image_patches_grouped[shape]:
                current_index += 1
                obj_idx_to_new_index_map[id(x)] = current_index

        processed_images = reorder_images(processed_image_patches_grouped, grouped_images_index)
        orig_idx_to_new_idx_map = {
            orig_idx: obj_idx_to_new_index_map[id(image)] for orig_idx, image in enumerate(processed_images)
        }

        flattened_patches = processed_images
        width = [width[orig_idx_to_new_idx_map[orig_idx]] for orig_idx in orig_idx_to_new_idx_map]
        height = [height[orig_idx_to_new_idx_map[orig_idx]] for orig_idx in orig_idx_to_new_idx_map]
        rows = [rows[orig_idx_to_new_idx_map[orig_idx]] for orig_idx in orig_idx_to_new_idx_map]
        cols = [cols[orig_idx_to_new_idx_map[orig_idx]] for orig_idx in orig_idx_to_new_idx_map]

        encoded_outputs = BatchFeature(
            data={
                "flattened_patches": torch.stack(flattened_patches, dim=0) if return_tensors else flattened_patches,
                "attention_mask": torch.stack(attention_masks, dim=0) if return_tensors else attention_masks,
                "width": width,
                "height": height,
                "rows": rows,
                "cols": cols,
            },
            tensor_type=return_tensors,
        )

        return encoded_outputs


__all__ = ["Kosmos2_5ImageProcessorFast"]
