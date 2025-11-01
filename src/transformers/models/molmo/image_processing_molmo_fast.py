# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
)
from ...utils import TensorType, auto_docstring, logging


logger = logging.get_logger(__name__)


def get_resize_output_image_size(
    image: torch.Tensor,
    size: Union[int, tuple[int, int], list[int], tuple[int]],
) -> dict:
    original_height, original_width = get_image_size(image, ChannelDimension.FIRST)

    scale_y = size["height"] / original_height
    scale_x = size["width"] / original_width
    scale = min(scale_x, scale_y)

    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    return {"height": new_height, "width": new_width}


def pad_to_bounding_box(
    image: torch.Tensor, offset_height: int, offset_width: int, target_height: int, target_width: int, value: int = 0
) -> torch.Tensor:
    height, width = image.shape[:2]
    top_padding = offset_height
    bottom_padding = max(0, target_height - height - offset_height)
    left_padding = offset_width
    right_padding = max(0, target_width - width - offset_width)
    image = image.permute(2, 0, 1)
    padding = [left_padding, top_padding, right_padding, bottom_padding]
    padded_image = F.pad(image, padding=padding, padding_mode="constant", fill=value)
    padded_image = padded_image.permute(1, 2, 0)
    return padded_image


@auto_docstring
class MolmoImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["pixel_values", "input_ids", "image_token_indices", "image_masks"]
    resample = PILImageResampling.BILINEAR
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 336, "width": 336}
    do_resize = True
    do_pad = True
    padding_value = 1.0
    padding_mode = "constant"
    do_split_into_crops = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True

    def __init__(
        self,
        max_num_crops: int = 12,
        overlap_margins: tuple[int, int] = (4, 4),
        size: Optional[dict[str, int]] = None,
        tokens_per_image_width: int = 12,
        tokens_per_image_height: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_pad: Optional[bool] = True,
        padding_value: float = 1.0,
        padding_mode: str = "constant",
        do_split_into_crops: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        image_patch_token: str = "<im_patch>",
        image_column_token: str = "<im_col>",
        image_start_token: str = "<im_start>",
        image_end_token: str = "<im_end>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 336, "width": 336}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_pad = do_pad
        self.padding_value = padding_value
        self.padding_mode = padding_mode
        self.do_split_into_crops = do_split_into_crops
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.max_num_crops = max_num_crops
        self.overlap_margins = (
            overlap_margins if isinstance(overlap_margins, (tuple, list)) else [overlap_margins, overlap_margins]
        )
        self.tokens_per_image_width = tokens_per_image_width
        self.tokens_per_image_height = tokens_per_image_height
        self.image_patch_size = image_patch_size
        self.image_padding_mask = image_padding_mask
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.image_patch_token = image_patch_token
        self.image_column_token = image_column_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token

        self.tokens_per_image = tokens_per_image_height * tokens_per_image_width
        self.patches_per_image_width = size["width"] // image_patch_size
        self.patches_per_image_height = size["height"] // image_patch_size
        self.total_margin_pixels = image_patch_size * (self.overlap_margins[1] + self.overlap_margins[0])
        self.crop_patches = self.size["width"] // self.image_patch_size
        self.crop_window_patches = self.crop_patches - (self.overlap_margins[1] + self.overlap_margins[0])
        self.crop_window_size = self.crop_window_patches * self.image_patch_size
        self.crop_size = size["width"]

        if ((self.patches_per_image_height + 1) // 2 != self.tokens_per_image_height) or (
            (self.patches_per_image_width + 1) // 2 != self.tokens_per_image_width
        ):
            raise ValueError("Number of patches per crop does not fit number of tokens per image dimension.")

    def resize(
        self,
        image: torch.Tensor,
        size: dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> torch.Tensor:
        output_size = (size["height"], size["width"])
        if input_data_format == ChannelDimension.LAST:
            image = image.permute(2, 0, 1)
        resized_image = F.resize(image, size=output_size, interpolation=F.InterpolationMode.BILINEAR)
        if input_data_format == ChannelDimension.LAST:
            resized_image = resized_image.permute(1, 2, 0)
        return resized_image

    def pad(
        self,
        image: torch.Tensor,
        size: dict[str, int],
        mode: str = "constant",
        constant_values: float = 1.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "height" not in size or "width" not in size:
            raise ValueError("Size must contain 'height' and 'width'.")

        current_height, current_width = get_image_size(image, input_data_format)

        new_size = get_resize_output_image_size(image, size)
        padding_height = size["height"] - new_size["height"]
        padding_width = size["width"] - new_size["width"]
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        if input_data_format == ChannelDimension.LAST:
            image = image.permute(2, 0, 1)

        padding = [padding_left, padding_top, padding_right, padding_bottom]
        padded_image = F.pad(image, padding=padding, fill=constant_values, padding_mode=mode)

        if input_data_format == ChannelDimension.FIRST:
            image_to_pad = image[0, :, :]
        else:
            image_to_pad = image[:, :, 0]

        image_mask = torch.ones_like(image_to_pad, dtype=torch.bool, device=image.device)
        image_mask = F.pad(image_mask.unsqueeze(0), padding=padding, fill=0).squeeze(0)

        if input_data_format == ChannelDimension.LAST:
            padded_image = padded_image.permute(1, 2, 0)

        return padded_image, image_mask

    def find_best_crop_grid_for_image_size(self, image: torch.Tensor):
        original_size = torch.tensor(
            [image.shape[-2] - self.total_margin_pixels, image.shape[-1] - self.total_margin_pixels],
            dtype=torch.float32,
            device=image.device,
        )
        crop_grid = [(i, j) for i in range(1, self.max_num_crops + 1) for j in range(1, (self.max_num_crops // i) + 1)]
        crop_grid.sort(key=lambda x: (x[0] * x[1], x[0]))
        candidate_crop_grid = torch.tensor(crop_grid, dtype=torch.int32, device=image.device)
        candidate_resolutions = candidate_crop_grid.float() * self.crop_window_size
        required_scale_step = candidate_resolutions / original_size
        required_scale, _ = torch.min(required_scale_step, dim=-1, keepdim=True)
        if torch.all(required_scale < 1):
            selected_index = torch.argmax(required_scale)
        else:
            required_scale = torch.where(required_scale < 1.0, float("inf"), required_scale)
            selected_index = torch.argmin(required_scale)
        return candidate_crop_grid[selected_index]

    def reshape_into_patches(self, global_image, input_data_format):
        if input_data_format == ChannelDimension.FIRST:
            global_image = global_image.permute(1, 2, 0)
        channels = global_image.shape[-1]
        global_image = global_image.reshape(
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
            channels,
        )
        global_image = global_image.permute(0, 2, 1, 3, 4)
        global_image = global_image.reshape(
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size * channels,
        )
        return global_image

    def split_image_into_crops(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor,
        crop_grid: tuple[int, int],
        input_data_format,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_data_format == ChannelDimension.FIRST:
            image = image.permute(1, 2, 0)
        crops = []
        cropped_masks = []
        patch_orderings = []
        patch_index = 0
        for row in range(crop_grid[0]):
            crop_y_start = row * self.crop_window_size

            current_crop_height = self.patches_per_image_height - (self.overlap_margins[1] + self.overlap_margins[0])
            if row == 0:
                current_crop_height += self.overlap_margins[0]
            if row == (crop_grid[0] - 1):
                current_crop_height += self.overlap_margins[1]
            pooled_height = (current_crop_height + 1) // 2
            crop_y_offset = self.overlap_margins[0] // 2 if row > 0 else 0
            for column in range(crop_grid[1]):
                crop_x_start = column * self.crop_window_size

                current_crop_width = self.patches_per_image_width - (self.overlap_margins[1] + self.overlap_margins[0])
                if column == 0:
                    current_crop_width += self.overlap_margins[0]
                if column == (crop_grid[1] - 1):
                    current_crop_width += self.overlap_margins[1]

                pooled_width = (current_crop_width + 1) // 2

                crop_x_offset = self.overlap_margins[0] // 2 if column > 0 else 0
                reshaped_image = torch.arange(
                    patch_index,
                    patch_index + pooled_height * pooled_width,
                    dtype=torch.int32,
                    device=image.device,
                ).reshape(pooled_height, pooled_width, 1)
                patch_orderings.append(
                    pad_to_bounding_box(
                        reshaped_image,
                        offset_height=crop_y_offset,
                        offset_width=crop_x_offset,
                        target_height=self.tokens_per_image_height,
                        target_width=self.tokens_per_image_width,
                        value=-1,
                    )[:, :, 0]
                )

                crop = image[
                    crop_y_start : crop_y_start + self.crop_size,
                    crop_x_start : crop_x_start + self.crop_size,
                ]
                crops.append(crop)

                cropped_mask = image_mask[
                    crop_y_start : crop_y_start + self.crop_size,
                    crop_x_start : crop_x_start + self.crop_size,
                ]
                cropped_masks.append(cropped_mask)

                patch_index += pooled_height * pooled_width

        crops = torch.stack(crops)
        patch_orderings = torch.stack(patch_orderings)
        cropped_masks = torch.stack(cropped_masks)

        leading_crops_dim, h, w, channels = crops.shape
        crops = crops.view(
            leading_crops_dim,
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
            channels,
        )
        crops = crops.permute(0, 1, 3, 2, 4, 5)
        crops = crops.contiguous()
        crops = crops.view(
            leading_crops_dim,
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size * channels,
        )

        leading_mask_dim = cropped_masks.shape[0]
        cropped_masks = cropped_masks.view(
            leading_mask_dim,
            self.patches_per_image_height,
            self.image_patch_size,
            self.patches_per_image_width,
            self.image_patch_size,
        )
        cropped_masks = cropped_masks.permute(0, 1, 3, 2, 4)
        cropped_masks = cropped_masks.contiguous()
        cropped_masks = cropped_masks.view(
            leading_mask_dim,
            self.patches_per_image_width * self.patches_per_image_height,
            self.image_patch_size * self.image_patch_size,
        )

        cropped_masks = cropped_masks.float().mean(dim=-1)
        cropped_masks = torch.nn.functional.pad(cropped_masks, (0, 0, 0, 1), value=-1)
        patch_orderings = patch_orderings.view(-1)
        return crops, patch_orderings, cropped_masks

    def transpose_patch_orderings(self, crop_grid, patch_orderings):
        patch_ordering_left_right = patch_orderings.reshape(
            crop_grid[0], crop_grid[1], self.tokens_per_image_height, self.tokens_per_image_width
        )
        patch_ordering_left_right = patch_ordering_left_right.permute(0, 2, 1, 3)
        patch_ordering_left_right = patch_ordering_left_right.reshape(-1)
        mask = patch_orderings >= 0
        patch_orderings[mask] = patch_ordering_left_right[patch_ordering_left_right >= 0]
        return patch_orderings

    def _prepare_crop_grids(self, data):
        crop_grids = data["crop_grids"]
        data["crop_grids"] = torch.stack(crop_grids)

    def _pad_patch_orderings(self, data, device):
        patch_orderings = data["patch_orderings"]
        batch_size = len(patch_orderings)
        max_length = max(ordering.shape[0] for ordering in patch_orderings)
        fill_value = -2
        batched_patch_orderings = torch.full(
            (batch_size, max_length), fill_value=fill_value, dtype=patch_orderings[0].dtype, device=device
        )

        for idx, ordering in enumerate(patch_orderings):
            length = ordering.shape[0]
            batched_patch_orderings[idx, :length] = ordering

        data["patch_orderings"] = batched_patch_orderings

    def _pad_for_batching(self, data: dict, device: str):
        crops = data["pixel_values"]
        max_num_crops = max(image.shape[0] for image in crops)
        batch_size = len(crops)
        crop_shape = crops[0].shape[1:]

        batched_crops = torch.zeros((batch_size, max_num_crops, *crop_shape), dtype=crops[0].dtype, device=device)
        for idx, image in enumerate(crops):
            num_crops = image.shape[0]
            batched_crops[idx, :num_crops, ...] = image

        data["pixel_values"] = batched_crops

        image_masks = data["image_masks"]
        mask_shape = image_masks[0].shape[1:]
        batched_image_masks = torch.full(
            (batch_size, max_num_crops, *mask_shape), fill_value=-1, dtype=image_masks[0].dtype, device=device
        )
        for idx, mask in enumerate(image_masks):
            num_crops = mask.shape[0]
            batched_image_masks[idx, :num_crops, ...] = mask

        data["image_masks"] = batched_image_masks
        self._pad_patch_orderings(data, device=device)
        self._prepare_crop_grids(data)
        return data

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_pad: Optional[bool] = None,
        do_split_into_crops: Optional[bool] = None,
        padding_value: Optional[float] = None,
        padding_mode: Optional[str] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_split_into_crops = do_split_into_crops if do_split_into_crops is not None else self.do_split_into_crops
        padding_value = padding_value if padding_value is not None else self.padding_value
        padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images_tensor = self._prepare_image_like_inputs(
            images=images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=None,
        )

        all_images = []
        all_crop_grids = []
        all_cropped_masks = []
        all_patch_orderings = []

        for image in images_tensor:
            if input_data_format is None:
                from ...image_utils import infer_channel_dimension_format

                input_data_format = infer_channel_dimension_format(image)

            crop_grid = self.find_best_crop_grid_for_image_size(image)

            if do_resize:
                global_image_size = get_resize_output_image_size(image, size)
                global_image = self.resize(
                    image=image, size=global_image_size, resample=resample, input_data_format=input_data_format
                )

                new_crop_size = {}
                new_crop_size["height"] = crop_grid[0] * self.crop_window_size + self.total_margin_pixels
                new_crop_size["width"] = crop_grid[1] * self.crop_window_size + self.total_margin_pixels
                crop_output_size = get_resize_output_image_size(image, size=new_crop_size)
                image = self.resize(
                    image=image, size=crop_output_size, resample=resample, input_data_format=input_data_format
                )

            if do_pad:
                image, image_mask = self.pad(
                    image=image, size=new_crop_size, input_data_format=input_data_format, constant_values=0
                )
                global_image, _ = self.pad(
                    image=global_image, size=size, input_data_format=input_data_format, constant_values=0
                )

            if do_rescale and do_normalize:
                new_mean = torch.tensor(image_mean, device=global_image.device) * (1.0 / rescale_factor)
                new_std = torch.tensor(image_std, device=global_image.device) * (1.0 / rescale_factor)
                if input_data_format == ChannelDimension.LAST:
                    image = image.permute(2, 0, 1)
                    global_image = global_image.permute(2, 0, 1)
                image = F.normalize(image.to(dtype=torch.float32), new_mean, new_std)
                global_image = F.normalize(global_image.to(dtype=torch.float32), new_mean, new_std)
                if input_data_format == ChannelDimension.LAST:
                    image = image.permute(1, 2, 0)
                    global_image = global_image.permute(1, 2, 0)

            if do_split_into_crops:
                crops, patch_orderings, cropped_masks = self.split_image_into_crops(
                    image=image, image_mask=image_mask, crop_grid=crop_grid, input_data_format=input_data_format
                )
                patch_orderings = self.transpose_patch_orderings(crop_grid, patch_orderings)

            global_image = self.reshape_into_patches(global_image, input_data_format=input_data_format)
            new_crops = torch.empty(
                (crops.shape[0] + 1, crops.shape[1], crops.shape[2]), device=crops.device, dtype=crops.dtype
            )
            new_crops[0] = global_image
            new_crops[1:] = crops
            crops = new_crops

            patch_orderings = torch.where(patch_orderings >= 0, patch_orderings + self.tokens_per_image, -1)
            prefix = torch.arange(0, self.tokens_per_image, device=global_image.device)
            new_patch_orderings = torch.empty(
                (patch_orderings.shape[0] + prefix.shape[0],),
                device=patch_orderings.device,
                dtype=patch_orderings.dtype,
            )
            new_patch_orderings[: prefix.shape[0]] = prefix
            new_patch_orderings[prefix.shape[0] :] = patch_orderings
            patch_orderings = new_patch_orderings

            all_images.append(crops)
            all_crop_grids.append(crop_grid)
            all_cropped_masks.append(cropped_masks)
            all_patch_orderings.append(patch_orderings)

        data = {
            "pixel_values": all_images,
            "crop_grids": all_crop_grids,
            "patch_orderings": all_patch_orderings,
            "image_masks": all_cropped_masks,
        }
        if do_pad:
            data = self._pad_for_batching(data, device=global_image.device)
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["MolmoImageProcessorFast"]
