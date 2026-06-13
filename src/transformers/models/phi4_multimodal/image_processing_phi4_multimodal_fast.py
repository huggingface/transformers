# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    Unpack,
)
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
)


logger = logging.get_logger(__name__)


class Phi4MultimodalFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    patch_size (`int`, *optional*):
        The size of the patch.
    dynamic_hd (`int`, *optional*):
        The maximum number of crops per image.
    """

    patch_size: Optional[int]
    dynamic_hd: Optional[int]


@auto_docstring
class Phi4MultimodalImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    size = {"height": 448, "width": 448}
    patch_size = 14
    dynamic_hd = 36
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = Phi4MultimodalFastImageProcessorKwargs
    model_input_names = ["image_pixel_values", "image_sizes", "image_attention_mask"]

    def __init__(self, **kwargs: Unpack[Phi4MultimodalFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, image_size, patch_size, mask_size, max_num=36, min_num=1):
        orig_height, orig_width = image.shape[-2:]

        w_crop_num = math.ceil(orig_width / float(image_size))
        h_crop_num = math.ceil(orig_height / float(image_size))
        if w_crop_num * h_crop_num > max_num:
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            }
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)

        # Calculate the ratio
        ratio_width = target_width / orig_width
        ratio_height = target_height / orig_height
        if ratio_width < ratio_height:
            new_size = (target_width, int(orig_height * ratio_width))
            padding_width = 0
            padding_height = target_height - int(orig_height * ratio_width)
        else:
            new_size = (int(orig_width * ratio_height), target_height)
            padding_width = target_width - int(orig_width * ratio_height)
            padding_height = 0

        attention_mask = torch.ones((int(mask_size * target_aspect_ratio[1]), int(mask_size * target_aspect_ratio[0])))
        if padding_width >= patch_size:
            attention_mask[:, -math.floor(padding_width / patch_size) :] = 0
        if padding_height >= patch_size:
            attention_mask[-math.floor(padding_height / patch_size) :, :] = 0

        if min(new_size[1], target_height) < 10 or min(new_size[0], target_width) < 10:
            raise ValueError(f"the aspect ratio is very extreme {new_size}")

        image = F.resize(image, [new_size[1], new_size[0]])
        resized_img = F.pad(image, [0, 0, padding_width, padding_height], fill=[255, 255, 255])

        return resized_img, attention_mask

    def pad_to_max_num_crops(self, images, max_crops=5):
        """
        images: B x 3 x H x W, B<=max_crops
        """
        B, _, H, W = images.shape
        if B < max_crops:
            pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
            images = torch.cat([images, pad], dim=0)
        return images

    def pad_mask_to_max_num_crops(self, masks, max_crops=5):
        B, H, W = masks.shape
        if B < max_crops:
            pad = torch.ones(max_crops - B, H, W, dtype=masks.dtype, device=masks.device)
            masks = torch.cat([masks, pad], dim=0)
        return masks

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Phi4MultimodalFastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        patch_size: int,
        dynamic_hd: int,
        do_rescale: bool,
        rescale_factor: Optional[float],
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        if size.height != size.width:
            raise ValueError("Phi4MultimodalFastImageProcessor only supports square sizes.")
        mask_size = size.height // patch_size
        images_transformed = []
        masks_transformed = []
        images_tokens = []
        image_sizes = []
        for image in images:
            resized_image, attention_mask = self.dynamic_preprocess(
                image, size.height, patch_size, mask_size, max_num=dynamic_hd
            )
            processed_image = self.rescale_and_normalize(
                resized_image, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            global_image = self.resize(processed_image, size, interpolation=interpolation, antialias=False)
            height, width = processed_image.shape[-2:]
            mask_height, mask_width = attention_mask.shape[-2:]
            global_attention_mask = torch.ones((1, mask_size, mask_size))

            hd_image_reshape = processed_image.reshape(
                1, 3, height // size.height, size.height, width // size.width, size.width
            )
            hd_image_reshape = hd_image_reshape.permute(0, 2, 4, 1, 3, 5)
            hd_image_reshape = hd_image_reshape.reshape(-1, 3, size.height, size.width).contiguous()

            attention_mask_reshape = attention_mask.reshape(
                mask_height // mask_size, mask_size, mask_width // mask_size, mask_size
            )
            attention_mask_reshape = attention_mask_reshape.transpose(1, 2)
            attention_mask_reshape = attention_mask_reshape.reshape(-1, mask_size, mask_size).contiguous()

            downsample_attention_mask = attention_mask_reshape[:, 0::2, 0::2]
            downsample_attention_mask = downsample_attention_mask.reshape(
                mask_height // mask_size,
                mask_width // mask_size,
                mask_size // 2 + mask_size % 2,
                mask_size // 2 + mask_size % 2,
            )
            downsample_attention_mask = downsample_attention_mask.transpose(1, 2)
            downsample_attention_mask = downsample_attention_mask.reshape(
                downsample_attention_mask.size(0) * downsample_attention_mask.size(1),
                downsample_attention_mask.size(2) * downsample_attention_mask.size(3),
            )

            num_img_tokens = (
                256
                + 1
                + int(downsample_attention_mask.sum().item())
                + int(downsample_attention_mask[:, 0].sum().item())
                + 16
            )

            hd_image_reshape = torch.cat([global_image.unsqueeze(0), hd_image_reshape], dim=0)
            hd_attention_mask_reshape = torch.cat([global_attention_mask, attention_mask_reshape], dim=0)

            images_transformed.append(hd_image_reshape)
            masks_transformed.append(hd_attention_mask_reshape)
            images_tokens.append(num_img_tokens)
            image_sizes.append([height, width])
            max_crops = hd_image_reshape.size(0)
        max_crops = max([img.size(0) for img in images_transformed])
        images_transformed = [self.pad_to_max_num_crops(im, max_crops) for im in images_transformed]
        images_transformed = torch.stack(images_transformed, dim=0)
        masks_transformed = [self.pad_mask_to_max_num_crops(mask, max_crops) for mask in masks_transformed]
        masks_transformed = torch.stack(masks_transformed, dim=0)
        image_sizes = torch.tensor(image_sizes, dtype=torch.long)

        data = {
            "image_pixel_values": images_transformed,
            "image_sizes": image_sizes,
            "image_attention_mask": masks_transformed,
            "num_img_tokens": images_tokens,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Phi4MultimodalImageProcessorFast"]
