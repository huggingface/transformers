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

"""
Processor class for Phi4Multimodal
"""

import math
from typing import List, Optional, Union

import torch
from torchvision.transforms import functional as F

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    Unpack,
    convert_to_rgb,
)
from ...image_utils import ImageInput, make_flat_list_of_images, valid_images
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class Phi4MultimodalFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    image_size: Optional[int]
    patch_size: Optional[int]
    dynamic_hd: Optional[int]


class Phi4MultimodalImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a Phi4Multimodal image processor.
    """

    image_size = 448
    patch_size = 14
    dynamic_hd = 36
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    valid_init_kwargs = Phi4MultimodalFastImageProcessorKwargs
    model_input_names = ["image_pixel_values", "image_sizes", "image_attention_mask"]

    def __init__(self, **kwargs: Unpack[Phi4MultimodalFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
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
                if area > 0.5 * self.image_size * self.image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, max_num=36, min_num=1):
        image_size = self.image_size
        patch_size = self.patch_size
        mask_size = image_size // patch_size
        orig_width, orig_height = image.size

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
            target_aspect_ratio = self.find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height)

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

    def preprocess(
        self,
        images: ImageInput,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_flat_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        images = [convert_to_rgb(image) for image in images]

        image_size = self.image_size
        patch_size = self.patch_size
        mask_size = image_size // patch_size
        imgs_and_masks = [self.dynamic_preprocess(image, max_num=self.dynamic_hd) for image in images]
        images, image_attention_masks = [x[0] for x in imgs_and_masks], [x[1] for x in imgs_and_masks]

        images = [F.to_tensor(image) for image in images]
        hd_images = [F.normalize(image, image_mean, image_std) for image in images]
        global_image = [
            torch.nn.functional.interpolate(
                image.unsqueeze(0).float(),
                size=(image_size, image_size),
                mode="bicubic",
            ).to(image.dtype)
            for image in hd_images
        ]

        shapes = [[image.size(1), image.size(2)] for image in hd_images]
        mask_shapes = [[mask.size(0), mask.size(1)] for mask in image_attention_masks]
        global_attention_mask = [torch.ones((1, mask_size, mask_size)) for _ in hd_images]

        hd_images_reshape = []
        for im, (h, w) in zip(hd_images, shapes):
            im = im.reshape(1, 3, h // image_size, image_size, w // image_size, image_size)
            im = im.permute(0, 2, 4, 1, 3, 5)
            im = im.reshape(-1, 3, image_size, image_size)
            hd_images_reshape.append(im.contiguous())

        attention_masks_reshape = []
        for mask, (h, w) in zip(image_attention_masks, mask_shapes):
            mask = mask.reshape(h // mask_size, mask_size, w // mask_size, mask_size)
            mask = mask.transpose(1, 2)
            mask = mask.reshape(-1, mask_size, mask_size)
            attention_masks_reshape.append(mask.contiguous())

        downsample_attention_masks = []
        for mask, (h, w) in zip(attention_masks_reshape, mask_shapes):
            mask = mask[:, 0::2, 0::2]
            mask = mask.reshape(
                h // mask_size, w // mask_size, mask_size // 2 + mask_size % 2, mask_size // 2 + mask_size % 2
            )
            mask = mask.transpose(1, 2)
            mask = mask.reshape(mask.size(0) * mask.size(1), mask.size(2) * mask.size(3))
            downsample_attention_masks.append(mask)

        num_img_tokens = [
            256 + 1 + int(mask.sum().item()) + int(mask[:, 0].sum().item()) + 16 for mask in downsample_attention_masks
        ]

        hd_images_reshape = [
            torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)
        ]
        hd_masks_reshape = [
            torch.cat([_global_mask] + [_mask], dim=0)
            for _global_mask, _mask in zip(global_attention_mask, attention_masks_reshape)
        ]
        max_crops = max([img.size(0) for img in hd_images_reshape])
        image_transformed = [self.pad_to_max_num_crops(im, max_crops) for im in hd_images_reshape]
        image_transformed = torch.stack(image_transformed, dim=0)
        mask_transformed = [self.pad_mask_to_max_num_crops(mask, max_crops) for mask in hd_masks_reshape]
        mask_transformed = torch.stack(mask_transformed, dim=0)

        returned_input_image_embeds = image_transformed
        returned_image_sizes = torch.tensor(shapes, dtype=torch.long)
        returned_image_attention_mask = mask_transformed
        returned_num_img_tokens = num_img_tokens

        data = {
            "image_pixel_values": returned_input_image_embeds,
            "image_sizes": returned_image_sizes,
            "image_attention_mask": returned_image_attention_mask,
            "num_img_tokens": returned_num_img_tokens,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Phi4MultimodalImageProcessorFast"]
