# coding=utf-8
# Copyright 2024 FIXME copyright?
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
"""Image processor class for Molmo"""

from typing import List, Optional, Union, Mapping

import numpy as np
import einops
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
)
from transformers.processing_utils import ImagesKwargs
from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import logging


logger = logging.get_logger(__name__)


def resize_and_pad(
    image,
    desired_output_size,
    resize_method="torch-bilinear",
    pad_value=0,
    normalize=True,
    image_mean=OPENAI_CLIP_MEAN,
    image_std=OPENAI_CLIP_STD,
):
    """Resize an image while padding to preserve uts aspect ratio."""
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Cast into float32 since the training code did this in float32 and it (very rarely) effects
    # the results after rounding.
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    if resize_method == "tensorflow":
        # This how the original training code did resizing, it can produce slightly different
        # results then using torch resize so we keep it just in case
        import tensorflow as tf
        image = tf.image.convert_image_dtype(tf.constant(image), dtype=tf.float32)
        image = tf.image.resize(
            image,
            [scaled_height, scaled_width],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=True,
        )
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = image.numpy()
    elif resize_method == "torch-bilinear":
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)  # resize in float32 to match the training code
        image = torchvision.transforms.Resize(
            [scaled_height, scaled_width], InterpolationMode.BILINEAR, antialias=True
        )(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0]
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    return image, image_mask


def select_tiling(h, w, patch_size, max_num_crops):
    """Divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = np.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i*j <= max_num_crops:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]
    required_scale_d = candidate_resolutions.astype(np.float32) / original_size
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = np.argmax(required_scale)
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


class MolmoImagesKwargs(ImagesKwargs, total=False):
    max_crops: Optional[int]
    overlap_margins: Optional[List[int]]
    base_image_input_size: Optional[List[int]]
    image_token_length_w: Optional[int]
    image_token_length_h: Optional[int]
    image_patch_size: Optional[int]
    image_padding_mask: Optional[bool]


class MolmoImageProcessor(BaseImageProcessor):
    """Preprocess images and multi-model inputs"""

    def __init__(
        self,
        max_crops: int = 12,
        overlap_margins: List[int] = (4, 4),
        base_image_input_size: List[int] = (336, 336),
        image_token_length_w: int = 12,
        image_token_length_h: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_crops = max_crops
        self.overlap_margins = overlap_margins
        self.base_image_input_size = base_image_input_size
        self.image_token_length_w = image_token_length_w
        self.image_token_length_h = image_token_length_h
        self.image_patch_size = image_patch_size
        self.image_padding_mask = image_padding_mask
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def _normalize(self, image):
        if self.do_normalize:
            image -= np.array(self.image_mean, dtype=np.float32)[None, None, :]
            image /= np.array(self.image_std, dtype=np.float32)[None, None, :]
        return image

    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[List[int]] = None,
        base_image_input_size: Optional[Union[int, List[int]]] = None,
        image_token_length_w: Optional[int] = None,
        image_token_length_h: Optional[int] = None,
        image_patch_size: Optional[int] = None,
    ):
        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        tokens_per_image = image_token_length_w * image_token_length_h
        image_base_patch_w = base_image_input_size[1] // base_image_input_d
        image_base_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        # Discard this many patches from the (left/top, right/bottom) of crops
        left_margin, right_margin = overlap_margins
        # left_margin, right_margin = 2, 2
        assert left_margin % 2 == 0  # Required for compatibility with 2x2 pooling
        total_margin_pixels = base_image_input_d*(right_margin + left_margin)  # pixels removed per dim
        crop_patches = base_image_input_size[0] // base_image_input_d  # patches per crop dim
        crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
        crop_window_size = crop_window_patches * base_image_input_d

        # Decide how to tile the image, to account for the overlap margins we compute the tiling
        # as if we had an image without the margins and were using a crop size without the margins
        tiling = select_tiling(
            original_image_h - total_margin_pixels,
            original_image_w - total_margin_pixels,
            crop_window_size,
            max_crops
        )
        src, img_mask = resize_and_pad(
            image,
            [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels]
        )
        src = self._normalize(src)

        # Now we have to split the image into crops, while keeping track of how each patch in the
        # each crop should be ordered in the global image, this require a lot of tricky booking
        n_crops = tiling[0] * tiling[1]
        patches_arr = []
        mask_arr = []
        patch_ordering_arr = []

        # We assume 2x2 pooling, but can allow padding the right/bottom with extra
        # patches if the number of patches per side is not even
        assert (crop_patches+1)//2 == image_token_length_h
        assert (crop_patches+1)//2 == image_token_length_w
        on = 0
        on_patch = 0
        for i in range(tiling[0]):
            y0 = i*crop_window_size
            if i == 0:
                crop_y0 = 0
            else:
                crop_y0 = left_margin // 2

            crop_h = image_base_patch_h - (right_margin + left_margin)
            if i == 0:
                crop_h += left_margin
            if i == (tiling[0]-1):
                crop_h += right_margin
            for j in range(tiling[1]):
                x0 = j*crop_window_size
                if j == 0:
                    crop_x0 = 0
                else:
                    crop_x0 = left_margin // 2

                crop_w = image_base_patch_w - (right_margin + left_margin)
                if j == 0:
                    crop_w += left_margin
                if j == (tiling[1]-1):
                    crop_w += right_margin

                pooled_w = (crop_w + 1) // 2
                pooled_h = (crop_h + 1) // 2
                after_padding_width = image_token_length_w - pooled_w - crop_x0
                after_padding_height = image_token_length_h - pooled_h - crop_y0
                patch_ordering_arr.append(
                    np.pad(
                        np.reshape(
                            np.arange(on, on+pooled_h*pooled_w, dtype=np.int32),
                            (pooled_h, pooled_w)),
                        [[crop_y0, crop_x0], [image_token_length_h, image_token_length_w]], value=-1
                    )
                )
                patches_arr.append(src[y0:y0+crop_size, x0:x0+crop_size])
                mask_arr.append(img_mask[y0:y0+crop_size, x0:x0+crop_size])

                on += pooled_h*pooled_w
                on_patch += 1
        patches = np.stack(patches_arr)
        patch_ordering = np.stack(patch_ordering_arr)
        img_mask = np.stack(mask_arr)

        # Switch to [n_crops, n_patches, pixels_per_patch] format
        image_layout_impatch_w, image_layout_impatch_h = tiling[0], tiling[1]
        patches = einops.rearrange(
            patches, 'p (h dh) (w dw) c -> p (h w) (dh dw c)',
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w
        )
        img_mask = einops.rearrange(
            img_mask, 'p (h dh) (w dw) -> p (h w) (dh dw)',
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w
        )

        img_mask = img_mask.astype(np.float32).mean(axis=-1)
        patch_ordering = np.reshape(patch_ordering, [-1])
        valid = patch_ordering >= 0

        # Path order numbers the patches crop-by-crop, here we transpose
        # it to get left-to-right order
        patch_ordering_rh = np.reshape(
            patch_ordering,
            [tiling[0], tiling[1], image_token_length_h, image_token_length_w]
        )
        patch_ordering_rh = np.transpose(patch_ordering_rh, [0, 2, 1, 3])
        patch_ordering_rh = np.reshape(patch_ordering_rh, [-1])

        # The transpose will screw up which patches are masked, project the
        # new order into sparse structure of `patch_ordering` to fix it
        patch_ordering[valid] = patch_ordering_rh[patch_ordering_rh >= 0]

        # Now build the output tokens
        h = tiling[0] * crop_window_patches + (right_margin+left_margin)
        w = tiling[1] * crop_window_patches + (right_margin+left_margin)
        per_row = np.full(
            ((w+1)//2,),
            image_patch_token_id,
        )
        per_row = np.concatenate([per_row, [image_col_token_id]], 0)

        joint = np.tile(per_row, [(h+1)//2])
        joint = [
            [image_start_token_id],
            joint,
            [image_end_token_id]
        ]

        # Finally do the same for the global image
        resized, _ = resize_and_pad(image, base_image_input_size)
        resized = self._normalize(resized)
        resized = einops.rearrange(
            resized, '(h dh) (w dw) c -> (h w) (dh dw c)',
            dh=base_image_input_d,
            dw=base_image_input_d,
            h=image_base_patch_h,
            w=image_base_patch_w
        )
        patches = np.concatenate([np.expand_dims(resized, 0), patches], 0)

        # Global image goes first, so the order of patches in previous crops gets increased
        patch_ordering = np.where(
            patch_ordering >= 0,
            patch_ordering + tokens_per_image,
            -1
        )
        patch_ordering = np.concatenate([np.arange(0, tokens_per_image), patch_ordering], 0)
        per_row = np.full(
            (image_token_length_w,),
            image_patch_token_id,
        )
        per_row = np.concatenate([per_row, [image_col_token_id]], 0)
        extra_tokens = np.tile(per_row, [image_token_length_h])
        joint = [
                    [image_start_token_id],
                    extra_tokens,
                    [image_end_token_id],
                ] + joint

        joint = np.concatenate(joint, 0)
        img_mask = np.pad(img_mask, [[0, 1], [0, 0]], constant_values=-1)
        return patches, joint, patch_ordering, img_mask

    def build_image_input_idx(
        self,
        image_tokens: np.ndarray,
        patch_order: np.ndarray,
        image_patch_token_id: int,
        image_token_length_w: int,
        image_token_length_h: int,
    ):
        """Converts `patch_order` into a mapping of token_id -> patch_id"""

        tokens_per_image = image_token_length_w * image_token_length_h

        # Indices to insert the patches
        image_input_idx = image_tokens == image_patch_token_id
        image_input_idx = np.nonzero(image_input_idx)[0].astype(np.int32)

        if patch_order is not None:
            n_tokens = image_input_idx.shape[0]
            patch_order = np.reshape(patch_order, [-1])
            n_patches = patch_order.shape[0]

            valid = patch_order >= 0
            n_valid_patches = valid.sum()
            assert len(image_input_idx) == n_valid_patches

            sorted_patch_ixs = np.zeros([n_tokens], np.int32)
            sorted_patch_ixs[patch_order[valid]] = np.arange(n_valid_patches, dtype=np.int32)

            # Project the inverted mapping into same sparse structure
            sorted_patch_ixs_ex = np.full(np.shape(patch_order), -1)
            sorted_patch_ixs_ex[valid] = sorted_patch_ixs

            # Do the gather and then re-masked outputs that were masked in `sorted_patch_ixs`
            valid = (sorted_patch_ixs_ex >= 0).astype(np.int32)
            image_input_idx = image_input_idx[sorted_patch_ixs_ex*valid]
            image_input_idx = image_input_idx*valid - 100*(1 - valid)
            image_input_idx = np.reshape(image_input_idx, [-1, tokens_per_image])
        return image_input_idx

    def preprocess(
        self,
        image: np.ndarray,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        max_crops: Optional[int] = None,
        overlap_margins: Optional[List[int]] = None,
        base_image_input_size: Optional[Union[int, List[int]]] = None,
        image_token_length_w: Optional[int] = None,
        image_token_length_h: Optional[int] = None,
        image_patch_size: Optional[int] = None,
        **kwargs,
    ):
        """Preprocesses a single image

        Returns:
            crops: (n_crops, n_patches, patch_dim) individual crops, `n_crops` might
                   change between images but the other dimension are fixed
            tokens: (n_tokens,) int32 tokens, pad tokens indicate where to insert the
                                patch features, might include other special tokens as well
            image_idx: (n_crops, n_patches) index in `tokens` to put the patch features from the
                       crops after pooling, negative values indicates patches features to exclude
            padding_mask: (n_crops, n_patches) what percent of each crop is padding, can be None
                          if the image mask is not being used.
        """

        max_crops = max_crops or self.max_crops
        overlap_margins = overlap_margins or self.overlap_margins
        base_image_input_size = base_image_input_size or self.base_image_input_size
        image_token_length_w = image_token_length_w or self.image_token_length_w
        image_token_length_h = image_token_length_h or self.image_token_length_h
        image_patch_size = image_patch_size or self.image_patch_size

        crops, image_tokens, patch_ordering, img_mask = self.image_to_patches_and_tokens(
            image,
            image_patch_token_id,
            image_col_token_id,
            image_start_token_id,
            image_end_token_id,
            max_crops,
            overlap_margins,
            base_image_input_size,
            image_token_length_w,
            image_token_length_h,
            image_patch_size,
        )
        patch_idx = self.build_image_input_idx(
            image_tokens,
            patch_ordering,
            image_patch_token_id,
            image_token_length_w=image_token_length_w,
            image_token_length_h=image_token_length_h,
        )
        return crops, image_tokens, patch_idx, img_mask

    def multimodal_preprocess(
        self,
        images: np.ndarray,
        tokens: List[int],
        image_idx: np.ndarray,
        sequence_length: int,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        **kwargs,
    ):
        """Merge images and text tokens into multi-modal features for the model

        :param images: images to use as input
        :param tokens: input text tokens
        :param image_idx: where to insert the images into `tokens`
        :params image_patch_token_id: id to use of tokens that will contain image features
        :params image_col_token_id: token id for image column special tokens
        :params image_start_token_id: token id for image start special tokens
        :params image_end_token_id: token id for image end special tokens
        :params kwargs: override preprocessor default args
        """
        if images is None:
            return {"input_ids": tokens}

        max_total_crops = kwargs.get("max_crops") or self.max_crops
        image_token_length_w = kwargs.get("image_token_length_w") or self.image_token_length_w
        image_token_length_h = kwargs.get("image_token_length_h") or self.image_token_length_h
        image_patch_size = kwargs.get("image_patch_size") or self.image_patch_size
        base_image_input_size = kwargs.get("base_image_input_size") or self.base_image_input_size
        image_num_patch = (
            base_image_input_size[0] // image_patch_size,
            base_image_input_size[1] // image_patch_size,
        )
        image_padding_mask = kwargs.get("image_padding_mask") or self.image_padding_mask

        tokens_per_image = image_token_length_w * image_token_length_h
        n_pixels = image_patch_size * image_patch_size * 3
        n_patches = image_num_patch[0] * image_num_patch[1]

        n = len(images)
        all_crops = []
        all_image_idx = []
        out_tokens = []
        all_crop_masks = []

        for ix in range(n):
            token_ix = image_idx[ix]
            crops, image_tokens, patch_idx, img_mask = self.preprocess(
                images[ix],
                image_patch_token_id,
                image_col_token_id,
                image_start_token_id,
                image_end_token_id,
                **kwargs,
            )

            if token_ix == -1:  # -1 is an image inserted at the very start
                start = 0
                token_ix = 0
                end = 0
            else:
                start = 0 if ix == 0 else image_idx[ix-1] + 1
                end = token_ix + 1

            all_image_idx.append(patch_idx + token_ix)
            all_crops.append(crops)
            out_tokens.append(tokens[start:token_ix])
            out_tokens.append(image_tokens)
            if ix == (n - 1):
                out_tokens.append(tokens[end:])
            if image_padding_mask:
                all_crop_masks.append(img_mask)

        input_ids = np.concatenate(out_tokens, 0)
        images = np.concatenate(all_crops, 0)
        image_input_idx = np.concatenate(all_image_idx, 0)
        if image_padding_mask:
            image_masks = np.concatenate(all_crop_masks, 0)
        else:
            image_masks = None

        out = {
            "input_ids": input_ids,
            "images": images,
            "image_input_idx": image_input_idx
        }
        if image_masks is not None:
            out["image_masks"] = image_masks
        return out


MolmoImageProcessor.register_for_auto_class()