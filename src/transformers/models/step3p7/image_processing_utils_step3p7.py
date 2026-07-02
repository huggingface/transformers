# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from itertools import product
from math import ceil
from typing import Any

import numpy as np

from ...utils import is_vision_available


if is_vision_available():
    from PIL import Image


MAX_IMAGE_SIZE: int = 3024
ImageWithPatches = tuple[Any, list[Any], list[bool] | None]


class ImagePatcher:
    def determine_window_size(self, long: int, short: int) -> int:
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
        img_rate_thr: float = 0.6,
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
        assert 1 >= img_rate_thr >= 0, "The `in_rate_thr` should lie in 0~1"
        windows = []
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step

            x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w

            y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)

        return [(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])) for box in windows], (
            x_num,
            y_num,
        )

    def square_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    def get_image_size_for_padding(self, img_width: int, img_height: int) -> tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(self, img_width: int, img_height: int) -> tuple[int, int]:
        if max(img_height, img_width) > MAX_IMAGE_SIZE:
            scale_factor = MAX_IMAGE_SIZE / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
        return img_width, img_height

    def get_image_size_for_crop(self, img_width: int, img_height: int, window_size: int):
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size

        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio
        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio
        return int(width_new), int(height_new)

    def patch_crop(self, img: Image.Image, i: int, j: int, th: int, tw: int):
        target = img.crop((j, i, j + tw, i + th))
        return target

    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
        img_width, img_height = self.get_image_size_for_padding(img_width, img_height)
        img_width, img_height = self.get_image_size_for_preprocess(img_width, img_height)
        window_size = self.determine_window_size(max(img_height, img_width), min(img_height, img_width))
        if window_size == 0:
            return 0, 0

        img_width, img_height = self.get_image_size_for_crop(img_width, img_height, window_size)
        center_list, (x_num, _) = self.slide_window(
            img_width, img_height, [(window_size, window_size)], [(window_size, window_size)]
        )
        full_rows = (len(center_list) - 1) // x_num + 1
        if len(center_list) > 0 and len(center_list) % x_num == 0:
            full_rows -= 1
        return len(center_list), full_rows

    def __call__(self, img: Image.Image) -> ImageWithPatches:
        img_width, img_height = img.size
        new_img_width, new_img_height = self.get_image_size_for_padding(img_width, img_height)
        if new_img_width != img_width or new_img_height != img_height:
            img = self.square_pad(img)
            img_width, img_height = img.size

        new_img_width, new_img_height = self.get_image_size_for_preprocess(img_width, img_height)
        img = img.resize((new_img_width, new_img_height), Image.Resampling.BILINEAR)
        window_size = self.determine_window_size(
            max(new_img_height, new_img_width), min(new_img_height, new_img_width)
        )
        if window_size == 0:
            return img, [], None

        new_img_width, new_img_height = self.get_image_size_for_crop(new_img_width, new_img_height, window_size)
        if (new_img_width, new_img_height) != (img_width, img_height):
            img_for_crop = img.resize((new_img_width, new_img_height), Image.Resampling.BILINEAR)
        else:
            img_for_crop = img

        patches = []
        newlines = []
        center_list, (x_num, _) = self.slide_window(
            new_img_width, new_img_height, [(window_size, window_size)], [(window_size, window_size)]
        )
        for patch_id, center_lf_point in enumerate(center_list):
            x, y, patch_w, patch_h = center_lf_point
            big_patch = self.patch_crop(img_for_crop, y, x, patch_h, patch_w)
            patches.append(big_patch)
            if (patch_id + 1) % x_num == 0:
                newlines.append(patch_id)

        if newlines and newlines[-1] == len(patches) - 1:
            newlines.pop()

        return img, patches, [i in newlines for i in range(len(patches))] if len(patches) > 0 else None


class Step3ImagePatcherMixin:
    image_patcher = ImagePatcher()

    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
        return self.image_patcher.get_num_patches(img_width, img_height)

    def split_image_to_patches(self, image: Image.Image) -> ImageWithPatches:
        return self.image_patcher(image)

    def split_images_to_patches(self, images: list[Image.Image]) -> list[ImageWithPatches]:
        return [self.split_image_to_patches(image) for image in images]

    def _prepare_step3_images(self, images: Any) -> list[Any]:
        images = self.fetch_images(images)
        if not isinstance(images, list):
            return [images]
        if images and isinstance(images[0], list):
            return images[0]
        return images

    def prepare_image_inputs(self, images: Any, **kwargs) -> tuple[dict[str, Any], list[list[bool] | None]]:
        """
        Prepare Step3p7 image tensors and placeholder metadata.

        Returns a pair of `(image_inputs, patch_newline_masks)`. `image_inputs` contains `pixel_values`,
        `num_patches`, and, when crops are produced, `patch_pixel_values` and `patch_newline_mask`. The second
        element keeps the per-image newline masks used by [`Step3VLProcessor`] to expand image placeholders.
        """
        import torch

        images = self._prepare_step3_images(images)
        if len(images) == 0:
            return {}, []

        kwargs = kwargs.copy()
        kwargs.pop("is_patch", None)
        if kwargs.get("return_tensors") is None:
            kwargs["return_tensors"] = "pt"

        pixel_values = []
        patch_pixel_values = []
        patch_newline_masks = []
        patch_newline_mask = []
        num_patches = []

        for image, image_patches, image_patch_newline_mask in self.split_images_to_patches(images):
            pixel_values.append(self(image, **kwargs)["pixel_values"])
            if image_patches:
                patch_pixel_values.append(self(image_patches, is_patch=True, **kwargs)["pixel_values"])
            num_patches.append(len(image_patches))
            patch_newline_masks.append(image_patch_newline_mask)
            if image_patch_newline_mask is not None:
                patch_newline_mask.extend(image_patch_newline_mask)

        image_inputs = {
            "pixel_values": torch.cat(pixel_values),
            "num_patches": num_patches,
        }
        if patch_pixel_values:
            image_inputs["patch_pixel_values"] = torch.cat(patch_pixel_values)
        if patch_newline_mask:
            image_inputs["patch_newline_mask"] = torch.tensor(patch_newline_mask, dtype=torch.bool)

        return image_inputs, patch_newline_masks


__all__ = ["ImagePatcher", "ImageWithPatches", "Step3ImagePatcherMixin"]
