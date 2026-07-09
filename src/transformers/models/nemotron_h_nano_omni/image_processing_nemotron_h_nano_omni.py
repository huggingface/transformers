# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Fast image processor for the NemotronH Nano Omni model."""

import math

import numpy as np

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_base import BatchFeature
from ...image_utils import ImageInput, ImageType, get_image_type, make_list_of_images
from ...utils import TensorType, is_torch_available


if is_torch_available():
    import torch


class NemotronH_Nano_Omni_Reasoning_V3ImageProcessor(TorchvisionBackend):
    """
    Dynamic-resolution image processor for the NemotronH Nano Omni model.

    Each image is resized to a single tile whose patch-grid `(h_patches, w_patches)` is chosen to
    land between `min_num_patches` and `max_num_patches` (on a `patch_size` grid), respecting the
    aspect ratio, then normalized with `norm_mean` / `norm_std`. Video frames use a separate fixed
    target-patch budget with aspect ratio preserved (toggled via `_is_video_mode`).
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        norm_mean=None,
        norm_std=None,
        patch_size=16,
        downsample_ratio=0.5,
        min_num_patches=1024,
        max_num_patches=13312,
        max_model_len=16384,
        video_target_num_patches=1024,
        video_maintain_aspect_ratio=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        # Integer reduction factor for pixel_shuffle (downsample_ratio = 0.5 -> factor 2).
        self._downsample_factor = int(round(1.0 / downsample_ratio))
        # Per-image patch-grid bounds (on the pre-pixel-shuffle `patch_size` grid).
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.max_model_len = max_model_len
        # Video frames use a separate (fixed) target-patch budget with aspect-ratio preserved.
        self.video_target_num_patches = video_target_num_patches
        self.video_maintain_aspect_ratio = video_maintain_aspect_ratio

    # Keep the PIL image through to `_preprocess`: we need PIL.resize semantics to match the
    # reference tiling algorithm exactly; resizing a tensor via `torchvision.transforms.Resize`
    # uses different kernels and breaks bit-exact agreement.
    def _process_image(self, image: ImageInput, **kwargs):
        if get_image_type(image) == ImageType.PIL:
            if image.mode != "RGB":
                image = image.convert("RGB")
        return image

    # transformers 5.6 renamed this hook from `_process_image` to `process_image`; alias both.
    process_image = _process_image

    # Toggled by the processor around video calls (the strict `ImagesKwargs` validator won't let us
    # thread a new kwarg down, so we use a flag on the instance instead).
    _is_video_mode: bool = False

    def _preprocess(self, images, return_tensors=None, **kwargs) -> BatchFeature:
        """Resize each image to its dynamic-resolution tile, normalize, and return pixel values.

        When `self._is_video_mode=True` (flipped by the processor before the video call), each
        input is resized using the **video** target-size rule (`video_target_num_patches`,
        aspect-ratio preserved) instead of the image dynamic-res rule.
        """
        is_video = self._is_video_mode
        images = make_list_of_images(images)

        target_sizes = []
        if is_video:
            for img in images:
                target_w_patches, target_h_patches = self._compute_target_patches_video(img)
                target_sizes.append((target_w_patches, target_h_patches))
        else:
            # Image path: per-image budget bounded by [min_num_patches, max_num_patches], with a
            # global cap derived from `max_model_len` x pixel-shuffle factor^2.
            num_tokens_available = self.max_model_len - 4
            budget = num_tokens_available * (self._downsample_factor**2)
            budget = max(budget, self.min_num_patches * len(images))
            max_budget = self.max_num_patches if (self.max_num_patches and self.max_num_patches > 0) else float("inf")
            per_image_budget = [max(min(budget, max_budget), self.min_num_patches) for _ in images]
            for img, tokens_for_media in zip(images, per_image_budget):
                target_w_patches, target_h_patches = self._compute_target_patches(img, tokens_for_media)
                target_sizes.append((target_w_patches, target_h_patches))

        norm_mean = torch.tensor(self.norm_mean).view(1, 3, 1, 1)
        norm_std = torch.tensor(self.norm_std).view(1, 3, 1, 1)

        pixel_values_list = []
        num_tokens_per_image = []
        imgs_sizes = []
        for img, (wp, hp) in zip(images, target_sizes):
            target_w = wp * self.patch_size
            target_h = hp * self.patch_size
            # Antialiased bicubic interpolation via `torch.nn.functional.interpolate`. PIL's bicubic
            # uses a different kernel (and no antialiasing), producing pixel values that amplify
            # through the ViT / mamba stack and cause outputs to diverge past the first few tokens.
            arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)  # (1, 3, H, W)
            if t.shape[-2] != target_h or t.shape[-1] != target_w:
                t = torch.nn.functional.interpolate(
                    t, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True
                )
            t = (t / 255.0 - norm_mean) / norm_std
            pixel_values_list.append(t.squeeze(0))  # (3, H, W)
            num_tokens_per_image.append((wp * hp) // (self._downsample_factor**2))
            imgs_sizes.append((target_h, target_w))

        # Stack if all images have the same target size; otherwise keep as a list of
        # (3, H_i, W_i) tensors (the model's `extract_feature` handles both).
        all_same_shape = all(t.shape == pixel_values_list[0].shape for t in pixel_values_list)
        pixel_values = torch.stack(pixel_values_list, dim=0) if all_same_shape else pixel_values_list

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "num_patches": [1] * len(images),
                "num_tokens": num_tokens_per_image,
                "imgs_sizes": imgs_sizes,
            },
            tensor_type=(return_tensors if all_same_shape else None),
        )

    def _compute_target_patches(self, img, tokens_available: int):
        """Choose the `(w_patches, h_patches)` tile that best fits `tokens_available` (image path)."""
        orig_w, orig_h = img.width, img.height
        # Ceil-ish: `round(x + 0.5)` == `floor(x) + 1` for non-integer x, `x` for integer.
        closest_patch_h = round(orig_h / self.patch_size + 0.5)
        closest_patch_w = round(orig_w / self.patch_size + 0.5)
        patches = closest_patch_h * closest_patch_w

        # Downscale to fit the token budget.
        factor = min(math.sqrt(tokens_available / patches), 1.0)
        target_h = math.floor(factor * closest_patch_h)
        target_w = math.floor(factor * closest_patch_w)

        # Scale up if below the per-image minimum.
        if tokens_available > self.min_num_patches and target_h * target_w < self.min_num_patches:
            up = math.sqrt(self.min_num_patches / (target_h * target_w))
            target_h = math.ceil(up * target_h)
            target_w = math.ceil(up * target_w)

        # Round each dim to a multiple of the pixel_shuffle factor so tokens divide evenly.
        divisor = self._downsample_factor
        rem_h = target_h % divisor
        if rem_h:
            inc_h = divisor - rem_h
            if (target_h + inc_h) * target_w <= tokens_available:
                target_h += inc_h
            else:
                target_h = max(divisor, target_h - rem_h)
        rem_w = target_w % divisor
        if rem_w:
            inc_w = divisor - rem_w
            if target_h * (target_w + inc_w) <= tokens_available:
                target_w += inc_w
            else:
                target_w = max(divisor, target_w - rem_w)

        return target_w, target_h

    def _compute_target_patches_video(self, img):
        """Choose an aspect-preserving `(w_patches, h_patches)` tile near `video_target_num_patches`."""
        orig_w, orig_h = img.width, img.height
        target = self.video_target_num_patches
        divisor = self._downsample_factor
        if self.video_maintain_aspect_ratio:
            aspect_wh = orig_w / max(orig_h, 1)
            ph = max(round(math.sqrt(target / aspect_wh)), 1)
            pw = max(round(math.sqrt(target * aspect_wh)), 1)
            if divisor > 1:
                rem_h = ph % divisor
                rem_w = pw % divisor
                ph_up = ph + (divisor - rem_h if rem_h else 0)
                ph_down = ph - rem_h
                pw_up = pw + (divisor - rem_w if rem_w else 0)
                pw_down = pw - rem_w
                if ph_up * pw_up <= target:
                    ph, pw = ph_up, pw_up
                else:
                    ph = max(divisor, ph_down)
                    pw = max(divisor, pw_down)
        else:
            side = int(math.sqrt(target))
            side = max(divisor, (side // divisor) * divisor)
            ph = pw = side
        return pw, ph


__all__ = ["NemotronH_Nano_Omni_Reasoning_V3ImageProcessor"]
