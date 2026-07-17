# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import itertools
import math

import torch
from PIL import Image
from torchvision import transforms as T

from ...image_processing_utils import BaseImageProcessor


def _grid_size(img_w: int, img_h: int, patch_hw: int, max_tokens: int) -> tuple[int, int, int]:
    """Pick the integer (H, W) grid closest to the aspect ratio under the token cap.

    Mirrors ``OnyxVisionEncoder._compute_grid_size`` so the processor needs no
    torch model import. Returns ``(target_h, target_w, n_tokens)``.
    """
    i_nph = img_h / patch_hw
    i_npw = img_w / patch_hw
    ratio = i_npw / i_nph if i_nph > 0 else 1.0
    if i_nph * i_npw > max_tokens:
        i_nph = (max_tokens / ratio) ** 0.5
        i_npw = i_nph * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(i_nph), math.ceil(i_nph)],
                [math.floor(i_npw), math.ceil(i_npw)],
            )
        )
    )
    candidates = [(nph, npw) for nph, npw in candidates if nph >= 1 and npw >= 1 and nph * npw <= max_tokens]
    if not candidates:
        candidates = [(max(1, round(i_nph)), max(1, round(i_npw)))]
    nph, npw = min(candidates, key=lambda c: abs(c[0] / c[1] - img_h / img_w))
    return nph * patch_hw, npw * patch_hw, nph * npw


class OnyxImageProcessor(BaseImageProcessor):
    """Resize + normalize Onyx images and compute patch-token counts.

    Variable-resolution: each image is resized to the grid that best matches its
    aspect ratio under the per-image token cap, then normalized with mean/std 0.5.
    Returns per-image tensors (not stacked) because Onyx consumes a list of
    variable-size images. Video frames are handled by ``OnyxVideoProcessor``.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        patch_size: int = 14,
        downsample_factor: int = 2,
        max_image_tokens: int = 4096,
        image_mean: float = 0.5,
        image_std: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.max_image_tokens = max_image_tokens
        self.image_mean = image_mean
        self.image_std = image_std

    def _to_norm_tensor(self, image: Image.Image) -> torch.Tensor:
        return T.functional.normalize(
            T.functional.to_tensor(image),
            [self.image_mean] * 3,
            [self.image_std] * 3,
        )

    def compute_image_size(self, img_w: int, img_h: int) -> tuple[int, int, int]:
        ph = self.patch_size * self.downsample_factor
        return _grid_size(img_w, img_h, ph, self.max_image_tokens)

    def preprocess_image(self, image: Image.Image) -> tuple[torch.Tensor, int]:
        """Return ``(pixel tensor [3, H, W], n_patch_tokens)`` for one image."""
        image = image.convert("RGB")
        target_h, target_w, n_tokens = self.compute_image_size(image.width, image.height)
        image = image.resize((target_w, target_h), Image.LANCZOS)
        return self._to_norm_tensor(image), n_tokens


__all__ = ["OnyxImageProcessor"]
