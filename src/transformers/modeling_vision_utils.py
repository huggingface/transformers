# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional

import torch.nn as nn

from .utils import is_torch_available, torch_int
from .utils.logging import get_logger


if is_torch_available():
    import torch

logger = get_logger(__name__)


def interpolate_pos_encoding(
    embeddings: torch.Tensor,
    position_embeddings: torch.Tensor,
    height: int,
    width: int,
    patch_size: int | List[int],
    num_class_embeds: int = 1,
    interpolate_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
    images. This method is also adapted to support models that do not have class embeddings (e.g., SigLIP or Hiera) and
    to enable torch.jit tracing.

    Adapted from:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194 and
    https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
    """

    num_patches = embeddings.shape[1] - num_class_embeds
    num_positions = position_embeddings.shape[1] - num_class_embeds

    # always interpolate when tracing to ensure the exported model works for dynamic input shapes
    if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
        return position_embeddings

    class_pos_embed = position_embeddings[:, :num_class_embeds]
    patch_pos_embed = position_embeddings[:, num_class_embeds:]

    dim = embeddings.shape[-1]

    ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    new_height = height // ph
    new_width = width // pw

    sqrt_num_positions = torch_int(num_positions**0.5)
    patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

    target_dtype = patch_pos_embed.dtype
    if interpolate_dtype is not None:
        patch_pos_embed = patch_pos_embed.to(interpolate_dtype)

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )
    if interpolate_dtype is not None:
        patch_pos_embed = patch_pos_embed.to(dtype=target_dtype)

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
