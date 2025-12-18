# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
This integration has for unique goal to allow re-initialization of the non-persistent buffers of `timm` models.
Indeed, as we load models fully on meta by default, we need a way to get back the correct value of non-persistent buffers.
We assume that everything else, i.e. parameters and persistent buffers, will correctly reside in model checkpoints, so we
won't need to reinit them.
Do not rely on it, as we will work to integrate it directly in `timm`, to then remove this file without warning.
"""

from .. import initialization as init
from ..utils import is_timm_available


if is_timm_available():
    from timm.layers.pos_embed_sincos import (
        FourierEmbed,
        RotaryEmbedding,
        RotaryEmbeddingCat,
        RotaryEmbeddingDinoV3,
        RotaryEmbeddingMixed,
        freq_bands,
        pixel_freq_bands,
    )


def _maybe_reinit_non_persistent_buffer(module):
    # This is a loooong list of hardcoded combinations from timm, as the modules do not provide a nice way to do
    # it natively
    if isinstance(module, FourierEmbed):
        init.copy_(module.bands, pixel_freq_bands(module.max_res, module.num_bands))
    elif isinstance(module, RotaryEmbedding):
        if module.bands is not None:
            bands = (
                pixel_freq_bands(module.dim // 4, float(module.max_res), linear_bands=module.linear_bands)
                if module.in_pixels
                else freq_bands(module.dim // 4, temperature=module.temperature, step=1)
            )
            init.copy_(module.bands, bands)
        elif module.pos_embed_sin is not None:
            emb_sin, emb_cos = module._get_pos_embed_values(module.feat_shape)
            init.copy_(module.pos_embed_sin, emb_sin)
            init.copy_(module.pos_embed_cos, emb_cos)
    elif isinstance(module, RotaryEmbeddingCat):
        if module.bands is not None:
            bands = (
                pixel_freq_bands(module.dim // 4, float(module.max_res), linear_bands=module.linear_bands)
                if module.in_pixels
                else freq_bands(module.dim // 4, temperature=module.temperature, step=1)
            )
            init.copy_(module.bands, bands)
        elif module.pos_embed is not None:
            init.copy_(module.pos_embed, module._get_pos_embed_values(feat_shape=module.feat_shape))
    elif isinstance(module, RotaryEmbeddingMixed):
        if module.t_x is not None:
            t_x, t_y = module._get_grid_values(module.feat_shape)
            init.copy_(module.t_x, t_x)
            init.copy(module.t_y, t_y)
    elif isinstance(module, RotaryEmbeddingDinoV3):
        init.copy_(module.periods, module._compute_periods())
        if module.pos_embed_cached is not None:
            init.copy_(module.pos_embed_cached, module._create_embed(module.feat_shape, no_aug=True))
