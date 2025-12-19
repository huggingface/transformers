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

from math import comb

import torch

from .. import initialization as init
from ..utils import is_timm_available


if is_timm_available():
    from timm.layers import ndgrid
    from timm.layers.blur_pool import BlurPool2d
    from timm.layers.lambda_layer import LambdaLayer, rel_pos_indices
    from timm.layers.pos_embed_rel import (
        RelPosBias,
        RelPosBiasTf,
        RelPosMlp,
        gen_relative_log_coords,
        gen_relative_position_index,
        generate_lookup_tensor,
    )
    from timm.layers.pos_embed_sincos import (
        FourierEmbed,
        RotaryEmbedding,
        RotaryEmbeddingCat,
        RotaryEmbeddingDinoV3,
        RotaryEmbeddingMixed,
        freq_bands,
        pixel_freq_bands,
    )
    from timm.models.beit import Attention
    from timm.models.beit import gen_relative_position_index as beit_gen_relative_position_index
    from timm.models.efficientformer_v2 import Attention2d, Attention2dDownsample
    from timm.models.eva import EvaAttention
    from timm.models.levit import AttentionDownsample
    from timm.models.swin_transformer import SwinTransformerBlock, get_relative_position_index
    from timm.models.swin_transformer import WindowAttention as SwinWindowAttention
    from timm.models.swin_transformer_v2 import SwinTransformerV2Block
    from timm.models.swin_transformer_v2 import WindowAttention as Swin2WindowAttention
    from timm.models.swin_transformer_v2_cr import SwinTransformerV2CrBlock, WindowMultiHeadAttention
    from timm.models.vision_transformer import ParallelScalingBlock

    # This one is very recent and is not necesarily in all versions we support (we require timm>=1.0.20)
    try:
        from timm.models.csatv2 import _DCT_MEAN, _DCT_VAR, LearnableDct2d
    except Exception:
        _DCT_MEAN, _DCT_VAR, LearnableDct2d = None, None, type(None)


def _maybe_reinit_non_persistent_buffer(module):
    """Reinit the non-persistent buffers of `module` if it matches any timm Module which has any."""
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
    elif isinstance(module, RelPosBias):
        has_class_token = module.relative_position_bias_table.shape[0] > (2 * module.window_size[0] - 1) * (
            2 * module.window_size[1] - 1
        )
        init.copy_(
            module.relative_position_index,
            gen_relative_position_index(module.window_size, class_token=has_class_token).view(-1),
        )
    elif isinstance(module, RelPosMlp):
        init.copy_(module.relative_position_index, gen_relative_position_index(module.window_size).view(-1))
        # This one is supposed to pass args `pretrained_window_size` as well to `gen_relative_log_coords`, but it's
        # not recorded as class attributes in `__init__` and we have no way to infer its value back as we do for `mode` here...
        # Let's hope it's always default value
        mode = "cr" if module.bias_gain is None else "swin"
        init.copy_(module.rel_coords_log, gen_relative_log_coords(module.window_size, mode=mode))
    elif isinstance(module, RelPosBiasTf):
        init.copy_(module.height_lookup, generate_lookup_tensor(module.window_size[0]))
        init.copy_(module.width_lookup, generate_lookup_tensor(module.window_size[1]))
    elif isinstance(module, LearnableDct2d):
        init.copy_(module.mean, torch.tensor(_DCT_MEAN))
        init.copy_(module.var, torch.tensor(_DCT_VAR))
        init.copy_(module.imagenet_mean, torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        init.copy_(module.imagenet_std, torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    elif isinstance(module, LambdaLayer):
        if module.rel_pos_indices is not None:
            rel_size = module.pos_enb.shape[:2]
            feat_size = [(s + 1) // 2 for s in rel_size]
            init.copy_(module.rel_pos_indices, rel_pos_indices(feat_size))
    elif isinstance(module, AttentionDownsample):
        k_pos = torch.stack(
            ndgrid(
                torch.arange(module.resolution[0], dtype=torch.long),
                torch.arange(module.resolution[1], dtype=torch.long),
            )
        ).flatten(1)
        q_pos = torch.stack(
            ndgrid(
                torch.arange(0, module.resolution[0], step=module.stride, dtype=torch.long),
                torch.arange(0, module.resolution[1], step=module.stride, dtype=torch.long),
            )
        ).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * module.resolution[1]) + rel_pos[1]
        init.copy_(module.attention_bias_idxs, rel_pos)
    elif isinstance(
        module,
        EvaAttention,
    ):
        if module.k_bias is not None:
            init.zeros_(module.k_bias)
    elif isinstance(module, ParallelScalingBlock):
        if module.qkv_bias is not None:
            init.zeros_(module.qkv_bias)
    elif isinstance(module, Attention):
        if module.k_bias is not None:
            init.zeros_(module.k_bias)
        if module.relative_position_index is not None:
            init.copy_(module.relative_position_index, beit_gen_relative_position_index(module.window_size))
    elif isinstance(module, SwinTransformerV2CrBlock):
        if module.attn_mask is not None:
            init.copy_(module.attn_mask, module.get_attn_mask())
    elif isinstance(module, WindowMultiHeadAttention):
        module._make_pair_wise_relative_positions()
    elif isinstance(module, BlurPool2d):
        coeffs = torch.tensor(
            [comb(module.filt_size - 1, k) for k in range(module.filt_size)], dtype=torch.float32
        ) / (2 ** (module.filt_size - 1))
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :]
        if module.channels is not None:
            blur_filter = blur_filter.repeat(module.channels, 1, 1, 1)
        init.copy_(module.filt, blur_filter)
    elif isinstance(module, Swin2WindowAttention):
        module._make_pair_wise_relative_positions()
        if module.k_bias is not None:
            init.zeros_(module.k_bias)
    elif isinstance(module, SwinTransformerV2Block):
        if module.attn_mask is not None:
            init.copy_(module.attn_mask, module.get_attn_mask())
    elif isinstance(module, SwinWindowAttention):
        init.copy_(module.relative_position_index, get_relative_position_index(*module.window_size))
    elif isinstance(module, SwinTransformerBlock):
        if module.attn_mask is not None:
            init.copy_(module.attn_mask, module.get_attn_mask())
    elif isinstance(module, Attention2d):
        pos = torch.stack(
            ndgrid(
                torch.arange(module.resolution[0], dtype=torch.long),
                torch.arange(module.resolution[1], dtype=torch.long),
            )
        ).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * module.resolution[1]) + rel_pos[1]
        init.copy_(module.attention_bias_idxs, rel_pos)
    elif isinstance(module, Attention2dDownsample):
        k_pos = torch.stack(
            ndgrid(
                torch.arange(module.resolution[0], dtype=torch.long),
                torch.arange(module.resolution[1], dtype=torch.long),
            )
        ).flatten(1)
        q_pos = torch.stack(
            ndgrid(
                torch.arange(0, module.resolution[0], step=2, dtype=torch.long),
                torch.arange(0, module.resolution[1], step=2, dtype=torch.long),
            )
        ).flatten(1)
        rel_pos = (q_pos[..., :, None] - k_pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * module.resolution[1]) + rel_pos[1]
        init.copy_(module.attention_bias_idxs, rel_pos)
