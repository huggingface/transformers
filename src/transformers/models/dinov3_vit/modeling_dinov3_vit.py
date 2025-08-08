# coding=utf-8
# Copyright 2025 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch DINOv3 model."""

import collections.abc
import math
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import check_model_inputs
from .configuration_dinov3_vit import DINOv3ViTConfig


logger = logging.get_logger(__name__)

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# Copied from transformers.models.dinov2.modeling_dinov2.Dinov2PatchEmbeddings with Dinov2 -> DINOv3ViT
class DINOv3ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class DINOv3ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = DINOv3ViTPatchEmbeddings(config)

    def forward(self, pixel_values: Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> Tensor:
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype), embeddings)

        # Add CLS and register tokens
        batch_size = embeddings.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_token, register_tokens, embeddings], dim=1)

        return embeddings


class DINOv3ViTRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor
    
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()

        self.config = config
        self.base = config.pos_embed_rope_base
        self.head_dim = config.hidden_size // config.num_attention_heads

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)  # (head_dim / 4,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def augment_coords_(self, coords: torch.Tensor) -> torch.Tensor:

        # Shift coords by adding a uniform value in [-shift, shift]
        if shift := self.config.pos_embed_rope_shift_coords is not None:
            shift_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
            shift_hw = shift_hw.uniform_(-shift, shift)
            coords += shift_hw

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if jitter := self.config.pos_embed_rope_jitter_coords is not None:
            jitter_range = np.log(jitter)
            jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
            jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
            coords *= jitter_hw

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if rescale := self.config.pos_embed_rope_rescale_coords is not None:
            rescale_range = np.log(rescale)
            rescale_hw = torch.empty(1, device=coords.device, dtype=coords.dtype)
            rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
            coords *= rescale_hw

        return coords

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.inv_freq.device
        dtype = torch.float32
        dd = {"device": device, "dtype": dtype}
        coords_h = torch.arange(0.5, H, **dd) / H  # [H]
        coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        if self.training:
            coords = self.augment_coords_(coords)

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q: Tensor, k: Tensor, rope: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
    q_dtype = q.dtype
    k_dtype = k.dtype
    sin, cos = rope
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    N = q.shape[-2]
    prefix = N - sin.shape[-2]
    assert prefix >= 0
    q_prefix = q[:, :, :prefix, :]
    q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
    q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
    k_prefix = k[:, :, :prefix, :]
    k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
    k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
    q = q.to(dtype=q_dtype)
    k = k.to(dtype=k_dtype)
    return q, k


# # Copied from transformers.models.llama.modeling_llama.rotate_half
# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# Copied from transformers.models.pixtral.modeling_pixtral.PixtralAttention with Pixtral->DINOv3ViT
class DINOv3ViTAttention(nn.Module):
    """
    Multi-headed attention compatible with ALL_ATTENTION_FUNCTIONS.
    """

    def __init__(self, config: DINOv3ViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = False

        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.dropout = config.attention_dropout

        # NOTE: modified for granular control over bias, DINOv3ViT has no bias in the key projection
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.value_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.output_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        # cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position_embeddings)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Since we use packing, if flash_attention_2 is selected we rely on position_ids
        if self.config._attn_implementation == "flash_attention_2":
            kwargs["position_ids"] = kwargs["position_ids"].to(hidden_states.device, non_blocking=True)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(config.hidden_size))
        self.init_values = config.layerscale_value

    def init_weights(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.gamma


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class DINOv3ViTDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class DINOv3ViTMLP(nn.Module):
    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class DINOv3ViTSwiGLUFFN(nn.Module):
    def __init__(
        self,
        config,
        device=None,
    ) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % config.swiglu_align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=True, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=True, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=True, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = nn.functional.silu(x1) * x2
        return self.w3(hidden)


class DINOv3ViTLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.drop_path = DINOv3ViTDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_swiglu_ffn:
            self.mlp = DINOv3ViTSwiGLUFFN(config)
        else:
            self.mlp = DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in DINOv3, layernorm is applied before self-attention
            head_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  #
        attention_output = self.layer_scale1(attention_output)

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in DINOv3, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        return (layer_output,) + outputs


@auto_docstring
class DINOv3ViTPreTrainedModel(PreTrainedModel):
    config: DINOv3ViTConfig
    base_model_prefix = "DINOv3ViT"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DINOv3ViTLayer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _can_record_outputs = {
        "hidden_states": "DINOv3ViTLayer",
        "attentions": "DINOv3ViTAttention",
    }

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, DINOv3ViTEmbeddings):
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)
            if module.config.num_register_tokens > 0:
                module.register_tokens.data = nn.init.trunc_normal_(
                    module.register_tokens.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.register_tokens.dtype)
            module.mask_token.data.zero_()
        elif isinstance(module, DINOv3ViTLayerScale):
            module.gamma.data.fill_(self.config.layerscale_value)


@auto_docstring
class DINOv3ViTModel(DINOv3ViTPreTrainedModel):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.layer = nn.ModuleList([DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> DINOv3ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.
        """

        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        num_patches_height = self.config.image_size // self.config.patch_size
        num_patches_width = self.config.image_size // self.config.patch_size
        position_embeddings = self.rope_embeddings(H=num_patches_height, W=num_patches_width)

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                head_mask=layer_head_mask,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        sequence_output = self.norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


__all__ = ["DINOv3ViTModel", "DINOv3ViTPreTrainedModel"]
