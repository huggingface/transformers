# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Mllama model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn, Tensor

from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_mllama import MllamaConfig, MllamaCrossAttentionTextConfig, MllamaCrossAttentionVisionConfig
import collections
from functools import partial
import torch.nn.functional as F
from ...configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MllamaConfig"

import collections
import torch
import math


def get_negative_inf_value(dtype):
    return torch.finfo(dtype).min


def to_tuple(x) -> Tuple[int, int]:
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def get_full_row_masked_out_mask(
    attn_bias,
    negative_inf_value,
):
    """
    attn_bias should be a 4D tensor of shape [B, H, S1, S2]
    where B is the batch size, H is the number of heads,
    and S1/S2 are the sequence lengths. This returns
    a 4D tensor of shape [B, H, S1, 1] which stores boolean
    values which are 0 if the a full row in the last dimension
    contains negative infinity values, otherwise it's 1.
    """
    return (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]


def convert_sparse_cross_attention_mask_to_dense(
    cross_attention_token_mask: List[List[List[int]]],
    num_tiles: List[List[int]],
    max_num_tiles: int,
    total_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    
    inf_value = get_negative_inf_value(dtype)

    batch_size = len(cross_attention_token_mask)
    max_num_images = max([len(masks) for masks in cross_attention_token_mask])

    out_masks = torch.full(
        size=(batch_size, total_len, max_num_images, max_num_tiles),
        fill_value=inf_value,
        dtype=dtype,
        device=device,
    )

    for idx, (mask_i, num_tiles_i) in enumerate(zip(cross_attention_token_mask, num_tiles)):
        for mask_idx, (token_locations, mask_num_chunks) in enumerate(zip(mask_i, num_tiles_i)):
            if len(token_locations) == 2:
                start, end = token_locations
                end = min(end, total_len)
                if end == -1:
                    end = total_len
                out_masks[idx, start:end, mask_idx, :mask_num_chunks].fill_(0.0)

    return out_masks


def prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    vision_tokens: torch.Tensor,
) -> Tuple[Tensor, Tensor]:
    
    if vision_tokens is None:
        raise ValueError("Vision tokens must be provided")
    if vision_tokens.shape[1] != cross_attention_mask.shape[2]:
        raise ValueError(f"Mismatch in number of images given and number of masks given {vision_tokens.shape} {cross_attention_mask.shape}")
    if vision_tokens.shape[2] != cross_attention_mask.shape[3]:
        raise ValueError(f"Vision tokens shape {vision_tokens.shape} mismatch with xattn shape {cross_attention_mask.shape}")

    seq_length = vision_tokens.shape[3]
    batch_size, text_total_length, _, _ = cross_attention_mask.shape

    cross_attention_mask = cross_attention_mask.repeat_interleave(seq_length, dim=2)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    inf_value = get_negative_inf_value(cross_attention_mask.dtype)
    full_text_row_masked_out_mask = get_full_row_masked_out_mask(cross_attention_mask, inf_value)
    
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask, full_text_row_masked_out_mask



# vision encoder utils 
# --------------------

def build_encoder_attention_mask(
    x: torch.Tensor,
    ar: torch.Tensor,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):
    """
    Build vision encoder attention mask that omits padding tokens.
    """
    masks = []
    for arx in ar:
        mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * get_negative_inf_value(x.dtype)
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = torch.stack(masks).to(x.device).expand(-1, n_heads, -1, -1)
    return masks

def expand_num_tokens_to_mult8(x):
    num_pad_tokens = (8 - (x.shape[-2] % 8))
    if num_pad_tokens == 0:
        return x, 0
    else:
        return torch.cat(
            [
                x,
                torch.zeros(
                    (x.shape[0], x.shape[1], num_pad_tokens, x.shape[-1]),
                    dtype=x.dtype,
                    device=x.device
                ),
            ],
            dim=-2,
        ), num_pad_tokens

def contract_num_tokens_from_mult8(x, num_pad_tokens):
    if num_pad_tokens == 0:
        return x
    return x[:, :, :-num_pad_tokens]

# --------------------
# end of vision encoder utils


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MllamaPatchEmbedding(nn.Module):
    """Conv2D Patching layer implemented as linear layer operation.
    (can be later replaced with Conv2d(bias=False) + .flatten(2).transpose(1, 2) with small logits mismatch)
    
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
    ) -> None:
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)

        # param equivalent to Conv2d weight, it will be reshaped in forward to fit Linear layer,
        # to be fully equivalent original implementation
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.unfold(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 1)
        hidden_state = F.linear(hidden_state, self.weight.view(self.out_channels, -1))
        return hidden_state


class MllamaTilePositionEmbedding(nn.Module):
    # originally TilePositionEmbedding
    def __init__(
        self,
        max_num_tiles: int,
        hidden_size: int,
        is_gated: bool = False,
    ):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.hidden_size = hidden_size
        self.is_gated = is_gated

        scale = hidden_size ** -0.5
        self.embedding = nn.Parameter(torch.randn(self.max_num_tiles, self.max_num_tiles, 1, self.hidden_size) / scale)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))


    @staticmethod
    def _dynamic_resize(embedding: torch.Tensor, num_tiles: int):
        embedding = embedding.permute(2, 3, 0, 1)

        embedding_new = F.interpolate(
            embedding,
            size=(num_tiles, num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # reshape the weights to the correct shape
        embedding_new = embedding_new.permute(2, 3, 0, 1)
        return embedding_new


    def forward(self, hidden_state: torch.Tensor, aspect_ratios: torch.Tensor, num_tiles: Optional[int] = None) -> torch.Tensor:
        
        if num_tiles is None:
            num_tiles = self.max_num_tiles
        elif num_tiles > self.max_num_tiles:
            self.embedding = self._dynamic_resize(self.embedding, num_tiles)

        batch_size = hidden_state.shape[0]
        out_tile_embedding = torch.zeros(
            batch_size, num_tiles, 1, self.hidden_size, device=hidden_state.device, dtype=hidden_state.dtype
        )
        for idx, aspect_ratio_i in enumerate(aspect_ratios):
            height, width = aspect_ratio_i
            tile_embedding_i = self.embedding[:height, :width]
            out_tile_embedding[idx, :height * width] = tile_embedding_i.reshape(height * width, 1, self.hidden_size)
        
        if self.is_gated:
            out_tile_embedding = out_tile_embedding * self.gate.tanh()

        hidden_state = hidden_state + out_tile_embedding
        return hidden_state


class MllamaVisionMLP(nn.Module):
    # originally ImageMllamaTextMLP

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.activation_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation_fn(hidden_state)
        
        # surprisingly this is not equivalent to self.fc2(hidden_state) for this model (??)
        # saving original implementation for logits full match
        hidden_state = F.linear(hidden_state, self.fc2.weight) + self.fc2.bias
    
        return hidden_state


class MllamaVisionSdpaAttention(nn.Module):
    # originally ImageAttention

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
    ):
        super().__init__()

        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        return output


class MllamaVisionEncoderLayer(nn.Module):
    # originally ImageTransformerBlock

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        is_gated: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.is_gated = is_gated
        self.intermediate_size = intermediate_size

        self.self_attn = MllamaVisionSdpaAttention(self.hidden_size, self.num_attention_heads)
        self.mlp = MllamaVisionMLP(self.hidden_size, self.intermediate_size)

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)

        if self.is_gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        # Self Attention
        residual = hidden_state
        hidden_state = self.layer_norm1(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)
        gate_attn = 1 if not self.is_gated else self.gate_attn.tanh()
        hidden_state = residual + gate_attn * hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.layer_norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        gate_ffn = 1 if not self.is_gated else self.gate_ffn.tanh()
        hidden_state = residual + gate_ffn * hidden_state

        return hidden_state


class MllamaVisionEncoder(nn.Module):
    # originally ImageTransformer

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        is_gated: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.is_gated = is_gated
        self.intermediate_size = 4 * hidden_size

        self.layers = nn.ModuleList(
            [
                MllamaVisionEncoderLayer(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    intermediate_size=self.intermediate_size,
                    is_gated=is_gated,
                )
                for _ in range(self.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediate: Optional[List[int]] = None,
    ):
        intermediate = []

        for idx, layer in enumerate(self.layers):

            if return_intermediate is not None and idx in return_intermediate:
                intermediate.append(hidden_state)
            
            hidden_state = layer(hidden_state, attention_mask=attention_mask)

        if return_intermediate is not None:
            intermediate = torch.stack(intermediate, dim=-1)
            return hidden_state, intermediate
        
        return hidden_state


class MllamaVisionTransformer(nn.Module):
    # originally VisionEncoder

    def __init__(
        self,
        max_num_tiles: int,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (14, 14),
        hidden_size: int = 1280,
        num_layers: int = 32,
        heads: int = 16,
        in_channels: int = 3,
        n_global_layers: int = 2,
        return_intermediate: Optional[List[int]] = None,
    ):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.return_intermediate = return_intermediate
        self.image_size = to_tuple(image_size)
        self.patch_size = to_tuple(patch_size)
        self.num_patches_height = self.image_size[0] // self.patch_size[0]
        self.num_patches_width = self.image_size[1] // self.patch_size[1]
        self.num_patches = self.num_patches_height * self.num_patches_width + 1
        self.scale = hidden_size ** -0.5

        # patch embedding
        self.patch_embedding = MllamaPatchEmbedding(
            in_channels=self.in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # class embedding
        self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))

        # positional embeddings and gate
        positional_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.positional_embedding = nn.Parameter(self.scale * positional_embedding)

        gated_positional_embedding = torch.randn(self.max_num_tiles, self.max_num_tiles, self.num_patches, self.hidden_size)
        self.gated_positional_embedding = nn.Parameter(self.scale * gated_positional_embedding)
        self.gated_positional_embedding_gate = nn.Parameter(torch.zeros(1))

        # pre and post tile position embedding
        self.pre_tile_pos_embed = MllamaTilePositionEmbedding(
            max_num_tiles=max_num_tiles,
            hidden_size=self.hidden_size,
            is_gated=True,
        )
        self.post_tile_pos_embed = MllamaTilePositionEmbedding(
            max_num_tiles=max_num_tiles,
            hidden_size=self.hidden_size,
            is_gated=True,
        )

        # layer norms
        self.ln_post = nn.LayerNorm(self.hidden_size)
        self.ln_pre = nn.LayerNorm(self.hidden_size)

        # encoders
        self.transformer = MllamaVisionEncoder(self.hidden_size, num_layers, heads, is_gated=False)
        self.global_transformer = MllamaVisionEncoder(self.hidden_size, n_global_layers, heads, is_gated=True)

    def apply_positional_embedding(self, hidden_state: torch.Tensor, aspect_ratios: torch.Tensor) -> torch.Tensor:
        
        bsz, num_chunks, num_tokens, dim = hidden_state.shape
        hidden_state = hidden_state.view(bsz * num_chunks, num_tokens, dim)

        # apply regular positional embedding with gate
        gate = 1 - self.gated_positional_embedding_gate.tanh()
        hidden_state = hidden_state + gate * self.positional_embedding

        hidden_state = hidden_state.view(bsz, num_chunks, num_tokens, dim)
        
        # apply gated positional embedding with gate
        for idx, (aspect_ratio_h, aspect_ratio_w) in enumerate(aspect_ratios):
            num_tiles = aspect_ratio_h * aspect_ratio_w
            gated_positional_embedding = self.gated_positional_embedding[:aspect_ratio_h, :aspect_ratio_w]
            embedding_height, embedding_width = gated_positional_embedding.shape[2:]
            gated_positional_embedding = gated_positional_embedding.reshape(num_tiles, embedding_height, embedding_width)
            gate = self.gated_positional_embedding_gate.tanh()
            gated_positional_embedding_with_gate = gate * gated_positional_embedding
            hidden_state[idx, :num_tiles] += gated_positional_embedding_with_gate

        return hidden_state

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def forward(self, pixel_values: torch.Tensor, aspect_ratios: torch.Tensor) -> torch.Tensor:

        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = pixel_values.shape

        pixel_values = pixel_values.reshape(batch_size * num_concurrent_media * num_tiles, num_channels, height, width)
        aspect_ratios = aspect_ratios.reshape(batch_size * num_concurrent_media, 2)

        # patch embedding
        hidden_state = self.patch_embedding(pixel_values)

        # tile embeddings
        _, num_tokens, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, -1, dim)
        hidden_state = self.pre_tile_pos_embed(hidden_state, aspect_ratios)

        # apply cls token
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media * num_tiles, num_tokens, dim)
        hidden_state = self.apply_class_embedding(hidden_state)
        num_tokens += 1

        # apply position embeddings
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_tokens, dim)
        hidden_state = self.apply_positional_embedding(hidden_state, aspect_ratios)

        # apply encoder
        hidden_state = self.ln_pre(hidden_state)
        hidden_state, num_pad_tokens = expand_num_tokens_to_mult8(hidden_state)
        attention_mask = build_encoder_attention_mask(hidden_state, aspect_ratios, num_tokens, num_tiles, 1)
        hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
        hidden_state, intermediate_hidden_state = self.transformer(
            hidden_state, return_intermediate=self.return_intermediate, attention_mask=attention_mask
        )

        # apply global encoder
        hidden_state = self.ln_post(hidden_state)
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_tokens + num_pad_tokens, dim)
        hidden_state = self.post_tile_pos_embed(hidden_state, aspect_ratios)
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles * (num_tokens + num_pad_tokens), dim)
        hidden_state = self.global_transformer(hidden_state, attention_mask=attention_mask)
        hidden_state = hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_tokens + num_pad_tokens, dim)
        hidden_state = contract_num_tokens_from_mult8(hidden_state, num_pad_tokens)

        # adding intermediate layer outputs
        hidden_state = hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_tokens, dim)
        intermediate_hidden_state = intermediate_hidden_state.reshape(batch_size * num_concurrent_media, num_tiles, num_tokens + num_pad_tokens, -1)
        intermediate_hidden_state = contract_num_tokens_from_mult8(intermediate_hidden_state, num_pad_tokens)
        intermediate_hidden_state = intermediate_hidden_state.reshape(batch_size, num_concurrent_media, num_tiles, num_tokens, -1)
        hidden_state = torch.cat([hidden_state, intermediate_hidden_state], dim=-1)

        return hidden_state


class MllamaTextSdpaAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, config: MllamaCrossAttentionTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.n_kv_heads = config.n_kv_heads

        self.n_local_heads = config.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len

        self.q_proj = nn.Linear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.n_heads * self.head_dim,
            config.dim,
            bias=False,

        )
        self.n_heads = config.n_heads
        
    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        cache_shape = (
            max_batch_size,
            self.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )
        device = next(self.parameters()).device
        self.register_buffer(
            "key_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: torch.LongTensor,
    ):

        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, seq_length, _ = query.shape

        query = query.view(batch_size, seq_length, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, key.shape[1], self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, value.shape[1], self.n_local_kv_heads, self.head_dim)

        query, key = apply_rotary_emb(query, key, freqs_cis)

        self.key_cache[:batch_size, position_ids, ...] = key
        self.value_cache[:batch_size, position_ids, ...] = value

        # TODO: we can avoid slicing on first dimension by always padding to max_batch_size()
        key = self.key_cache[:batch_size, ...]
        value = self.value_cache[:batch_size, ...]

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        key = key.repeat_interleave(self.n_rep, dim=1)
        value = value.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
        out = self.o_proj(attn_output)

        return out


class MllamaTextMLP(nn.Module):
    # originally FeedForward

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        activation = "silu",
    ):
        """
        Initialize the MllamaTextMLP module.
        Args:
            hidden_size (int): Input dimension.
            intermediate_size (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
        Attributes:
            gate_proj (Linear): Linear transformation for the first layer.
            down_proj (Linear): Linear transformation for the second layer.
            up_proj (Linear): Linear transformation for the third layer.
        """
        super().__init__()

        # custom dim factor multiplier
        intermediate_size = int(2 / 3 * intermediate_size)
        if ffn_dim_multiplier is not None:
            intermediate_size = int(ffn_dim_multiplier * intermediate_size)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.activation_fn = ACT2FN[activation]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        
        hidden_state_gate = self.gate_proj(hidden_state)
        hidden_state_up = self.up_proj(hidden_state)
        hidden_state = self.activation_fn(hidden_state_gate) * hidden_state_up
        hidden_state = self.down_proj(hidden_state)

        return hidden_state


class MllamaTextEncoderLayer(nn.Module):
    # originally TransformerBlock

    def __init__(self, config: PretrainedConfig, layer_id: int):
        """
        Initialize a MllamaTextEncoderLayer.
        Args:
            config (PretrainedConfig): configuration object for the layer.
            layer_id (int): Identifier for the layer.
        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            mlp (MllamaTextMLP): MllamaTextMLP module.
            layer_id (int): Identifier for the layer.
            input_layernorm (RMSNorm): Layer normalization for attention output.
            post_attention_layernorm (RMSNorm): Layer normalization for feedforward output.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.self_attn = MllamaTextSdpaAttention(config)
        self.mlp = MllamaTextMLP(
            hidden_size=self.dim,
            intermediate_size=4 * self.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        self.self_attn.setup_cache(max_batch_size, dtype)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the MllamaTextEncoderLayer.
        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.
        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        
        # Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(
            hidden_state=hidden_state,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state


class MllamaSdpaCrossAttention(nn.Module):
    # originally CrossAttention

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()

        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )

        # local heads
        self.n_local_heads = self.num_heads
        self.n_local_kv_heads = self.num_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

    def compute_cross_attention_key_value(self, hidden_state: torch.Tensor) -> torch.Tensor:
        
        batch_size = hidden_state.shape[0]
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        seq_length = key.shape[1]
        key = key.view(batch_size, seq_length, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.n_local_kv_heads, self.head_dim)

        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key = key.repeat_interleave(self.n_rep, dim=1)
        value = value.repeat_interleave(self.n_rep, dim=1)

        key = self.k_norm(key)

        return torch.stack([key, value])


    def forward(
        self,
        hidden_state: torch.Tensor,
        cross_attention_key_value: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        bsz, seq_length, _ = hidden_state.shape
        key, value = cross_attention_key_value

        query = self.q_proj(hidden_state)
        query = query.view(bsz, seq_length, self.n_local_heads, self.head_dim)

        query = self.q_norm(query)
        query = query.transpose(1, 2)

        # FIXME shape issue here (?)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=cross_attention_mask, dropout_p=0.0
        )
        attn_output = attn_output * full_text_row_masked_out_mask
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_length, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class MllamaCrossAttentionLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""
    # originally CrossAttentionTransformerBlock

    def __init__(
        self,
        config: MllamaCrossAttentionTextConfig,
        layer_id: int,
        no_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.norm_eps = config.norm_eps
        self.ffn_dim_multiplier = config.ffn_dim_multiplier
        self.multiple_of = config.multiple_of

        self.self_attn = MllamaSdpaCrossAttention(
            hidden_size=self.dim,
            head_dim=self.head_dim,
            num_heads=self.n_heads,
            num_kv_heads=self.n_kv_heads,
            norm_eps=self.norm_eps,
        )

        self.input_layernorm = RMSNorm(
            self.dim,
            eps=self.norm_eps,
        )
        self.gate_attn = torch.nn.Parameter(torch.zeros(1))

        self.mlp = MllamaTextMLP(
            hidden_size=self.dim,
            intermediate_size=4 * self.dim,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
            multiple_of=self.multiple_of,
        )
        self.post_attention_layernorm = RMSNorm(
            self.dim,
            eps=self.norm_eps,
        )
        self.ffn_gate = torch.nn.Parameter(torch.zeros(1))

        self.no_ffn = no_ffn

    def compute_cross_attention_key_value(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.self_attn.compute_cross_attention_key_value(hidden_state)

    def forward(
        self,
        hidden_state: torch.Tensor,
        cross_attention_key_value: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(
            hidden_state=hidden_state,
            cross_attention_mask=cross_attention_mask,
            cross_attention_key_value=cross_attention_key_value,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        )

        hidden_state = residual + self.gate_attn.tanh() * hidden_state

        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = full_text_row_masked_out_mask[:, 0] * hidden_state  # type: ignore
        hidden_state = residual + self.ffn_gate.tanh() * hidden_state * float(not self.no_ffn)
        return hidden_state


class MllamaCrossAttentionVisionModel(torch.nn.Module):
    def __init__(self, config: MllamaCrossAttentionVisionConfig):
        super().__init__()
        return_intermediate = config.return_intermediate
        self.vision_input_dim = config.vision_input_dim
        self.image_res = config.vision_chunk_size
        self.max_num_chunks = config.vision_max_num_chunks
        if return_intermediate is not None:
            return_intermediate = [int(l) for l in return_intermediate.split(",")]
            self.vision_input_dim = (
                len(return_intermediate) + 1
            ) * self.vision_input_dim
        self.patch_size = config.patch_size
        self.vision_encoder = MllamaVisionTransformer(
            max_num_tiles=config.max_num_tiles,
            image_size=self.image_res,
            patch_size=self.patch_size,
            n_global_layers=config.global_vision_layers,
            return_intermediate=return_intermediate,
        )
        # vision token projection
        self.vision_projection = nn.Linear(
            self.vision_input_dim,
            config.projection_dim,
            #! originally bias=True, but bias was not used in original forward pass
            bias=False,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        aspect_ratios: torch.Tensor,
    ) -> torch.Tensor:
        
        pixel_values = pixel_values.to(dtype=torch.bfloat16)
        hidden_state = self.vision_encoder(pixel_values, aspect_ratios)
        hidden_state = self.vision_projection(hidden_state)

        return hidden_state


class MllamaCrossAttentionTextModel(PreTrainedModel):
    INFERENCE_IMAGE_TOKEN_ID = 128010

    def __init__(self, config: MllamaCrossAttentionTextConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.rope_theta = config.rope_theta
        self.use_scaled_rope = config.use_scaled_rope
        self.max_seq_len = config.max_seq_len
        self.vision_num_cross_attention_layers = config.vision_num_cross_attention_layers
        self.pos_embeddings = None
        self.n_llama_layers = self.n_layers
        self.model_dim = self.dim
        self.fusion_schedule = self._init_fusion_schedule(
            self.vision_num_cross_attention_layers
        )
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.learnable_embedding = nn.Embedding(8, self.dim)
        self.num_frozen_embeddings = self.embed_tokens.num_embeddings
        self._thresh = self.num_frozen_embeddings - 1

        # self-attention layers
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            layer = MllamaTextEncoderLayer(config=config, layer_id=i)
            self.layers.append(layer)

        # cross-attention layers
        self.cross_attention_layers = torch.nn.ModuleList()
        for i in range(self.vision_num_cross_attention_layers):
            cross_attention_layer = MllamaCrossAttentionLayer(config, layer_id=i + self.n_layers)
            self.cross_attention_layers.append(cross_attention_layer)

        freqs_cis = precompute_freqs_cis(
            self.dim // self.n_heads,
            self.max_seq_len * 2,
            self.rope_theta,
            self.use_scaled_rope,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.cache_is_setup = False
        self.max_seq_len = self.max_seq_len

    def _init_fusion_schedule(
        self,
        num_layers: int,
    ) -> List[int]:
        llama_layers = list(range(self.n_llama_layers))

        # uniformly spread the layers
        k = math.ceil(len(llama_layers) / num_layers)
        return llama_layers[::-1][::k][:num_layers][::-1]

    def setup_cache(self, max_batch_size: int, dtype=torch.bfloat16):
        ones = torch.ones(
            (self.max_seq_len, self.max_seq_len),
            dtype=torch.bool,
            device=self.device,
        )
        mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask_cache", mask, persistent=False)

        for layer in self.layers:
            layer.setup_cache(max_batch_size, dtype=dtype)

        self.cache_is_setup = True

    def get_partially_trainable_embedding(self, x):
        xz = torch.zeros_like(x, device=x.device)
        oz = torch.ones_like(x, device=x.device)
        x_orig = torch.minimum(x, torch.tensor(self._thresh, device=x.device))
        x_new = (
            torch.maximum(x, torch.tensor(self._thresh + 1, device=x.device))
            - self.num_frozen_embeddings
        )

        mask_orig = torch.where(x >= self.num_frozen_embeddings, xz, oz).unsqueeze(-1)
        mask_new = torch.where(x < self.num_frozen_embeddings, xz, oz).unsqueeze(-1)

        x_orig = self.embed_tokens(x_orig)
        x_new = self.learnable_embedding(x_new).type_as(x_orig)
        return x_orig * mask_orig.type_as(x_orig) + x_new * mask_new.type_as(x_new)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_state: torch.Tensor,
        cross_attention_key_value: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
    ):
        if not self.cache_is_setup:
            raise RuntimeError("Please set up cache before calling forward")

        attention_mask = self.mask_cache[:, :, position_ids]
        freqs_cis = self.freqs_cis[position_ids]

        for idx, layer in enumerate(self.layers):

            if idx in self.fusion_schedule:
                cross_attention_layer_idx = self.fusion_schedule.index(idx)
                cross_attention_layer = self.cross_attention_layers[cross_attention_layer_idx]
                layer_cross_attention_key_value = cross_attention_key_value[cross_attention_layer_idx]
                hidden_state = cross_attention_layer(
                    hidden_state=hidden_state,
                    cross_attention_key_value=layer_cross_attention_key_value,
                    cross_attention_mask=cross_attention_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                )

            hidden_state = layer(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                position_ids=position_ids,
            )

        hidden_state = self.norm(hidden_state)

        return hidden_state


class MllamaPreTrainedModel(PreTrainedModel):
    config_class = MllamaConfig
    base_model_prefix = "model"
    _no_split_modules = []


class MllamaModel(MllamaPreTrainedModel):
    def __init__(self, config: MllamaConfig):
        super().__init__(config)
        self.vision_model = MllamaCrossAttentionVisionModel(config.vision_config)
        self.language_model = MllamaCrossAttentionTextModel(config.text_config)
        self.post_init()

MLLAMA_START_DOCSTRING = "" # TODO add docstring to MLLAMA start and other classes

@add_start_docstrings(
    """The MLLAMA model which consists of a vision backbone and a language model.""",
    MLLAMA_START_DOCSTRING,
)
class MllamaForConditionalGeneration(MllamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MllamaModel(config)

        self.lm_head = nn.Linear(
            config.text_config.dim, config.text_config.vocab_size, bias=False
        )
        self.vision_max_num_chunks = config.vision_config.vision_max_num_chunks
        self.vision_chunk_size = config.vision_config.vision_chunk_size 

        self.post_init()

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        self.model.language_model.setup_cache(max_batch_size, dtype)

    def compute_vision_cross_attention_key_value_and_masks(
        self,
        pixel_values: torch.Tensor,  # shape: [batch_size, num_images, num_tiles, channels, height, width]
        aspect_ratios: torch.Tensor, # shape: [batch_size, num_images, 2]
        num_tiles: List[List[int]],  # shape: [batch_size, num_images]; num tiles per image
        cross_attention_token_mask: List[List[List[int]]],  # shape: [batch_size, num_images, 2]; start token, end token
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # TODO: make sure vision transformer with pure text also works
        # Skip for now
        skip_vision_encoder = False
        if skip_vision_encoder:  # create dummy vision tokens
            max_num_images = 0
            num_tiles = [[self.vision_max_num_chunks] for _ in pixel_values]
            vision_tokens = torch.zeros(
                (
                    len(pixel_values),  # batch size 
                    max_num_images,   # most likely 1 but take it from model_inputs
                    self.vision_max_num_chunks,
                    self.model.vision_model.vision_encoder.num_patches,
                    self.config.vision_config.projection_dim,
                ),
                device=self.device,
            )

        # get vision tokens from vision model
        vision_tokens = self.model.vision_model(pixel_values, aspect_ratios)

        # compute key value pairs for cross-attention with vision tokens
        cross_attention_key_value = []
        batch_size, *_, dim = vision_tokens.shape
        vision_tokens_flattened = vision_tokens.view(batch_size, -1, dim)
        for layer in self.model.language_model.cross_attention_layers:
            layer_cross_attention_key_value = layer.compute_cross_attention_key_value(vision_tokens_flattened)
            cross_attention_key_value.append(layer_cross_attention_key_value)
        cross_attention_key_value = torch.stack(cross_attention_key_value)

        # create masks for cross-attention based on image token locations
        cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
            cross_attention_token_mask=cross_attention_token_mask,
            num_tiles=num_tiles,
            total_len=total_len,
            max_num_tiles=self.vision_max_num_chunks,
            device=self.device,
            dtype=self.dtype,
        )

        cross_attention_mask, full_text_row_masked_out_mask = prepare_cross_attention_mask(
            cross_attention_mask=cross_attention_mask,
            vision_tokens=vision_tokens,
        )

        return cross_attention_key_value, cross_attention_mask, full_text_row_masked_out_mask

    def forward(
        self,
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
        cross_attention_key_value: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        input_ids = input_ids[:, position_ids]
        embeddings = self.model.language_model.get_partially_trainable_embedding(input_ids)

        logits = self.model.language_model.forward(
            position_ids=position_ids,
            hidden_state=embeddings,
            cross_attention_key_value=cross_attention_key_value,
            cross_attention_mask=cross_attention_mask[:, :, position_ids],
            full_text_row_masked_out_mask=full_text_row_masked_out_mask[:, :, position_ids],
        )

        logits = self.lm_head(logits)

        return logits


    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    # likely not needed here. 
    # TODO do we support multi-image per message at arbitrary positions? if yes, we can keep the merge method
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids
    # TODO add return docstrings
    # TODO add return type (MllamaCausalLMOutputWithPast)
    #@add_start_docstrings_to_model_forward(MLLAMA_INPUTS_DOCSTRING)
    #@replace_return_docstrings(output_type=MllamaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def llava_forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

        >>> model = MllamaForConditionalGeneration.from_pretrained("mllama-hf/mllama-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("mllama-hf/mllama-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Mllama + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MllamaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.model.language_model._reorder_cache(*args, **kwargs)
