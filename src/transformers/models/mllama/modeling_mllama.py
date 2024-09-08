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
import collections
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache, StaticCache
from ...configuration_utils import PretrainedConfig
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...utils import (
    add_start_docstrings,
    logging,
)
from .configuration_mllama import MllamaConfig, MllamaCrossAttentionTextConfig, MllamaCrossAttentionVisionConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MllamaConfig"




def get_negative_inf_value(dtype):
    return torch.finfo(dtype).min


def to_tuple(x) -> Tuple[int, int]:
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

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
    attention_mask: torch.Tensor,
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

    for batch_idx, (mask_i, num_tiles_i, text_attn_mask) in enumerate(zip(cross_attention_token_mask, num_tiles, attention_mask)):
        for mask_idx, (token_locations, mask_num_chunks) in enumerate(zip(mask_i, num_tiles_i)):
            if len(token_locations) == 2:
                start, end = token_locations
                end = min(end, total_len)
                if end == -1:
                    end = total_len
                out_masks[batch_idx, start:end, mask_idx, :mask_num_chunks].fill_(0.0)
        curr_attn_mask = text_attn_mask[-(out_masks.shape[1] - 100): ]
        masked_locations = torch.where(curr_attn_mask == 0)[0]
        out_masks[batch_idx, masked_locations]  = inf_value # text_attn_mask is 1D of total_len-100 length

    return out_masks


def prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
) -> Tuple[Tensor, Tensor]:

    batch_size, text_total_length, _, _ = cross_attention_mask.shape # 2, 109, 1, 4

    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=2)
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
    # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    if x.shape[0] == 1: # if batch-size = 1
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else: # if batchsize > 1
        shape = [*freqs_cis.shape[:2], 1, freqs_cis.shape[-1]]
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


@dataclass
class MllamaOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: List[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attention_key_value: Optional[torch.FloatTensor] = None


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
        self.layer_idx = layer_idx

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


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: torch.LongTensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
    ):

        output_attentions = False # SDPA doesn't support it

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        batch_size, seq_length, _ = query.shape

        query = query.view(batch_size, seq_length, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, key.shape[1], self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, value.shape[1], self.n_local_kv_heads, self.head_dim)

        query, key = apply_rotary_emb(query, key, freqs_cis)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key, value = past_key_value.update(key, value, self.layer_idx, cache_kwargs)

        key = key.repeat_interleave(self.n_rep, dim=1)
        value = value.repeat_interleave(self.n_rep, dim=1)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=causal_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_state_gate = self.gate_proj(hidden_states)
        hidden_state_up = self.up_proj(hidden_states)
        hidden_states = self.activation_fn(hidden_state_gate) * hidden_state_up
        hidden_states = self.down_proj(hidden_states)

        return hidden_states


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
        self.self_attn = MllamaTextSdpaAttention(config, layer_id)
        self.mlp = MllamaTextMLP(
            hidden_size=self.dim,
            intermediate_size=4 * self.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: torch.LongTensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Feed forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


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
        hidden_states: torch.Tensor,
        cross_attention_key_value: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
    ) -> torch.Tensor:

        bsz, seq_length, _ = hidden_states.shape
        key, value = cross_attention_key_value

        query = self.q_proj(hidden_states)
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
        hidden_states: torch.Tensor,
        cross_attention_key_value: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cross_attention_mask=cross_attention_mask,
            cross_attention_key_value=cross_attention_key_value,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        )

        hidden_states = residual + self.gate_attn.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.ffn_gate.tanh() * hidden_states * float(not self.no_ffn)
        return hidden_states


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

    def _init_fusion_schedule(
        self,
        num_layers: int,
    ) -> List[int]:
        llama_layers = list(range(self.n_llama_layers))

        # uniformly spread the layers
        k = math.ceil(len(llama_layers) / num_layers)
        return llama_layers[::-1][::k][:num_layers][::-1]

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
        hidden_states: torch.Tensor,
        cross_attention_key_value: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        cache_position=None,
    ):
        freqs_cis = self.freqs_cis[position_ids]

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if idx in self.fusion_schedule:
                cross_attention_layer_idx = self.fusion_schedule.index(idx)
                cross_attention_layer = self.cross_attention_layers[cross_attention_layer_idx]
                layer_cross_attention_key_value = cross_attention_key_value[cross_attention_layer_idx]
                hidden_states = cross_attention_layer(
                    hidden_states=hidden_states,
                    cross_attention_key_value=layer_cross_attention_key_value,
                    cross_attention_mask=cross_attention_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                )

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                position_ids=position_ids,
                use_cache=use_cache,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MllamaPreTrainedModel(PreTrainedModel):
    config_class = MllamaConfig
    base_model_prefix = "model"
    _no_split_modules = []
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_sdpa = True
    _supports_quantized_cache = True


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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None, # shape: [batch_size, num_images, num_tiles, channels, height, width]
        aspect_ratios: Optional[torch.Tensor] = None, # shape: [batch_size, num_images, 2]
        num_tiles: Optional[List[List[int]]] = None,  # shape: [batch_size, num_images]; num tiles per image
        cross_attention_token_mask: Optional[List[List[List[int]]]] = None,  # shape: [batch_size, num_images, 2]; start token, end token
        cross_attention_key_value: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[torch.Tensor] = None,

        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> MllamaOutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_key_value is not None:
            raise ValueError(
                "`pixel_values` and `cross_attention_key_value` cannot be provided simultaneously"
            )

        if cross_attention_token_mask is None or num_tiles is None:
            raise ValueError("`cross_attention_token_mask` and `num_tiles` must be provided!")

        if pixel_values is not None:
            if aspect_ratios is None:
                raise ValueError("`aspect_ratios` must be provided if `pixel_values` is provided")

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

        # @raushan: can we prepare the dense attn in processing code so we dont't have to convert here?
        # or probably this is somewhat similar to `update_causal_mask`, have to see where and how it's used
        # create masks for cross-attention based on image token locations
        cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
            cross_attention_token_mask=cross_attention_token_mask,
            attention_mask=attention_mask,
            num_tiles=num_tiles,
            total_len=input_ids.shape[-1] + 100,
            max_num_tiles=self.vision_max_num_chunks,
            device=self.device,
            dtype=self.dtype,
        )

        print(cross_attention_mask.shape, cross_attention_token_mask)
        cross_attention_mask, full_text_row_masked_out_mask = prepare_cross_attention_mask(
            cross_attention_mask=cross_attention_mask,
            num_vision_tokens=self.model.vision_model.vision_encoder.num_patches,
        )

        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.get_partially_trainable_embedding(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # temp workaround for rope, for some reason it expects a 1d tensor
        position_ids = position_ids.squeeze(0)

        hidden_states = inputs_embeds

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        outputs = self.model.language_model(
            position_ids=position_ids,
            hidden_states=hidden_states,
            cross_attention_key_value=cross_attention_key_value,
            cross_attention_mask=cross_attention_mask[:, :, :position_ids.shape[-1]],
            full_text_row_masked_out_mask=full_text_row_masked_out_mask[:, :, :position_ids.shape[-1]],
            attention_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # TODO: fix and test loss calculation and backward
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
            output = (logits, cross_attention_key_value, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (loss,) + output if loss is not None else output

        return MllamaOutput(
            loss=loss,
            logits=logits,
            cross_attention_key_value=cross_attention_key_value,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        using_static_cache = isinstance(past_key_values, StaticCache)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratios=None,
        num_tiles=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        # TODO @raushan: Uncomment when cache is added
        # if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        #     if model_inputs["inputs_embeds"] is not None:
        #         batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
        #         device = model_inputs["inputs_embeds"].device
        #     else:
        #         batch_size, sequence_length = model_inputs["input_ids"].shape
        #         device = model_inputs["input_ids"].device
#
        #     dtype = self.lm_head.weight.dtype
        #     min_dtype = torch.finfo(dtype).min
#
        #     attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        #         attention_mask,
        #         sequence_length=sequence_length,
        #         target_length=past_key_values.get_max_length(),
        #         dtype=dtype,
        #         device=device,
        #         min_dtype=min_dtype,
        #         cache_position=cache_position,
        #         batch_size=batch_size,
        #     )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "num_tiles": num_tiles,
                "cross_attention_token_mask": kwargs.get("cross_attention_token_mask"),
            }
        )

        # We need pixel values to calculate `cross_attention_key_value` in modeling, if it isn't available
        # Otherwise it was pre-computed earlier, so we can simply pass it further
        if kwargs.get("cross_attention_key_value", None) is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratios"] = aspect_ratios
        else:
            model_inputs["cross_attention_key_value"] = kwargs["cross_attention_key_value"]

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # Get the precomputed image_hidden_states
        model_kwargs["cross_attention_key_value"] = outputs.cross_attention_key_value
        return model_kwargs
