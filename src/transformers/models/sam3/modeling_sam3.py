# coding=utf-8
# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
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


import collections.abc
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

from transformers import CLIPTextModelWithProjection

from ... import initialization as init
from ...activations import ACT2FN
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import auto_docstring
from ...utils.generic import TransformersKwargs, check_model_inputs
from ..auto import AutoModel
from .configuration_sam3 import (
    Sam3Config,
    Sam3DETRDecoderConfig,
    Sam3DETREncoderConfig,
    Sam3GeometryEncoderConfig,
    Sam3MaskDecoderConfig,
    Sam3VisionConfig,
    Sam3ViTConfig,
)


@dataclass
@auto_docstring
class Sam3VisionEncoderOutput(ModelOutput):
    r"""
    fpn_hidden_states (`tuple[torch.FloatTensor]`):
        Tuple of multi-level FPN feature maps.
    fpn_position_encoding (`tuple[torch.FloatTensor]`):
        Tuple of position encodings for each FPN level.
    hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of hidden states from all ViT layers.
    attentions (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of attention weights from all ViT layers.
    """

    last_hidden_state: torch.FloatTensor = None
    fpn_hidden_states: tuple[torch.FloatTensor, ...] = None
    fpn_position_encoding: tuple[torch.FloatTensor, ...] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring
class Sam3GeometryEncoderOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_prompts, hidden_size)`):
        Encoded geometry prompt features (boxes).
    attention_mask (`torch.BoolTensor` of shape `(batch_size, num_prompts)`, *optional*):
        Attention mask for geometry prompts where True indicates valid positions and False indicates padding.
    """

    last_hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.BoolTensor] = None


@dataclass
@auto_docstring
class Sam3DETREncoderOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Encoded vision features (flattened from multi-level features).
    pos_embeds_flattened (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Flattened position embeddings for the vision features.
    text_features (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`, *optional*):
        Text features (may be pooled after encoder processing).
    spatial_shapes (`torch.LongTensor` of shape `(num_levels, 2)`, *optional*):
        Spatial shapes (height, width) for each feature pyramid level.
    hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of hidden states from all encoder layers.
    attentions (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of attention weights from all encoder layers.
    """

    last_hidden_state: torch.FloatTensor = None
    pos_embeds_flattened: Optional[torch.FloatTensor] = None
    text_features: Optional[torch.FloatTensor] = None
    spatial_shapes: Optional[torch.LongTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring
class Sam3DETRDecoderOutput(ModelOutput):
    r"""
    intermediate_hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, num_queries, hidden_size)`):
        Decoder hidden states from all layers.
    reference_boxes (`torch.FloatTensor` of shape `(num_layers, batch_size, num_queries, 4)`):
        Predicted reference boxes from all decoder layers in (cx, cy, w, h) format.
    presence_logits (`torch.FloatTensor` of shape `(num_layers, batch_size)`, *optional*):
        Presence logits from all decoder layers (None if using instance queries).
    hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of hidden states from all decoder layers.
    attentions (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of attention weights from all decoder layers (self-attention and cross-attention).
    """

    intermediate_hidden_states: torch.FloatTensor = None
    reference_boxes: torch.FloatTensor = None
    presence_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring
class Sam3MaskDecoderOutput(ModelOutput):
    r"""
    pred_masks (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`):
        Predicted segmentation masks for each query.
    semantic_seg (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
        Semantic segmentation output.
    attentions (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of attention weights from mask decoder cross-attention layers.
    """

    pred_masks: torch.FloatTensor = None
    semantic_seg: Optional[torch.FloatTensor] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring
class Sam3ImageSegmentationOutput(ModelOutput):
    r"""
    pred_masks (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`):
        Predicted segmentation masks for each query.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Predicted bounding boxes in (x1, y1, x2, y2) format.
    pred_logits (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
        Classification confidence scores for each query, computed via dot product between
        decoder query features and text features.
    presence_logits (`torch.FloatTensor` of shape `(batch_size, 1)`, *optional*):
        Presence logits from the DETR decoder presence token (last layer only). These indicate whether objects
        are present in the scene. Can be used to compute final scores by multiplying with pred_logits:
        `final_scores = pred_logits.sigmoid() * presence_logits.sigmoid()`.
    semantic_seg (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
        Semantic segmentation output.
    decoder_hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of hidden states from all DETR decoder layers. Each tensor has shape `(batch_size, num_queries, hidden_size)`.
    decoder_reference_boxes (`torch.FloatTensor` of shape `(num_layers, batch_size, num_queries, 4)`, *optional*):
        Reference boxes from all DETR decoder layers.
    encoder_hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of hidden states from all DETR encoder layers.
    vision_hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Tuple of hidden states from all vision encoder (ViT) layers.
    vision_attentions (`tuple[torch.FloatTensor]`, *optional*):
        Attention weights from vision encoder (ViT) layers.
    detr_encoder_attentions (`tuple[torch.FloatTensor]`, *optional*):
        Attention weights from DETR encoder layers.
    detr_decoder_attentions (`tuple[torch.FloatTensor]`, *optional*):
        Attention weights from DETR decoder layers (self-attention and cross-attention).
    mask_decoder_attentions (`tuple[torch.FloatTensor]`, *optional*):
        Attention weights from mask decoder layers.
    """

    pred_masks: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    pred_logits: Optional[torch.FloatTensor] = None
    presence_logits: Optional[torch.FloatTensor] = None
    semantic_seg: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    decoder_reference_boxes: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[tuple[torch.FloatTensor]] = None
    detr_encoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    detr_decoder_attentions: Optional[tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[tuple[torch.FloatTensor]] = None


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """The inverse function for sigmoid activation function."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """
    Concatenates two right-padded sequences, such that the resulting sequence
    is contiguous and also right-padded.

    Tensors are batch-first, masks are batch-first with True=valid, False=padding.

    Args:
        seq1: A tensor of shape (batch_size, seq1_length, hidden_size).
        mask1: A tensor of shape (batch_size, seq1_length) with True=valid, False=padding.
        seq2: A tensor of shape (batch_size, seq2_length, hidden_size).
        mask2: A tensor of shape (batch_size, seq2_length) with True=valid, False=padding.
        return_index: If True, also returns the index of the ids of the element of seq2
            in the concatenated sequence. This can be used to retrieve the elements of seq2.

    Returns:
        A tuple (concatenated_sequence, concatenated_mask) if return_index is False,
        otherwise (concatenated_sequence, concatenated_mask, index).
        The concatenated_mask uses True=valid, False=padding convention.
    """
    batch_size, seq1_length, hidden_size = seq1.shape
    batch_size2, seq2_length, hidden_size2 = seq2.shape

    assert batch_size == batch_size2 == mask1.size(0) == mask2.size(0)
    assert hidden_size == hidden_size2
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)

    actual_seq1_lengths = mask1.sum(dim=-1)
    actual_seq2_lengths = mask2.sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length

    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) < final_lengths[:, None]
    )

    concatenated_sequence = torch.zeros((batch_size, max_length, hidden_size), device=seq2.device, dtype=seq2.dtype)
    concatenated_sequence[:, :seq1_length, :] = seq1

    # Shift seq2 elements to start at the end of valid seq1
    index = torch.arange(seq2_length, device=seq2.device)[None].repeat(batch_size, 1)
    index = index + actual_seq1_lengths[:, None]

    # Scatter seq2 into the right positions
    concatenated_sequence = concatenated_sequence.scatter(1, index[:, :, None].expand(-1, -1, hidden_size), seq2)

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


def box_cxcywh_to_xyxy(x):
    """Convert boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class Sam3MLP(nn.Module):
    def __init__(self, config: Union[Sam3ViTConfig]):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Sam3Attention(nn.Module):
    """
    Multi-head attention.
    Handles standard [batch_size, seq_len, hidden_size] tensors.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, query_len, hidden_size]
            key: [batch_size, key_len, hidden_size]
            value: [batch_size, value_len, hidden_size]
            attention_mask: [batch_size, num_heads, query_len, key_len] or broadcastable

        Returns:
            Tuple of (output, attention_weights)
                output: [batch_size, query_len, hidden_size]
                attention_weights: [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]

        query = self.q_proj(query).view(batch_size, query_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key).view(batch_size, key_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value).view(batch_size, key_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, query_len, self.num_attention_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Sam3ViTRotaryEmbedding(nn.Module):
    """
    Vision Rotary Position Embedding for SAM3, following transformers library standards.
    Supports 2D (axial) rotary embeddings for spatial dimensions.
    """

    def __init__(self, config: Sam3ViTConfig, end_x: int, end_y: int, scale: float = 1.0):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        # Ensure even dimension for proper axial splitting
        if dim % 4 != 0:
            raise ValueError("Dimension must be divisible by 4 for axial RoPE")
        freqs = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

        flattened_indices = torch.arange(end_x * end_y, dtype=torch.long)
        x_positions = (flattened_indices % end_x) * scale
        y_positions = torch.div(flattened_indices, end_x, rounding_mode="floor") * scale
        freqs_x = torch.outer(x_positions, freqs).float()
        freqs_y = torch.outer(y_positions, freqs).float()
        inv_freq = torch.cat([freqs_x, freqs_y], dim=-1)
        inv_freq = inv_freq.repeat_interleave(2, dim=-1)
        # directly register the cos and sin embeddings as we have a fixed feature shape
        self.register_buffer("rope_embeddings_cos", inv_freq.cos(), persistent=False)
        self.register_buffer("rope_embeddings_sin", inv_freq.sin(), persistent=False)

    @torch.no_grad()
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        # As the feature map size is fixed for each stage, we can just return the pre-computed embeddings.
        return self.rope_embeddings_cos, self.rope_embeddings_sin


def rotate_pairwise(x):
    """
    pairwise rotation of the hidden dims of the input. Differerent from Llama Half-Tensor Rotation.

    This is an optimized version of the following more explicit implementation:
    ```python
    x_rotated = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    x_rotated[..., ::2] = -x[..., 1::2]
    x_rotated[..., 1::2] = x[..., ::2]
    return x_rotated
    ```
    """
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(start_dim=-2)


def apply_rotary_pos_emb_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors for self-attention.

    Args:
        q: Query tensor of shape (batch_size, num_windows, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch_size, num_windows, seq_len, num_heads, head_dim)
        cos: Cosine position embedding of shape (seq_len, head_dim)
        sin: Sine position embedding of shape (seq_len, head_dim)

    Returns:
        Rotated (q, k) tensors
    """
    q_embed = q.float()
    q_embed = (q_embed * cos) + (rotate_pairwise(q_embed) * sin)

    k_embed = k.float()
    k_embed = (k_embed * cos) + (rotate_pairwise(k_embed) * sin)

    return q_embed.type_as(q), k_embed.type_as(k)


class Sam3ViTRoPEAttention(nn.Module):
    """Self-attention with rotary position encoding."""

    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        batch_size, height, width, _ = hidden_states.shape
        seq_len = height * width
        new_shape = (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(*new_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(*new_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(*new_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_2d(query, key, cos=cos, sin=sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=self.is_causal,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, height, width, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Sam3ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: Sam3ViTConfig):
        super().__init__()
        image_size, patch_size = config.pretrain_image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class Sam3ViTEmbeddings(nn.Module):
    """
    Construct the patch embeddings and position embeddings for SAM3 ViT.

    Position embeddings are tiled (not interpolated) when resizing to match different input sizes.
    """

    def __init__(self, config: Sam3ViTConfig):
        super().__init__()

        self.patch_embeddings = Sam3ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches, config.hidden_size)
        )  # !Remove cls token in convert weights!

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.patch_size = config.patch_size

    def _tile_position_embeddings(
        self,
        position_embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Tile position embeddings to match target spatial dimensions.
        Args:
            position_embeddings: Shape [1, num_pretrain_patches, hidden_size]
            height: Target height in patches
            width: Target width in patches

        Returns:
            Shape [1, height * width, hidden_size]
        """
        pretrain_size = int(position_embeddings.shape[1] ** 0.5)

        # Skip tiling if sizes match (but always tile during tracing for consistent graph)
        if not torch.jit.is_tracing() and pretrain_size == height and pretrain_size == width:
            return position_embeddings.reshape(1, height * width, -1)

        # Tile position embeddings to match target spatial dimensions
        hidden_size = position_embeddings.shape[-1]
        pos_embed = position_embeddings.reshape(1, pretrain_size, pretrain_size, hidden_size).permute(0, 3, 1, 2)
        repeat_h = height // pretrain_size + 1
        repeat_w = width // pretrain_size + 1
        pos_embed = pos_embed.tile([1, 1, repeat_h, repeat_w])[:, :, :height, :width]
        return pos_embed.permute(0, 2, 3, 1).reshape(1, height * width, hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        embeddings = self.patch_embeddings(pixel_values)

        # Calculate spatial dimensions in patches
        height_patches = height // self.patch_size
        width_patches = width // self.patch_size

        position_embeddings = self._tile_position_embeddings(
            self.position_embeddings,
            height_patches,
            width_patches,
        )
        embeddings = embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


def window_partition(hidden_state, window_size):
    """
    Partition into non-overlapping windows with padding if needed.

    Args:
        hidden_state (`torch.Tensor`):
            Input tokens with [batch_size, height, width, num_channels].
        window_size (`int`):
            Window size.

    Returns:
        `tuple(torch.FloatTensor)` comprising various elements:
        - windows: windows after partition with [batch_size * num_windows, window_size, window_size, num_channels].
        - (padded_height, padded_width): padded height and width before partition
    """
    batch_size, height, width, num_channels = hidden_state.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size

    # Noop in case pad_width == 0 and pad_height == 0.
    hidden_state = nn.functional.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))

    padded_height, padded_width = height + pad_height, width + pad_width

    hidden_state = hidden_state.view(
        batch_size, padded_height // window_size, window_size, padded_width // window_size, window_size, num_channels
    )
    windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows, (padded_height, padded_width)


def window_unpartition(windows, window_size, pad_height_width, height_width):
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (`torch.Tensor`):
            Input tokens with [batch_size * num_windows, window_size, window_size, num_channels].
        window_size (`int`):
            Window size.
        pad_height_width (`tuple[int]`):
            Padded height and width (padded_height, padded_width).
        height_width (`tuple[int]`):
            Original height and width before padding.

    Returns:
        hidden_state: unpartitioned sequences with [batch_size, height, width, num_channels].
    """
    padded_height, padded_width = pad_height_width
    height, width = height_width
    batch_size = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
    hidden_state = windows.view(
        batch_size, padded_height // window_size, padded_width // window_size, window_size, window_size, -1
    )
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous()
    hidden_state = hidden_state.view(batch_size, padded_height, padded_width, -1)

    # We always have height <= padded_height and width <= padded_width
    hidden_state = hidden_state[:, :height, :width, :].contiguous()
    return hidden_state


class Sam3ViTLayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layer_scale_init_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class Sam3ViTLayer(GradientCheckpointingLayer):
    """Vision Transformer layer with rotary position embeddings and optional windowed attention."""

    def __init__(self, config: Sam3ViTConfig, window_size: int = 0) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        image_size = config.image_size
        image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)

        patch_size = config.patch_size
        patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)

        input_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        rotary_input_size = input_size if window_size == 0 else (window_size, window_size)
        rotary_scale = config.window_size / rotary_input_size[0]
        self.rotary_emb = Sam3ViTRotaryEmbedding(
            config, end_x=rotary_input_size[0], end_y=rotary_input_size[1], scale=rotary_scale
        )
        self.attention = Sam3ViTRoPEAttention(config)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = Sam3MLP(config)
        self.dropout = nn.Dropout(config.hidden_dropout)

        self.window_size = window_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            # Partition into non-overlapping windows for efficient attention
            hidden_states, pad_height_width = window_partition(hidden_states, self.window_size)

        position_embeddings = self.rotary_emb()
        hidden_states, _ = self.attention(hidden_states, position_embeddings, **kwargs)

        if self.window_size > 0:
            # Reverse window partition to restore original spatial layout
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_height_width, (height, width))

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states


@auto_docstring
class Sam3PreTrainedModel(PreTrainedModel):
    config_class = Sam3Config
    base_model_prefix = "sam3"
    main_input_name = "pixel_values"
    input_modalities = ["image", "text"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Sam3ViTEmbeddings):
            init.normal_(module.position_embeddings, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class Sam3ViTModel(Sam3PreTrainedModel):
    def __init__(self, config: Sam3ViTConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = Sam3ViTEmbeddings(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            [
                Sam3ViTLayer(config, window_size=config.window_size if i not in config.global_attn_indexes else 0)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.post_init()

    def get_input_embeddings(self) -> Sam3ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = self.embeddings(pixel_values)  # [batch_size, seq_len, hidden_size]

        batch_size = hidden_states.shape[0]
        height = pixel_values.shape[-2] // self.config.patch_size
        width = pixel_values.shape[-1] // self.config.patch_size
        hidden_size = hidden_states.shape[-1]

        # Reshape to spatial format for windowed attention: [batch_size, height, width, hidden_size]
        hidden_states = hidden_states.view(batch_size, height, width, hidden_size)

        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)

        # Reshape back to sequence format: [batch_size, height*width, hidden_size]
        hidden_states = hidden_states.view(batch_size, height * width, hidden_size)

        return BaseModelOutput(last_hidden_state=hidden_states)


class Sam3SinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def encode_1d_positions(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode 1D coordinate pairs using sine/cosine positional embeddings.

        Args:
            x: 1D tensor of x coordinates (flattened)
            y: 1D tensor of y coordinates (flattened)

        Returns:
            Tuple of (pos_x, pos_y) positional embeddings
        """
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).to(x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Encode 4D box coordinates (x, y, w, h) for decoder conditioning using sine/cosine embeddings.

        Args:
            boxes: Box coordinates [batch_size, num_queries, 4] in (x, y, w, h) format

        Returns:
            Position embeddings [batch_size, num_queries, num_pos_feats*4]
        """
        assert boxes.size(-1) == 4, f"Expected 4D box coordinates (x, y, w, h), got shape {boxes.shape}"
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=boxes.device).to(boxes.dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        x_embed = boxes[:, :, 0] * self.scale
        y_embed = boxes[:, :, 1] * self.scale
        w_embed = boxes[:, :, 2] * self.scale
        h_embed = boxes[:, :, 3] * self.scale

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)

        return pos

    @compile_compatible_method_lru_cache(maxsize=4)
    def forward(
        self,
        shape: torch.Size,
        device: Union[torch.device, str],
        dtype: torch.dtype,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if mask is None:
            mask = torch.zeros((shape[0], shape[2], shape[3]), device=device, dtype=torch.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=device).to(dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Sam3FPNLayer(nn.Module):
    def __init__(self, in_channels: int, fpn_dim: int, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

        # Build the upsampling/downsampling layers based on scale factor
        self.scale_layers = nn.ModuleList()

        if scale_factor == 4.0:
            self.scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            self.scale_layers.append(nn.GELU())
            self.scale_layers.append(nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2))
            intermediate_channels = in_channels // 4
        elif scale_factor == 2.0:
            self.scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            intermediate_channels = in_channels // 2
        elif scale_factor == 1.0:
            intermediate_channels = in_channels
        elif scale_factor == 0.5:
            self.scale_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            intermediate_channels = in_channels
        else:
            raise NotImplementedError(f"scale_factor={scale_factor} is not supported yet.")

        self.proj1 = nn.Conv2d(in_channels=intermediate_channels, out_channels=fpn_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels=fpn_dim, out_channels=fpn_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.scale_layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.proj1(hidden_states)
        hidden_states = self.proj2(hidden_states)

        return hidden_states


class Sam3VisionNeck(nn.Module):
    def __init__(self, config: Sam3VisionConfig):
        super().__init__()
        self.config = config

        self.position_encoding = Sam3SinePositionEmbedding(num_pos_feats=config.fpn_hidden_size // 2, normalize=True)

        # Create one FPN layer per scale factor
        self.fpn_layers = nn.ModuleList(
            [
                Sam3FPNLayer(
                    in_channels=config.backbone_config.hidden_size, fpn_dim=config.fpn_hidden_size, scale_factor=scale
                )
                for scale in config.scale_factors
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        fpn_hidden_states = ()
        fpn_position_encoding = ()

        for fpn_layer in self.fpn_layers:
            fpn_output = fpn_layer(hidden_states)
            fpn_hidden_states += (fpn_output,)
            # Generate position encoding for this FPN level
            pos_enc = self.position_encoding(fpn_output.shape, fpn_output.device, fpn_output.dtype)
            fpn_position_encoding += (pos_enc,)

        return fpn_hidden_states, fpn_position_encoding


@auto_docstring(
    custom_intro="""
    The vision model from Sam without any head or projection on top.
    """
)
class Sam3VisionModel(Sam3PreTrainedModel):
    config_class = Sam3VisionConfig
    main_input_name = "pixel_values"
    _can_record_outputs = {
        "hidden_states": Sam3ViTLayer,
        "attentions": Sam3ViTRoPEAttention,
    }

    def __init__(self, config: Sam3VisionConfig):
        super().__init__(config)
        self.config = config
        self.backbone = AutoModel.from_config(config.backbone_config)
        self.neck = Sam3VisionNeck(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @check_model_inputs
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Sam3VisionEncoderOutput]:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        backbone_output = self.backbone(pixel_values, **kwargs)
        hidden_states = backbone_output.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Reshape for FPN neck: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, height, width]
        batch_size = hidden_states.shape[0]
        height = pixel_values.shape[-2] // self.config.backbone_config.patch_size
        width = pixel_values.shape[-1] // self.config.backbone_config.patch_size
        hidden_states_spatial = hidden_states.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        fpn_hidden_states, fpn_position_encoding = self.neck(hidden_states_spatial)

        return Sam3VisionEncoderOutput(
            last_hidden_state=hidden_states,
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
        )


class Sam3GeometryEncoderLayer(nn.Module):
    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.self_attn = Sam3Attention(config)
        self.dropout = nn.Dropout(config.dropout)

        self.cross_attn = Sam3Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

        self.mlp = Sam3MLP(config)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        prompt_feats: Tensor,
        vision_feats: Tensor,
        vision_pos_encoding: Tensor,
        prompt_mask: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = prompt_feats
        hidden_states = self.layer_norm1(prompt_feats)
        hidden_states, _ = self.self_attn(
            query=hidden_states, key=hidden_states, value=hidden_states, attention_mask=prompt_mask, **kwargs
        )
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        key = vision_feats + vision_pos_encoding
        hidden_states, _ = self.cross_attn(query=hidden_states, key=key, value=vision_feats, **kwargs)
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual

        return hidden_states


class Sam3GeometryEncoder(nn.Module):
    """
    Encoder for geometric prompts (boxes).

    Boxes are encoded using three approaches:
     - Direct projection: linear projection from coordinate space to hidden_size
     - Pooling: pool features from the backbone at the specified location (ROI align for boxes)
     - Position encoding: use position encoding of the box center

    These encodings are combined additively and further processed with transformer layers.
    """

    def __init__(self, config: Sam3GeometryEncoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.roi_size = config.roi_size

        self.position_encoding = Sam3SinePositionEmbedding(num_pos_feats=config.hidden_size // 2, normalize=True)
        self.label_embed = nn.Embedding(2, self.hidden_size)
        self.cls_embed = nn.Embedding(1, self.hidden_size)

        # Box encoding layers
        self.boxes_direct_project = nn.Linear(4, self.hidden_size)
        self.boxes_pool_project = nn.Conv2d(self.hidden_size, self.hidden_size, self.roi_size)
        self.boxes_pos_enc_project = nn.Linear(self.hidden_size + 2, self.hidden_size)

        # Image feature normalization
        self.vision_layer_norm = nn.LayerNorm(self.hidden_size)

        # Prompt projection and normalization
        self.final_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.prompt_layer_norm = nn.LayerNorm(self.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([Sam3GeometryEncoderLayer(config) for _ in range(config.num_layers)])
        self.output_layer_norm = nn.LayerNorm(self.hidden_size)

    def _encode_box_coordinates(
        self, center_x: torch.Tensor, center_y: torch.Tensor, width: torch.Tensor, height: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode box coordinates by combining position-encoded centers with raw width/height.

        Args:
            center_x: 1D tensor of box center x coordinates
            center_y: 1D tensor of box center y coordinates
            width: 1D tensor of box widths
            height: 1D tensor of box heights

        Returns:
            Encoded box coordinates [N, embedding_dim]
        """
        pos_x, pos_y = self.position_encoding.encode_1d_positions(center_x, center_y)
        pos = torch.cat((pos_y, pos_x, height[:, None], width[:, None]), dim=1)
        return pos

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, vision_features):
        """Encode box prompts. Mask convention: True=valid, False=padding."""
        batch_size, num_boxes = boxes.shape[:2]
        height, width = vision_features.shape[-2:]
        boxes_embed = self.boxes_direct_project(boxes)

        # Pool features using ROI align
        # Convert boxes from CxCyWH to xyxy format and denormalize
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
        scale = scale.view(1, 1, 4)
        boxes_xyxy = boxes_xyxy * scale
        # ROI align expects list of boxes per batch element,
        # convert from bfloat16 to float16 as roi_align only supports float16 and float32
        dtype = torch.float16 if vision_features.dtype == torch.bfloat16 else vision_features.dtype
        sampled_features = torchvision.ops.roi_align(
            vision_features.to(dtype), boxes_xyxy.to(dtype).unbind(0), self.roi_size
        ).to(vision_features.dtype)

        pooled_projection = self.boxes_pool_project(sampled_features)
        pooled_projection = pooled_projection.view(batch_size, num_boxes, self.hidden_size)
        boxes_embed = boxes_embed + pooled_projection

        # Add position encoding
        center_x, center_y, box_width, box_height = boxes.unbind(-1)
        pos_enc = self._encode_box_coordinates(
            center_x.flatten(), center_y.flatten(), box_width.flatten(), box_height.flatten()
        )
        pos_enc = pos_enc.view(batch_size, num_boxes, pos_enc.shape[-1])
        pos_projection = self.boxes_pos_enc_project(pos_enc)
        boxes_embed = boxes_embed + pos_projection

        # Add label embeddings (positive/negative)
        label_embed = self.label_embed(boxes_labels.long())
        return label_embed + boxes_embed, boxes_mask

    def forward(
        self,
        box_embeddings: torch.Tensor,
        box_mask: torch.Tensor,
        box_labels: torch.Tensor,
        img_feats: tuple[torch.Tensor, ...],
        img_pos_embeds: Optional[tuple[torch.Tensor, ...]] = None,
    ):
        """
        Forward pass for encoding geometric prompts.

        Args:
            box_embeddings: Box coordinates in CxCyWH format [batch_size, num_boxes, 4]
            box_mask: Attention mask for boxes [batch_size, num_boxes]
            box_labels: Labels for boxes (positive/negative) [batch_size, num_boxes]
            img_feats: Image features from vision encoder
            img_pos_embeds: Optional position embeddings for image features

        Returns:
            Sam3GeometryEncoderOutput containing encoded geometry features and attention mask.
        """
        batch_size = box_embeddings.shape[0]

        # Prepare vision features for cross-attention: flatten spatial dimensions
        vision_feats = img_feats[-1]  # [B, C, H, W]
        vision_pos_embeds = img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(vision_feats)
        vision_feats_flat = vision_feats.flatten(2).transpose(1, 2)  # [B, H*W, C]
        vision_pos_embeds_flat = vision_pos_embeds.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Normalize image features for pooling operations
        img_feats_last = img_feats[-1]  # [B, C, H, W]
        img_feats_last = img_feats_last.permute(0, 2, 3, 1)  # [B, H, W, C]
        normalized_img_feats = self.vision_layer_norm(img_feats_last)
        normalized_img_feats = normalized_img_feats.permute(0, 3, 1, 2)  # [B, C, H, W]

        prompt_embeds, prompt_mask = self._encode_boxes(box_embeddings, box_mask, box_labels, normalized_img_feats)

        # Add CLS token (always valid)
        cls_embed = self.cls_embed.weight.view(1, self.hidden_size).unsqueeze(0).expand(batch_size, -1, -1)
        cls_mask = torch.ones(batch_size, 1, dtype=prompt_mask.dtype, device=prompt_mask.device)
        prompt_embeds, prompt_mask = concat_padded_sequences(prompt_embeds, prompt_mask, cls_embed, cls_mask)

        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))

        # Create bidirectional attention mask for transformer layers
        prompt_attention_mask = None
        if prompt_mask is not None:
            prompt_attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=prompt_embeds,
                attention_mask=prompt_mask,
            )

        # Apply transformer layers with cross-attention to vision features
        for layer in self.layers:
            prompt_embeds = layer(
                prompt_feats=prompt_embeds,
                vision_feats=vision_feats_flat,
                vision_pos_encoding=vision_pos_embeds_flat,
                prompt_mask=prompt_attention_mask,
            )

        # Final output normalization
        prompt_embeds = self.output_layer_norm(prompt_embeds)

        return Sam3GeometryEncoderOutput(
            last_hidden_state=prompt_embeds,
            attention_mask=prompt_mask,
        )


class Sam3DetrEncoderLayer(nn.Module):
    """DETR encoder layer with self-attention and cross-attention."""

    def __init__(self, config: Sam3DETREncoderConfig):
        super().__init__()
        self.config = config
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.self_attn = Sam3Attention(config)
        self.dropout = nn.Dropout(config.dropout)

        self.cross_attn = Sam3Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

        self.mlp = Sam3MLP(config)
        self.layer_norm3 = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        vision_feats: Tensor,
        prompt_feats: Tensor,
        vision_pos_encoding: Tensor,
        prompt_mask: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Forward pass for DETR encoder layer.

        Args:
            vision_feats: Vision features [batch_size, vision_len, hidden_size] (main hidden states)
            prompt_feats: Text prompt features [batch_size, text_len, hidden_size]
            vision_pos_encoding: Position encoding for vision [batch_size, vision_len, hidden_size]
            prompt_mask: Padding mask for prompts [batch_size, text_len] where True=valid, False=padding

        Returns:
            Updated vision features [batch_size, vision_len, hidden_size]
        """
        # Self-attention on vision features with position encoding
        residual = vision_feats
        hidden_states = self.layer_norm1(vision_feats)
        hidden_states_with_pos = hidden_states + vision_pos_encoding
        hidden_states, _ = self.self_attn(
            query=hidden_states_with_pos,
            key=hidden_states_with_pos,
            value=hidden_states,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states) + residual

        # Cross-attention: vision queries attend to text/prompt features
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        prompt_cross_attn_mask = None
        if prompt_mask is not None:
            prompt_cross_attn_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=hidden_states,
                attention_mask=prompt_mask,
                encoder_hidden_states=prompt_feats,
            )

        hidden_states, _ = self.cross_attn(
            query=hidden_states,
            key=prompt_feats,
            value=prompt_feats,
            attention_mask=prompt_cross_attn_mask,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states) + residual

        # MLP
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual

        return hidden_states


class Sam3DetrEncoder(Sam3PreTrainedModel):
    """
    DETR-style encoder that processes multi-level vision features with text fusion.

    This encoder processes vision features from multiple levels (e.g., FPN features at different
    resolutions) and fuses them with text prompts through a stack of transformer encoder layers.
    """

    _can_record_outputs = {
        "hidden_states": Sam3DetrEncoderLayer,
        "attentions": Sam3Attention,
    }

    def __init__(self, config: Sam3DETREncoderConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.layers = nn.ModuleList([Sam3DetrEncoderLayer(config) for _ in range(config.num_layers)])

    def _prepare_multilevel_features(
        self,
        vision_features: list[torch.Tensor],
        vision_pos_embeds: list[torch.Tensor],
    ):
        """
        Prepare multi-level vision features by flattening spatial dimensions and adding level embeddings.

        Args:
            vision_features: List of vision features at different levels [batch_size, channels, height, width]
            vision_pos_embeds: List of position embeddings for each level [batch_size, channels, height, width]

        Returns:
            Tuple containing flattened features, position embeddings, and spatial metadata
        """
        features_flattened = []
        pos_embeds_flattened = []
        spatial_shapes = []

        for features, pos_embed in zip(vision_features, vision_pos_embeds):
            height, width = features.shape[-2:]
            spatial_shapes.append((height, width))

            # Flatten spatial dimensions: [batch_size, channels, height, width] -> [batch_size, height*width, channels]
            features = features.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)

            features_flattened.append(features)
            pos_embeds_flattened.append(pos_embed)

        # Concatenate all levels into single sequence
        features_flattened = torch.cat(features_flattened, dim=1)
        pos_embeds_flattened = torch.cat(pos_embeds_flattened, dim=1)

        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=features_flattened.device)

        return (
            features_flattened,
            pos_embeds_flattened,
            spatial_shapes,
        )

    @check_model_inputs
    def forward(
        self,
        vision_features: list[torch.Tensor],
        text_features: torch.Tensor,
        vision_pos_embeds: Optional[list[torch.Tensor]] = None,
        text_mask: Optional[torch.Tensor] = None,
        spatial_sizes: Optional[list[tuple[int, int]]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        Forward pass for the DETR encoder.

        Args:
            vision_features: List of vision features at different levels
            text_features: Text prompt features [batch_size, seq_len, hidden_size]
            vision_pos_embeds: Optional list of position embeddings for each level
            text_mask: Optional text padding mask [batch_size, seq_len]
            spatial_sizes: Optional list of (height, width) tuples for reshaping

        Returns:
            Sam3DETREncoderOutput containing encoded features and metadata.
        """
        batch_size = vision_features[0].shape[0] if vision_features[0].dim() == 4 else vision_features[0].shape[1]

        # TODO: See if we can remove that reshaping and just use the features as is.
        if spatial_sizes is not None:
            for i, (height, width) in enumerate(spatial_sizes):
                # Reshape from [height*width, batch_size, channels] to [batch_size, channels, height, width]
                vision_features[i] = vision_features[i].reshape(height, width, batch_size, -1).permute(2, 3, 0, 1)
                vision_pos_embeds[i] = vision_pos_embeds[i].reshape(height, width, batch_size, -1).permute(2, 3, 0, 1)

        # Flatten multi-level features for encoder processing
        (
            features_flattened,
            pos_embeds_flattened,
            spatial_shapes,
        ) = self._prepare_multilevel_features(vision_features, vision_pos_embeds)

        hidden_states = features_flattened
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                prompt_feats=text_features,
                vision_pos_encoding=pos_embeds_flattened,
                prompt_mask=text_mask,
                **kwargs,
            )
        return Sam3DETREncoderOutput(
            last_hidden_state=hidden_states,
            pos_embeds_flattened=pos_embeds_flattened,
            text_features=text_features,
            spatial_shapes=spatial_shapes,
        )


class Sam3DecoderMLP(nn.Module):
    """Simple 2 or 3-layer MLP for decoder components."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers == 2:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.layer3 = None
        elif num_layers == 3:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Only 2 or 3 layers supported, got {num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        if self.layer3 is not None:
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.layer2(x)
        return x


class Sam3DetrDecoderLayer(nn.Module):
    """DETR decoder layer with self-attention, text cross-attention, and vision cross-attention."""

    def __init__(self, config: Sam3DETRDecoderConfig):
        super().__init__()
        self.config = config
        self.self_attn = Sam3Attention(config)
        self.self_attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        self.text_cross_attn = Sam3Attention(config)
        self.text_cross_attn_dropout = nn.Dropout(config.dropout)
        self.text_cross_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        self.vision_cross_attn = Sam3Attention(config)
        self.vision_cross_attn_dropout = nn.Dropout(config.dropout)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        self.mlp = Sam3MLP(config)
        self.mlp_layer_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vision_cross_attn_mask: Optional[torch.Tensor] = None,
        presence_token: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: Query features [batch_size, num_queries, hidden_size]
            query_pos: Query position embeddings [batch_size, num_queries, hidden_size]
            text_features: Text features [batch_size, seq_len, hidden_size]
            vision_features: Vision features [batch_size, height*width, hidden_size]
            vision_pos_encoding: Vision position encoding [batch_size, height*width, hidden_size]
            text_mask: Text padding mask [batch_size, seq_len] where True=valid, False=padding
            vision_cross_attn_mask: Vision cross-attention mask [batch_size, num_heads, num_queries, height*width]
            presence_token: Optional presence token [batch_size, 1, hidden_size]

        Returns:
            Tuple of (updated hidden states, updated presence token)
        """
        # Concatenate presence token if provided
        if presence_token is not None:
            hidden_states = torch.cat([presence_token, hidden_states], dim=1)
            query_pos = torch.cat([torch.zeros_like(presence_token), query_pos], dim=1)

        # Self-attention with query position encoding
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        attn_output, _ = self.self_attn(
            query=query_with_pos,
            key=query_with_pos,
            value=hidden_states,
            attention_mask=None,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_dropout(attn_output)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Text cross-attention: queries attend to text features
        residual = hidden_states
        query_with_pos = hidden_states + query_pos

        text_cross_attn_mask = None
        if text_mask is not None:
            text_cross_attn_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=hidden_states,
                attention_mask=text_mask,
                encoder_hidden_states=text_features,
            )

        attn_output, _ = self.text_cross_attn(
            query=query_with_pos,
            key=text_features,
            value=text_features,
            attention_mask=text_cross_attn_mask,
            **kwargs,
        )
        hidden_states = residual + self.text_cross_attn_dropout(attn_output)
        hidden_states = self.text_cross_attn_layer_norm(hidden_states)

        # Expand vision cross-attention mask for presence token if needed
        combined_vision_mask = vision_cross_attn_mask
        if presence_token is not None and combined_vision_mask is not None:
            batch_size, num_heads = combined_vision_mask.shape[:2]
            presence_mask = torch.zeros(
                batch_size,
                num_heads,
                1,
                combined_vision_mask.shape[-1],
                device=combined_vision_mask.device,
                dtype=combined_vision_mask.dtype,
            )
            combined_vision_mask = torch.cat([presence_mask, combined_vision_mask], dim=2)

        # Vision cross-attention: queries attend to vision features (with RPB)
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        key_with_pos = vision_features + vision_pos_encoding
        attn_output, _ = self.vision_cross_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=vision_features,
            attention_mask=combined_vision_mask,
            **kwargs,
        )
        hidden_states = residual + self.vision_cross_attn_dropout(attn_output)
        hidden_states = self.vision_cross_attn_layer_norm(hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_dropout(hidden_states)
        hidden_states = self.mlp_layer_norm(hidden_states)

        # Extract presence token if it was added
        presence_token_out = None
        if presence_token is not None:
            presence_token_out = hidden_states[:, :1]
            hidden_states = hidden_states[:, 1:]

        return hidden_states, presence_token_out


class Sam3DetrDecoder(Sam3PreTrainedModel):
    """
    DETR-style decoder with box refinement and presence token.

    Simplified version that assumes:
    - Box refinement is always enabled
    - Intermediate outputs are always returned
    - BoxRPB (relative position bias) with log-scale encoding
    - Presence token is used
    """

    _can_record_outputs = {
        "hidden_states": Sam3DetrDecoderLayer,
        "attentions": Sam3Attention,
    }

    def __init__(
        self,
        config: Sam3DETRDecoderConfig,
    ):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.layers = nn.ModuleList([Sam3DetrDecoderLayer(config) for _ in range(config.num_layers)])

        self.output_layer_norm = nn.LayerNorm(config.hidden_size)

        self.box_head = Sam3DecoderMLP(config.hidden_size, config.hidden_size, 4, 3)

        self.query_embed = nn.Embedding(config.num_queries, config.hidden_size)
        self.reference_points = nn.Embedding(config.num_queries, 4)

        self.presence_token = nn.Embedding(1, config.hidden_size)
        self.presence_head = Sam3DecoderMLP(config.hidden_size, config.hidden_size, 1, 3)
        self.presence_layer_norm = nn.LayerNorm(config.hidden_size)
        self.clamp_presence_logit_max_val = 10.0

        self.ref_point_head = Sam3DecoderMLP(2 * config.hidden_size, config.hidden_size, config.hidden_size, 2)

        self.box_rpb_embed_x = Sam3DecoderMLP(2, config.hidden_size, config.num_attention_heads, 2)
        self.box_rpb_embed_y = Sam3DecoderMLP(2, config.hidden_size, config.num_attention_heads, 2)

        self.position_encoding = Sam3SinePositionEmbedding(num_pos_feats=config.hidden_size // 2, normalize=False)

    @compile_compatible_method_lru_cache(maxsize=1)
    def _get_coords(
        self, height: torch.Tensor, width: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate normalized coordinate grids."""
        coords_h = torch.arange(0, height, device=device, dtype=dtype) / height
        coords_w = torch.arange(0, width, device=device, dtype=dtype) / width
        return coords_h, coords_w

    def _get_rpb_matrix(
        self, reference_boxes: torch.Tensor, spatial_shape: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute box relative position bias (RPB) matrix using log-scale encoding.
        RPB helps the decoder attend to relevant spatial locations based on predicted box positions.

        Args:
            reference_boxes: Reference boxes [batch_size, num_queries, 4] in sigmoid space
            spatial_shape: (height, width) of the vision features as tensors

        Returns:
            RPB matrix [batch_size, num_heads, num_queries, height*width]
        """
        height, width = spatial_shape
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        batch_size, num_queries, _ = boxes_xyxy.shape

        # Generate coordinate grids
        coords_h, coords_w = self._get_coords(
            height, width, dtype=reference_boxes.dtype, device=reference_boxes.device
        )

        # Compute deltas between coordinates and box boundaries
        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(batch_size, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(batch_size, num_queries, -1, 2)

        # Apply log-scale encoding
        deltas_x_log = deltas_x * 8
        deltas_x_log = torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / math.log2(8)
        deltas_y_log = deltas_y * 8
        deltas_y_log = torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / math.log2(8)

        # Embed deltas
        deltas_x = self.box_rpb_embed_x(deltas_x_log)  # [batch_size, num_queries, width, num_heads]
        deltas_y = self.box_rpb_embed_y(deltas_y_log)  # [batch_size, num_queries, height, num_heads]

        # Combine into 2D bias matrix
        rpb_matrix = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(
            2
        )  # [batch_size, num_queries, height, width, num_heads]
        rpb_matrix = rpb_matrix.flatten(2, 3)  # [batch_size, num_queries, height*width, num_heads]
        rpb_matrix = rpb_matrix.permute(0, 3, 1, 2).contiguous()  # [batch_size, num_heads, num_queries, height*width]
        return rpb_matrix

    @check_model_inputs
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for the DETR decoder.

        Args:
            vision_features: Vision features [batch_size, height*width, hidden_size]
            text_features: Text features [batch_size, seq_len, hidden_size]
            vision_pos_encoding: Vision position encoding [batch_size, height*width, hidden_size]
            text_mask: Text padding mask [batch_size, seq_len] where True=valid, False=padding
            spatial_shapes: Spatial shapes [num_levels, 2]

        Returns:
            Sam3DETRDecoderOutput containing decoder outputs from all layers.
        """
        batch_size = vision_features.shape[0]

        hidden_states = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = reference_boxes.sigmoid()
        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)

        intermediate_outputs = []
        intermediate_boxes = [reference_boxes]
        intermediate_presence_logits = []

        for layer in self.layers:
            # Generate sine embeddings for conditional queries
            reference_points_input = reference_boxes.unsqueeze(2)
            query_sine_embed = self.position_encoding.encode_boxes(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # Compute box relative position bias (RPB) attention mask
            vision_cross_attn_mask = None
            if spatial_shapes is not None and spatial_shapes.shape[0] == 1:
                spatial_shape = (spatial_shapes[0, 0], spatial_shapes[0, 1])
                vision_cross_attn_mask = self._get_rpb_matrix(reference_boxes, spatial_shape)

            hidden_states, presence_token = layer(
                hidden_states,
                query_pos=query_pos,
                text_features=text_features,
                vision_features=vision_features,
                vision_pos_encoding=vision_pos_encoding,
                text_mask=text_mask,
                vision_cross_attn_mask=vision_cross_attn_mask,
                presence_token=presence_token,
                **kwargs,
            )

            # Box refinement: predict delta and update reference boxes
            reference_boxes_before_sigmoid = inverse_sigmoid(reference_boxes)
            delta_boxes = self.box_head(self.output_layer_norm(hidden_states))
            new_reference_boxes = (delta_boxes + reference_boxes_before_sigmoid).sigmoid()
            reference_boxes = new_reference_boxes.detach()

            intermediate_outputs.append(self.output_layer_norm(hidden_states))
            intermediate_boxes.append(new_reference_boxes)

            # Process presence token
            if presence_token is not None:
                presence_logits = self.presence_head(self.presence_layer_norm(presence_token)).squeeze(-1)
                presence_logits = presence_logits.clamp(
                    min=-self.clamp_presence_logit_max_val, max=self.clamp_presence_logit_max_val
                )
                intermediate_presence_logits.append(presence_logits)

        # Stack outputs from all layers
        intermediate_outputs = torch.stack(intermediate_outputs)
        intermediate_boxes = torch.stack(intermediate_boxes[:-1])
        intermediate_presence_logits = (
            torch.stack(intermediate_presence_logits) if intermediate_presence_logits else None
        )

        return Sam3DETRDecoderOutput(
            intermediate_hidden_states=intermediate_outputs,
            reference_boxes=intermediate_boxes,
            presence_logits=intermediate_presence_logits,
        )


class Sam3DotProductScoring(nn.Module):
    """
    Computes classification scores by computing dot product between projected decoder queries and pooled text features.
    This is used to determine confidence/presence scores for each query.
    """

    def __init__(self, config: Sam3Config):
        super().__init__()
        self.config = config
        hidden_size = config.detr_decoder_config.hidden_size
        projection_dim = config.detr_decoder_config.hidden_size

        self.text_mlp = Sam3DecoderMLP(
            input_dim=hidden_size,
            hidden_dim=config.detr_decoder_config.intermediate_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.text_mlp_dropout = nn.Dropout(config.detr_decoder_config.dropout)
        self.text_mlp_out_norm = nn.LayerNorm(hidden_size)

        # Projections for text and query features
        self.text_proj = nn.Linear(hidden_size, projection_dim)
        self.query_proj = nn.Linear(hidden_size, projection_dim)

        # Scale factor for dot product
        self.scale = float(1.0 / np.sqrt(projection_dim))

        # Clamping to avoid numerical issues
        self.clamp_logits = True
        self.clamp_max_val = 12.0

    def _pool_text_features(self, text_features: torch.Tensor, text_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Mean pool text features, accounting for padding.

        Args:
            text_features: [batch_size, seq_len, hidden_size]
            text_mask: [batch_size, seq_len] where True indicates valid tokens, False indicates padding

        Returns:
            pooled_text: [batch_size, hidden_size]
        """
        if text_mask is None:
            # No padding, simple mean
            return text_features.mean(dim=1)

        is_valid = text_mask.to(text_features.dtype).unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Count valid tokens per batch
        num_valid = is_valid.sum(dim=1).clamp(min=1.0)  # [batch_size, 1]

        # Mean pool only over valid tokens
        pooled_text = (text_features * is_valid).sum(dim=1) / num_valid  # [batch_size, hidden_size]

        return pooled_text

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute classification scores via dot product.

        Args:
            decoder_hidden_states: [num_layers, batch_size, num_queries, hidden_size]
            text_features: [batch_size, seq_len, hidden_size]
            text_mask: [batch_size, seq_len] where True=valid, False=padding

        Returns:
            scores: [num_layers, batch_size, num_queries, 1]
        """
        orig_text_features = text_features
        text_features = self.text_mlp(text_features)
        text_features = self.text_mlp_dropout(text_features)
        text_features = text_features + orig_text_features
        text_features = self.text_mlp_out_norm(text_features)

        pooled_text = self._pool_text_features(text_features, text_mask)

        proj_text = self.text_proj(pooled_text)
        proj_queries = self.query_proj(decoder_hidden_states)

        proj_text = proj_text.unsqueeze(-1)
        scores = torch.matmul(proj_queries, proj_text.unsqueeze(0))
        scores = scores * self.scale
        if self.clamp_logits:
            scores = scores.clamp(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class Sam3MaskEmbedder(nn.Module):
    """
    MLP that embeds object queries for mask prediction.
    Similar to MaskFormer's mask embedder.
    """

    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ]
        )
        self.activation = nn.ReLU()

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Query embeddings [batch_size, num_queries, hidden_size]

        Returns:
            Mask embeddings [batch_size, num_queries, hidden_size]
        """
        hidden_states = queries
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i < len(self.layers) - 1:
                hidden_states = self.activation(hidden_states)
        return hidden_states


class Sam3PixelDecoder(nn.Module):
    """
    Feature Pyramid Network (FPN) decoder that generates pixel-level features.
    Inspired by MaskFormer's pixel decoder.
    """

    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_upsampling_stages = config.num_upsampling_stages

        # Create conv layers and norms for FPN
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
                for _ in range(num_upsampling_stages)
            ]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden_size) for _ in range(num_upsampling_stages)])

        self.out_channels = hidden_size

    def forward(self, backbone_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            backbone_features: List of backbone features [batch_size, hidden_size, H_i, W_i]
                              from low to high resolution (assumes already projected to hidden_size)

        Returns:
            Pixel embeddings [batch_size, hidden_size, H, W] at the finest resolution
        """
        # Start from the coarsest feature (last in list)
        prev_fpn = backbone_features[-1]
        # Iterate through features from coarse to fine (excluding the last which we started with)
        for layer_idx, backbone_feat in enumerate(reversed(backbone_features[:-1])):
            # Upsample previous FPN output to match current backbone feature size
            prev_fpn = F.interpolate(prev_fpn, size=backbone_feat.shape[-2:], mode="nearest")

            # Add skip connection
            prev_fpn = prev_fpn + backbone_feat

            # Apply conv and norm
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = self.norms[layer_idx](prev_fpn)
            prev_fpn = F.relu(prev_fpn)

        return prev_fpn


class Sam3MaskDecoder(Sam3PreTrainedModel):
    """
    Mask decoder that combines object queries with pixel-level features to predict instance masks.
    Also produces a semantic segmentation output and supports cross-attention to prompts.
    """

    _can_record_outputs = {
        "attentions": Sam3Attention,
    }

    def __init__(self, config: Sam3MaskDecoderConfig):
        super().__init__(config)
        self.config = config
        hidden_size = config.hidden_size

        # Pixel decoder (FPN)
        self.pixel_decoder = Sam3PixelDecoder(config)

        # Mask embedder (MLP to transform queries)
        self.mask_embedder = Sam3MaskEmbedder(config)

        # Projection from pixel decoder output to mask embedding space
        self.instance_projection = nn.Conv2d(self.pixel_decoder.out_channels, hidden_size, kernel_size=1)

        # Semantic segmentation head (always present in UniversalSegmentationHead)
        self.semantic_projection = nn.Conv2d(self.pixel_decoder.out_channels, 1, kernel_size=1)

        self.prompt_cross_attn = Sam3Attention(config)
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden_size)
        self.prompt_cross_attn_dropout = nn.Dropout(config.dropout)

    @check_model_inputs
    def forward(
        self,
        decoder_queries: torch.Tensor,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        prompt_features: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            decoder_queries: Decoder output queries [batch_size, num_queries, hidden_size]
            backbone_features: List of backbone features to process through FPN
            encoder_hidden_states: Encoder outputs [batch_size, seq_len, hidden_size]
            prompt_features: Prompt features (text + geometry) for cross-attention [batch_size, prompt_len, hidden_size]
            prompt_mask: Padding mask [batch_size, prompt_len] where True=valid, False=padding

        Returns:
            Sam3MaskDecoderOutput containing predicted masks and semantic segmentation.
        """
        if prompt_features is not None:
            # Cross-attention: encoder features attend to prompt features
            residual = encoder_hidden_states
            normed_hidden_states = self.prompt_cross_attn_norm(encoder_hidden_states)

            cross_attn_mask = None
            if prompt_mask is not None:
                cross_attn_mask = create_bidirectional_mask(
                    config=self.config,
                    input_embeds=normed_hidden_states,
                    encoder_hidden_states=prompt_features,
                    attention_mask=prompt_mask,
                )

            attn_output, _ = self.prompt_cross_attn(
                query=normed_hidden_states,
                key=prompt_features,
                value=prompt_features,
                attention_mask=cross_attn_mask,
                **kwargs,
            )
            encoder_hidden_states = residual + self.prompt_cross_attn_dropout(attn_output)

        # Process backbone features through FPN to get pixel embeddings
        pixel_embed = self._embed_pixels(
            backbone_features=backbone_features,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Predict instance masks via dot product between query embeddings and pixel embeddings
        instance_embeds = self.instance_projection(pixel_embed)
        mask_embeddings = self.mask_embedder(decoder_queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeddings, instance_embeds)

        # Generate semantic segmentation
        semantic_seg = self.semantic_projection(pixel_embed)

        return Sam3MaskDecoderOutput(
            pred_masks=pred_masks,
            semantic_seg=semantic_seg,
        )

    def _embed_pixels(
        self,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed pixels by combining backbone FPN features with encoder vision features.
        The encoder vision features replace the finest-resolution backbone feature.

        Args:
            backbone_features: List of backbone features [batch_size, C, H_i, W_i]
            encoder_hidden_states: Encoder outputs [batch_size, seq_len, hidden_size]

        Returns:
            Pixel embeddings [batch_size, hidden_size, H, W]
        """
        backbone_visual_feats = [feat.clone() for feat in backbone_features]

        # Extract vision features from encoder output and reshape to spatial format
        spatial_dim = backbone_features[-1].shape[-2] * backbone_features[-1].shape[-1]
        encoder_visual_embed = encoder_hidden_states[:, :spatial_dim, :]
        batch_size, _, hidden_size = encoder_visual_embed.shape
        height, width = backbone_features[-1].shape[-2:]
        encoder_visual_embed = encoder_visual_embed.transpose(1, 2).reshape(batch_size, hidden_size, height, width)

        # Replace finest backbone feature with encoder vision features
        backbone_visual_feats[-1] = encoder_visual_embed

        # Process through FPN decoder
        pixel_embed = self.pixel_decoder(backbone_visual_feats)

        return pixel_embed


class Sam3Model(Sam3PreTrainedModel):
    input_modalities = ["image", "text"]
    _checkpoint_conversion_mapping = {
        r"detector_model.(.+)": r"\1"  # the regex allows to remove the prefix, and add it back in revert mode
    }
    _keys_to_ignore_on_load_unexpected = [
        r"^tracker_model.",
        r"^tracker_neck.",
    ]

    def __init__(self, config: Sam3Config):
        # loading from a sam3_video config
        if hasattr(config, "detector_config") and config.detector_config is not None:
            detector_config = config.detector_config
            if isinstance(detector_config, dict):
                detector_config = Sam3Config(**detector_config)
            config = detector_config
        super().__init__(config)
        self.vision_encoder = Sam3VisionModel(config.vision_config)
        self.text_encoder = CLIPTextModelWithProjection(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        # Project text features from text encoder hidden size to model hidden size
        # CLIP text encoder outputs 1024-dim features, but we need 256-dim for DETR
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.detr_encoder_config.hidden_size)

        # Pass _attn_implementation to subconfigs BEFORE creating modules
        config.geometry_encoder_config._attn_implementation = config._attn_implementation
        config.detr_encoder_config._attn_implementation = config._attn_implementation
        config.detr_decoder_config._attn_implementation = config._attn_implementation
        config.mask_decoder_config._attn_implementation = config._attn_implementation

        self.geometry_encoder = Sam3GeometryEncoder(config.geometry_encoder_config)
        self.detr_encoder = Sam3DetrEncoder(config.detr_encoder_config)
        self.detr_decoder = Sam3DetrDecoder(config.detr_decoder_config)
        self.mask_decoder = Sam3MaskDecoder(config.mask_decoder_config)

        # Dot product scoring to compute classification scores
        self.dot_product_scoring = Sam3DotProductScoring(config)

        self.post_init()

    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Text embeddings that can be passed as `text_embeds` to the forward method.

        Example:

        ```python
        >>> from transformers import Sam3Model, Sam3Processor
        >>> from PIL import Image
        >>> import requests

        >>> model = Sam3Model.from_pretrained("facebook/sam3")
        >>> processor = Sam3Processor.from_pretrained("facebook/sam3")

        >>> # Pre-compute text embeddings
        >>> text_inputs = processor(text="cat", return_tensors="pt")
        >>> text_embeds = model.get_text_features(**text_inputs)

        >>> # Reuse text embeddings for multiple images
        >>> img_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
        >>> image = Image.open(requests.get(img_url, stream=True).raw)
        >>> img_inputs = processor(images=image, return_tensors="pt")
        >>> outputs = model(pixel_values=img_inputs.pixel_values, text_embeds=text_embeds)
        ```
        """
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        ).last_hidden_state
        text_features = self.text_projection(text_features)
        return text_features

    @auto_docstring
    def get_vision_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sam3VisionEncoderOutput:
        r"""
        Returns:
            vision_embeds (`Sam3VisionEncoderOutput`):
                Vision embeddings that can be passed as `vision_embeds` to the forward method.

        Example:

        ```python
        >>> from transformers import Sam3Model, Sam3Processor
        >>> from PIL import Image
        >>> import requests

        >>> model = Sam3Model.from_pretrained("facebook/sam3")
        >>> processor = Sam3Processor.from_pretrained("facebook/sam3")

        >>> # Pre-compute vision embeddings
        >>> img_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
        >>> image = Image.open(requests.get(img_url, stream=True).raw)
        >>> img_inputs = processor(images=image, return_tensors="pt")
        >>> vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

        >>> # Reuse vision embeddings for multiple text prompts
        >>> text_inputs = processor(text="cat", return_tensors="pt")
        >>> outputs = model(vision_embeds=vision_embeds, input_ids=text_inputs.input_ids)
        ```
        """
        vision_outputs = self.vision_encoder(pixel_values, **kwargs)
        return vision_outputs

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_embeds: Optional[Sam3VisionEncoderOutput] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_boxes_labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Sam3ImageSegmentationOutput:
        r"""
        vision_embeds (`Sam3VisionEncoderOutput`, *optional*):
            Pre-computed vision embeddings. Can be used to easily reuse vision embeddings. If provided, `pixel_values`
            should not be passed. Mutually exclusive with `pixel_values`.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Pre-computed text embeddings. Can be used to easily reuse text embeddings. If provided, `input_ids`
            should not be passed. Mutually exclusive with `input_ids`.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`, *optional*):
            Normalized box coordinates in [0, 1] range, in (cx, cy, w, h) format.
        input_boxes_labels (`torch.LongTensor` of shape `(batch_size, num_boxes)`, *optional*):
            Labels for boxes: 1 (positive), 0 (negative).

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/sam3")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam3")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> text = "car"
        >>> inputs = processor(images=raw_image, text=text, return_tensors="pt")

        >>> # Get segmentation output
        >>> outputs = model(**inputs)
        >>> pred_masks = outputs.pred_masks
        >>> pred_boxes = outputs.pred_boxes
        ```
        """
        if (pixel_values is None) == (vision_embeds is None):
            raise ValueError("You must specify exactly one of pixel_values or vision_embeds")

        if (input_ids is None) == (text_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or text_embeds")

        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
        else:
            batch_size = vision_embeds.fpn_hidden_states[0].shape[0]
            device = vision_embeds.fpn_hidden_states[0].device

        if vision_embeds is None:
            vision_outputs = self.vision_encoder(pixel_values, **kwargs)
        else:
            vision_outputs = vision_embeds

        fpn_hidden_states = vision_outputs.fpn_hidden_states[:-1]
        fpn_position_encoding = vision_outputs.fpn_position_encoding[:-1]

        if text_embeds is None:
            text_features = self.get_text_features(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        else:
            text_features = text_embeds

        text_mask = attention_mask.bool() if attention_mask is not None else None
        has_geometry_prompts = input_boxes is not None and input_boxes.numel() > 0

        geometry_prompt_features = None
        geometry_prompt_mask = None

        if has_geometry_prompts:
            if input_boxes is not None and input_boxes.numel() > 0:
                box_embeddings = input_boxes  # [batch_size, num_boxes, 4]
                box_labels = (
                    input_boxes_labels
                    if input_boxes_labels is not None
                    else torch.ones_like(box_embeddings[..., 0], dtype=torch.long)
                )
                box_mask = (
                    (input_boxes_labels != -10)
                    if input_boxes_labels is not None
                    else torch.ones(batch_size, input_boxes.shape[1], dtype=torch.bool, device=device)
                )
                box_labels = torch.where(box_labels == -10, 0, box_labels)
            else:
                box_embeddings = torch.zeros(batch_size, 0, 4, dtype=text_features.dtype, device=device)
                box_labels = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
                box_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)

            geometry_outputs = self.geometry_encoder(
                box_embeddings=box_embeddings,
                box_mask=box_mask,
                box_labels=box_labels,
                img_feats=fpn_hidden_states,
                img_pos_embeds=fpn_position_encoding,
            )

            geometry_prompt_features = geometry_outputs.last_hidden_state
            geometry_prompt_mask = geometry_outputs.attention_mask

        if geometry_prompt_features is not None:
            # Repeat text_features for all geometry prompts
            if text_features.shape[0] == 1 and geometry_prompt_features.shape[0] > 1:
                text_features = text_features.repeat(geometry_prompt_features.shape[0], 1, 1)
            combined_prompt_features = torch.cat([text_features, geometry_prompt_features], dim=1)
            if text_mask is not None and text_mask.shape[0] == 1 and geometry_prompt_mask.shape[0] > 1:
                text_mask = text_mask.repeat(geometry_prompt_mask.shape[0], 1)

            if text_mask is not None and geometry_prompt_mask is not None:
                combined_prompt_mask = torch.cat([text_mask, geometry_prompt_mask], dim=1)
            elif text_mask is not None:
                geo_valid_mask = torch.ones(
                    batch_size, geometry_prompt_features.shape[1], dtype=torch.bool, device=device
                )
                combined_prompt_mask = torch.cat([text_mask, geo_valid_mask], dim=1)
            elif geometry_prompt_mask is not None:
                text_valid_mask = torch.ones(batch_size, text_features.shape[1], dtype=torch.bool, device=device)
                combined_prompt_mask = torch.cat([text_valid_mask, geometry_prompt_mask], dim=1)
            else:
                combined_prompt_mask = None
        else:
            combined_prompt_features = text_features
            combined_prompt_mask = text_mask

        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_hidden_states[-1]],
            text_features=combined_prompt_features,
            vision_pos_embeds=[fpn_position_encoding[-1]],
            text_mask=combined_prompt_mask,
            **kwargs,
        )

        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs.last_hidden_state,
            text_features=encoder_outputs.text_features,
            vision_pos_encoding=encoder_outputs.pos_embeds_flattened,
            text_mask=combined_prompt_mask,
            spatial_shapes=encoder_outputs.spatial_shapes,
            **kwargs,
        )

        # Refine boxes from decoder
        all_box_offsets = self.detr_decoder.box_head(decoder_outputs.intermediate_hidden_states)
        reference_boxes_inv_sig = inverse_sigmoid(decoder_outputs.reference_boxes)
        all_pred_boxes_cxcywh = (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs.intermediate_hidden_states,
            text_features=encoder_outputs.text_features,
            text_mask=combined_prompt_mask,
        ).squeeze(-1)

        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_outputs.intermediate_hidden_states[-1]
        presence_logits = decoder_outputs.presence_logits[-1]

        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=list(fpn_hidden_states),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            prompt_features=combined_prompt_features,
            prompt_mask=combined_prompt_mask,
            **kwargs,
        )

        return Sam3ImageSegmentationOutput(
            pred_masks=mask_outputs.pred_masks,
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            presence_logits=presence_logits,
            semantic_seg=mask_outputs.semantic_seg,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_reference_boxes=decoder_outputs.reference_boxes,
            encoder_hidden_states=encoder_outputs.hidden_states,
            vision_hidden_states=vision_outputs.hidden_states,
            vision_attentions=vision_outputs.attentions,
            detr_encoder_attentions=encoder_outputs.attentions,
            detr_decoder_attentions=decoder_outputs.attentions,
            mask_decoder_attentions=mask_outputs.attentions,
        )


__all__ = ["Sam3Model", "Sam3VisionModel", "Sam3ViTModel", "Sam3PreTrainedModel"]
