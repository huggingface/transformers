# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Swin Transformer model."""

import collections.abc
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...backbone_utils import BackboneMixin, filter_output_hidden_states
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging, torch_int
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..vit.modeling_vit import (
    PreTrainedModel,
    ViTAttention,
    ViTLayer,
    ViTMLP,
    ViTPreTrainedModel,
    eager_attention_forward,
)
from .configuration_swin import SwinConfig


logger = logging.get_logger(__name__)


class SwinDropPath(nn.Module):
    """Stochastic depth (DropPath) per sample, for residual blocks.

    Identity when ``drop_prob`` is 0 or outside training. See `Deep Networks with Stochastic Depth
    <https://arxiv.org/abs/1603.09382>`_.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prob = 1 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return hidden_states.div(keep_prob) * random_tensor

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


@auto_docstring(
    custom_intro="""
    Swin encoder's outputs, with potential hidden states and attentions.
    """
)
@dataclass
class SwinEncoderOutput(ModelOutput):
    r"""
    reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, hidden_size, height, width)`.

        Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
        include the spatial dimensions.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Swin model's outputs that also contains a pooling of the last hidden states.
    """
)
@dataclass
class SwinModelOutput(ModelOutput):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
        Average pooling of the last layer hidden-state.
    reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, hidden_size, height, width)`.

        Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
        include the spatial dimensions.
    """

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Swin masked image model outputs.
    """
)
@dataclass
class SwinMaskedImageModelingOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
        Masked image modeling (MLM) loss.
    reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        Reconstructed pixel values.
    reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, hidden_size, height, width)`.

        Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
        include the spatial dimensions.
    """

    loss: torch.FloatTensor | None = None
    reconstruction: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Swin outputs for image classification.
    """
)
@dataclass
class SwinImageClassifierOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Classification (or regression if config.num_labels==1) loss.
    logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
        Classification (or regression if config.num_labels==1) scores (before SoftMax).
    reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
        shape `(batch_size, hidden_size, height, width)`.

        Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
        include the spatial dimensions.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    reshaped_hidden_states: tuple[torch.FloatTensor, ...] | None = None


def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.transpose(2, 3).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.transpose(2, 3).contiguous().view(-1, height, width, num_channels)
    return windows


class SwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = SwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        self.position_embeddings = (
            nn.Parameter(torch.zeros(1, num_patches, config.embed_dim)) if config.use_absolute_embeddings else None
        )

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Interpolate pre-trained position encodings to support higher-resolution images at inference.
        Unlike ViT, Swin has no CLS token, so position embeddings cover patch positions only.
        """
        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = self.position_embeddings.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def forward(
        self,
        pixel_values: torch.FloatTensor | None,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple[torch.Tensor]:
        _, num_channels, height, width = pixel_values.shape
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


class SwinPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def maybe_pad(self, pixel_values, height, width):
        """Pad pixel_values to be divisible by patch_size if needed."""
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: torch.FloatTensor | None) -> tuple[torch.Tensor, tuple[int]]:
        _, num_channels, height, width = pixel_values.shape
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions


class SwinPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        dim (`int`):
            Number of input channels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def maybe_pad(self, input_feature: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Pad input feature map to be divisible by 2 in both spatial dimensions if needed."""
        if (height % 2 == 1) or (width % 2 == 1):
            input_feature = nn.functional.pad(input_feature, (0, 0, 0, width % 2, 0, height % 2))
        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be divisible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # Interleave rows and columns to produce [batch_size, height/2*width/2, 4*num_channels]
        input_feature = torch.cat(
            [input_feature[:, row::2, col::2, :] for col in range(2) for row in range(2)], dim=-1
        )
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


class SwinRelativePositionBias(nn.Module):
    """
    Relative position bias for Swin's window-based attention, following the style of BeitRelativePositionBias.

    Unlike BeiT, Swin has no CLS token, so the table covers exactly (2*ws_h-1)*(2*ws_w-1) unique
    relative positions. The lookup index is purely determined by window_size (static), so it is stored
    as a non-persistent buffer (recomputed from config on load, never serialised). The table values
    are learned parameters and must be re-read on every forward call.
    """

    def __init__(self, num_heads: int, window_size: tuple[int, int]):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        # Non-persistent: fully determined by window_size, no need to serialise.
        # Stored flat so forward avoids an extra .view() call.
        self.register_buffer(
            "relative_position_index",
            self._create_relative_position_index().view(-1),
            persistent=False,
        )

    def _create_relative_position_index(self) -> torch.Tensor:
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        # shift to start from 0 and compute a unique flat index for each (dh, dw) pair
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

    def forward(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]
        relative_position_bias = relative_position_bias.view(self.window_area, self.window_area, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # 1, num_heads, Wh*Ww, Wh*Ww


class SwinAttention(ViTAttention):
    def __init__(self, config: SwinConfig, hidden_size: int, num_attention_heads: int, window_size: int):
        super().__init__(config)
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=config.qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.relative_position_bias = SwinRelativePositionBias(num_attention_heads, (window_size, window_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: (batch_size * num_windows, window_size * window_size, channels)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Combine relative position bias with the cyclic-shift attention mask for SW-MSA
        relative_position_bias = self.relative_position_bias()  # 1, num_heads, ws*ws, ws*ws
        if attention_mask is not None:
            # attention_mask: (num_windows, ws*ws, ws*ws)
            num_windows = attention_mask.shape[0]
            batch_size = input_shape[0] // num_windows
            seq_len = input_shape[1]
            # Expand to (batch * num_windows, 1, ws*ws, ws*ws) for broadcasting
            attention_mask = (
                attention_mask.unsqueeze(1)  # (num_windows, 1, ws*ws, ws*ws)
                .unsqueeze(0)  # (1, num_windows, 1, ws*ws, ws*ws)
                .expand(batch_size, -1, -1, -1, -1)  # (batch, num_windows, 1, ws*ws, ws*ws)
                .reshape(-1, 1, seq_len, seq_len)  # (batch * num_windows, 1, ws*ws, ws*ws)
            )
            combined_mask = relative_position_bias + attention_mask
        else:
            combined_mask = relative_position_bias

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class SwinMLP(ViTMLP):
    def __init__(self, config: SwinConfig, dim: int):
        nn.Module.__init__(self)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(dim, int(config.mlp_ratio * dim))
        self.fc2 = nn.Linear(int(config.mlp_ratio * dim), dim)


class SwinLayer(ViTLayer):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        drop_path_rate: float = 0.0,
        shift_size: int = 0,
    ):
        super().__init__()
        self.window_size = config.window_size
        self.attention = SwinAttention(config, dim, num_heads, window_size=config.window_size)
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = SwinMLP(config, dim)
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        self.drop_path = SwinDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def set_shift_and_window_size(self, input_resolution: tuple[int, int]) -> None:
        """Clamp window and shift sizes when the window is larger than the input resolution."""
        if min(input_resolution) <= self.window_size:
            self.shift_size = torch_int(0)
            self.window_size = (
                torch.min(torch.tensor(input_resolution)) if torch.jit.is_tracing() else min(input_resolution)
            )

    def get_attn_mask(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        """Build the cyclic-shift attention mask for shifted-window MSA; returns None when shift_size is 0."""
        if self.shift_size > 0:
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Pad feature map so both spatial dimensions are divisible by window_size."""
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def cyclic_shift(self, hidden_states: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """Apply a cyclic shift along the spatial dimensions for shifted-window attention."""
        if self.shift_size > 0:
            direction = 1 if reverse else -1
            hidden_states = torch.roll(
                hidden_states,
                shifts=(direction * self.shift_size, direction * self.shift_size),
                dims=(1, 2),
            )
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        always_partition: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape

        hidden_states_windows = window_partition(self.cyclic_shift(hidden_states), self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(
            height_pad, width_pad, dtype=hidden_states.dtype, device=hidden_states_windows.device
        )

        attention_output, attn_weights = self.attention(hidden_states_windows, attn_mask, **kwargs)
        attention_output = self.dropout(attention_output)

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        attention_windows = self.cyclic_shift(
            window_reverse(attention_windows, self.window_size, height_pad, width_pad), reverse=True
        )

        if pad_values[3] > 0 or pad_values[5] > 0:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = shortcut + self.drop_path(attention_windows)

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual

        return hidden_states, attn_weights


class SwinStage(GradientCheckpointingLayer):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        drop_path: list[float],
        downsample,
    ):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                SwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    drop_path_rate=drop_path[i],
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        self.downsample = downsample(dim=dim) if downsample is not None else None

    def get_reshaped_hidden_states(
        self,
        hidden_states: torch.Tensor,
        hidden_states_before_downsampling: torch.Tensor,
        height: int,
        width: int,
        output_hidden_states_before_downsampling: bool,
    ) -> torch.Tensor:
        """
        Select the spatial hidden states for this stage and reshape from (B, L, C) to (B, C, H, W).

        The chosen state and its resolution depend on output_hidden_states_before_downsampling:
        - True  → pre-downsampling states at (height, width) — used by the backbone.
        - False → post-downsampling states at half the resolution (if a downsampler exists).
        """
        if output_hidden_states_before_downsampling:
            spatial_state, h, w = hidden_states_before_downsampling, height, width
        elif self.downsample is not None:
            spatial_state, h, w = hidden_states, (height + 1) // 2, (width + 1) // 2
        else:
            spatial_state, h, w = hidden_states, height, width

        batch_size, _, hidden_size = spatial_state.shape
        return spatial_state.view(batch_size, h, w, hidden_size).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        always_partition: bool = False,
        output_hidden_states_before_downsampling: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        height, width = input_dimensions
        last_attn_weights = None
        for layer_module in self.blocks:
            hidden_states, last_attn_weights = layer_module(
                hidden_states, input_dimensions, always_partition=always_partition, **kwargs
            )

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)

        reshaped_hidden_states = self.get_reshaped_hidden_states(
            hidden_states, hidden_states_before_downsampling, height, width, output_hidden_states_before_downsampling
        )

        return hidden_states, reshaped_hidden_states, last_attn_weights


@auto_docstring
class SwinPreTrainedModel(ViTPreTrainedModel):
    config: SwinConfig
    _no_split_modules = ["SwinStage"]
    _supports_flash_attn = False
    _supports_flex_attn = False
    # relative_position_index is now a non-persistent buffer (recomputed from window_size in __init__).
    _keys_to_ignore_on_load_unexpected = [
        r"attention\.self\.relative_position_index",
        r"attention\.relative_position_bias\.relative_position_index",
    ]
    _can_record_outputs = {
        # capture_initial_hidden_state=True: prepend the embedding input (args[0] of SwinStage 0) so that
        # hidden_states[0] has the same shape as the patch embeddings (num_patches, embed_dim).
        "hidden_states": OutputRecorder(SwinStage, index=0, capture_initial_hidden_state=True),
        # reshaped_hidden_states are collected explicitly by SwinEncoder (per stage) and the stem
        # is prepended in SwinModel.forward, so they are NOT captured via hooks here.
        # index=2: SwinStage returns (hidden_states, reshaped_hidden_states, last_attn_weights);
        # capture the last block's attention weights at index 2, giving one entry per stage.
        "attentions": OutputRecorder(SwinStage, index=2, capture_initial_hidden_state=False),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, SwinEmbeddings):
            if module.mask_token is not None:
                init.zeros_(module.mask_token)
            if module.position_embeddings is not None:
                init.zeros_(module.position_embeddings)
        elif isinstance(module, SwinRelativePositionBias):
            init.zeros_(module.relative_position_bias_table)
            init.copy_(module.relative_position_index, module._create_relative_position_index().view(-1))


class SwinEncoder(SwinPreTrainedModel):
    def __init__(self, config: SwinConfig, grid_size: tuple[int, int]):
        super().__init__(config)
        self.num_layers = len(config.depths)
        self.config = config
        dpr = [config.drop_path_rate * i / max(sum(config.depths) - 1, 1) for i in range(sum(config.depths))]
        self.layers = nn.ModuleList(
            [
                SwinStage(
                    config=config,
                    dim=int(config.embed_dim * 2**layer_idx),
                    input_resolution=(grid_size[0] // (2**layer_idx), grid_size[1] // (2**layer_idx)),
                    depth=config.depths[layer_idx],
                    num_heads=config.num_heads[layer_idx],
                    drop_path=dpr[sum(config.depths[:layer_idx]) : sum(config.depths[: layer_idx + 1])],
                    downsample=SwinPatchMerging if (layer_idx < self.num_layers - 1) else None,
                )
                for layer_idx in range(self.num_layers)
            ]
        )
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        always_partition: bool = False,
        output_hidden_states: bool = False,
        output_hidden_states_before_downsampling: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SwinEncoderOutput:
        r"""
        input_dimensions (`tuple[int, int]`):
            Spatial `(height, width)` of the patch grid entering the encoder.
        always_partition (`bool`, *optional*, defaults to `False`):
            If `True`, always apply window partitioning regardless of input resolution.
        output_hidden_states_before_downsampling (`bool`, *optional*, defaults to `False`):
            If `True`, `reshaped_hidden_states` contains pre-downsampling feature maps.
        """
        all_reshaped_hidden_states = None
        if output_hidden_states:
            # Prepend the stem: hidden_states is the patch embedding output (B, N, C),
            # reshape it to spatial (B, C, H, W) as the first reshaped hidden state.
            batch_size, _, hidden_size = hidden_states.shape
            stem_spatial = (
                hidden_states.view(batch_size, *input_dimensions, hidden_size).permute(0, 3, 1, 2).contiguous()
            )
            all_reshaped_hidden_states = (stem_spatial,)

        for layer_module in self.layers:
            hidden_states, reshaped_hidden_state, _ = layer_module(
                hidden_states,
                input_dimensions,
                always_partition=always_partition,
                output_hidden_states_before_downsampling=output_hidden_states_before_downsampling,
                **kwargs,
            )
            if output_hidden_states:
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if layer_module.downsample is not None:
                input_dimensions = ((input_dimensions[0] + 1) // 2, (input_dimensions[1] + 1) // 2)

        return SwinEncoderOutput(
            last_hidden_state=hidden_states,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


@auto_docstring
class SwinModel(SwinPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        r"""
        add_pooling_layer (`bool`, *optional*, defaults to `True`):
            Whether or not to apply pooling layer.
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether or not to create and apply mask tokens in the embedding layer.
        """
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = SwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SwinModelOutput:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # FIXME: output_hidden_states must be popped manually here because SwinEncoder takes it as an
        # explicit argument (not via **kwargs), so it is not captured by the @capture_outputs decorator.
        output_hidden_states = kwargs.pop("output_hidden_states", self.config.output_hidden_states)

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        return SwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    Swin Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://huggingface.co/papers/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """
)
class SwinForMaskedImageModeling(SwinPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.swin = SwinModel(config, add_pooling_layer=False, use_mask_token=True)

        num_features = int(config.embed_dim * 2 ** (config.num_layers - 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=config.encoder_stride**2 * config.num_channels, kernel_size=1
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SwinMaskedImageModelingOutput:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, SwinForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
        >>> model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-base-simmim-window6-192")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 192, 192]
        ```"""
        outputs = self.swin(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output.transpose(1, 2)
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        return SwinMaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    Swin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune Swin on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """
)
class SwinForImageClassification(SwinPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.swin = SwinModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(self.swin.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SwinImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.swin(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        pooled_output = outputs.pooler_output

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return SwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    Swin backbone, to be used with frameworks like DETR and MaskFormer.
    """
)
class SwinBackbone(BackboneMixin, SwinPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"swin.layernorm.*"]

    def __init__(self, config: SwinConfig):
        super().__init__(config)

        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]
        self.swin = SwinModel(config, add_pooling_layer=False)

        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self.out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/swin-tiny-patch4-window7-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 7, 7]
        ```
        """
        kwargs["output_hidden_states"] = True  # required to extract layers for the stages
        # always_partition=True preserves shifted-window attention at all resolutions.
        # output_hidden_states_before_downsampling=True captures pre-downsampling feature maps per stage.
        outputs = self.swin(
            pixel_values,
            always_partition=True,
            output_hidden_states_before_downsampling=True,
            **kwargs,
        )

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, outputs.reshaped_hidden_states):
            if stage in self.out_features:
                batch_size, num_channels, height, width = hidden_state.shape
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.reshaped_hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "SwinForImageClassification",
    "SwinForMaskedImageModeling",
    "SwinModel",
    "SwinPreTrainedModel",
    "SwinBackbone",
]
