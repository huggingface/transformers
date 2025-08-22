# coding=utf-8
# Copyright 2023 The LAION-AI Team and The HuggingFace Team. All rights reserved.
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
"""PyTorch CLAP model."""

import collections
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging, torch_int
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig


logger = logging.get_logger(__name__)


# Adapted from: https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/utils.py#L191
def interpolate(hidden_states, ratio):
    """
    Interpolate data in time domain. This is used to compensate the resolution reduction in downsampling of a CNN.

    Args:
        hidden_states (`torch.FloatTensor` of shape (batch_size, time_length, classes_num)):
            Input hidden states
        ratio (`int`):
            The ratio of the length of the output to the length of the input.
    """
    (batch_size, time_length, classes_num) = hidden_states.shape
    upsampled = hidden_states[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_length * ratio, classes_num)
    return upsampled


# Adapted from https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/htsat.py#L249
def window_partition(hidden_states, window_size):
    """
    Returns the resized hidden states. The output shape should be `(batch_size * num_windows, window_size, window_size,
    num_channels)`

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, height, width, num_channels)`):
            Input hidden states
        window_size (`int`):
            Window size
    """
    batch_size, height, width, num_channels = hidden_states.shape

    hidden_states = hidden_states.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Adapted from https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/htsat.py#L263
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    Args:
        windows (`torch.FloatTensor` of shape `(num_windows * batch_size, window_size, window_size, num_channels)`):
            Input windows
        window_size (`int`):
            Window size
        height (`int`):
            Height of the resized audio
        width (`int`):
            Width of the resized audio
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html#CLIP-loss-function
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for text model's outputs that also contains a pooling of the last hidden states.
    """
)
# Copied from transformers.models.clip.modeling_clip.CLIPTextModelOutput with CLIP->Clap
class ClapTextModelOutput(ModelOutput):
    r"""
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The text embeddings obtained by applying the projection layer to the pooler_output.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    ClapAudio model output to mimic the output of the original implementation.
    """
)
class ClapAudioModelOutput(ModelOutput):
    r"""
    audio_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        The Audio embeddings obtained by applying the projection layer to the pooler_output.
    """

    audio_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->Clap, vision->audio, Vision->Audio, image->audio
class ClapOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for audio-text similarity.
    logits_per_audio (`torch.FloatTensor` of shape `(audio_batch_size, text_batch_size)`):
        The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, audio_batch_size)`):
        The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`ClapTextModel`].
    audio_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The audio embeddings obtained by applying the projection layer to the pooled output of [`ClapAudioModel`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`ClapTextModel`].
    audio_model_output (`BaseModelOutputWithPooling`):
        The output of the [`ClapAudioModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_audio: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    audio_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "audio_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Adapted from transformers.models.swin.modeling_swin.SwinDropPath
class ClapDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is a slightly
    refactored version of the `SwinDropPath` implementation.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states):
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states

        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor.floor_()  # binarize
        output = hidden_states.div(keep_prob) * random_tensor
        return output


# Adapted from https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/feature_fusion.py#L133
class ClapAudioAFFBlock(nn.Module):
    r"""
    ATTENTIONAL FEATURE FUSION Block from CLAP, since in CLAP we are always in 2D mode, it is not needed to implement
    the 1D version.
    """

    def __init__(self, config: ClapAudioConfig):
        super().__init__()
        channels = config.patch_embeds_hidden_size
        downsize_ratio = config.aff_block_r
        inter_channels = int(channels // downsize_ratio)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, residual):
        attention_input = hidden_states + residual

        fused_layer_output = self.local_att(attention_input) + self.global_att(attention_input)
        fused_layer_output = self.sigmoid(fused_layer_output)

        output = 2 * hidden_states * fused_layer_output + 2 * residual * (1 - fused_layer_output)
        return output


class ClapAudioPatchEmbed(nn.Module):
    """
    This module converts the hidden states reshaped as an image to patch embeddings ready to be passed to the
    Transformer block.
    """

    def __init__(self, config: ClapAudioConfig):
        super().__init__()
        img_size = (config.spec_size, config.spec_size) if isinstance(config.spec_size, int) else config.spec_size
        patch_size = (
            (config.patch_size, config.patch_size) if isinstance(config.patch_size, int) else config.patch_size
        )
        patch_stride = (
            (config.patch_stride, config.patch_stride) if isinstance(config.patch_stride, int) else config.patch_stride
        )

        self.img_size = img_size
        self.patch_stride = patch_stride

        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = config.flatten_patch_embeds
        self.enable_fusion = config.enable_fusion

        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        scale_factor = 4 if (self.enable_fusion) and (config.fusion_type == "channel_map") else 1

        self.proj = nn.Conv2d(
            config.patch_embed_input_channels * scale_factor,
            config.patch_embeds_hidden_size,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=padding,
        )

        self.norm = nn.LayerNorm(config.patch_embeds_hidden_size) if config.enable_patch_layer_norm else nn.Identity()
        if self.enable_fusion:
            self.fusion_model = ClapAudioAFFBlock(config)
            self.mel_conv2d = nn.Conv2d(
                config.patch_embed_input_channels,
                config.patch_embeds_hidden_size,
                kernel_size=(patch_size[0], patch_size[1] * 3),
                stride=(patch_stride[0], patch_stride[1] * 3),
                padding=padding,
            )

    def forward(self, hidden_states, is_longer_idx=None):
        if self.enable_fusion:
            # retrieve the last mel as we have transposed the input
            global_hidden_states = hidden_states[:, 0:1, :, :]

            # global processing
            batch_size, num_channels, height, width = global_hidden_states.shape

            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(
                    f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                )

            global_hidden_states = self.proj(global_hidden_states)
            output_width = global_hidden_states.size(-1)
            if len(is_longer_idx) > 0:
                # local processing
                local_hidden_states = hidden_states[is_longer_idx, 1:, :, :].contiguous()
                batch_size, num_channels, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size * num_channels, 1, height, width)

                local_hidden_states = self.mel_conv2d(local_hidden_states)

                _, features, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size, num_channels, features, height, width)
                local_hidden_states = local_hidden_states.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)

                local_width = local_hidden_states.size(-1)
                local_hidden_states = torch.nn.functional.pad(
                    local_hidden_states, (0, output_width - local_width), "constant", 0
                )

                global_hidden_states[is_longer_idx] = self.fusion_model(
                    global_hidden_states[is_longer_idx], local_hidden_states
                )
            hidden_states = global_hidden_states
        else:
            _, _, height, width = hidden_states.shape
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(
                    f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                )
            hidden_states = self.proj(hidden_states)

        if self.flatten:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinSelfAttention with Swin->ClapAudio
class ClapAudioSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        hidden_shape = (batch_size, dim, -1, self.attention_head_size)

        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ClapAudioModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->ClapAudio
class ClapAudioSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinAttention with Swin->ClapAudio
class ClapAudioAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        self.self = ClapAudioSelfAttention(config, dim, num_heads, window_size)
        self.output = ClapAudioSelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate with Swin->ClapAudio
class ClapAudioIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinOutput with Swin->ClapAudio
class ClapAudioOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinLayer with SwinDropPath->ClapDropPath, Swin->ClapAudio
class ClapAudioLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, drop_path_rate=0.0, shift_size=0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = ClapAudioAttention(config, dim, num_heads, window_size=self.window_size)
        self.drop_path = ClapDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = ClapAudioIntermediate(config, dim)
        self.output = ClapAudioOutput(config, dim)

    def set_shift_and_window_size(self, input_resolution):
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = torch_int(0)
            self.window_size = (
                torch.min(torch.tensor(input_resolution)) if torch.jit.is_tracing() else min(input_resolution)
            )

    def get_attn_mask(self, height, width, dtype, device):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
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

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)

        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(
            height_pad, width_pad, dtype=hidden_states.dtype, device=hidden_states_windows.device
        )

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(attention_windows)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


# Copied from transformers.models.swin.modeling_swin.SwinStage with Swin->ClapAudio
class ClapAudioStage(GradientCheckpointingLayer):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                ClapAudioLayer(
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

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> tuple[torch.Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# Copied from transformers.models.swin.modeling_swin.SwinPatchMerging with Swin->ClapAudio
class ClapAudioPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be divisible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


class ClapAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = len(config.depths)

        self.config = config
        self.patch_embed = ClapAudioPatchEmbed(config)
        self.enable_fusion = config.enable_fusion
        self.patch_stride = self.patch_embed.patch_stride
        self.spec_size = config.spec_size
        self.freq_ratio = config.spec_size // config.num_mel_bins

        self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))

        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths), device="cpu")]

        grid_size = self.patch_embed.grid_size
        self.input_resolutions = [(grid_size[0] // (2**i), grid_size[1] // (2**i)) for i in range(self.num_layers)]

        self.layers = nn.ModuleList(
            [
                ClapAudioStage(
                    config=config,
                    dim=int(config.patch_embeds_hidden_size * 2**i_layer),
                    input_resolution=self.input_resolutions[i_layer],
                    depth=config.depths[i_layer],
                    num_heads=config.num_attention_heads[i_layer],
                    drop_path=drop_path_rate[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=ClapAudioPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

        self.batch_norm = nn.BatchNorm2d(config.num_mel_bins)
        self.norm = nn.LayerNorm(self.num_features)
        self.depths = config.depths
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def reshape_mel2img(self, normalized_input_features):
        """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
        _, _, time_length, freq_length = normalized_input_features.shape

        spec_width = int(self.spec_size * self.freq_ratio)
        spec_height = self.spec_size // self.freq_ratio

        if time_length > spec_width or freq_length > spec_height:
            raise ValueError("the wav size should be less than or equal to the swin input size")

        # to avoid bicubic zero error
        if time_length < spec_width:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (spec_width, freq_length), mode="bicubic", align_corners=True
            )
        if freq_length < spec_height:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (time_length, spec_height), mode="bicubic", align_corners=True
            )

        batch, channels, time, freq = normalized_input_features.shape

        # batch_size, channels, spec_width, spec_height --> batch_size, channels, spec_height * freq_ratio, spec_width // freq_ratio
        normalized_input_features = normalized_input_features.reshape(
            batch, channels * self.freq_ratio, time // self.freq_ratio, freq
        )
        normalized_input_features = normalized_input_features.permute(0, 1, 3, 2).contiguous()
        normalized_input_features = normalized_input_features.reshape(
            batch, channels, freq * self.freq_ratio, time // self.freq_ratio
        )

        return normalized_input_features

    def forward(
        self,
        input_features,
        is_longer: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[tuple, ClapAudioModelOutput]:
        input_features = input_features.transpose(1, 3)
        normalized_input_features = self.batch_norm(input_features)
        normalized_input_features = normalized_input_features.transpose(1, 3)

        is_longer_list_idx = None
        if self.enable_fusion:
            is_longer_list = is_longer.to(input_features.device)
            is_longer_list_idx = torch.where(is_longer_list == 1)[0]

        hidden_states = self.reshape_mel2img(normalized_input_features)

        frames_num = hidden_states.shape[2]

        hidden_states = self.patch_embed(hidden_states, is_longer_list_idx)

        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        input_dimensions = self.input_resolutions[0]

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange batch_size (height width) channels -> batch_size channel height width
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            input_dimensions = self.input_resolutions[i]

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            hidden_states = layer_outputs[0]

            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange batch_size (height width) channels -> batch_size channel height width
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange batch_size (height width) channels -> batch_size channel height width
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        last_hidden_state = self.norm(hidden_states)

        batch_size, _, n_channels = last_hidden_state.shape

        freq_shape = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        temporal_shape = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]

        last_hidden_state = (
            last_hidden_state.permute(0, 2, 1).contiguous().reshape(batch_size, n_channels, freq_shape, temporal_shape)
        )

        batch_size, n_channels, n_frequencies, n_temp = last_hidden_state.shape
        # group 2D CNN
        c_freq_bin = n_frequencies // self.freq_ratio
        last_hidden_state = last_hidden_state.reshape(
            batch_size, n_channels, n_frequencies // c_freq_bin, c_freq_bin, n_temp
        )
        last_hidden_state = (
            last_hidden_state.permute(0, 1, 3, 2, 4).contiguous().reshape(batch_size, n_channels, c_freq_bin, -1)
        )
        latent_output = self.avgpool(torch.flatten(last_hidden_state, 2))
        latent_output = torch.flatten(latent_output, 1)

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    latent_output,
                    all_reshaped_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=latent_output,
            hidden_states=all_reshaped_hidden_states,
            attentions=all_self_attentions,
        )


class ClapProjectionLayer(nn.Module):
    def __init__(self, config: Union[ClapAudioConfig, ClapTextConfig]):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        projection_dim = config.projection_dim

        self.linear1 = nn.Linear(hidden_size, projection_dim)
        self.activation = ACT2FN[config.projection_hidden_act]
        self.linear2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->ClapText, persistent=False->persistent=True
class ClapTextEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=True
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=True
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.align.modeling_align.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# Copied from transformers.models.align.modeling_align.AlignTextSelfAttention with Align->Clap
class ClapTextSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_dropout = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            head_mask=head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class ClapTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.align.modeling_align.AlignTextAttention with Align->Clap
class ClapTextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ClapTextSelfAttention(config)
        self.output = ClapTextSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class ClapTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class ClapTextOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.align.modeling_align.AlignTextLayer with Align->Clap
class ClapTextLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ClapTextAttention(config)
        self.intermediate = ClapTextIntermediate(config)
        self.output = ClapTextOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.align.modeling_align.AlignTextEncoder with Align->Clap
class ClapTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ClapTextLayer(config) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class ClapTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@auto_docstring
class ClapPreTrainedModel(PreTrainedModel):
    config: ClapConfig
    base_model_prefix = "clap"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        factor = self.config.initializer_factor

        if isinstance(module, ClapTextEmbeddings):
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.token_type_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, ClapModel):
            module.logit_scale_a.data.fill_(math.log(self.config.logit_scale_init_value))
            module.logit_scale_t.data.fill_(math.log(self.config.logit_scale_init_value))
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            in_proj_std = (self.config.hidden_size**-0.5) * ((2 * self.config.num_hidden_layers) ** -0.5) * factor
            nn.init.normal_(module.weight, std=in_proj_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ClapAudioSelfAttention):
            module.relative_position_bias_table.data.zero_()


class ClapAudioModel(ClapPreTrainedModel):
    config: ClapAudioConfig
    main_input_name = "input_features"

    def __init__(self, config: ClapAudioConfig):
        super().__init__(config)
        self.audio_encoder = ClapAudioEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.audio_encoder.patch_embed.proj

    @auto_docstring
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        is_longer (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*):
            Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
            the features.

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ClapAudioModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

        >>> inputs = processor(audios=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return self.audio_encoder(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@auto_docstring(
    custom_intro="""
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://huggingface.co/papers/1706.03762
    """
)
class ClapTextModel(ClapPreTrainedModel):
    config: ClapTextConfig

    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = ClapTextEmbeddings(config)
        self.encoder = ClapTextEncoder(config)

        self.pooler = ClapTextPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring
class ClapModel(ClapPreTrainedModel):
    config: ClapConfig

    def __init__(self, config: ClapConfig):
        super().__init__(config)

        if not isinstance(config.text_config, ClapTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type ClapTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.audio_config, ClapAudioConfig):
            raise TypeError(
                "config.audio_config is expected to be of type ClapAudioConfig but is of type"
                f" {type(config.audio_config)}."
            )

        text_config = config.text_config
        audio_config = config.audio_config

        self.logit_scale_a = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))
        self.logit_scale_t = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))

        self.projection_dim = config.projection_dim

        self.text_model = ClapTextModel(text_config)
        self.text_projection = ClapProjectionLayer(text_config)

        self.audio_model = ClapAudioModel(audio_config)
        self.audio_projection = ClapProjectionLayer(audio_config)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ClapTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapModel

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1] if return_dict is not None else text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        text_features = F.normalize(text_features, dim=-1)

        return text_features

    @auto_docstring
    def get_audio_features(
        self,
        input_features: Optional[torch.Tensor] = None,
        is_longer: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        is_longer (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*):
            Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
            the features.

        Returns:
            audio_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by
            applying the projection layer to the pooled output of [`ClapAudioModel`].

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, ClapModel
        >>> import torch

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        >>> random_audio = torch.rand((16_000))
        >>> inputs = feature_extractor(random_audio, return_tensors="pt")
        >>> audio_features = model.get_audio_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=return_dict,
        )

        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        audio_features = self.audio_projection(pooled_output)
        audio_features = F.normalize(audio_features, dim=-1)

        return audio_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ClapOutput]:
        r"""
        is_longer (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*):
            Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
            the features.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ClapModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

        >>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

        >>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

        >>> outputs = model(**inputs)
        >>> logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
        >>> probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        audio_embeds = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(audio_embeds)

        text_embeds = text_outputs[1] if not return_dict else text_outputs.pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale_text = self.logit_scale_t.exp()
        logit_scale_audio = self.logit_scale_a.exp()
        logits_per_text = torch.matmul(text_embeds, audio_embeds.t()) * logit_scale_text
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio

        loss = None
        if return_loss:
            caption_loss = contrastive_loss(logits_per_text)
            audio_loss = contrastive_loss(logits_per_audio.t())
            loss = (caption_loss + audio_loss) / 2.0

        return ClapOutput(
            loss=loss,
            logits_per_audio=logits_per_audio,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
        )


@auto_docstring
class ClapTextModelWithProjection(ClapPreTrainedModel):
    config: ClapTextConfig

    def __init__(self, config: ClapTextConfig):
        super().__init__(config)
        self.text_model = ClapTextModel(config)
        self.text_projection = ClapProjectionLayer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_model.embeddings.word_embeddings = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ClapTextModelOutput]:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapTextModelWithProjection

        >>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = text_outputs[1] if not return_dict else text_outputs.pooler_output

        text_embeds = self.text_projection(pooled_output)

        return ClapTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


@auto_docstring
class ClapAudioModelWithProjection(ClapPreTrainedModel):
    config: ClapAudioConfig
    main_input_name = "input_features"

    def __init__(self, config: ClapAudioConfig):
        super().__init__(config)
        self.audio_model = ClapAudioModel(config)
        self.audio_projection = ClapProjectionLayer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.audio_model.audio_encoder.patch_embed.proj

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ClapAudioModelOutput]:
        r"""
        is_longer (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*):
            Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
            the features.

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import ClapAudioModelWithProjection, ClapProcessor

        >>> model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        >>> processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> inputs = processor(audios=audio_sample, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> audio_embeds = outputs.audio_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        audio_embeds = self.audio_projection(pooled_output)

        return ClapAudioModelOutput(
            audio_embeds=audio_embeds,
            last_hidden_state=audio_outputs.last_hidden_state,
            attentions=audio_outputs.attentions,
            hidden_states=audio_outputs.hidden_states,
        )


__all__ = [
    "ClapModel",
    "ClapPreTrainedModel",
    "ClapTextModel",
    "ClapTextModelWithProjection",
    "ClapAudioModel",
    "ClapAudioModelWithProjection",
]
