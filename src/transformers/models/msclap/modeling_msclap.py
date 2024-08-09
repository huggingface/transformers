# coding=utf-8
# Copyright 2024 The Microsoft Team and The HuggingFace Team. All rights reserved.
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
"""PyTorch MSCLAP model."""

import collections
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from ..auto.modeling_auto import AutoModel
from .configuration_msclap import MSClapAudioConfig, MSClapConfig, MSClapTextConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "kamilakesbi/ms_clap"


# Copied from transformers.models.clap.modeling_clap.interpolate
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


# Copied from transformers.models.clap.modeling_clap.window_partition
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


# Copied from transformers.models.clap.modeling_clap.window_reverse
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


# Copied from transformers.models.clap.modeling_clap.create_position_ids_from_input_ids
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


# Copied from transformers.models.clap.modeling_clap.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


@dataclass
# Copied from transformers.models.clap.modeling_clap.ClapTextModelOutput with Clap->MSClap
class MSClapTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
# Copied from transformers.models.clap.modeling_clap.ClapAudioModelOutput with Clap->MSClap
class MSClapAudioModelOutput(ModelOutput):
    """
    MSClapAudio model output to mimic the output of the original implementation.

    Args:
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The Audio embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    audio_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
# Copied from transformers.models.clap.modeling_clap.ClapOutput with Clap->MSClap
class MSClapOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for audio-text similarity.
        logits_per_audio:(`torch.FloatTensor` of shape `(audio_batch_size, text_batch_size)`):
            The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, audio_batch_size)`):
            The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`MSClapTextModel`].
        audio_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The audio embeddings obtained by applying the projection layer to the pooled output of [`MSClapAudioModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`MSClapTextModel`].
        audio_model_output(`BaseModelOutputWithPooling`):
            The output of the [`MSClapAudioModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_audio: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "audio_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.clap.modeling_clap.ClapDropPath with Clap->MSClap
class MSClapDropPath(nn.Module):
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


# Copied from transformers.models.clap.modeling_clap.ClapAudioAFFBlock with Clap->MSClap, CLAP->MSCLAP
class MSClapAudioAFFBlock(nn.Module):
    r"""
    ATTENTIONAL FEATURE FUSION Block from MSCLAP, since in MSCLAP we are always in 2D mode, it is not needed to implement
    the 1D version.
    """

    def __init__(self, config: MSClapAudioConfig):
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


# Copied from transformers.models.clap.modeling_clap.ClapAudioPatchEmbed with Clap->MSClap
class MSClapAudioPatchEmbed(nn.Module):
    """
    This module converts the hidden states reshaped as an image to patch embeddings ready to be passed to the
    Transformer block.
    """

    def __init__(self, config: MSClapAudioConfig):
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
            self.fusion_model = MSClapAudioAFFBlock(config)
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


# Copied from transformers.models.clap.modeling_clap.ClapAudioSelfAttention with Clap->MSClap
class MSClapAudioSelfAttention(nn.Module):
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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

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
            # Apply the attention mask is (precomputed for all layers in MSClapAudioModel forward() function)
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


# Copied from transformers.models.clap.modeling_clap.ClapAudioSelfOutput with Clap->MSClap
class MSClapAudioSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.clap.modeling_clap.ClapAudioAttention with Clap->MSClap
class MSClapAudioAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        self.self = MSClapAudioSelfAttention(config, dim, num_heads, window_size)
        self.output = MSClapAudioSelfOutput(config, dim)
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
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.clap.modeling_clap.ClapAudioIntermediate with Clap->MSClap
class MSClapAudioIntermediate(nn.Module):
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


# Copied from transformers.models.clap.modeling_clap.ClapAudioOutput with Clap->MSClap
class MSClapAudioOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.clap.modeling_clap.ClapAudioLayer with Clap->MSClap
class MSClapAudioLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = MSClapAudioAttention(config, dim, num_heads, window_size=self.window_size)
        self.drop_path = MSClapDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.intermediate = MSClapAudioIntermediate(config, dim)
        self.output = MSClapAudioOutput(config, dim)

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
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
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
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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


# Copied from transformers.models.clap.modeling_clap.ClapAudioStage with Clap->MSClap
class MSClapAudioStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                MSClapAudioLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
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
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
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


# Copied from transformers.models.clap.modeling_clap.ClapAudioPatchMerging with Clap->MSClap
class MSClapAudioPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
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

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be disible by width and height, if needed
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


# Adpated from transformers.models.clap.modeling_clap.ClapAudioEncoder with Clap->MSClap
class MSClapAudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = len(config.depths)

        self.config = config
        self.patch_embed = MSClapAudioPatchEmbed(config)
        self.enable_fusion = config.enable_fusion
        self.patch_stride = self.patch_embed.patch_stride
        self.spec_size = config.spec_size
        self.freq_ratio = config.spec_size // config.num_mel_bins

        self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))

        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        grid_size = self.patch_embed.grid_size
        self.input_resolutions = [(grid_size[0] // (2**i), grid_size[1] // (2**i)) for i in range(self.num_layers)]

        self.layers = nn.ModuleList(
            [
                MSClapAudioStage(
                    config=config,
                    dim=int(config.patch_embeds_hidden_size * 2**i_layer),
                    input_resolution=self.input_resolutions[i_layer],
                    depth=config.depths[i_layer],
                    num_heads=config.num_attention_heads[i_layer],
                    drop_path=drop_path_rate[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=MSClapAudioPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

        self.batch_norm = nn.BatchNorm2d(config.num_mel_bins)
        self.norm = nn.LayerNorm(self.num_features)
        self.depths = config.depths
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        SF = self.spec_size // (2 ** (len(self.depths) - 1)) // self.patch_stride[0] // self.freq_ratio
        self.tscam_conv = nn.Conv2d(
            in_channels=self.num_features, out_channels=config.num_classes, kernel_size=(SF, 3), padding=(0, 1)
        )

    def reshape_mel2img(self, normalized_input_features):
        """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
        _, _, time_length, freq_length = normalized_input_features.shape

        spec_width = int(self.spec_size * self.freq_ratio)
        spec_heigth = self.spec_size // self.freq_ratio

        if time_length > spec_width or freq_length > spec_heigth:
            raise ValueError("the wav size should be less than or equal to the swin input size")

        # to avoid bicubic zero error
        if time_length < spec_width:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (spec_width, freq_length), mode="bicubic", align_corners=True
            )
        if freq_length < spec_heigth:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (time_length, spec_heigth), mode="bicubic", align_corners=True
            )

        batch, channels, time, freq = normalized_input_features.shape

        # batch_size, channels, spec_width, spec_heigth --> batch_size, channels, spec_heigth * freq_ratio, spec_width // freq_ratio
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
    ) -> Union[Tuple, MSClapAudioModelOutput]:
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

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__, hidden_states, input_dimensions, layer_head_mask, output_attentions
                )
            else:
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

        last_hidden_state = self.tscam_conv(last_hidden_state)
        last_hidden_state = torch.flatten(last_hidden_state, 2)

        last_hidden_state = self.avgpool(torch.flatten(last_hidden_state, 2))
        last_hidden_state = torch.flatten(last_hidden_state, 1)

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


MSCLAP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MSClapConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MSCLAP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

MSCLAP_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Input audio features. This should be returnes by the [`ClapFeatureExtractor`] class that you can also
            retrieve from [`AutoFeatureExtractor`]. See [`ClapFeatureExtractor.__call__`] for details.
        is_longer (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*):
            Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
            the features.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

MSCLAP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Input audio features. This should be returnes by the [`ClapFeatureExtractor`] class that you can also
            retrieve from [`AutoFeatureExtractor`]. See [`ClapFeatureExtractor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Adapted from https://github.com/microsoft/CLAP/blob/59bc8446e3e9426bf4158810e572b0798a30cf4d/msclap/models/clap.py#L8
class MSClapProjectionLayer(nn.Module):
    def __init__(self, config: Union[MSClapAudioConfig, MSClapTextConfig]):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        projection_dim = config.projection_dim

        self.linear1 = nn.Linear(hidden_size, projection_dim, bias=False)
        self.linear2 = nn.Linear(projection_dim, projection_dim, bias=False)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.drop = nn.Dropout(config.projection_dropout_prob)
        self.activation = ACT2FN[config.projection_hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states1 = self.linear1(hidden_states)
        hidden_states2 = self.activation(hidden_states1)
        hidden_states2 = self.linear2(hidden_states2)
        hidden_states2 = self.drop(hidden_states2)

        hidden_states = self.layer_norm(hidden_states1 + hidden_states2)
        return hidden_states


class MSClapPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MSClapConfig
    base_model_prefix = "msclap"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            in_proj_std = (self.config.hidden_size**-0.5) * ((2 * self.config.num_hidden_layers) ** -0.5) * factor
            nn.init.normal_(module.weight, std=in_proj_std)
            if module.bias is not None:
                module.bias.data.zero_()


# Copied from transformers.models.clap.modeling_clap.ClapAudioModel with Clap->MSClap, laion/clap-htsat-fused->microsoft/ms_clap, CLAP->MSCLAP
class MSClapAudioModel(MSClapPreTrainedModel):
    config_class = MSClapAudioConfig
    main_input_name = "input_features"

    def __init__(self, config: MSClapAudioConfig):
        super().__init__(config)
        self.audio_encoder = MSClapAudioEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.audio_encoder.patch_embed.proj

    @add_start_docstrings_to_model_forward(MSCLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=MSClapAudioConfig)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, MSClapAudioModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = MSClapAudioModel.from_pretrained("microsoft/ms_clap")
        >>> processor = AutoProcessor.from_pretrained("microsoft/ms_clap")

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


@add_start_docstrings(MSCLAP_START_DOCSTRING)
# Copied from transformers.models.clap.modeling_clap.ClapModel with Clap->MSClap, CLAP->MSCLAP,laion/clap-htsat-unfused->microsoft/ms_clap
class MSClapModel(MSClapPreTrainedModel):
    config_class = MSClapConfig

    # Ignore copy
    def __init__(
        self,
        config: MSClapConfig = None,
        text_model: Optional[PreTrainedModel] = None,
        audio_model: Optional[PreTrainedModel] = None,
        text_projection: Optional[PreTrainedModel] = None,
        audio_projection: Optional[PreTrainedModel] = None,
    ):
        if config is None and (
            text_model is None or audio_model is None or text_projection is None or audio_projection is None
        ):
            raise ValueError(
                "Either a configuration has to be provided, or all four of text model, audio model and projection layers."
            )
        if config is None:
            config = MSClapConfig.from_text_audio_configs(text_model.config, audio_model.config)

        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        super().__init__(config)

        text_config = config.text_config
        audio_config = config.audio_config

        self.logit_scale = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))

        self.projection_dim = config.projection_dim

        if not text_model:
            self.text_model = AutoModel.from_config(text_config.text_model_config)

        if not text_projection:
            self.text_projection = MSClapProjectionLayer(text_config)

        if not audio_model:
            self.audio_model = MSClapAudioModel(audio_config)

        if not audio_projection:
            self.audio_projection = MSClapProjectionLayer(audio_config)

        if self.text_model.config.to_dict() != self.config.text_config.text_model_config.to_dict():
            logger.warning(
                f"Config of the text_model: {self.text_model.__class__} is overwritten by shared text_model config:"
                f" {self.config.text_config.text_model_config}"
            )

        if self.audio_model.config.to_dict() != self.config.audio_config.to_dict():
            logger.warning(
                f"Config of the audio_model: {self.audio_model.__class__} is overwritten by shared audio_model config:"
                f" {self.config.audio_model}"
            )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MSCLAP_TEXT_INPUTS_DOCSTRING)
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
            applying the projection layer to the pooled output of [`MSClapTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MSClapModel

        >>> model = MSClapModel.from_pretrained("microsoft/ms_clap")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/ms_clap")

        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use MSCLAP model's config for some fields (if specified) instead of those of audio & text components.
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

    @add_start_docstrings_to_model_forward(MSCLAP_AUDIO_INPUTS_DOCSTRING)
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
        Returns:
            audio_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by
            applying the projection layer to the pooled output of [`MSClapAudioModel`].

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, MSClapModel
        >>> import torch

        >>> model = MSClapModel.from_pretrained("microsoft/ms_clap")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/ms_clap")
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

    @add_start_docstrings_to_model_forward(MSCLAP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MSClapOutput, config_class=MSClapConfig)
    # Ignore copy
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
    ) -> Union[Tuple, MSClapOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, MSClapModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = MSClapModel.from_pretrained("microsoft/ms_clap")
        >>> processor = AutoProcessor.from_pretrained("microsoft/ms_clap")

        >>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

        >>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

        >>> outputs = model(**inputs)
        >>> logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
        >>> probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use MSCLAP model's config for some fields (if specified) instead of those of audio & text components.
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
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        audio_embeds = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(audio_embeds)

        text_embeds = text_outputs[0] if not return_dict else text_outputs.last_hidden_state

        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(axis=-1) - 1
        else:
            sequence_lengths = torch.ne(input_ids, 0).sum(-1) - 1

        text_embeds = text_embeds[torch.arange(input_ids.shape[0], device=text_embeds.device), sequence_lengths]

        text_embeds = self.text_projection(text_embeds)

        # normalized features
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, audio_embeds.t()) * logit_scale
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t()) * logit_scale

        loss = None
        if return_loss:
            caption_loss = contrastive_loss(logits_per_text)
            audio_loss = contrastive_loss(logits_per_audio.t())
            loss = (caption_loss + audio_loss) / 2.0

        if not return_dict:
            output = (logits_per_audio, logits_per_text, text_embeds, audio_embeds, text_outputs, audio_outputs)
            return ((loss,) + output) if loss is not None else output

        return MSClapOutput(
            loss=loss,
            logits_per_audio=logits_per_audio,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
        )


@add_start_docstrings(
    """
    MSCLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    MSCLAP_START_DOCSTRING,
)
# Adapted from transformers.models.clap.modeling_clap.ClapTextModelWithProjection with Clap->MSClap, CLAP->MSCLAP
class MSClapTextModelWithProjection(MSClapPreTrainedModel):
    config_class = MSClapTextConfig

    def __init__(self, config: MSClapTextConfig):
        super().__init__(config)

        self.text_model = AutoModel.from_config(config.text_model_config)

        self.text_projection = MSClapProjectionLayer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_model.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(MSCLAP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MSClapTextModelOutput, config_class=MSClapTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MSClapTextModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MSClapTextModelWithProjection

        >>> model = MSClapTextModelWithProjection.from_pretrained("microsoft/ms_clap")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/ms_clap")

        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[0] if not return_dict else text_outputs.last_hidden_state

        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(axis=-1) - 1
        else:
            sequence_lengths = torch.ne(input_ids, 0).sum(-1) - 1
        text_embeds = text_embeds[torch.arange(input_ids.shape[0], device=text_embeds.device), sequence_lengths]

        text_embeds = self.text_projection(text_embeds)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return MSClapTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


@add_start_docstrings(
    """
    MSCLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    MSCLAP_START_DOCSTRING,
)
# Copied from transformers.models.clap.modeling_clap.ClapAudioModelWithProjection with Clap->MSClap, CLAP->MSCLAP, laion/clap-htsat-fused->microsoft/ms_clap
class MSClapAudioModelWithProjection(MSClapPreTrainedModel):
    config_class = MSClapAudioConfig
    main_input_name = "input_features"

    def __init__(self, config: MSClapAudioConfig):
        super().__init__(config)
        self.audio_model = MSClapAudioModel(config)
        self.audio_projection = MSClapProjectionLayer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.audio_model.audio_encoder.patch_embed.proj

    @add_start_docstrings_to_model_forward(MSCLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MSClapAudioModelOutput, config_class=MSClapAudioConfig)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MSClapAudioModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import MSClapAudioModelWithProjection, MSClapProcessor

        >>> model = MSClapAudioModelWithProjection.from_pretrained("microsoft/ms_clap")
        >>> processor = MSClapProcessor.from_pretrained("microsoft/ms_clap")

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
            return_dict=return_dict,
        )

        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        audio_embeds = self.audio_projection(pooled_output)

        if not return_dict:
            outputs = (audio_embeds, audio_outputs[0]) + audio_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return MSClapAudioModelOutput(
            audio_embeds=audio_embeds,
            last_hidden_state=audio_outputs.last_hidden_state,
            attentions=audio_outputs.attentions,
            hidden_states=audio_outputs.hidden_states,
        )
