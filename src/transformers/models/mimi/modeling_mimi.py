# coding=utf-8
# Copyright 2024 Kyutai, and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Mimi model."""

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from .configuration_mimi import MimiConfig


if is_flash_attn_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
class MimiOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
        Discret code embeddings computed using `model.encode`.
    audio_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Decoded audio values, obtained using the decoder part of Mimi.
    encoder_past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
        This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

        The model will output the same cache format that is fed as input.

        If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
        have their past key value states given to this model).
    decoder_past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
        This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

        The model will output the same cache format that is fed as input.

        If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
        have their past key value states given to this model).
    """

    audio_codes: Optional[torch.LongTensor] = None
    audio_values: Optional[torch.FloatTensor] = None
    encoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None
    decoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None


class MimiConv1dPaddingCache:
    """
    Padding cache for MimiConv1d causal convolutions in order to support streaming via cache padding.
    See: https://huggingface.co/papers/2005.06720 & https://huggingface.co/papers/2204.07064

    A padding cache is a list of cached partial hidden states for each convolution layer.
    Hidden states are cached from the previous call to the MimiConv1d forward pass, given the padding size.
    """

    def __init__(
        self,
        num_layers: int,
        per_layer_padding: list[int],
        per_layer_padding_mode: list[str],
        per_layer_in_channels: list[int],
    ):
        # ensure correct number of layers for each arg
        from_args_num_layers = {len(per_layer_padding), len(per_layer_padding_mode), len(per_layer_in_channels)}

        if len(from_args_num_layers) != 1 or from_args_num_layers.pop() != num_layers:
            raise ValueError(
                f"Expected `num_layers` ({num_layers}) values in `per_layer_padding`, `per_layer_padding_mode` and `per_layer_in_channels`"
            )
        elif not all(mode in ["constant", "replicate"] for mode in per_layer_padding_mode):
            raise NotImplementedError(
                "`padding_cache` is not supported for convolutions using other than `constant` or `replicate` padding mode"
            )

        self.per_layer_padding = per_layer_padding
        self.per_layer_padding_mode = per_layer_padding_mode
        self.per_layer_in_channels = per_layer_in_channels
        self.per_layer_is_init = [True] * num_layers

        self.padding_cache = [None] * num_layers

    def update(self, hidden_states: torch.Tensor, layer_idx: int):
        """
        Updates the padding cache with the new padding states for the layer `layer_idx` and returns the current cache.

        Parameters:
            hidden_states (`torch.Tensor`):
                The hidden states to be partially cached.
            layer_idx (`int`):
                The index of the layer to cache the states for.
        Returns:
            `torch.Tensor` or `None`, the current padding cache.
        """
        batch_size, dtype, device = hidden_states.shape[0], hidden_states.dtype, hidden_states.device
        padding = self.per_layer_padding[layer_idx]
        padding_mode = self.per_layer_padding_mode[layer_idx]
        in_channels = self.per_layer_in_channels[layer_idx]

        if self.padding_cache[layer_idx] is None:
            if padding_mode == "constant":
                current_cache = torch.zeros(
                    batch_size,
                    in_channels,
                    padding,
                    device=device,
                    dtype=dtype,
                )
            elif padding_mode == "replicate":
                current_cache = (
                    torch.ones(
                        batch_size,
                        in_channels,
                        padding,
                        device=device,
                        dtype=dtype,
                    )
                    * hidden_states[..., :1]
                )
        else:
            current_cache = self.padding_cache[layer_idx]

        # update the cache
        if padding > 0:
            padding_states = hidden_states[:, :, -padding:]
        else:
            padding_states = torch.empty(batch_size, in_channels, padding, dtype=dtype, device=device)
        self.padding_cache[layer_idx] = padding_states

        return current_cache


@dataclass
@auto_docstring
class MimiEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
        Discret code embeddings computed using `model.encode`.
    encoder_past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
        This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

        The model will output the same cache format that is fed as input.

        If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
        have their past key value states given to this model).
    padding_cache (`MimiConv1dPaddingCache`, *optional*):
        Padding cache for MimiConv1d causal convolutions in order to support streaming via cache padding.
    """

    audio_codes: Optional[torch.LongTensor] = None
    encoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None
    padding_cache: Optional[MimiConv1dPaddingCache] = None


@dataclass
@auto_docstring
class MimiDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*):
        Decoded audio values, obtained using the decoder part of Mimi.
    decoder_past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
        This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

        The model will output the same cache format that is fed as input.

        If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
        have their past key value states given to this model).
    """

    audio_values: Optional[torch.FloatTensor] = None
    decoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None


class MimiConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode: Optional[str] = None,
        bias: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode
        self.layer_idx = layer_idx
        self.in_channels = in_channels

        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logger.warning(
                "MimiConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias
        )

        kernel_size = self.conv.kernel_size[0]
        stride = torch.tensor(self.conv.stride[0], dtype=torch.int64)
        dilation = self.conv.dilation[0]

        # Effective kernel size with dilations.
        kernel_size = torch.tensor((kernel_size - 1) * dilation + 1, dtype=torch.int64)

        self.register_buffer("stride", stride, persistent=False)
        self.register_buffer("kernel_size", kernel_size, persistent=False)
        self.register_buffer("padding_total", kernel_size - stride, persistent=False)

        # Asymmetric padding required for odd strides
        self.padding_right = self.padding_total // 2
        self.padding_left = self.padding_total - self.padding_right

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    # Copied from transformers.models.encodec.modeling_encodec.EncodecConv1d._get_extra_padding_for_conv1d
    def _get_extra_padding_for_conv1d(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """See `pad_for_conv1d`."""
        length = hidden_states.shape[-1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total

        return ideal_length - length

    @staticmethod
    # Copied from transformers.models.encodec.modeling_encodec.EncodecConv1d._pad1d
    def _pad1d(hidden_states: torch.Tensor, paddings: tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if mode != "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def _get_output_length(self, input_length: torch.LongTensor) -> torch.LongTensor:
        """
        Return the length of the output of the MimiConv1d.
        """
        # padding size
        n_frames = (input_length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        extra_padding = ideal_length - input_length

        if self.causal:
            padding_left = self.padding_total
            padding_right = extra_padding
        else:
            padding_left = self.padding_left
            padding_right = self.padding_right + extra_padding

        # padding
        input_length = input_length + padding_left + padding_right

        # conv
        output_length = (
            input_length + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
        ) // self.conv.stride[0] + 1
        return output_length

    def forward(self, hidden_states, padding_cache=None):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

        if not self.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is not supported for non-causal convolutions.")

        if self.causal and padding_cache is not None:
            layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)

        elif self.causal:
            # Left padding for causal
            hidden_states = self._pad1d(hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode)

        else:
            hidden_states = self._pad1d(
                hidden_states, (self.padding_left, self.padding_right + extra_padding), mode=self.pad_mode
            )

        hidden_states = self.conv(hidden_states)
        return hidden_states


class MimiConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias=True,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias)

        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")

        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            # Asymmetric padding required for odd strides
            self.padding_right = padding_total // 2

        self.padding_left = padding_total - self.padding_right

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        # unpad
        end = hidden_states.shape[-1] - self.padding_right
        hidden_states = hidden_states[..., self.padding_left : end]
        return hidden_states


class MimiResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by Mimi.
    """

    def __init__(self, config: MimiConfig, dim: int, dilations: list[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [MimiConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]
        self.block = nn.ModuleList(block)

        if config.use_conv_shortcut:
            self.shortcut = MimiConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, hidden_states, padding_cache=None):
        residual = hidden_states

        for layer in self.block:
            if isinstance(layer, MimiConv1d):
                hidden_states = layer(hidden_states, padding_cache=padding_cache)
            else:
                hidden_states = layer(hidden_states)

        if isinstance(self.shortcut, MimiConv1d):
            residual = self.shortcut(residual, padding_cache=padding_cache)
        else:
            residual = self.shortcut(residual)

        return residual + hidden_states


class MimiEncoder(nn.Module):
    """SEANet encoder as used by Mimi."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        model = [MimiConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1

        # keep track of MimiConv1d submodule layer names for easy encoded length computation
        mimiconv1d_layer_names = ["layers.0"]

        # Downsample to raw audio scale
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            # Add residual layers
            for j in range(config.num_residual_layers):
                mimiconv1d_layer_names.extend([f"layers.{len(model)}.block.1", f"layers.{len(model)}.block.3"])
                model += [MimiResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])]
            # Add downsampling layers
            model += [nn.ELU()]
            mimiconv1d_layer_names.append(f"layers.{len(model)}")
            model += [MimiConv1d(config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio)]
            scaling *= 2

        model += [nn.ELU()]
        mimiconv1d_layer_names.append(f"layers.{len(model)}")
        model += [MimiConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]

        self.layers = nn.ModuleList(model)
        self._mimiconv1d_layer_names = mimiconv1d_layer_names

        # initialize layer_idx for MimiConv1d submodules, necessary for padding_cache
        for layer_idx, layername in enumerate(self._mimiconv1d_layer_names):
            conv_layer = self.get_submodule(layername)
            setattr(conv_layer, "layer_idx", layer_idx)

    def forward(self, hidden_states, padding_cache=None):
        for layer in self.layers:
            if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
                hidden_states = layer(hidden_states, padding_cache=padding_cache)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class MimiLayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://huggingface.co/papers/2103.17239).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.
    """

    def __init__(self, config):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.scale * x


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Mimi
class MimiRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: MimiConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MimiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    # Copied from transformers.models.clip.modeling_clip.CLIPMLP.forward
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# copied from transformers.models.gemma.modeling_gemma.GemmaAttention with Gemma->Mimi
# no longer copied after attention refactors
class MimiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MimiConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = 1 / math.sqrt(config.head_dim)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = MimiRotaryEmbedding(config)
        self.sliding_window = config.sliding_window  # Ignore copy

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# NO LONGER EXIST Copied from transformers.models.gemma.modeling_gemma.GemmaFlashAttention2 with Gemma->Mimi
# TODO cyril: modular
class MimiFlashAttention2(MimiAttention):
    """
    Mimi flash attention module. This module inherits from `MimiAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = flash_attn_supports_top_left_mask()

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if isinstance(past_key_values, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (MimiRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        device_type = query_states.device.type if query_states.device.type != "mps" else "cpu"
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = (
                    torch.get_autocast_dtype(device_type)
                    if hasattr(torch, "get_autocast_dtype")
                    else torch.get_autocast_gpu_dtype()
                )
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# NO LONGER EXIST Copied from transformers.models.gemma.modeling_gemma.GemmaSdpaAttention with Gemma->Mimi
# TODO cyril: modular
class MimiSdpaAttention(MimiAttention):
    """
    Mimi attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MimiAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MimiAttention.forward
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MimiModel is using MimiSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = causal_mask is None and q_len > 1

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None


MIMI_ATTENTION_CLASSES = {
    "eager": MimiAttention,
    "flash_attention_2": MimiFlashAttention2,
    "sdpa": MimiSdpaAttention,
}


class MimiTransformerLayer(GradientCheckpointingLayer):
    def __init__(self, config: MimiConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MIMI_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = MimiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn_layer_scale = MimiLayerScale(config)
        self.mlp_layer_scale = MimiLayerScale(config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class MimiTransformerModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MimiTransformerLayer`]

    Args:
        config: MimiConfig
    """

    def __init__(self, config: MimiConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [MimiTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation

        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Embedded representation that will be contextualized by the model
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
                `past_key_values`).

                If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
                and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
                information on the default strategy.

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            past_key_values (`Cache`, *optional*):
                It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

                If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
                have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
                of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MimiDecoder(nn.Module):
    """SEANet decoder as used by Mimi."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [MimiConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)]

        # Upsample to raw audio scale
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            # Add upsampling layers
            model += [nn.ELU()]
            model += [
                MimiConvTranspose1d(config, current_scale, current_scale // 2, kernel_size=ratio * 2, stride=ratio)
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [MimiResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]
            scaling //= 2

        # Add final layers
        model += [nn.ELU()]
        model += [MimiConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)]
        self.layers = nn.ModuleList(model)

    # Copied from transformers.models.encodec.modeling_encodec.EncodecDecoder.forward
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MimiEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config: MimiConfig, epsilon: float = 1e-5):
        super().__init__()
        embed = torch.zeros(config.codebook_size, config.codebook_dim)

        self.codebook_size = config.codebook_size

        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
        self.register_buffer("cluster_usage", torch.ones(config.codebook_size))
        self.register_buffer("embed_sum", embed)
        self._embed = None
        self.epsilon = epsilon

    @property
    def embed(self) -> torch.Tensor:
        if self._embed is None:
            self._embed = self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return self._embed

    def quantize(self, hidden_states):
        # Projects each vector in `hidden_states` over the nearest centroid and return its index.
        # `hidden_states` should be `[N, D]` with `N` the number of input vectors and `D` the dimension.
        dists = torch.cdist(hidden_states[None].float(), self.embed[None].float(), p=2)[0]
        embed_ind = dists.argmin(dim=-1)
        return embed_ind

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.encode
    def encode(self, hidden_states):
        shape = hidden_states.shape
        # pre-process
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        # quantize
        embed_ind = self.quantize(hidden_states)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    # Copied from transformers.models.encodec.modeling_encodec.EncodecEuclideanCodebook.decode
    def decode(self, embed_ind):
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize


# Copied from transformers.models.encodec.modeling_encodec.EncodecVectorQuantization with Encodec->Mimi
class MimiVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: MimiConfig):
        super().__init__()
        self.codebook = MimiEuclideanCodebook(config)

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class MimiResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config: MimiConfig, num_quantizers: Optional[int] = None):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.num_quantizers = num_quantizers if num_quantizers is not None else config.num_quantizers
        self.layers = nn.ModuleList([MimiVectorQuantization(config) for _ in range(self.num_quantizers)])

        self.input_proj = None
        self.output_proj = None
        if config.vector_quantization_hidden_dimension != config.hidden_size:
            self.input_proj = torch.nn.Conv1d(
                config.hidden_size, config.vector_quantization_hidden_dimension, 1, bias=False
            )
            self.output_proj = torch.nn.Conv1d(
                config.vector_quantization_hidden_dimension, config.hidden_size, 1, bias=False
            )

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        num_quantizers = num_quantizers if num_quantizers is not None else self.num_quantizers

        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes of shape [B, K, T] to the quantized representation."""
        quantized_out = torch.tensor(0.0, device=codes.device)
        codes = codes.transpose(0, 1)
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized

        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out)
        return quantized_out


class MimiSplitResidualVectorQuantizer(nn.Module):
    """Split Residual Vector Quantizer."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.max_num_quantizers = config.num_quantizers

        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = config.num_quantizers - config.num_semantic_quantizers

        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(config, self.num_semantic_quantizers)
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(config, self.num_acoustic_quantizers)

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[float] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given number of quantizers / codebooks. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """

        num_quantizers = self.max_num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.max_num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.max_num_quantizers}, but is currently {num_quantizers}."
            )

        if num_quantizers < self.num_semantic_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be higher than the number of semantic quantizers {self.num_semantic_quantizers}, but is currently {num_quantizers}."
            )

        # codes is [K, B, T], with T frames, K nb of codebooks.
        codes = self.semantic_residual_vector_quantizer.encode(embeddings)

        if num_quantizers > self.num_semantic_quantizers:
            acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers
            )
            codes = torch.cat([codes, acoustic_codes], dim=0)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""

        # The first num_semantic_quantizers codebooks are decoded using the semantic RVQ
        quantized_out = self.semantic_residual_vector_quantizer.decode(codes[:, : self.num_semantic_quantizers])

        # The rest of the codebooks are decoded using the acoustic RVQ
        if codes.shape[1] > self.num_semantic_quantizers:
            quantized_out += self.acoustic_residual_vector_quantizer.decode(codes[:, self.num_semantic_quantizers :])
        return quantized_out


@auto_docstring
class MimiPreTrainedModel(PreTrainedModel):
    config: MimiConfig
    base_model_prefix = "mimi"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MimiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, MimiLayerScale):
            module.scale.data.fill_(self.config.layer_scale_initial_scale)


@auto_docstring(
    custom_intro="""
    The Mimi neural audio codec model.
    """
)
class MimiModel(MimiPreTrainedModel):
    def __init__(self, config: MimiConfig):
        super().__init__(config)
        self.config = config

        self.encoder = MimiEncoder(config)
        self.encoder_transformer = MimiTransformerModel(config)

        self.downsample = None
        self.upsample = None
        if config.frame_rate != config.encodec_frame_rate:
            self.downsample = MimiConv1d(
                config,
                config.hidden_size,
                config.hidden_size,
                kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
                stride=2,
                bias=False,
                pad_mode="replicate",
                layer_idx=len(self.encoder._mimiconv1d_layer_names),
            )

            self.upsample = MimiConvTranspose1d(
                config,
                config.hidden_size,
                config.hidden_size,
                kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
                stride=2,
                bias=False,
                groups=config.upsample_groups,
            )

        self.decoder_transformer = MimiTransformerModel(config)
        self.decoder = MimiDecoder(config)

        self.quantizer = MimiSplitResidualVectorQuantizer(config)

        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def _encode_frame(
        self,
        input_values: torch.Tensor,
        num_quantizers: int,
        padding_mask: int,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        padding_cache: Optional[MimiConv1dPaddingCache] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes the given input using the underlying VQVAE. The padding mask is required to compute the correct scale.
        """

        # TODO: @eustlb, let's make the encoder support padding_mask so that batched inputs are supported.
        embeddings = self.encoder(input_values, padding_cache=padding_cache)

        # TODO: @eustlb, convert the padding mask to attention mask.
        encoder_outputs = self.encoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
        )
        if return_dict:
            past_key_values = encoder_outputs.get("past_key_values")
        elif len(encoder_outputs) > 1:
            past_key_values = encoder_outputs[1]
        embeddings = encoder_outputs[0].transpose(1, 2)
        embeddings = self.downsample(embeddings, padding_cache=padding_cache)

        codes = self.quantizer.encode(embeddings, num_quantizers)
        codes = codes.transpose(0, 1)
        return codes, past_key_values, padding_cache

    def get_encoded_length(self, input_length: torch.LongTensor) -> torch.LongTensor:
        """
        Return the number of frames of the encoded audio waveform.
        """
        output_length = input_length

        # encoder
        for layer_name in self.encoder._mimiconv1d_layer_names:
            output_length = self.encoder.get_submodule(layer_name)._get_output_length(output_length)

        # downsample
        output_length = self.downsample._get_output_length(output_length)

        return output_length

    def get_audio_codes_mask(self, padding_mask: torch.Tensor, padding_side: str = "right"):
        """
        Get the mask for the audio codes from the original padding mask.
        """
        encoded_lengths = self.get_encoded_length(padding_mask.sum(dim=-1))

        audio_codes_mask = torch.arange(encoded_lengths.max(), device=encoded_lengths.device).expand(
            len(encoded_lengths), -1
        )
        audio_codes_mask = audio_codes_mask < encoded_lengths.unsqueeze(1)
        audio_codes_mask = audio_codes_mask.to(padding_mask.device)

        if padding_side == "right":
            return audio_codes_mask
        else:
            return audio_codes_mask.flip(dims=[-1])

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_quantizers: Optional[float] = None,
        encoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        padding_cache: Optional[MimiConv1dPaddingCache] = None,
        use_streaming: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, Optional[torch.Tensor]], MimiEncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            num_quantizers (`int`, *optional*):
                Number of quantizers (i.e codebooks) to use. By default, all quantizers are used.
            encoder_past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
                This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

                The model will output the same cache format that is fed as input.

                If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
                have their past key value states given to this model).
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `codebook` of shape `[batch_size, num_codebooks, frames]`, the discrete encoded codes for the input audio waveform.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_streaming = use_streaming if use_streaming is not None else self.config.use_streaming

        num_quantizers = self.config.num_quantizers if num_quantizers is None else num_quantizers

        if num_quantizers > self.config.num_quantizers:
            raise ValueError(
                f"The number of quantizers (i.e codebooks) asked should be lower than the total number of quantizers {self.config.num_quantizers}, but is currently {num_quantizers}."
            )

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if use_streaming and padding_cache is None:
            per_layer_padding, per_layer_padding_mode, per_layer_in_channels = [], [], []
            for layer_name in self.encoder._mimiconv1d_layer_names:
                per_layer_padding.append(self.encoder.get_submodule(layer_name).padding_total)
                per_layer_padding_mode.append(self.encoder.get_submodule(layer_name).pad_mode)
                per_layer_in_channels.append(self.encoder.get_submodule(layer_name).in_channels)

            # downsample layer
            per_layer_padding.append(self.downsample.padding_total)
            per_layer_padding_mode.append(self.downsample.pad_mode)
            per_layer_in_channels.append(self.downsample.in_channels)

            padding_cache = MimiConv1dPaddingCache(
                num_layers=len(self.encoder._mimiconv1d_layer_names) + 1,
                per_layer_padding=per_layer_padding,
                per_layer_padding_mode=per_layer_padding_mode,
                per_layer_in_channels=per_layer_in_channels,
            )

        encoded_frames, encoder_past_key_values, padding_cache = self._encode_frame(
            input_values,
            num_quantizers,
            padding_mask.bool(),
            past_key_values=encoder_past_key_values,
            padding_cache=padding_cache,
            return_dict=return_dict,
        )

        if not return_dict:
            return (
                encoded_frames,
                encoder_past_key_values,
                padding_cache,
            )

        return MimiEncoderOutput(encoded_frames, encoder_past_key_values, padding_cache)

    def _decode_frame(
        self,
        codes: torch.Tensor,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        embeddings = self.quantizer.decode(codes)

        embeddings = self.upsample(embeddings)
        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2), past_key_values=past_key_values, return_dict=return_dict
        )
        if return_dict:
            past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            past_key_values = decoder_outputs[1]
        embeddings = decoder_outputs[0].transpose(1, 2)
        outputs = self.decoder(embeddings)
        return outputs, past_key_values

    def decode(
        self,
        audio_codes: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        decoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], MimiDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            decoder_past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
                This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

                The model will output the same cache format that is fed as input.

                If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
                have their past key value states given to this model).
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        audio_values, decoder_past_key_values = self._decode_frame(
            audio_codes, past_key_values=decoder_past_key_values, return_dict=return_dict
        )

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (
                audio_values,
                decoder_past_key_values,
            )
        return MimiDecoderOutput(audio_values, decoder_past_key_values)

    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_quantizers: Optional[int] = None,
        audio_codes: Optional[torch.Tensor] = None,
        encoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        decoder_past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], MimiOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Raw audio input converted to Float.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
            for *masked*.
        num_quantizers (`int`, *optional*):
            Number of quantizers (i.e codebooks) to use. By default, all quantizers are used.
        audio_codes (`torch.LongTensor`  of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        encoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the encoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).
        decoder_past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            The model will output the same cache format that is fed as input.

            If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
            have their past key value states given to this model).

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoFeatureExtractor, MimiModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "kyutai/mimi"
        >>> model = MimiModel.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> inputs = feature_extractor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if audio_codes is None:
            encoder_outputs = self.encode(
                input_values, padding_mask, num_quantizers, encoder_past_key_values, return_dict=return_dict
            )
            audio_codes = encoder_outputs[0]
            if return_dict:
                encoder_past_key_values = encoder_outputs.get("past_key_values")
            elif len(encoder_outputs) > 1:
                encoder_past_key_values = encoder_outputs[1]

        decoder_outputs = self.decode(audio_codes, padding_mask, decoder_past_key_values, return_dict=return_dict)
        audio_values = decoder_outputs[0]
        if return_dict:
            decoder_past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            decoder_past_key_values = decoder_outputs[1]

        if not return_dict:
            return (audio_codes, audio_values, encoder_past_key_values, decoder_past_key_values)

        return MimiOutput(
            audio_codes=audio_codes,
            audio_values=audio_values,
            encoder_past_key_values=encoder_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
        )


__all__ = ["MimiModel", "MimiPreTrainedModel"]
