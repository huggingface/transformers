# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MossAudioTokenizer model."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import DynamicSlidingWindowLayer
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedAudioTokenizerBase
from ...utils import ModelOutput, auto_docstring, logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_moss_audio_tokenizer import MossAudioTokenizerConfig, MossAudioTokenizerQuantizerConfig


logger = logging.get_logger(__name__)


def _quantizer_config_to_kwargs(config: MossAudioTokenizerQuantizerConfig) -> dict[str, int | str]:
    return {
        "input_dim": config.input_dim,
        "rvq_dim": config.rvq_dim,
        "output_dim": config.output_dim,
        "num_quantizers": config.num_quantizers,
        "codebook_size": config.codebook_size,
        "codebook_dim": config.codebook_dim,
        "quantizer_type": config.quantizer_type,
    }


# =============================================================================
# Output Classes
# =============================================================================


@dataclass
@auto_docstring
class MossAudioTokenizerEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(num_quantizers, batch_size, sequence_length)`, *optional*):
        Discrete audio codes computed using the encoder and quantizer.
    audio_codes_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Valid lengths for each sample's audio codes.
    encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, hidden_size, sequence_length)`, *optional*):
        Hidden states from the encoder before quantization.
    """

    audio_codes: torch.Tensor | None = None
    audio_codes_lengths: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None


@dataclass
@auto_docstring
class MossAudioTokenizerDecoderOutput(ModelOutput):
    r"""
    audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
        Decoded audio waveform.
    audio_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Valid lengths for each sample's audio.
    """

    audio: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None


@dataclass
@auto_docstring
class MossAudioTokenizerOutput(ModelOutput):
    r"""
    audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
        Decoded audio waveform.
    audio_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Valid lengths for each sample's audio.
    audio_codes (`torch.LongTensor` of shape `(num_quantizers, batch_size, sequence_length)`, *optional*):
        Discrete audio codes computed using the encoder and quantizer.
    audio_codes_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Valid lengths for each sample's audio codes.
    """

    audio: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None
    audio_codes: torch.Tensor | None = None
    audio_codes_lengths: torch.Tensor | None = None


# =============================================================================
# Streaming Module Base Classes
# =============================================================================


@dataclass
class StreamingState:
    """Base state for streaming modules."""

    batch_size: int
    device: torch.device

    def __post_init__(self):
        self.exec_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def set_exec_mask(self, exec_mask: torch.Tensor):
        self.exec_mask[:] = exec_mask

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.exec_mask[:] = torch.where(reset_mask, torch.ones_like(self.exec_mask), self.exec_mask)

    def __enter__(self):
        # ExitStack expects a context manager; returning self is conventional and useful for debugging.
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class StreamingModule(nn.Module):
    """Base class for streaming components."""

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: StreamingState | None = None
        self._streaming_detached: bool = False
        self._cached_children: list[tuple[str, nn.Module]] | None = None

    @property
    def is_streaming(self):
        return self._streaming_state is not None

    def _apply_named_streaming(self, fn):
        def _handle_module(prefix: str, module: nn.Module):
            if isinstance(module, StreamingModule) or getattr(module, "_supports_streaming", False):
                if getattr(module, "_streaming_detached", False) and prefix != "":
                    return
                if self._cached_children is None:
                    raise RuntimeError("Internal error: _cached_children should be initialized before traversal.")
                self._cached_children.append((prefix, module))
            for name, child in module.named_children():
                new_prefix = f"{prefix}.{name}" if prefix else name
                _handle_module(new_prefix, child)

        if self._cached_children is None:
            self._cached_children = []
            _handle_module("", self)
        for name, child in self._cached_children:
            fn(name, child)

    def _start_streaming(self, batch_size: int, exit_stack: ExitStack):
        def _start_streaming_fn(name: str, module):
            if module._streaming_state is not None:
                raise RuntimeError(f"{name} is already streaming!")
            state = module._init_streaming_state(batch_size)
            exit_stack.enter_context(state)
            module._streaming_state = state

        self._apply_named_streaming(_start_streaming_fn)

    def _stop_streaming(self) -> None:
        def _stop_streaming_fn(name: str, module):
            module._streaming_state = None

        self._apply_named_streaming(_stop_streaming_fn)

    def _init_streaming_state(self, batch_size: int) -> StreamingState:
        device = next(iter(self.parameters())).device
        return StreamingState(batch_size, device)

    def streaming(self, batch_size: int) -> ExitStack:
        """Context manager to enter streaming mode."""
        exit_stack = ExitStack()
        self._start_streaming(batch_size, exit_stack)
        exit_stack.callback(self._stop_streaming)
        return exit_stack


class StreamingContainer(StreamingModule):
    """Container for streaming modules."""

    pass


# =============================================================================
# Normalization Layers
# =============================================================================


class MossAudioTokenizerRMSNorm(LlamaRMSNorm):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
        device=None,
    ):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.variance_epsilon = eps
        self._register_load_state_dict_pre_hook(self._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        alpha_key = prefix + "alpha"
        weight_key = prefix + "weight"
        if alpha_key in state_dict and weight_key not in state_dict:
            state_dict[weight_key] = state_dict.pop(alpha_key).reshape_as(module.weight)


class MossAudioTokenizerLayerScale(nn.Module):
    """Layer scale from Touvron et al. 2021."""

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.full((channels,), init, requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module."""
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm"}:
        return MossAudioTokenizerRMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return MossAudioTokenizerRMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# =============================================================================
# Rotary Position Embedding
# =============================================================================


def _interleaved_to_half_rotary(x: torch.Tensor) -> torch.Tensor:
    first_half, second_half = x[..., ::2], x[..., 1::2]
    return torch.cat((first_half, second_half), dim=-1)


class MossAudioTokenizerRotaryEmbedding(LlamaRotaryEmbedding):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, max_period: float = 10000.0, head_dim: int = 64, device=None):
        config = SimpleNamespace(
            max_position_embeddings=2048,
            rope_parameters=RopeParameters(rope_type="default", rope_theta=max_period),
            head_dim=head_dim,
            hidden_size=head_dim,
            num_attention_heads=1,
        )
        super().__init__(config, device=device)


# =============================================================================
# Gating Modules
# =============================================================================


class MossAudioTokenizerActivationGating(nn.Module):
    """Gating FFN layer with activation."""

    def __init__(self, dim: int, dim_feedforward: int, activation, **factory_kwargs):
        super().__init__()
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3

        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.linear_in(x)
        B, T, _ = x.shape
        x = x.view(B, T, 2, -1)
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        x = self.linear_out(x)
        return x


def _get_activation(name: str):
    if name in ["sigmoid", "tanh", "relu"]:
        return getattr(torch, name)
    elif name in ["leaky_relu", "elu", "gelu", "silu", "mish", "softsign"]:
        return getattr(F, name)
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation {name}")


def make_gating(name: str, dim: int, dim_feedforward: int, **factory_kwargs) -> nn.Module:
    return MossAudioTokenizerActivationGating(dim, dim_feedforward, _get_activation(name), **factory_kwargs)


# =============================================================================
# Positional Embeddings
# =============================================================================


def create_sin_embedding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create sinusoidal positional embedding with shape [B, T, C]."""
    if dim % 2 != 0:
        raise ValueError(f"Sinusoidal embedding requires even dim, got dim={dim}")
    half_dim = dim // 2
    if half_dim <= 1:
        raise ValueError(f"Sinusoidal embedding requires dim >= 4, got dim={dim}")
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


# =============================================================================
# Multi-Head Attention
# =============================================================================


@dataclass
class MHAState(StreamingState):
    kv_cache: DynamicSlidingWindowLayer | None
    offset: torch.Tensor
    offset_cpu: int

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offset[:] = torch.where(reset_mask, torch.zeros_like(self.offset), self.offset)
        if self.kv_cache is not None:
            self.kv_cache.reset()
        self.offset_cpu = 0


def apply_weights_per_step(
    modules: nn.ModuleList,
    schedule: list[int] | None,
    x: torch.Tensor,
    offset: int | None,
) -> torch.Tensor:
    """Apply different weights for each time step."""
    if len(modules) == 1:
        return modules[0](x)

    if offset is None:
        raise ValueError("offset must be provided when using per-step weights (len(modules) > 1).")
    ys = []
    B, T, C = x.shape
    for t in range(T):
        module_index = t + offset
        if schedule is not None:
            if module_index >= len(schedule) or module_index < 0:
                raise ValueError(
                    f"weights_per_step_schedule is too short for module_index={module_index} (len={len(schedule)})."
                )
            module_index = schedule[module_index]
        if module_index >= len(modules) or module_index < 0:
            raise ValueError(f"module_index={module_index} out of range for len(modules)={len(modules)}.")
        y = modules[module_index](x[:, t : t + 1])
        ys.append(y)
    return torch.cat(ys, 1)


class MossAudioTokenizerAttention(LlamaAttention):
    """Multi-head attention with streaming support."""

    _supports_streaming = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: int | None = None,
        rope: MossAudioTokenizerRotaryEmbedding | None = None,
        weights_per_step: int = 0,
        weights_per_step_schedule: list[int] | None = None,
        device=None,
        dtype=None,
    ):
        nn.Module.__init__(self)
        self._streaming_state: MHAState | None = None
        self._streaming_detached: bool = False
        factory_kwargs = {"device": device, "dtype": dtype}

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got embed_dim={embed_dim}, num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.causal = causal
        self.is_causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.num_key_value_groups = 1
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self._attn_implementation = "sdpa"
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule

        out_dim = 3 * embed_dim
        mult = 1
        if weights_per_step:
            mult = max(weights_per_step_schedule) + 1 if weights_per_step_schedule else weights_per_step
        self.mult = mult

        self.out_projs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs) for _ in range(mult)]
        )
        self.in_projs = nn.ModuleList(
            [nn.Linear(embed_dim, out_dim, bias=False, **factory_kwargs) for _ in range(mult)]
        )

        self._register_load_state_dict_pre_hook(self._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        mappings = {
            "in_proj_weight": "in_projs.{i}.weight",
            "in_proj.weight": "in_projs.{i}.weight",
            "out_proj.weight": "out_projs.{i}.weight",
        }
        mult = module.mult
        for suffix in ["", "_scb"]:
            for source, target in mappings.items():
                this_source = prefix + source + suffix
                if this_source in state_dict:
                    weight = state_dict[this_source]
                    _, *OD = weight.shape
                    weight = weight.view(mult, -1, *OD)
                    for i in range(mult):
                        state_dict[prefix + target.format(i=i) + suffix] = weight[i]
                    state_dict.pop(this_source)

    def _init_streaming_state(self, batch_size: int) -> MHAState:
        in_proj = cast(nn.Linear, self.in_projs[0])
        device = cast(torch.device, in_proj.weight.device)

        if self.context is None:
            capacity = self.weights_per_step if self.weights_per_step else 1024
        else:
            capacity = self.context

        kv_cache = DynamicSlidingWindowLayer(sliding_window=capacity)
        return MHAState(
            batch_size,
            cast(torch.device, device),
            kv_cache,
            offset=torch.zeros(batch_size, device=cast(torch.device, device), dtype=torch.long),
            offset_cpu=0,
        )

    def _complete_kv(
        self, k: torch.Tensor, v: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = cast(MHAState | None, self._streaming_state)
        if state is None:
            batch_size, _, sequence_length, _ = k.shape
            positions = torch.arange(sequence_length, device=k.device, dtype=torch.long)
            return k, v, positions.expand(batch_size, -1)
        if state.kv_cache is None:
            batch_size, _, sequence_length, _ = k.shape
            positions = torch.arange(sequence_length, device=k.device, dtype=torch.long)
            return k, v, positions.expand(batch_size, -1)

        query_length = k.shape[-2]
        key_states, value_states = state.kv_cache.update(k, v)
        key_length = key_states.shape[-2]
        previous_length = key_length - query_length
        key_start = offset - previous_length
        positions = key_start.view(-1, 1) + torch.arange(key_length, device=k.device, dtype=torch.long)
        return key_states, value_states, positions

    def _prepare_attention_mask(
        self,
        query_states: torch.Tensor,
        key_positions: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.causal:
            return None

        _, _, query_length, _ = query_states.shape
        query_positions = offset.view(-1, 1, 1) + torch.arange(
            query_length, device=query_states.device, dtype=torch.long
        ).view(1, -1, 1)
        key_positions = key_positions[:, None, :]
        delta = query_positions - key_positions
        allowed = (key_positions >= 0) & (delta >= 0)
        if self.context is not None:
            allowed = allowed & (delta < self.context)

        attention_mask = torch.zeros(allowed.shape, device=query_states.device, dtype=query_states.dtype)
        attention_mask = attention_mask.masked_fill(~allowed, torch.finfo(query_states.dtype).min)
        return attention_mask[:, None, :, :]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        state = cast(MHAState | None, self._streaming_state)
        B, T = query.shape[:2]

        if state is None:
            offset = torch.zeros(B, device=query.device, dtype=torch.long)
            offset_cpu = 0
        else:
            offset = state.offset
            offset_cpu = state.offset_cpu

        projected = apply_weights_per_step(self.in_projs, self.weights_per_step_schedule, query, offset_cpu)
        projected = projected.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = projected[0], projected[1], projected[2]

        if self.rope:
            position_ids = offset.view(B, 1) + torch.arange(T, device=query.device).view(1, -1)
            cos, sin = self.rope(q, position_ids)
            q, k = _interleaved_to_half_rotary(q), _interleaved_to_half_rotary(k)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k, v, pos_k = self._complete_kv(k, v, offset)
        attention_mask = self._prepare_attention_mask(q, pos_k, offset)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self._attn_implementation, eager_attention_forward
        )
        x, _ = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
        )
        x = x.reshape(B, T, self.embed_dim)
        x = apply_weights_per_step(self.out_projs, self.weights_per_step_schedule, x, offset_cpu)

        if state is not None:
            state.offset[:] = torch.where(state.exec_mask, state.offset + T, state.offset)
            state.offset_cpu += T
        return x


# =============================================================================
# Transformer Layer
# =============================================================================


@dataclass
class LayerState(StreamingState):
    offset_cpu: int = 0

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offset_cpu = 0


class MossAudioTokenizerTransformerLayer(StreamingModule):
    """Transformer layer with streaming support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        causal: bool = False,
        context: int | None = None,
        rope: MossAudioTokenizerRotaryEmbedding | None = None,
        norm: str = "layer_norm",
        layer_scale: float | None = None,
        gating: str = "none",
        weights_per_step: int = 0,
        weights_per_step_schedule: list[int] | None = None,
        activation=F.gelu,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = MossAudioTokenizerAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            causal=causal,
            context=context,
            rope=rope,
            weights_per_step=weights_per_step,
            weights_per_step_schedule=weights_per_step_schedule,
            **factory_kwargs,
        )
        self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)

        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        self.gating: nn.Module | nn.ModuleList | None = None
        self.linear1: nn.Module | None = None
        self.linear2: nn.Module | None = None
        self.activation = activation

        num_weights = 1
        if weights_per_step:
            num_weights = max(weights_per_step_schedule) + 1 if weights_per_step_schedule else weights_per_step

        if gating == "none":
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False, **factory_kwargs)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False, **factory_kwargs)
        else:
            if weights_per_step:
                dim_ff_list = [dim_feedforward] * num_weights if isinstance(dim_feedforward, int) else dim_feedforward
                self.gating = nn.ModuleList(
                    [make_gating(gating, d_model, dim, **factory_kwargs) for dim in dim_ff_list]
                )
            else:
                self.gating = make_gating(gating, d_model, dim_feedforward, **factory_kwargs)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = MossAudioTokenizerLayerScale(
                channels=d_model, init=layer_scale, channel_last=True, **cast(dict[str, object], factory_kwargs)
            )
            self.layer_scale_2 = MossAudioTokenizerLayerScale(
                channels=d_model, init=layer_scale, channel_last=True, **cast(dict[str, object], factory_kwargs)
            )

    def _init_streaming_state(self, batch_size: int) -> LayerState:
        device = next(iter(self.parameters())).device
        return LayerState(batch_size, device, offset_cpu=0)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        state = self._streaming_state
        offset = state.offset_cpu if isinstance(state, LayerState) else 0

        x_orig = x
        x = self.norm2(x)

        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                update = apply_weights_per_step(self.gating, self.weights_per_step_schedule, x, offset)
            else:
                update = self.gating(x)
        return x_orig.to(update) + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor):
        x_orig = x
        x = self.norm1(x)
        update = self.self_attn(x, x, x)
        return x_orig.to(update) + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor):
        x = self._sa_block(x)
        x = self._ff_block(x)
        state = self._streaming_state
        if state is not None:
            assert isinstance(state, LayerState)
            state.offset_cpu += x.shape[1]
        return x


# =============================================================================
# Streaming Transformer
# =============================================================================


@dataclass
class TransformerState(StreamingState):
    offsets: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offsets[:] = torch.where(reset_mask, torch.zeros_like(self.offsets), self.offsets)


class MossAudioTokenizerTransformer(StreamingModule):
    """Transformer with streaming/causal support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        causal: bool = False,
        context: int | None = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got d_model={d_model}, num_heads={num_heads}")

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale

        self.rope: MossAudioTokenizerRotaryEmbedding | None = None
        if positional_embedding in {"rope", "sin_rope"}:
            self.rope = MossAudioTokenizerRotaryEmbedding(max_period=max_period, head_dim=d_model // num_heads)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                MossAudioTokenizerTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def _init_streaming_state(self, batch_size: int) -> TransformerState:
        device = next(self.parameters()).device
        return TransformerState(
            batch_size,
            device,
            offsets=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape
        state = self._streaming_state
        offsets = (
            torch.zeros(1, dtype=torch.long, device=x.device)
            if state is None
            else (
                state.offsets
                if isinstance(state, TransformerState)
                else torch.zeros(1, dtype=torch.long, device=x.device)
            )
        )

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        if state is not None:
            assert isinstance(state, TransformerState)
            state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)
        return x


class MossAudioTokenizerProjectedTransformer(StreamingContainer):
    """Transformer with input/output projections."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        d_model: int,
        *,
        conv_layout: bool = False,
        module_type: str,
        **kwargs,
    ):
        super().__init__()
        self.module_type = module_type
        self.downsample_ratio: int = 1
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.input_proj = (
            nn.Linear(input_dimension, d_model, bias=False) if d_model != input_dimension else nn.Identity()
        )
        self.transformer = MossAudioTokenizerTransformer(d_model=d_model, **kwargs)
        self.conv_layout = conv_layout
        self.output_proj = (
            nn.Linear(d_model, output_dimension, bias=False) if d_model != output_dimension else nn.Identity()
        )

    def forward(self, x, input_lengths, *args, **kwargs):
        x = self.input_proj(x.transpose(1, 2))  # (B, D, T) -> (B, T, D)
        x = self.transformer(x, *args, **kwargs)
        x = self.output_proj(x).transpose(1, 2)  # (B, T, D) -> (B, D, T)
        return x, input_lengths


# =============================================================================
# Patched Pretransform Module
# =============================================================================


class MossAudioTokenizerPatchedPretransform(nn.Module):
    """Patching module for downsampling/upsampling."""

    def __init__(self, patch_size: int, is_downsample: bool, module_type: str, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio: int = patch_size
        self.is_downsample = is_downsample
        self.module_type = module_type

    def encode(self, x, input_lengths):
        b, d, _ = x.shape
        h = self.patch_size
        x = x.reshape(b, d, -1, h).permute(0, 1, 3, 2).reshape(b, d * h, -1)
        # We pad the input waveform to a multiple of `downsample_rate` before applying the encoder.
        # Use a ceil division to match that padding and avoid dropping the last (partially padded) frame.
        output_lengths = input_lengths // self.patch_size
        return x, output_lengths

    def decode(self, x, input_lengths):
        b, dh, l = x.shape
        h = self.patch_size
        d = dh // h
        x = x.reshape(b, d, h, l).permute(0, 1, 3, 2).reshape(b, d, l * h)
        output_lengths = input_lengths * self.patch_size
        return x, output_lengths

    def forward(self, x, input_lengths):
        if self.is_downsample:
            return self.encode(x, input_lengths)
        else:
            return self.decode(x, input_lengths)


# =============================================================================
# Vector Quantization
# =============================================================================


def WNConv1d(*args, **kwargs):
    """Weight-normalized Conv1d."""
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


class MossAudioTokenizerVectorQuantize(nn.Module):
    """Single codebook vector quantization (inference only)."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if input_dim != codebook_dim:
            self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor of shape (B, D, T)
        Returns:
            z_q: Quantized tensor of shape (B, D, T)
            indices: Code indices of shape (B, T)
            z_e: Encoded tensor before quantization
        """
        z = z.float()
        z_e = self.in_proj(z).float()

        encodings = z_e.transpose(1, 2).reshape(-1, z_e.shape[1])

        codebook_weight = self.codebook.weight
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook_weight.float().t()
            + codebook_weight.float().pow(2).sum(1, keepdim=True).t()
        )

        indices = (-dist).max(1)[1]
        indices = indices.reshape(z.size(0), -1)

        z_q = self.decode_code(indices)
        z_q = self.out_proj(z_q).float()

        return z_q, indices, z_e

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Decode code indices to embeddings."""
        return self.codebook(embed_id).transpose(1, 2).float()


class MossAudioTokenizerLFQ(nn.Module):
    """LFQ (inference-only) used by ResidualLFQ."""

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if self.input_dim != self.codebook_dim:
            self.in_proj = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_proj = WNConv1d(self.codebook_dim, self.input_dim, kernel_size=1)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize z into codebook vectors."""
        z = z.float()
        z_e = self.in_proj(z).float()
        z_q, indices = self.decode_latents(z_e)
        z_q = (z_e + (z_q - z_e).detach()).float()
        z_q = self.out_proj(z_q).float()
        return z_q, indices, z_e

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code_wo_out_proj(self, embed_id: torch.Tensor) -> torch.Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        z_q = self.decode_code_wo_out_proj(embed_id).float()
        z_q = self.out_proj(z_q).float()
        return z_q

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match training LFQ: L2-normalize then argmin squared distance."""
        encodings = latents.transpose(1, 2).reshape(-1, latents.shape[1]).float()
        codebook = self.codebook.weight.float()

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1]
        indices = indices.reshape(latents.size(0), -1)
        z_q = self.decode_code_wo_out_proj(indices).float()
        return z_q, indices


class MossAudioTokenizerResidualVQ(nn.Module):
    """Residual Vector Quantization (inference only)."""

    def __init__(
        self,
        input_dim: int = 1024,
        rvq_dim: int | None = None,
        output_dim: int | None = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim or input_dim
        self.output_dim = output_dim or input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.input_proj = (
            WNConv1d(input_dim, self.rvq_dim, kernel_size=1) if input_dim != self.rvq_dim else nn.Identity()
        )
        self.output_proj = (
            WNConv1d(self.rvq_dim, self.output_dim, kernel_size=1)
            if self.rvq_dim != self.output_dim
            else nn.Identity()
        )

        self.quantizers = nn.ModuleList(
            [
                MossAudioTokenizerVectorQuantize(
                    input_dim=self.rvq_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,
        input_length: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor of shape (B, D, T)
            input_length: Valid lengths for each sample (B,)
            n_quantizers: Number of quantizers to use
        Returns:
            quantized_out: Quantized output (B, D, T)
            all_indices: All code indices (N, B, T)
            output_length: Output lengths (B,)
        """
        z = self.input_proj(z)

        batch_size, _, max_time = z.shape
        mask = torch.arange(max_time, device=z.device).expand(batch_size, max_time) < input_length.unsqueeze(1)

        quantized_out = torch.zeros_like(z, dtype=torch.float32)
        residual = z.clone().float()
        all_indices = []

        n_quantizers = n_quantizers or self.num_quantizers

        for i, quantizer in enumerate(self.quantizers):
            if i >= n_quantizers:
                break

            masked_residual = residual * mask.unsqueeze(1)
            z_q_i, indices_i, _ = quantizer(masked_residual)

            update_mask = mask.unsqueeze(1)
            quantized_out = quantized_out + z_q_i * update_mask
            residual = residual - z_q_i * update_mask
            all_indices.append(indices_i)

        all_indices = torch.stack(all_indices)  # (N, B, T)
        quantized_out = self.output_proj(quantized_out)

        return quantized_out, all_indices, input_length

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes from multiple quantizers to embeddings."""
        nq, B, T = codes.shape
        emb = torch.zeros(B, self.rvq_dim, T, device=codes.device, dtype=torch.float32)

        for i, quantizer in enumerate(self.quantizers[:nq]):
            quantizer = cast(MossAudioTokenizerVectorQuantize, quantizer)
            quantized_i = quantizer.decode_code(codes[i])
            emb += quantized_i

        emb = self.output_proj(emb)
        return emb


class MossAudioTokenizerResidualLFQ(nn.Module):
    """Residual LFQ (inference only)."""

    def __init__(
        self,
        input_dim: int = 1024,
        rvq_dim: int | None = None,
        output_dim: int | None = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim or input_dim
        self.output_dim = output_dim or input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.input_proj = (
            WNConv1d(input_dim, self.rvq_dim, kernel_size=1) if input_dim != self.rvq_dim else nn.Identity()
        )
        self.output_proj = (
            WNConv1d(self.rvq_dim, self.output_dim, kernel_size=1)
            if self.rvq_dim != self.output_dim
            else nn.Identity()
        )

        self.quantizers = nn.ModuleList(
            [
                MossAudioTokenizerLFQ(
                    input_dim=self.rvq_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    @torch.no_grad()
    def forward(
        self,
        z: torch.Tensor,
        input_length: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference quantization."""
        z = self.input_proj(z).float()

        batch_size, _, max_time = z.shape
        mask = torch.arange(max_time, device=z.device).expand(batch_size, max_time) < input_length.unsqueeze(1)

        quantized_out = torch.zeros_like(z, dtype=torch.float32)
        residual = z.clone().float()
        all_indices = []

        n_quantizers = n_quantizers or self.num_quantizers
        for i, quantizer in enumerate(self.quantizers):
            if i >= n_quantizers:
                break

            masked_residual = residual * mask.unsqueeze(1)
            z_q_i, indices_i, _ = quantizer(masked_residual)

            update_mask = mask.unsqueeze(1)
            quantized_out = quantized_out + z_q_i * update_mask
            residual = residual - z_q_i * update_mask
            all_indices.append(indices_i)

        all_indices = (
            torch.stack(all_indices)
            if all_indices
            else torch.empty(0, batch_size, max_time, device=z.device, dtype=torch.long)
        )
        quantized_out = self.output_proj(quantized_out)
        return quantized_out, all_indices, input_length

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        nq, B, T = codes.shape
        emb = torch.zeros(B, self.rvq_dim, T, device=codes.device, dtype=torch.float32)
        for i, quantizer in enumerate(self.quantizers[:nq]):
            quantizer = cast(MossAudioTokenizerLFQ, quantizer)
            emb += quantizer.decode_code(codes[i]).float()
        emb = self.output_proj(emb)
        return emb


# =============================================================================
# Main Model Classes
# =============================================================================


@auto_docstring
class MossAudioTokenizerPreTrainedModel(PreTrainedAudioTokenizerBase):
    """Base class for MossAudioTokenizer models."""

    config_class = MossAudioTokenizerConfig
    base_model_prefix = ""
    main_input_name = "input_values"
    input_modalities = "audio"
    supports_gradient_checkpointing = False
    _no_split_modules = [
        "MossAudioTokenizerTransformerLayer",
        "MossAudioTokenizerResidualVQ",
        "MossAudioTokenizerResidualLFQ",
    ]


@auto_docstring(
    custom_intro="""
    The MossAudioTokenizer neural audio codec model for audio tokenization and synthesis.
    """
)
class MossAudioTokenizerModel(MossAudioTokenizerPreTrainedModel):
    """
    MossAudioTokenizer model for audio tokenization and synthesis.

    This model can encode audio waveforms into discrete tokens and decode
    tokens back into audio waveforms.
    """

    def __init__(self, config: MossAudioTokenizerConfig):
        super().__init__(config)

        _ = config.version

        # Build encoder
        current_frame_rate: float = float(self.config.sampling_rate)
        self.encoder = nn.ModuleList()

        for encoder_kwargs_i in config.encoder_config.to_module_configs():
            encoder_kwargs_i = dict(encoder_kwargs_i)  # Make a copy
            if encoder_kwargs_i["module_type"] == "PatchedPretransform":
                self.encoder.append(MossAudioTokenizerPatchedPretransform(**encoder_kwargs_i, is_downsample=True))
            elif encoder_kwargs_i["module_type"] == "Transformer":
                self.encoder.append(
                    MossAudioTokenizerProjectedTransformer(
                        **encoder_kwargs_i,
                        context=int(current_frame_rate * self.config.causal_transformer_context_duration),
                    )
                )
            current_frame_rate /= self.encoder[-1].downsample_ratio

        # Build quantizer
        quantizer_kwargs = _quantizer_config_to_kwargs(config.quantizer_config)
        quantizer_type = quantizer_kwargs.get("quantizer_type", getattr(config, "quantizer_type", "rvq"))
        if quantizer_type in {"rvq", "spec_rvq"}:
            self.quantizer = MossAudioTokenizerResidualVQ(**quantizer_kwargs)
        elif quantizer_type in {"rlfq", "random_prefix_rlfq"}:
            self.quantizer = MossAudioTokenizerResidualLFQ(**quantizer_kwargs)
        else:
            raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")

        # Build decoder
        self.decoder = nn.ModuleList()

        for decoder_kwargs_i in config.decoder_config.to_module_configs():
            decoder_kwargs_i = dict(decoder_kwargs_i)
            if decoder_kwargs_i["module_type"] == "PatchedPretransform":
                self.decoder.append(MossAudioTokenizerPatchedPretransform(**decoder_kwargs_i, is_downsample=False))
            elif decoder_kwargs_i["module_type"] == "Transformer":
                self.decoder.append(
                    MossAudioTokenizerProjectedTransformer(
                        **decoder_kwargs_i,
                        context=int(current_frame_rate * self.config.causal_transformer_context_duration),
                    )
                )
            current_frame_rate *= self.decoder[-1].downsample_ratio

        self.post_init()

    def _start_streaming(self, batch_size: int):
        """Start streaming mode for all modules."""

        def _start(module):
            if isinstance(module, StreamingModule) or getattr(module, "_supports_streaming", False):
                module._streaming_state = module._init_streaming_state(batch_size)

        self.apply(_start)

    def _stop_streaming(self):
        """Stop streaming mode for all modules."""

        def _stop(module):
            if isinstance(module, StreamingModule) or getattr(module, "_supports_streaming", False):
                module._streaming_state = None

        self.apply(_stop)

    @contextmanager
    def streaming(self, batch_size: int = 1):
        """Context manager for streaming mode."""
        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming()

    @torch.no_grad()
    def batch_encode(
        self, wav_list: list[torch.Tensor], num_quantizers: int | None = None
    ) -> MossAudioTokenizerEncoderOutput:
        """Batch encode a list of audio waveforms.

        Args:
            wav_list: List of audio tensors, each of shape `(num_samples,)`.
            num_quantizers: Number of quantizers to use. By default, all quantizers are used.

        Returns:
            [`MossAudioTokenizerEncoderOutput`] with `audio_codes` and `audio_codes_lengths`.
        """
        if len(wav_list) == 0:
            raise ValueError("`wav_list` must contain at least one waveform.")

        device = wav_list[0].device
        batch_size = len(wav_list)

        max_length = max(wav.shape[-1] for wav in wav_list)
        input_values = torch.zeros(batch_size, 1, max_length, device=device)
        input_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)

        for i, wav in enumerate(wav_list):
            input_values[i, 0, : wav.shape[-1]] = wav
            input_lengths[i] = wav.shape[-1]

        return self._encode_frame(input_values, input_lengths, n_quantizers=num_quantizers)

    @torch.no_grad()
    def batch_decode(
        self, codes_list: list[torch.Tensor], num_quantizers: int | None = None
    ) -> MossAudioTokenizerDecoderOutput:
        """Batch decode a list of audio codes.

        Args:
            codes_list: List of audio code tensors, each of shape `(num_quantizers, codes_length)`.
            num_quantizers: If provided, decode only the first `num_quantizers` quantizers from each element in
                `codes_list`. If omitted, all elements in `codes_list` must have the same number of quantizers.

        Returns:
            [`MossAudioTokenizerDecoderOutput`] with `audio` and `audio_lengths`.
        """
        if len(codes_list) == 0:
            raise ValueError("`codes_list` must contain at least one code tensor.")

        batch_size = len(codes_list)
        device = codes_list[0].device
        nqs = [codes.shape[0] for codes in codes_list]
        if num_quantizers is None:
            num_quantizers = nqs[0]
            if any(nq != num_quantizers for nq in nqs):
                raise ValueError(
                    "All elements in `codes_list` must have the same number of quantizers when `num_quantizers` is None. "
                    "Pass `num_quantizers=...` to decode a common prefix."
                )
        else:
            min_nq = min(nqs)
            if min_nq < num_quantizers:
                raise ValueError(
                    "`num_quantizers` must be <= the number of quantizers for every element in `codes_list`. "
                    f"Got num_quantizers={num_quantizers}, min(codes.shape[0])={min_nq}."
                )
        max_length = max(codes.shape[-1] for codes in codes_list)

        audio_codes = torch.zeros(num_quantizers, batch_size, max_length, device=device, dtype=torch.long)
        audio_codes_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)

        for i, codes in enumerate(codes_list):
            codes = codes[:num_quantizers]
            audio_codes[:, i, : codes.shape[-1]] = codes
            audio_codes_lengths[i] = codes.shape[-1]

        return self._decode_frame(audio_codes, audio_codes_lengths)

    @torch.no_grad()
    def _encode_frame(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        n_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        """Tokenize audio waveform into discrete tokens."""
        # Handle input shape
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        B, _, T = input_values.shape
        device = input_values.device

        if input_lengths is None:
            input_lengths = torch.full((B,), T, device=device, dtype=torch.long)

        # Pad to multiple of downsample_rate
        if T % self.config.downsample_rate != 0:
            pad_length = self.config.downsample_rate - (T % self.config.downsample_rate)
            input_values = F.pad(input_values, (0, pad_length))

        # Encode
        e, e_lengths = input_values, input_lengths
        for encoder_module in self.encoder:
            e, e_lengths = encoder_module(e, e_lengths)

        # Quantize
        quantizer = cast(MossAudioTokenizerResidualVQ | MossAudioTokenizerResidualLFQ, self.quantizer)
        zq, audio_codes, audio_codes_lengths = quantizer(e, e_lengths, n_quantizers)

        return MossAudioTokenizerEncoderOutput(
            audio_codes=audio_codes, audio_codes_lengths=audio_codes_lengths, encoder_hidden_states=e
        )

    @torch.no_grad()
    def _decode_frame(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        """Detokenize discrete tokens into audio waveform."""
        nq, B, T = codes.shape
        device = codes.device

        if codes_lengths is None:
            codes_lengths = torch.full((B,), T, device=device, dtype=torch.long)

        # Decode from codes
        quantizer = cast(MossAudioTokenizerResidualVQ | MossAudioTokenizerResidualLFQ, self.quantizer)
        zq = quantizer.decode_codes(codes)

        d, d_lengths = zq, codes_lengths
        for decoder_module in self.decoder:
            d, d_lengths = decoder_module(d, d_lengths)

        return MossAudioTokenizerDecoderOutput(audio=d, audio_lengths=d_lengths)

    def encode(  # type: ignore[override]
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        return_dict: bool | None = None,
        chunk_duration: float | None = None,
    ):
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to indicate valid audio samples.
            num_quantizers (`int`, *optional*):
                Number of quantizers to use. By default, all quantizers are used.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            chunk_duration (`float`, *optional*):
                If provided, encode the input waveform in successive chunks of `chunk_duration` seconds while keeping a
                streaming KV cache for the causal transformers.

                `chunk_duration` must be <= `config.causal_transformer_context_duration`, and
                `chunk_duration * config.sampling_rate` must be divisible by `config.downsample_rate`.

        Returns:
            `MossAudioTokenizerEncoderOutput` or tuple containing audio codes and lengths.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Handle input shape
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        B, _, T = input_values.shape
        device = input_values.device

        if padding_mask is not None:
            input_lengths = padding_mask.sum(dim=-1).long()
        else:
            input_lengths = torch.full((B,), T, device=device, dtype=torch.long)

        if chunk_duration is None:
            encoder_output = self._encode_frame(input_values, input_lengths, num_quantizers)
        else:
            if chunk_duration <= 0:
                raise ValueError("`chunk_duration` must be > 0 when provided.")
            if chunk_duration > self.config.causal_transformer_context_duration:
                raise ValueError(
                    "`chunk_duration` must be <= `config.causal_transformer_context_duration` "
                    f"({self.config.causal_transformer_context_duration}), got {chunk_duration}."
                )
            if B != 1:
                raise ValueError("Streaming encode via `chunk_duration` currently only supports batch_size=1.")

            chunk_length = int(round(chunk_duration * self.config.sampling_rate))
            if chunk_length <= 0:
                raise ValueError("`chunk_duration` is too small and results in chunk_length <= 0.")
            if chunk_length % self.config.downsample_rate != 0:
                raise ValueError(
                    "`chunk_duration * config.sampling_rate` must be divisible by `config.downsample_rate`. "
                    f"Got chunk_length={chunk_length}, downsample_rate={self.config.downsample_rate}."
                )

            input_length = int(input_lengths[0].item())
            if input_length <= chunk_length:
                encoder_output = self._encode_frame(input_values[..., :input_length], input_lengths, num_quantizers)
            else:
                codes_chunks: list[torch.Tensor] = []
                hidden_chunks: list[torch.Tensor] = []

                with ExitStack() as exit_stack:
                    for encoder_module in self.encoder:
                        if isinstance(encoder_module, StreamingModule):
                            exit_stack.enter_context(encoder_module.streaming(batch_size=B))

                    for start_idx in range(0, input_length, chunk_length):
                        input_length_i = min(chunk_length, input_length - start_idx)
                        if input_length_i <= 0:
                            break

                        input_lengths_i = torch.tensor([input_length_i], device=device, dtype=torch.long)
                        input_values_i = input_values[..., start_idx : start_idx + input_length_i]
                        result_i = self._encode_frame(input_values_i, input_lengths_i, num_quantizers)

                        if result_i.audio_codes is None or result_i.audio_codes_lengths is None:
                            raise RuntimeError("Internal error: `_encode_frame` returned empty audio codes.")
                        if result_i.encoder_hidden_states is None:
                            raise RuntimeError("Internal error: `_encode_frame` returned empty encoder hidden states.")

                        codes_length_i = result_i.audio_codes_lengths
                        codes_chunks.append(result_i.audio_codes[:, :, : codes_length_i[0]])
                        hidden_chunks.append(result_i.encoder_hidden_states[:, :, : codes_length_i[0]])

                audio_codes = torch.cat(codes_chunks, dim=-1)
                encoder_hidden_states = torch.cat(hidden_chunks, dim=-1)
                audio_codes_lengths = torch.tensor([audio_codes.shape[-1]], device=device, dtype=torch.long)
                encoder_output = MossAudioTokenizerEncoderOutput(
                    audio_codes=audio_codes,
                    audio_codes_lengths=audio_codes_lengths,
                    encoder_hidden_states=encoder_hidden_states,
                )

        if not return_dict:
            assert encoder_output.audio_codes is not None
            assert encoder_output.audio_codes_lengths is not None
            return (
                cast(torch.Tensor, encoder_output.audio_codes),
                cast(torch.Tensor, encoder_output.audio_codes_lengths),
            )
        return encoder_output

    def decode(  # type: ignore[override]
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
        chunk_duration: float | None = None,
        num_quantizers: int | None = None,
    ):
        """
        Decodes the given codes into an output audio waveform.

        Args:
            audio_codes (`torch.LongTensor` of shape `(num_quantizers, batch_size, sequence_length)`):
                Discrete code embeddings computed using `model.encode`.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to indicate valid code positions.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            chunk_duration (`float`, *optional*):
                If provided, decode the input codes in successive chunks of `chunk_duration` seconds while keeping a
                streaming KV cache for the causal transformers.

            num_quantizers (`int`, *optional*):
                Number of quantizers to use. By default, all quantizers in `audio_codes` are used.

                `chunk_duration` must be <= `config.causal_transformer_context_duration`, and
                `chunk_duration * config.sampling_rate` must be divisible by `config.downsample_rate`.

        Returns:
            `MossAudioTokenizerDecoderOutput` or tuple containing decoded audio.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(1)  # nq, T -> nq, B=1, T

        if num_quantizers is not None:
            if num_quantizers > audio_codes.shape[0]:
                raise ValueError(
                    f"`num_quantizers` ({num_quantizers}) must be <= audio_codes.shape[0] ({audio_codes.shape[0]})."
                )
            audio_codes = audio_codes[:num_quantizers]

        _, B, T = audio_codes.shape
        device = audio_codes.device

        if padding_mask is not None:
            codes_lengths = padding_mask.sum(dim=-1).long()
        else:
            codes_lengths = torch.full((B,), T, device=device, dtype=torch.long)

        if chunk_duration is None:
            decoder_output = self._decode_frame(audio_codes, codes_lengths)
        else:
            if chunk_duration <= 0:
                raise ValueError("`chunk_duration` must be > 0 when provided.")
            if chunk_duration > self.config.causal_transformer_context_duration:
                raise ValueError(
                    "`chunk_duration` must be <= `config.causal_transformer_context_duration` "
                    f"({self.config.causal_transformer_context_duration}), got {chunk_duration}."
                )
            if B != 1:
                raise ValueError("Streaming decode via `chunk_duration` currently only supports batch_size=1.")

            chunk_length = int(round(chunk_duration * self.config.sampling_rate))
            if chunk_length <= 0:
                raise ValueError("`chunk_duration` is too small and results in chunk_length <= 0.")
            if chunk_length % self.config.downsample_rate != 0:
                raise ValueError(
                    "`chunk_duration * config.sampling_rate` must be divisible by `config.downsample_rate`. "
                    f"Got chunk_length={chunk_length}, downsample_rate={self.config.downsample_rate}."
                )

            chunk_frame_length = chunk_length // self.config.downsample_rate
            codes_length = int(codes_lengths[0].item())
            if codes_length <= chunk_frame_length:
                decoder_output = self._decode_frame(audio_codes[..., :codes_length], codes_lengths)
            else:
                wav_chunks: list[torch.Tensor] = []
                with ExitStack() as exit_stack:
                    for decoder_module in self.decoder:
                        if isinstance(decoder_module, StreamingModule):
                            exit_stack.enter_context(decoder_module.streaming(batch_size=B))

                    for start_idx in range(0, codes_length, chunk_frame_length):
                        codes_length_i = min(chunk_frame_length, codes_length - start_idx)
                        if codes_length_i <= 0:
                            break

                        codes_lengths_i = torch.tensor([codes_length_i], device=device, dtype=torch.long)
                        codes_i = audio_codes[:, :, start_idx : start_idx + codes_length_i]
                        result_i = self._decode_frame(codes_i, codes_lengths_i)

                        if result_i.audio is None or result_i.audio_lengths is None:
                            raise RuntimeError("Internal error: `_decode_frame` returned empty audio.")

                        wav_chunks.append(result_i.audio[:, :, : result_i.audio_lengths[0]])

                wav = torch.cat(wav_chunks, dim=-1)
                audio_lengths = torch.tensor([wav.shape[-1]], device=device, dtype=torch.long)
                decoder_output = MossAudioTokenizerDecoderOutput(audio=wav, audio_lengths=audio_lengths)

        if not return_dict:
            assert decoder_output.audio is not None
            return (cast(torch.Tensor, decoder_output.audio),)
        return decoder_output

    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        audio_codes: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | MossAudioTokenizerOutput:  # type: ignore[override]
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Raw audio input converted to Float.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid computing on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_codes (`torch.LongTensor` of shape `(num_quantizers, batch_size, sequence_length)`, *optional*):
            Discrete code embeddings computed using `model.encode`.
        num_quantizers (`int`, *optional*):
            Number of quantizers (codebooks) to use. By default, all quantizers are used.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import MossAudioTokenizerModel

        >>> model = MossAudioTokenizerModel.from_pretrained("moss_audio_tokenizer-model")

        >>> # Create dummy audio input
        >>> audio = torch.randn(1, 1, 24000)  # 1 second of audio at 24kHz

        >>> outputs = model(input_values=audio)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        output_audio_codes: torch.Tensor | None = None
        output_audio_codes_lengths: torch.Tensor | None = None
        output_audio: torch.Tensor | None = None
        output_audio_lengths: torch.Tensor | None = None
        decoded_from_encoded_codes = False

        # Encode if input_values provided
        if input_values is not None:
            encoder_output = self.encode(input_values, padding_mask, num_quantizers, return_dict=True)
            encoder_output = cast(MossAudioTokenizerEncoderOutput, encoder_output)
            output_audio_codes = encoder_output.audio_codes
            output_audio_codes_lengths = encoder_output.audio_codes_lengths

            # If codes not provided separately, use encoded codes for decoding
            if audio_codes is None:
                audio_codes = output_audio_codes
                decoded_from_encoded_codes = True

        # Decode if codes available
        if audio_codes is not None:
            # If we're decoding the codes we just produced, use the computed lengths so we don't decode padded garbage.
            if decoded_from_encoded_codes and output_audio_codes_lengths is not None:
                decoder_output = self._decode_frame(audio_codes, output_audio_codes_lengths)
            else:
                decoder_output = self.decode(
                    audio_codes,
                    padding_mask=padding_mask,
                    return_dict=True,
                    num_quantizers=num_quantizers,
                )
                decoder_output = cast(MossAudioTokenizerDecoderOutput, decoder_output)
            output_audio = decoder_output.audio
            output_audio_lengths = decoder_output.audio_lengths

        if not return_dict:
            return (output_audio_codes, output_audio, output_audio_lengths)

        return MossAudioTokenizerOutput(
            audio=output_audio,
            audio_lengths=output_audio_lengths,
            audio_codes=output_audio_codes,
            audio_codes_lengths=output_audio_codes_lengths,
        )


__all__ = ["MossAudioTokenizerModel", "MossAudioTokenizerPreTrainedModel"]
