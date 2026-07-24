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
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import DynamicSlidingWindowLayer
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedAudioTokenizerBase
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_moss_audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerDecoderConfig,
    MossAudioTokenizerEncoderConfig,
    MossAudioTokenizerQuantizerConfig,
)


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
class MossAudioTokenizerEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, sequence_length)`, *optional*):
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
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, sequence_length)`, *optional*):
        Discrete audio codes computed using the encoder and quantizer.
    audio_codes_lengths (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Valid lengths for each sample's audio codes.
    """

    audio: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None
    audio_codes: torch.Tensor | None = None
    audio_codes_lengths: torch.Tensor | None = None


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


class MossAudioTokenizerLayerScale(nn.Module):
    """Layer scale from Touvron et al. 2021."""

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init, requires_grad=True, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor):
        return self.scale * hidden_states


class MossAudioTokenizerRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: PreTrainedConfig, device=None):
        super().__init__(config, device=device)

    @staticmethod
    def compute_default_rope_parameters(
        config: PreTrainedConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


def _create_rope_config(
    hidden_size: int,
    num_attention_heads: int,
    max_position_embeddings: int,
) -> PreTrainedConfig:
    return PreTrainedConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        head_dim=hidden_size // num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )


@dataclass
class MHAState(StreamingState):
    kv_cache: DynamicSlidingWindowLayer
    offset: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offset[:] = torch.where(reset_mask, torch.zeros_like(self.offset), self.offset)
        if self.kv_cache is not None:
            self.kv_cache.reset()


class MossAudioTokenizerAttention(LlamaAttention):
    """Multi-head attention with streaming support."""

    _supports_streaming = True

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        context: int,
        rope: MossAudioTokenizerRotaryEmbedding,
        device=None,
        dtype=None,
    ):
        nn.Module.__init__(self)
        self._streaming_state: MHAState | None = None
        self._streaming_detached: bool = False
        factory_kwargs = {"device": device, "dtype": dtype}

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads, "
                f"got hidden_size={hidden_size}, num_attention_heads={num_attention_heads}"
            )

        self.hidden_size = hidden_size
        self.context = context
        self.rope = rope
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.num_key_value_groups = 1
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self._attn_implementation = "sdpa"

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False, **factory_kwargs)
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False, **factory_kwargs)

        self._register_load_state_dict_pre_hook(self._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        for suffix in ["", "_scb"]:
            source = prefix + "in_proj_weight" + suffix
            target = prefix + "in_proj.weight" + suffix
            if source in state_dict:
                state_dict[target] = state_dict.pop(source)

    def _init_streaming_state(self, batch_size: int) -> MHAState:
        in_proj = cast(nn.Linear, self.in_proj)
        device = cast(torch.device, in_proj.weight.device)

        kv_cache = DynamicSlidingWindowLayer(sliding_window=self.context)
        return MHAState(
            batch_size,
            cast(torch.device, device),
            kv_cache,
            offset=torch.zeros(batch_size, device=cast(torch.device, device), dtype=torch.long),
        )

    def _complete_kv(
        self, key_states: torch.Tensor, value_states: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = cast(MHAState | None, self._streaming_state)
        if state is None:
            batch_size, _, sequence_length, _ = key_states.shape
            positions = torch.arange(sequence_length, device=key_states.device, dtype=torch.long)
            return key_states, value_states, positions.expand(batch_size, -1)

        current_sequence_length = key_states.shape[-2]
        completed_key_states, completed_value_states = state.kv_cache.update(key_states, value_states)
        key_length = completed_key_states.shape[-2]
        previous_length = key_length - current_sequence_length
        key_start = offset - previous_length
        positions = key_start.view(-1, 1) + torch.arange(key_length, device=key_states.device, dtype=torch.long)
        return completed_key_states, completed_value_states, positions

    def _prepare_attention_mask(
        self,
        query_states: torch.Tensor,
        key_positions: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor | None:
        _, _, query_length, _ = query_states.shape
        query_positions = offset.view(-1, 1, 1) + torch.arange(
            query_length, device=query_states.device, dtype=torch.long
        ).view(1, -1, 1)
        key_positions = key_positions[:, None, :]
        delta = query_positions - key_positions
        allowed = (key_positions >= 0) & (delta >= 0) & (delta < self.context)

        attention_mask = torch.zeros(allowed.shape, device=query_states.device, dtype=query_states.dtype)
        attention_mask = attention_mask.masked_fill(~allowed, torch.finfo(query_states.dtype).min)
        return attention_mask[:, None, :, :]

    def forward(self, hidden_states: torch.Tensor):
        state = cast(MHAState | None, self._streaming_state)
        batch_size, sequence_length = hidden_states.shape[:2]

        if state is None:
            offset = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.long)
        else:
            offset = state.offset

        projected_states = self.in_proj(hidden_states)
        projected_states = projected_states.reshape(
            batch_size, sequence_length, 3, self.num_attention_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = (
            projected_states[0],
            projected_states[1],
            projected_states[2],
        )

        position_ids = offset.view(batch_size, 1) + torch.arange(sequence_length, device=hidden_states.device).view(
            1, -1
        )
        cos, sin = self.rope(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states, value_states, key_positions = self._complete_kv(key_states, value_states, offset)
        attention_mask = self._prepare_attention_mask(query_states, key_positions, offset)
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self._attn_implementation, eager_attention_forward
        )
        attention_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=False,
        )
        attention_output = attention_output.reshape(batch_size, sequence_length, self.hidden_size)
        attention_output = self.out_proj(attention_output)

        if state is not None:
            state.offset[:] = torch.where(state.exec_mask, state.offset + sequence_length, state.offset)
        return attention_output


class MossAudioTokenizerTransformerLayer(StreamingModule):
    """Transformer layer with streaming support."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        context: int,
        rope: MossAudioTokenizerRotaryEmbedding,
        intermediate_size: int = 2048,
        layer_scale_init_value: float = 0.01,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = MossAudioTokenizerAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            context=context,
            rope=rope,
            **factory_kwargs,
        )
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5, **factory_kwargs)

        self.linear1 = nn.Linear(hidden_size, intermediate_size, bias=False, **factory_kwargs)
        self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=False, **factory_kwargs)

        self.layer_scale_1 = MossAudioTokenizerLayerScale(
            channels=hidden_size, init=layer_scale_init_value, **cast(dict[str, object], factory_kwargs)
        )
        self.layer_scale_2 = MossAudioTokenizerLayerScale(
            channels=hidden_size, init=layer_scale_init_value, **cast(dict[str, object], factory_kwargs)
        )

    def _ff_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        update = self.linear2(F.gelu(self.linear1(hidden_states)))
        return residual.to(update) + self.layer_scale_2(update)

    def _sa_block(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        update = self.self_attn(hidden_states)
        return residual.to(update) + self.layer_scale_1(update)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self._sa_block(hidden_states)
        hidden_states = self._ff_block(hidden_states)
        return hidden_states


class MossAudioTokenizerTransformer(StreamingModule):
    """Transformer with streaming/causal support."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        context: int,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads, "
                f"got hidden_size={hidden_size}, num_attention_heads={num_attention_heads}"
            )

        rope_config = _create_rope_config(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
        )
        self.rope = MossAudioTokenizerRotaryEmbedding(config=rope_config, device=device)

        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(
                MossAudioTokenizerTransformerLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, *args, **kwargs)
        return hidden_states


class MossAudioTokenizerProjectedTransformer(StreamingContainer):
    """Transformer with input/output projections."""

    def __init__(
        self,
        config: MossAudioTokenizerConfig,
        stage_index: int,
        context: int,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        input_hidden_size = config.input_hidden_sizes[stage_index]
        output_hidden_size = config.output_hidden_sizes[stage_index]
        hidden_size = config.hidden_sizes[stage_index]
        self.sampling_ratio: int = 1
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_proj = (
            nn.Linear(input_hidden_size, hidden_size, bias=False, **factory_kwargs)
            if hidden_size != input_hidden_size
            else nn.Identity()
        )
        self.transformer = MossAudioTokenizerTransformer(
            hidden_size=hidden_size,
            num_attention_heads=config.num_attention_heads[stage_index],
            num_hidden_layers=config.num_hidden_layers[stage_index],
            intermediate_size=config.intermediate_sizes[stage_index],
            layer_scale_init_value=config.layer_scale_init_value,
            max_position_embeddings=config.max_position_embeddings,
            context=context,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.output_proj = (
            nn.Linear(hidden_size, output_hidden_size, bias=False, **factory_kwargs)
            if hidden_size != output_hidden_size
            else nn.Identity()
        )

    def forward(self, hidden_states, input_lengths, *args, **kwargs):
        hidden_states = self.input_proj(hidden_states.transpose(1, 2))
        hidden_states = self.transformer(hidden_states, *args, **kwargs)
        hidden_states = self.output_proj(hidden_states).transpose(1, 2)
        return hidden_states, input_lengths


class MossAudioTokenizerDownsample(nn.Module):
    """Patching module for downsampling."""

    def __init__(self, sampling_ratio: int):
        super().__init__()
        self.sampling_ratio = sampling_ratio

    def forward(self, hidden_states, input_lengths):
        batch_size, num_channels, _ = hidden_states.shape
        sampling_ratio = self.sampling_ratio
        hidden_states = (
            hidden_states.reshape(batch_size, num_channels, -1, sampling_ratio)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, num_channels * sampling_ratio, -1)
        )
        output_lengths = torch.div(input_lengths + sampling_ratio - 1, sampling_ratio, rounding_mode="floor")
        return hidden_states, output_lengths


class MossAudioTokenizerUpsample(nn.Module):
    """Patching module for upsampling."""

    def __init__(self, sampling_ratio: int):
        super().__init__()
        self.sampling_ratio = sampling_ratio

    def forward(self, hidden_states, input_lengths):
        batch_size, patch_channels, sequence_length = hidden_states.shape
        sampling_ratio = self.sampling_ratio
        num_channels = patch_channels // sampling_ratio
        hidden_states = (
            hidden_states.reshape(batch_size, num_channels, sampling_ratio, sequence_length)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, num_channels, sequence_length * sampling_ratio)
        )
        output_lengths = input_lengths * self.sampling_ratio
        return hidden_states, output_lengths


class MossAudioTokenizerLFQ(nn.Module):
    """LFQ (inference-only) used by ResidualLFQ."""

    def __init__(self, config: MossAudioTokenizerQuantizerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim

        self.in_proj = nn.Conv1d(self.hidden_size, self.codebook_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(self.codebook_dim, self.hidden_size, kernel_size=1)

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize hidden states into codebook vectors."""
        hidden_states = hidden_states.float()
        projected_hidden_states = self.in_proj(hidden_states).float()
        quantized_hidden_states, indices = self.decode_latents(projected_hidden_states)
        quantized_hidden_states = (
            projected_hidden_states + (quantized_hidden_states - projected_hidden_states).detach()
        ).float()
        quantized_hidden_states = self.out_proj(quantized_hidden_states).float()
        return quantized_hidden_states, indices, projected_hidden_states

    def embed_code(self, code_indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(code_indices, self.codebook.weight)

    def decode_code_wo_out_proj(self, code_indices: torch.Tensor) -> torch.Tensor:
        return self.embed_code(code_indices).transpose(1, 2)

    def decode_code(self, code_indices: torch.Tensor) -> torch.Tensor:
        quantized_hidden_states = self.decode_code_wo_out_proj(code_indices).float()
        quantized_hidden_states = self.out_proj(quantized_hidden_states).float()
        return quantized_hidden_states

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match training LFQ: L2-normalize then argmin squared distance."""
        encodings = latents.transpose(1, 2).reshape(-1, latents.shape[1]).float()
        codebook = self.codebook.weight.float()

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        distances = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distances).max(1)[1]
        indices = indices.reshape(latents.size(0), -1)
        quantized_hidden_states = self.decode_code_wo_out_proj(indices).float()
        return quantized_hidden_states, indices


class MossAudioTokenizerResidualLFQ(nn.Module):
    """Residual LFQ (inference only)."""

    def __init__(self, config: MossAudioTokenizerQuantizerConfig):
        super().__init__()
        self.input_hidden_size = config.input_hidden_size
        self.hidden_size = config.hidden_size
        self.output_hidden_size = config.output_hidden_size
        self.n_codebooks = config.n_codebooks
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim

        self.input_proj = nn.Conv1d(self.input_hidden_size, self.hidden_size, kernel_size=1)
        self.output_proj = nn.Conv1d(self.hidden_size, self.output_hidden_size, kernel_size=1)

        self.quantizers = nn.ModuleList([MossAudioTokenizerLFQ(config) for _ in range(self.n_codebooks)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_length: torch.Tensor,
        num_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference quantization."""
        hidden_states = self.input_proj(hidden_states).float()

        batch_size, _, max_time = hidden_states.shape
        mask = torch.arange(max_time, device=hidden_states.device).expand(
            batch_size, max_time
        ) < input_length.unsqueeze(1)

        quantized_out = torch.zeros_like(hidden_states, dtype=torch.float32)
        residual = hidden_states.clone().float()
        all_indices = []

        num_quantizers = num_quantizers or self.n_codebooks
        for index, quantizer in enumerate(self.quantizers):
            if index >= num_quantizers:
                break

            masked_residual = residual * mask.unsqueeze(1)
            quantized_hidden_states, indices, _ = quantizer(masked_residual)

            update_mask = mask.unsqueeze(1)
            quantized_out = quantized_out + quantized_hidden_states * update_mask
            residual = residual - quantized_hidden_states * update_mask
            all_indices.append(indices)

        all_indices = (
            torch.stack(all_indices)
            if all_indices
            else torch.empty(0, batch_size, max_time, device=hidden_states.device, dtype=torch.long)
        )
        quantized_out = self.output_proj(quantized_out)
        return quantized_out, all_indices, input_length

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        num_quantizers, batch_size, sequence_length = codes.shape
        embeddings = torch.zeros(
            batch_size, self.hidden_size, sequence_length, device=codes.device, dtype=torch.float32
        )
        for index, quantizer in enumerate(self.quantizers[:num_quantizers]):
            quantizer = cast(MossAudioTokenizerLFQ, quantizer)
            embeddings += quantizer.decode_code(codes[index]).float()
        embeddings = self.output_proj(embeddings)
        return embeddings


class MossAudioTokenizerEncoder(StreamingContainer):
    """MOSS Audio Tokenizer encoder."""

    def __init__(self, config: MossAudioTokenizerEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        current_frame_rate = float(config.sampling_rate)

        for stage_index, sampling_ratio in enumerate(config.downsampling_ratios):
            self.layers.append(MossAudioTokenizerDownsample(sampling_ratio=sampling_ratio))
            current_frame_rate /= sampling_ratio
            self.layers.append(
                MossAudioTokenizerProjectedTransformer(
                    config=config,
                    stage_index=stage_index,
                    context=int(current_frame_rate * config.sliding_window_duration),
                )
            )

    def forward(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor):
        for layer in self.layers:
            hidden_states, input_lengths = layer(hidden_states, input_lengths)
        return hidden_states, input_lengths


class MossAudioTokenizerDecoder(StreamingContainer):
    """MOSS Audio Tokenizer decoder."""

    def __init__(self, config: MossAudioTokenizerDecoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        current_frame_rate = float(config.sampling_rate) / config.hop_length

        for stage_index, sampling_ratio in enumerate(config.upsampling_ratios):
            self.layers.append(
                MossAudioTokenizerProjectedTransformer(
                    config=config,
                    stage_index=stage_index,
                    context=int(current_frame_rate * config.sliding_window_duration),
                )
            )
            self.layers.append(MossAudioTokenizerUpsample(sampling_ratio=sampling_ratio))
            current_frame_rate *= sampling_ratio

    def forward(self, hidden_states: torch.Tensor, input_lengths: torch.Tensor):
        for layer in self.layers:
            hidden_states, input_lengths = layer(hidden_states, input_lengths)
        return hidden_states, input_lengths


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
        "MossAudioTokenizerResidualLFQ",
    ]

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        for module in self.modules():
            if isinstance(module, nn.Conv1d) and not nn.utils.parametrize.is_parametrized(module, "weight"):
                weight_norm(module)

    def remove_weight_norm(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and nn.utils.parametrize.is_parametrized(module, "weight"):
                nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


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

        self.encoder = MossAudioTokenizerEncoder(config.encoder_config)
        self.quantizer = MossAudioTokenizerResidualLFQ(config.quantizer_config)
        self.decoder = MossAudioTokenizerDecoder(config.decoder_config)

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

    def _encode_frame(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        """Tokenize audio waveform into discrete tokens."""
        # Handle input shape
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        batch_size, _, sequence_length = input_values.shape
        device = input_values.device

        if input_lengths is None:
            input_lengths = torch.full((batch_size,), sequence_length, device=device, dtype=torch.long)

        if sequence_length % self.config.hop_length != 0:
            raise ValueError(
                "`input_values` length must be a multiple of `config.hop_length`. "
                "Use `MossAudioTokenizerFeatureExtractor` to prepare and pad audio inputs."
            )

        encoder_hidden_states, encoder_hidden_states_lengths = self.encoder(input_values, input_lengths)

        quantizer = cast(MossAudioTokenizerResidualLFQ, self.quantizer)
        _, audio_codes, audio_codes_lengths = quantizer(
            encoder_hidden_states, encoder_hidden_states_lengths, num_quantizers
        )

        return MossAudioTokenizerEncoderOutput(
            audio_codes=audio_codes.transpose(0, 1).contiguous(),
            audio_codes_lengths=audio_codes_lengths,
            encoder_hidden_states=encoder_hidden_states,
        )

    @can_return_tuple
    @auto_docstring
    def encode(  # type: ignore[override]
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        chunk_duration: float | None = None,
    ):
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
            Float values of the input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to indicate valid audio samples.
        num_quantizers (`int`, *optional*):
            Number of quantizers to use. By default, all quantizers are used.
        chunk_duration (`float`, *optional*):
            If provided, encode the input waveform in successive chunks of `chunk_duration` seconds while keeping a
            streaming KV cache for the causal transformers.

            `chunk_duration` must be <= `config.sliding_window_duration`, and
            `chunk_duration * config.sampling_rate` must be divisible by `config.hop_length`.

        Returns:
            `MossAudioTokenizerEncoderOutput` or tuple containing audio codes and lengths.
        """
        # Handle input shape
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(1)

        batch_size, _, sequence_length = input_values.shape
        device = input_values.device

        if padding_mask is not None:
            input_lengths = padding_mask.sum(dim=-1).long()
        else:
            input_lengths = torch.full((batch_size,), sequence_length, device=device, dtype=torch.long)

        if chunk_duration is None:
            encoder_output = self._encode_frame(input_values, input_lengths, num_quantizers)
        else:
            if chunk_duration <= 0:
                raise ValueError("`chunk_duration` must be > 0 when provided.")
            if chunk_duration > self.config.sliding_window_duration:
                raise ValueError(
                    "`chunk_duration` must be <= `config.sliding_window_duration` "
                    f"({self.config.sliding_window_duration}), got {chunk_duration}."
                )
            if batch_size != 1:
                raise ValueError("Streaming encode via `chunk_duration` currently only supports batch_size=1.")

            chunk_length = int(round(chunk_duration * self.config.sampling_rate))
            if chunk_length <= 0:
                raise ValueError("`chunk_duration` is too small and results in chunk_length <= 0.")
            if chunk_length % self.config.hop_length != 0:
                raise ValueError(
                    "`chunk_duration * config.sampling_rate` must be divisible by `config.hop_length`. "
                    f"Got chunk_length={chunk_length}, hop_length={self.config.hop_length}."
                )

            input_length = int(input_lengths[0].item())
            padded_length = int(input_values.shape[-1])
            if input_length <= chunk_length:
                encoder_output = self._encode_frame(input_values, input_lengths, num_quantizers)
            else:
                codes_chunks: list[torch.Tensor] = []
                hidden_chunks: list[torch.Tensor] = []

                with self.encoder.streaming(batch_size=batch_size):
                    for start_idx in range(0, padded_length, chunk_length):
                        input_length_i = max(0, min(chunk_length, input_length - start_idx))
                        if input_length_i <= 0:
                            break

                        input_lengths_i = torch.tensor([input_length_i], device=device, dtype=torch.long)
                        input_values_i = input_values[..., start_idx : start_idx + chunk_length]
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

        return encoder_output

    @can_return_tuple
    @auto_docstring
    def decode(  # type: ignore[override]
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        chunk_duration: float | None = None,
        num_quantizers: int | None = None,
    ):
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, sequence_length)`):
            Discrete code embeddings computed using `model.encode`.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to indicate valid code positions.
        chunk_duration (`float`, *optional*):
            If provided, decode the input codes in successive chunks of `chunk_duration` seconds while keeping a
            streaming KV cache for the causal transformers.
        num_quantizers (`int`, *optional*):
            Number of quantizers to use. By default, all quantizers in `audio_codes` are used.

            `chunk_duration` must be <= `config.sliding_window_duration`, and
            `chunk_duration * config.sampling_rate` must be divisible by `config.hop_length`.

        Returns:
            `MossAudioTokenizerDecoderOutput` or tuple containing decoded audio.
        """
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(1)
        elif audio_codes.dim() == 3:
            audio_codes = audio_codes.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "`audio_codes` must have shape `(num_quantizers, sequence_length)` or `(batch_size, num_quantizers, sequence_length)`."
            )

        if num_quantizers is not None:
            if num_quantizers > audio_codes.shape[0]:
                raise ValueError(
                    f"`num_quantizers` ({num_quantizers}) must be <= audio_codes.shape[0] ({audio_codes.shape[0]})."
                )
            audio_codes = audio_codes[:num_quantizers]

        _, batch_size, sequence_length = audio_codes.shape
        device = audio_codes.device

        if padding_mask is not None:
            codes_lengths = padding_mask.sum(dim=-1).long()
        else:
            codes_lengths = torch.full((batch_size,), sequence_length, device=device, dtype=torch.long)

        if chunk_duration is None:
            quantizer = cast(MossAudioTokenizerResidualLFQ, self.quantizer)
            decoder_hidden_states = quantizer.decode_codes(audio_codes)
            audio, audio_lengths = self.decoder(decoder_hidden_states, codes_lengths)
            decoder_output = MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)
        else:
            if chunk_duration <= 0:
                raise ValueError("`chunk_duration` must be > 0 when provided.")
            if chunk_duration > self.config.sliding_window_duration:
                raise ValueError(
                    "`chunk_duration` must be <= `config.sliding_window_duration` "
                    f"({self.config.sliding_window_duration}), got {chunk_duration}."
                )
            if batch_size != 1:
                raise ValueError("Streaming decode via `chunk_duration` currently only supports batch_size=1.")

            chunk_length = int(round(chunk_duration * self.config.sampling_rate))
            if chunk_length <= 0:
                raise ValueError("`chunk_duration` is too small and results in chunk_length <= 0.")
            if chunk_length % self.config.hop_length != 0:
                raise ValueError(
                    "`chunk_duration * config.sampling_rate` must be divisible by `config.hop_length`. "
                    f"Got chunk_length={chunk_length}, hop_length={self.config.hop_length}."
                )

            chunk_frame_length = chunk_length // self.config.hop_length
            codes_length = int(codes_lengths[0].item())
            if codes_length <= chunk_frame_length:
                quantizer = cast(MossAudioTokenizerResidualLFQ, self.quantizer)
                decoder_hidden_states = quantizer.decode_codes(audio_codes[..., :codes_length])
                audio, audio_lengths = self.decoder(decoder_hidden_states, codes_lengths)
                decoder_output = MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)
            else:
                wav_chunks: list[torch.Tensor] = []
                quantizer = cast(MossAudioTokenizerResidualLFQ, self.quantizer)
                with self.decoder.streaming(batch_size=batch_size):
                    for start_idx in range(0, codes_length, chunk_frame_length):
                        codes_length_i = min(chunk_frame_length, codes_length - start_idx)
                        if codes_length_i <= 0:
                            break

                        codes_lengths_i = torch.tensor([codes_length_i], device=device, dtype=torch.long)
                        codes_i = audio_codes[:, :, start_idx : start_idx + codes_length_i]
                        decoder_hidden_states_i = quantizer.decode_codes(codes_i)
                        audio_i, audio_lengths_i = self.decoder(decoder_hidden_states_i, codes_lengths_i)
                        result_i = MossAudioTokenizerDecoderOutput(audio=audio_i, audio_lengths=audio_lengths_i)

                        if result_i.audio is None or result_i.audio_lengths is None:
                            raise RuntimeError("Internal error: decoder returned empty audio.")

                        wav_chunks.append(result_i.audio[:, :, : result_i.audio_lengths[0]])

                wav = torch.cat(wav_chunks, dim=-1)
                audio_lengths = torch.tensor([wav.shape[-1]], device=device, dtype=torch.long)
                decoder_output = MossAudioTokenizerDecoderOutput(audio=wav, audio_lengths=audio_lengths)

        return decoder_output

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        audio_codes: torch.Tensor | None = None,
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerOutput:  # type: ignore[override]
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Raw audio input converted to Float.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid computing on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, sequence_length)`, *optional*):
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
        output_audio_codes: torch.Tensor | None = None
        output_audio_codes_lengths: torch.Tensor | None = None
        output_audio: torch.Tensor | None = None
        output_audio_lengths: torch.Tensor | None = None
        decoded_from_encoded_codes = False

        if input_values is not None:
            encoder_output = self.encode(input_values, padding_mask, num_quantizers, return_dict=True)
            encoder_output = cast(MossAudioTokenizerEncoderOutput, encoder_output)
            output_audio_codes = encoder_output.audio_codes
            output_audio_codes_lengths = encoder_output.audio_codes_lengths

            if audio_codes is None:
                audio_codes = output_audio_codes
                decoded_from_encoded_codes = True

        if audio_codes is not None:
            audio_codes_padding_mask = padding_mask
            if decoded_from_encoded_codes and output_audio_codes_lengths is not None:
                code_positions = torch.arange(audio_codes.shape[-1], device=audio_codes.device)
                audio_codes_padding_mask = code_positions[None, :] < output_audio_codes_lengths[:, None]

            decoder_output = self.decode(
                audio_codes,
                padding_mask=audio_codes_padding_mask,
                return_dict=True,
                num_quantizers=num_quantizers,
            )
            decoder_output = cast(MossAudioTokenizerDecoderOutput, decoder_output)
            output_audio = decoder_output.audio
            output_audio_lengths = decoder_output.audio_lengths

        return MossAudioTokenizerOutput(
            audio=output_audio,
            audio_lengths=output_audio_lengths,
            audio_codes=output_audio_codes,
            audio_codes_lengths=output_audio_codes_lengths,
        )


__all__ = [
    "MossAudioTokenizerDecoder",
    "MossAudioTokenizerEncoder",
    "MossAudioTokenizerModel",
    "MossAudioTokenizerPreTrainedModel",
]
