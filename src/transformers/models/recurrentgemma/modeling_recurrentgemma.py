# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
""" PyTorch RecurrentGemma model."""

from dataclasses import dataclass
import math
import warnings
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import einops
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
)
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, \
    is_torch_greater_or_equal_than_1_13
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import is_torch_fx_available
from .configuration_recurrentgemma import RecurrentGemmaConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, \
        unpad_input  # noqa

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(
        _prepare_4d_causal_attention_mask)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RecurrentGemmaConfig"

_MAX_SQRT_GRADIENT = 1000.0
_MIN_LOGITS_VALUE = -2.3819763e38  # Set to a large negative number.
_MAX_WAVELENGTH = 10_000


# class RecurrentBlockCache(NamedTuple):
#     """The cache for a recurrent block."""
#
#     rg_lru_state: torch.Tensor
#     conv1d_state: torch.Tensor
#
#
# class AttentionBlockCache(NamedTuple):
#     """The cache for an attention block."""
#
#     keys: torch.Tensor
#     values: torch.Tensor
#     num_tokens: torch.Tensor
#
#
# ResidualBlockCache = RecurrentBlockCache | AttentionBlockCache


def _apply_rope(
    inputs: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> torch.Tensor:
    """Applies RoPE to the first half of inputs.

    Args:
      inputs: Queries or keys..
      positions: Positions of each token in the sequence.
      max_wavelength: The maximum wavelength used for the sin and cos.

    Returns:
      Rotated keys or queries in first half (along with original in second half).
    """
    batch_size, sequence_length, *_ = inputs.shape
    x_rope, x = torch.chunk(inputs, 2, dim=-1)
    positions = positions.reshape(1, sequence_length, 1, 1)

    freq = torch.arange(x_rope.shape[-1] // 2, device=x.device)
    freq_exponents = 2 * freq / x_rope.shape[-1]
    timescale = max_wavelength ** freq_exponents
    inv_frequencies = 1.0 / timescale

    sinusoid_imp = positions * inv_frequencies
    sin = torch.sin(sinusoid_imp).type_as(inputs)
    cos = torch.cos(sinusoid_imp).type_as(inputs)

    first_half, second_half = torch.chunk(x_rope, 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin

    return torch.concatenate([first_part, second_part, x], dim=-1)


def _compute_causal_mask(
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    window_size: int,
    q_segment_ids: torch.Tensor | None,
    k_segment_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Computes the causal mask for local attention.

    Args:
      q_positions: Position of each query token in the sequence.
      k_positions: Position of each key token in the sequence.
      window_size: The local attention window size.
      q_segment_ids: Optional segment id for each query token.
      k_segment_ids: Optional segment id for each key token.

    Returns:
      The mask that needs to be applied to the logits of the local attention.
    """
    # Mask for attending only to the same segment.
    if q_segment_ids is not None or k_segment_ids is not None:
        assert q_segment_ids is not None and k_segment_ids is not None
        same_segment_mask = q_segment_ids[..., None] == k_segment_ids[..., None,
                                                        :]
    else:
        same_segment_mask = (k_positions >= 0)[..., None, :]

    # Mask for attending only to previous tokens.
    causal_mask = q_positions[..., None] >= k_positions[..., None, :]

    # Mask for attending only to things within the window size.
    window_cond = q_positions[..., None] <= (
        k_positions[..., None, :] + window_size
    )

    mask = torch.logical_and(causal_mask, window_cond)
    mask = torch.logical_and(same_segment_mask, mask)
    return mask


def compute_forward_pass_mask(
    segment_pos: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Compute the forward pass mask.

    Args:
      segment_pos: Position of each token in the sequence.
      window_size: The local attention window size.

    Returns:
      The mask that needs to be applied to the logits when performing a forward
      pass (e.g. prompt processing) of the local attention.
    """
    segment_ids = torch.cumsum(segment_pos == 0, dim=-1)
    positions = torch.arange(segment_pos.shape[-1], device=segment_pos.device)
    return _compute_causal_mask(positions, positions, window_size, segment_ids, segment_ids)


def _compute_cache_mask(
    num_tokens: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Computes the mask when there a KV-cache is present.

    Args:
      num_tokens: The number of active tokens currently stored in the KV-cache.
      window_size: The local attention window size.

    Returns:
      The mask that needs to be applied to the logits when performing a single
      inference step with a KV-cache of the local attention.
    """
    device = num_tokens.device
    q_positions = num_tokens[None]
    k_positions = torch.arange(window_size + 1, device=device) - window_size
    k_positions = k_positions + num_tokens
    # Add batch dimension
    return _compute_causal_mask(q_positions, k_positions, window_size, None, None)


def _update_attention_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    cache: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Updates the cache with the new keys and values.

    Args:
      keys: The new keys to be added to the cache.
      values: The new values to be added to the cache.
      cache: The dictionary with the cache to be updated.

    Returns:
      The updated cache dictionary.
    """
    l = keys.shape[-3]
    window_size = cache["keys"].shape[-3]
    n_fill = min(window_size, l)

    new_keys = [cache["keys"][:, n_fill:], keys[:, -n_fill:]]
    new_values = [cache["values"][:, n_fill:], values[:, -n_fill:]]
    return dict(
        keys=torch.concatenate(new_keys, axis=-3),
        values=torch.concatenate(new_values, axis=-3),
        num_tokens=cache["num_tokens"] + keys.shape[-3],
    )


def _attention_cache_from_prompt(
    keys: torch.Tensor,
    values: torch.Tensor,
    segment_pos: torch.Tensor,
    window_size: int,
) -> dict[str, torch.Tensor]:
    """Creates a new cache from a prompt.

    Args:
      keys: The new keys to be added to an empty cache.
      values: The new values to be added to an empty cache.
      segment_pos: Positions of each token in the sequence.
      window_size: The local attention window size.

    Returns:
      An empty initialized KV-cache updated with the given keys and values.
    """
    w = min(window_size, keys.shape[1])
    k_padding = torch.zeros(
        (keys.shape[0], window_size - w, keys.shape[2], keys.shape[3]),
        dtype=keys.dtype,
        device=keys.device,
    )
    v_padding = torch.zeros(
        (values.shape[0], window_size - w, values.shape[2], values.shape[3]),
        dtype=values.dtype,
        device=values.device,
    )
    return dict(
        keys=torch.concatenate([k_padding, keys[:, -w:]], dim=1),
        values=torch.concatenate([v_padding, values[:, -w:]], dim=1),
        num_tokens=segment_pos[-1] + 1,
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Returns the GELU activation function with the same approximation as JAX."""
    return nn.functional.gelu(x, approximate="tanh")


class LocalAttentionBlock(nn.Module):
    """Local Multi-Head Attention (MHA) block."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        window_size: int,
        final_w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the local attention block.

        Args:
          width: The width of the block.
          num_heads: The number of heads for the attention mechanism.
          window_size: The local attention window size.
          final_w_init_variance_scale: The scale for the initialization of the last
            layer of the block.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.window_size = window_size
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # Layers.
        self.proj_q = nn.Linear(
            in_features=self.width,
            out_features=self.width,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.proj_k = nn.Linear(
            in_features=self.width,
            out_features=self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.proj_v = nn.Linear(
            in_features=self.width,
            out_features=self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.proj_final = nn.Linear(
            in_features=self.width,
            out_features=self.width,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # Initialization.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.proj_q.weight)
        self.w_init_(self.proj_k.weight)
        self.w_init_(self.proj_v.weight)
        self.out_w_init_(self.proj_final.weight)
        torch.nn.init.zeros_(self.proj_final.bias)

    @property
    def head_dim(self) -> int:
        return self.width // self.num_heads

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weights of the queries, keys and values projections."""
        torch.nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))

    def out_w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weights of the final projection."""
        std = math.sqrt(self.final_w_init_variance_scale / self.width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calls the local attention block.

        Args:
          x: Sequence of input activations.
          segment_pos: Positions of each token in the sequence.
          attention_mask: The attention mask for the block.
          cache: Optional KV-cache for the block, of previous keys and values.

        Returns:
          Output of the block together with the updated cache. If `cache` is None
          than the returned updated cache is empty initialized and filled in from
          the input sequence.
        """
        b, t, _ = x.shape
        assert segment_pos.shape == (t,), f"{segment_pos.shape} != {(t,)}"

        # Generate keys, values and queries.
        queries = self.proj_q(x)
        keys = self.proj_k(x)
        values = self.proj_v(x)
        queries = einops.rearrange(
            queries, "... (n h) -> ... n h", n=self.num_heads
        )
        keys = einops.rearrange(keys, "... (n h) -> ... n h", n=1)
        values = einops.rearrange(values, "... (n h) -> ... n h", n=1)

        # Apply rotary embeddings.
        queries = _apply_rope(queries, segment_pos)
        keys = _apply_rope(keys, segment_pos)

        cache = getattr(self, "cache", cache)
        if cache is not None:
            assert t == 1, f"When cache is provided only `t=1` is supported, not {t=}"

            new_cache = _update_attention_cache(keys, values, cache)

            keys = torch.concatenate([cache["keys"], keys], dim=-3)
            values = torch.concatenate([cache["values"], values], dim=-3)

            if attention_mask is None:
                attention_mask = _compute_cache_mask(segment_pos, self.window_size)
        else:
            new_cache = _attention_cache_from_prompt(
                keys, values, segment_pos, self.window_size
            )

            if attention_mask is None:
                attention_mask = compute_forward_pass_mask(segment_pos, self.window_size)

        # Compute attention.
        logits = einops.einsum(queries, keys, "b t n h, b s n h -> b n t s")
        logits = logits * (self.head_dim ** -0.5)

        # Expand for batch and heads axis.
        attn_mask = attention_mask[None, None].type(torch.bool)

        masked_logits = torch.where(attn_mask, logits, _MIN_LOGITS_VALUE)
        masked_logits = masked_logits.type(torch.float32)

        probs = nn.functional.softmax(masked_logits, dim=-1).type_as(x)
        encoded = einops.einsum(probs, values, "b n t s, b s n h -> b t n h")
        encoded = einops.rearrange(
            encoded, "... n h -> ... (n h)", n=self.num_heads
        )
        attn_output = self.proj_final(encoded)

        return attn_output, new_cache

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        window_size: int,
        heads_dim: int,
        dtype: torch.dtype,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Initializes an empty KV-cache for the block."""
        shape = (batch_size, window_size, 1, heads_dim)
        return dict(
            keys=torch.zeros(shape, device=device, dtype=dtype),
            values=torch.zeros(shape, device=device, dtype=dtype),
            num_tokens=torch.zeros([], dtype=torch.int32, device=device),
        )


class RecurrentBlock(nn.Module):
    """Griffin and Hawk's recurrent block."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
        final_w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the recurrent block.

        Args:
          width: The width of the block.
          num_heads: The number of RG-LRU heads/blocks to use.
          lru_width: Internal dimension to be projected into for RG-LRU to operate
            on.
          conv1d_temporal_width: The temporal width of the 1d convolution.
          final_w_init_variance_scale: The scale for the initialization of the last
            layer of the block.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.lru_width = lru_width or width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # Layers.
        self.linear_y = nn.Linear(
            in_features=self.width,
            out_features=self.lru_width,
            device=device,
            dtype=dtype,
        )
        self.linear_x = nn.Linear(
            in_features=self.width,
            out_features=self.lru_width,
            device=device,
            dtype=dtype,
        )
        self.linear_out = nn.Linear(
            in_features=self.lru_width,
            out_features=self.width,
            device=device,
            dtype=dtype,
        )
        self.conv_1d = Conv1D(
            width=self.lru_width,
            temporal_width=self.conv1d_temporal_width,
            device=device,
            dtype=dtype,
        )
        self.rg_lru = RGLRU(
            width=self.lru_width,
            num_heads=self.num_heads,
            device=device,
            dtype=dtype,
        )

        # Initialization.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.linear_x.weight)
        torch.nn.init.zeros_(self.linear_x.bias)
        self.w_init_(self.linear_y.weight)
        torch.nn.init.zeros_(self.linear_y.bias)
        self.out_w_init_(self.linear_out.weight)
        torch.nn.init.zeros_(self.linear_out.bias)
        self.conv_1d.reset_parameters()
        self.rg_lru.reset_parameters()

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weights of the linear x and y layers of the block."""
        torch.nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))

    def out_w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weights of the last layer of the block."""
        std = math.sqrt(self.final_w_init_variance_scale / self.lru_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calls the recurrent block.

        Args:
          x: Sequence of input activations.
          segment_pos: Position of each token in the sequence.
          attention_mask: Unused attention mask.
          cache: Optional cache with the previous state of the RG-LRU and Conv1D.

        Returns:
          Output of the block together with the updated cache. If `cache` is None
          than the returned updated cache is empty initialized and filled in from
          the input sequence.
        """
        del attention_mask
        # y branch.
        y = self.linear_y(x)
        y = gelu(y)

        # x branch.
        x = self.linear_x(x)
        x, conv1d_state = self.conv_1d(
            x=x,
            segment_pos=segment_pos,
            state=None if cache is None else cache["conv1d_state"],
        )
        x, rg_lru_state = self.rg_lru(
            x=x,
            segment_pos=segment_pos,
            prev_h=None if cache is None else cache["rg_lru_state"],
        )

        # Join branches.
        x = x * y
        x = self.linear_out(x)

        return x, dict(
            conv1d_state=conv1d_state,
            rg_lru_state=rg_lru_state,
        )

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        lru_width: int,
        dtype: torch.dtype,
        conv1d_temporal_width: int = 4,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Initializes an empty RG-LRU and Conv1D cache for the block."""
        return dict(
            rg_lru_state=RGLRU.init_cache(
                batch_size=batch_size,
                width=lru_width,
                device=device,
            ),
            conv1d_state=Conv1D.init_cache(
                batch_size=batch_size,
                width=lru_width,
                dtype=dtype,
                conv1d_temporal_width=conv1d_temporal_width,
                device=device,
            ),
        )


class MLPBlock(nn.Module):
    """MLP block."""

    def __init__(
        self,
        width: int,
        expanded_width: int,
        final_w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the MLP block.

        Args:
          width: The width of the block.
          expanded_width: The width of the expansion inside the MLP block.
          final_w_init_variance_scale: The scale for the initialization of the last
            layer of the block.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.width = width
        self.expanded_width = expanded_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # Layers.
        self.ffw_up = Einsum(
            w_shape=(2, self.width, self.expanded_width),
            b_shape=(2, 1, 1, self.expanded_width),
            eqn="...td,cdD->c...tD",
            device=device,
            dtype=dtype,
        )
        self.ffw_down = nn.Linear(
            in_features=self.expanded_width,
            out_features=self.width,
            device=device,
            dtype=dtype,
        )

        # Initialization.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.ffw_up.reset_parameters()
        self.out_w_init_(self.ffw_down.weight)
        torch.nn.init.zeros_(self.ffw_down.bias)

    def out_w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weights of the last layer of the block."""
        std = math.sqrt(self.final_w_init_variance_scale / self.expanded_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls the MLP block.

        Args:
          x: Sequence of input activations.

        Returns:
          Output of the block.
        """
        out = self.ffw_up(x)
        gate_value = gelu(out[0])
        activations = gate_value * out[1]
        return self.ffw_down(activations)


class ResidualBlock(nn.Module):
    """Griffin and Hawk's residual block."""

    def __init__(
        self,
        width: int,
        mlp_expanded_width: int,
        num_heads: int,
        attention_window_size: int,
        temporal_block_type: str,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
        final_w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the residual block.

        Args:
          width: The width of the block.
          mlp_expanded_width: The width of the expansion inside the MLP block.
          num_heads: The number of heads for the Attention or the RG-LRU.
          attention_window_size: The window size for the local attention block.
          temporal_block_type: Either "recurrent" or "attention", specifying the
            type of recurrent block to use.
          lru_width: The width of the RG-LRU if different from `width`.
          conv1d_temporal_width: The width of the temporal convolution.
          final_w_init_variance_scale: The scale for the variance of the
            initializations of the sub blocks.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.num_heads = num_heads
        self.attention_window_size = attention_window_size
        self.temporal_block_type = temporal_block_type
        self.lru_width = lru_width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # Sub-blocks and layers.
        self.temporal_pre_norm = RMSNorm(
            width=self.width, device=device, dtype=dtype
        )

        match self.temporal_block_type:
            case "recurrent":
                self.recurrent_block = RecurrentBlock(
                    width=self.width,
                    num_heads=self.num_heads,
                    lru_width=self.lru_width,
                    conv1d_temporal_width=self.conv1d_temporal_width,
                    final_w_init_variance_scale=self.final_w_init_variance_scale,
                    device=device,
                    dtype=dtype,
                )

            case "attention":
                self.attention_block = LocalAttentionBlock(
                    width=self.width,
                    num_heads=self.num_heads,
                    window_size=self.attention_window_size,
                    final_w_init_variance_scale=self.final_w_init_variance_scale,
                    device=device,
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unrecognized {temporal_block_type=}.")

        self.channel_pre_norm = RMSNorm(
            width=width, device=device, dtype=dtype,
        )
        self.mlp_block = MLPBlock(
            width=self.width,
            expanded_width=self.mlp_expanded_width,
            final_w_init_variance_scale=self.final_w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.temporal_pre_norm.reset_parameters()
        self.temporal_block.reset_parameters()
        self.channel_pre_norm.reset_parameters()
        self.mlp_block.reset_parameters()

    @property
    def temporal_block(self) -> nn.Module:
        """Alias for the temporal block.

        This creates a common interface while making the layer / parameter types
        easily identifiable by name in a state dictionary.
        """
        match self.temporal_block_type:
            case "recurrent":
                return self.recurrent_block
            case "attention":
                return self.attention_block
            case _:
                raise ValueError(f"Unrecognized {self.temporal_block_type=}.")

    def forward(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calls the residual block.

        Args:
          x: Sequence of input activations.
          segment_pos: Positions of each token in the sequence.
          attention_mask: The attention mask for local attention blocks.
          cache: Optional cache for the block.

        Returns:
          Output of the block together with the updated cache. If `cache` is None
          than the returned updated cache is empty initialized and filled in from
          the input sequence.
        """
        assert segment_pos.shape == (x.shape[1],), f"{segment_pos.shape} != {(x.shape[1],)}"
        raw_x = x

        inputs_normalized = self.temporal_pre_norm(raw_x)
        x, cache = self.temporal_block(
            inputs_normalized,
            segment_pos,
            attention_mask,
            cache,
        )

        residual = x + raw_x

        x = self.channel_pre_norm(residual)
        x = self.mlp_block(x)

        x = x + residual

        return x, cache

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        width: int,
        num_heads: int,
        attention_window_size: int,
        temporal_block_type: str,
        dtype: torch.dtype,
        lru_width: int | None = None,
        conv1d_temporal_width: int = 4,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Initializes an empty cache for the block."""
        match temporal_block_type:
            case "recurrent":
                return RecurrentBlock.init_cache(
                    batch_size=batch_size,
                    lru_width=lru_width or width,
                    dtype=dtype,
                    conv1d_temporal_width=conv1d_temporal_width,
                    device=device,
                )
            case "attention":
                return LocalAttentionBlock.init_cache(
                    batch_size=batch_size,
                    window_size=attention_window_size,
                    heads_dim=width // num_heads,
                    dtype=dtype,
                    device=device,
                )
            case _:
                raise ValueError(f"Unrecognized {temporal_block_type=}.")


# class Embedder(nn.Module):
#     """Embedder module."""
#
#     def __init__(
#         self,
#         vocab_size: int,
#         embed_dim: int,
#         scale_by_sqrt_dim: bool,
#         device: str | torch.device | None = None,
#         dtype: torch.dtype | None = None,
#     ):
#         """Initializes the embedder.
#
#         Args:
#           vocab_size: The size of the token vocabulary.
#           embed_dim: The dimensionality of each token embedding.
#           scale_by_sqrt_dim: Whether to scale the output of the block by
#             `sqrt(self.embed_dim)`
#           device: On what device to initialize parameters. Needed to allow for
#             initializing the module without parameter initialization.
#           dtype: What dtype to use for initialization.
#         """
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.scale_by_sqrt_dim = scale_by_sqrt_dim
#
#         # Parameters.
#         self.input_embedding = nn.Parameter(
#             torch.empty(
#                 [self.vocab_size, self.embed_dim], device=device, dtype=dtype
#             )
#         )
#
#         # Initialization
#         self.reset_parameters()
#
#     def reset_parameters(self) -> None:
#         """Resets the parameters of the module."""
#         torch.nn.init.normal_(
#             self.input_embedding,
#             mean=0.0,
#             std=math.sqrt(1.0 / self.embed_dim),
#         )
#
#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         """Encodes an input sequence of tokens."""
#         x = self.input_embedding[(x,)]
#         if self.scale_by_sqrt_dim:
#             # Cast to bfloat16 to match training.
#             x = x * torch.tensor(math.sqrt(self.embed_dim)).type(torch.bfloat16)
#         return x
#
#     def decode(self, x: torch.Tensor) -> torch.Tensor:
#         """Decodes an input sequence of activations."""
#         return x @ self.input_embedding.T


class RMSNorm(nn.Module):
    """RMS Norm."""

    def __init__(
        self,
        width: int,
        eps: float = 1e-6,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the RMSNorm.

        Args:
          width: The number of dimensions of the input and output.
          eps: Small constant added to the square root when normalizing.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.width = width
        self.eps = eps

        # Parameters.
        self.scale = nn.Parameter(torch.empty(
            [self.width], device=device, dtype=dtype
        ))

        # Initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        torch.nn.init.zeros_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls the RMSNorm."""
        var = torch.mean(torch.square(x), axis=-1, keepdims=True)
        normed_x = x * torch.rsqrt(var + self.eps)

        scale = torch.reshape(self.scale, [1 for _ in range(x.ndim - 1)] + [-1])

        return normed_x * (scale + 1)


class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer."""

    def __init__(
        self,
        width: int,
        num_blocks: int,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the BlockDiagonalLinear.

        Args:
          width: The number of dimensions of the input and output.
          num_blocks: The number of diagonal blocks in the layer.
          w_init_variance_scale: A parameters that scales the variance of the
            initialization of the weights.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.block_width = self.width // self.num_blocks

        # Parameters.
        self.w = nn.Parameter(torch.empty(
            [self.num_blocks, self.block_width, self.block_width],
            device=device,
            dtype=dtype
        ))
        self.b = nn.Parameter(torch.empty(
            [self.num_blocks, self.block_width], device=device, dtype=dtype
        ))

        # Initialization.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.w)
        torch.nn.init.zeros_(self.b)

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weight `w` of the layer."""
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls the BlockDiagonalLinear."""
        # Split x to blocks.
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias.
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

        # Flatten the output.
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


def rnn_scan(
    x: torch.Tensor,
    a: torch.Tensor,
    reset: torch.Tensor,
    h0: torch.Tensor | None,
    acc_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the recurrence of a linear RNN.

    Args:
      x: The input sequence.
      a: The diagonal of the recurrence matrix `A`.
      reset: Indicator of document boundaries, e.g. when to reset the hidden state
        of the RNN.
      h0: The initial hidden state.
      acc_dtype: The data type for the accumulation.

    Returns:
      The output of the linear recurrence.
    """
    assert x.ndim == 3
    assert a.shape == x.shape[-a.ndim:]
    assert a.dtype == x.dtype
    assert type(a) is type(x)
    assert h0 is None or h0.dtype == acc_dtype

    # Multiply `a` by the reset.
    a = a * ~reset

    if x.shape[1] == 1:
        # Using scan in sampling mode.
        if h0 is None:
            return x, x[:, 0].type(acc_dtype)

        else:
            y = a.type(acc_dtype) * h0[:, None] + x.type(acc_dtype)
            return y.type(x.dtype), y[:, -1]

    else:
        # Using scan in linear mode.
        if h0 is not None:
            h_t = h0
        else:
            h_t = torch.zeros(x[:, 0].shape, dtype=acc_dtype, device=x.device)

        y = torch.zeros_like(x)
        for t in range(x.shape[1]):
            h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
            y[:, t] = h_t.type(x.dtype)

    return y, h_t


def rnn_param_init(
    tensor: torch.Tensor,
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Initializes the `A` parameter of the RG-LRU uniformly on a ring."""
    with torch.no_grad():
        # Proportional to area in a ring.
        # 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + 1e-8)
        tensor.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
        tensor.log_().mul_(0.5)

        if transform == "softplus":
            # Inverse transform.
            # jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
            return tensor.neg_().exp_().sub_(1.0).log_()
        else:
            raise NotImplementedError()


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x,
                                       min=1 / (_MAX_SQRT_GRADIENT ** 2))
        return grad_output / torch.sqrt(clipped_x_times_4)


class RGLRU(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the RG-LRU.

        Args:
          width: The number of dimensions of the input and output.
          num_heads: The number of diagonal blocks in the input and A gate layers.
          w_init_variance_scale: Initialization parameter for the
            BlockDiagonalLinear layers of the gates. See the `BlockDiagonalLinear`
            layer for details.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters and layers.
        self.a_param = nn.Parameter(torch.empty(
            [self.width], device=device, dtype=dtype
        ))
        self.input_gate = BlockDiagonalLinear(
            width=self.width,
            num_blocks=self.num_heads,
            w_init_variance_scale=w_init_variance_scale,
            device=device,
            dtype=dtype,
        )
        self.a_gate = BlockDiagonalLinear(
            width=self.width,
            num_blocks=self.num_heads,
            w_init_variance_scale=self.w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

        # Initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.input_gate.reset_parameters()
        self.a_gate.reset_parameters()
        self.a_param_init(self.a_param)

    def a_param_init(self, w: torch.Tensor) -> torch.Tensor:
        """Initializes the `A` parameter of the RG-LRU."""
        return rnn_param_init(w, min_rad=0.9, max_rad=0.999)

    def __call__(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        prev_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls the RG-LRU.

        Args:
          x: Sequence of input activations.
          segment_pos: Position of each token in the sequence.
          prev_h: The previous hidden state of the RG-LRU.

        Returns:
          Output of the block together with the updated hidden state.
        """

        bs, l, _ = x.shape
        assert segment_pos.shape == (l,), f"{segment_pos.shape} != {(l,)}"
        reset = segment_pos[None, :, None] == 0

        # Gates for x and a.
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))

        # Compute the parameter `A` of the recurrence.
        log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
        a = torch.exp(log_a)
        a_square = torch.exp(2 * log_a)

        # Gate the input.
        gated_x = x * gate_x

        # Apply gamma normalization to the input. We need to clip the derivatives of
        # `sqrt` in order to prevent NaNs during training in bfloat16.
        multiplier = SqrtBoundDerivative.apply(1 - a_square)
        multiplier = reset + ~reset * multiplier
        normalized_x = gated_x * multiplier.type(x.dtype)

        y, last_h = rnn_scan(
            x=normalized_x,
            a=a,
            reset=reset,
            h0=prev_h,
        )
        return y, last_h

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        width: int,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Returns an empty initialized cache for the RG-LRU."""
        # RG-LRU cache always in float32.
        return torch.zeros((batch_size, width), dtype=torch.float32,
                           device=device)


class Conv1D(nn.Module):
    """A 1D temporal convolution layer."""

    def __init__(
        self,
        width: int,
        temporal_width: int,
        w_init_variance_scale: float = 0.01,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the Conv1D.

        Args:
          width: The number of features for both inputs and outputs.
          temporal_width: The size of the temporal receptive field of the
            convolution. In other words, how much back in time the convolution can
            look to produce an output.
          w_init_variance_scale: A parameter that scales the variance of the
            initialization of the weights.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialzation.
          dtype: What dtype to use for initialziation.
        """
        super().__init__()
        self.width = width
        self.temporal_width = temporal_width
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters.
        self.w = nn.Parameter(torch.empty(
            [self.temporal_width, self.width], device=device, dtype=dtype
        ))
        self.b = nn.Parameter(torch.empty([width], device=device, dtype=dtype))

        # Initialization.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.w)
        torch.nn.init.zeros_(self.b)

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weight matrix `w` of the Conv1D."""
        std = math.sqrt(self.w_init_variance_scale / self.temporal_width)
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def forward(
        self,
        x: torch.Tensor,
        segment_pos: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls the Conv1D.

        Args:
          x: Sequence of input activations.
          segment_pos: Position of each token in the sequence.
          state: The state containing the previous `self.temporal_width-1` inputs
            This is set to `None` in training mode.

        Returns:
          The output of the convolution and the updated state.
        """
        assert segment_pos.shape == (x.shape[1],)

        if state is not None:
            # 1. Decoding mode:
            # - We have access to the previous `self.temporal_width - 1` inputs.
            # - Only a single token needs to be output.
            x = self._concatenate_with_state(x, state)
            prompt_len = self.temporal_width - 1
            output_len = 1
            state_dtype = state.dtype
        else:
            # 1. Training mode:
            # - The full sequence length need to be output.
            prompt_len = 0
            output_len = x.shape[1]
            state_dtype = x.dtype

        # 3. Perform the convolution:
        # - Initialize an accumulator for the convolution output.
        convolution_output = 0.0

        # - We cannot look back by more than the total sequence length
        #   ("valid" convolution).
        temporal_width = min(self.temporal_width, prompt_len + output_len)

        # - The convolution is implemented as a manual loop so that we can
        #   incorporate the window masking further below.
        for temporal_shift in range(temporal_width):
            start_idx, end_idx = self._convolution_window_indices(
                prompt_len=prompt_len,
                shift_back=temporal_shift,
                output_len=output_len,
            )
            x_window = x[:, start_idx:end_idx]

            if state is None:
                # - Ensure that the mask prevents accessing tokens from a different
                #   document in training mode.
                window_mask = self._compute_document_mask(
                    segment_pos=segment_pos,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    max_look_ahead=temporal_shift,
                )
                x_window *= window_mask[None, :, None].type(x.dtype).to(device=x.device)

            x_window = self._pad_window(x_window, output_len)

            # - Select w for this temporal shift, and expand on the batch and time
            #   dimensions.
            w = self.w[self.temporal_width - temporal_shift - 1][None, None, :]

            # - Accumulate the convolution result.
            convolution_output += x_window * w

        # - Add the bias of the convolution.
        convolution_output += self.b[None, None]

        # 4. Store the new (potentially padded) state for future decoding.
        new_state = x[:, 1 - self.temporal_width:].type(state_dtype)
        new_state = self._pad_state(new_state)

        return convolution_output, new_state

    def _concatenate_with_state(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenates the current input `x` with the previous state for decoding.

        Args:
          x: The current input activations (shape: [batch_size, 1, width]).
          state: State tensor storing previous inputs (shape: [batch_size,
            temporal_width - 1, width]).

        Returns:
          The concatenated input sequence
          (shape: [batch_size, temporal_width, width]).
        """
        b, num_tokens, d = x.shape
        assert state.shape == (b, self.temporal_width - 1, d)
        assert num_tokens == 1
        return torch.concatenate([state.type(x.dtype), x], dim=1)

    def _convolution_window_indices(
        self,
        *,
        prompt_len: int,
        shift_back: int,
        output_len: int,
    ) -> tuple[int, int]:
        """Calculates the start and end indices for the convolution window.

        Args:
          prompt_len: Length of the prompt (zero in training mode).
          shift_back: By how much the window should be shifted backwards.
          output_len: Sequence length of the output (sequence length in training
            mode, one in decoding mode).

        Returns:
          start_idx: The starting index for the convolution window.
          end_idx: The ending index for the convolution window.
        """
        start_idx = max(prompt_len - shift_back, 0)
        end_idx = prompt_len + output_len - shift_back
        return start_idx, end_idx

    def _compute_document_mask(
        self,
        *,
        segment_pos: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_look_ahead: int,
    ) -> torch.Tensor:
        """Creates a mask to prevent mixing of information between documents.

        Args:
            segment_pos: Position of each token in the sequence. In particular,
              a zero indicates the start of a new document.
            start_idx: The starting index of the convolution window.
            end_idx: The ending index of the convolution window.
            max_look_ahead: How much to look ahead at most to detect a document
              boundary (depends on the convolution).

        Returns:
            An integer mask where `1` indicates a position that should be
            included in the convolution, and `0` a position that should be excluded.
        """
        not_a_document_boundary = (segment_pos != 0).type(torch.int32)
        mask = torch.ones((end_idx - start_idx), device=segment_pos.device)
        for shift in range(1, max_look_ahead + 1):
            # At each position, look ahead by `shift` tokens to see if a
            # document boundary is present there.
            mask *= not_a_document_boundary[start_idx + shift: end_idx + shift]
        return mask

    def _pad_window(
        self,
        window: torch.Tensor,
        output_len: int,
    ) -> torch.Tensor:
        """Left-pads the window if it is shorter than the output sequence length."""
        batch_size, window_len, width = window.shape
        padding_len = output_len - window_len
        padding = torch.zeros(
            (batch_size, padding_len, width),
            dtype=window.dtype,
            device=window.device,
        )
        return torch.concatenate([padding, window], dim=1)

    def _pad_state(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Left-pads the state if it is shorter than the temporal width."""
        b, state_seq_len, d = state.shape
        padding_len = self.temporal_width - state_seq_len - 1
        padding = torch.zeros(
            (b, padding_len, d),
            dtype=state.dtype,
            device=state.device,
        )
        return torch.concatenate([padding, state], dim=1)

    @classmethod
    def init_cache(
        cls,
        *,
        batch_size: int,
        width: int,
        dtype: torch.dtype,
        conv1d_temporal_width: int = 4,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Returns an empty initialized cache for the Conv1D."""
        shape = (batch_size, conv1d_temporal_width - 1, width)
        return torch.zeros(shape, dtype=dtype, device=device)


class Einsum(nn.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""

    def __init__(
        self,
        w_shape: Sequence[int],
        b_shape: Sequence[int],
        eqn: str,
        w_init_variance_scale: float = 1.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the Einsum.

        Args:
          w_shape: The shape of the weight matrix w.
          b_shape: The shape of the bias.
          eqn: The einsum string.
          w_init_variance_scale: A parameters that scales the variance of the
            initialization of the weights.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.w_shape = tuple(w_shape)
        self.b_shape = tuple(b_shape)
        self.eqn = eqn
        self.w_init_variance_scale = w_init_variance_scale

        # Parameters.
        self.w = nn.Parameter(
            torch.empty(self.w_shape, device=device, dtype=dtype))
        self.b = nn.Parameter(
            torch.empty(self.b_shape, device=device, dtype=dtype))

        # Initialization.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters of the module."""
        self.w_init_(self.w)
        torch.nn.init.zeros_(self.b)

    def w_init_(self, w: torch.Tensor) -> None:
        """Initializes the weight matrix `w` of the Einsum."""
        std = math.sqrt(self.w_init_variance_scale / self.w_shape[1])
        torch.nn.init.normal_(w, mean=0.0, std=std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Calls the Einsum."""
        return torch.einsum(self.eqn, x, self.w) + self.b


# ======================================== TRANSITION ========================================
# Above: GDM code
# Below: HF library code


# BEGIN: adapted from mamba.
@dataclass
class RecurrentBlockCache:
    rnn_state: torch.Tensor
    conv1d_state: torch.Tensor


class GriffinCache:
    def __init__(self, config: RecurrentGemmaConfig, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.states = []
        for block_type in config.block_types:
            self.states.append(ResidualBlock.init_cache(
                batch_size=batch_size,
                width=config.hidden_size,
                num_heads=config.num_attention_heads,
                attention_window_size=config.attention_window_size,
                temporal_block_type=block_type,
                lru_width=config.lru_width,
                conv1d_temporal_width=config.conv1d_width,
                dtype=dtype,
                device=device,
            ))


@dataclass
class GriffinOutput(ModelOutput):
    """
    Class for the Griffin model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (`GriffinCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[GriffinCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class GriffinCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`GriffinCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[GriffinCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: tuple | None = None


# END: adapted from mamba.


ALL_LAYERNORM_LAYERS.append(RMSNorm)

# TODO(lberrada, botev): adapt all doctsrings.

RECURRENTGEMMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RecurrentGemmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.",
    RECURRENTGEMMA_START_DOCSTRING,
)
class RecurrentGemmaPreTrainedModel(PreTrainedModel):
    config_class = RecurrentGemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keep_in_fp32_modules = ["inv_freq", "rotary_emb", "cos_cached", "sin_cached"]
    _no_split_modules = ["RecurrentGemmaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    # TODO(lberrada, botev): decide whether we want to support the various implementations of attention
    # in first version.
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = False

    def _init_weights(self, module):
        if isinstance(module, nn.ModuleList):
            for block_module in module:
                block_module.reset_parameters()
        elif isinstance(module, RMSNorm) or isinstance(module, nn.Embedding):
            module.reset_parameters()

    # TODO(lberrada, botev): adapt this to handle recurrent states.
    #  PS(botev): Mamba doesn't seem to use this
    # def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
    #
    #     if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
    #         raise ValueError(
    #             "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
    #             "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
    #         )
    #
    #     for layer in self.model.layers:
    #         weights = layer.self_attn.o_proj.weight
    #         layer.self_attn.past_key_value = cache_cls(
    #             self.config, max_batch_size, max_cache_len, device=weights.device, dtype=weights.dtype
    #         )
    #
    # def _reset_cache(self):
    #     for layer in self.model.layers:
    #         layer.self_attn.past_key_value = None


RECURRENTGEMMA_INPUTS_DOCSTRING = r"""
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

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention  See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all  See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.",
    RECURRENTGEMMA_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaModel with LLAMA->RECURRENTGEMMA,Llama->RecurrentGemma
class RecurrentGemmaModel(RecurrentGemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers*  Each layer is a [`RecurrentGemmaDecoderLayer`]

    Args:
        config: RecurrentGemmaConfig
    """

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.config = config

        # TODO(lberrada, botev): fix device and dtype
        device = dtype = None

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.blocks = nn.ModuleList([
            ResidualBlock(
                width=self.config.hidden_size,
                mlp_expanded_width=self.config.intermediate_size,
                num_heads=self.config.num_attention_heads,
                attention_window_size=self.config.attention_window_size,
                temporal_block_type=block_type,
                lru_width=self.config.lru_width,
                final_w_init_variance_scale=2.0 / self.config.num_hidden_layers,
                device=device,
                dtype=dtype,
            )
            for block_type in self.config.block_types
        ])
        self.final_norm = RMSNorm(
            width=self.config.hidden_size, device=device, dtype=dtype
        )
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embedder

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(RECURRENTGEMMA_INPUTS_DOCSTRING)
    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[GriffinCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, GriffinOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if cache_position is None:
            if input_ids is not None:
                cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
            else:
                cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        if self.config.embeddings_scale_by_sqrt_dim:
            normalizer = torch.tensor(self.config.hidden_size ** 0.5)
            hidden_states = hidden_states * normalizer.type(torch.bfloat16)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        new_cache = None
        cache = past_key_values

        for i, residual_block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    residual_block.__call__,
                    hidden_states,
                    cache_position,
                    attention_mask,
                    None if cache is None else cache.states[i],
                )
            else:
                layer_outputs = residual_block(
                    hidden_states,
                    cache_position,
                    attention_mask,
                    None if cache is None else cache.states[i],
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                if new_cache is None:
                    new_cache = GriffinCache(
                        config=self.config,
                        batch_size=hidden_states.shape[0],
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                new_cache.states[i] = layer_outputs[1]

        hidden_states = self.final_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, new_cache, all_hidden_states] if v is not None)

        return GriffinOutput(
            last_hidden_state=hidden_states,
            past_key_values=new_cache,
            hidden_states=all_hidden_states,
        )

    # TODO(botev): We don't need this as we generate it on the fly.
    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    # def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
    #     if self.config._attn_implementation == "flash_attention_2":
    #         if attention_mask is not None and 0.0 in attention_mask:
    #             return attention_mask
    #         return None
    #
    #     dtype, device = input_tensor.dtype, input_tensor.device
    #     min_dtype = torch.finfo(dtype).min
    #     sequence_length = input_tensor.shape[1]
    #     if hasattr(self.layers[0].self_attn, "past_key_value"):  # static cache
    #         target_length = self.config.max_position_embeddings
    #     else:  # dynamic cache
    #         target_length = (
    #             attention_mask.shape[-1] if isinstance(attention_mask,
    #                                                    torch.Tensor) else
    #             cache_position[-1] + 1
    #         )
    #
    #     causal_mask = torch.full((sequence_length, target_length),
    #                              fill_value=min_dtype, dtype=dtype,
    #                              device=device)
    #     if sequence_length != 1:
    #         causal_mask = torch.triu(causal_mask, diagonal=1)
    #     causal_mask *= torch.arange(target_length,
    #                                 device=device) > cache_position.reshape(-1,
    #                                                                         1)
    #     causal_mask = causal_mask[None, None, :, :].expand(
    #         input_tensor.shape[0], 1, -1, -1)
    #     if attention_mask is not None:
    #         causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
    #         if attention_mask.dim() == 2:
    #             mask_length = attention_mask.shape[-1]
    #             padding_mask = causal_mask[..., :mask_length].eq(
    #                 0.0) * attention_mask[:, None, None, :].eq(0.0)
    #             causal_mask[..., :mask_length] = causal_mask[...,
    #                                              :mask_length].masked_fill(
    #                 padding_mask, min_dtype)
    #         elif attention_mask.dim() == 4:
    #             # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
    #             # cache. In that case, the 4D attention mask attends to the newest tokens only.
    #             if attention_mask.shape[-2] < cache_position[
    #                 0] + sequence_length:
    #                 offset = cache_position[0]
    #             else:
    #                 offset = 0
    #             mask_shape = attention_mask.shape
    #             mask_slice = (attention_mask.eq(0.0)).to(
    #                 dtype=dtype) * min_dtype
    #             causal_mask[
    #             : mask_shape[0], : mask_shape[1],
    #             offset: mask_shape[2] + offset, : mask_shape[3]
    #             ] = mask_slice
    #
    #     if (
    #         self.config._attn_implementation == "sdpa"
    #         and attention_mask is not None
    #         and attention_mask.device.type == "cuda"
    #     ):
    #         # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
    #         # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
    #         # Details: https://github.com/pytorch/pytorch/issues/110213
    #         causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask,
    #                                                                 min_dtype)
    #
    #     return causal_mask


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->RECURRENTGEMMA,Llama->RecurrentGemma,llama->gemma
class RecurrentGemmaForCausalLM(RecurrentGemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__(config)
        self.model = RecurrentGemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Ignore copy
    @add_start_docstrings_to_model_forward(RECURRENTGEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GriffinCausalLMOutput,
                               config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[GriffinCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, GriffinCausalLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RecurrentGemmaForCausalLM

        >>> model = RecurrentGemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Soft-cap the logits
        if self.config.logits_soft_cap is not None:
            c = self.config.logits_soft_cap
            logits = nn.functional.tanh(logits / c) * c

        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return GriffinCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=(),
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values: Optional[GriffinCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None or cache_position.shape[0] == 1:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if cache_position is not None:
                cache_position = cache_position[-1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["past_key_values"] = past_key_values
        model_inputs["cache_position"] = cache_position

        if past_key_values is not None:
            attn_mask = _compute_cache_mask(
                torch.zeros([], dtype=torch.int32, device=input_ids.device),
                self.config.attention_window_size,
            )
        else:
            attn_mask = compute_forward_pass_mask(
                cache_position,
                self.config.attention_window_size,
            )

        model_inputs["attention_mask"] = attn_mask
        return model_inputs
