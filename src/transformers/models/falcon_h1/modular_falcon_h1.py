# coding=utf-8
# Copyright 2025 Technology Innovation Institute and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch FalconH1 model."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.mamba2.modeling_mamba2 import (
    MambaRMSNormGated,
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
)

from ...cache_utils import Cache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    auto_docstring,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_attn_2_available,
    is_mamba_2_ssm_available,
)
from .configuration_falcon_h1 import FalconH1Config


if is_flash_attn_2_available():
    pass

if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    selective_state_update = None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FalconH1Config"


class FalconHybridMambaAttentionDynamicCache(HybridMambaAttentionDynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(
        self,
        config: FalconH1Config,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        devices: Optional[List[str]] = None,
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.has_previous_state = False
        self.conv_kernel_size = config.mamba_d_conv

        self._seen_tokens = 0

        self.intermediate_size = (
            config.mamba_d_ssm if config.mamba_d_ssm is not None else int(config.mamba_expand * config.hidden_size)
        )

        self.conv_states = {
            i: torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.mamba_n_groups * config.mamba_d_state,
                self.conv_kernel_size,
                device=devices[i],
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(
                batch_size,
                config.mamba_n_heads,
                config.mamba_d_head,
                config.mamba_d_state,
                device=devices[i],
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }

        self.transformer_layers = []
        for i in range(config.num_hidden_layers):
            self.transformer_layers.append(i)

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        if len(cache_position) > 1:
            conv_state[:, :, :] = new_conv_state.to(conv_state.device)
        else:
            conv_state[:, :, -1] = new_conv_state[:, :, -1].to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


class FalconH1RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class FalconH1Attention(LlamaAttention):
    def __init__(self, config: FalconH1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.key_multiplier = config.key_multiplier

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) * self.key_multiplier
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class FalconH1RMSNormGated(MambaRMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, n_groups=1, norm_before_gate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype

        if not self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, dim = hidden_states.shape
        else:
            batch_size, dim = hidden_states.shape
            seq_len = 1
        hidden_states = hidden_states.to(torch.float32)

        hidden_states = hidden_states.view(batch_size, seq_len, self.n_groups, int(dim // self.n_groups))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight.view(self.n_groups, int(dim // self.n_groups)) * hidden_states
        hidden_states = hidden_states.view(batch_size, seq_len, dim)

        if seq_len == 1:
            hidden_states = hidden_states.squeeze(1)

        if self.norm_before_gate and gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


# Adapted from transformers.models.mamba2.modeling_mamba2.Mamba2Mixer
class FalconH1Mixer(nn.Module):
    """
    FalconH1Mixer is identical to classic Mamba2 mixer classes but differs on two different things
    - Users can pass custom intermediate_size through `config.mamba_d_ssm`
    - The use of gated RMS normalization layer is optional
    """

    def __init__(self, config: FalconH1Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.mamba_n_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = (
            int(config.mamba_expand * self.hidden_size) if config.mamba_d_ssm is None else config.mamba_d_ssm
        )
        self.layer_idx = layer_idx
        self.use_conv_bias = config.mamba_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.use_bias = config.mamba_proj_bias

        self.layer_norm_epsilon = config.rms_norm_eps
        self.groups_time_state_size = config.mamba_n_groups * self.ssm_state_size

        self.n_groups = config.mamba_n_groups
        self.head_dim = config.mamba_d_head
        self.chunk_size = config.mamba_chunk_size

        # FIXME:
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.mamba_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=self.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.mamba_rms_norm = config.mamba_rms_norm

        if self.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=self.layer_norm_epsilon,
                n_groups=self.n_groups,
                norm_before_gate=config.mamba_norm_before_gate,
            )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=config.projectors_bias)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )
        else:
            logger.warning_once("The fast path for FalconH1 will be used when running the model on a GPU")

        self.zxbcdt_multipliers = config.ssm_multipliers
        self.ssm_in_multiplier = config.ssm_in_multiplier

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        hidden_states = hidden_states * self.ssm_in_multiplier
        projected_states = self.in_proj(hidden_states)
        projected_states = projected_states * self.mup_vector
        d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
            and cache_position is not None
            and cache_position[0] > 0
        )

        # getting projected states from cache if it exists
        if use_precomputed_states:
            d_mlp = (projected_states.squeeze(1).shape[-1] - d_to_remove) // 2

            z0, x0, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )

            # 2. Convolution sequence transformation
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )

            # 3. SSM transformation
            A = -torch.exp(self.A_log.float())  # (nheads,)
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=gate.view(batch_size, self.num_heads, self.head_dim) if not self.mamba_rms_norm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)

            if self.mamba_rms_norm:
                hidden_states = self.norm(hidden_states, gate)

            if d_mlp > 0:
                hidden_states = torch.cat([F.silu(z0) * x0, hidden_states], dim=-1)

            # 4. Final linear projection
            out = self.out_proj(hidden_states[:, None, ...])
        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            if self.training and cache_params is None:
                out = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight if self.mamba_rms_norm else None,
                    rmsnorm_eps=self.norm.variance_epsilon if self.mamba_rms_norm else None,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )

            else:
                d_mlp = (
                    projected_states.shape[-1]
                    - 2 * self.intermediate_size
                    - 2 * self.n_groups * self.ssm_state_size
                    - self.num_heads
                ) // 2
                if attention_mask is not None:
                    projected_states = projected_states * attention_mask[..., None]
                _, gate, hidden_states_B_C, dt = projected_states.split(
                    [
                        2 * d_mlp,
                        self.intermediate_size,
                        self.conv_dim,
                        self.num_heads,
                    ],
                    dim=-1,
                )

                if cache_params is not None:
                    conv_states = F.pad(
                        hidden_states_B_C.permute(0, 2, 1),
                        (self.conv_kernel_size - hidden_states_B_C.shape[-2], 0),
                    )
                    cache_params.update_conv_state(self.layer_idx, conv_states, cache_position)

                time_step = nn.functional.softplus(dt + self.dt_bias)
                # 1D Convolution
                if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
                    )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)[:, :seq_len]

                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [
                        self.intermediate_size,
                        groups_time_state_size,
                        groups_time_state_size,
                    ],
                    dim=-1,
                )

                if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
                # This is a hack to make sure multi-GPU inference works with HF accelerate
                # see: https://github.com/Dao-AILab/flash-attention/issues/523 for more details
                with torch.cuda.device(hidden_states.device):
                    scan_output, ssm_state = mamba_chunk_scan_combined(
                        hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                        time_step,
                        A,
                        B.view(batch_size, seq_len, self.n_groups, -1),
                        C.view(batch_size, seq_len, self.n_groups, -1),
                        chunk_size=self.chunk_size,
                        D=self.D,
                        z=None,
                        seq_idx=None,
                        return_final_states=True,
                        **dt_limit_kwargs,
                    )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                if self.mamba_rms_norm:
                    out = self.norm(scan_output, gate)
                else:
                    out = scan_output * torch.nn.functional.silu(gate)
                out = self.out_proj(out)
        return out

    # fmt: off
    def torch_forward(
        self,
        input_states,
        cache_params: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        projected_states = self.in_proj(input_states)
        gate, hidden_states_B_C, dt = projected_states.split(
                [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
            and cache_position is not None
            and cache_position[0] > 0
        )

        # 2. Convolution sequence transformation
        if use_precomputed_states:
            cache_params.conv_states[self.layer_idx] = cache_params.conv_states[self.layer_idx].roll(shifts=-1, dims=-1)
            cache_params.conv_states[self.layer_idx][:, :, -1] = hidden_states_B_C[:, 0, :].to(cache_params.conv_states[self.layer_idx].device)

            # We need to guarantee that anything regarding the cache is on the same device
            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)

            hidden_states_B_C = torch.sum(
                conv_states * self.conv1d.weight.squeeze(1), dim=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Init cache
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = nn.functional.pad(
                    hidden_states_B_C_transposed, (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_states)

            hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())                            # [num_heads]
        if use_precomputed_states:
            # We need to guarantee that anything regarding the cache is on the same device
            cache_device = cache_params.ssm_states[self.layer_idx].device

            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)

            # State calculation
            cache_params.ssm_states[self.layer_idx].copy_(
                cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]

            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(segment_sum(A))

            # Contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

            # Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            # 2. Compute the state for each intra-chunk
            # (right term of low-rank factorization of off-diagonal blocks; B terms)
            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
            # (middle term of factorization of off-diag blocks; A terms)
            if use_precomputed_states:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # 4. Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])

            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            # Init cache
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        if self.mamba_rms_norm:
            scan_output = self.norm(y, gate)
        else:
            scan_output = y * torch.nn.functional.silu(gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)


class FalconH1MLP(LlamaMLP):
    def __init__(self, config: FalconH1Config = None):
        super().__init__()
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers

    def forward(self, x):
        y = self.up_proj(x) * self.act_fn(self.gate_proj(x) * self.gate_multiplier)
        y = self.down_proj(y) * self.down_multiplier
        return y


class FalconH1RMSNorm(LlamaRMSNorm):
    pass


class FalconH1DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FalconH1Config, layer_idx: int):
        super().__init__()
        self.feed_forward = FalconH1MLP(config)

        head_dim = config.hidden_size // config.num_attention_heads
        self.channels_attn = config.num_attention_heads * head_dim + 2 * config.num_key_value_heads * head_dim

        self.mamba = FalconH1Mixer(config=config, layer_idx=layer_idx)

        self.self_attn = FalconH1Attention(config, layer_idx)

        self.attention_in_multiplier = config.attention_in_multiplier
        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

        self.input_layernorm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mamba_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`FalconHybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        mamba_hidden_states = self.mamba(
            hidden_states=hidden_states,
            cache_params=past_key_value,
            cache_position=cache_position,
            attention_mask=mamba_attention_mask,
        )
        mamba_hidden_states = mamba_hidden_states * self.ssm_out_multiplier

        attention_hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states * self.attention_in_multiplier,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attention_hidden_states = attention_hidden_states * self.attn_out_multiplier

        hidden_states = mamba_hidden_states + attention_hidden_states

        # residual connection after attention
        hidden_states = residual + hidden_states

        # feed-forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class FalconH1PreTrainedModel(PreTrainedModel):
    config_class = FalconH1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FalconH1DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True  # Note: only supports FalconHybridMambaAttentionDynamicCache
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        for name, param in module.named_parameters(recurse=True):
            if not param.requires_grad:
                continue
            if "layernorm" in name.lower() and "weight" in name:
                # LayerNorm weights usually initialized to 1
                param.data.fill_(1.0)
            elif "bias" in name:
                param.data.zero_()
            else:
                try:
                    param.data.normal_(mean=0.0, std=std)
                except Exception as e:
                    print(f"Skipping init for {name} due to error: {e}")


def compute_mup_vector(config):
    """
    Computes the MuP vector based on model configuration.

    FalconH1 applies different MuP multiplier for each dimension of the hidden states.
    The MuP vector is partitioned into chunks, and each chunk is multiplied with its
    corresponding projected dimension.

    Args:
        config: FalconH1Config object

    Returns:
        torch.Tensor: The computed MuP vector
    """
    # We'll need some values from the config to compute the vector dimensions
    intermediate_size = (
        config.mamba_d_ssm if config.mamba_d_ssm is not None else int(config.mamba_expand * config.hidden_size)
    )
    groups_time_state_size = config.mamba_n_groups * config.mamba_d_state
    num_heads = config.mamba_n_heads
    zxbcdt_multipliers = config.ssm_multipliers

    vector_shape = 2 * intermediate_size + 2 * groups_time_state_size + num_heads
    mup_vector = torch.ones(1, 1, vector_shape)

    # Apply multipliers to different sections of the vector
    mup_vector[:, :, :intermediate_size] *= zxbcdt_multipliers[0]
    mup_vector[:, :, intermediate_size : 2 * intermediate_size] *= zxbcdt_multipliers[1]
    mup_vector[:, :, 2 * intermediate_size : 2 * intermediate_size + groups_time_state_size] *= zxbcdt_multipliers[2]
    mup_vector[
        :, :, 2 * intermediate_size + groups_time_state_size : 2 * intermediate_size + 2 * groups_time_state_size
    ] *= zxbcdt_multipliers[3]
    mup_vector[:, :, 2 * intermediate_size + 2 * groups_time_state_size :] *= zxbcdt_multipliers[4]

    return mup_vector


@auto_docstring
# Adapted from transformers.models.jamba.modeling_jamba.JambaModel
class FalconH1Model(FalconH1PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`FalconH1DecoderLayer`]

    Args:
        config: FalconH1Config
    """

    def __init__(self, config: FalconH1Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        decoder_layers = []
        for i in range(config.num_hidden_layers):
            decoder_layers.append(FalconH1DecoderLayer(config, layer_idx=i))
        self.layers = nn.ModuleList(decoder_layers)

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = FalconH1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = FalconH1RotaryEmbedding(config=config)

        self.embedding_multiplier = config.embedding_multiplier
        self.lm_head_multiplier = config.lm_head_multiplier

        self.gradient_checkpointing = False
        # Compute the MuP vector once and register it for all layers
        mup_vector = compute_mup_vector(config)
        for layer in self.layers:
            layer.mamba.register_buffer("mup_vector", mup_vector, persistent=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,  # NOOP kwargs, for now
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embedding_multiplier
        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            logger.warning_once(
                "FalconH1 requires an initialized `FalconHybridMambaAttentionDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                mamba_attention_mask=mamba_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                if layer_outputs[1] is not None:
                    # append attentions only of attention layers. Mamba layers return `None` as the attention weights
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None if not use_cache else past_key_values

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_mamba_mask(self, attention_mask, cache_position):
        """
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None
        return mamba_mask

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: FalconHybridMambaAttentionDynamicCache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_attention_mask = (attention_mask[:, None, None, :] == attention_mask[:, None, :, None])[
                    :, :, -sequence_length:, :
                ].to(dtype)
                padding_mask = causal_mask[:, :, :, :mask_length] + padding_attention_mask
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class FalconH1ForCausalLM(LlamaForCausalLM):
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @auto_docstring
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FalconH1ForCausalLM

        >>> model = FalconH1ForCausalLM.from_pretrained("...")
        >>> tokenizer = AutoTokenizer.from_pretrained("...")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]) * self.model.lm_head_multiplier

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwitten -- has a unique cache type, `FalconHybridMambaAttentionDynamicCache`

        empty_past_kv = past_key_values is None

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if not empty_past_kv:
            if (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = FalconHybridMambaAttentionDynamicCache(
                self.config,
                input_ids.shape[0],
                self.dtype,
                devices=[
                    self.model.layers[i].mamba.conv1d.weight.device for i in range(self.config.num_hidden_layers)
                ],
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": cache_position,
            }
        )
        return model_inputs


__all__ = ["FalconH1Model", "FalconH1ForCausalLM", "FalconH1PreTrainedModel"]
