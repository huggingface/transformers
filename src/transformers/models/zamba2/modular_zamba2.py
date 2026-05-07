# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
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
import math
from collections.abc import Callable
from itertools import cycle

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...integrations.hub_kernels import lazy_load_kernel
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import resolve_internal_import
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from ..mamba2.modeling_mamba2 import pad_tensor_by_size, reshape_into_chunks, segment_sum
from ..zamba.modeling_zamba import (
    ZambaAttention,
    ZambaAttentionDecoderLayer,
    ZambaForCausalLM,
    ZambaForSequenceClassification,
    ZambaHybridLayer,
    ZambaMambaDecoderLayer,
    ZambaModel,
    ZambaRMSNorm,
    eager_attention_forward,
)
from .configuration_zamba2 import Zamba2Config


_CONFIG_FOR_DOC = "Zyphra/Zamba2-2.7B"

logger = logging.get_logger(__name__)


class Zamba2RMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, group_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        *prefix_dims, last_dim = hidden_states.shape
        group_count = last_dim // self.group_size
        hidden_states_group = hidden_states.view(*prefix_dims, group_count, self.group_size)
        variance = hidden_states_group.pow(2).mean(-1, keepdim=True)
        hidden_states_group = hidden_states_group * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states_group.view(*prefix_dims, group_count * self.group_size)
        return self.weight * hidden_states.to(input_dtype)


class Zamba2RMSNorm(ZambaRMSNorm):
    pass


class Zamba2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Zamba2Attention(ZambaAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
    The input dimension here is attention_hidden_size = 2 * hidden_size, and head_dim = attention_hidden_size // num_heads.
    The extra factor of 2 comes from the input being the concatenation of original_hidden_states with the output of the previous (mamba) layer
    (see fig. 2 in https://huggingface.co/papers/2405.16712).
    Additionally, replaced
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) with
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim/2)
    Finally, this attention layer contributes to tied transformer blocks aimed to increasing compute without increasing model size. Because this
    layer is tied, un-tied adapters (formally the same as LoRA but used in the base model) modules are added to the q, k, v projectors to increase
    expressivity with a small memory overhead (see Fig. 2 of https://huggingface.co/papers/2411.15242).
    """

    def __init__(
        self,
        config: Zamba2Config,
        layer_idx: int | None = None,
        num_fwd_mem_blocks: int | None = None,
        block_id: int | None = None,
    ):
        super().__init__(config, layer_idx)
        self.num_fwd_mem_blocks = num_fwd_mem_blocks
        self.layer_block_map = config.hybrid_layer_ids
        self.block_id = block_id

        if config.use_shared_attention_adapter:
            self.linear_q_adapter_list = nn.ModuleList([])
            self.linear_k_adapter_list = nn.ModuleList([])
            self.linear_v_adapter_list = nn.ModuleList([])

            for i in range(self.num_fwd_mem_blocks):
                if i % config.num_mem_blocks == block_id:
                    linear_q_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                    linear_k_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                    linear_v_adapter = nn.Sequential(
                        nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False),
                        nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False),
                    )
                else:
                    linear_q_adapter = nn.Identity()
                    linear_k_adapter = nn.Identity()
                    linear_v_adapter = nn.Identity()
                self.linear_q_adapter_list.append(linear_q_adapter)
                self.linear_k_adapter_list.append(linear_k_adapter)
                self.linear_v_adapter_list.append(linear_v_adapter)

        self.layer_dic = {value: index for index, value in enumerate(self.layer_block_map)}

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.config.use_shared_attention_adapter:
            adapter_layer_idx = self.layer_dic[layer_idx]
            query_states = query_states + self.linear_q_adapter_list[adapter_layer_idx](hidden_states)
            key_states = key_states + self.linear_k_adapter_list[adapter_layer_idx](hidden_states)
            value_states = value_states + self.linear_v_adapter_list[adapter_layer_idx](hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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


class Zamba2MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Zamba2Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * self.hidden_size)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = "silu"
        self.act = nn.SiLU()
        self.use_mem_eff_path = config.use_mem_eff_path

        self.n_groups = config.mamba_ngroups
        self.head_dim = config.mamba_headdim
        self.num_heads = self.config.n_mamba_heads
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=True,
            kernel_size=config.mamba_d_conv,
            groups=self.conv_dim,
            padding=config.mamba_d_conv - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.add_bias_linear,
        )
        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Zamba2RMSNormGated(
            self.intermediate_size, group_size=self.intermediate_size // self.n_groups, eps=1e-5
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)

        global causal_conv1d_update, causal_conv1d_fn
        global selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
        global is_fast_path_available

        if config.use_mamba_kernels:
            causal_conv1d = lazy_load_kernel("causal-conv1d")
            causal_conv1d_update = getattr(causal_conv1d, "causal_conv1d_update", None)
            causal_conv1d_fn = getattr(causal_conv1d, "causal_conv1d_fn", None)

            mamba_ssm = lazy_load_kernel("mamba-ssm")
            selective_state_update = resolve_internal_import(
                mamba_ssm, chained_path="ops.triton.selective_state_update.selective_state_update"
            )
            mamba_chunk_scan_combined = resolve_internal_import(
                mamba_ssm, chained_path="ops.triton.ssd_combined.mamba_chunk_scan_combined"
            )
            mamba_split_conv1d_scan_combined = resolve_internal_import(
                mamba_ssm, chained_path="ops.triton.ssd_combined.mamba_split_conv1d_scan_combined"
            )

            is_fast_path_available = all(
                (
                    selective_state_update,
                    mamba_chunk_scan_combined,
                    mamba_split_conv1d_scan_combined,
                    causal_conv1d_fn,
                    causal_conv1d_update,
                )
            )
        else:
            causal_conv1d_update = None
            causal_conv1d_fn = None
            selective_state_update = None
            mamba_chunk_scan_combined = None
            mamba_split_conv1d_scan_combined = None
            is_fast_path_available = False

        if getattr(config, "use_mamba_kernels", True) and not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        # set up dimensions for reshapes later

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        # getting projected states from cache if it exists
        if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
            in_projected_states = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
            d_mlp = (in_projected_states.shape[-1] - d_to_remove) // 2
            split_projection_dim = [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads]
            _, _, gate, hidden_states_B_C, dt = torch.split(in_projected_states, split_projection_dim, dim=-1)

            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.layers[self.layer_idx].conv_states,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())  # (nheads,)

            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.layers[self.layer_idx].recurrent_states,
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        # if no cache is found, calling the kernel
        else:
            if attention_mask is not None and not torch.all(attention_mask == 1):
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                dtype = hidden_states.dtype
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
            # 1. Gated MLP's linear projection
            projected_states = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}
            if attention_mask is not None:
                input_not_masked = torch.all(attention_mask == 1)
            else:
                input_not_masked = True

            if self.use_mem_eff_path and self.training and cache_params is None and input_not_masked:
                out, ssm_state = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=True,
                    **dt_limit_kwargs,
                )

            else:
                gate, hidden_states_B_C, time_step = torch.split(
                    projected_states,
                    [self.intermediate_size, self.conv_dim, self.num_heads],
                    dim=-1,
                )

                # 1D Convolution
                if cache_params is not None:
                    hidden_states_B_C_t = hidden_states_B_C.transpose(1, 2)
                    conv_state = nn.functional.pad(
                        hidden_states_B_C_t, (self.conv_kernel_size - hidden_states_B_C_t.shape[-1], 0)
                    )
                    conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
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
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )
                if attention_mask is not None and not torch.all(attention_mask == 1):
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
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
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_recurrent_state(ssm_state, self.layer_idx)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)
        return out

    # fmt: off
    def torch_forward(self, input_states, cache_params: Cache | None=None, attention_mask: torch.Tensor | None = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # Gated MLP's linear projection
        if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
            projected_states = self.in_proj(input_states)
        else:
            if attention_mask is not None:
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                input_states = (input_states * attention_mask[:, :, None]).to(dtype)
            projected_states = self.in_proj(input_states)
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size- self.num_heads) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
                [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
        )
        hidden_states = hidden_states.transpose(1, 2)

        use_precomputed_state = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        # Convolution sequence transformation
        if use_precomputed_state:
            conv_state = cache_params.update_conv_state(hidden_states, self.layer_idx)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]         # [batch, 1, intermediate_size] : decoding
        else:
            if cache_params is not None:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)

            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len].transpose(1, 2))
            if attention_mask is not None:
                dtype = hidden_states.dtype
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)
        A = -torch.exp(self.A_log.float())                            # [num_heads]
        if use_precomputed_state:
            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_min) #, self.time_step_max)
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = torch.exp(dt[..., None] * A)

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
            dBx = dB * hidden_states[..., None]

            # State calculation
            ssm_states = cache_params.layers[self.layer_idx].recurrent_states.clone()
            ssm_states = ssm_states * dA + dBx
            ssm_states = cache_params.update_recurrent_state(ssm_states, self.layer_idx)

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = ssm_states.to(C.dtype)  # Shape: [b, h, d, n]
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
            dt = torch.clamp(dt, self.time_step_min)
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len,  -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
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

            # First, contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, : ,:]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)


            # Step 2: Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Step 3: Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

            # (right term of low-rank factorization of off-diagonal blocks; B terms)

            decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
            # permute back B * decay states
            states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]  * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
            previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

            states_permuted = states.permute(0, 2, 1, 3, 4)
            result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
            new_states = result.permute(0, 2, 1, 3, 4)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            # compute Yoff
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
            if ssm_state is not None and cache_params is not None:
                cache_params.update_recurrent_state(ssm_state, self.layer_idx)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type and not is_torchdynamo_compiling():
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

        return self.torch_forward(hidden_states, cache_params, attention_mask)


class Zamba2MLP(nn.Module):
    def __init__(self, config: Zamba2Config, num_fwd_mem_blocks=None, block_id: int | None = None):
        """
        This MLP layer contributes to tied transformer blocks aimed to increasing compute without increasing model size. Because this layer
        is tied, un-tied adapter modules (formally same as LoRA, but used in the base model) are added to the up and gate projectors to increase expressivity with a small memory overhead.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_fwd_mem_blocks = num_fwd_mem_blocks
        self.block_id = block_id

        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=config.add_bias_linear)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_up_proj_adapter_list = nn.ModuleList([])
        for i in range(self.num_fwd_mem_blocks):
            if i % config.num_mem_blocks == block_id:
                gate_up_proj_adapter = nn.Sequential(
                    nn.Linear(self.config.hidden_size, self.config.adapter_rank, bias=False),
                    nn.Linear(self.config.adapter_rank, 2 * self.intermediate_size, bias=False),
                )
            else:
                gate_up_proj_adapter = nn.Identity()
            self.gate_up_proj_adapter_list.append(gate_up_proj_adapter)

        layer_block_map = config.hybrid_layer_ids
        self.layer_dic = {value: index for index, value in enumerate(layer_block_map)}

    def forward(self, hidden_state, layer_idx=None):
        gate_up_state = self.gate_up_proj(hidden_state)
        layer_idx = self.layer_dic[layer_idx]
        gate_up_state = gate_up_state + self.gate_up_proj_adapter_list[layer_idx](hidden_state)

        gate_up_state = torch.chunk(gate_up_state, 2, dim=-1)
        hidden_state = self.act_fn(gate_up_state[0]) * gate_up_state[1]
        output = self.down_proj(hidden_state)
        return output


class Zamba2AttentionDecoderLayer(ZambaAttentionDecoderLayer):
    def __init__(self, config: Zamba2Config, block_id: int | None = None, layer_idx: int | None = None):
        self.block_id = block_id
        num_gs = len(config.hybrid_layer_ids)
        super().__init__(config, layer_idx)
        self.self_attn = Zamba2Attention(config, layer_idx=-1, num_fwd_mem_blocks=num_gs, block_id=block_id)
        self.feed_forward = Zamba2MLP(config, num_fwd_mem_blocks=num_gs, block_id=block_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): output of previous Mamba layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output of shape `(batch, seq_len, embed_dim)`.
                This is concatenated with `hidden_states` (which is the output of the previous (mamba) layer). The
                concatenated tensor is then used as input of the pre-attention RMSNorm
                (see fig. 2 in https://huggingface.co/papers/2405.16712).
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, layer_idx)

        return hidden_states


class Zamba2MambaDecoderLayer(ZambaMambaDecoderLayer):
    def __init__(self, config: Zamba2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mamba = Zamba2MambaMixer(config=config, layer_idx=layer_idx)
        self.input_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Zamba2HybridLayer(ZambaHybridLayer):
    def __init__(
        self, shared_transformer: Zamba2AttentionDecoderLayer, linear: nn.Linear, mamba: Zamba2MambaDecoderLayer
    ):
        super().__init__(shared_transformer, linear, mamba)
        del self.shared_transf
        self.shared_transformer = shared_transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor | None = None,
        layer_idx: int | None = None,
        attention_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output that will be concatenated with
            hidden activations to form the input of the shared transformer layer.
            layer_idx (`int`): layer number.
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """

        transformer_hidden_states = self.shared_transformer(
            hidden_states,
            original_hidden_states=original_hidden_states,
            layer_idx=layer_idx,
            attention_mask=causal_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            **kwargs,
        )
        transformer_hidden_states = self.linear(transformer_hidden_states)

        hidden_states = self.mamba_decoder(
            hidden_states,
            transformer_hidden_states=transformer_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        return hidden_states


@auto_docstring
class Zamba2PreTrainedModel(PreTrainedModel):
    config: Zamba2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Zamba2HybridLayer", "Zamba2MambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_sdpa = True
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": Zamba2MambaDecoderLayer,
        "attentions": Zamba2Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Zamba2MambaMixer):
            dt = torch.exp(
                torch.rand(self.config.n_mamba_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            init.copy_(module.dt_bias, inv_dt)

            A = torch.arange(1, module.num_heads + 1)
            init.copy_(module.A_log, torch.log(A))
            init.ones_(module.D)


class Zamba2Model(ZambaModel, Zamba2PreTrainedModel):
    """
    Model consisting of *config.num_hidden_layers* layers.

    Args:
        config: Zamba2Config
    """

    def __init__(self, config: Zamba2Config):
        Zamba2PreTrainedModel.__init__(self, config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers_block_type = config.layers_block_type
        self.layers = self.get_layers()

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_mem_rope:
            if config.use_long_context:
                logger.warning_once(
                    "`use_long_context` set to `True`: using rescaled `rope_theta` and extended `max_position_embeddings`."
                )
            self.rotary_emb = Zamba2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_layers(self):
        layers = []
        self._tied_weights_keys = {}
        self.first_transformer_layer_id = 0
        unique_hybrid_blocks = []

        for layer_id, layer_type in enumerate(self.layers_block_type):
            mamba_layer = Zamba2MambaDecoderLayer(self.config, layer_idx=layer_id)
            if layer_type == "hybrid":
                prefix_pattern = f"layers.{layer_id}.shared_transformer"

                # Zamba ties Hybrid module weights by repeating blocks after every
                # `num_mem_blocks`. So if `num_mem_blocks=2`, the blocks looks like
                # [1, 2, 1, 2, 1, 2] where all "ones" share the same set of weights.
                if (
                    not isinstance(unique_hybrid_blocks, list)
                    or len(unique_hybrid_blocks) >= self.config.num_mem_blocks
                ):
                    if isinstance(unique_hybrid_blocks, list):
                        unique_hybrid_blocks = cycle(unique_hybrid_blocks)
                    target_pattern = next(unique_hybrid_blocks)
                    self._tied_weights_keys.update({prefix_pattern: target_pattern})
                else:
                    # Store source patterns to which the subsequent modules will be tied
                    unique_hybrid_blocks.append(prefix_pattern)

                block_id = layer_id % self.config.num_mem_blocks
                attn_block = Zamba2AttentionDecoderLayer(self.config, block_id=block_id)
                linear_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
                layers.append(Zamba2HybridLayer(attn_block, linear_layer, mamba_layer))
            else:
                layers.append(mamba_layer)
        return nn.ModuleList(layers)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        original_hidden_states = torch.clone(inputs_embeds)
        # original_hidden_states: word embedding output that will be concatenated with hidden activations to form the input of the shared transformer layer

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # create position embeddings to be shared across the decoder layers
        if self.config.use_mem_rope:
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        else:
            position_embeddings = None

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                original_hidden_states,
                layer_idx,
                attention_mask,
                causal_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Zamba2ForCausalLM(ZambaForCausalLM):
    def __init__(self, config: Zamba2Config):
        super().__init__(config)
        self.model = Zamba2Model(config)
        self.post_init()


class Zamba2ForSequenceClassification(ZambaForSequenceClassification):
    def __init__(self, config: Zamba2Config):
        super().__init__(config)
        self.model = Zamba2Model(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=pooled_logits, labels=labels, pooled_logits=pooled_logits, config=self.config, **kwargs
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = [
    "Zamba2ForCausalLM",
    "Zamba2ForSequenceClassification",
    "Zamba2Model",
    "Zamba2PreTrainedModel",
]
