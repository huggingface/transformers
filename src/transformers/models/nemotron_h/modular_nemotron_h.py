# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...integrations import use_experts_implementation
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE, DeepseekV3TopkRouter
from ...models.jamba.modeling_jamba import JambaAttention
from ...models.llama.modeling_llama import LlamaRMSNorm
from ...models.mamba2.modeling_mamba2 import pad_tensor_by_size, reshape_into_chunks, segment_sum
from ...models.nemotron.modeling_nemotron import NemotronMLP
from ...models.zamba.modeling_zamba import ZambaForCausalLM
from ...models.zamba2.modeling_zamba2 import Zamba2MambaMixer, Zamba2RMSNormGated
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_nemotron_h import NemotronHConfig


logger = logging.get_logger(__name__)

is_fast_path_available = False


class NemotronHMamba2Mixer(Zamba2MambaMixer):
    def __init__(self, config: NemotronHConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.mamba_hidden_act
        self.act = ACT2FN[config.mamba_hidden_act]
        self.use_mem_eff_path = True

        self.n_groups = config.n_groups
        self.head_dim = config.mamba_head_dim
        self.num_heads = config.mamba_num_heads

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads

        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )

        self.norm = Zamba2RMSNormGated(
            self.intermediate_size, group_size=self.intermediate_size // self.n_groups, eps=config.layer_norm_epsilon
        )

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)

    def forward(
        self,
        hidden_states,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type and not is_torchdynamo_compiling():
            # Use cuda stream to avoid NaN when using multiple GPUs, which is caused by multi-GPU synchronization issue.
            # Mamba might launch on the default cuda stream that not strictly respect the current Pytorch cuda stream.
            # This leads to kernel reading uninitialized memory before the data transfer is complete.
            with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
                return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

        return self.torch_forward(hidden_states, cache_params, attention_mask)

    # fmt: off
    def torch_forward(
        self,
        input_states,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        Override of Zamba2MambaMixer.torch_forward that fixes two bugs on the slow
        (no-kernel) path:

        1. **Inter-chunk SSM recurrence** (step 3): the inherited implementation
           permutes ``states`` to reduce over the wrong axis, producing incorrect
           output and SSM cache states whenever the sequence spans more than one
           chunk or when generation continues from a non-empty SSM cache.  The
           correct form – matching canonical Mamba2 (``modeling_mamba2.py``, fixed
           in #35154) – transposes ``decay_chunk`` instead and reduces over
           ``dim=1``.

        2. **``dt`` clamping**: the inherited code calls
           ``torch.clamp(dt, self.time_step_min)`` (one-sided lower bound only),
           introducing a spurious floor on the slow path that the CUDA-kernel path
           does not apply.  The correct call uses ``self.time_step_limit`` (a
           2-tuple) to match the kernel path.

        See issue #47246 for the full analysis and a self-contained reproduction.
        """
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
            projected_states = self.in_proj(input_states)
        else:
            if attention_mask is not None:
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                input_states = (input_states * attention_mask[:, :, None]).to(dtype)
            projected_states = self.in_proj(input_states)

        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )
        hidden_states = hidden_states.transpose(1, 2)

        use_precomputed_state = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        )
        if use_precomputed_state:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]

        # 2. Convolution sequence transformation
        if use_precomputed_state and seq_len == 1:
            conv_states = cache_params.update_conv_state(hidden_states, self.layer_idx)[..., -self.conv_kernel_size:]
            hidden_states = torch.sum(conv_states * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states = hidden_states + self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]  # [batch, 1, intermediate_size]
        else:
            if use_precomputed_state:
                # chunked prefill / speculative verify: prepend cached left context
                hidden_states = torch.cat([conv_state, hidden_states], dim=-1)
            if cache_params is not None:
                conv_states = F.pad(
                    hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                conv_states = cache_params.update_conv_state(conv_states, self.layer_idx)[..., -self.conv_kernel_size:]
            hidden_states = self.act(
                self.conv1d(hidden_states)[..., :hidden_states.shape[-1]].transpose(1, 2)
            )
            if use_precomputed_state:
                hidden_states = hidden_states[:, -seq_len:, :]
            if attention_mask is not None:
                dtype = hidden_states.dtype
                # tune out hidden states for pad tokens
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        hidden_states, B, C = torch.split(
            hidden_states,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())  # [num_heads]

        # 3. SSM transformation
        if use_precomputed_state and seq_len == 1:
            # Single-step decode path
            dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
            dt = F.softplus(dt + dt_bias.to(dt.dtype))
            # FIX 2: clamp with full time_step_limit (2-tuple) to match cuda_kernels_forward
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dA = torch.exp(dt[..., None] * A)
            # Discretize B
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            dB = dt[..., None] * B[..., None, :]
            # Discretize x into dB
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = dB * hidden_states[..., None]
            # State update
            ssm_states = cache_params.layers[self.layer_idx].recurrent_states[0].clone()
            ssm_states = ssm_states * dA + dBx
            ssm_states = cache_params.update_recurrent_state(ssm_states, self.layer_idx)
            # Output
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            ssm_states = ssm_states.to(C.dtype)
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # Prefill path: chunked SSD naive implementation
            dt = F.softplus(dt + self.dt_bias)
            # FIX 2: clamp with full time_step_limit (2-tuple) to match cuda_kernels_forward
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [
                reshape_into_chunks(t, pad_size, self.chunk_size)
                for t in (hidden_states, A, B, C)
            ]

            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # Step 1: Intra-chunk output (diagonal blocks)
            L = torch.exp(segment_sum(A))
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # (b, c, l, s, h)
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            # Step 2: Per-chunk states (right term of off-diagonal factorization; B terms)
            decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
            states = (
                B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]
                * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]
            ).sum(dim=3).permute(0, 1, 2, 4, 3)
            previous_states = (
                cache_params.layers[self.layer_idx].recurrent_states[0][:, None].to(
                    dtype=states.dtype, device=states.device
                )
                if use_precomputed_state
                else torch.zeros_like(states[:, :1])
            )
            states = torch.cat([previous_states, states], dim=1)

            # Step 3: Inter-chunk SSM recurrence (middle term; A terms)
            # FIX 1: transpose decay_chunk over dim (1,3) and reduce over dim=1,
            # matching canonical Mamba2 (#35154).  The inherited Zamba2 code
            # instead permutes `states` (dims 1↔2) and reduces over dim=2, which
            # collapses the wrong axis and produces incorrect outputs whenever
            # num_chunks > 1 or the SSM cache is non-zero.
            decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # Step 4: State → output (left term of off-diagonal factorization; C terms)
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

            # Combine intra-chunk and inter-chunk contributions
            y = Y_diag + Y_off
            # [bsz, -1, chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
            y = y + D_residual
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)
            if ssm_state is not None and cache_params is not None:
                cache_params.update_recurrent_state(ssm_state, self.layer_idx)

        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states
    # fmt: on


class NemotronHRMSNorm(LlamaRMSNorm):
    pass


class NemotronHMLP(NemotronMLP, nn.Module):
    def __init__(self, config, intermediate_size=None, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]


@use_experts_implementation(has_gate=False)
class NemotronHExperts(nn.Module):
    """
    Collection of expert weights stored as 3D tensors.

    **Architecture Note**: Unlike Mixtral or DeepSeek which use gated MLPs,
    NemotronH uses a standard MLP architecture with only up_proj and down_proj
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # Determine input/output dimension based on whether latent projection is used
        input_dim = config.moe_latent_size if config.moe_latent_size is not None else config.hidden_size

        # All expert weights stored as 3D tensors: (num_experts, out_dim, in_dim)
        # up_proj: (num_experts, intermediate_dim, input_dim)
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, input_dim))
        # down_proj: (num_experts, input_dim, intermediate_dim)
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, input_dim, self.intermediate_dim))

        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)

        # Create expert mask to identify which tokens go to which experts
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, num_experts_per_tok, num_tokens)
            # Only iterate over experts that have at least one token assigned
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)

        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            # Find which tokens are routed to this expert
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if token_idx.numel() == 0:
                continue

            # Get input for this expert
            current_state = hidden_states[token_idx]

            # Expert computation: down_proj(act_fn(up_proj(x)))
            # No gating mechanism unlike Mixtral which uses: down_proj(act_fn(gate_proj(x)) * up_proj(x))
            current_hidden_states = torch.nn.functional.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(current_hidden_states)
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])

            # Apply routing weights
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            # Accumulate into final output
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


class NemotronHMoE(DeepseekV3MoE):
    """
    Mixture-of-Experts (MoE) module for NemotronH.

    Unique architectures:
    - Uses non-gated MLP experts (NemotronHExperts) instead of gated experts
    - Adds optional latent projection for computational efficiency
    """

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config)

        # Replace with NemotronH-specific experts (non-gated MLP architecture)
        self.experts = NemotronHExperts(config)
        self.gate = NemotronHTopkRouter(config)

        # Override shared_experts to use NemotronHMLP with correct intermediate size
        self.shared_experts = NemotronHMLP(config=config, intermediate_size=config.moe_shared_expert_intermediate_size)

        # NemotronH-specific latent projection layers
        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=config.mlp_bias)
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # NemotronH-specific: latent projection
        hidden_states = self.fc1_latent_proj(hidden_states)
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights)
        hidden_states = self.fc2_latent_proj(hidden_states)

        hidden_states = hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class NemotronHTopkRouter(DeepseekV3TopkRouter):
    pass


class NemotronHAttention(JambaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return super().forward(hidden_states, attention_mask, past_key_values, **kwargs)


MIXER_TYPES = {
    "linear_attention": NemotronHMamba2Mixer,
    "full_attention": NemotronHAttention,
    "moe": NemotronHMoE,
    "mlp": NemotronHMLP,
}


class NemotronHBlock(GradientCheckpointingLayer):
    """
    A single transformer block in the NemotronH model.

    This block can contain different types of mixers (Mamba, Attention, MLP, or MoE)
    depending on the configuration. Each block applies pre-normalization followed by
    the mixer, then adds a residual connection.

    Args:
        config (`NemotronHConfig`):
            Model configuration specifying the block architecture.
        layer_idx (`int`):
            Index of this block in the model. Used to determine the block type from
            `config.layers_block_type[layer_idx]`.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.block_type = config.layers_block_type[layer_idx]
        self.mixer = MIXER_TYPES[self.block_type](config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

        if self.block_type == "linear_attention":
            hidden_states = self.mixer(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)
        elif self.block_type == "full_attention":
            hidden_states, _ = self.mixer(
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class NemotronHPreTrainedModel(PreTrainedModel):
    config: NemotronHConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NemotronHBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _is_stateful = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": NemotronHBlock,
        "attentions": NemotronHAttention,
    }
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
    ]
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        super()._init_weights(module)
        if isinstance(module, NemotronHMamba2Mixer):
            # Only re-initialise params that were NOT loaded from a checkpoint.
            # `_is_hf_initialized` is set by `from_pretrained` on each loaded
            # parameter; without this guard a post-load safety pass of
            # `_init_weights` would overwrite checkpoint values of
            # A_log / D / dt_bias with fresh random draws.
            if not getattr(module.A_log, "_is_hf_initialized", False):
                A = torch.arange(1, self.config.mamba_num_heads + 1)
                init.copy_(module.A_log, torch.log(A))
            if not getattr(module.D, "_is_hf_initialized", False):
                init.ones_(module.D)
            if not getattr(module.dt_bias, "_is_hf_initialized", False):
                dt = torch.exp(
                    torch.rand(self.config.mamba_num_heads)
                    * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                    + math.log(self.config.time_step_min)
                ).clamp(min=self.config.time_step_floor)

                # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                with torch.no_grad():
                    init.copy_(module.dt_bias, inv_dt)
        elif isinstance(module, NemotronHTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, NemotronHExperts):
            # Initialize expert weights
            init.normal_(module.up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_is_hf_initialized", False):
                    init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/sqrt(N) where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name == "out_proj.weight":
                    # Skip checkpoint-loaded weights so a post-load safety
                    # pass of `_init_weights` doesn't silently overwrite them.
                    if getattr(p, "_is_hf_initialized", False):
                        continue
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p_new = p / math.sqrt(self.config.num_hidden_layers)
                        init.copy_(p, p_new)


class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NemotronHBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        hidden_states = inputs_embeds

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Under a compileable cache, `generate()` precomputes per-pattern masks and hands them in as a dict;
        # otherwise we build them here.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        for layer_idx, mixer_block in enumerate(self.layers):
            hidden_states = mixer_block(
                hidden_states,
                attention_mask=causal_mask_mapping.get(mixer_block.block_type),
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm_f(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class NemotronHForCausalLM(ZambaForCausalLM):
    _tied_weights_keys = {}

    @staticmethod
    def create_masks_for_generate(config, inputs_embeds, attention_mask, past_key_values, position_ids=None, **_):
        # Nemotron-H layer_types include non-attention block types (moe / mlp) that the default dispatch
        # table doesn't enumerate, so we return both masks the forward needs as a dict.
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        return {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }

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
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "NemotronHPreTrainedModel",
    "NemotronHModel",
    "NemotronHForCausalLM",
]
