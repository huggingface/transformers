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
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...integrations import use_experts_implementation
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE, DeepseekV3TopkRouter
from ...models.jamba.modeling_jamba import JambaAttention
from ...models.llama.modeling_llama import LlamaRMSNorm
from ...models.nemotron.modeling_nemotron import NemotronMLP
from ...models.zamba.modeling_zamba import ZambaForCausalLM
from ...models.zamba2.modeling_zamba2 import Zamba2HybridDynamicCache, Zamba2MambaMixer, Zamba2RMSNormGated
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..granitemoehybrid.modular_granitemoehybrid import GraniteMoeHybridDecoderLayer
from .configuration_nemotron_h import NemotronHConfig


logger = logging.get_logger(__name__)

is_fast_path_available = False


class NemotronHHybridDynamicCache(Zamba2HybridDynamicCache):
    def __init__(
        self, config: NemotronHConfig, batch_size: int, dtype: torch.dtype = torch.float16, device: str | None = None
    ):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.has_previous_state = False
        self.intermediate_size = int(config.mamba_num_heads * config.mamba_head_dim)
        self.ssm_state_size = config.ssm_state_size
        self.conv_kernel_size = config.conv_kernel
        self.n_mamba_heads = config.mamba_num_heads
        self.transformer_layers = []
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self.conv_states = {}
        self.ssm_states = {}
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] != "attention":
                # Allocate mamba cache for mamba and moe layers (both have a mamba sublayer)
                self.conv_states[i] = torch.zeros(
                    batch_size,
                    self.intermediate_size + 2 * config.n_groups * self.ssm_state_size,
                    self.conv_kernel_size,
                    device=device,
                    dtype=dtype,
                )
                self.ssm_states[i] = torch.zeros(
                    batch_size,
                    self.n_mamba_heads,
                    config.mamba_head_dim,
                    self.ssm_state_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                self.conv_states[i] = torch.tensor([[]] * batch_size, device=device)
                self.ssm_states[i] = torch.tensor([[]] * batch_size, device=device)

            if self.layers_block_type[i] == "attention":
                self.transformer_layers.append(i)
        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]


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
        cache_params: NemotronHHybridDynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type and not is_torchdynamo_compiling():
            with torch.cuda.stream(torch.cuda.default_stream(hidden_states.device)):
                return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

        return self.torch_forward(hidden_states, cache_params, attention_mask)


class NemotronHRMSNorm(LlamaRMSNorm):
    pass


class NemotronHMLP(NemotronMLP):
    def __init__(self, config, intermediate_size=None):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.mlp_hidden_act]


class NemotronHAttention(JambaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: NemotronHHybridDynamicCache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # position_embeddings not used; RoPE is computed internally by JambaAttention
        return super().forward(hidden_states, attention_mask, past_key_values, **kwargs)


@use_experts_implementation(has_gate=False)
class NemotronHExperts(nn.Module):
    """
    Non-gated parallel expert weights. Unlike gated MLP experts (Mixtral, DeepSeek),
    NemotronH uses: down_proj(act_fn(up_proj(x))).
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        input_dim = config.moe_latent_size if config.moe_latent_size is not None else config.hidden_size

        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, input_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, input_dim, self.intermediate_dim))

        self.act_fn = ACT2FN[config.mlp_hidden_act]

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)

        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if token_idx.numel() == 0:
                continue

            current_state = hidden_states[token_idx]
            current_hidden_states = torch.nn.functional.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(current_hidden_states)
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


class NemotronHMoE(DeepseekV3MoE):
    """MoE with non-gated experts and optional latent projection for efficiency."""

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config)

        self.experts = NemotronHExperts(config)
        self.gate = NemotronHTopkRouter(config)
        self.shared_experts = NemotronHMLP(config=config, intermediate_size=config.moe_shared_expert_intermediate_size)

        if config.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=config.mlp_bias)
            self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=config.mlp_bias)
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        hidden_states = self.fc1_latent_proj(hidden_states)
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights)
        hidden_states = self.fc2_latent_proj(hidden_states)

        hidden_states = hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class NemotronHTopkRouter(DeepseekV3TopkRouter):
    pass


class NemotronHMambaDecoderLayer(GraniteMoeHybridDecoderLayer):
    """
    Mamba-primary decoder layer for NemotronH.

    Handles both "mamba" and "moe" block types:
    - stage 1: NemotronHMamba2Mixer (always present)
    - stage 2: NemotronHMoE (for "moe" blocks) or NemotronHMLP (for "mamba" blocks)

    Inherits forward from GraniteMoeHybridDecoderLayer.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        nn.Module.__init__(self)
        block_type = config.layers_block_type[layer_idx]
        self.mamba = NemotronHMamba2Mixer(config, layer_idx)
        self.self_attn = None
        self.has_experts = block_type == "moe"
        self.block_sparse_moe = NemotronHMoE(config) if self.has_experts else None
        self.shared_mlp = NemotronHMLP(config)
        self.input_layernorm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.residual_multiplier = getattr(config, "residual_multiplier", 1.0)


class NemotronHAttentionDecoderLayer(GraniteMoeHybridDecoderLayer):
    """
    Attention decoder layer for NemotronH.

    Handles "attention" block type:
    - stage 1: NemotronHAttention
    - stage 2: NemotronHMLP

    Inherits forward from GraniteMoeHybridDecoderLayer.
    """

    def __init__(self, config: NemotronHConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.mamba = None
        self.self_attn = NemotronHAttention(config, layer_idx)
        self.has_experts = False
        self.block_sparse_moe = None
        self.shared_mlp = NemotronHMLP(config)
        self.input_layernorm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.residual_multiplier = getattr(config, "residual_multiplier", 1.0)


class NemotronHPreTrainedModel(PreTrainedModel):
    config: NemotronHConfig
    base_model_prefix = "model"
    _no_split_modules = ["NemotronHMambaDecoderLayer", "NemotronHAttentionDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": (NemotronHMambaDecoderLayer, NemotronHAttentionDecoderLayer),
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
            A = torch.arange(1, self.config.mamba_num_heads + 1)
            init.copy_(module.A_log, torch.log(A))
            init.ones_(module.D)

            dt = torch.exp(
                torch.rand(self.config.mamba_num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                init.copy_(module.dt_bias, inv_dt)
            module.dt_bias._no_reinit = True
        elif isinstance(module, NemotronHTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, NemotronHExperts):
            init.normal_(module.up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name == "out_proj.weight":
                    init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p_new = p / math.sqrt(self.config.num_hidden_layers)
                        init.copy_(p, p_new)


class NemotronHModel(NemotronHPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            NemotronHMambaDecoderLayer(config, i) if config.layers_block_type[i] != "attention"
            else NemotronHAttentionDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])

        self.norm_f = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
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
        past_key_values: NemotronHHybridDynamicCache | None = None,
        use_cache: bool | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = NemotronHHybridDynamicCache(
                config=self.config,
                batch_size=inputs_embeds.shape[0],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        hidden_states = inputs_embeds

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        mamba_mask = self._update_mamba_mask(attention_mask, past_key_values)

        for layer in self.layers:
            # Attention layers use causal mask; mamba/moe layers use mamba mask
            layer_mask = causal_mask if isinstance(layer, NemotronHAttentionDecoderLayer) else mamba_mask

            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm_f(hidden_states)

        if past_key_values is not None and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def _update_mamba_mask(self, attention_mask, past_key_values):
        """
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        mamba_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            mamba_mask = None
        return mamba_mask


class NemotronHForCausalLM(ZambaForCausalLM):
    _tied_weights_keys = {}

    def __init__(self, config: NemotronHConfig):
        # Call PreTrainedModel.__init__ directly to avoid ZambaForCausalLM creating ZambaModel
        NemotronHPreTrainedModel.__init__(self, config)
        self.model = NemotronHModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: NemotronHHybridDynamicCache | None = None,
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
