# Copyright 2026 Reflection AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Beam model."""

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3PreTrainedModel,
    DeepseekV3TopkRouter,
)
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from ..deepseek_v32.modeling_deepseek_v32 import DeepseekV32DecoderLayer
from ..olmo3.modeling_olmo3 import Olmo3RMSNorm, Olmo3RotaryEmbedding, apply_rotary_pos_emb
from ..qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
    eager_attention_forward,
)


@auto_docstring(checkpoint="reflection/Beam-1.0-501B-A23B")
@strict
class BeamConfig(Qwen3MoeConfig):
    r"""
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type, `"dense"` for a SwiGLU MLP or `"sparse"` for the MoE block. Length must
        equal `num_hidden_layers`. Defaults to three dense layers followed by sparse ones.
    n_group (`int`, *optional*, defaults to 8):
        Number of groups for routed experts.

    ```python
    >>> from transformers import BeamConfig

    >>> # Initializing a Beam configuration
    >>> configuration = BeamConfig()
    ```
    """

    model_type = "beam"
    # Beam uses `num_experts` directly; Qwen3-MoE's `num_local_experts` alias does not apply.
    attribute_map = AttributeError()

    base_model_tp_plan = {
        # trf-ignore: TRF035
        # TODO: the attn gate bias needs to be local -> new colwise local style  # noqa: NC001
        # "layers.*.self_attn.q_proj": "colwise",
        # "layers.*.self_attn.k_proj": "colwise",
        # "layers.*.self_attn.v_proj": "colwise",
        # "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        # "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        # "layers.*.self_attn.gate_proj": "colwise",
        # "layers.*.self_attn.gate_bias": "colwise",
        # "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    hidden_size: int = 5120
    intermediate_size: int = 20480
    num_hidden_layers: int = 52
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 8192
    sliding_window: int | None = 512
    moe_intermediate_size: int = 2560
    num_experts: int = 256
    norm_topk_prob: bool = True

    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None

    n_group: int | None = 8
    topk_group: int | None = 4
    routed_scaling_factor: int | float = 2.5

    final_logit_softcapping: int | float | None = 30.0

    # Fields declared by Qwen3MoeConfig but not used by Beam.
    decoder_sparse_step = AttributeError()
    mlp_only_layers = AttributeError()
    router_aux_loss_coef = AttributeError()
    use_sliding_window = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            # Every sixth layer runs full attention, and so does the last one.
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 6) else "full_attention" for i in range(self.num_hidden_layers)
            ]
            self.layer_types[-1] = "full_attention"
        if self.mlp_layer_types is None:
            num_dense_layers = min(3, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * num_dense_layers + ["sparse"] * (
                self.num_hidden_layers - num_dense_layers
            )
        if self.rope_parameters is None:
            # Keyed by layer type; `None` marks a layer type that runs without positional encoding.
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": self.default_theta},
                "full_attention": None,
            }
        PreTrainedConfig.__post_init__(self, **kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # The generic converter assumes a flat mapping and would add top-level RoPE keys.
        return kwargs

    def validate_architecture(self):
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{self.num_attention_heads} % {self.num_key_value_heads} != 0"
            )


def depth_scaled_residual_add(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    residual_add_idx: int,
) -> torch.Tensor:
    hidden_states_scale = 1.0 / math.sqrt(residual_add_idx)
    residual_scale = math.sqrt((residual_add_idx - 1) / residual_add_idx)
    return hidden_states * hidden_states_scale + residual * residual_scale


class BeamRMSNorm(Olmo3RMSNorm):
    pass


class BeamRotaryEmbedding(Olmo3RotaryEmbedding):
    pass


def compute_attention_scaling(position_ids: torch.LongTensor) -> torch.Tensor:
    """Scalable-softmax query scale, `(1 + position / 2048) ** 0.1`, for layers that run without RoPE."""
    return torch.exp(0.1 * torch.log1p((position_ids.float() + 1) / 2048))[:, None, :, None]


class BeamAttention(Qwen3MoeAttention):
    def __init__(self, config: BeamConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        # One gate bias per head, broadcast over `head_dim`.
        self.gate_bias = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        attention_scaling: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Sliding-attention layers use RoPE; full-attention layers use NoPE with position-dependent query scaling.
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if attention_scaling is not None:
            query_states = (query_states.float() * attention_scaling).to(query_states.dtype)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        gate = self.gate_proj(hidden_states).view_as(attn_output) + self.gate_bias[None, None, :, None]
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class BeamExperts(DeepseekV4Experts):
    def __init__(self, config: BeamConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_dim = config.moe_intermediate_size
        del self.limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return (self.act_fn(gate.float()) * up.float()).to(gate.dtype)


class BeamMLP(Qwen3MoeMLP):
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Beam evaluates the SwiGLU activation and product in FP32.
        activated = (self.act_fn(gate.float()) * up.float()).to(gate.dtype)
        return self.down_proj(activated)


class BeamTopKRouter(DeepseekV3TopkRouter):
    def __init__(self, config: BeamConfig):
        super().__init__()
        self.num_experts = config.num_experts
        del self.routed_scaling_factor

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.num_group, self.num_experts // self.num_group)
            .reshape(-1, self.num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        # `routed_scaling_factor` is applied after expert aggregation in `BeamMoE`.
        return router_logits, topk_weights, topk_indices


class BeamMoE(Qwen3MoeSparseMoeBlock):
    def __init__(self, config: BeamConfig):
        super().__init__(config)
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        final_hidden_states = final_hidden_states * self.routed_scaling_factor
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class BeamDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: BeamConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.pre_feedforward_layernorm = BeamRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = BeamRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_residual_add_idx = 2 * (layer_idx + 1)
        self.mlp_residual_add_idx = self.attn_residual_add_idx + 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_scaling: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            attention_scaling=attention_scaling,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = depth_scaled_residual_add(hidden_states, residual, self.attn_residual_add_idx)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = depth_scaled_residual_add(hidden_states, residual, self.mlp_residual_add_idx)
        return hidden_states


class BeamPreTrainedModel(DeepseekV3PreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(BeamTopKRouter, index=0),
        "hidden_states": BeamDecoderLayer,
        "attentions": BeamAttention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "mlp.gate.weight"]
    # DeepseekV3 uses this to drop its MTP layer, which Beam does not have.
    _keys_to_ignore_on_load_unexpected = AttributeError()

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, BeamTopKRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, BeamExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BeamRotaryEmbedding):
            for layer_type in module.rope_type:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, BeamAttention):
            init.zeros_(module.gate_bias)


class BeamModel(Qwen3MoeModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_creation_functions = {
                "full_attention": lambda: create_causal_mask(**mask_kwargs),
                "sliding_attention": lambda: create_sliding_window_causal_mask(**mask_kwargs),
            }
            causal_mask_mapping = {}
            for layer_type in set(self.config.layer_types):
                causal_mask_mapping[layer_type] = mask_creation_functions[layer_type]()

        hidden_states = inputs_embeds
        position_embeddings, attention_scalings = {}, {}
        for layer_type in set(self.config.layer_types):
            if self.config.rope_parameters[layer_type] is None:
                # Full-attention layers use NoPE with position-dependent attention scaling.
                position_embeddings[layer_type] = None
                attention_scalings[layer_type] = compute_attention_scaling(position_ids)
            else:
                position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)
                attention_scalings[layer_type] = None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings[layer_type],
                attention_scaling=attention_scalings[layer_type],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class BeamForCausalLM(DeepseekV3ForCausalLM):
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
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            # Softcap in float32, as bf16 tanh quantizes badly near saturation.
            logits = logits.float() / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "BeamConfig",
    "BeamForCausalLM",
    "BeamModel",
    "BeamPreTrainedModel",
]
