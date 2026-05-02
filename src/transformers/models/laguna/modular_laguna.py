# Copyright 2026 Poolside and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Laguna model."""

from collections.abc import Callable
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ...utils.generic import TransformersKwargs
from ..afmoe.modeling_afmoe import AfmoeAttention
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer
from ..llama.modeling_llama import LlamaModel, eager_attention_forward
from ..qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeMLP, Qwen2MoePreTrainedModel, Qwen2MoeRMSNorm
from ..qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter, apply_rotary_pos_emb
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts, Qwen3MoeSparseMoeBlock


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="poolside/laguna-XS.2")
@strict
class LagunaConfig(Qwen2MoeConfig):
    r"""
    num_attention_heads_per_layer (`list[int]`, *optional*):
        Per-layer override for ``num_attention_heads``. Length must equal ``num_hidden_layers``.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type — ``"dense"`` or ``"sparse"``. Length must equal
        ``num_hidden_layers``. Defaults to first layer dense, rest sparse.
    moe_routed_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scalar applied to routed-expert output before combining with the shared-expert output.
    moe_apply_router_weight_on_input (`bool`, *optional*, defaults to `False`):
        Whether to apply router weights to the MoE input rather than the output. Not supported
        in transformers yet; ``True`` will raise a ``NotImplementedError`` for now.
    moe_router_logit_softcapping (`float`, *optional*, defaults to 0.0):
        Scaling factor when applying tanh softcapping on the logits of the MoE router logits.

    Example:

    ```python
    >>> from transformers import LagunaModel, LagunaConfig

    >>> configuration = LagunaConfig()
    >>> model = LagunaModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "laguna"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }

    vocab_size: int = 100352
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    sliding_window: int = 512

    # Laguna-specific attention
    head_dim: int = 128
    attention_bias: bool = False
    num_attention_heads_per_layer: list[int] | None = None
    # Laguna-specific MoE
    mlp_layer_types: list[str] | None = None
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0

    # Fields declared by Qwen2MoeConfig but not used by Laguna. ``= AttributeError()``
    # tells modular to drop these from the materialized child.
    decoder_sparse_step = AttributeError()
    mlp_only_layers = AttributeError()
    qkv_bias = AttributeError()
    norm_topk_prob = AttributeError()
    use_sliding_window = AttributeError()
    max_window_layers = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers

        default_rope_params: dict[Literal["full_attention", "sliding_attention"], dict[str, Any]] = {
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        # rope_parameters is keyed by layer type; tell the validator those keys are intentional.
        PreTrainedConfig.__post_init__(
            self, **kwargs, ignore_keys_at_rope_validation={"sliding_attention", "full_attention"}
        )

    def convert_rope_params_to_dict(self, **kwargs):
        # No need to handle BC for new models, because they have no old-format `rope_scaling`
        return kwargs

    def validate_architecture(self):
        """Part of ``@strict``-powered validation."""
        if self.moe_apply_router_weight_on_input:
            raise NotImplementedError(
                "moe_apply_router_weight_on_input=True is not yet supported in the "
                "transformers implementation of Laguna."
            )
        if (
            self.num_attention_heads_per_layer is not None
            and len(self.num_attention_heads_per_layer) != self.num_hidden_layers
        ):
            raise ValueError(
                f"num_attention_heads_per_layer length ({len(self.num_attention_heads_per_layer)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types length ({len(self.mlp_layer_types)}) "
                f"must equal num_hidden_layers ({self.num_hidden_layers})."
            )


class LagunaRMSNorm(Qwen2MoeRMSNorm):
    pass


class LagunaRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: LagunaConfig):
        super().__init__(config)

    @staticmethod
    def compute_default_rope_parameters(
        config: LagunaConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters[layer_type]["rope_theta"]
        # key difference to gemma3: partial rope
        partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class LagunaMLP(Qwen2MoeMLP):
    pass


class LagunaTopKRouter(Qwen3_5MoeTopKRouter):
    def __init__(self, config):
        super().__init__()
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)
        self.router_logit_softcapping = config.moe_router_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight).float()
        # Optional logits softcapping
        if self.router_logit_softcapping > 0.0:
            router_logits = torch.tanh(router_logits / self.router_logit_softcapping) * self.router_logit_softcapping
        # Sigmoid instead of softmax normalization
        routing_scores = torch.sigmoid(router_logits)

        scores_for_selection = routing_scores + self.e_score_correction_bias.to(routing_scores.dtype)
        _, selected_experts = torch.topk(scores_for_selection, self.top_k, dim=-1)
        routing_weights = routing_scores.gather(-1, selected_experts)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        return router_logits, routing_weights, selected_experts


class LagunaExperts(Qwen3MoeExperts):
    pass


class LagunaSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def __init__(self, config: LagunaConfig):
        super().__init__(config)
        self.shared_experts = LagunaMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.routed_scaling_factor = config.moe_routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = self.shared_experts(hidden_states)

        _, routing_weights, selected_experts = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
        # Additional scaling
        hidden_states = hidden_states * self.routed_scaling_factor
        hidden_states = hidden_states + shared_output

        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states


class LagunaAttention(AfmoeAttention):
    """Afmoe-style SWA/GQA attention with Laguna-specific gating and per-layer head count."""

    def __init__(self, config: LagunaConfig, layer_idx: int, num_heads: int):
        # Number of heads is controlled via `config.num_attention_heads_per_layer`
        self.num_heads = num_heads

        super().__init__(config, layer_idx)
        self.num_key_value_groups = self.num_heads // config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # Custom per-head gating
        del self.gate_proj
        self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        gate = F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype)
        attn_output = (attn_output.view(*input_shape, -1, self.head_dim) * gate.unsqueeze(-1)).view(*input_shape, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LagunaDecoderLayer(Glm4MoeLiteDecoderLayer):
    def __init__(self, config: LagunaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = LagunaAttention(config, layer_idx, config.num_attention_heads_per_layer[layer_idx])
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = LagunaSparseMoeBlock(config)
        else:
            self.mlp = LagunaMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LagunaPreTrainedModel(Qwen2MoePreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LagunaTopKRouter):
            torch.nn.init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, LagunaRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


class LagunaModel(LlamaModel):
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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

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
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class LagunaForCausalLM(Qwen2MoeForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return super().forward(**super_kwargs)


__all__ = [
    "LagunaConfig",
    "LagunaForCausalLM",
    "LagunaModel",
    "LagunaPreTrainedModel",
]
