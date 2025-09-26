# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All Rights Reserved.
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
"""PyTorch Ernie 4.5 MoE model."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import OutputRecorder, check_model_inputs
from ..ernie4_5.modeling_ernie4_5 import Ernie4_5RotaryEmbedding, apply_rotary_pos_emb, rotate_half  # noqa: F401
from ..llama.modeling_llama import LlamaAttention, LlamaRMSNorm
from ..mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralPreTrainedModel,
)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeMLP
from .configuration_ernie4_5_moe import Ernie4_5_MoeConfig


logger = logging.get_logger(__name__)


class Ernie4_5_MoeRMSNorm(LlamaRMSNorm):
    pass


class Ernie4_5_MoeMLP(Qwen3MoeMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config, intermediate_size)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)


class Ernie4_5_MoeRotaryEmbedding(Ernie4_5RotaryEmbedding):
    def __init__(self, config: Ernie4_5_MoeConfig, device=None):
        super().__init__(config, device)


class Ernie4_5_MoeAttention(LlamaAttention):
    def __init__(self, config: Ernie4_5_MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.attention_dropout = 0.0

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.use_bias)


class Ernie4_5_MoeStatics(nn.Module):
    """
    Stores MoE (Mixture of Experts) statistics
        - Bias for the gating
        - Additionally, usage per expert in the original codebase
    """

    def __init__(self, config):
        super().__init__()

        num_experts_groups = 1
        num_experts = config.moe_num_experts

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts_groups, num_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        # NOTE: This is a workaround to enable TP with a module that only has parameters
        #
        # Otherwise, it stays as `DTensor` when called in the "super" forward
        #   1. All other tensors are local (`torch.Tensor`)
        #   2. Isolate does not work on `nn.Module` which only has parameters
        return hidden_states + self.e_score_correction_bias.squeeze()


class Ernie4_5_MoeSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.

    Ernie 4.5 MoE's original formula is based on case (2) with
    (optional) shared experts and a corrections bias during gating.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_k

        # correction bias (yes it seems to be a typo with statics <> statistics)
        self.moe_statics = Ernie4_5_MoeStatics(config)

        # gating
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False, dtype=torch.float32)
        self.experts = nn.ModuleList(
            [Ernie4_5_MoeMLP(config, config.moe_intermediate_size) for _ in range(config.moe_num_experts)]
        )
        self.norm_min = config.moe_norm_min

        # (optional) shared experts for all forwards
        self.shared_experts = None
        if config.moe_num_shared_experts > 0:
            self.shared_experts = Ernie4_5_MoeMLP(config, config.moe_intermediate_size * config.moe_num_shared_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # (Optional) shared experts
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states.float())

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(self.moe_statics(routing_weights), self.top_k, dim=-1)
            routing_weights = torch.gather(routing_weights, dim=-1, index=selected_experts)
            routing_weights = routing_weights / torch.clamp(
                routing_weights.sum(dim=-1, keepdim=True), min=self.norm_min
            )
            routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # Add (optional) shared experts to the result
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Ernie4_5_MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = Ernie4_5_MoeAttention(config, layer_idx)

        if (
            ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= config.moe_layer_start_index
            and layer_idx <= config.moe_layer_end_index
        ):
            self.mlp = Ernie4_5_MoeSparseMoeBlock(config)
        else:
            self.mlp = Ernie4_5_MoeMLP(config)

        self.input_layernorm = Ernie4_5_MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Ernie4_5_MoeRMSNorm(config.hidden_size, config.rms_norm_eps)


@auto_docstring
class Ernie4_5_MoePreTrainedModel(MixtralPreTrainedModel):
    config: Ernie4_5_MoeConfig
    _no_split_modules = ["Ernie4_5_MoeDecoderLayer"]
    _keep_in_fp32_modules_strict = ["gate", "moe_statics"]
    # Not supporting multi-token prediction (MTP) atm
    _keys_to_ignore_on_load_unexpected = ["mtp"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Ernie4_5_MoeSparseMoeBlock, index=1),
        "hidden_states": Ernie4_5_MoeDecoderLayer,
        "attentions": Ernie4_5_MoeAttention,
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Ernie4_5_MoeStatics):
            module.e_score_correction_bias.data.zero_()


@auto_docstring
class Ernie4_5_MoeModel(Ernie4_5_MoePreTrainedModel):
    def __init__(self, config: Ernie4_5_MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Ernie4_5_MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Ernie4_5_MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Ernie4_5_MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class Ernie4_5_MoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Ernie4_5_MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.use_bias)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.moe_num_experts
        self.num_experts_per_tok = config.moe_k

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        super().forward(**super_kwargs)


__all__ = [
    "Ernie4_5_MoeForCausalLM",
    "Ernie4_5_MoeModel",
    "Ernie4_5_MoePreTrainedModel",
]
