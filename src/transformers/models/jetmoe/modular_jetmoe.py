# coding=utf-8
# Copyright 2024 JetMoe AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch JetMoe model."""

import math
from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import (
    GenericForSequenceClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from .configuration_jetmoe import JetMoeConfig
from ..mixtral.modeling_mixtral import MixtralModel, MixtralPreTrainedModel, load_balancing_loss_func

logger = logging.get_logger(__name__)

class JetMoeParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """
        Initialize the JetMoeParallelExperts module.
        The experts weights are stored in [num_experts, output_size, input_size] format. Such that it's compatible with
        many MoE libraries, such as [Megablock](https://github.com/databricks/megablocks) and
        [ScatterMoE](https://github.com/shawntan/scattermoe), as well as the
        [MoE kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
        used in vllm.

        Args:
            num_experts (int):
                Number of experts.
            input_size (int):
                Size of the input.
            output_size (int):
                Size of the output.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs, expert_size):
        """
        Forward pass of the JetMoeParallelExperts module.

        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], self.weight[i]))
        results = torch.cat(output_list, dim=0)
        return results


class JetMoeTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (`int`):
                Size of the input.
            num_experts (`int`):
                Number of experts.
            top_k (`int`):
                Number of top experts to select.
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, hidden_states):
        # compute the top_k routing decision
        logits = self.layer(hidden_states).float()  # [batch_size x seq_len, num_experts]
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)  # [num_tokens, top_k]
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)  # [num_tokens, top_k]

        # compute number of input given to each expert
        zeros = torch.zeros(
            [top_k_gates.size(0), self.num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device
        )  # [num_tokens, num_experts]
        gates = zeros.scatter(1, top_k_indices, 1)  # [num_tokens, num_experts]
        expert_size = gates.long().sum(0)  # [num_experts,]
        # (This cause torch.compile to fail with `torch._dynamo.exc.Unsupported: Backend compiler failed with a fake tensor exception at`)
        # (and `DataDependentOutputException`)
        expert_size = expert_size.tolist()

        # sort and group input tokens according to expert assignment
        top_k_experts = top_k_indices.flatten()  # [num_tokens * top_k]
        _, index_sorted_experts = top_k_experts.sort(0)  # [num_tokens * top_k]
        batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")  # [num_tokens * top_k]

        # gather the gate values for grouped input tokens
        top_k_gates = top_k_gates.flatten()  # [num_tokens * top_k]
        batch_gates = top_k_gates[index_sorted_experts]  # [num_tokens * top_k]

        return index_sorted_experts, batch_index, batch_gates, expert_size, logits


class JetMoeMoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """

    def __init__(self, config: JetMoeConfig):
        super().__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.intermediate_size
        self.activation = ACT2FN[config.activation_function]
        self.bias = torch.nn.Parameter(torch.empty(self.input_size))
        self.input_linear = JetMoeParallelExperts(config.num_local_experts, self.input_size, self.hidden_size * 2)
        self.output_linear = JetMoeParallelExperts(config.num_local_experts, self.hidden_size, self.input_size)

        self.router = JetMoeTopKGating(
            input_size=self.input_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

    def forward(self, layer_input):
        """
        Forward pass of the mixture of experts layer.

        Args:
            layer_input (Tensor):
                Input tensor.

        Returns:
            Tensor:
                Output tensor.
            Tensor:
                Router logits.
        """
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)
        _, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)

        expert_inputs = layer_input[batch_index]
        hidden_states = self.input_linear(expert_inputs, expert_size)
        chunked_hidden_states = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
        expert_outputs = self.output_linear(hidden_states, expert_size)

        expert_outputs = expert_outputs * batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        layer_output = zeros.index_add(0, batch_index, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.input_size)
        layer_output = layer_output + self.bias
        return layer_output, router_logits


class JetMoeMoA(nn.Module):
    """
    A Sparsely gated mixture of attention layer with pairs of query- and output-projections as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """

    def __init__(self, config: JetMoeConfig):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.input_size = config.hidden_size
        self.hidden_size = config.kv_channels * config.num_key_value_heads
        self.top_k = config.num_experts_per_tok
        self.bias = torch.nn.Parameter(torch.empty(self.input_size))

        self.input_linear = JetMoeParallelExperts(self.num_experts, self.input_size, self.hidden_size)
        self.output_linear = JetMoeParallelExperts(self.num_experts, self.hidden_size, self.input_size)

        self.router = JetMoeTopKGating(
            input_size=self.input_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

    def map(self, layer_input):
        """
        Map inputs to attention experts according to routing decision and compute query projection inside each experts.
        """

        # Compute gating topology
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)  # [bsz * length, emb_size]
        index_sorted_experts, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)
        topo_info = (index_sorted_experts, batch_index, batch_gates, expert_size)

        # Group inputs according to topology and compute query projection
        expert_inputs = layer_input[batch_index]  # [bsz * length * top_k, emb_size]
        expert_outputs = self.input_linear(expert_inputs, expert_size)  # [bsz * length * top_k, hidden_size]

        # Ungroup queries back to original order
        zeros = torch.zeros(
            (bsz * length * self.top_k, self.hidden_size), dtype=expert_outputs.dtype, device=expert_outputs.device
        )
        layer_output = zeros.index_add(0, index_sorted_experts, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.top_k, -1)  # [bsz, length, top_k, hidden_size]
        return layer_output, router_logits, topo_info

    def reduce(self, layer_input, topo_info):
        """
        Compute output projection inside each attention experts and merge the outputs of different experts.
        """
        bsz, length, k, hidden_size = layer_input.size()
        layer_input = layer_input.reshape(-1, hidden_size)  # [bsz * length * k, hidden_size]
        index_sorted_experts, batch_index, batch_gates, expert_size = topo_info

        # Group inputs according to topology and compute output projection
        expert_inputs = layer_input[index_sorted_experts]  # [bsz * length * top_k, hidden_size]
        expert_outputs = self.output_linear(expert_inputs, expert_size)  # [bsz * length * top_k, emb_size]

        # Apply gates to attention expert outputs
        expert_outputs = expert_outputs * batch_gates[:, None]

        # Ungroup and merge outputs to original order
        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        layer_output = zeros.index_add(0, batch_index, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.input_size)
        layer_output = layer_output + self.bias
        return layer_output

    def forward(self, layer_input):
        raise NotImplementedError("This module doesn't support call and forward.")




class JetMoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    """

    def __init__(self, config: JetMoeConfig, layer_idx: Optional[int] = None):
        """
        Initialize the JetMoeAttention module.

        Args:
            config:
                Configuration object with model hyperparameters.
            layer_idx:
                Index of the layer in the model.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_causal = True
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.top_k = config.num_experts_per_tok
        self.attention_dropout = config.attention_dropout
        self.kv_projection_size = config.kv_channels * config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.head_dim = config.kv_channels

        self.experts = JetMoeMoA(config)

        self.kv_proj = torch.nn.Linear(config.hidden_size, self.kv_projection_size * 2, bias=False)

        self.rotary_emb = JetMoeRotaryEmbedding(config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states, router_logits, topo_info = self.experts.map(hidden_states)
        key_states, value_states = self.kv_proj(hidden_states).chunk(2, dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads for top-k attention experts
        key_states = key_states.repeat(1, self.top_k, 1, 1)
        value_states = value_states.repeat(1, self.top_k, 1, 1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.top_k, self.kv_projection_size)

        attn_output = self.experts.reduce(attn_output, topo_info)
        attn_output = attn_output.view(bsz, q_len, -1)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, router_logits


class JetMoeBlock(GradientCheckpointingLayer):
    def __init__(self, config: JetMoeConfig, layer_idx: Optional[int] = None):
        """
        Initialize the JetMoeBlock module.

        Args:
            config:
                Configuration object with model hyperparameters.
        """
        super().__init__()
        self.input_layernorm = JetMoeRMSNorm(config.hidden_size)
        self.self_attention = JetMoeAttention(config, layer_idx)
        self.post_attention_layernorm = JetMoeRMSNorm(config.hidden_size)

        self.mlp = JetMoeMoE(config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.Tensor], Optional[tuple[torch.Tensor, tuple[torch.FloatTensor, ...]]]]:
        # Self Attention
        attn_output, _, _ = self.self_attention(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = hidden_states + attn_output
        x_mlp, mlp_router_logits = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + x_mlp
        return hidden_states


@auto_docstring
class JetMoePreTrainedModel(MixtralPreTrainedModel):
    pass


@auto_docstring
class JetMoeModel(MixtralModel):
    def __init__(self, config: JetMoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([JetMoeBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = JetMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class JetMoeForCausalLM(JetMoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = JetMoeModel(config)
        self.vocab_size = config.vocab_size
        self.aux_loss_coef = config.aux_loss_coef
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_word_embeddings = config.tie_word_embeddings

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class JetMoeForSequenceClassification(GenericForSequenceClassification, JetMoePreTrainedModel): ...


__all__ = ["JetMoeForCausalLM", "JetMoeModel", "JetMoePreTrainedModel", "JetMoeForSequenceClassification"]
