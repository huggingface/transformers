import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel
from .configuration_deepseek_v3 import DeepseekV3Config


logger = logging.get_logger(__name__)


class DeepseekV3RMSNorm(LlamaRMSNorm):
    pass


class DeepseekV3RotaryEmbedding(LlamaRotaryEmbedding):
    pass


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))

    def forward(self, hidden_states):
        batch_size, seq_length = hidden_states.shape[:-1]
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = router_logits.sigmoid()
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(batch_size * seq_length, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )  # [n, e]
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        _, topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weights = scores.gather(1, topk_indices)
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor  # must multiply the scaling factor
        return topk_indices, topk_weights, router_logits


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(config=config, intermediate_size=config.moe_intermediate_size)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe_infer(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states, router_logits

    def moe_infer(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        """
        Perform inference using a Mixture of Experts (MoE) model.
        Args:
            hidden_states (torch.Tensor): Input hidden states.
            topk_indices (torch.Tensor): Indices of the top-k experts for each token.
            topk_weights (torch.Tensor): Weights associated with the top-k experts.

        Returns:
            torch.Tensor: Output of the MoE model.
        """
        num_experts = len(self.experts)
        batch_size, num_topk = topk_indices.shape
        with torch.no_grad():
            # Count the number of tokens assigned to each expert
            expert_counts = topk_indices.new_zeros((batch_size, num_experts))
            expert_counts.scatter_(1, topk_indices, 1)
            tokens_per_expert = expert_counts.sum(dim=0)

            # Sort tokens by their assigned expert
            sorted_indices = topk_indices.view(-1).argsort()
            sorted_tokens = hidden_states[sorted_indices // num_topk]
            tokens_per_expert = tokens_per_expert.cpu().numpy()

            # Process tokens through their assigned experts
            expert_outputs = []
            current_pos = 0

        for expert_idx, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue

            next_pos = current_pos + num_tokens
            expert = self.experts[expert_idx]
            expert_tokens = sorted_tokens[current_pos:next_pos]
            expert_outputs.append(expert(expert_tokens))
            current_pos = next_pos

        # Combine the outputs from all experts
        expert_outputs = torch.cat(expert_outputs, dim=0) if expert_outputs else sorted_tokens.new_empty(0)

        # Reorder the outputs to match the original token sequence
        reordered_outputs = torch.empty_like(expert_outputs)
        reordered_outputs[sorted_indices] = expert_outputs

        # Reshape and apply the expert weights
        reordered_outputs = reordered_outputs.view(batch_size, num_topk, -1).type(topk_weights.dtype)
        moe_output = torch.matmul(topk_weights.unsqueeze(1), reordered_outputs)
        moe_output = moe_output.sum(dim=1).type(hidden_states.dtype)
        return moe_output


class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.q_head_dim

        self.is_causal = True
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

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
        hidden_shape = (*input_shape, self.num_heads, -1)
        batch_size, seq_length = input_shape

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states))).view(hidden_shape).transpose(1, 2)
        q_rot, q_pass = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(hidden_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(*input_shape, 1, self.qk_rope_head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        query_states = torch.cat(q_rot, q_pass, dim=-1)
        key_states = torch.cat(k_rot, k_pass, dim=-1)

        if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

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

        if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = (torch.zeros((1,), device=hidden_states.device, dtype=torch.int64),)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class DeepseekV3PreTrainedModel(LlamaPreTrainedModel):
    pass


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


class DeepseekV3Model(MixtralModel):
    def __init__(self, config):
        super().__init__(config)
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        """
        Weights have to be permutted for correct rope formulation. We can't do this in the weights
        as every other framework already uses the `Llama` orginal function (which is copyrighted btw).
        And I am not even sure it's better.... anyways end of my rant
        """
        for k in state_dict:
            if "q_b_proj." in k:
                weight = state_dict.pop(k[: self.qk_nope_head_dim])
            if "k_b_proj." in k:
                weight = state_dict.pop(k[self.qk_nope_head_dim :])
            state_dict[k] = permute_for_rope(weight, weight.shape[0], weight.shape[1], weight.shape[2])


class DeepseekV3ForCausalLM(MixtralForCausalLM):
    pass


class DeepseekV3ForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = [
    "DeepseekV3PreTrainedModel",
    "DeepseekV3Model",
    "DeepseekV3ForCausalLM",
    "DeepseekV3ForSequenceClassification",
]
