import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from tqdm import tqdm

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GenericForSequenceClassification
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils.deprecation import deprecate_kwarg
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)
from .configuration_deepseek_v3 import DeepseekV3Config


class DeepseekV3RMSNorm(LlamaRMSNorm):
    pass


class DeepseekV3RotaryEmbedding(LlamaRotaryEmbedding):
    pass


def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""
    TODO let's just use the original freqcis computation to not have the view
    transpose + reshape! This is not optimized!
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class GroupedLinear(nn.Module):
    def __init__(self, num_groups: int, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_groups, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        raise NotImplementedError("`GroupedLinear` is a weight container for use with specialized kernels.")

    def extra_repr(self) -> str:
        return f"num_groups={self.num_groups}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class GroupedDeepseekV3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_experts = config.n_routed_experts
        hidden_size = config.hidden_size
        intermediate_size = config.moe_intermediate_size

        # Use the new GroupedLinear layer for a cleaner, more consistent structure
        self.gate_proj = GroupedLinear(num_experts, hidden_size, intermediate_size, bias=False)
        self.up_proj = GroupedLinear(num_experts, hidden_size, intermediate_size, bias=False)
        self.down_proj = GroupedLinear(num_experts, intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        raise NotImplementedError("The MoE computation is performed in the parent `DeepseekV3MoE` module.")


class DeepseekV3MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_grouped_gemm = config.use_grouped_gemm
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        if self.use_grouped_gemm:
            self.experts = GroupedDeepseekV3MLP(config)
        else:
            self.experts = nn.ModuleList(
                [
                    DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                    for _ in range(config.n_routed_experts)
                ]
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_grouped_gemm:
            return self.grouped_forward(hidden_states)
        else:
            return self.vanilla_forward(hidden_states)

    def grouped_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import grouped_gemm.ops as ops

        residuals = hidden_states
        orig_shape = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        topk_indices, topk_weights = self.gate(hidden_states)
        topk_indices = topk_indices.to(torch.int32)
        batch_sizes = torch.bincount(topk_indices.flatten().cpu(), minlength=self.config.n_routed_experts)
        permuted_hidden_states, row_id_map = ops.permute(flat_hidden_states, topk_indices)

        gate_out = ops.gmm(permuted_hidden_states, self.experts.gate_proj.weight, batch_sizes, trans_b=True)
        up_out = ops.gmm(permuted_hidden_states, self.experts.up_proj.weight, batch_sizes, trans_b=True)
        intermediate_out = self.experts.act_fn(gate_out) * up_out
        expert_out = ops.gmm(intermediate_out, self.experts.down_proj.weight, batch_sizes, trans_b=True)

        moe_output = ops.unpermute(expert_out, row_id_map, topk_weights)
        return moe_output.view(*orig_shape) + self.shared_experts(residuals)

    def vanilla_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        topk_indices, topk_weights = self.gate(hidden_states)
        final_hidden_states = torch.zeros_like(flat_hidden_states)
        expert_mask = F.one_hot(topk_indices, num_classes=len(self.experts)).permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            token_indices, weight_indices = torch.where(expert_mask[expert_idx])
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices].unsqueeze(1)
                expert_output = expert(flat_hidden_states[token_indices])
                final_hidden_states.index_add_(
                    0, token_indices, (expert_output * expert_weights).to(final_hidden_states.dtype)
                )

        return final_hidden_states.view(*orig_shape) + self.shared_experts(residuals)


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
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
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

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV3PreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DeepseekV3TopkRouter):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


class DeepseekV3Model(LlamaModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]

    def _fuse_experts(self):
        for layer in tqdm(self.layers, desc="Fusing experts"):
            if isinstance(layer.mlp, DeepseekV3MoE) and not layer.mlp.use_grouped_gemm:
                grouped_experts = GroupedDeepseekV3MLP(layer.mlp.config)

                gate_weights = torch.stack([expert.gate_proj.weight for expert in layer.mlp.experts])
                up_weights = torch.stack([expert.up_proj.weight for expert in layer.mlp.experts])
                down_weights = torch.stack([expert.down_proj.weight for expert in layer.mlp.experts])

                grouped_experts.gate_proj.weight.data = gate_weights
                grouped_experts.up_proj.weight.data = up_weights
                grouped_experts.down_proj.weight.data = down_weights

                layer.mlp.experts = grouped_experts
                layer.mlp.use_grouped_gemm = True

    def _unfuse_experts(self):
        for layer in tqdm(self.layers, desc="Unfusing experts"):
            if isinstance(layer.mlp, DeepseekV3MoE) and layer.mlp.use_grouped_gemm:
                grouped_experts = layer.mlp.experts
                gate_weights = grouped_experts.gate_proj.weight.data
                up_weights = grouped_experts.up_proj.weight.data
                down_weights = grouped_experts.down_proj.weight.data

                config = layer.mlp.config
                num_experts = config.n_routed_experts
                experts = nn.ModuleList(
                    [DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(num_experts)]
                )

                for i in range(num_experts):
                    experts[i].gate_proj.weight.data = gate_weights[i]
                    experts[i].up_proj.weight.data = up_weights[i]
                    experts[i].down_proj.weight.data = down_weights[i]

                layer.mlp.experts = experts
                layer.mlp.use_grouped_gemm = False


class DeepseekV3ForCausalLM(LlamaForCausalLM):
    def fuse_experts(self):
        import importlib.util

        if importlib.util.find_spec("grouped_gemm") is None:
            raise ImportError(
                "Please install grouped_gemm to use use_grouped_gemm=True. "
                "You can install it with `pip install git+https://github.com/fanshiqing/grouped_gemm@main`"
            )
        self.model._fuse_experts()

    def unfuse_experts(self):
        self.model._unfuse_experts()


class DeepseekV3ForSequenceClassification(GenericForSequenceClassification, DeepseekV3PreTrainedModel):
    pass


__all__ = [
    "DeepseekV3PreTrainedModel",
    "DeepseekV3Model",
    "DeepseekV3ForCausalLM",
    "DeepseekV3ForSequenceClassification",
]
