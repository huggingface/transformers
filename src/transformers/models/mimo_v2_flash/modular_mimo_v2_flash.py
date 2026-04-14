# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_experts_implementation, use_kernelized_func
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..glm4_moe.modeling_glm4_moe import apply_rotary_pos_emb  # noqa: F401
from ..llama.modeling_llama import LlamaDecoderLayer, eager_attention_forward, repeat_kv
from ..mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2-Flash")
@strict
class MiMoV2FlashConfig(PreTrainedConfig):
    r"""
    head_dim (`int`, *optional*, defaults to 192):
        Dimension of query and key heads.
    v_head_dim (`int`, *optional*, defaults to 128):
        Dimension of value heads (special case because MiMo uses a smaller v head dim than (qk) head dim )
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for group-based top-k routing.
    topk_group (`int`, *optional*, defaults to 1):
        Number of groups selected per token in group-based top-k routing.
    moe_layer_freq (`list`, *optional*):
        Per-layer binary flag indicating MoE (1) vs dense MLP (0).
    add_swa_attention_sink_bias (`bool`, *optional*, defaults to `True`):
        Whether to add attention sink bias to sliding window attention layers.
    add_full_attention_sink_bias (`bool`, *optional*, defaults to `False`):
        Whether to add attention sink bias to full attention layers.
    """

    model_type = "mimo_v2_flash"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = {"full_attention": 5_000_000.0, "sliding_attention": 10_000.0}
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 152576
    hidden_size: int = 4096
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 128
    layer_types: list[str] | None = None
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float | None = 1.0
    router_jitter_noise: float = 0.0
    moe_layer_freq: list | None = None
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        # Full attention: first layer and every 6th layer; rest are SWA
        hybrid_layer_pattern = kwargs.pop("hybrid_layer_pattern", None)
        if self.layer_types is None:
            if hybrid_layer_pattern is not None:
                self.layer_types = ["sliding_attention" if p == 1 else "full_attention" for p in hybrid_layer_pattern]
            else:
                self.layer_types = [
                    "full_attention" if (i == 0 or not ((i + 1) % 6)) else "sliding_attention"
                    for i in range(self.num_hidden_layers)
                ]

        # BC: hub-only fields not modeled in the config or redundant that can be derived.
        for _hub_only in (
            "scoring_func",
            "topk_method",
            "attention_value_scale",
            "attention_chunk_size",
            "sliding_window_size",
            "n_shared_experts",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "swa_qk_head_dim",
            "swa_v_head_dim",
            "swa_head_dim",
        ):
            kwargs.pop(_hub_only, None)

        if self.routed_scaling_factor is None:
            self.routed_scaling_factor = 1.0

        if self.moe_layer_freq is None:
            self.moe_layer_freq = [0] * self.num_hidden_layers

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.334)

        # Similar to Gemma3:
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`. If we find `rope_parameters`
        # as arg in the inputs, we can safely assume that it is in the new format. New naming used -> new format
        default_rope_params = {
            "full_attention": {"rope_type": "default"},
            "sliding_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters or default_rope_params

        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)

        for attn_type, theta_key in (("full_attention", "rope_theta"), ("sliding_attention", "swa_rope_theta")):
            if self.rope_parameters.get(attn_type) is None:
                self.rope_parameters[attn_type] = {"rope_type": "default"}
            self.rope_parameters[attn_type].setdefault(
                "rope_theta", kwargs.pop(theta_key, self.default_theta[attn_type])
            )
            self.rope_parameters[attn_type].setdefault("partial_rotary_factor", partial_rotary_factor)

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.standardize_rope_params()
        return kwargs

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.sinks": "colwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }


class MiMoV2FlashRMSNorm(MixtralRMSNorm):
    pass


class MiMoV2FlashRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: MiMoV2FlashConfig, device=None):
        super().__init__(config, device=device)

    @staticmethod
    def compute_default_rope_parameters(
        config: MiMoV2FlashConfig | None = None,
        device=None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ):
        rope_params = config.rope_parameters[layer_type]
        base = rope_params["rope_theta"]
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.334)
        dim = int(config.head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


# NOTE @casinca: Concerning this TopKRouter:
# This is the "fixed" TopKRouter from the remote DSV3 implementation with correct masked_fill=-inf and not 0.0 like the
# old native transformers DSV3 implementation currently in the repo. see the update:
# https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/commit/e9b33add76883f293d6bf61f6bd89b497e80e335#d2h-632685
#
# OR HF fixes the DSV3 masking and then I can try to inherit from DSV3, so that I don't override the forward.
#
# On top of that, I made this class as it is, so that it's a direct drop-in replacement for the MixtralSparseMoeBlock:
# I can just do inheritance from MixtralSparseMoeBlock and override the self.gate = MiMoV2FlashTopKRouter(config)
#
# tldr: this MiMo class is refactored (with DSV3 fix + internals) to be compatible with the MixtralSparseMoeBlock,
# following newer style like minimax M2 in the repo, for fused experts etc... I think this is what Vasqu prefers
class MiMoV2FlashTopKRouter(nn.Module):
    """MiMo gating with sigmoid scoring and group-based top-k selection."""

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )

    def forward(self, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]

        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = scores_for_choice.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return logits, topk_weight, topk_idx


@use_experts_implementation
class MiMoV2FlashExperts(MixtralExperts):
    """Fused experts (V5). Checkpoint layout: `mlp.experts.gate_up_proj`, `mlp.experts.down_proj`.
    Original reference used per-expert `nn.ModuleList` MLPs (`experts.{i}.gate_proj` etc.)
    """

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        # MoE experts use moe_intermediate_size, not the dense MLP's intermediate_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))


class MiMoV2FlashSparseMoeBlock(MixtralSparseMoeBlock):
    pass


class MiMoV2FlashMLP(Qwen2MoeMLP):
    pass


# There is no prior occurrence of this function in the repo, except for gpt-oss.
# But for enabling dual eager attention, with a sink path and a non-sink path, we'd need to import both functions from
# gpt-oss and llama with aliases (they have the same name).
# However this cause problems with `modular_model_converter.py` and HuggingFace prefers not to refactor other models,
# see: https://github.com/huggingface/transformers/issues/45141
# So I'm creating this function as a direct copy of the gpt-oss eager attention forward (with sinks).
def eager_attention_forward_with_sink(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    """Eager attention with attention sinks (copy from gpt-oss `eager_attention_forward`)."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# NOTE: @casinca I have to do this in the case users enable backends. The upcoming attention mask (which depends on the
# backend used) needs to be converted for the sink path which always expects a float mask (gpt-oss eager attn).
def _prepare_sink_eager_attention_mask(
    attention_mask: torch.Tensor | None,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    config: MiMoV2FlashConfig,
    layer_idx: int,
) -> torch.Tensor:
    """
    This is a helper needed to build or normalize the upcoming attention mask for the gpt-oss sink eager attention path.

    SDPA/FlashAttention can pass None, flex attention can pass a boolean mask.
    But Sink eager attention always needs an explicit additive float mask (0 for allowed, finfo(dtype).min for masked).
    """
    if attention_mask is None:
        seq_len, key_len = query_states.shape[2], key_states.shape[2]
        dtype, device = query_states.dtype, query_states.device
        min_val = torch.finfo(dtype).min
        row_pos = torch.arange(seq_len, device=device).unsqueeze(1) + (key_len - seq_len)
        col_pos = torch.arange(key_len, device=device).unsqueeze(0)
        mask = col_pos > row_pos  # causal
        if config.layer_types[layer_idx] == "sliding_attention":
            mask = mask | (row_pos - col_pos >= config.sliding_window)
        return torch.where(mask, min_val, 0.0).to(dtype)[None, None]
    if attention_mask.dtype == torch.bool:
        min_dtype = torch.finfo(query_states.dtype).min
        return torch.where(attention_mask, 0.0, min_dtype).to(query_states.dtype)
    return attention_mask


@use_kernelized_func(apply_rotary_pos_emb)
class MiMoV2FlashAttention(nn.Module):
    def __init__(self, config: MiMoV2FlashConfig, layer_idx: int):
        super().__init__()
        is_swa = config.layer_types[layer_idx] == "sliding_attention"

        # SWA layers double the kv heads vs full-attention layers
        num_kv_heads = config.num_key_value_heads * 2 if is_swa else config.num_key_value_heads
        num_attn_heads = config.num_attention_heads
        head_dim = config.head_dim
        v_head_dim = config.v_head_dim

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.num_key_value_groups = num_attn_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, num_attn_heads * head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * v_head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(num_attn_heads * v_head_dim, config.hidden_size, bias=False)

        # Dispatch attention: sink layers use GPT-OSS always-sink eager, non-sink layers use standard Llama eager
        # (which is compatible with SDPA/FA2/flex backends)
        add_sink = config.add_swa_attention_sink_bias if is_swa else config.add_full_attention_sink_bias
        if add_sink:
            self.sinks = nn.Parameter(torch.empty(num_attn_heads), requires_grad=False)
            self._eager_attention_forward = eager_attention_forward_with_sink
        else:
            self.sinks = None
            self._eager_attention_forward = eager_attention_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        qk_hidden_shape = (*input_shape, -1, self.head_dim)
        v_hidden_shape = (*input_shape, -1, self.v_head_dim)

        query_states = self.q_proj(hidden_states).view(qk_hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(qk_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(v_hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # NOTE @casinca: this is a remnant for trying GLM inheritance and reusing their apply_rotary_pos_emb to avoid
        # duplicating code. but could just recreate a another apply_rotary_pos_emb here too.
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Sink layers always use their custom eager attention (incompatible with SDPA/FA2/flex).
        # Non-sink layers can dispatch to any configured backend.
        if self.sinks is not None:
            attention_interface = self._eager_attention_forward
            attention_mask = _prepare_sink_eager_attention_mask(
                attention_mask, query_states, key_states, self.config, self.layer_idx
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, self._eager_attention_forward
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


class MiMoV2FlashDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MiMoV2FlashConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # TODO @casinca: for now attn_type is dead code but might need for new attn rework
        self.attention_type = config.layer_types[layer_idx]
        # Replace the dense MLP with an MoE block on MoE layers (per `moe_layer_freq`).
        if config.moe_layer_freq[layer_idx]:
            self.mlp = MiMoV2FlashSparseMoeBlock(config)


@auto_docstring
class MiMoV2FlashPreTrainedModel(MixtralPreTrainedModel):
    config_class = MiMoV2FlashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiMoV2FlashDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # non-sink layers can use backends (SDPA/FA2/flex_attention compatible)
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"^model\.mtp\."]
    _can_record_outputs = {
        "hidden_states": MiMoV2FlashDecoderLayer,
        "attentions": MiMoV2FlashAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, MiMoV2FlashExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, MiMoV2FlashAttention) and module.sinks is not None:
            init.normal_(module.sinks, mean=0.0, std=std)
        elif isinstance(module, MiMoV2FlashTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, MiMoV2FlashRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class MiMoV2FlashModel(MixtralModel):
    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        self.has_sliding_layers = "sliding_attention" in self.rotary_emb.layer_types

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
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

        # Build causal mask mapping: full attention + optional sliding window
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings_mapping = {
            lt: self.rotary_emb(hidden_states, position_ids, layer_type=lt) for lt in self.rotary_emb.layer_types
        }

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[layer_idx]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=position_embeddings_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MiMoV2FlashForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "MiMoV2FlashConfig",
    "MiMoV2FlashForCausalLM",
    "MiMoV2FlashModel",
    "MiMoV2FlashPreTrainedModel",
]
