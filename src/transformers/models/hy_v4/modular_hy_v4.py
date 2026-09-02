# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HYV4 model."""

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4UnweightedRMSNorm,
)
from ..deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer
from ..glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMLP,
    Glm4MoeLiteModel,
    Glm4MoeLiteMoE,
    Glm4MoeLitePreTrainedModel,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteTopkRouter,
    apply_rotary_pos_emb,
)
from ..glm5_next.modeling_glm5_next import Glm5NextTextExperts
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding
from ..gpt_oss.modeling_gpt_oss import eager_attention_forward


@auto_docstring(checkpoint="tencent/Hy4-preview")
@strict
class HYV4Config(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for routing. HYV4 selects experts globally, so this is 1 (one group holding every expert).
    topk_group (`int`, *optional*, defaults to 1):
        Number of expert groups kept during routing. With `n_group=1` this reuses `Glm4MoeLiteTopkRouter` as a plain global top-k.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type, either `"dense"` or `"sparse"`. Defaults to one dense layer followed by
        sparse MoE layers.
    index_topk (`int`, *optional*, defaults to 2048):
        Maximum number of key positions selected by each DSA query.
    index_head_dim (`int`, *optional*, defaults to 128):
        Hidden dimension of each DSA indexer head.
    index_n_heads (`int`, *optional*, defaults to 16):
        Number of DSA indexer heads.
    indexer_types (`list[str]`, *optional*):
        Per-layer DSA indexer type, either `"full"` or `"shared"`. A shared layer reuses the
        most recent full indexer in the same forward request.
    hc_mult (`int`, *optional*, defaults to 4):
        Number of hidden-state channels maintained by iHC.
    hc_magnitude (`float`, *optional*, defaults to 2.0):
        Scale applied to the iHC post-gating branch.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical epsilon added to iHC sigmoid gates.
    learnable_sink (`bool`, *optional*, defaults to `True`):
        Whether to add a learned per-head attention sink.
    learnable_sink_init (`float`, *optional*, defaults to 0.0):
        Initial value of each learned attention-sink logit.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Magnitude of the routed-expert SwiGLU clamp. Values at or below zero disable the clamp.
    """

    model_type = "hy_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.gate_proj": "colwise",
        "layers.*.self_attn.sinks": "colwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    vocab_size: int = 120832
    hidden_size: int = 2816
    intermediate_size: int = 6912
    moe_intermediate_size: int = 768
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.006
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 120002
    bos_token_id: int | None = 120000
    eos_token_id: int | list[int] | None = 120025
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    indexer_types: list[str] | None = None
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    learnable_sink: bool = True
    learnable_sink_init: float = 0.0
    swiglu_limit: float = 10.0
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        # MLA expands the latent to one key/value per query head, so keys are never grouped.
        self.num_key_value_heads = self.num_attention_heads

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so `head_dim` points at it.
        self.head_dim = self.qk_rope_head_dim

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, self.num_hidden_layers) + ["sparse"] * max(
                self.num_hidden_layers - 1, 0
            )
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        if self.indexer_types is None:
            self.indexer_types = [
                "full" if layer_idx == 0 or (layer_idx - 1) % 4 == 0 else "shared"
                for layer_idx in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


class HYV4RMSNorm(Glm4MoeLiteRMSNorm):
    pass


class HYV4UnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We compute the inverse (i.e. we do not apply `x * ...`)
        return torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class HYV4RotaryEmbedding(GlmMoeDsaRotaryEmbedding):
    pass


class HYV4Indexer(DeepseekV32Indexer):
    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,  # Kept for BC
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        # Flipped RoPE position (later portion instead of the first)
        q_pass, q_rot = torch.split(q, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Norm is kept in fp32
        k = (
            self.k_norm(self.wk(hidden_states).to(self.k_norm.weight.dtype)).to(hidden_states.dtype).unsqueeze(2)
        )  # [B, S, 1, D]
        k_pass, k_rot = torch.split(k, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pass, q_rot], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_pass, k_rot], dim=-1).squeeze(2)  # [B, S, D]

        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
        # Apply softmax scale later
        weights = (
            self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
            * (self.n_heads**-0.5)
            * self.softmax_scale
        )
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
        if attention_mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
        else:
            index_scores = index_scores + attention_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]


class HYV4Attention(GlmMoeDsaAttention):
    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.gate_projection_size = self.v_head_dim
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.gate_projection_size, bias=False)
        self.sinks = nn.Parameter(torch.full((self.num_heads,), config.learnable_sink_init))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        # Key difference is the gating mechanism
        gate_states = self.gate_proj(hidden_states).view(batch_size, seq_length, -1, self.gate_projection_size)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # Both latents are viewed as single-head, 4D tensors, as expected by `expand_kv`
        k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        # Non-interleave RoPE
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        query_states = torch.cat((q_pass, q_rot), dim=-1)

        key_states, value_states = self.expand_kv(k_pass, k_rot)

        # Sparse-attention models cache the expanded K/V, not the compressed latents. TODO (remi-or): fix this with topk
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # DSA: select this layer's top-k tokens, or reuse the previous full layer's on `"shared"` layers.
        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask[:, 0, :, :],
                position_ids,  # Kept for BC
                past_key_values=past_key_values,
            )  # [B, S, topk]
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        sparse_indices = None
        if self.config._attn_implementation in ("eager", "sdpa"):
            index_mask = (
                topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                .scatter(-1, topk_indices.long(), False)
                .unsqueeze(1)
            )

            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask & ~index_mask
            else:
                attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
        else:
            sparse_indices = topk_indices

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
            indices=sparse_indices,  # consumed by flash_mla_with_kvcache; ignored by eager / SDPA
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = attn_output * torch.sigmoid(gate_states)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        return self.o_proj(attn_output), attn_weights, topk_indices


class HYV4MLP(Glm4MoeLiteMLP):
    pass


class HYV4TopkRouter(Glm4MoeLiteTopkRouter):
    pass


class HYV4Experts(Glm5NextTextExperts):
    pass


class HYV4MoE(Glm4MoeLiteMoE):
    pass


class HYV4HyperConnection(DeepseekV4HyperConnection):
    """
    Independent Hyper-Connection following the DeepSeek-V4 form (parameters on the module).

    The overall difference lies in the dropping of the `comb` output which skips the sinkhorn part of the algorithm.
    """

    def __init__(self, config: HYV4Config):
        super().__init__()
        del self.hc_sinkhorn_iters
        mix = 2 * self.hc_mult  # noqa: F841
        self.hc_post_magnitude = config.hc_magnitude
        self.scale = nn.Parameter(torch.empty(2))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent HC implementation with forced fp32 application"""
        # Key difference is to force fp32 in any case
        device_type = hidden_streams.device.type if hidden_streams.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            flat = hidden_streams.flatten(2).float()
            # Norm as residual
            mixes = F.linear(flat, self.fn.float()) * self.input_norm(flat)

            # Can be seen as non-combined variation of the DSv4 implementation
            pre_b, post_b = self.base.split(self.hc_mult, dim=-1)
            pre_scale, post_scale = self.scale.unbind(0)
            pre_logits, post_logits = mixes.split(self.hc_mult, dim=-1)

            pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + self.hc_eps
            # Magnitude can be set within the config (difference to the constant to 2 in dsv4)
            post = self.hc_post_magnitude * torch.sigmoid(post_logits * post_scale + post_b) + self.hc_eps
            out = torch.sum(pre.unsqueeze(-1) * hidden_streams, dim=2)

        return post, out.to(hidden_streams.dtype)


class HYV4HyperHead(DeepseekV4HyperHead):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Key difference is to force fp32 in any case
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            flat = x.flatten(2).float()
            # Norm as residual
            mixes = F.linear(flat, self.hc_fn.float()) * self.input_norm(flat)
            pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
            out = (pre.unsqueeze(-1) * x).sum(dim=2)
        return out.to(x.dtype)


class HYV4DecoderLayer(DeepseekV4DecoderLayer):
    """Similar to DSv4 but with a different Hyper Connection and mixed MLP/MoE patterns"""

    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__(self, config, layer_idx)
        self.mlp = HYV4MoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else HYV4MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.LongTensor | None]:
        dtype = hidden_states.dtype

        # Key difference is the hyper connection and its residual connection
        residual = hidden_states
        post, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
        hidden_states = (post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2) + residual.float()).to(dtype)

        residual = hidden_states
        post, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = (post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2) + residual.float()).to(dtype)

        return hidden_states, topk_indices


class HYV4PreTrainedModel(Glm4MoeLitePreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp_layers\..*"]
    # Combination of sinks and DSA disable anything but eager atm
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "fn",
        "scale",
        "base",
        "hc_fn",
        "hc_scale",
        "hc_base",
        "weights_proj",
        "k_norm",
        "sinks",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, HYV4TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, HYV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, HYV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=self.config.initializer_range)
            init.constant_(module.scale, 0.01)

            base_value = -math.log(max(module.hc_mult - 1, 1))
            base = torch.zeros_like(module.base)
            base[: module.hc_mult] = base_value
            init.copy_(module.base, base)
        elif isinstance(module, HYV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=self.config.initializer_range)
            init.constant_(module.hc_scale, 0.01)

            base_value = -math.log(max(module.hc_mult - 1, 1))
            init.constant_(module.hc_base, base_value)
        elif isinstance(module, HYV4Attention):
            init.constant_(module.sinks, self.config.learnable_sink_init)


class HYV4Model(Glm4MoeLiteModel):
    def __init__(self, config: HYV4Config):
        super().__init__(config)
        self.hc_head = HYV4HyperHead(config)

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

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "allow_is_causal_skip": False,  # Always force creation to account for causality in the indexer
            }
            causal_mask_mapping = {"deepseek_sparse_attention": create_causal_mask(**mask_kwargs)}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        # Prepare HC connections
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        topk_indices = None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping["deepseek_sparse_attention"],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        # Difference with the HC head at the end
        hidden_states = self.norm(self.hc_head(hidden_states))

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class HYV4ForCausalLM(Glm4MoeLiteForCausalLM):
    # Same as base but with the additional lm head
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "fn",
        "scale",
        "base",
        "hc_fn",
        "hc_scale",
        "hc_base",
        "weights_proj",
        "k_norm",
        "sinks",
        "lm_head",
    ]

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
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # Key difference of the lm_head being kept in float
        logits = self.lm_head(hidden_states[:, slice_indices, :].to(dtype=self.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["HYV4Config", "HYV4PreTrainedModel", "HYV4Model", "HYV4ForCausalLM"]
