# Copyright 2026 Sarvam AI team. All rights reserved.
#
# This code is based on Llama and Deepseek MoE implementations
# in this library. It has been modified from its original forms to
# accommodate Sarvam's MLA (multi-latent attention) MoE architecture.
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
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepSeekV3Attention,
    DeepSeekV3ForCausalLM,
    DeepSeekV3MLP,
    DeepSeekV3Model,
    DeepSeekV3PreTrainedModel,
    MoEGate,
    SarvamMLAMoE,
    SarvamMLAMLP,
    SarvamMLARMSNorm,
)


logger = logging.get_logger(__name__)


class SarvamMLAConfig(PreTrainedConfig):
    model_type = "sarvam_mla"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        kv_lora_rank=128,
        v_head_dim=128,
        qk_nope_head_dim=128,
        q_head_dim=256,
        rope_scaling=None,
        num_experts=16,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        moe_intermediate_size=11008,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        num_shared_experts=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = q_head_dim
        self.rope_scaling = rope_scaling
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.num_shared_experts = num_shared_experts
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class SarvamMLARotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


class SarvamMLAYarnRotaryEmbedding(SarvamMLARotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=40.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=1.0,
    ):
        self.scaling_factor = float(scaling_factor)
        self.original_max_position_embeddings = int(original_max_position_embeddings)
        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)
        self.mscale = float(mscale)
        self.mscale_all_dim = float(mscale_all_dim)
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )

        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SarvamMLAAttention(DeepSeekV3Attention):
    def __init__(self, config: SarvamMLAConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self._init_rope()

    def _init_rope(self):
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if rope_scaling is None or rope_scaling.get("type", None) in (None, "default"):
            self.rotary_emb = SarvamMLARotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            return

        rope_type = rope_scaling.get("type")
        if rope_type == "deepseek_yarn":
            self.rotary_emb = SarvamMLAYarnRotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
                scaling_factor=rope_scaling.get("factor", 40.0),
                original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 4096),
                beta_fast=rope_scaling.get("beta_fast", 32),
                beta_slow=rope_scaling.get("beta_slow", 1),
                mscale=rope_scaling.get("mscale", 1.0),
                mscale_all_dim=rope_scaling.get("mscale_all_dim", 1.0),
            )
            return
        raise ValueError(f"Unknown rope_scaling type: {rope_type}")


class SarvamMLADecoderLayer(nn.Module):
    def __init__(self, config: SarvamMLAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SarvamMLAAttention(config=config, layer_idx=layer_idx)

        use_moe = (
            hasattr(config, "num_experts")
            and config.num_experts is not None
            and layer_idx >= getattr(config, "first_k_dense_replace", 0)
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        )

        self.mlp = SarvamMLAMoE(config) if use_moe else SarvamMLAMLP(config)
        self.input_layernorm = SarvamMLARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SarvamMLARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class SarvamMLAPreTrainedModel(DeepSeekV3PreTrainedModel):
    config_class = SarvamMLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SarvamMLADecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SarvamMLAModel(DeepSeekV3Model):
    def __init__(self, config: SarvamMLAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SarvamMLADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_flash_attention_2 = False
        self.norm = SarvamMLARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()


class SarvamMLAForCausalLM(DeepSeekV3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = SarvamMLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()