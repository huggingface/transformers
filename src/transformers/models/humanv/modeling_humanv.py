# coding=utf-8
# Copyright 2025 The HumanV Team. All rights reserved.
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

from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_humanv import HumanVConfig
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS


logger = logging.get_logger(__name__)


class HumanVRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, config: HumanVConfig, device=None):
        super().__init__()
        self.config = config

        rope_type = "default"
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_type = rope_parameters.get("rope_type", rope_type)

        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or rope_type

        rope_init_fn = ROPE_INIT_FUNCTIONS.get(rope_type)
        if rope_init_fn is None:
            rope_init_fn = ROPE_INIT_FUNCTIONS.get("default") or next(iter(ROPE_INIT_FUNCTIONS.values()))

        inv_freq, self.attention_scaling = rope_init_fn(config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class HumanVMLP(nn.Module):
    def __init__(self, config: HumanVConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=getattr(config, "mlp_bias", False))
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=getattr(config, "mlp_bias", False))
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=getattr(config, "mlp_bias", False))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True

        self.attn_implementation = getattr(config, "attn_implementation", "eager")
        self.use_sliding_window = getattr(config, "use_sliding_window", False)
        self.sliding_window = getattr(config, "sliding_window", None) if self.use_sliding_window else None

        layer_types = getattr(config, "layer_types", None)
        if isinstance(layer_types, list) and layer_idx < len(layer_types):
            self.attention_type = layer_types[layer_idx]
        else:
            self.attention_type = getattr(config, "attention_type", "full_attention")

        self.sparse_block_size = int(getattr(config, "sparse_block_size", 64))
        self.sparse_local_num_blocks = int(getattr(config, "sparse_local_num_blocks", 4))
        self.sparse_global_num_blocks = int(getattr(config, "sparse_global_num_blocks", 2))
        self.sparse_global_stride = int(getattr(config, "sparse_global_stride", 8))

        self.qk_norm = bool(getattr(config, "qk_norm", False))
        self.qk_norm_eps = float(getattr(config, "qk_norm_eps", 1e-6))
        self.scaling = 1.0 if self.qk_norm else (self.head_dim**-0.5)

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
        )

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        qn = nn.functional.normalize(q.to(torch.float32), dim=-1, eps=self.qk_norm_eps).to(q_dtype)
        kn = nn.functional.normalize(k.to(torch.float32), dim=-1, eps=self.qk_norm_eps).to(k_dtype)
        return qn, kn

    def _make_2d_mask_full(self, attention_mask_2d: Optional[torch.Tensor], seq_len: int, device: torch.device) -> torch.Tensor:
        if attention_mask_2d is None:
            return torch.ones((1, seq_len), device=device, dtype=torch.float32)
        if attention_mask_2d.dtype != torch.float32:
            attention_mask_2d = attention_mask_2d.to(torch.float32)
        return attention_mask_2d

    def _pad_left_to_multiple(self, x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
        seq_len = x.size(-2)
        pad = (multiple - (seq_len % multiple)) % multiple
        if pad == 0:
            return x, 0
        return torch.nn.functional.pad(x, (0, 0, pad, 0)), pad

    def _build_global_block_indices(
        self, n_blocks: int, global_num_blocks: int, global_stride: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_num_blocks <= 0:
            return torch.empty((n_blocks, 0), device=device, dtype=torch.long), torch.empty((n_blocks, 0), device=device, dtype=torch.float32)

        g_idx = torch.zeros((n_blocks, global_num_blocks), device=device, dtype=torch.long)
        g_valid = torch.zeros((n_blocks, global_num_blocks), device=device, dtype=torch.float32)

        stride = max(1, int(global_stride))
        for i in range(n_blocks):
            if i <= 0:
                continue
            cand = torch.arange(0, i, device=device, dtype=torch.long)
            if stride > 1:
                cand = cand[::stride]
            if cand.numel() == 0:
                continue
            if cand.numel() > global_num_blocks:
                cand = cand[-global_num_blocks:]
            start = global_num_blocks - cand.numel()
            g_idx[i, start:] = cand
            g_valid[i, start:] = 1.0

        return g_idx, g_valid

    def _apply_sliding_window_scores(self, scores: torch.Tensor, window: int, past_len: int) -> torch.Tensor:
        if window is None or window <= 0:
            return scores
        bsz, n_heads, q_len, k_len = scores.shape
        q_pos = torch.arange(past_len, past_len + q_len, device=scores.device).view(1, 1, q_len, 1)
        k_pos = torch.arange(0, k_len, device=scores.device).view(1, 1, 1, k_len)
        too_old = k_pos < (q_pos - (window - 1))
        return scores.masked_fill(too_old, -1e9)

    def _local_global_block_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
        block_size: int,
        local_num_blocks: int,
        global_num_blocks: int,
        global_stride: int,
        dropout_p: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_heads, q_len, d = q.shape
        k_len = k.size(-2)

        key_mask_2d = self._make_2d_mask_full(attention_mask_2d, k_len, q.device)
        if key_mask_2d.size(0) == 1 and bsz != 1:
            key_mask_2d = key_mask_2d.expand(bsz, -1)

        q, q_pad = self._pad_left_to_multiple(q, block_size)
        k, k_pad = self._pad_left_to_multiple(k, block_size)
        v, v_pad = self._pad_left_to_multiple(v, block_size)

        if q_pad != k_pad:
            delta = q_pad - k_pad
            if delta > 0:
                k = torch.nn.functional.pad(k, (0, 0, delta, 0))
                v = torch.nn.functional.pad(v, (0, 0, delta, 0))
                key_mask_2d = torch.nn.functional.pad(key_mask_2d, (delta, 0), value=0.0)
                k_pad = q_pad
            else:
                q = torch.nn.functional.pad(q, (0, 0, -delta, 0))
                q_pad = k_pad

        total_len = q.size(-2)
        n_blocks = total_len // block_size

        q_blocks = q.view(bsz, n_heads, n_blocks, block_size, d)
        k_blocks = k.view(bsz, n_heads, n_blocks, block_size, d)
        v_blocks = v.view(bsz, n_heads, n_blocks, block_size, d)

        km = torch.nn.functional.pad(key_mask_2d, (k_pad, 0), value=0.0)
        km_blocks = km.view(bsz, n_blocks, block_size)

        local_num_blocks = max(1, int(local_num_blocks))
        local_pad = local_num_blocks - 1

        k_blocks_p = torch.nn.functional.pad(k_blocks, (0, 0, 0, 0, local_pad, 0))
        v_blocks_p = torch.nn.functional.pad(v_blocks, (0, 0, 0, 0, local_pad, 0))
        km_blocks_p = torch.nn.functional.pad(km_blocks, (0, 0, local_pad, 0), value=0.0)

        k_local = k_blocks_p.unfold(dimension=2, size=local_num_blocks, step=1)
        v_local = v_blocks_p.unfold(dimension=2, size=local_num_blocks, step=1)
        km_local = km_blocks_p.unfold(dimension=1, size=local_num_blocks, step=1)

        k_local = k_local.permute(0, 1, 2, 5, 3, 4).contiguous()
        v_local = v_local.permute(0, 1, 2, 5, 3, 4).contiguous()

        km_local = km_local.permute(0, 1, 3, 2).contiguous()
        km_local = km_local.unsqueeze(1).unsqueeze(3)

        g_idx, g_valid = self._build_global_block_indices(n_blocks, global_num_blocks, global_stride, q.device)
        g = g_idx.size(1)
        if g == 0:
            g_len = 0
            k_g = None
            v_g = None
            km_g = None
        else:
            idx = g_idx.view(1, 1, n_blocks, g, 1, 1).expand(bsz, n_heads, n_blocks, g, block_size, d)
            k_g = torch.gather(
                k_blocks.unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size, d),
                dim=2,
                index=idx,
            )
            v_g = torch.gather(
                v_blocks.unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size, d),
                dim=2,
                index=idx,
            )

            idxm = g_idx.view(1, 1, n_blocks, g, 1).expand(bsz, n_heads, n_blocks, g, block_size)
            km_g = torch.gather(
                km_blocks.unsqueeze(1).unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size),
                dim=2,
                index=idxm,
            )
            km_g = km_g * g_valid.view(1, 1, n_blocks, g, 1)

            g_len = g * block_size

        qf = q_blocks.to(torch.float32)
        klf = k_local.to(torch.float32)
        vlf = v_local.to(torch.float32)

        s_local = torch.einsum("bhqtd,bhqwsd->bhqtws", qf, klf) * self.scaling
        s_local = s_local + (1.0 - km_local.to(torch.float32)) * -1e9

        intra = torch.triu(torch.full((block_size, block_size), -1e9, device=q.device, dtype=torch.float32), diagonal=1)
        s_local[:, :, :, :, -1, :] = s_local[:, :, :, :, -1, :] + intra[None, None, None, :, :]

        s_local = s_local.reshape(bsz, n_heads, n_blocks, block_size, local_num_blocks * block_size)

        if g_len > 0:
            kgf = k_g.to(torch.float32)
            s_g = torch.einsum("bhqtd,bhqwsd->bhqtws", qf, kgf) * self.scaling
            s_g = s_g + (1.0 - km_g.to(torch.float32).unsqueeze(3)) * -1e9
            s_g = s_g.reshape(bsz, n_heads, n_blocks, block_size, g_len)

            scores = torch.cat([s_local, s_g], dim=-1)

            v_g_flat = v_g.to(torch.float32).reshape(bsz, n_heads, n_blocks, g_len, d)
            v_l_flat = vlf.reshape(bsz, n_heads, n_blocks, local_num_blocks * block_size, d)
            v_all = torch.cat([v_l_flat, v_g_flat], dim=-2)
        else:
            scores = s_local
            v_all = vlf.reshape(bsz, n_heads, n_blocks, local_num_blocks * block_size, d)

        probs = torch.softmax(scores, dim=-1)
        probs = torch.nn.functional.dropout(probs, p=dropout_p, training=self.training)

        out = torch.einsum("bhqtk,bhqkd->bhqtd", probs, v_all)
        out = out.reshape(bsz, n_heads, total_len, d)
        out = out[:, :, q_pad:, :]
        return out, probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.qk_norm:
            query_states, key_states = self._apply_qk_norm(query_states, key_states)

        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values.get_seq_length()
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        use_sparse = self.attention_type in {"local_global_block_sparse", "block_sparse", "lg_block_sparse"}
        if use_sparse and past_len == 0 and q_len == key_states.size(-2):
            out, _ = self._local_global_block_sparse(
                query_states,
                key_states,
                value_states,
                attention_mask_2d,
                block_size=self.sparse_block_size,
                local_num_blocks=self.sparse_local_num_blocks,
                global_num_blocks=self.sparse_global_num_blocks,
                global_stride=self.sparse_global_stride,
                dropout_p=self.attention_dropout,
            )
            out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            return self.o_proj(out), None

        attn_scores = torch.matmul(query_states.to(torch.float32), key_states.transpose(2, 3).to(torch.float32)) * float(
            self.scaling
        )

        if self.sliding_window is not None and self.attention_type == "sliding_attention":
            attn_scores = self._apply_sliding_window_scores(attn_scores, int(self.sliding_window), past_len)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.to(torch.float32)

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, value_states.to(torch.float32))
        attn_output = attn_output.to(dtype=query_states.dtype)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, attn_probs
        return attn_output, None


class HumanVDecoderLayer(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HumanVAttention(config=config, layer_idx=layer_idx)
        self.mlp = HumanVMLP(config)
        self.input_layernorm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attention_mask_2d=attention_mask_2d,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


class HumanVPreTrainedModel(PreTrainedModel):
    config_class = HumanVConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HumanVDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HumanVModel(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HumanVDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HumanVRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _prepare_decoder_attention_mask(
        self,
        attention_mask_2d: Optional[torch.Tensor],
        input_shape: tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> torch.Tensor:
        batch_size, tgt_len = input_shape
        device = inputs_embeds.device
        dtype = torch.float32

        src_len = tgt_len + past_key_values_length
        if attention_mask_2d is not None:
            src_len = attention_mask_2d.shape[-1]

        causal_mask = torch.triu(
            torch.full((tgt_len, src_len), -1e9, device=device, dtype=dtype),
            diagonal=1 + past_key_values_length,
        )
        expanded_mask = causal_mask[None, None, :, :]

        if getattr(self.config, "use_sliding_window", False) and getattr(self.config, "sliding_window", None):
            window = int(self.config.sliding_window)
            q_pos = torch.arange(past_key_values_length, past_key_values_length + tgt_len, device=device).view(tgt_len, 1)
            k_pos = torch.arange(0, src_len, device=device).view(1, src_len)
            too_old = k_pos < (q_pos - (window - 1))
            expanded_mask = expanded_mask.masked_fill(too_old[None, None, :, :], -1e9)

        if attention_mask_2d is not None:
            expanded_attn_mask = attention_mask_2d[:, None, None, :].to(device=device, dtype=dtype)
            inverted_mask = (1.0 - expanded_attn_mask) * -1e9
            expanded_mask = expanded_mask + inverted_mask

        return expanded_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = bool(output_attentions) if output_attentions is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False
        return_dict = bool(return_dict) if return_dict is not None else True

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify either input_ids or inputs_embeds.")
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        batch_size, seq_length = inputs_embeds.shape[:2]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attn_mask_4d = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_length)

        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attn_mask_4d,
                    attention_mask,
                    position_embeddings,
                    past_key_values,
                    output_attentions,
                )
            else:
                hidden_states, attn_weights = layer(
                    hidden_states,
                    attention_mask=attn_mask_4d,
                    attention_mask_2d=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_attentions.append(attn_weights)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            out = (hidden_states, past_key_values)
            if output_hidden_states:
                out = out + (all_hidden_states,)
            if output_attentions:
                out = out + (all_attentions,)
            return out  # type: ignore[return-value]

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class HumanVForCausalLM(HumanVPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.model = HumanVModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            out = (logits, outputs[1])
            if loss is not None:
                out = (loss,) + out
            if output_hidden_states:
                out = out + (outputs[2],)
            if output_attentions:
                out = out + (outputs[3],)
            return out  # type: ignore[return-value]

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
            "inputs_embeds": inputs_embeds,
        }


__all__ = ["HumanVForCausalLM", "HumanVModel", "HumanVPreTrainedModel"]