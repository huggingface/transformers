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
from ...modeling_utils import PreTrainedModel
from .configuration_humanv import HumanVConfig


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
        dim = config.head_dim
        base = config.rope_parameters.get("rope_theta", 10000.0) if config.rope_parameters else 10000.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].to(device=x.device, dtype=torch.float32).expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].to(device=x.device, dtype=torch.float32)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


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
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if config.layer_types is not None else "full_attention"
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def _pad_left_to_multiple(self, x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
        l = x.size(-2)
        pad_left = (multiple - (l % multiple)) % multiple
        if pad_left == 0:
            return x, 0
        return torch.nn.functional.pad(x, (0, 0, pad_left, 0)), pad_left

    def _make_2d_mask_full(self, attention_mask_2d: Optional[torch.Tensor], k_len: int, device) -> torch.Tensor:
        if attention_mask_2d is None:
            return torch.ones((1, k_len), device=device, dtype=torch.float32)
        m = attention_mask_2d.to(device=device, dtype=torch.float32)
        if m.size(-1) < k_len:
            pad = torch.ones((m.size(0), k_len - m.size(-1)), device=device, dtype=torch.float32)
            m = torch.cat([pad, m], dim=-1)
        elif m.size(-1) > k_len:
            m = m[:, -k_len:]
        return m

    def _build_global_block_indices(self, n_blocks: int, num_global: int, stride: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        if num_global <= 0:
            idx = torch.empty((n_blocks, 0), device=device, dtype=torch.long)
            valid = torch.empty((n_blocks, 0), device=device, dtype=torch.float32)
            return idx, valid
        num_global = int(num_global)
        stride = max(1, int(stride))
        i = torch.arange(n_blocks, device=device, dtype=torch.long)
        if num_global == 1:
            idx = torch.zeros((n_blocks, 1), device=device, dtype=torch.long)
            valid = torch.ones((n_blocks, 1), device=device, dtype=torch.float32)
            return idx, valid
        offsets = torch.arange(num_global - 1, 0, -1, device=device, dtype=torch.long) * stride
        raw = i[:, None] - offsets[None, :]
        raw = torch.clamp(raw, min=0)
        idx = torch.cat([torch.zeros((n_blocks, 1), device=device, dtype=torch.long), raw], dim=1)
        valid = torch.ones_like(idx, dtype=torch.float32)
        dup = idx[:, 1:] == idx[:, :-1]
        valid[:, 1:][dup] = 0.0
        idx[:, 1:][dup] = 0
        return idx, valid

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
        km_local = km_local.unsqueeze(1)

        g_idx, g_valid = self._build_global_block_indices(n_blocks, global_num_blocks, global_stride, q.device)
        g = g_idx.size(1)
        if g == 0:
            g_len = 0
            k_g = None
            v_g = None
            km_g = None
        else:
            idx = g_idx.view(1, 1, n_blocks, g, 1, 1).expand(bsz, n_heads, n_blocks, g, block_size, d)
            k_g = torch.gather(k_blocks.unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size, d), dim=2, index=idx)
            v_g = torch.gather(v_blocks.unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size, d), dim=2, index=idx)

            idxm = g_idx.view(1, 1, n_blocks, g, 1).expand(bsz, n_heads, n_blocks, g, block_size)
            km_g = torch.gather(km_blocks.unsqueeze(1).unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size), dim=2, index=idxm)
            km_g = km_g * g_valid.view(1, 1, n_blocks, g, 1)

            g_len = g * block_size

        qf = q_blocks.to(torch.float32)
        klf = k_local.to(torch.float32)
        vlf = v_local.to(torch.float32)

        s_local = torch.einsum("bhqtd,bhqwsd->bhqtws", qf, klf) * self.scaling
        s_local = s_local + (1.0 - km_local) * -1e9

        intra = torch.triu(torch.full((block_size, block_size), -1e9, device=q.device, dtype=torch.float32), diagonal=1)
        s_local[:, :, :, :, -1, :] = s_local[:, :, :, :, -1, :] + intra[None, None, None, :, :]

        s_local = s_local.reshape(bsz, n_heads, n_blocks, block_size, local_num_blocks * block_size)

        if g_len > 0:
            kgf = k_g.to(torch.float32)
            s_g = torch.einsum("bhqtd,bhqwsd->bhqtws", qf, kgf) * self.scaling
            s_g = s_g + (1.0 - km_g.to(torch.float32).unsqueeze(1)) * -1e9
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        dtype = hidden_states.dtype

        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_length = 0
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            past_length = past_key_values.get_seq_length()

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if (
            self.layer_type == "sliding_attention"
            and self.config.sparse_attention_impl == "local_global_block"
            and past_length == 0
        ):
            block_size = int(self.config.sparse_block_size)
            local_num_blocks = int(self.config.sparse_local_num_blocks)
            global_num_blocks = int(self.config.sparse_global_num_blocks)
            global_stride = int(self.config.sparse_global_block_stride)
            dropout_p = self.attention_dropout if self.training else 0.0

            out, probs = self._local_global_block_sparse(
                q=query_states,
                k=key_states,
                v=value_states,
                attention_mask_2d=attention_mask_2d,
                block_size=block_size,
                local_num_blocks=local_num_blocks,
                global_num_blocks=global_num_blocks,
                global_stride=global_stride,
                dropout_p=dropout_p,
            )
            out = out.to(dtype=dtype)
            out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            return self.o_proj(out), probs.to(dtype=dtype)

        q = query_states.to(torch.float32)
        k = key_states.to(torch.float32)
        v = value_states.to(torch.float32)

        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.to(dtype=torch.float32)

        if self.layer_type == "sliding_attention":
            window = int(self.config.sparse_attention_window)
            if window > 0:
                k_len = attn_scores.size(-1)
                q_pos = torch.arange(past_length, past_length + q_len, device=attn_scores.device)
                k_pos = torch.arange(0, k_len, device=attn_scores.device)
                diff = q_pos[:, None] - k_pos[None, :]
                invalid = (diff < 0) | (diff >= window)
                sw = torch.zeros((q_len, k_len), device=attn_scores.device, dtype=torch.float32).masked_fill(invalid, -1e9)
                attn_scores = attn_scores + sw[None, None, :, :]

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nn.functional.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_probs, v).to(dtype=dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

        return self.o_proj(attn_output), attn_probs.to(dtype=dtype)


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
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attention_mask_2d=attention_mask_2d,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HumanVPreTrainedModel(PreTrainedModel):
    config_class = HumanVConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HumanVDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

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


class HumanVModel(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        _ = config.layer_types
        _ = config.attn_implementation
        _ = config.max_position_embeddings
        _ = config.use_sparse_attention
        _ = config.sparse_attention_impl
        _ = config.sparse_attention_window
        _ = config.sparse_block_size
        _ = config.sparse_local_num_blocks
        _ = config.sparse_global_num_blocks
        _ = config.sparse_global_block_stride

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([HumanVDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HumanVRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        batch_size, tgt_len = input_shape
        device = inputs_embeds.device
        dtype = torch.float32

        src_len = attention_mask.shape[-1] if attention_mask is not None else (tgt_len + past_key_values_length)

        causal_mask = torch.triu(
            torch.full((tgt_len, src_len), -1e9, device=device, dtype=dtype),
            diagonal=1 + past_key_values_length,
        )

        expanded_mask = causal_mask[None, None, :, :]

        if attention_mask is not None:
            expanded_attn_mask = attention_mask[:, None, None, :].to(device=device, dtype=dtype)
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
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if use_cache is None:
            use_cache = self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        batch_size, seq_length = inputs_embeds.shape[:2]

        if past_key_values is not None:
            past_length = past_key_values.get_seq_length()
        else:
            past_length = 0

        if (past_length + seq_length) > int(self.config.max_position_embeddings):
            raise ValueError(
                f"Sequence length {past_length + seq_length} exceeds max_position_embeddings={self.config.max_position_embeddings}."
            )

        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attention_mask_2d = attention_mask
        attn_mask_4d = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_length)

        hidden_states = inputs_embeds

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attn_mask_4d,
                    attention_mask_2d,
                    position_embeddings,
                    past_key_values,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attn_mask_4d,
                    attention_mask_2d=attention_mask_2d,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class HumanVForCausalLM(HumanVPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.model = HumanVModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }


__all__ = [
    "HumanVForCausalLM",
    "HumanVModel",
    "HumanVPreTrainedModel",
]
