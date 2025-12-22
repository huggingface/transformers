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

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_humanv import HumanVConfig


logger = logging.get_logger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


class HumanVRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(dtype=input_dtype)


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, config: HumanVConfig, device=None):
        super().__init__()
        dim = config.head_dim
        base = float(getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].to(dtype=torch.float32)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class HumanVMLP(nn.Module):
    def __init__(self, config: HumanVConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.dropout_p = float(config.attention_dropout)

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * self.scaling
        if attention_mask_4d is not None:
            scores = scores + attention_mask_4d.to(dtype=torch.float32)
        probs = torch.softmax(scores, dim=-1)
        probs = nn.functional.dropout(probs, p=self.dropout_p, training=self.training)
        vf = v.to(torch.float32)
        out = torch.matmul(probs, vf)
        return out.to(dtype=q.dtype)

    def _dense_sliding_mask(
        self,
        attention_mask_2d: Optional[torch.Tensor],
        q_len: int,
        k_len: int,
        past_len: int,
        window: int,
        device,
    ) -> torch.Tensor:
        dtype = torch.float32
        i = torch.arange(q_len, device=device) + past_len
        j = torch.arange(k_len, device=device)
        causal = j[None, :] <= i[:, None]
        lower = j[None, :] >= (i[:, None] - (window - 1))
        keep = causal & lower
        mask = torch.where(keep, torch.zeros((), device=device, dtype=dtype), torch.full((), -1e9, device=device, dtype=dtype))
        mask = mask[None, None, :, :]
        if attention_mask_2d is not None:
            am = attention_mask_2d[:, None, None, :].to(device=device, dtype=dtype)
            mask = mask + (1.0 - am) * -1e9
        return mask

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
        window_tokens: int,
    ) -> torch.Tensor:
        bsz, n_heads, q_len, d = q.shape
        k_len = k.shape[-2]
        if attention_mask_2d is None:
            attention_mask_2d = torch.ones((bsz, k_len), device=q.device, dtype=torch.float32)
        else:
            attention_mask_2d = attention_mask_2d.to(dtype=torch.float32)
            if attention_mask_2d.size(0) == 1 and bsz != 1:
                attention_mask_2d = attention_mask_2d.expand(bsz, -1)

        def pad_left_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
            seq_len = x.size(-2)
            pad = (multiple - (seq_len % multiple)) % multiple
            if pad == 0:
                return x, 0
            return nn.functional.pad(x, (0, 0, pad, 0)), pad

        q, q_pad = pad_left_to_multiple(q, block_size)
        k, k_pad = pad_left_to_multiple(k, block_size)
        v, v_pad = pad_left_to_multiple(v, block_size)

        if q_pad != k_pad:
            delta = q_pad - k_pad
            if delta > 0:
                k = nn.functional.pad(k, (0, 0, delta, 0))
                v = nn.functional.pad(v, (0, 0, delta, 0))
                attention_mask_2d = nn.functional.pad(attention_mask_2d, (delta, 0), value=0.0)
                k_pad = q_pad
            else:
                q = nn.functional.pad(q, (0, 0, -delta, 0))
                q_pad = k_pad

        total_len = q.size(-2)
        n_blocks = total_len // block_size

        q_blocks = q.view(bsz, n_heads, n_blocks, block_size, d)
        k_blocks = k.view(bsz, n_heads, n_blocks, block_size, d)
        v_blocks = v.view(bsz, n_heads, n_blocks, block_size, d)

        km = nn.functional.pad(attention_mask_2d, (k_pad, 0), value=0.0).view(bsz, n_blocks, block_size)

        local_num_blocks = max(1, int(local_num_blocks))
        local_pad = local_num_blocks - 1

        k_blocks_p = nn.functional.pad(k_blocks, (0, 0, 0, 0, local_pad, 0))
        v_blocks_p = nn.functional.pad(v_blocks, (0, 0, 0, 0, local_pad, 0))
        km_p = nn.functional.pad(km, (0, 0, local_pad, 0), value=0.0)

        k_local = (
            k_blocks_p.unfold(dimension=2, size=local_num_blocks, step=1)
            .permute(0, 1, 2, 5, 3, 4)
            .contiguous()
        )
        v_local = (
            v_blocks_p.unfold(dimension=2, size=local_num_blocks, step=1)
            .permute(0, 1, 2, 5, 3, 4)
            .contiguous()
        )

        km_local = km_p.unfold(dimension=1, size=local_num_blocks, step=1)
        km_local = km_local.permute(0, 1, 3, 2).contiguous()
        km_local = km_local.unsqueeze(1).unsqueeze(3)

        g = max(0, int(global_num_blocks))
        if g > 0:
            stride = max(1, int(global_stride))
            max_blocks_away = max(1, int(window_tokens) // block_size) if int(window_tokens) > 0 else n_blocks

            g_idx = torch.zeros((n_blocks, g), device=q.device, dtype=torch.long)
            g_valid = torch.zeros((n_blocks, g), device=q.device, dtype=torch.float32)

            for i in range(n_blocks):
                if i == 0:
                    continue
                start_block = max(0, i - max_blocks_away)
                cand = torch.arange(start_block, i, device=q.device, dtype=torch.long)
                if cand.numel() == 0:
                    continue
                if stride > 1:
                    cand = cand[::stride]
                if cand.numel() == 0:
                    continue
                if cand.numel() > g:
                    cand = cand[-g:]
                s = g - cand.numel()
                g_idx[i, s:] = cand
                g_valid[i, s:] = 1.0

            idx = g_idx.view(1, 1, n_blocks, g, 1, 1).expand(bsz, n_heads, n_blocks, g, block_size, d)
            k_g = torch.gather(k_blocks.unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size, d), dim=2, index=idx)
            v_g = torch.gather(v_blocks.unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size, d), dim=2, index=idx)

            idxm = g_idx.view(1, 1, n_blocks, g, 1).expand(bsz, n_heads, n_blocks, g, block_size)
            km_g = torch.gather(km.unsqueeze(1).unsqueeze(3).expand(bsz, n_heads, n_blocks, g, block_size), dim=2, index=idxm)
            km_g = km_g * g_valid.view(1, 1, n_blocks, g, 1)

            g_len = g * block_size
        else:
            k_g = None
            v_g = None
            km_g = None
            g_len = 0

        qf = q_blocks.to(torch.float32)
        klf = k_local.to(torch.float32)
        vlf = v_local.to(torch.float32)

        s_local = torch.einsum("bhqtd,bhqwsd->bhqtws", qf, klf) * self.scaling
        s_local = s_local + (1.0 - km_local.to(torch.float32)) * -1e9

        intra = torch.triu(
            torch.full((block_size, block_size), -1e9, device=q.device, dtype=torch.float32), diagonal=1
        )
        s_local[:, :, :, :, -1, :] = s_local[:, :, :, :, -1, :] + intra[None, None, None, :, :]

        s_local = s_local.reshape(bsz, n_heads, n_blocks, block_size, local_num_blocks * block_size)
        v_l_flat = vlf.reshape(bsz, n_heads, n_blocks, local_num_blocks * block_size, d)

        if g_len > 0:
            kgf = k_g.to(torch.float32)
            s_g = torch.einsum("bhqtd,bhqwsd->bhqtws", qf, kgf) * self.scaling
            s_g = s_g + (1.0 - km_g.to(torch.float32)).unsqueeze(3) * -1e9
            s_g = s_g.reshape(bsz, n_heads, n_blocks, block_size, g_len)

            scores = torch.cat([s_local, s_g], dim=-1)
            v_g_flat = v_g.to(torch.float32).reshape(bsz, n_heads, n_blocks, g_len, d)
            v_all = torch.cat([v_l_flat, v_g_flat], dim=-2)
        else:
            scores = s_local
            v_all = v_l_flat

        probs = torch.softmax(scores, dim=-1)
        probs = nn.functional.dropout(probs, p=self.dropout_p, training=self.training)
        out = torch.einsum("bhqtk,bhqkd->bhqtd", probs, v_all).to(dtype=q.dtype)

        out = out.reshape(bsz, n_heads, total_len, d)
        out = out[:, :, q_pad:, :]
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        layer_type = "full_attention"
        if hasattr(self.config, "layer_types") and self.config.layer_types is not None:
            layer_type = self.config.layer_types[self.layer_idx]

        use_sparse = bool(getattr(self.config, "use_sparse_attention", False))
        sparse_impl = str(getattr(self.config, "sparse_attention_impl", "local_global_block"))

        if layer_type == "sliding_attention" and use_sparse and sparse_impl == "local_global_block" and past_key_values is None:
            out = self._local_global_block_sparse(
                q=q,
                k=k,
                v=v,
                attention_mask_2d=attention_mask_2d,
                block_size=int(getattr(self.config, "sparse_block_size", 64)),
                local_num_blocks=int(getattr(self.config, "sparse_local_num_blocks", 4)),
                global_num_blocks=int(getattr(self.config, "sparse_global_num_blocks", 2)),
                global_stride=int(getattr(self.config, "sparse_global_block_stride", 4)),
                window_tokens=int(getattr(self.config, "sparse_attention_window", 256)),
            )
            out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            return self.o_proj(out), None

        if layer_type == "sliding_attention" and not use_sparse:
            window = int(getattr(self.config, "sliding_window", 256) or 256)
            past_len = 0 if past_key_values is None else past_key_values.get_seq_length()
            mask = self._dense_sliding_mask(
                attention_mask_2d=attention_mask_2d,
                q_len=q_len,
                k_len=k.size(-2),
                past_len=past_len,
                window=window,
                device=hidden_states.device,
            )
            attn_out = self._dense_attention(q, k, v, mask)
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            return self.o_proj(attn_out), None

        attn_out = self._dense_attention(q, k, v, attention_mask_4d)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_out), None


class HumanVDecoderLayer(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.self_attn = HumanVAttention(config=config, layer_idx=layer_idx)
        self.mlp = HumanVMLP(config)
        self.input_layernorm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask_4d=attention_mask_4d,
            attention_mask_2d=attention_mask_2d,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([HumanVDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HumanVRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _prepare_4d_causal_mask(
        self,
        attention_mask_2d: Optional[torch.Tensor],
        q_len: int,
        k_len: int,
        past_len: int,
        device,
    ) -> torch.Tensor:
        dtype = torch.float32
        causal = torch.triu(
            torch.full((q_len, k_len), -1e9, device=device, dtype=dtype),
            diagonal=1 + past_len,
        )
        mask = causal[None, None, :, :]
        if attention_mask_2d is not None:
            am = attention_mask_2d[:, None, None, :].to(device=device, dtype=dtype)
            mask = mask + (1.0 - am) * -1e9
        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = bool(self.config.use_cache)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        bsz, seq_len = inputs_embeds.shape[:2]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if seq_len + past_len > int(self.config.max_position_embeddings):
            raise ValueError(
                f"Sequence length {seq_len + past_len} exceeds max_position_embeddings={self.config.max_position_embeddings}"
            )

        position_ids = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attention_mask_2d = attention_mask
        if attention_mask_2d is not None and attention_mask_2d.dim() != 2:
            attention_mask_2d = None

        attention_mask_4d = self._prepare_4d_causal_mask(
            attention_mask_2d=attention_mask_2d,
            q_len=seq_len,
            k_len=seq_len + past_len,
            past_len=past_len,
            device=inputs_embeds.device,
        )

        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask_4d,
                    attention_mask_2d,
                    position_embeddings,
                    past_key_values,
                    output_attentions,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask_4d=attention_mask_4d,
                    attention_mask_2d=attention_mask_2d,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            return (hidden_states, past_key_values, all_hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class HumanVForCausalLM(HumanVPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.model = HumanVModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            return (loss, logits, outputs.past_key_values, outputs.hidden_states)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
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


__all__ = ["HumanVForCausalLM", "HumanVModel", "HumanVPreTrainedModel"]
