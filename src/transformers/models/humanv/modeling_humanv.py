# coding=utf-8
# Copyright 2025 The HumanV Team.
# Licensed under the Apache License, Version 2.0

from typing import Optional

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


def _as_fp32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)


class HumanVRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x).to(dtype=dtype)


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, config: HumanVConfig, device=None):
        super().__init__()
        dim = config.head_dim
        base = float(getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].to(device=x.device)
        pos = position_ids[:, None, :].to(dtype=torch.float32, device=x.device)
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class HumanVMLP(nn.Module):
    def __init__(self, config: HumanVConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, kvh, t, d = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(b, kvh, n_rep, t, d)
    return x.reshape(b, kvh * n_rep, t, d)


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.scaling = self.head_dim**-0.5
        self.dropout_p = float(config.attention_dropout)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.attn_compute_fp32 = str(getattr(config, "attention_compute_dtype", "fp32")).lower() == "fp32"

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
        window: Optional[int],
    ) -> torch.Tensor:
        if self.attn_compute_fp32:
            qf, kf, vf = _as_fp32(q), _as_fp32(k), _as_fp32(v)
        else:
            qf, kf, vf = q, k, v

        scores = torch.matmul(qf, kf.transpose(-2, -1)) * self.scaling

        if attention_mask_4d is not None:
            scores = scores + attention_mask_4d.to(dtype=scores.dtype)

        if window is not None and window > 0:
            b, h, tq, tk = scores.shape
            device = scores.device
            qi = torch.arange(tq, device=device)[:, None]
            kj = torch.arange(tk, device=device)[None, :]
            local = (kj <= qi) & (kj >= (qi - (window - 1)))
            scores = scores + (~local)[None, None, :, :].to(dtype=scores.dtype) * (-1e9)

        probs = torch.softmax(scores, dim=-1)
        if self.dropout_p > 0:
            probs = nn.functional.dropout(probs, p=self.dropout_p, training=self.training)

        out = torch.matmul(probs, vf)
        return out.to(dtype=q.dtype)

    def _local_global_block_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask_2d: Optional[torch.Tensor],
        block_size: int,
        local_num_blocks: int,
        global_num_blocks: int,
        global_stride: int,
        dropout_p: float,
    ) -> torch.Tensor:
        b, h, t, d = q.shape
        device = q.device

        if key_padding_mask_2d is None:
            key_padding_mask_2d = torch.ones((b, t), device=device, dtype=torch.bool)
        else:
            key_padding_mask_2d = key_padding_mask_2d.to(device=device, dtype=torch.bool)

        n_blocks = (t + block_size - 1) // block_size
        t_pad = n_blocks * block_size
        pad = t_pad - t

        if pad > 0:
            q = nn.functional.pad(q, (0, 0, 0, pad))
            k = nn.functional.pad(k, (0, 0, 0, pad))
            v = nn.functional.pad(v, (0, 0, 0, pad))
            key_padding_mask_2d = nn.functional.pad(key_padding_mask_2d, (0, pad), value=False)

        q_blk = q.view(b, h, n_blocks, block_size, d)
        k_blk = k.view(b, h, n_blocks, block_size, d)
        v_blk = v.view(b, h, n_blocks, block_size, d)
        km_blk = key_padding_mask_2d.view(b, n_blocks, block_size)

        max_local_by_window = max(1, int(getattr(self.config, "sparse_attention_window", block_size) // block_size))
        local_num_blocks = int(min(local_num_blocks, max_local_by_window))

        block_ids = torch.arange(n_blocks, device=device)
        offsets = torch.arange(local_num_blocks - 1, -1, -1, device=device)
        idx_local = block_ids[:, None] - offsets[None, :]
        valid_local = idx_local >= 0
        idx_local = idx_local.clamp_min(0)

        k_local = torch.stack([k_blk[:, :, idx_local[:, i], :, :] for i in range(local_num_blocks)], dim=3)
        v_local = torch.stack([v_blk[:, :, idx_local[:, i], :, :] for i in range(local_num_blocks)], dim=3)
        km_local = torch.stack([km_blk[:, idx_local[:, i], :] for i in range(local_num_blocks)], dim=2)

        valid_local_mask = valid_local[None, :, :, None].expand(b, -1, -1, block_size)
        km_local = km_local & valid_local_mask

        if global_num_blocks > 0 and global_stride > 0:
            g_offsets = torch.arange(global_num_blocks - 1, -1, -1, device=device)
            gpos = (block_ids - 1).clamp_min(0) // global_stride
            idx_global = (gpos[:, None] - g_offsets[None, :]) * global_stride
            valid_global = (block_ids[:, None] > 0) & (idx_global >= 0)
            idx_global = idx_global.clamp_min(0)

            k_gblk = torch.stack([k_blk[:, :, idx_global[:, i], :, :] for i in range(global_num_blocks)], dim=3)
            v_gblk = torch.stack([v_blk[:, :, idx_global[:, i], :, :] for i in range(global_num_blocks)], dim=3)
            km_gblk = torch.stack([km_blk[:, idx_global[:, i], :] for i in range(global_num_blocks)], dim=2)

            valid_g = valid_global[None, :, :, None].expand(b, -1, -1, block_size)
            km_gblk = km_gblk & valid_g

            denom = km_gblk.to(torch.float32).sum(dim=-1).clamp_min(1.0)
            k_gsum = (k_gblk * km_gblk[:, None, :, :, :, None].to(k_gblk.dtype)).sum(dim=-2) / denom[:, None, :, :, None]
            v_gsum = (v_gblk * km_gblk[:, None, :, :, :, None].to(v_gblk.dtype)).sum(dim=-2) / denom[:, None, :, :, None]
            km_gsum = valid_global[None, :, :].expand(b, -1, -1)
        else:
            k_gsum = None
            v_gsum = None
            km_gsum = None
            global_num_blocks = 0

        if self.attn_compute_fp32:
            qf = _as_fp32(q_blk)
            k_local_f = _as_fp32(k_local)
            v_local_f = _as_fp32(v_local)
            if k_gsum is not None:
                k_gsum_f = _as_fp32(k_gsum)
                v_gsum_f = _as_fp32(v_gsum)
        else:
            qf = q_blk
            k_local_f = k_local
            v_local_f = v_local
            if k_gsum is not None:
                k_gsum_f = k_gsum
                v_gsum_f = v_gsum

        s_local = torch.einsum("bhnqd,bhnlkd->bhnqlk", qf, k_local_f) * self.scaling
        s_local = s_local + (~km_local[:, None, :, :, None, :]).to(s_local.dtype)[:, :, :, None, :, :] * (-1e9)

        intra = torch.triu(torch.full((block_size, block_size), -1e9, device=device, dtype=s_local.dtype), diagonal=1)
        s_local[:, :, :, :, -1, :] = s_local[:, :, :, :, -1, :] + intra[None, None, None, :, :]

        s_local = s_local.reshape(b, h, n_blocks, block_size, local_num_blocks * block_size)

        if global_num_blocks > 0:
            s_global = torch.einsum("bhnqd,bhngd->bhnqg", qf, k_gsum_f) * self.scaling
            s_global = s_global + (~km_gsum[:, None, :, None, :]).to(s_global.dtype) * (-1e9)

            scores = torch.cat([s_local, s_global], dim=-1)
        else:
            scores = s_local

        probs = torch.softmax(scores, dim=-1)
        if dropout_p > 0:
            probs = nn.functional.dropout(probs, p=dropout_p, training=self.training)

        p_local = probs[..., : local_num_blocks * block_size].reshape(b, h, n_blocks, block_size, local_num_blocks, block_size)
        out_local = torch.einsum("bhnqlk,bhnlkd->bhnqd", p_local, v_local_f)

        if global_num_blocks > 0:
            p_global = probs[..., local_num_blocks * block_size :]
            out_global = torch.einsum("bhnqg,bhngd->bhnqd", p_global, v_gsum_f)
            out = out_local + out_global
        else:
            out = out_local

        out = out.to(dtype=q.dtype).reshape(b, h, t_pad, d)
        return out[:, :, :t, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        use_sparse = bool(getattr(self.config, "use_sparse_attention", False)) and self.layer_type == "sliding_attention"
        sparse_impl = str(getattr(self.config, "sparse_attention_impl", "local_global_block"))

        if use_sparse and sparse_impl == "local_global_block" and (past_key_values is None or q_len > 1):
            out = self._local_global_block_sparse(
                q=q,
                k=k,
                v=v,
                key_padding_mask_2d=attention_mask_2d,
                block_size=int(getattr(self.config, "sparse_block_size", 64)),
                local_num_blocks=int(getattr(self.config, "sparse_local_num_blocks", 4)),
                global_num_blocks=int(getattr(self.config, "sparse_global_num_blocks", 2)),
                global_stride=int(getattr(self.config, "sparse_global_block_stride", 4)),
                dropout_p=self.dropout_p,
            )
        else:
            window = None
            if self.layer_type == "sliding_attention":
                window = int(getattr(self.config, "sparse_attention_window", 256))
            out = self._dense_attention(q, k, v, attention_mask, window=window)

        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(out), None


class HumanVDecoderLayer(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_type = config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "full_attention"
        self.self_attn = HumanVAttention(config=config, layer_idx=layer_idx, layer_type=layer_type)
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

    def _init_weights(self, module: nn.Module):
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

    def _prepare_decoder_attention_mask_4d(
        self,
        attention_mask_2d: Optional[torch.Tensor],
        input_shape: tuple[int, int],
        past_key_values_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        bsz, tgt_len = input_shape
        src_len = tgt_len + past_key_values_length

        causal = torch.triu(
            torch.full((tgt_len, src_len), -1e9, device=device, dtype=torch.float32),
            diagonal=1 + past_key_values_length,
        )
        mask = causal[None, None, :, :]

        if attention_mask_2d is not None:
            am = attention_mask_2d[:, None, None, :].to(device=device, dtype=torch.float32)
            mask = mask + (1.0 - am) * -1e9

        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify input_ids or inputs_embeds.")
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = bool(self.config.use_cache)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        bsz, seq_len = inputs_embeds.shape[:2]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        position_ids = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attention_mask_2d = attention_mask
        attention_mask_4d = self._prepare_decoder_attention_mask_4d(
            attention_mask_2d=attention_mask_2d,
            input_shape=(bsz, seq_len),
            past_key_values_length=past_len,
            device=inputs_embeds.device,
        )

        hidden_states = inputs_embeds

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask_4d,
                    attention_mask_2d,
                    position_embeddings,
                    past_key_values,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
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

    def get_output_embeddings(self):
        return self.lm_head

    def set_input_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).to(torch.float32)

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
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }


__all__ = ["HumanVForCausalLM", "HumanVModel", "HumanVPreTrainedModel"]
