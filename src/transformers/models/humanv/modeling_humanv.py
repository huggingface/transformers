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

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_humanv import HumanVConfig

logger = logging.get_logger(__name__)


class HumanVRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hs).to(input_dtype)


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


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, config: HumanVConfig, device=None):
        super().__init__()
        dim = int(config.head_dim)
        rope_theta = 10000.0
        if getattr(config, "rope_parameters", None) is not None:
            rope_theta = float(config.rope_parameters.get("rope_theta", 10000.0))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].to(device=x.device, dtype=torch.float32)
        inv_freq = inv_freq.expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].to(device=x.device, dtype=torch.float32)
        freqs = (inv_freq @ pos).transpose(1, 2)
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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, num_kv_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hs = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seqlen, head_dim)
    return hs.reshape(bsz, num_kv_heads * n_rep, seqlen, head_dim)


def _pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -2, value: float = 0.0) -> tuple[torch.Tensor, int]:
    length = x.size(dim)
    pad_len = (multiple - (length % multiple)) % multiple
    if pad_len == 0:
        return x, 0
    pad = [0, 0] * x.dim()
    pad_index = (-dim - 1) * 2
    pad[pad_index] = 0
    pad[pad_index + 1] = pad_len
    x = torch.nn.functional.pad(x, pad, value=value)
    return x, pad_len


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int, layer_type: str = "full_attention"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        self.head_dim = int(config.head_dim)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self._compute_fp32 = str(getattr(config, "attention_compute_dtype", "fp32")).lower() == "fp32"
        self._attn_impl = str(getattr(config, "attn_implementation", "eager")).lower()

        self._sparse_enabled = bool(getattr(config, "use_sparse_attention", False))
        self._sparse_impl = str(getattr(config, "sparse_attention_impl", "local_global_block"))

        self._sparse_block = int(getattr(config, "sparse_block_size", 64))
        self._sparse_local_blocks = int(getattr(config, "sparse_local_num_blocks", 4))
        self._sparse_global_blocks = int(getattr(config, "sparse_global_num_blocks", 2))
        self._sparse_global_stride = int(getattr(config, "sparse_global_block_stride", 4))

        self._oow_enabled = bool(getattr(config, "sparse_oow_enabled", False))
        gate_init = float(getattr(config, "sparse_oow_gate_init", 0.0))
        if self._oow_enabled:
            self.oow_gate = nn.Parameter(torch.full((self.num_heads, 1, 1), gate_init, dtype=torch.float32))
        else:
            self.oow_gate = None

    def _apply_attention_mask_slice(
        self, scores: torch.Tensor, attention_mask: Optional[torch.Tensor], q_pos: slice, k_pos: torch.Tensor
    ) -> torch.Tensor:
        if attention_mask is None:
            return scores
        am = attention_mask
        if am.dim() != 4:
            raise ValueError(f"Expected attention_mask with shape [bsz, 1, q_len, kv_len], got {am.shape}")
        bsz, _, q_len, kv_len = am.shape
        max_k = int(k_pos.max().item()) if k_pos.numel() > 0 else -1
        if max_k >= kv_len:
            pad_k = max_k + 1 - kv_len
            am = torch.nn.functional.pad(am, (0, pad_k), value=-1e9)
        q_end = q_pos.stop
        if q_end is not None and q_end > q_len:
            pad_q = q_end - q_len
            am = torch.nn.functional.pad(am, (0, 0, 0, pad_q), value=-1e9)
        mask_slice = am[:, :, q_pos, :].index_select(-1, k_pos).to(dtype=scores.dtype)
        return scores + mask_slice

    def _dense_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if self._compute_fp32:
            scores = scores.to(torch.float32)
        if attention_mask is not None:
            scores = scores + attention_mask.to(dtype=scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nn.functional.dropout(probs, p=self.attention_dropout, training=self.training)
        if self._compute_fp32:
            out = torch.matmul(probs.to(torch.float32), v.to(torch.float32)).to(dtype=q.dtype)
        else:
            out = torch.matmul(probs.to(dtype=q.dtype), v)
        return out

    def _sliding_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        window: int,
    ) -> torch.Tensor:
        bsz, heads, q_len, _ = q.shape
        kv_len = k.size(-2)
        device = q.device

        pos_k = torch.arange(kv_len, device=device)
        pos_q = torch.arange(q_len, device=device)[:, None]
        too_old = pos_k[None, :] < (pos_q - int(window))
        sw_mask = torch.zeros((q_len, kv_len), device=device, dtype=torch.float32)
        sw_mask = sw_mask.masked_fill(too_old, -1e9)
        sw_mask = sw_mask[None, None, :, :]

        if attention_mask is None:
            combined = sw_mask
        else:
            combined = attention_mask.to(dtype=torch.float32) + sw_mask

        if self._compute_fp32:
            combined = combined.to(torch.float32)
        else:
            combined = combined.to(dtype=q.dtype)

        return self._dense_attention(q, k, v, combined)

    def _local_global_block_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        block_size: int,
        local_num_blocks: int,
        global_num_blocks: int,
        global_stride: int,
    ) -> torch.Tensor:
        bsz, heads, seqlen, dim = q.shape
        B = int(block_size)
        if B <= 0:
            raise ValueError("sparse_block_size must be > 0")

        q_pad, q_pad_len = _pad_to_multiple(q, B, dim=-2, value=0.0)
        k_pad, k_pad_len = _pad_to_multiple(k, B, dim=-2, value=0.0)
        v_pad, v_pad_len = _pad_to_multiple(v, B, dim=-2, value=0.0)

        padded_len = q_pad.size(-2)
        num_blocks = padded_len // B

        q_blk = q_pad.view(bsz, heads, num_blocks, B, dim)
        k_blk = k_pad.view(bsz, heads, num_blocks, B, dim)
        v_blk = v_pad.view(bsz, heads, num_blocks, B, dim)

        out_blk = torch.zeros((bsz, heads, num_blocks, B, dim), device=q.device, dtype=q.dtype)

        intra = torch.triu(torch.full((B, B), -1e9, device=q.device, dtype=torch.float32), diagonal=1)

        for t in range(num_blocks):
            q_t = q_blk[:, :, t, :, :]
            local_start = max(0, t - int(local_num_blocks))
            local_idx = list(range(local_start, t + 1))

            global_candidates = list(range(0, t + 1, max(1, int(global_stride))))
            if global_num_blocks > 0 and len(global_candidates) > 0:
                global_idx = global_candidates[-int(global_num_blocks) :]
            else:
                global_idx = []

            idx = sorted(set(local_idx + global_idx))
            idx_t = torch.tensor(idx, device=q.device, dtype=torch.long)
            k_sel = k_blk.index_select(2, idx_t)
            v_sel = v_blk.index_select(2, idx_t)

            scores = torch.einsum("bhqd,bhspd->bhqsp", q_t, k_sel) * self.scaling
            if self._compute_fp32:
                scores = scores.to(torch.float32)

            if t in idx:
                cur_pos = idx.index(t)
                scores[:, :, :, cur_pos, :] = scores[:, :, :, cur_pos, :] + intra[None, None, :, :]

            scores = scores.reshape(bsz, heads, B, len(idx) * B)

            q_pos = slice(t * B, (t + 1) * B)
            k_positions = (idx_t[:, None] * B + torch.arange(B, device=q.device)[None, :]).reshape(-1)
            scores = self._apply_attention_mask_slice(scores, attention_mask, q_pos, k_positions)

            probs = torch.softmax(scores, dim=-1)
            probs = torch.nn.functional.dropout(probs, p=self.attention_dropout, training=self.training)

            v_flat = v_sel.reshape(bsz, heads, len(idx) * B, dim)
            if self._compute_fp32:
                out = torch.matmul(probs.to(torch.float32), v_flat.to(torch.float32)).to(dtype=q.dtype)
            else:
                out = torch.matmul(probs.to(dtype=q.dtype), v_flat)

            out_blk[:, :, t, :, :] = out

        out = out_blk.reshape(bsz, heads, padded_len, dim)
        out = out[:, :, :seqlen, :]
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_probs = None

        if self._sparse_enabled and self._sparse_impl == "local_global_block":
            attn_out = self._local_global_block_sparse(
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask,
                block_size=self._sparse_block,
                local_num_blocks=self._sparse_local_blocks,
                global_num_blocks=self._sparse_global_blocks,
                global_stride=self._sparse_global_stride,
            )
        elif self.layer_type == "sliding_attention":
            window = int(getattr(self.config, "sliding_window", None) or getattr(self.config, "sparse_attention_window", 256))
            attn_out = self._sliding_window_attention(q, k, v, attention_mask, window)
        else:
            if self._attn_impl == "sdpa" and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                q_ = q
                k_ = k
                v_ = v
                am = None
                if attention_mask is not None:
                    am = attention_mask.to(dtype=torch.float32)
                if self._compute_fp32:
                    q_ = q_.to(torch.float32)
                    k_ = k_.to(torch.float32)
                    v_ = v_.to(torch.float32)
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q_, k_, v_, attn_mask=am, dropout_p=self.attention_dropout if self.training else 0.0, is_causal=False
                )
                if self._compute_fp32:
                    attn_out = attn_out.to(dtype=q.dtype)
            else:
                attn_out = self._dense_attention(q, k, v, attention_mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), attn_probs


class HumanVDecoderLayer(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int, layer_type: str = "full_attention"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HumanVAttention(config=config, layer_idx=layer_idx, layer_type=layer_type)
        self.mlp = HumanVMLP(config)
        self.input_layernorm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
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

    def _init_weights(self, module: nn.Module) -> None:
        std = float(self.config.initializer_range)
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

        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * config.num_hidden_layers
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError("config.layer_types must have length == num_hidden_layers")

        self.layers = nn.ModuleList(
            [HumanVDecoderLayer(config, layer_idx=i, layer_type=layer_types[i]) for i in range(config.num_hidden_layers)]
        )
        self.norm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HumanVRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> torch.Tensor:
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
            inverted = (1.0 - expanded_attn_mask) * -1e9
            expanded_mask = expanded_mask + inverted

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
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide either input_ids or inputs_embeds.")
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = bool(self.config.use_cache)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        batch_size, seq_length = inputs_embeds.shape[:2]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if (seq_length + past_length) > int(self.config.max_position_embeddings):
            raise ValueError(
                f"Sequence length {seq_length + past_length} exceeds max_position_embeddings={self.config.max_position_embeddings}."
            )

        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attn_mask_4d = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_length)

        hidden_states = inputs_embeds

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attn_mask_4d,
                    position_embeddings,
                    past_key_values,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attn_mask_4d,
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

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self):
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
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }


class HumanVForSequenceClassification(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.model = HumanVModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=True)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state

        if attention_mask is None:
            pooled = hidden_states[:, -1, :]
        else:
            lengths = attention_mask.to(torch.long).sum(dim=-1) - 1
            lengths = lengths.clamp(min=0)
            pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), lengths]

        logits = self.score(pooled)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = torch.nn.functional.mse_loss(logits.squeeze(-1), labels.to(logits.dtype))
            else:
                loss = torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HumanVForTokenClassification(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.model = HumanVModel(config)
        self.dropout = nn.Dropout(float(getattr(config, "classifier_dropout", 0.0)))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active]
                active_labels = labels.view(-1)[active]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HumanVForQuestionAnswering(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.model = HumanVModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.clamp(0, start_logits.size(1) - 1)
            end_positions = end_positions.clamp(0, end_logits.size(1) - 1)
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HumanVForCausalLM",
    "HumanVForQuestionAnswering",
    "HumanVForSequenceClassification",
    "HumanVForTokenClassification",
    "HumanVModel",
    "HumanVPreTrainedModel",
]
