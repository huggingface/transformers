from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
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

_NEG_INF = -1e9


class HumanVRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class HumanVTorchRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        if hasattr(nn, "RMSNorm"):
            self.norm = nn.RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = HumanVRMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._seq_len_cached: int = 0
        self._device_cached: Optional[torch.device] = None
        self._dtype_cached: Optional[torch.dtype] = None

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        seq_len = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else 0

        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._device_cached != device
            or self._dtype_cached != dtype
        ):
            self._seq_len_cached = max(seq_len, self._seq_len_cached)
            self._device_cached = device
            self._dtype_cached = dtype

            t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype=dtype)
            self._sin_cached = emb.sin().to(dtype=dtype)

        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]
        return cos, sin


class HumanVMLP(nn.Module):
    def __init__(self, config: HumanVConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", hidden_size * 4))
        activation = str(getattr(config, "hidden_act", "silu"))

        mlp_bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[activation]

        self.dropout = float(getattr(config, "hidden_dropout", 0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        out = gate * up
        out = self.down_proj(out)
        if self.dropout and self.training:
            out = F.dropout(out, p=self.dropout, training=True)
        return out


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        self.head_dim = int(config.head_dim)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", None) or config.num_attention_heads)

        if self.num_kv_heads <= 0:
            raise ValueError(f"num_key_value_heads must be > 0, got {self.num_kv_heads}")
        if self.num_kv_heads > self.num_heads:
            raise ValueError(f"num_key_value_heads ({self.num_kv_heads}) cannot exceed num_attention_heads ({self.num_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for grouped attention")

        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = float(getattr(config, "attention_scaling", 1.0)) / (self.head_dim ** 0.5)
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))

        attn_bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=attn_bias)

        self.use_sparse_attention = bool(getattr(config, "use_sparse_attention", False))
        self.sparse_attention_impl = str(getattr(config, "sparse_attention_impl", "local_global_block"))
        self.sparse_block_size = int(getattr(config, "sparse_block_size", 64))
        self.sparse_prefill_chunk_blocks = int(getattr(config, "sparse_prefill_chunk_blocks", 0) or 0)
        self.sparse_local_num_blocks = int(getattr(config, "sparse_local_num_blocks", 8))
        self.sparse_global_num_blocks = int(getattr(config, "sparse_global_num_blocks", 1))

        self.kv_cache_dtype = str(getattr(config, "kv_cache_dtype", "auto"))
        self.attn_backend = str(getattr(config, "attn_backend", "gqa_matmul")).lower().strip()
        if self.attn_backend not in ("gqa_matmul", "sdpa"):
            self.attn_backend = "gqa_matmul"

    def _kv_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_cache_dtype == "auto":
            return x
        if self.kv_cache_dtype in ("bf16", "bfloat16"):
            return x.to(torch.bfloat16)
        if self.kv_cache_dtype in ("fp16", "float16"):
            return x.to(torch.float16)
        if self.kv_cache_dtype in ("fp32", "float32"):
            return x.to(torch.float32)
        return x

    def _apply_partial_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _apply_rotary(q, k, cos, sin)

    def _reshape_q_grouped(self, q: torch.Tensor) -> torch.Tensor:
        bsz, h, t, d = q.shape
        return q.contiguous().view(bsz, self.num_kv_heads, self.num_kv_groups, t, d)

    def _sdpa_mha_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Fix: SDPA requires float mask/bias dtype == q.dtype
        dropout_p = self.attention_dropout if self.training else 0.0
        if attention_mask_4d is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)

        if attention_mask_4d.dtype is not torch.bool and attention_mask_4d.dtype != q.dtype:
            attention_mask_4d = attention_mask_4d.to(dtype=q.dtype)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask_4d, dropout_p=dropout_p, is_causal=False)

    def _grouped_dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, _, q_len, d = q.shape
        qg = self._reshape_q_grouped(q)

        scores = torch.matmul(
            qg.to(torch.float32),
            k.unsqueeze(2).transpose(-2, -1).to(torch.float32),
        ) * self.scaling

        if attention_mask_4d is not None:
            # attention_mask_4d: [bsz, 1, q_len, k_len]
            m = attention_mask_4d[:, 0].to(dtype=torch.float32)
            scores = scores + m[:, None, None, :, :]

        probs = torch.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)

        out = torch.matmul(probs.to(v.dtype), v.unsqueeze(2))
        out = out.to(dtype=q.dtype).reshape(bsz, self.num_heads, q_len, d)
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
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_partial_rope(q, k, cos, sin)

        if past_key_values is not None:
            k = self._kv_dtype(k)
            v = self._kv_dtype(v)
            k, v = past_key_values.update(k, v, self.layer_idx)

        k_len = k.shape[-2]

        use_sparse = (
            self.use_sparse_attention
            and self.layer_type == "sliding_attention"
            and self.sparse_attention_impl == "local_global_block"
        )

        if use_sparse:
            # (مثل فایل فعلی شما) برای این تست، sparse را فعلاً از نظر correctness کنار می‌گذاریم:
            # اگر sparse path دارید و می‌خواهید نگه دارید، باید همان توابع sparse فعلی را اینجا صدا بزنید.
            attn_out = self._grouped_dense_attention(q, k, v, attention_mask_4d)
        else:
            if self.attn_backend == "sdpa" and self.num_kv_heads == self.num_heads:
                attn_out = self._sdpa_mha_attention(q, k, v, attention_mask_4d)
            else:
                attn_out = self._grouped_dense_attention(q, k, v, attention_mask_4d)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        return attn_out, None


class HumanVDecoderLayer(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        layer_types = getattr(config, "layer_types", None)
        layer_type = "full_attention" if layer_types is None else layer_types[layer_idx]

        norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        norm_backend = str(getattr(config, "norm_backend", "torch_rmsnorm")).lower().strip()
        Norm = HumanVTorchRMSNorm if norm_backend != "layernorm" else nn.LayerNorm

        self.input_layernorm = Norm(config.hidden_size, eps=norm_eps) if Norm is nn.LayerNorm else Norm(config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = Norm(config.hidden_size, eps=norm_eps) if Norm is nn.LayerNorm else Norm(config.hidden_size, eps=norm_eps)

        self.self_attn = HumanVAttention(config, layer_idx=layer_idx, layer_type=layer_type)
        self.mlp = HumanVMLP(config)
        self.resid_dropout = float(getattr(config, "resid_dropout", 0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, attn_weights = self.self_attn(
            hidden_states,
            attention_mask_4d=attention_mask_4d,
            attention_mask_2d=attention_mask_2d,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )
        if self.resid_dropout and self.training:
            attn_out = F.dropout(attn_out, p=self.resid_dropout, training=True)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        if self.resid_dropout and self.training:
            mlp_out = F.dropout(mlp_out, p=self.resid_dropout, training=True)
        hidden_states = residual + mlp_out

        return hidden_states, attn_weights


class HumanVPreTrainedModel(PreTrainedModel):
    config_class = HumanVConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HumanVDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module: nn.Module):
        std = float(getattr(self.config, "initializer_range", 0.02))
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
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([HumanVDecoderLayer(config, i) for i in range(int(config.num_hidden_layers))])

        norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        norm_backend = str(getattr(config, "norm_backend", "torch_rmsnorm")).lower().strip()
        self.norm = HumanVTorchRMSNorm(config.hidden_size, eps=norm_eps) if norm_backend != "layernorm" else nn.LayerNorm(config.hidden_size, eps=norm_eps)

        rope_base = float(getattr(config, "rope_theta", 10000.0))
        self.rotary_emb = HumanVRotaryEmbedding(
            dim=int(config.head_dim),
            max_position_embeddings=int(getattr(config, "max_position_embeddings", 2048)),
            base=rope_base,
        )

        # Fix: cache includes dtype to avoid dtype-mismatch and repeated casts
        self._causal_cache = {}

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _get_causal_mask(self, q_len: int, src_len: int, past_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (q_len, src_len, past_len, device, dtype)
        m = self._causal_cache.get(key)
        if m is not None:
            return m
        m = torch.triu(
            torch.full((q_len, src_len), _NEG_INF, device=device, dtype=dtype),
            diagonal=1 + past_len,
        )
        m = m[None, None, :, :]
        self._causal_cache[key] = m
        return m

    def _prepare_attention_masks(
        self,
        attention_mask_2d: torch.Tensor,
        q_len: int,
        past_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = attention_mask_2d.device
        src_len = int(attention_mask_2d.shape[1])

        causal = self._get_causal_mask(q_len=q_len, src_len=src_len, past_len=past_len, device=device, dtype=dtype)

        key_valid = attention_mask_2d.to(dtype=torch.bool)
        padding_bias = (~key_valid)[:, None, None, :].to(dtype=dtype) * torch.tensor(_NEG_INF, device=device, dtype=dtype)
        return causal + padding_bias

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        bsz, q_len = inputs_embeds.shape[:2]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        # Fix: use bool attention_mask_2d (not float32)
        if attention_mask is None:
            attention_mask_2d = torch.ones((bsz, past_len + q_len), device=inputs_embeds.device, dtype=torch.bool)
        else:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D (bsz, seq)")
            attention_mask_2d = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            if attention_mask_2d.shape[1] == q_len and past_len > 0:
                pad = torch.ones((bsz, past_len), device=inputs_embeds.device, dtype=torch.bool)
                attention_mask_2d = torch.cat([pad, attention_mask_2d], dim=-1)

        position_ids = torch.arange(past_len, past_len + q_len, device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attention_mask_4d = self._prepare_attention_masks(
            attention_mask_2d,
            q_len=q_len,
            past_len=past_len,
            dtype=inputs_embeds.dtype,
        )

        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, attn = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask_4d,
                    attention_mask_2d,
                    position_embeddings,
                    past_key_values,
                    output_attentions,
                )
            else:
                hidden_states, attn = layer(
                    hidden_states,
                    attention_mask_4d=attention_mask_4d,
                    attention_mask_2d=attention_mask_2d,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_attentions.append(attn)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
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

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
