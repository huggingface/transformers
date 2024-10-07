import logging
import math
from copy import deepcopy
from dataclasses import fields, dataclass, replace
from enum import Enum
from typing import List, Optional, Tuple, Union, Dict, Any, Sequence, Callable, cast, MutableMapping

import torch
from einops import einsum, einops
from transformers import PreTrainedModel, GenerationConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.auto import AutoModelForCausalLM
from torch import nn

from .config_molmo import MolmoConfig
from torch.nn import functional as F


log = logging.getLogger(__name__)


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class ImageProjectType(StrEnum):
    mlp = "mlp"
    mlpx2 = "2mlp"
    linear = "linear"


class ImagePooling2DType(StrEnum):
    attention = "attention"
    attention_meanq = "attention-meanq"
    attention_2wide = "attention_2wide"
    attention_v2 = "attention-v2"
    none = "none"
    stack = "stack"


class ActivationType(StrEnum):
    quick_gelu = "quick_gelu"
    gelu = "gelu"
    gelu_tanh = "gelu_tanh"
    relu = "relu"
    silu = "silu"
    llama_geglu = "llama_geglu"
    llama_geglu_tanh = "llama_geglu_tanh"
    llama_swiglu = "llama_swiglu"
    swiglu = "swiglu"


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


class MolmoConfigurationError(Exception):
    pass


def _non_meta_init_device(config) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: MolmoConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(
            config.max_position_embeddings or config.max_sequence_length,
            _non_meta_init_device(config)
        )

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            if self.config.rope_impl == "interleave":
                positions = freqs.repeat_interleave(2, dim=-1)
            else:
                positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, hs // 2, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.view(B, nh, T, hs)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.rope_impl == "interleave":
            return ((t * pos_cos) + (self.rotate_every_two(t) * pos_sin)).to(t.dtype)
        else:
            return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            batch_size = q_.shape[0]
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            if position_ids is not None:
                freqs_cis_len = (self.config.max_position_embeddings or self.config.max_sequence_length)
            else:
                freqs_cis_len = key_len
            pos_sin, pos_cos = self.get_rotary_embedding(freqs_cis_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            if position_ids is not None:
                assert query_len == key_len, "Query and key lengths must be equal when using position IDs."
                pos_sin = pos_sin[0, 0][position_ids].view(
                    (batch_size, 1, key_len, pos_sin.shape[-1])
                )
                pos_cos = pos_cos[0, 0][position_ids].view(
                    (batch_size, 1, key_len, pos_cos.shape[-1])
                )
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class MolmoBlock(nn.Module):

    def __init__(self, layer_id: int, config: MolmoConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        input_dim = config.d_model
        self.attn_out = nn.Linear(
            input_dim, config.d_model,
            bias=config.include_bias,
            device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        if config.attention_type == "flash":
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        response_dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)

        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if self.config.use_position_ids and self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k, position_ids=position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key.to(k.device), k), dim=-2)
            v = torch.cat((past_value.to(v.device), v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if not self.config.use_position_ids and self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            response_dropout_p=0.0 if not self.training else self.config.response_attention_dropout,
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: 'FullMolmoConfig', cache: BufferCache):
        if config.moe_num_experts > 0:
            return MolmoeBlock(layer_id=layer_id, config=config, cache=cache)
        else:
            return MolmoSequentialBlock(layer_id, config, cache)


class MolmoeMLP(nn.Module):
    def __init__(self, config):
        from transformers.activations import ACT2FN
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.hidden_size = (
                               config.mlp_hidden_size if config.mlp_hidden_size is not None \
                                   else config.mlp_ratio * config.d_model
                           ) // 2
        self.gate_proj = nn.Linear(self.d_model, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.hidden_size, bias=False)
        self.down_proj = nn.Linear(self.hidden_size, self.d_model, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MolmoeMlpExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.gate = nn.Linear(config.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MolmoeMLP(config) for _ in range(self.num_experts)])
        self.ff_norm = RMSLayerNorm(
            config,
            size=config.d_model,
            eps=config.layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ff_norm(hidden_states)
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class MolmoeAttention(nn.Module):
    def __init__(
        self,
        config: MolmoConfig,
        cache: BufferCache
    ):
        super().__init__()
        self.config = config
        self.__cache = cache
        self.rotary_emb = RotaryEmbedding(config, self.__cache)
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None \
                else config.mlp_ratio * config.d_model
        )

        if config.attention_layer_norm:
            if config.n_kv_heads is None:
                config.n_kv_heads = config.n_heads
            self.q_norm = RMSLayerNorm(
                config,
                size=config.d_model,
                eps=config.layer_norm_eps
            )
            self.k_norm = RMSLayerNorm(
                config,
                size=config.d_model,
                eps=config.layer_norm_eps
            )

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function
        self.act = SwiGLU()
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        input_dim = config.d_model
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.n_kv_heads * head_dim,
            config.n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims),
            bias=config.include_bias or config.qkv_bias,
            device=config.init_device
        )
        self.attn_out = nn.Linear(
            input_dim, config.d_model,
            bias=config.include_bias,
            device=config.init_device
        )
        self.attn_norm = RMSLayerNorm(
            config,
            size=config.d_model,
            eps=config.layer_norm_eps)

    def attention(self,
                  q: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor,
                  attention_bias: Optional[torch.Tensor] = None,
                  position_ids: Optional[torch.Tensor] = None,
                  drop_mask: Optional[torch.Tensor] = None,
                  layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                  use_cache: bool = False,
                  ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if self.config.use_position_ids and self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k, position_ids=position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key.to(k.device), k), dim=-2)
            v = torch.cat((past_value.to(v.device), v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if not self.config.use_position_ids and self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            drop_mask=drop_mask,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            response_dropout_p=0.0 if not self.training else self.config.response_attention_dropout,
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        response_dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)

        # if self.flash_attn_func is not None and attn_mask is None:
        #     raise NotImplementedError()
        #     r = self.flash_attn_func(
        #         q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
        #     )
        #     return r.transpose(1, 2)
        # else:
        # torch's sdpa doesn't support GQA, so we're doing this
        assert k.size(1) == v.size(1)
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    def forward(
        self,
        x,
        attention_bias,
        position_ids,
        drop_mask,
        layer_past,
        use_cache
    ):
        if not self.config.norm_after:
            atten_in = self.attn_norm(x)
        else:
            atten_in = x

        qkv = self.att_proj(atten_in)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        att, cache = self.attention(
            q, k, v,
            attention_bias,
            position_ids=position_ids,
            drop_mask=drop_mask,
            layer_past=layer_past,
            use_cache=use_cache
        )

        if self.config.norm_after:
            att = self.attn_norm(att)

        return att, cache


class MolmoeBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """
    def __init__(
        self,
        layer_id: int,
        config: MolmoConfig,
        cache: BufferCache
    ):
        super().__init__()
        self.attn = MolmoeAttention(config, cache)
        assert config.moe_num_experts > 0
        self.mlp = MolmoeMlpExpert(config)
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get query, key, value projections.
        shape:
            for regular attn q, k, v: (batch_size, seq_len, d_model)
            for multi-query attn q: (batch_size, seq_len, d_model)
                              k, v: (batch_size, seq_len, d_model // n_heads)
            for group query attn q: (batch_size, seq_len, d_model)
                            k, v: (batch_size, seq_len, d_model // n_kv_heads)
        """

        att, cache = self.attn(
            x,
            attention_bias=attention_bias,
            position_ids=position_ids,
            drop_mask=drop_mask,
            layer_past=layer_past,
            use_cache=use_cache
        )
        x = x + self.dropout(att)
        og_x = x
        x, _ = self.mlp(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class MolmoSequentialBlock(MolmoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: MolmoConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)

        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims),
            bias=config.include_bias or config.qkv_bias,
            device=config.init_device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                atten_in = self._activation_checkpoint_fn(self.attn_norm, x)
            else:
                atten_in = self.attn_norm(x)
        else:
            atten_in = x
        qkv = self.att_proj(atten_in)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x

        return x, cache


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        device: Union[str, torch.device],
        initializer_range: float = 0.02,
        new_embed_initializer_range: float = 0.02,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.new_embed_initializer_range = new_embed_initializer_range
        self.embedding = nn.Parameter(
            torch.zeros(num_embeddings, features, device=device),
        )
        self.new_embedding = nn.Parameter(
            torch.zeros(num_new_embeddings, features, device=device),
        )

    def reset_parameters(self):
        nn.init.normal_(self.embedding, std=self.initializer_range)
        nn.init.normal_(self.new_embedding, std=self.new_embed_initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


class Dropout(nn.Dropout):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        mask_p: float = 0,
        broadcast_dims: Sequence[int] = (),
    ):
        super().__init__(p, inplace)
        self.mask_p = mask_p
        self.broadcast_dims = broadcast_dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input: A tensor of shape `(batch_size, seq_len, embed_dim)`
        """
        if self.p == 0.0 and (self.mask_p is None or self.mask_p == 0.0):
            return input
        else:
            if self.p > 0. and len(self.broadcast_dims) > 0 and self.training:
                keep_prob = 1.0 - self.p
                dropout_shape = list(input.shape)
                for dim in self.broadcast_dims:
                    dropout_shape[dim] = 1
                keep = input.new_empty(dropout_shape).bernoulli_(keep_prob)
                multiplier = keep.broadcast_to(input.shape)
                multiplier.div_(keep_prob)
                input = input * multiplier
            else:
                return F.dropout(input, self.p, self.training, self.inplace)


@dataclass
class VisionBackboneConfig:
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 24
    image_head_dim: int = 64
    image_mlp_dim: int = 4096
    image_mlp_activations: str = "gelu"
    image_dropout_rate: float = 0.0
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    fsdp_wrap: bool = False
    resize_mode: str = "default"

    def __post_init__(self):
        self.image_default_input_size = tuple(self.image_default_input_size)  # type: ignore[assignment]

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class FullMolmoConfig:
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    qkv_bias: bool = False
    clip_qkv: Optional[float] = None
    n_layers: int = 12
    mlp_ratio: int = 4
    mlp_hidden_size: Optional[int] = None
    activation_type: str = "swiglu"
    block_group_size: int = 1
    rope: bool = True
    rope_full_precision: bool = True
    rope_theta: float = 10000.
    rope_impl: str = "interleave"
    vision_backbone: Optional[VisionBackboneConfig] = None
    attention_type: str = "sdpa"
    float32_attention: bool = True
    attention_dropout: float = 0.1
    response_attention_dropout: float = 0.0
    multi_query_attention: Optional[bool] = None
    attention_layer_norm: bool = False
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    layer_norm_type: str = "default"
    layer_norm_with_affine: bool = True
    layer_norm_eps: Optional[float] = None
    attention_layer_norm_with_affine: bool = True
    max_sequence_length: int = 1024
    max_position_embeddings: Optional[int] = None
    include_bias: bool = True
    bias_for_layer_norm: Optional[bool] = None
    scale_logits: bool = False
    vocab_size: int = 50257
    embedding_size: Optional[int] = 50304
    additional_vocab_size: Optional[int] = None
    new_embedding_init_range: float = 0.02
    weight_tying: bool = True
    pad_token_id: int = -1
    init_device: Optional[str] = None
    init_std: float = 0.02
    init_cutoff_factor: Optional[float] = None
    norm_after: bool = False
    precision: Optional[str] = None
    image_padding_embed: Optional[str] = None
    vit_layers: Tuple = (-1,)
    image_pooling_h: int = 2
    image_pooling_w: int = 2
    image_pooling_2d: str = "attention"
    image_projector: str = "mlp"
    image_feature_dropout: float = 0.0
    initializer_range: float = 0.02
    normalize_input_embeds: bool = False
    use_position_ids: bool = True
    moe_num_experts: int = None
    moe_top_k: int = None

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise MolmoConfigurationError(
                    "You can't set `multi_query_attention` and `n_kv_heads` at the same time."
                )

    @property
    def image_num_patch(self):
        assert self.vision_backbone is not None
        return self.vision_backbone.image_num_patch

    @property
    def image_patch_size(self):
        assert self.vision_backbone is not None
        return self.visoin_backbone.image_patch_size

    def llm_patches_per_crop(self):
        h, w = self.image_num_patch
        # Round up in case we need to pad the image features for pooling
        h = (h + self.image_pooling_h - 1) // self.image_pooling_h
        w = (w + self.image_pooling_w - 1) // self.image_pooling_w
        return h, w


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class ViTMLP(nn.Module):
    def __init__(self, config: FullMolmoConfig):
        super().__init__()
        self.config = config
        v_cfg = config.vision_backbone

        self.w1 = nn.Linear(
            v_cfg.image_emb_dim,
            v_cfg.image_mlp_dim,
            bias=True,
            device=config.init_device,
        )
        # Activation function.
        cfg = deepcopy(config)
        cfg.activation_type = v_cfg.image_mlp_activations
        self.act = Activation.build(cfg)
        self.w2 = nn.Linear(
            v_cfg.image_mlp_dim,
            v_cfg.image_emb_dim,
            bias=True,
            device=config.init_device,
        )

    def reset_parameters(self):
        v_cfg = self.config.vision_backbone
        nn.init.trunc_normal_(self.w1.weight, std=math.sqrt(1 / v_cfg.image_emb_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=math.sqrt(1 / v_cfg.image_mlp_dim), a=-2.0, b=2.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class ResidualAttentionBlock(nn.Module):

    def __init__(self, config: FullMolmoConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        self.attention = MultiHeadDotProductAttention(config)
        self.feed_forward = ViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )
        self.ffn_norm = nn.LayerNorm(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
            device=config.init_device,
        )

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class BlockCollection(nn.Module):

    def __init__(self, config: FullMolmoConfig):
        super().__init__()
        self.config = config
        self.grad_checkpointing: bool = False

        v_cfg = config.vision_backbone
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(config) for _ in range(v_cfg.image_num_layers)
        ])

    def reset_parameters(self):
        for r in self.resblocks:
            r.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            x = r(x)
            hidden_states.append(x)
        return hidden_states


class LayerNormFp32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight.to(torch.float32),
                         self.bias.to(torch.float32), self.eps)
        return x.to(orig_type)


class VisionTransformer(nn.Module):

    def __init__(self, config: FullMolmoConfig):
        super().__init__()
        self.config = config

        v_cfg = config.vision_backbone
        # class embeddings and positional embeddings
        self.scale = v_cfg.image_emb_dim ** -0.5
        self.class_embedding = nn.Parameter(
            torch.zeros(v_cfg.image_emb_dim, device=config.init_device),
        )
        self.num_prefix_tokens: int = 1
        self.positional_embedding = nn.Parameter(
            torch.zeros(v_cfg.image_num_pos, v_cfg.image_emb_dim, device=config.init_device),
        )

        image_patch_size = v_cfg.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            v_cfg.image_emb_dim,
            bias=False,
            device=config.init_device,
            )

        self.pre_ln = LayerNormFp32(
            v_cfg.image_emb_dim,
            eps=v_cfg.image_norm_eps,
        )

        self.transformer = BlockCollection(config)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def reset_parameters(self):
        nn.init.normal_(self.class_embedding, std=self.scale)
        nn.init.normal_(self.positional_embedding, std=self.scale)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.pre_ln.reset_parameters()
        self.transformer.reset_parameters()

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        pos_emb = pos_emb.reshape(
            (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + torch.cat([cls_emb[None, :, :], pos_emb[None, :, :]], dim=1).to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.vision_backbone.image_num_patch
        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = self.add_pos_emb(x, patch_num)

        x = self.pre_ln(x)

        hidden_states = self.transformer(x)
        return hidden_states


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, config: FullMolmoConfig, use_bias: bool = True, is_vit_layer: Optional[bool] = True):
        super().__init__()
        self.config = config
        self.use_bias = use_bias

        v_cfg = config.vision_backbone
        self.embed_dim = v_cfg.image_emb_dim
        self.num_heads = v_cfg.image_num_heads
        self.head_dim = v_cfg.image_head_dim
        self.num_key_value_heads = v_cfg.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.initializer_range = v_cfg.initializer_range
        self.is_vit_layer = is_vit_layer

        nlayers = 1 if (is_vit_layer or config.vit_layers is None) else len(config.vit_layers)

        self.wq = nn.Linear(
            nlayers * self.embed_dim,
            self.num_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.wk = nn.Linear(
            nlayers * self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.wv = nn.Linear(
            nlayers * self.embed_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            self.embed_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.attention_dropout: Optional[Dropout] = None
        if v_cfg.attention_dropout > 0:
            self.attention_dropout = Dropout(v_cfg.attention_dropout, broadcast_dims=(0, 1))
        self.residual_dropout = Dropout(v_cfg.residual_dropout)

    def reset_parameters(self):
        nn.init.normal_(self.wq.weight, std=self.initializer_range)
        nn.init.normal_(self.wk.weight, std=self.initializer_range)
        nn.init.normal_(self.wv.weight, std=self.initializer_range)
        nn.init.normal_(self.wo.weight, std=self.initializer_range)
        if self.use_bias:
            nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wk.bias, 0)
            nn.init.constant_(self.wv.bias, 0)
            nn.init.constant_(self.wo.bias, 0)

    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(self, inputs_q: torch.Tensor, inputs_kv: Optional[torch.Tensor] = None) -> torch.Tensor:

        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        og_dtype = xq.dtype

        if self.config.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)

        if self.config.attention_type == "direct":
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            if self.attention_dropout is not None:
                attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)

        elif self.config.attention_type == "sdpa":
            if self.config.float32_attention and not torch.is_autocast_enabled():
                xv = xv.to(torch.float32)
            attn_output = F.scaled_dot_product_attention(
                xq.transpose(1, 2).contiguous(),
                xk.transpose(1, 2).contiguous(),
                xv.transpose(1, 2).contiguous(),
                is_causal=False,
                dropout_p=self.config.vision_backbone.attention_dropout
            ).transpose(1, 2)
        else:
            raise NotImplementedError(self.config.attention_type)
        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class MultiHeadAttentionPool(nn.Module):
    def __init__(
        self,
        config: FullMolmoConfig,
        factor: int = 1,
        use_bias: bool = True,
        dropout: bool = True,
        output_layer: bool = True,
        mean_residual: bool = False,
        query: str = "mean",
        is_vit_layer: Optional[bool] = True
    ):
        super().__init__()
        self.config = config
        self.factor = factor
        self.use_bias = use_bias
        self.dropout = dropout
        self.output_layer = output_layer
        self.mean_residual = mean_residual
        self.query = query

        v_cfg = config.vision_backbone
        input_dim = v_cfg.image_emb_dim
        self.embed_dim = v_cfg.image_emb_dim * factor
        self.num_heads = v_cfg.image_num_heads
        self.head_dim = v_cfg.image_head_dim * factor
        self.num_key_value_heads = v_cfg.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.initializer_range = v_cfg.initializer_range

        nlayers = 1 if (is_vit_layer or config.vit_layers is None) else len(config.vit_layers)

        if query != "vector":
            self.wq = nn.Linear(
                nlayers * input_dim,
                self.num_heads * self.head_dim,
                bias=use_bias,
                device=config.init_device,
                )
        self.wk = nn.Linear(
            nlayers * input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
            )
        self.wv = nn.Linear(
            nlayers * input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=config.init_device,
            )

        if query == "vector":
            self.attention_query = nn.Parameter(
                torch.zeros(
                    1, self.num_key_value_heads * self.head_dim, device=config.init_device,
                       ),
            )

        if output_layer:
            self.wo = nn.Linear(
                self.num_heads * self.head_dim,
                self.embed_dim,
                bias=use_bias,
                device=config.init_device,
                )
        self.attention_dropout = Dropout(v_cfg.attention_dropout, broadcast_dims=(0, 1))
        if dropout:
            self.residual_dropout = Dropout(v_cfg.residual_dropout)

    def reset_parameters(self):
        if self.query != "vector":
            nn.init.normal_(self.wq.weight, std=self.initializer_range)
        nn.init.normal_(self.wk.weight, std=self.initializer_range)
        nn.init.normal_(self.wv.weight, std=self.initializer_range)
        if self.output_layer:
            nn.init.normal_(self.wo.weight, std=self.initializer_range)
        if self.use_bias:
            if self.query != "vector":
                nn.init.constant_(self.wq.bias, 0)
            nn.init.constant_(self.wk.bias, 0)
            nn.init.constant_(self.wv.bias, 0)
            if self.output_layer:
                nn.init.constant_(self.wo.bias, 0)
        if self.query == "vector":
            nn.init.normal_(self.attention_query, std=self.initializer_range)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(self, inputs_kv: torch.Tensor) -> torch.Tensor:

        xk, xv = self.wk(inputs_kv), self.wv(inputs_kv)

        if self.query == "mean":
            inputs_q = inputs_kv.mean(dim=1, keepdim=True)
            xq = self.wq(inputs_q)
        elif self.query == "first":
            inputs_q = inputs_kv[:, :1]
            xq = self.wq(inputs_q)
        elif self.query == "vector":
            xq = self.attention_query.expand(inputs_kv.size(0), -1, -1)
        elif self.query == "constant":
            inputs_q = torch.ones_like(inputs_kv[:, :1]) / math.sqrt(inputs_kv.shape[-1])
            xq = self.wq(inputs_q)
        else:
            raise ValueError(f"Unknown query type: {self.query}")

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        xq = xq.to(torch.float)
        xk = xk.to(torch.float)

        xq = xq / math.sqrt(xq.size(-1))
        attn_weights = torch.einsum("...qhd,...khd->...hqk", xq, xk)

        attn_weights = F.softmax(attn_weights, dim=-1).to(xq.dtype)

        attn_weights = self.attention_dropout(attn_weights).to(xv.dtype)

        attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, xv)
        attn_output = self._merge_heads(attn_output)
        if self.output_layer:
            attn_output = self.wo(attn_output)
        if self.dropout:
            attn_output = self.residual_dropout(attn_output)
        if self.mean_residual:
            attn_output += inputs_kv.mean(dim=1, keepdim=True)

        return attn_output


class MLP(nn.Module):
    def __init__(self, config: FullMolmoConfig, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.initializer_range = config.initializer_range

        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=config.init_device,
            )
        self.w2 = nn.Linear(
            self.hidden_size // 2,
            config.d_model,
            bias=False,
            device=config.init_device,
            )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=config.init_device,
            )
        # Activation function.
        self.act = Activation.build(config)
        self.dropout = Dropout(dropout)

    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=self.initializer_range)
        nn.init.normal_(self.w2.weight, std=self.initializer_range)
        nn.init.normal_(self.w3.weight, std=self.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.act(self.w1(x), self.w3(x)))
        x = self.dropout(x)
        return x


class Residual(nn.Module):
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    def reset_parameters(self):
        self.submodule.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class OLMoVisionBackbone(nn.Module):
    def __init__(self, config: FullMolmoConfig):
        super().__init__()
        self.config = config
        self.image_vit = VisionTransformer(config)

        input_dim: int = None
        self.image_pooling_2d: nn.Module = None
        if config.image_pooling_2d in {ImagePooling2DType.attention, ImagePooling2DType.attention_meanq}:
            self.image_pooling_2d = MultiHeadDotProductAttention(config, is_vit_layer=False)
            input_dim = config.vision_backbone.image_emb_dim
        elif config.image_pooling_2d == ImagePooling2DType.attention_2wide:
            cfg = deepcopy(config)
            cfg.vision_backbone.image_emb_dim *= 2
            cfg.vision_backbone.image_head_dim *= 2
            self.image_pooling_2d = MultiHeadDotProductAttention(cfg, is_vit_layer=False)
            input_dim = cfg.vision_backbone.image_emb_dim
        elif config.image_pooling_2d == ImagePooling2DType.attention_v2:
            assert config.vit_layers is not None
            use_bias = True
            dropout = True
            output_layer = True
            query = "mean"
            mean_residual = False
            factor = len(config.vit_layers)
            self.image_pooling_2d = MultiHeadAttentionPool(
                config,
                factor=factor,
                use_bias=use_bias,
                dropout=dropout,
                output_layer=output_layer,
                mean_residual=mean_residual,
                query=query,
                is_vit_layer=False,
            )
            input_dim = config.vision_backbone.image_emb_dim * factor
        elif config.image_pooling_2d in [ImagePooling2DType.none, ImagePooling2DType.stack]:
            self.image_pooling_2d = None
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            input_dim = nlayers * config.vision_backbone.image_emb_dim
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.image_pooling_2d}")

        self.input_dim = input_dim

        # `MLP` assume the activation takes two inputs, so it must be a 'llama' version
        if config.activation_type == ActivationType.swiglu:
            mlp_config = replace(config, activation_type=ActivationType.llama_swiglu)
        elif config.activation_type == ActivationType.gelu:
            mlp_config = replace(config, activation_type=ActivationType.llama_geglu)
        else:
            mlp_config = config
        if config.image_projector == ImageProjectType.mlpx2:
            self.image_projector = nn.ModuleList(
                [MLP(mlp_config, input_dim), Residual(MLP(config, input_dim))]
            )
        elif config.image_projector == ImageProjectType.mlp:
            self.image_projector = MLP(mlp_config, input_dim)
        elif config.image_projector == ImageProjectType.linear:
            self.image_projector = nn.Linear(
                input_dim,
                config.d_model,
                bias=False,
                device=config.init_device,
            )
        else:
            raise NotImplementedError(f"Unknown image projector: {config.image_projector}")

        self.image_feature_dropout = Dropout(config.image_feature_dropout)

    def reset_parameters(self):
        if self.image_pooling_2d is not None:
            self.image_pooling_2d.reset_parameters()
        if self.config.image_projector == "2mlp":
            for module in self.image_projector:
                module.reset_parameters()
        elif self.config.image_projector == "linear":
            nn.init.xavier_uniform_(self.image_projector.weight)
        else:
            self.image_projector.reset_parameters()

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


class OLMoPretrainedVisionBackbone(OLMoVisionBackbone):
    def __init__(self, config: FullMolmoConfig):
        super().__init__(config)
        v_cfg = self.config.vision_backbone
        self.grad_checkpointing = False

        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {0, 1}, "Only 0 or 1 prefix tokens are supported"

        self.pad_embed = None
        if config.image_padding_embed:
            image_dim = v_cfg.image_emb_dim*len(self.config.vit_layers)
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(
                    torch.zeros((image_dim,), device=config.init_device))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(
                    torch.zeros((2, image_dim), device=config.init_device))
            else:
                raise ValueError(config.image_padding_embed)

    def reset_parameters(self):
        super().reset_parameters()
        self.image_vit.reset_parameters()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        v_cfg = self.config.vision_backbone
        B, T, N, D = images.shape

        mask = ~torch.all(images.view(B * T, N, D) == -1, dim=(1, 2), keepdim=True)

        # Output all hidden states
        # n_layers x (batch_num_crops, (1+)n_tokens, image_emb_dim)
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        if cfg.vit_layers is not None:
            features = []
            for layer in cfg.vit_layers:
                features.append(image_features[layer])
            image_features = torch.cat(features, dim=-1)
        else:
            image_features = image_features[-1]

        cls_embed: torch.Tensor = None
        if self.num_prefix_tokens > 0:
            cls_embed = image_features[:, 0]
            image_features = image_features[:, 1:]

        image_features = image_features * mask
        image_features = image_features.view(B, T, N, -1)

        cls_embed = cls_embed.view(B, T, -1) if cls_embed is not None else None

        return image_features, cls_embed

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        image_features, cls_embed = self.encode_image(images)

        if cfg.image_padding_embed:
            assert image_masks is not None
            if cfg.image_padding_embed == "pad_embed":
                all_pad = (image_masks == 0).to(dtype=torch.float32)
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
            elif cfg.image_padding_embed == "regress":
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(torch.maximum(image_masks, torch.zeros_like(image_masks)), -1)
            elif cfg.image_padding_embed == "pad_and_partial_pad":
                pad_embed = self.pad_embed[:, None, None, None, :]
                all_pad = image_masks == 0
                partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=image_features.dtype)
                all_pad = all_pad.to(dtype=image_features.dtype)
                image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
                image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
            else:
                raise ValueError(cfg.image_padding_embed)

        image_features = self.image_feature_dropout(image_features)
        if cls_embed is not None:
            cls_embed = self.image_feature_dropout(cls_embed)

        image_features = image_features.reshape(
            (batch_size, num_image) + cfg.image_num_patch + (-1,),
            )

        if cfg.image_num_patch[0] % cfg.image_pooling_h == 1:
            # Pad so we can still pool 2x2 patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
            )

        # image pooling
        image_features = einops.rearrange(
            image_features,
            'b n (h dh) (w dw) c -> (b n h w) (dh dw) c',
            dh=cfg.image_pooling_h,
            dw=cfg.image_pooling_w,
        )

        if cfg.image_pooling_2d == ImagePooling2DType.attention_meanq:
            query = image_features.mean(-2, keepdim=True)
            image_features = self.image_pooling_2d(query, image_features)
        elif cfg.image_pooling_2d not in {ImagePooling2DType.none, ImagePooling2DType.stack}:
            if self.grad_checkpointing:
                from torch.utils.checkpoint import checkpoint
                image_features = checkpoint(self.image_pooling_2d, image_features[:, :1, :], image_features, use_reentrant=False)
            else:
                image_features = self.image_pooling_2d(image_features[:, :1, :], image_features)

        h, w = cfg.llm_patches_per_crop()
        image_features = image_features.reshape(batch_size, num_image, h * w, -1)

        # MLP layer to map the feature.
        if self.grad_checkpointing:
            from torch.utils.checkpoint import checkpoint
            image_features = checkpoint(self.image_projector, image_features, use_reentrant=False)
        else:
            image_features = self.image_projector(image_features)

        # image_features: (batch_size, num_image, num_patch, d_model)
        # cls_embed: (batch_size, num_image, d_model)
        return image_features, cls_embed


class ModuleType(str, Enum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: FullMolmoConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    d = d if d is not None else config.d_model
    std = config.init_std * std_factor
    if config.init_cutoff_factor is not None:
        cutoff_value = config.init_cutoff_factor * std
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=std)


class LlamaSwiGLU(nn.Module):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return F.silu(x1) * x2

    @property
    def output_multiplier(self) -> float:
        return 0.5


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


class Activation(nn.Module):
    def __init__(self, config: FullMolmoConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: FullMolmoConfig) -> 'Activation':
        if config.activation_type == "quick_gelu":
            return QuickGELU(config)
        elif config.activation_type == "gelu":
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == "gelu_tanh":
            return cast(Activation, GELU(approximate="tanh"))
        elif config.activation_type == "relu":
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == "silu":
            return cast(Activation, SiLU(inplace=False))
        # elif config.activation_type == "llama_geglu":
        #     return LlamaGEGLU(config)
        # elif config.activation_type == "llama_geglu_tanh":
        #     return LlamaGEGLUTanh(config)
        elif config.activation_type == "llama_swiglu":
            return LlamaSwiGLU()
        elif config.activation_type == "swiglu":
            return SwiGLU()
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class QuickGELU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    @property
    def output_multiplier(self) -> float:
        return 1.0


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: MolmoConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
        weight_initializer: Optional[Callable] = torch.ones,
        bias_initializer: Optional[Callable] = torch.zeros,
    ):
        super().__init__()
        self.config = config
        self.eps = self.config.layer_norm_eps or eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(weight_initializer(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(bias_initializer(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @classmethod
    def build(cls, config: FullMolmoConfig, size: Optional[int] = None, **kwargs):
        if config.layer_norm_type == "default":
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == "low_precision":
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == "rms":
            return RMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: FullMolmoConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: FullMolmoConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class Molmo(nn.Module):
    def __init__(self, config: FullMolmoConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise MolmoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        wte = None
        if self.config.additional_vocab_size is not None:
            wte = Embedding(
                config.embedding_size or config.vocab_size,
                config.additional_vocab_size,
                config.d_model,
                device=config.init_device,
                initializer_range=config.initializer_range,
                new_embed_initializer_range=config.new_embedding_init_range
            )
        else:
            wte=nn.Embedding(
                config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
            )

        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [MolmoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.group_size > 1:
            raise NotImplementedError()
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not self.config.rope:
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                        )
                }
            )

        self.vision_backbone: Optional[OLMoVisionBackbone] = None
        if config.vision_backbone is not None:
            self.vision_backbone = OLMoPretrainedVisionBackbone(config)

        self.__num_fwd_flops: Optional[int] = None

    def reset_parameters(self):
        if self.vision_backbone is not None:
            self.vision_backbone.reset_parameters()
        self.reset_non_vision_parameters()

    def reset_non_vision_parameters(self):
        self.transformer.wte.reset_parameters()
        if hasattr(self.transformer.wte, "new_embedding"):
            nn.init.normal_(self.transformer.wte.new_embedding, std=self.config.new_embedding_init_range)

        if hasattr(self.transformer, "wpe"):
            nn.init.normal_(self.transformer.wpe, mean=0.0, std=1.0)

        self.transformer.ln_f.reset_parameters()  # type: ignore

        if hasattr(self.transformer, "ff_out"):
            nn.init.normal_(self.transformer.ff_out, mean=0.0, std=0.02)

        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()


    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param response_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            the response mask. A `1` value in the mask means that the corresponding token
            is a response token. A `0` means that the corresponding token is not
            a response token.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        has_image = images is not None

        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1

        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = (
                subsegment_mask.to(attention_mask.dtype) *
                attention_mask.unsqueeze(2) *
                attention_mask.unsqueeze(1))
            if position_ids is None:
                raise ValueError(f"Positioned ids must be given if using subsegment_ids")
        else:
            if self.config.use_position_ids and position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                    ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        num_image: Optional[int] = None
        if images is not None:
            # shape: (batch_size, num_image, num_patch, d_model)
            # cls_embed: (batch_size, num_image, d_model)
            image_features, cls_embed = self.vision_backbone(images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            assert image_input_idx.shape == (batch_size, num_image, num_patch)

            # inster the image feature into the embedding.
            image_features = image_features.view(batch_size, num_image * num_patch, -1)
            image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

            valid = image_input_idx >= 0
            batch_idx = torch.arange(batch_size, device=x.device)
            batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])

            # For hf demo/endpoint
            image_features = image_features.to(x.device)

            x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]

        if not self.config.rope:
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # normalized
        if self.config.normalize_input_embeds:
            x = x * (self.config.d_model ** 0.5)

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, :past_length + seq_len]
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            else:
                attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.float)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                x, cache = block(x, attention_bias=attention_bias, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache)

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                         group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                         ]
                )
                x, cache = block_group(
                    x, attention_bias=attention_bias, position_ids=position_ids, layers_past=layers_past, use_cache=use_cache
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        if not last_logits_only and append_last_valid_logits is not None:
            last_valid_logit = logits[
                torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
            logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)

        return ModelOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]


class MolmoForCausalLM(PreTrainedModel):
    config_class = MolmoConfig
    base_model_prefix = "model"
    _no_split_modules = ["MolmoBlock"]

    def __init__(self, config: MolmoConfig, model: Optional[Molmo] = None, init_params: bool = False):
        super().__init__(config)

        if not model:
            full_config = FullMolmoConfig(
                mlp_ratio=1 if config.moe_num_experts else 4,
                image_padding_embed="pad_and_partial_pad",
                image_pooling_2d="attention-meanq",
                attention_layer_norm=config.attention_layer_norm,
                rope_impl="llama",
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                qkv_bias=config.qkv_bias,
                norm_after=config.norm_after,
                embedding_size=config.embedding_size,
                attention_type="sdpa",
                embedding_dropout=0,
                attention_dropout=0,
                residual_dropout=0,
                rope=True,
                weight_tying=False,
                include_bias=False,
                d_model=config.hidden_size,
                mlp_hidden_size=config.intermediate_size,
                n_layers=config.num_hidden_layers,
                additional_vocab_size=128,
                n_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                rope_theta=config.rope_theta,
                layer_norm_eps=config.layer_norm_eps,
                layer_norm_type=config.layer_norm_type,
                vit_layers=[-2, -9],
                vision_backbone=VisionBackboneConfig(
                    image_default_input_size=(336, 336),
                    image_patch_size=14,
                    image_pos_patch_size=14,
                    image_emb_dim=1024,
                    image_num_heads=16,
                    image_num_key_value_heads=16,
                    image_num_layers=23,
                    image_head_dim=64,
                    image_mlp_dim=4096,
                    image_mlp_activations="quick_gelu",
                    image_dropout_rate=0.0,
                    image_num_pos=577,
                    image_norm_eps=1e-5,
                    attention_dropout=0.0,
                    residual_dropout=0.0,
                    initializer_range=0.02,
                ),
                moe_num_experts=config.moe_num_experts,
                moe_top_k=config.moe_top_k
            )
            self.model = Molmo(full_config, init_params=init_params)
        else:
            self.model = model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        last_logits_only: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[
            Cache
        ] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in Molmo")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            response_mask=response_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            subsegment_ids=subsegment_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            output_hidden_states=output_hidden_states,
            append_last_valid_logits=append_last_valid_logits,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            if loss_masks is not None:
                loss_masks = loss_masks * (loss_masks > 0)
                batch_size_in_tokens = max(loss_masks.sum().item(), 1)
                labels = labels.long()
                labels.masked_fill_(~(loss_masks > 0), -100)
                labels = labels.view(-1)
                logits_for_loss = logits.to(torch.float32).view(-1, logits.size(-1))
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(logits_for_loss, labels)
                loss = loss.view(input_ids.shape[0], -1)
                loss = loss * loss_masks
                loss = loss.sum() / batch_size_in_tokens
                use_zloss = getattr(self.config, "softmax_auxiliary_loss", False)
                if use_zloss:
                    z_squared = logits_for_loss.logsumexp(-1).pow(2)
                    z_loss = self.config.softmax_auxiliary_loss_scale * z_squared
                    z_loss = z_loss.view(input_ids.shape[0], -1)
                    z_loss = z_loss * loss_masks
                    z_loss = z_loss.sum() / batch_size_in_tokens
                    loss += z_loss
            else:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.embedding_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    @torch.no_grad()
    def generate_from_batch(
        self,
        batch: Dict[str, Any],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        if generation_config is not None:
            assert generation_config.use_cache

        images = batch.get("images")
        image_masks = batch.get("image_masks")
        image_input_idx = batch.get("image_input_idx")

        # Validate inputs.
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        attention_mask = batch.get("attention_mask", None)
        max_new_tokens = generation_config.max_new_tokens
        assert max_new_tokens is not None
        mask_len = seq_len + max_new_tokens if self.config.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)

        out = super().generate(
            batch["input_ids"],
            generation_config,
            attention_mask=attention_mask,
            images=images,
            image_masks=image_masks,
            image_input_idx=image_input_idx,
            position_ids=position_ids,
            append_last_valid_logits=append_last_valid_logits,
            **kwargs,
        )

        return out

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]

        if self.config.use_position_ids:
            attention_mask = kwargs.get("attention_mask")
            images = kwargs.get("images")
            image_masks = kwargs.get("image_masks")
            image_input_idx = kwargs.get("image_input_idx")
            position_ids = kwargs.get("position_ids")
            append_last_valid_logits = kwargs.get("append_last_valid_logits")
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "last_logits_only": True,
            }
            if past_key_values is None:
                model_inputs["images"] = images
                model_inputs["image_masks"] = image_masks
                model_inputs["image_input_idx"] = image_input_idx
                model_inputs["append_last_valid_logits"] = append_last_valid_logits
        else:
            model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

            model_inputs.update(kwargs)
            model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        if self.config.use_position_ids:
            model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
            if "append_last_valid_logits" in model_kwargs:
                del model_kwargs["append_last_valid_logits"]
            if "images" in model_kwargs:
                del model_kwargs["images"]
                del model_kwargs["image_masks"]
                del model_kwargs["image_input_idx"]
        cache_name, cache = super()._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        return model_kwargs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        """
        This function is intentionally left as a no-op.

        Weight tying is handled as follows:
        - When the model is initialized, the `ff_out` layer is conditionally defined based on the `weight_tying` configuration.
        See: `if not config.weight_tying: self.transformer.update(...)` in `olmo/model.py`.
        - When computing logits, the `wte` weights are used directly if `weight_tying` is enabled.
        See: `if self.config.weight_tying: logits = F.linear(x, self.transformer.wte.weight, None)` in the `forward` method.

        Therefore, there is no need to explicitly tie the weights in this function.
        """
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> torch.nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.embedding_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.

        Note:
            This method differs from the base class implementation by resizing the `embedding_size` attribute of the
            model configuration instead of the `vocab_size`. It also includes a warning if the resized `embedding_size`
            is less than the `vocab_size`. In OLMo, `embedding_size` refers to the dimensionality of the model's token
            embeddings, while `vocab_size` refers to the number of unique tokens in the vocabulary.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.embedding_size = model_embeds.weight.shape[0]
        self.model.config.embedding_size = model_embeds.weight.shape[0]

        # Check if the embedding size is less than the vocab size
        if self.config.embedding_size < self.config.vocab_size:
            warning_message = (
                f"Resizing token embeddings to size {self.config.embedding_size}, which is less than the vocab size "
                f"{self.config.vocab_size} defined in the model configuration. Make sure your tokenizer's vocabulary "
                "size is less than or equal to the new token embedding size."
            )
            log.warning(warning_message)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds


# Always register for multi-modal features
AutoModelForCausalLM.register(MolmoConfig, MolmoForCausalLM)