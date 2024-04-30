"""PyTorch Granite model."""

import math
import numbers
import warnings
from enum import Enum
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import get_activation as get_base_activation
from ...cache_utils import DynamicCache, StaticCache, Cache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    is_flash_attn_2_available,
    logging,
)
from .configuration_granite import GraniteConfig
from ...modeling_attn_mask_utils import AttentionMaskConverter


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)


class PositionEmbeddingType(Enum):
    learned_absolute = "learned_absolute"
    alibi = "alibi"
    rope = "rope"


class AttentionHeadType(Enum):
    mha = "mha"
    mqa = "mqa"
    gqa = "gqa"


_CONFIG_FOR_DOC = "GraniteConfig"


def get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def repeat_key_value(x: torch.Tensor, num_heads: int, num_key_value_heads: int) -> torch.Tensor:
    num_groups = num_heads // num_key_value_heads

    # mha
    if num_groups == 1:
        return x

    # mqa
    if num_key_value_heads == 1:
        return x.expand(-1, num_heads, -1, -1)

    # gqa
    return x.repeat_interleave(num_groups, dim=1)


_GLU_BASE_MAPPING = {
    "geglu": "gelu",
    "miglu": "mish",
    "mishglu": "mish",
    "swiglu": "swish",
}


class GLUActivation(nn.Module):
    def __init__(self, base_activation: nn.Module) -> None:
        super().__init__()
        self.base_activation = base_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.chunk(2, dim=-1)
        return x[0] * self.base_activation(x[1])


def is_glu(name: str) -> bool:
    return name.endswith("glu")


def get_activation_function(name: str) -> nn.Module:
    if is_glu(name):
        # for glu and sigmoid_glu, we directly return the pytorch's GLU
        if name in ["glu", "sigmoid_glu"]:
            activation_function = nn.modules.GLU()
        else:
            if name in _GLU_BASE_MAPPING:
                name = _GLU_BASE_MAPPING[name]
            elif name.endswith("_glu"):
                name = name.rstrip("_glu")
            else:
                raise ValueError("invalid activation function")

            base_activation = get_base_activation(name)
            activation_function = GLUActivation(base_activation)
    else:
        activation_function = get_base_activation(name)

    return activation_function


class GraniteRMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype

        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)

        return self.weight * input.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


ALL_LAYERNORM_LAYERS.append(GraniteRMSNorm)


_NORMALIZATION_FUNCTIONS = {
    "layernorm": nn.LayerNorm,
    "rmsnorm": GraniteRMSNorm,
}


def get_normalization_function(name: str, normalized_shape: int, eps: float = 1e-5) -> nn.Module:
    if name in _NORMALIZATION_FUNCTIONS:
        return _NORMALIZATION_FUNCTIONS[name](normalized_shape, eps=eps)

    raise ValueError(f"unexpected `normalization_function` {name}")


class GraniteAttention(nn.Module):
    def __init__(self, config: GraniteConfig, causal: bool, layer_idx: Optional[int] = None) -> None:
        super().__init__()

        self.causal = causal
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.num_key_value_heads = config.num_key_value_heads
        self.add_bias = config.add_bias

        assert (
            self.hidden_size % self.num_heads == 0
        ), f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.hidden_size // self.num_heads
        self.attention_head_type = AttentionHeadType(config.attention_head_type)

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)
        self.scale_attn_weights = config.scale_attn_weights
        self.attention_multiplier = config.attention_multiplier

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )

        if self.attention_head_type == AttentionHeadType.mha:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.num_heads

            assert (
                self.num_heads == self.num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"
        elif self.attention_head_type == AttentionHeadType.gqa:
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert self.num_heads % self.num_key_value_heads == 0, (
                f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` "
                f"({self.num_key_value_heads})"
            )
        elif self.attention_head_type == AttentionHeadType.mqa:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, f"{self.__class__.__name__} should have 1 head for keys and values"
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        self.c_attn = nn.Linear(
            self.hidden_size, self.hidden_size + 2 * self.num_key_value_heads * self.head_dim, bias=self.add_bias
        )
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.add_bias)

        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop

        self.attn_dropout = nn.Identity() if self.attn_pdrop == 0 else nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Identity() if self.resid_pdrop == 0 else nn.Dropout(self.resid_pdrop)

    def _prepare_qkv_for_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # the output of following is a tuple if using MQA with tensor parallel
        hidden_states = self.c_attn(hidden_states)

        # for MHA, we can get away with doing just 1 transpose which is not true for GQA
        if self.attention_head_type == AttentionHeadType.mha:
            query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
        elif self.attention_head_type == AttentionHeadType.gqa:
            query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
        elif self.attention_head_type == AttentionHeadType.mqa:
            query, key, value = self._prepare_qkv_for_forward_mqa(hidden_states)
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        return query, key, value

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_heads, -1)
        hidden_states = hidden_states.transpose(1, 2)

        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_key_value_heads, -1)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(batch_size, query_length, -1, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        query, key, value = hidden_states.split((self.hidden_size, self.head_dim, self.head_dim), dim=-1)

        query = query.view(batch_size, query_length, self.num_heads, -1)

        query = query.transpose(1, 2)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        key = key.transpose(-1, -2)

        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype

        if self.scale_attn_weights:
            if self.attention_multiplier is None:
                scale_factor = 1 / self.head_dim**0.5
            else:
                scale_factor = self.attention_multiplier
        else:
            scale_factor = 1

        batch_size = query.shape[0]
        query_length = query.shape[2]
        key_length = key.shape[-1]

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # Always copies
        query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
        # No copy when layer_past is provided.
        key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        if attention_mask is None:
            attn_weights = torch.empty(
                (batch_size * self.num_heads, query_length, key_length), device=query.device, dtype=query.dtype
            )
            beta = 0
        else:
            attn_weights = attention_mask.expand(-1, self.num_heads, -1, -1).reshape(-1, query_length, key_length)
            beta = 1

        attn_weights = torch.baddbmm(attn_weights, query, key, beta=beta, alpha=scale_factor).view(
            batch_size, self.num_heads, query_length, key_length
        )

        attn_weights = F.softmax(attn_weights.to(softmax_dtype), dim=-1).to(dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class GraniteSDPA(GraniteAttention):
    """
    Granite attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GraniteAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_pdrop if self.training else 0,
            is_causal=self.causal if attention_mask is None else False,
            scale=self.attention_multiplier if self.scale_attn_weights else 1,
        )

        batch_size = attn_output.shape[0]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class GraniteFlashAttention2(GraniteAttention):
    """
    Granite flash attention module. This module inherits from `GraniteAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # TODO avoid this extra transpose
        query = query.transpose(1, 2)
        if self.attention_head_type == AttentionHeadType.mqa:
            key = key.squeeze(1).unsqueeze(2)
            value = value.squeeze(1).unsqueeze(2)
        else:
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        batch_size, query_length = query.shape[:2]
        key_length = key.shape[1]
        indices_k, cu_seqlens_k, max_seqlen_k = get_unpad_data(attention_mask)

        key = index_first_axis(
            key.reshape(batch_size * key_length, self.num_key_value_heads, self.head_dim), indices_k
        )
        value = index_first_axis(
            value.reshape(batch_size * key_length, self.num_key_value_heads, self.head_dim), indices_k
        )

        if query_length == key_length:
            query = index_first_axis(
                query.reshape(batch_size * key_length, self.num_heads, self.head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query = query.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query, attention_mask)

        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=self.attn_pdrop if self.training else 0,
            softmax_scale=self.attention_multiplier if self.scale_attn_weights else 1,
            causal=self.causal,
        )

        attn_output = pad_input(attn_output, indices_q, batch_size, query_length)
        attn_output = attn_output.view(batch_size, query_length, -1)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


_ATTENTION_MODULES = {
    "eager": GraniteAttention,
    "sdpa": GraniteSDPA,
    "flash_attention_2": GraniteFlashAttention2,
}


def get_attention_module(
    config: GraniteConfig, causal: bool, attention_implementation: str, layer_idx: int
) -> GraniteAttention:
    if attention_implementation in _ATTENTION_MODULES:
        return _ATTENTION_MODULES[attention_implementation](config, causal=causal, layer_idx=layer_idx)
    raise ValueError(f"unexpected `attention_implementation` {attention_implementation}")


class Alibi(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads

        self.reset_parameters()

    def forward(
        self, attention_mask: torch.Tensor, batch_size: int, key_length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
        relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
        `softmax(l+a) = softmax(l)`. Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

        Args:
            attention_mask (torch.Tensor): attention_mask tensor of shape (`batch_size`, `key_length`)
            num_heads (int): `num_heads` for the model
            batch_size (int): `batch_size`
            key_length (int): `key_length`
            device (torch.device): device for the tensors
            dtype (torch.dtype): dtype to use for the tensors

        Returns:
            torch.Tensor: alibi tensor of shape (`batch_size`, `num_heads`, `key_length`)
        """

        # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
        # => the query_length dimension will then be broadcasted correctly
        # This is more or less identical to T5's relative position bias:
        # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
        if attention_mask is None:
            arange_tensor = (
                torch.arange(key_length, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            )
        else:
            arange_tensor = (attention_mask.cumsum(dim=-1) - 1).masked_fill_(attention_mask == 0, 0).unsqueeze(1)

        alibi = self.slopes.unsqueeze(1) * arange_tensor
        return alibi.to(dtype)

    def reset_parameters(self) -> None:
        closest_power_of_2 = 2 ** math.floor(math.log2(self.num_heads))
        base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != self.num_heads:
            extra_base = torch.tensor(2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, self.num_heads - closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

        self.register_buffer("slopes", slopes, persistent=False)


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mscale = 1

        self.reset_parameters()

    def forward(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        cos = self.cos_cached[:seq_len].to(dtype)
        sin = self.sin_cached[:seq_len].to(dtype)

        return cos, sin

    def reset_parameters(self) -> None:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    @torch.no_grad()
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)


def apply_rotary_pos_emb(x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = cos_sin
    x = (x * cos) + (_rotate_half(x) * sin)
    return x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class GraniteMLP(nn.Module):
    def __init__(self, config: GraniteConfig) -> None:
        super().__init__()

        hidden_size = config.n_embd
        intermediate_size = config.n_inner
        activation_function = config.activation_function
        add_bias = config.add_bias
        residual_dropout = config.resid_pdrop

        self.c_fc = nn.Linear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            bias=add_bias,
        )
        self.act = get_activation_function(activation_function)
        self.c_proj = nn.Linear(intermediate_size, hidden_size, bias=add_bias)
        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GraniteBlock(nn.Module):
    def __init__(
        self,
        config: GraniteConfig,
        attention_implementation: str,
        layer_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.layer_idx = layer_idx

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attn = get_attention_module(config, True, attention_implementation, layer_idx)
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.mlp = GraniteMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
        )

        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


GRANITE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GraniteConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Granite Model outputting raw hidden-states without any specific head on top.",
    GRANITE_START_DOCSTRING,
)
class GranitePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GraniteConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GraniteBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config: GraniteConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.attention_implementation = self.config._attn_implementation
        self._use_eager_attention = self.attention_implementation == "eager"
        self._use_sdpa = self.attention_implementation == "sdpa"
        self._use_flash_attention_2 = self.attention_implementation == "flash_attention_2"

        self.initializer_range = config.initializer_range

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.LayerNorm, GraniteRMSNorm, Alibi, RoPE)):
            module.reset_parameters()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight[module.padding_idx].zero_()


@add_start_docstrings(
    "The bare Granite Model outputting raw hidden-states without any specific head on top.",
    GRANITE_START_DOCSTRING,
)
class GraniteModel(GranitePreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]
    mask_value = None

    def __init__(self, config: GraniteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        assert (
            self.embed_dim % self.num_heads == 0
        ), f"`embed_dim` ({self.embed_dim}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.embed_dim // self.num_heads

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.drop = nn.Identity() if config.embd_pdrop == 0 else nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GraniteBlock(config, self.attention_implementation, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = get_normalization_function(
            config.normalization_function,
            self.embed_dim,
            eps=config.layer_norm_epsilon,
        )

        self.position_embedding_type = PositionEmbeddingType(config.position_embedding_type)

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            self.wpe = nn.Embedding(config.n_positions, self.embed_dim)
        elif self.position_embedding_type == PositionEmbeddingType.alibi:
            assert not self._use_flash_attention_2, "alibi is not implemented with FlashAttention"

            self.alibi = Alibi(self.num_heads)
        elif self.position_embedding_type == PositionEmbeddingType.rope:
            self.rope = RoPE(self.head_dim, max_position_embeddings=config.n_positions, base=config.rope_theta)
        else:
            raise NotImplementedError()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: torch.Tensor = None,
    ) -> Union[Tuple]:
        (
            output_hidden_states,
            use_cache,
            return_dict,
            input_shape,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
            cache_position,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        output_shape = input_shape + (hidden_states.size(-1),)

        past_key_values = DynamicCache() if use_cache and past_key_values is None else past_key_values
        all_hidden_states = () if output_hidden_states else None
        for block in self.h:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = block(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
            )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    def _get_position_ids(
        self, attention_mask: torch.Tensor, past_length: int, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        if attention_mask is not None and len(attention_mask.shape) == 2:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_length > 0:
                position_ids = position_ids[:, past_length:key_length:]
        else:
            position_ids = torch.arange(past_length, key_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, query_length)

        return position_ids

    def _get_alibi_bias(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.position_embedding_type != PositionEmbeddingType.alibi:
            return None

        alibi_bias = self.alibi(attention_mask, batch_size, key_length, device, dtype)

        alibi_bias = alibi_bias.unsqueeze(2)
        if query_length != 1:
            alibi_bias = alibi_bias.expand(-1, -1, query_length, -1)

        return alibi_bias

    def _get_rope_cos_sin(
        self, key_length: int, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.position_embedding_type == PositionEmbeddingType.rope:
            cos, sin = self.rope(key_length, dtype=dtype, device=device)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
            return cos, sin

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if self.config._attn_implementation == "sdpa":
            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
            # in order to dispatch on Flash Attention 2.
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.h[0], "attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _get_initial_hidden_state(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if self.position_embedding_type == PositionEmbeddingType.learned_absolute:
            inputs_embeds = inputs_embeds + self.wpe(position_ids)

        if token_type_ids is not None:
            inputs_embeds = inputs_embeds + self.wte(token_type_ids)

        inputs_embeds = self.drop(inputs_embeds)

        return inputs_embeds

    def _prepare_a_bunch_of_stuff(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
        return_dict: bool,
        cache_position: torch.Tensor,
    ) -> Tuple[
        bool,
        bool,
        bool,
        torch.Size,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        DynamicCache,
        torch.Tensor,
    ]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            # TODO special handling for padding free transformer needed here if we support inputs_embeds argument
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size = input_shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if self.position_embedding_type == PositionEmbeddingType.alibi:
            if position_ids is not None:
                warnings.warn("`position_ids` have no functionality with Alibi.", FutureWarning)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        query_length = input_shape[-1]
        key_length = past_length + query_length

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_length, past_length + input_ids.shape[1], device=input_ids.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = self._get_initial_hidden_state(input_ids, inputs_embeds, position_ids, token_type_ids)

        alibi_bias = self._get_alibi_bias(
            attention_mask, batch_size, query_length, key_length, device, hidden_states.dtype
        )

        rope_cos_sin = self._get_rope_cos_sin(
            key_length, position_ids, dtype=hidden_states.dtype, device=hidden_states.device
        )

        attention_mask = self._update_causal_mask(attention_mask, hidden_states, cache_position, past_length)

        if alibi_bias is not None:
            attention_mask = attention_mask + alibi_bias

        return (
            output_hidden_states,
            use_cache,
            return_dict,
            input_shape,
            hidden_states,
            attention_mask,
            position_ids,
            rope_cos_sin,
            past_key_values,
            cache_position,
        )


class GraniteForCausalLM(GranitePreTrainedModel):
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config: GraniteConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.transformer = GraniteModel(config, **kwargs)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer.wte = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.transformer.h[0], "attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[Union[torch.Tensor]] = None,
        past_key_values: Optional[DynamicCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[Union[torch.Tensor]] = None,
        position_ids: Optional[Union[torch.Tensor]] = None,
        inputs_embeds: Optional[Union[torch.Tensor]] = None,
        labels: Optional[Union[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: torch.Tensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        # Shift so that tokens < n predict n
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
