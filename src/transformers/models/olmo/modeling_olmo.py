"""PyTorch OLMo model."""

from __future__ import annotations

import math
import sys
import warnings
from abc import abstractmethod
from functools import partial
from os import PathLike
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import einsum, nn

from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_olmo import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    FSDPWrapStrategy,
    InitFnType,
    LayerNormType,
    OLMoConfig,
    StrEnum,
)


if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


PathOrStr = Union[str, PathLike]

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "OLMoConfig"


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: OLMoConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: OLMoConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default [`LayerNorm`] implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: OLMoConfig,
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
            downcast_weight = self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified [`LayerNorm`] implementation
    """

    def __init__(
        self,
        config: OLMoConfig,
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


ALL_LAYERNORM_LAYERS.append(LayerNormBase)


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: OLMoConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: OLMoConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

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
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
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

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: OLMoConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: OLMoConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


def activation_checkpoint_function(cfg: OLMoConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify `x` in place to replace `float("-inf")` with the minimum value of the dtype when `check_neg_inf`
    is `True` and to replace `float("inf")` with the maximum value of the dtype when `check_pos_inf` is `True`.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: OLMoConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        `1 / sqrt(2 * (layer_id + 1))`.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = config.init_std_mitchell * std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


class OLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: OLMoConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=config.d_model // config.n_heads if config.multi_query_attention else None,
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
        self.attn_out = nn.Linear(config.d_model, config.d_model, bias=config.include_bias, device=config.init_device)

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

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

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
        is_causal: bool = False,
        use_pytorch_sdpa: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.

        This method is based on PyTorch's `scaled_dot_product_attention`.
        """
        if use_pytorch_sdpa and output_attentions:
            logger.warning_once(
                "OLMoModel is using SDPA, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                "but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `use_pytorch_sdpa=False` in `OLMoConfig`."
            )
            use_pytorch_sdpa = False

        if use_pytorch_sdpa:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            ), None

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if is_causal:
            assert attn_mask is None

            query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
            attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
        elif attn_mask is not None:
            attn_bias = attn_mask.to(q.dtype)
        else:
            attn_bias = torch.zeros_like(attn_weights)

        attn_weights += attn_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)
        return torch.matmul(attn_weights, v), attn_weights if output_attentions else None

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_pytorch_sdpa: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        if self.config.multi_query_attention:
            # shape: (B, 1, T, hs)
            k = k.view(B, T, 1, C // self.config.n_heads).transpose(1, 2)
            # shape: (B, 1, T, hs)
            v = v.view(B, T, 1, C // self.config.n_heads).transpose(1, 2)
        else:
            # shape: (B, nh, T, hs)
            k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
            # shape: (B, nh, T, hs)
            v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype)

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att, attn_weights = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
            use_pytorch_sdpa=use_pytorch_sdpa,
            output_attentions=output_attentions,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present, attn_weights

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_pytorch_sdpa: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: OLMoConfig, cache: BufferCache) -> OLMoBlock:
        if config.block_type == BlockType.sequential:
            return OLMoSequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.parallel:
            return OLMoParallelBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return OLMoLlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class OLMoSequentialBlock(OLMoBlock):
    """
    This is a typical transformer block where the output is computed as `MLP(LN(x + Attention(LN(x))))`
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: OLMoConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            self.fused_dims = (config.d_model, config.d_model // config.n_heads, config.d_model // config.n_heads)
        else:
            self.fused_dims = (config.d_model, config.d_model, config.d_model)
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device)

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
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_pytorch_sdpa: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        if self._activation_checkpoint_fn is not None:
            qkv = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x))
        else:
            qkv = self.att_proj(self.attn_norm(x))

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache, attn_weights = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                use_pytorch_sdpa=use_pytorch_sdpa,
                output_attentions=output_attentions,
            )
        else:
            att, cache, attn_weights = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                use_pytorch_sdpa=use_pytorch_sdpa,
                output_attentions=output_attentions,
            )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
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
        x = self.dropout(x)
        x = og_x + x

        return x, cache, attn_weights


class OLMoParallelBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as `MLP(LN(x)) + Attention(LN(x))`
    as in the PaLM architecture, as opposed to the typical `MLP(LN(x + Attention(LN(x))))`
    as in [`OLMoSequentialBlock`] (ignoring some skip connections).

    The decoupling of the MLP and Attention functions allow us to fuse the separate input projections
    into a single linear layer to increase throughput. In this configuration it's also straight-forward
    to fuse the output projections, but we found that didn't help.
    """

    def __init__(self, layer_id: int, config: OLMoConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.norm = LayerNorm.build(config)
        # Fused attention and feed-forward projection.
        # NOTE: we could also fuse the attention and feed-forward output projections but we
        # found that didn't help, possibly because of the overhead of joining the `att` and
        # `ff` activations together. See https://github.com/allenai/LLM/pull/79 for details.
        if config.multi_query_attention:
            self.fused_dims = (
                config.d_model,
                config.d_model // config.n_heads,
                config.d_model // config.n_heads,
                self.hidden_size,
            )
        else:
            self.fused_dims = (config.d_model, config.d_model, config.d_model, self.hidden_size)
        self.fused_attn_ff_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config,
            self.fused_attn_ff_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_pytorch_sdpa: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        # Get query, key, value, and feed-forward projections.
        # shape of q, k, v:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        # shape of ff:      (batch_size, seq_len, hidden_size)
        if self._activation_checkpoint_fn is not None:
            q, k, v, ff = self.fused_attn_ff_proj(self._activation_checkpoint_fn(self.norm, x)).split(
                self.fused_dims, dim=-1
            )
        else:
            q, k, v, ff = self.fused_attn_ff_proj(self.norm(x)).split(self.fused_dims, dim=-1)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        # shape: (B, T, C)
        if self._activation_checkpoint_fn is not None:
            att, cache, attn_weights = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                use_pytorch_sdpa=use_pytorch_sdpa,
                output_attentions=output_attentions,
            )
        else:
            att, cache, attn_weights = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                use_pytorch_sdpa=use_pytorch_sdpa,
                output_attentions=output_attentions,
            )

        # Apply output projections (and activation function) and sum the results.
        # We keep these projections separate because we found that we got better throughput this
        # way compared to fusing them.
        if self._activation_checkpoint_fn is not None:
            return (
                x + self.dropout(self.ff_out(self._activation_checkpoint_fn(self.act, ff))) + self.dropout(att),
                cache,
                attn_weights,
            )
        else:
            return (
                x + self.dropout(self.ff_out(self.act(ff))) + self.dropout(att),
                cache,
                attn_weights,
            )


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


def alibi_attention_bias(seq_len: int, config: OLMoConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class OLMoLlamaBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as `MLP(LN(x + Attention(LN(x))))`
    (plus another skip connection). This block is similar to `OLMoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: OLMoConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model // config.n_heads
            v_proj_out_dim = config.d_model // config.n_heads
        else:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model
            v_proj_out_dim = config.d_model
        self.q_proj = nn.Linear(config.d_model, q_proj_out_dim, bias=config.include_bias, device=config.init_device)
        self.k_proj = nn.Linear(config.d_model, k_proj_out_dim, bias=config.include_bias, device=config.init_device)
        self.v_proj = nn.Linear(config.d_model, v_proj_out_dim, bias=config.include_bias, device=config.init_device)

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_pytorch_sdpa: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        att, cache, attn_weights = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            use_pytorch_sdpa=use_pytorch_sdpa,
            output_attentions=output_attentions,
        )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
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
        x = self.dropout(x)
        x = og_x + x

        return x, cache, attn_weights


class OLMoBlockGroup(nn.ModuleList):
    def __init__(self, config: OLMoConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        use_pytorch_sdpa: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[torch.Tensor]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        attn_weights: Optional[List[torch.Tensor]] = [] if output_attentions else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                # shape: (batch_size, seq_len, d_model)
                x, cache, layer_attn_weights = self._activation_checkpoint_fn(  # type: ignore
                    block,
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    use_pytorch_sdpa=use_pytorch_sdpa,
                    output_attentions=output_attentions,
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache, layer_attn_weights = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    use_pytorch_sdpa=use_pytorch_sdpa,
                    output_attentions=output_attentions,
                )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
            if attn_weights is not None:
                assert layer_attn_weights is not None
                attn_weights.append(layer_attn_weights)
        return x, attn_key_values, attn_weights

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


OLMO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OLMoConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OLMo Model outputting raw hidden-states without any specific head on top.",
    OLMO_START_DOCSTRING,
)
class OLMoPreTrainedModel(PreTrainedModel):
    config_class = OLMoConfig
    base_model_prefix = "model"
    _no_split_modules = ["OLMoBlock"]
    # _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        # `OLMoModel.reset_parameters` initializes weights of itself and its children
        if isinstance(module, OLMoModel):
            module.reset_parameters()


OLMO_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        attention_bias (`torch.Tensor` of shape `(batch_size, 1, sequence_length, sequence_length)`,
            `(1, 1, sequence_length, sequence_length)`, or `(sequence_length, sequence_length)`, *optional*):

            This is used to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            One formats is currently allowed:
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        last_logits_only (`bool`, *optional*):
            If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare OLMo Model outputting raw hidden-states without any specific head on top.",
    OLMO_START_DOCSTRING,
)
class OLMoModel(OLMoPreTrainedModel):
    """
    Transformer decoder consisting of `config.n_layers` layers. Each layer is a [`OLMoBlock`]

    Args:
        config: OLMoConfig
    """

    def __init__(self, config: OLMoConfig):
        super().__init__(config)
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise ValueError("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise ValueError("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise ValueError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise ValueError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(self.config.flash_attention)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                "emb_drop": Dropout(config.embedding_dropout),
                "ln_f": LayerNorm.build(config),
            }
        )

        blocks = [OLMoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
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
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        logger.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[-1] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    @add_start_docstrings_to_model_forward(OLMO_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
    ) -> BaseModelOutputWithPast | Tuple:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Input ids or embeddings must be given to OLMo model")

        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if inputs_embeds is None else inputs_embeds.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if inputs_embeds is None else inputs_embeds  # type: ignore

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
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
        attn_weights: Optional[List[torch.Tensor]] = [] if output_attentions else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache, block_attn_weights = self._activation_checkpoint_fn(
                        block,
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        use_pytorch_sdpa=self.config.use_pytorch_sdpa,
                        output_attentions=output_attentions,
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache, block_attn_weights = block(
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        use_pytorch_sdpa=self.config.use_pytorch_sdpa,
                        output_attentions=output_attentions,
                    )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
                if attn_weights is not None:
                    assert block_attn_weights is not None
                    attn_weights.append(block_attn_weights)
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
                x, cache, block_attn_weights = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                    use_pytorch_sdpa=self.config.use_pytorch_sdpa,
                    output_attentions=output_attentions,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)
                if attn_weights is not None:
                    assert block_attn_weights is not None
                    attn_weights.append(block_attn_weights)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        if not return_dict:
            return tuple(
                v
                for v in [
                    x,
                    tuple(attn_key_values) if attn_key_values is not None else None,
                    tuple(all_hidden_states),
                    None,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=tuple(attn_key_values) if attn_key_values is not None else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(attn_weights) if attn_weights is not None else None,
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise ValueError("'by_block_group' FSDP wrapping strategy requires block group size greater than 1")

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlockGroup)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise ValueError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops
        n_params = self.num_params()
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.config.max_sequence_length
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * (self.config.max_sequence_length**2))
        )
        self.__num_fwd_flops = params_flops_per_seq + attn_flops_per_seq
        return self.__num_fwd_flops


@add_start_docstrings(
    "OLMo Model with a language modeling head on top (linear layer with weights that can tied to the input embeddings).",
    OLMO_START_DOCSTRING,
)
class OLMoForCausalLM(OLMoPreTrainedModel):
    _tied_weights_keys = []
    # _tied_weights_keys = ["transformer.wte.weight"]

    def __init__(self, config: OLMoConfig):
        super().__init__(config)
        self.model = OLMoModel(config)

        # Initialize weights and apply final processing
        self.post_init()

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

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(OLMO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OLMoForCausalLM

        >>> model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        base_output: BaseModelOutputWithPast | Tuple = self.model.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        last_hidden_state = base_output.last_hidden_state if return_dict else base_output[0]

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(last_hidden_state, self.model.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.model.transformer.ff_out(last_hidden_state)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + base_output[1:]
            return (loss,) + output if loss is not None else output

        assert isinstance(base_output, BaseModelOutputWithPast)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_output.past_key_values,
            hidden_states=base_output.hidden_states,
            attentions=base_output.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        kwargs.pop("cache_position")
        model_inputs.update(kwargs)
        # logger.warning("%s %s", kwargs.keys(), model_inputs.keys())
        # model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
