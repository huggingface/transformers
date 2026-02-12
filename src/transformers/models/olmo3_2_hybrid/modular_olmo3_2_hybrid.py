# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import layer_type_validation
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.generic import check_model_inputs
from ...utils.import_utils import is_flash_linear_attention_available
from ..olmo3.configuration_olmo3 import Olmo3Config
from ..olmo3.modeling_olmo3 import (
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3MLP,
    Olmo3RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextDynamicCache,
    Qwen3NextPreTrainedModel,
    Qwen3NextRMSNormGated,
    apply_mask_to_padding_states,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None
    ShortConvolution = None

is_fast_path_available = all(
    (ShortConvolution, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule, FusedRMSNormGated)
)

logger = logging.get_logger(__name__)


class Olmo3_2HybridConfig(Olmo3Config):
    r"""
    This is the configuration class to store the configuration of a [`Olmo3_2HybridModel`]. It is used to instantiate
    an OLMo 3.2 Hybrid model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [allenai/OLMo-3.2-1B-Hybrid](https://huggingface.co/allenai/OLMo-3.2-1B-Hybrid) model.

    The OLMo 3.2 Hybrid model combines standard transformer attention layers with GatedDeltaNet linear attention
    layers for improved efficiency while maintaining model quality.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the Olmo3_2Hybrid model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Olmo3_2HybridModel`].
        hidden_size (`int`, *optional*, defaults to 3840):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 30):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 65536):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 100277):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 100257):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`. Can be `None` to disable RoPE (e.g., during long context extension).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        sliding_window (`int`, *optional*):
            Size of the sliding window attention. Not used in OLMo 3.2 Hybrid.
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Can contain `"full_attention"` or `"linear_attention"`.
            Defaults to linear attention for most layers with full attention for every 4th layer.
        linear_num_key_heads (`int`, *optional*):
            Number of key heads for the linear attention layers. Defaults to `num_attention_heads`.
        linear_num_value_heads (`int`, *optional*):
            Number of value heads for the linear attention layers. Defaults to `num_attention_heads`.
        linear_key_head_dim (`int`, *optional*):
            Dimension of each key head in linear attention layers. Defaults to `0.75 * hidden_size / linear_num_key_heads`.
        linear_value_head_dim (`int`, *optional*):
            Dimension of each value head in linear attention layers. Defaults to `2 * linear_key_head_dim`.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.
        linear_use_gate (`bool`, *optional*, defaults to `True`):
            Whether to use gating in the linear attention output normalization.
        linear_allow_neg_eigval (`bool`, *optional*, defaults to `True`):
            Whether to allow negative eigenvalues in the GatedDeltaNet recurrence. When `True`, the beta
            parameter is scaled by 2.0 to allow values in range [0, 2] instead of [0, 1].
    ```python
    >>> from transformers import Olmo3_2HybridModel, Olmo3_2HybridConfig

    >>> # Initializing an Olmo3.2 Hybrid style configuration
    >>> configuration = Olmo3_2HybridConfig()

    >>> # Initializing a model from the Olmo3.2 Hybrid style configuration
    >>> model = Olmo3_2HybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo3_2_hybrid"

    def __init__(
        self,
        vocab_size: int | None = 100352,
        hidden_size: int | None = 3840,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 30,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 65536,
        initializer_range: float | None = 0.02,
        use_cache: bool | None = True,
        pad_token_id: int | None = 100277,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 100257,
        tie_word_embeddings: bool | None = False,
        rope_parameters=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        rms_norm_eps: float | None = 1e-06,
        layer_types: list[str] | None = None,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_conv_kernel_dim: int = 4,
        linear_use_gate: bool = True,
        linear_allow_neg_eigval: bool = True,
        **kwargs,
    ):
        if layer_types is None:
            # Default: linear attention for most layers, full attention every 4th layer
            layer_types = ["linear_attention"] * int(num_hidden_layers)
            for i in range(int(num_hidden_layers)):
                if i % 4 == 3:
                    layer_types[i] = "full_attention"
            # Ensure at least one full attention layer for small num_hidden_layers
            if "full_attention" not in layer_types:
                layer_types[-1] = "full_attention"

        layer_type_validation(layer_types, num_hidden_layers)

        if "linear_attention" not in layer_types:
            raise ValueError("OLMo3.2 Hybrid expects at least one 'linear_attention' layer.")
        if all(t == "linear_attention" for t in layer_types):
            raise ValueError("OLMo3.2 Hybrid expects at least one attention layer.")

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            layer_types=layer_types,
            rope_parameters=rope_parameters,
            **kwargs,
        )
        del self.sliding_window

        if linear_num_key_heads is None:
            linear_num_key_heads = num_attention_heads
        if linear_num_value_heads is None:
            linear_num_value_heads = num_attention_heads
        if linear_key_head_dim is None:
            linear_key_head_dim = int(0.75 * hidden_size / linear_num_key_heads)
        if linear_value_head_dim is None:
            linear_value_head_dim = 2 * linear_key_head_dim

        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_use_gate = linear_use_gate
        self.linear_allow_neg_eigval = linear_allow_neg_eigval


class Olmo3_2HybridDynamicCache(Qwen3NextDynamicCache):
    """
    Cache for hybrid model supporting both attention KV cache and linear attention state.

    Inherits from Qwen3NextDynamicCache. The main difference is that this cache
    stores separate conv states for q, k, v (instead of a single conv_states list).
    """

    def __init__(self, config: Olmo3_2HybridConfig):
        super().__init__(config)
        # Replace single conv_states with separate q, k, v conv states
        self.conv_states_q = [None for _ in range(config.num_hidden_layers)]
        self.conv_states_k = [None for _ in range(config.num_hidden_layers)]
        self.conv_states_v = [None for _ in range(config.num_hidden_layers)]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        batch_size = beam_idx.shape[0]
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                if self.key_cache[layer_idx].shape[0] < batch_size:
                    expand_ratio = batch_size // self.key_cache[layer_idx].shape[0]
                    self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(expand_ratio, dim=0)
                    self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(expand_ratio, dim=0)
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.conv_states_q[layer_idx] is not None:
                if self.conv_states_q[layer_idx].shape[0] < batch_size:
                    expand_ratio = batch_size // self.conv_states_q[layer_idx].shape[0]
                    self.conv_states_q[layer_idx] = self.conv_states_q[layer_idx].repeat_interleave(
                        expand_ratio, dim=0
                    )
                    self.conv_states_k[layer_idx] = self.conv_states_k[layer_idx].repeat_interleave(
                        expand_ratio, dim=0
                    )
                    self.conv_states_v[layer_idx] = self.conv_states_v[layer_idx].repeat_interleave(
                        expand_ratio, dim=0
                    )
                    self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].repeat_interleave(
                        expand_ratio, dim=0
                    )
                device = self.conv_states_q[layer_idx].device
                self.conv_states_q[layer_idx] = self.conv_states_q[layer_idx].index_select(0, beam_idx.to(device))
                self.conv_states_k[layer_idx] = self.conv_states_k[layer_idx].index_select(0, beam_idx.to(device))
                self.conv_states_v[layer_idx] = self.conv_states_v[layer_idx].index_select(0, beam_idx.to(device))
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(
                    0, beam_idx.to(device)
                )

    @property
    def has_previous_state(self):
        return self.conv_states_q[self.last_linear_layer] is not None


class Olmo3_2HybridRMSNormGated(Qwen3NextRMSNormGated):
    pass


class Olmo3_2HybridRMSNorm(Olmo3RMSNorm):
    pass


class Olmo3_2HybridShortConvolution(nn.Conv1d):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias,
        )
        self.hidden_size = hidden_size
        self.conv_kernel_size = kernel_size
        self.act_fn = ACT2FN[activation] if activation is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: torch.Tensor | None = None,
        use_precomputed: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = hidden_states.shape

        hidden_states = hidden_states.transpose(1, 2)

        if use_precomputed:
            # Single token update (decoding mode)
            x_with_state = torch.cat([cache, hidden_states], dim=-1)
            out = F.conv1d(
                x_with_state,
                self.weight,
                self.bias,
                padding=0,
                groups=dim,
            )
            conv_state = x_with_state[:, :, 1:]
        else:
            # Multi-token forward (prefill mode)
            out = F.conv1d(hidden_states, self.weight, self.bias, padding=self.conv_kernel_size - 1, groups=dim)
            out = out[:, :, :seq_len]
            conv_state = F.pad(hidden_states, (self.conv_kernel_size - 1 - hidden_states.shape[-1], 0))

        if self.act_fn is not None:
            out = self.act_fn(out)

        return out.transpose(1, 2), conv_state


class Olmo3_2HybridAttention(nn.Module):
    """
    Multi-headed attention for OLMo 3.2 Hybrid that supports optional RoPE.

    When position_embeddings is None (NoPE mode for long context extension),
    rotary position embeddings are skipped entirely.
    """

    def __init__(self, config: Olmo3_2HybridConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Olmo3_2HybridRMSNorm(self.num_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo3_2HybridRMSNorm(self.num_key_value_heads * self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # Apply RoPE only if position_embeddings are provided (not in NoPE mode)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {
                "sin": position_embeddings[1] if position_embeddings is not None else None,
                "cos": position_embeddings[0] if position_embeddings is not None else None,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Olmo3_2HybridRotaryEmbedding(nn.Module):
    """
    RoPE for OLMo 3.2 Hybrid that returns float32 cos/sin to match OLMo-core.

    The only difference from standard RoPE is NOT casting cos/sin back to x.dtype,
    preserving float32 precision like OLMo-core's full_precision=True.

    When rope_parameters is None or rope_theta is None,
    the embedding is disabled and forward() returns None.
    """

    def __init__(self, config, device=None):
        super().__init__()

        try:
            rope_params = getattr(config, "rope_parameters", None)
            if rope_params is None:
                self.disabled = True
                return
            rope_theta = rope_params["rope_theta"]
            if rope_theta is None:
                self.disabled = True
                return
        except (AttributeError, TypeError, KeyError):
            self.disabled = True
            return

        self.disabled = False
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        self.rope_type = rope_params.get("rope_type", "default") if isinstance(rope_params, dict) else "default"
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config=None,
        device=None,
        seq_len=None,
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.disabled:
            return None

        # Recompute inv_freq if sequence exceeds cached length (needed for dynamic NTK scaling)
        seq_len = torch.max(position_ids).item() + 1
        if seq_len > self.max_seq_len_cached:
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(self.config, self.inv_freq.device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # KEY FIX: Return float32, don't cast to x.dtype
        return cos, sin


class Olmo3_2HybridGatedDeltaNet(nn.Module):
    def __init__(self, config: Olmo3_2HybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.layer_idx = layer_idx
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.use_gate = config.linear_use_gate
        self.allow_neg_eigval = config.linear_allow_neg_eigval
        self.eps = config.rms_norm_eps

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        if self.use_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        Conv1dClass = ShortConvolution if ShortConvolution is not None else Olmo3_2HybridShortConvolution

        self.q_conv1d = Conv1dClass(
            hidden_size=self.key_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            activation="silu",
        )
        self.k_conv1d = Conv1dClass(
            hidden_size=self.key_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            activation="silu",
        )
        self.v_conv1d = Conv1dClass(
            hidden_size=self.value_dim,
            kernel_size=self.conv_kernel_size,
            bias=False,
            activation="silu",
        )

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Output norm - NOTE: FLA's FusedRMSNormGated uses eps=1e-5 by default
        o_norm_eps = 1e-5
        NormClass = FusedRMSNormGated if FusedRMSNormGated is not None else Olmo3_2HybridRMSNormGated
        if self.use_gate:
            self.o_norm = NormClass(self.head_v_dim, eps=o_norm_eps)
        else:
            self.o_norm = Olmo3_2HybridRMSNorm(self.head_v_dim, eps=o_norm_eps)

        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required libraries is not installed. "
                "Falling back to torch implementation. To install, follow: "
                "https://github.com/fla-org/flash-linear-attention#installation"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Olmo3_2HybridDynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Requires LEFT padding to work correctly
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        input_shape = hidden_states.shape[:-1]

        batch_size, seq_len, _ = hidden_states.shape

        use_cache = cache_params is not None
        use_precomputed = use_cache and getattr(cache_params, "has_previous_state", False) and seq_len == 1

        conv_state_q = cache_params.conv_states_q[self.layer_idx] if cache_params else None
        conv_state_k = cache_params.conv_states_k[self.layer_idx] if cache_params else None
        conv_state_v = cache_params.conv_states_v[self.layer_idx] if cache_params else None
        recurrent_state = cache_params.recurrent_states[self.layer_idx] if cache_params else None

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, new_conv_state_q = self.q_conv1d(
            q, cache=conv_state_q, use_precomputed=use_precomputed, output_final_state=use_cache
        )
        k, new_conv_state_k = self.k_conv1d(
            k, cache=conv_state_k, use_precomputed=use_precomputed, output_final_state=use_cache
        )
        v, new_conv_state_v = self.v_conv1d(
            v, cache=conv_state_v, use_precomputed=use_precomputed, output_final_state=use_cache
        )

        if cache_params is not None:
            cache_params.conv_states_q[self.layer_idx] = new_conv_state_q
            cache_params.conv_states_k[self.layer_idx] = new_conv_state_k
            cache_params.conv_states_v[self.layer_idx] = new_conv_state_v

        q = q.view(batch_size, seq_len, -1, self.head_k_dim)
        k = k.view(batch_size, seq_len, -1, self.head_k_dim)
        v = v.view(batch_size, seq_len, -1, self.head_v_dim)

        if self.num_v_heads > self.num_k_heads:
            expand_ratio = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(expand_ratio, dim=2)
            k = k.repeat_interleave(expand_ratio, dim=2)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        if use_precomputed:
            output, new_recurrent_state = self.recurrent_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            output, new_recurrent_state = self.chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = new_recurrent_state

        if self.use_gate:
            gate = self.g_proj(hidden_states)
            gate = gate.view(batch_size, seq_len, -1, self.head_v_dim)
            output_shape = gate.shape
            output = output.reshape(-1, output.shape[-1])
            gate = gate.reshape(-1, gate.shape[-1])
            output = self.o_norm(output, gate)
            output = output.reshape(output_shape)
        else:
            output = output.reshape(-1, output.shape[-1])
            output = self.o_norm(output)
            output = output.view(batch_size, seq_len, -1, self.head_v_dim)

        output = output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(output)

        return output


class Olmo3_2MLP(Olmo3MLP):
    pass


class Olmo3_2HybridAttentionDecoderLayer(Olmo3DecoderLayer):
    def __init__(self, config: Olmo3_2HybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_type = "full_attention"
        self.self_attn = Olmo3_2HybridAttention(config=config, layer_idx=layer_idx)


class Olmo3_2HybridLinearDecoderLayer(Olmo3DecoderLayer):
    def __init__(self, config: Olmo3_2HybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_type = "linear_attention"
        del self.self_attn
        del self.post_attention_layernorm
        del self.post_feedforward_layernorm
        self.linear_attn = Olmo3_2HybridGatedDeltaNet(config, layer_idx=layer_idx)
        self.attention_layer_norm = Olmo3_2HybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layer_norm = Olmo3_2HybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # GatedDeltaNet has a different call signature than standard attention
        # (cache_params instead of past_key_values, returns a single tensor),
        # so we override forward entirely.
        residual = hidden_states
        hidden_states = self.attention_layer_norm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.feedforward_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Olmo3_2HybridPreTrainedModel(Qwen3NextPreTrainedModel):
    _is_stateful = True
    _no_split_modules = ["Olmo3_2HybridAttentionDecoderLayer", "Olmo3_2HybridLinearDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": (Olmo3_2HybridAttentionDecoderLayer, Olmo3_2HybridLinearDecoderLayer),
        "attentions": Olmo3_2HybridAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv1d):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (Olmo3_2HybridRMSNorm, Olmo3_2HybridRMSNormGated)) or (
            FusedRMSNormGated is not None and isinstance(module, FusedRMSNormGated)
        ):
            init.ones_(module.weight)
        elif isinstance(module, Olmo3_2HybridGatedDeltaNet):
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(torch.rand_like(module.dt_bias) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            init.copy_(module.dt_bias, inv_dt)
        elif isinstance(module, Olmo3_2HybridRotaryEmbedding):
            if not module.disabled:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type]
                inv_freq, _ = rope_init_fn(module.config, module.inv_freq.device)
                init.copy_(module.inv_freq, inv_freq)
                init.copy_(module.original_inv_freq, inv_freq)


class Olmo3_2HybridModel(Olmo3_2HybridPreTrainedModel):
    def __init__(self, config: Olmo3_2HybridConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Olmo3_2HybridLinearDecoderLayer(config, layer_idx)
                if config.layer_types[layer_idx] == "linear_attention"
                else Olmo3_2HybridAttentionDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Olmo3_2HybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Olmo3_2HybridRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None or not isinstance(past_key_values, Olmo3_2HybridDynamicCache):
                past_key_values = Olmo3_2HybridDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask


class Olmo3_2HybridForCausalLM(Olmo3ForCausalLM):
    pass


__all__ = [
    "Olmo3_2HybridConfig",
    "Olmo3_2HybridForCausalLM",
    "Olmo3_2HybridModel",
    "Olmo3_2HybridPreTrainedModel",
    "Olmo3_2HybridLinearDecoderLayer",
    "Olmo3_2HybridAttentionDecoderLayer",
]
