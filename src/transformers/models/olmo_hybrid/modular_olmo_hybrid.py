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
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import layer_type_validation
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.import_utils import is_flash_linear_attention_available
from ...utils.output_capturing import capture_outputs
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaDecoderLayer
from ..olmo3.modeling_olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3MLP,
    Olmo3RMSNorm,
    Olmo3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextDynamicCache,
    Qwen3NextModel,
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


class OlmoHybridConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`OlmoHybridModel`]. It is used to instantiate
    an OLMo Hybrid model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [allenai/Olmo-Hybrid-7B](https://huggingface.co/allenai/Olmo-Hybrid-7B) model.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 100352):
            Vocabulary size of the OlmoHybrid model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`OlmoHybridModel`].
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
        linear_a_log_min (`float`, *optional*, defaults to 0.0):
            Minimum value for uniform initialization of A_log in GatedDeltaNet layers.
        linear_a_log_max (`float`, *optional*, defaults to 16.0):
            Maximum value for uniform initialization of A_log in GatedDeltaNet layers.
        linear_dt_min (`float`, *optional*, defaults to 0.001):
            Minimum value for dt initialization in GatedDeltaNet layers.
        linear_dt_max (`float`, *optional*, defaults to 0.1):
            Maximum value for dt initialization in GatedDeltaNet layers.
        linear_dt_init_floor (`float`, *optional*, defaults to 0.0001):
            Floor value for clamping dt during initialization in GatedDeltaNet layers.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.
        linear_allow_neg_eigval (`bool`, *optional*, defaults to `True`):
            Whether to allow negative eigenvalues in the GatedDeltaNet recurrence. When `True`, the beta
            parameter is scaled by 2.0 to allow values in range [0, 2] instead of [0, 1].
    ```python
    >>> from transformers import OlmoHybridModel, OlmoHybridConfig

    >>> # Initializing an OlmoHybrid style configuration
    >>> configuration = OlmoHybridConfig()

    >>> # Initializing a model from the OlmoHybrid style configuration
    >>> model = OlmoHybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo_hybrid"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_split_input",  # input is replicated due to the added norm on q and k
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

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
        linear_a_log_min: float = 0.0,
        linear_a_log_max: float = 16.0,
        linear_dt_min: float = 0.001,
        linear_dt_max: float = 0.1,
        linear_dt_init_floor: float = 1e-4,
        linear_conv_kernel_dim: int = 4,
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
            raise ValueError("OLMoHybrid expects at least one 'linear_attention' layer.")
        if all(t == "linear_attention" for t in layer_types):
            raise ValueError("OLMoHybrid expects at least one attention layer.")

        self.layer_types = layer_types

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
        self.linear_a_log_min = linear_a_log_min
        self.linear_a_log_max = linear_a_log_max
        self.linear_dt_min = linear_dt_min
        self.linear_dt_max = linear_dt_max
        self.linear_dt_init_floor = linear_dt_init_floor
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_allow_neg_eigval = linear_allow_neg_eigval

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
            rope_parameters=rope_parameters,
            **kwargs,
        )
        del self.pretraining_tp
        del self.mlp_bias
        del self.head_dim


class OlmoHybridDynamicCache(Qwen3NextDynamicCache):
    """
    Cache for hybrid model supporting both attention KV cache and linear attention state.

    Inherits from Qwen3NextDynamicCache. The main difference is that this cache
    stores separate conv states for q, k, v (instead of a single conv_states list).
    """

    def __init__(self, config: OlmoHybridConfig):
        super().__init__(config)
        del self.conv_states
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


class OlmoHybridRMSNormGated(Qwen3NextRMSNormGated):
    pass


class OlmoHybridRMSNorm(Olmo3RMSNorm):
    pass


class OlmoHybridShortConvolution(nn.Conv1d):
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
        self.act_fn = ACT2FN[activation]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: torch.Tensor | None = None,
        use_precomputed: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len, dim = hidden_states.shape[-2:]

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

        out = self.act_fn(out)

        return out.transpose(1, 2), conv_state


class OlmoHybridAttention(Olmo3Attention):
    """
    Multi-headed attention for OLMo Hybrid that supports optional RoPE (NoPE mode).

    Inherits from Olmo3Attention. The only behavioral difference is that when
    position_embeddings is None, rotary position embeddings are skipped entirely,
    enabling NoPE mode for long context extension.
    """

    def __init__(self, config: OlmoHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Hybrid model doesn't use sliding window attention
        del self.sliding_window
        del self.attention_type

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

        # NoPE mode: skip RoPE when position_embeddings is None
        cos, sin = None, None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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


class OlmoHybridRotaryEmbedding(Olmo3RotaryEmbedding):
    """
    RoPE for OLMo Hybrid that returns float32 cos/sin to match OLMo-core.
    """

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # KEY difference from parent: return float32, don't cast to x.dtype
        return cos, sin


class OlmoHybridGatedDeltaNet(nn.Module):
    """
    GatedDeltaNet linear attention for OLMo Hybrid.

    Key differences from Qwen3NextGatedDeltaNet:
    - Fully separate q/k/v/a/b projections (vs. fused qkvz + partially split ba)
    - Per-projection conv1d for q, k, v (vs. single conv1d over concatenated qkv)
    - Dedicated g_proj gate (vs. z derived from the fused qkvz projection)
    - Supports allow_neg_eigval: scales beta by 2.0 to allow range [0, 2]
    """

    def __init__(self, config: OlmoHybridConfig, layer_idx: int):
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
        self.allow_neg_eigval = config.linear_allow_neg_eigval
        self.eps = config.rms_norm_eps

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        Conv1dClass = ShortConvolution if ShortConvolution is not None else OlmoHybridShortConvolution

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

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(
            config.linear_a_log_min, config.linear_a_log_max
        )
        self.A_log = nn.Parameter(torch.log(A))

        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(config.linear_dt_max) - math.log(config.linear_dt_min))
            + math.log(config.linear_dt_min)
        )
        dt = torch.clamp(dt, min=config.linear_dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Output norm - NOTE: FLA's FusedRMSNormGated uses eps=1e-5 by default
        self.o_norm = (
            OlmoHybridRMSNormGated(self.head_v_dim, eps=1e-5)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=1e-5,
                device=torch.cuda.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )
        )

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
        cache_params: OlmoHybridDynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Requires LEFT padding to work correctly
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

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

        gate = self.g_proj(hidden_states)
        output = output.reshape(-1, self.head_v_dim)
        gate = gate.reshape(-1, self.head_v_dim)
        output = self.o_norm(output, gate)
        output = output.reshape(batch_size, seq_len, -1)

        output = self.o_proj(output)

        return output


class OlmoHybridMLP(Olmo3MLP):
    pass


class OlmoHybridAttentionDecoderLayer(Olmo3DecoderLayer):
    def __init__(self, config: OlmoHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_type = "full_attention"
        self.self_attn = OlmoHybridAttention(config=config, layer_idx=layer_idx)


class OlmoHybridLinearAttentionDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OlmoHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_type = "linear_attention"
        del self.self_attn
        self.linear_attn = OlmoHybridGatedDeltaNet(config, layer_idx=layer_idx)
        self.input_layernorm = OlmoHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = OlmoHybridMLP(config)

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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Main difference to llama - signature (`cache_params`) and linear attention
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class OlmoHybridPreTrainedModel(Qwen3NextPreTrainedModel):
    _is_stateful = True
    _no_split_modules = ["OlmoHybridAttentionDecoderLayer", "OlmoHybridLinearAttentionDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": (OlmoHybridAttentionDecoderLayer, OlmoHybridLinearAttentionDecoderLayer),
        "attentions": OlmoHybridAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, OlmoHybridGatedDeltaNet):
            cfg = self.config
            init.copy_(
                module.A_log,
                torch.empty_like(module.A_log).uniform_(cfg.linear_a_log_min, cfg.linear_a_log_max).log_(),
            )
            dt = torch.exp(
                torch.rand_like(module.dt_bias) * (math.log(cfg.linear_dt_max) - math.log(cfg.linear_dt_min))
                + math.log(cfg.linear_dt_min)
            )
            dt = torch.clamp(dt, min=cfg.linear_dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            init.copy_(module.dt_bias, inv_dt)


class OlmoHybridModel(Qwen3NextModel):
    def __init__(self, config: OlmoHybridConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                OlmoHybridLinearAttentionDecoderLayer(config, layer_idx)
                if config.layer_types[layer_idx] == "linear_attention"
                else OlmoHybridAttentionDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = (
            OlmoHybridRotaryEmbedding(config=config)
            if getattr(config, "rope_parameters", None) is not None
            and config.rope_parameters.get("rope_theta") is not None
            else None
        )
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
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

        if use_cache and past_key_values is None:
            past_key_values = OlmoHybridDynamicCache(config=self.config)

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
        # RoPE or NoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.rotary_emb is not None else None

        for decoder_layer in self.layers:
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            layer_position_embeddings = position_embeddings if decoder_layer.layer_type == "full_attention" else None

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=layer_position_embeddings,
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


class OlmoHybridForCausalLM(Olmo3ForCausalLM):
    pass


__all__ = [
    "OlmoHybridConfig",
    "OlmoHybridForCausalLM",
    "OlmoHybridModel",
    "OlmoHybridPreTrainedModel",
]
