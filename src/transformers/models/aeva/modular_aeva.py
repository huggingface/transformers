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


import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, is_grouped_mm_available
from ...utils.generic import check_model_inputs
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeForCausalLM,
    Glm4MoeMLP,
    Glm4MoeModel,
    Glm4MoeRMSNorm,
    Glm4MoeRotaryEmbedding,
    Glm4MoeTopkRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class AevaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AevaModel`]. It is used to instantiate an Aeva
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Aeva-M.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151552):
            Vocabulary size of the Aeva model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`AevaModel`].
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimension of the hidden representations. This is the core semantic anchor that determines all other
            architectural dimensions through continuous scaling functions.
        max_position_embeddings (`int`, *optional*):
            The maximum sequence length that this model might ever be used with. Computed as
            `min(hidden_size * 32, hidden_size * 25 + hidden_size**2 / 65536)` and aligned to 4096 if not specified.
        num_hidden_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Computed as
            `20 * log2(1 + hidden_size / 1024) * (1 + (hidden_size - 1024) / (hidden_size + 8192)) + 12`
            and rounded to multiples of 4 if not specified.
        intermediate_size (`int`, *optional*):
            Dimension of the MLP representations for dense layers. Computed as `hidden_size * (2.4 + hidden_size / 32768)`
            and aligned to 64 if not specified.
        first_k_dense_replace (`int`, *optional*):
            Number of dense layers at the beginning before MoE layers. Computed as `1 + int(3 * hidden_size / 5120)`
            if not specified.
        num_attention_heads (`int`, *optional*):
            Number of attention heads for each attention layer. Computed as
            `round(hidden_size / 64 * (1 + hidden_size / 16384))` and aligned to 8 if not specified.
        head_dim (`int`, *optional*):
            Dimension of each attention head. Computed as `hidden_size // num_attention_heads` if not specified.
        num_key_value_heads (`int`, *optional*):
            Number of key_value heads for Grouped Query Attention. Computed as
            `num_attention_heads // max(1, int(8 - hidden_size / 8192))` adjusted to ensure divisibility with
            num_attention_heads if not specified.
        use_qk_norm (`bool`, *optional*):
            Whether to use query-key normalization. Automatically enabled when head_dim > 128.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_parameters (`dict`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `True`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*):
            Size of the sliding window. Computed as 80% of max_position_embeddings and aligned to 128 if not specified.
        attn_layer_types (`list`, *optional*):
            Attention type pattern for each layer. If not specified, defaults to a hybrid pattern where layers in the top
            20% use 'sliding_attention'. For the remaining layers, 'linear_attention' is used at intervals that decrease
            from period 3 to period 1 as depth increases.
        n_routed_experts (`int`, *optional*):
            Number of routed experts. Computed as `hidden_size // 32` and aligned to 8 if not specified.
        num_experts_per_tok (`int`, *optional*):
            Number of selected experts per token. Computed as `min(int(4 + 4 * log2(1 + hidden_size / 4096)), n_routed_experts // 2)`
            if not specified.
        moe_intermediate_size (`int`, *optional*):
            Dimension of the MoE representations. Computed as `hidden_size * 0.3 + hidden_size**2 / 131072` and aligned
            to 64 if not specified.
        n_group (`int`, *optional*):
            Number of groups for routed experts. Computed as the next power of 2 greater than or equal to
            `max(1, int(n_routed_experts / 10))` if not specified.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token.
        num_local_groups (`int`, *optional*):
            Number of anchor expert groups. Computed as `max(8, int(n_routed_experts / (2 + hidden_size / 8192)))` and
            incremented until it divides `n_routed_experts` evenly if not specified.
        anchor_intermediate_size (`int`, *optional*):
            Dimension of the anchor expert representations. Computed as `moe_intermediate_size * 0.5` and aligned to 64
            if not specified.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        routed_scaling_factor (`float`, *optional*):
            Scaling factor for routed expert weights. Computed as `hidden_size / 5120` if not specified.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to return router logits.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Coefficient for the router auxiliary loss.
        anchor_alpha (`float`, *optional*, defaults to 0.1):
            Scaling factor for anchor expert outputs.
        anchor_intermediate_size_ratio (`float`, *optional*, defaults to 0.5):
            Ratio of anchor intermediate size to expert intermediate size.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        hidden_act (`str` or `function`, *optional*):
            The non-linear activation function in the decoder. Defaults to "silu".
        bos_token_id (`int`, *optional*, defaults to 151329):
            Beginning of stream token id.
        eos_token_id (`list[int]`, *optional*, defaults to [151329, 151336, 151338]):
            End of stream token ids.
        pad_token_id (`int`, *optional*, defaults to 151329):
            Padding token id.

    ```python
    >>> from transformers import AevaModel, AevaConfig
    >>> # Initializing an Aeva style configuration
    >>> configuration = AevaConfig()
    >>> # Initializing a model from the Aeva-M style configuration
    >>> model = AevaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "aeva"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Aeva`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        anchor_alpha: float | None = 0.1,
        anchor_intermediate_size: int | None = None,
        anchor_intermediate_size_ratio: float | None = 0.5,
        attention_bias: bool | None = True,
        attention_dropout: float | None = 0.0,
        attn_layer_types: list | None = None,
        bos_token_id: int | None = None,
        eos_token_id: list[int] | None = None,
        first_k_dense_replace: int | None = None,
        hidden_act: str | None = "silu",
        hidden_size: int | None = 5120,
        initializer_range: float | None = 0.02,
        intermediate_size: int | None = None,
        max_position_embeddings: int | None = None,
        moe_intermediate_size: int | None = None,
        n_group: int | None = None,
        n_routed_experts: int | None = None,
        n_shared_experts: int | None = None,
        num_attention_heads: int | None = None,
        num_experts_per_tok: int | None = None,
        num_hidden_layers: int | None = None,
        num_key_value_heads: int | None = None,
        num_local_groups: int | None = None,
        output_router_logits: bool | None = False,
        pad_token_id: int | None = None,
        rms_norm_eps: float | None = 1e-5,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        router_aux_loss_coef: float | None = 0.001,
        routed_scaling_factor: float | None = None,
        sliding_window: int | None = None,
        tie_word_embeddings: bool | None = False,
        topk_group: int | None = 1,
        use_cache: bool | None = True,
        use_qk_norm: bool | None = None,
        use_sliding_window: bool | None = True,
        vocab_size: int | None = 151552,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.tie_word_embeddings = tie_word_embeddings if tie_word_embeddings is not None else False
        self.use_cache = use_cache if use_cache is not None else True
        self.vocab_size = vocab_size
        raw_layers = max(12, int(hidden_size / 100))
        self.num_hidden_layers = num_hidden_layers if num_hidden_layers is not None else raw_layers - (raw_layers % 4)
        raw_max_pos = hidden_size * 16
        self.max_position_embeddings = (
            max_position_embeddings
            if max_position_embeddings is not None
            else int(raw_max_pos) - (int(raw_max_pos) % 4096)
        )
        self.first_k_dense_replace = (
            first_k_dense_replace if first_k_dense_replace is not None else max(2, int(self.num_hidden_layers * 0.1))
        )
        self.initializer_range = initializer_range if initializer_range is not None else 0.02
        self.rms_norm_eps = rms_norm_eps if rms_norm_eps is not None else 1e-5
        self.hidden_act = hidden_act

        # --- Attention Mechanism ---
        config_head_dim = 128 if hidden_size >= 8192 else 64
        self.num_attention_heads = (
            num_attention_heads
            if num_attention_heads is not None
            else (hidden_size // config_head_dim) - ((hidden_size // config_head_dim) % 8)
        )
        self.head_dim = config_head_dim
        kv_divisor = 4 if config_head_dim == 128 else 8
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else max(1, self.num_attention_heads // kv_divisor)
        )
        while self.num_attention_heads % self.num_key_value_heads != 0:
            self.num_key_value_heads += 1
        self.attention_bias = attention_bias if attention_bias is not None else True
        self.attention_dropout = attention_dropout if attention_dropout is not None else 0.0
        self.use_qk_norm = use_qk_norm if use_qk_norm is not None else (config_head_dim > 64)
        if rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 1.0}
        else:
            self.rope_parameters = rope_parameters
        raw_sw = min(hidden_size * 2, max(4096, self.max_position_embeddings * 0.5))
        self.sliding_window = sliding_window if sliding_window is not None else int(raw_sw) - (int(raw_sw) % 128)
        self.use_sliding_window = use_sliding_window if use_sliding_window is not None else True
        self.linear_attn_config = {
            "short_conv_kernel_size": 4,
            "head_dim": self.head_dim,
            "num_heads": self.num_attention_heads,
        }
        if attn_layer_types is not None:
            self.attn_layer_types = attn_layer_types
        else:
            stage1_end = int(self.num_hidden_layers * 0.2)
            stage2_end = int(self.num_hidden_layers * 0.7)
            stage2_mid = int(self.num_hidden_layers * 0.45)
            self.attn_layer_types = []
            for i in range(self.num_hidden_layers):
                if i < stage1_end:
                    self.attn_layer_types.append("sliding_attention")
                elif i >= stage2_end:
                    self.attn_layer_types.append("linear_attention")
                else:
                    rel_idx = i - stage1_end
                    period = 3 if i < stage2_mid else 2
                    if rel_idx % period == 0:
                        self.attn_layer_types.append("linear_attention")
                    else:
                        self.attn_layer_types.append("sliding_attention")

        # MoE
        self.anchor_alpha = anchor_alpha if anchor_alpha is not None else 0.1
        self.anchor_intermediate_size_ratio = (
            anchor_intermediate_size_ratio if anchor_intermediate_size_ratio is not None else 0.5
        )
        # Experts: d/32 aligned to 32.
        raw_experts = hidden_size // 32
        self.n_routed_experts = (
            n_routed_experts if n_routed_experts is not None else max(8, raw_experts - (raw_experts % 32))
        )
        self.num_experts_per_tok = (
            num_experts_per_tok if num_experts_per_tok is not None else (2 if hidden_size < 4096 else 4)
        )
        self.moe_intermediate_size = (
            moe_intermediate_size
            if moe_intermediate_size is not None
            else (hidden_size // 2) - ((hidden_size // 2) % 64)
        )
        self.routed_scaling_factor = routed_scaling_factor if routed_scaling_factor is not None else hidden_size / 5120
        self.n_group = n_group if n_group is not None else self.n_routed_experts // 8
        self.topk_group = topk_group if topk_group is not None else 1
        anchor_divisor = max(32, hidden_size // 256)
        self.num_local_groups = (
            num_local_groups if num_local_groups is not None else max(2, self.n_routed_experts // anchor_divisor)
        )
        self.n_shared_experts = n_shared_experts if n_shared_experts is not None else max(1, hidden_size // 4096)
        self.anchor_intermediate_size = (
            anchor_intermediate_size
            if anchor_intermediate_size is not None
            else int(self.moe_intermediate_size * 0.5) - (int(self.moe_intermediate_size * 0.5) % 64)
        )
        self.output_router_logits = output_router_logits if output_router_logits is not None else False
        self.router_aux_loss_coef = router_aux_loss_coef if router_aux_loss_coef is not None else 0.01
        raw_inter = hidden_size * (8 / 3)
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else int(raw_inter) - (int(raw_inter) % 64)
        )

        # Tokens
        self.bos_token_id = bos_token_id if bos_token_id is not None else None
        self.eos_token_id = eos_token_id if eos_token_id is not None else [151329, 151336, 151338]
        self.pad_token_id = pad_token_id if pad_token_id is not None else 151329

        super().__init__(**kwargs)


class AevaRMSNorm(Glm4MoeRMSNorm):
    pass


class RosaCache:
    """Rapid Online Suffix Automaton (ROSA) Cache
    Maintains the Suffix Automaton state and necessary buffers for each layer
    to support fast autoregressive decoding with statistical fallback.
    """

    def __init__(self) -> None:
        self.layers: dict[int, Any] = {}

    def get_rosa_context(self, layer_idx: int) -> Any:
        try:
            from rosa_cpp import RosaContext
        except ImportError as e:
            raise ImportError(
                "ROSA requires 'rosa-cpp' to be installed. Please install it via: `pip install git+https://github.com/wjie98/rosa_soft.git#subdirectory=rosa_cpp`"
            ) from e

        if layer_idx not in self.layers:
            self.layers[layer_idx] = RosaContext()
        return self.layers[layer_idx]

    def update(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> Any:
        """Updates the Suffix Automaton state with new tokens.
        Args:
            query: The query tensor for the current step.
            key: The key tensor to be incorporated into the automaton.
            value: The value tensor associated with the keys.
            layer_idx: The index of the transformer layer.
        Returns:
            The output from the ROSA context update (typically attention weights
            or hidden states).
        """
        return self.get_rosa_context(layer_idx).update(query, key, value)


class RosaBase(nn.Module):
    """Rapid Online Suffix Automaton (ROSA) Base Module with Statistical Fallback"""

    def __init__(self) -> None:
        super().__init__()

    def init_rosa(self, config: AevaConfig, layer_idx: int) -> None:
        """Initializes ROSA specific parameters and layers based on the config."""
        self.rosa_layer_idx = layer_idx
        self.hidden_size = getattr(config, "hidden_size")
        if self.hidden_size % 64 != 0:
            raise ValueError("hidden_size must be divisible by 64")
        bias = getattr(config, "attention_bias", False)
        self.rosa_num_qk_bits = 8
        self.rosa_num_v_bits = self.rosa_num_qk_bits
        self.rosa_num_heads = getattr(config, "num_attention_heads", self.hidden_size // 8)
        self.rosa_num_kv_heads = getattr(config, "num_key_value_heads", self.rosa_num_heads)
        if self.rosa_num_kv_heads > self.rosa_num_heads:
            self.rosa_num_kv_heads = self.rosa_num_heads
        max_pos = getattr(config, "max_position_embeddings", 131072)
        if max_pos <= 4096:
            self.rosa_suffix_window = 8
        elif max_pos <= 32768:
            self.rosa_suffix_window = 16
        else:
            self.rosa_suffix_window = 32
        self.rosa_suffix_factor = None
        wb_input_dim = self.hidden_size + 2 * (self.rosa_num_heads * self.rosa_num_v_bits)
        self.rosa_q_proj = nn.Linear(
            self.hidden_size,
            self.rosa_num_heads * self.rosa_num_qk_bits,
            bias=bias,
        )
        self.rosa_k_proj = nn.Linear(
            self.hidden_size,
            self.rosa_num_kv_heads * self.rosa_num_qk_bits,
            bias=bias,
        )
        self.rosa_v_proj = nn.Linear(
            self.hidden_size,
            self.rosa_num_kv_heads * self.rosa_num_v_bits,
            bias=bias,
        )
        self.rosa_o_proj = nn.Linear(
            self.rosa_num_heads * self.rosa_num_v_bits,
            self.hidden_size,
            bias=bias,
        )
        self.rosa_fallback_proj = nn.Linear(
            self.hidden_size,
            self.rosa_num_heads * self.rosa_num_v_bits,
            bias=bias,
        )
        self.rosa_wb_gate = nn.Sequential(
            nn.Linear(wb_input_dim, self.hidden_size // 4),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        self.rosa_v_emb0 = nn.Parameter(torch.zeros(self.rosa_num_heads * self.rosa_num_v_bits))
        self.rosa_v_emb1 = nn.Parameter(torch.zeros(self.rosa_num_heads * self.rosa_num_v_bits))
        self.rosa_o_gate = nn.Parameter(torch.zeros(self.hidden_size))
        self._reset_parameters(config)

    def rosa_dispatch(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
    ) -> tuple[Any, torch.Tensor]:
        try:
            from rosa_cpp import rosa_bits_ops
        except ImportError as e:
            raise ImportError(
                "ROSA requires 'rosa-cpp' to be installed. Please install it via: `pip install git+https://github.com/wjie98/rosa_soft.git#subdirectory=rosa_cpp`"
            ) from e

        batch_size, sequence_length, _ = hidden_states.size()
        query_states = self.rosa_q_proj(hidden_states)
        query_states = query_states.view(
            batch_size, sequence_length, self.rosa_num_heads, self.rosa_num_qk_bits
        ).transpose(1, 2)
        key_states = self.rosa_k_proj(hidden_states)
        key_states = key_states.view(
            batch_size, sequence_length, self.rosa_num_kv_heads, self.rosa_num_qk_bits
        ).transpose(1, 2)
        value_states = self.rosa_v_proj(hidden_states)
        value_states = value_states.view(
            batch_size, sequence_length, self.rosa_num_kv_heads, self.rosa_num_v_bits
        ).transpose(1, 2)
        if past_key_values is None:
            work = rosa_bits_ops(
                query_states,
                key_states,
                value_states,
                suffix_window=self.rosa_suffix_window,
                suffix_factor=self.rosa_suffix_factor,
                attention_mask=attention_mask,
                async_op=True,
            )
            return (work, hidden_states)
        else:
            if not hasattr(past_key_values, "_rosa_cache"):
                past_key_values._rosa_cache = RosaCache()
            cache = past_key_values._rosa_cache
            work = cache.get_rosa_context(layer_idx=self.rosa_layer_idx).update(
                query_states, key_states, value_states, 0, async_op=True
            )
            return (work, hidden_states)

    def rosa_combine(
        self,
        states: tuple[Any, torch.Tensor],
        inject_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combines the results from ROSA attention and statistical fallback.
        Args:
            states: A tuple containing the work object and original hidden states.
            inject_states: Optional tensor for residual injection.
        Returns:
            The final output tensor after combination and projection.
        """
        work, hidden_states = states
        batch_size, sequence_length, _ = hidden_states.size()
        sam_output_flat = work.wait().view(batch_size, sequence_length, -1)
        fb_output_flat = self.rosa_fallback_proj(hidden_states)
        gate_input = torch.cat([hidden_states, sam_output_flat, fb_output_flat], dim=-1)
        lam = self.rosa_wb_gate(gate_input)
        mixed_output_flat = lam * sam_output_flat + (1.0 - lam) * fb_output_flat
        output = mixed_output_flat.view(batch_size, -1, sequence_length).transpose(1, 2).contiguous()
        output = self.rosa_v_emb1 * output + self.rosa_v_emb0 * (1.0 - output)
        output = self.rosa_o_proj(output)
        if inject_states is not None:
            gate = torch.sigmoid(self.rosa_o_gate)
            output = output * gate + inject_states * (1.0 - gate)
        return output

    def _reset_parameters(self, config: AevaConfig) -> None:
        """Initializes layer weights following specific ROSA heuristics."""
        num_layers = getattr(config, "num_hidden_layers", 1)
        scale = 0.02 / max(1, num_layers)
        self.rosa_q_proj.weight.data.uniform_(-scale, scale)
        q_w = self.rosa_q_proj.weight.data
        k_w = self.rosa_k_proj.weight.data
        if q_w.size() == k_w.size():
            k_w.copy_(q_w)
        else:
            if k_w.size(0) <= q_w.size(0):
                k_w.copy_(q_w[: k_w.size(0)])
            else:
                repeats = int(math.ceil(k_w.size(0) / q_w.size(0)))
                rep = q_w.repeat(repeats, 1)[: k_w.size(0)]
                k_w.copy_(rep)
        s0 = self.rosa_v_proj.weight.size(0)
        s1 = self.rosa_v_proj.weight.size(1)
        gain = max(1.0, math.sqrt(s0 / s1))
        nn.init.orthogonal_(self.rosa_v_proj.weight.data, gain=gain)
        nn.init.zeros_(self.rosa_o_proj.weight.data)
        if self.rosa_o_proj.bias is not None:
            nn.init.zeros_(self.rosa_o_proj.bias.data)
        self.rosa_v_emb0.data.fill_(-1e-5)
        self.rosa_v_emb1.data.fill_(1e-5)
        self.rosa_o_gate.data.zero_()
        nn.init.xavier_normal_(self.rosa_fallback_proj.weight.data, gain=0.1)
        if self.rosa_fallback_proj.bias is not None:
            nn.init.zeros_(self.rosa_fallback_proj.bias.data)


@use_kernelized_func(apply_rotary_pos_emb)
class AevaAttention(RosaBase):
    """DeepEmbed Sliding Window Suffix Attention"""

    def __init__(self, config: AevaConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_parameters = config.rope_parameters
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_qk_norm = config.use_qk_norm
        self.sliding_window = getattr(config, "sliding_window", None)
        self.init_rosa(config, layer_idx)
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_low_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * (self.head_dim // 4),
            bias=config.attention_bias,
        )
        self.k_up_proj = nn.Linear(
            config.num_key_value_heads * (self.head_dim // 4),
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.k_deepemb = nn.Embedding(config.vocab_size, config.num_key_value_heads * self.head_dim)
        self.v_low_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * (self.head_dim // 4),
            bias=config.attention_bias,
        )
        self.v_up_proj = nn.Linear(
            config.num_key_value_heads * (self.head_dim // 4),
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_deepemb = nn.Embedding(config.vocab_size, config.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        if self.use_qk_norm:
            self.q_norm = AevaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = AevaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rosa_states = self.rosa_dispatch(
            hidden_states,
            attention_mask,
            past_key_values,
        )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = query_states.view(hidden_shape)
        if input_ids is not None:
            ids = input_ids[:, -hidden_states.shape[1] :]
        else:
            ids = torch.zeros_like(hidden_states[..., :1]).long()
        k_low = self.k_low_proj(hidden_states)
        k_up = self.k_up_proj(k_low)
        k_emb = self.k_deepemb(ids).to(k_up.dtype)
        key_states = k_up.view(*input_shape, -1, self.head_dim) * k_emb.view(*input_shape, -1, self.head_dim)
        v_low = self.v_low_proj(hidden_states)
        v_up = self.v_up_proj(v_low)
        v_emb = self.v_deepemb(ids).to(v_up.dtype)
        value_states = v_up.view(*input_shape, -1, self.head_dim) * v_emb.view(*input_shape, -1, self.head_dim)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface: Callable = eager_attention_forward
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
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        attn_output = self.rosa_combine(rosa_states, attn_output)
        return attn_output, attn_weights


class AevaDynamicCache:
    """Dynamic Cache with Synaptic Overwriting for Heterogeneous Attention States."""

    is_compileable = False

    def __init__(self, config: "AevaConfig"):
        super().__init__()
        self.config = config
        self.attn_layer_types = (
            config.attn_layer_types if config.attn_layer_types else ["sliding_attention"] * config.num_hidden_layers
        )
        self.max_cache_size = config.sliding_window or 4096
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.attn_layer_types[i] == "sliding_attention"
        ]
        linear_layers = [i for i in range(config.num_hidden_layers) if self.attn_layer_types[i] == "linear_attention"]
        self.last_linear_layer = linear_layers[-1] if linear_layers else -1
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]

    def __len__(self) -> int:
        """Return number of attention layers."""
        return len(self.attn_layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a specific layer with a rolling buffer strategy."""
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            curr_len = self.key_cache[layer_idx].shape[-2]
            if curr_len >= self.max_cache_size:
                shift = key_states.shape[-2]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].roll(-shift, -2)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].roll(-shift, -2)
                self.key_cache[layer_idx][..., -shift:, :] = key_states
                self.value_cache[layer_idx][..., -shift:, :] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)
            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx][0].device
                beam_idx = beam_idx.to(device)
                q_conv, k_conv, v_conv = self.conv_states[layer_idx]
                self.conv_states[layer_idx] = (
                    q_conv.index_select(0, beam_idx),
                    k_conv.index_select(0, beam_idx),
                    v_conv.index_select(0, beam_idx),
                )
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Return the sequence length of the cached states."""
        if self.transformer_layers and layer_idx not in self.transformer_layers:
            layer_idx = self.transformer_layers[0]
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length
        and offset that will be returned for the given layer at `layer_idx`.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self) -> bool:
        """Return True if we have a previous state."""
        if self.last_linear_layer == -1:
            return False
        return self.conv_states[self.last_linear_layer] is not None


class SparseProjection(nn.Linear):
    """Event-Driven Sparse Linear Projection."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for sparse linear projection.
        Applies the linear transformation to the input tensor `x`, and if a
        `cache_mask` attribute exists and is not None, applies element-wise
        multiplication with the mask to sparsify the output.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., out_features).
        """
        proj = super().forward(x)
        if hasattr(self, "cache_mask") and self.cache_mask is not None:
            proj = proj * self.cache_mask
        return proj


class AevaLinearAttention(RosaBase):
    """DeepEmbed Linear Suffix Attention."""

    def __init__(self, config: AevaConfig, layer_idx: int):
        super().__init__()
        try:
            from fla.modules import FusedRMSNormGated, ShortConvolution
        except ImportError as e:
            raise ImportError(
                "AevaLinearAttention requires 'fla-core' to be installed. Please install it via: `pip install -U fla-core`"
            ) from e
        self.config = config
        self.layer_idx = layer_idx
        self.mode = "chunk"
        if self.mode not in ["chunk", "fused_recurrent"]:
            raise ValueError(f"Not supported mode `{self.mode}`.")
        self.hidden_size = config.hidden_size
        linear_config = config.linear_attn_config
        self.head_dim = linear_config["head_dim"]
        self.num_heads = linear_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads
        self.conv_size = linear_config["short_conv_kernel_size"]
        kv_low_dim = self.head_dim // 4
        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads
        self.init_rosa(config, layer_idx)
        self.q_proj = SparseProjection(self.hidden_size, projection_k_size, bias=False)
        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation=self.config.hidden_act,
        )
        self.k_proj = SparseProjection(self.hidden_size, self.num_k_heads * kv_low_dim, bias=False)
        self.k_up_proj = nn.Linear(self.num_k_heads * kv_low_dim, projection_k_size, bias=False)
        self.k_deepemb = nn.Embedding(config.vocab_size, projection_k_size)
        self.k_conv1d = ShortConvolution(
            hidden_size=self.num_k_heads * kv_low_dim,
            kernel_size=self.conv_size,
            activation=self.config.hidden_act,
        )
        self.v_proj = SparseProjection(self.hidden_size, self.num_heads * kv_low_dim, bias=False)
        self.v_up_proj = nn.Linear(self.num_heads * kv_low_dim, projection_size, bias=False)
        self.v_deepemb = nn.Embedding(config.vocab_size, projection_size)
        self.v_conv1d = ShortConvolution(
            hidden_size=self.num_heads * kv_low_dim,
            kernel_size=self.conv_size,
            activation=self.config.hidden_act,
        )
        self.A_log = torch.nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1)
        )
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)
        nn.init.zeros_(self.k_deepemb.weight)
        nn.init.zeros_(self.v_deepemb.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: AevaDynamicCache | None = None,
        input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        try:
            from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
            from fla.ops.kda.gate import fused_kda_gate
        except ImportError as e:
            raise ImportError(
                "AevaLinearAttention requires 'fla-core' to be installed. Please install it via: `pip install -U fla-core`"
            ) from e
        rosa_states = self.rosa_dispatch(hidden_states, attention_mask)
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")
            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here."
                )
        use_cache = cache_params is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            if mode != "chunk":
                raise ValueError("Only chunk mode is supported in training.")
        ids = None
        if input_ids is not None:
            if attention_mask is None:
                ids = input_ids[:, -q_len:]
        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            if input_ids is not None:
                flat_ids = rearrange(input_ids[:, -q_len:], "b s -> (b s)")
                ids = flat_ids[indices].unsqueeze(0)
        if ids is None and input_ids is not None:
            ids = input_ids[:, -q_len:]
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        recurrent_state = None
        if cache_params is not None:
            if cache_params.conv_states[self.layer_idx] is not None:
                (
                    conv_state_q,
                    conv_state_k,
                    conv_state_v,
                ) = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        if cache_params is not None and recurrent_state is not None:
            mask = torch.zeros_like(hidden_states)
            mask[:, -1:, :] = 1.0
            self.q_proj.cache_mask = mask
            k_low_dim = self.num_k_heads * (self.head_dim // 4)
            k_mask = torch.zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                k_low_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            k_mask[:, -1:, :] = 1.0
            self.k_proj.cache_mask = k_mask
            v_low_dim = self.num_heads * (self.head_dim // 4)
            v_mask = torch.zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                v_low_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            v_mask[:, -1:, :] = 1.0
            self.v_proj.cache_mask = v_mask
        else:
            self.q_proj.cache_mask = None
            self.k_proj.cache_mask = None
            self.v_proj.cache_mask = None
        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k_low, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        k_up = self.k_up_proj(k_low)
        if ids is not None:
            k = k_up * self.k_deepemb(ids).to(k_up.dtype)
        else:
            k = k_up
        v_low, conv_state_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        v_up = self.v_up_proj(v_low)
        if ids is not None:
            v = v_up * self.v_deepemb(ids).to(v_up.dtype)
        else:
            v = v_up
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(g, self.A_log, self.dt_bias)
        beta = self.b_proj(hidden_states).float().sigmoid()
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)
        if g.dim() == 3:
            g = rearrange(g, "... (h d) -> ... h d", h=self.num_heads, d=self.head_dim)
        elif g.dim() != 4:
            raise ValueError(f"Unexpected g dimension: {g.dim()}")
        gate_intensity = g.abs().mean(dim=-1)
        threshold = gate_intensity.quantile(0.9, dim=1, keepdim=True)
        spike_mask = (gate_intensity > threshold).float()
        g_sparse = g * spike_mask.unsqueeze(-1)
        beta_sparse = beta * spike_mask
        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g_sparse,
                beta=beta_sparse,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g_sparse,
                beta=beta_sparse,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = recurrent_state
            cache_params.conv_states[self.layer_idx] = (
                conv_state_q,
                conv_state_k,
                conv_state_v,
            )
        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, g)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        o = self.rosa_combine(rosa_states, o)
        return o, None


class AevaMLP(Glm4MoeMLP):
    pass


class AevaTopkRouter(Glm4MoeTopkRouter):
    def __init__(self, config: AevaConfig):
        super().__init__()
        del self.top_k
        del self.routed_scaling_factor
        del self.n_group
        del self.topk_group
        del self.norm_topk_prob


class AevaNaiveMoe(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: AevaConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.config = config
        self.num_groups = config.num_local_groups
        self.group_size = self.num_experts // self.num_groups
        self.tile_size = self._calculate_tile_size()

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_modules: nn.ModuleList,
        anchor_modules: nn.ModuleList,
    ) -> torch.Tensor:
        """
        Forward pass for ExpertKernel.
        Args:
            hidden_states (torch.Tensor): Input tensor [Batch * SeqLen, HiddenDim].
            topk_indices (torch.Tensor): Selected expert indices
                [Batch * SeqLen, TopK].
            topk_weights (torch.Tensor): Routing weights [Batch * SeqLen, TopK].
            expert_modules (nn.ModuleList): List of Specialist MLPs (owned by
                HGSCMoE).
            anchor_modules (nn.ModuleList): List of Anchor MLPs (owned by
                HGSCMoE).
        Returns:
            torch.Tensor: Processed hidden states.
        """
        bt, hidden_dim = hidden_states.shape
        top_k = topk_indices.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        inputs = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
        flat_indices = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1, 1)

        group_ids = flat_indices // self.group_size
        local_ids = flat_indices % self.group_size

        sort_keys = group_ids * 1000 + local_ids
        sorted_indices = torch.argsort(sort_keys, stable=True)

        sorted_inputs = inputs[sorted_indices]
        sorted_weights = flat_weights[sorted_indices]
        sorted_group_ids = group_ids[sorted_indices]
        sorted_local_ids = local_ids[sorted_indices]

        tile_size = self.tile_size
        total_assignments = bt * top_k
        output_buffer = torch.empty(total_assignments, hidden_dim, device=device, dtype=torch.float32)

        unique_groups, group_sizes = torch.unique_consecutive(sorted_group_ids, return_counts=True)
        current_offset = 0

        for i, group_id in enumerate(unique_groups.tolist()):
            group_size = group_sizes[i].item()
            start = current_offset
            end = current_offset + group_size
            current_offset = end

            group_inputs = sorted_inputs[start:end]
            group_indices = sorted_indices[start:end]

            anchor = anchor_modules[group_id]

            n_tokens = group_size
            rounded_num = ((n_tokens + tile_size - 1) // tile_size) * tile_size
            if rounded_num > n_tokens:
                padding = rounded_num - n_tokens
                group_inputs_padded = F.pad(group_inputs, (0, 0, 0, padding))
            else:
                group_inputs_padded = group_inputs

            if self.training:
                anchor_out = torch.utils.checkpoint.checkpoint(anchor, group_inputs_padded, use_reentrant=False)
            else:
                anchor_out = anchor(group_inputs_padded)

            if rounded_num > n_tokens:
                anchor_out = anchor_out[:n_tokens]

            output_buffer[group_indices] += anchor_out * self.config.anchor_alpha

            sub_local_ids = sorted_local_ids[start:end]
            local_sort_args = torch.argsort(sub_local_ids, stable=True)

            local_sorted_weights = sorted_weights[start:end][local_sort_args]
            local_sorted_inputs = group_inputs[local_sort_args]
            local_sorted_buffer_indices = group_indices[local_sort_args]

            unique_locals, local_sizes = torch.unique_consecutive(sub_local_ids[local_sort_args], return_counts=True)

            local_offset = 0
            for j, local_id in enumerate(unique_locals.tolist()):
                local_size = local_sizes[j].item()
                l_start = local_offset
                l_end = local_offset + local_size
                local_offset = l_end

                expert_inputs = local_sorted_inputs[l_start:l_end]
                expert_weights = local_sorted_weights[l_start:l_end]

                e_tokens = local_size
                e_rounded = ((e_tokens + tile_size - 1) // tile_size) * tile_size
                if e_rounded > e_tokens:
                    e_padding = e_rounded - e_tokens
                    expert_inputs_padded = F.pad(expert_inputs, (0, 0, 0, e_padding))
                    expert_weights = F.pad(expert_weights, (0, 0, 0, e_padding))
                else:
                    expert_inputs_padded = expert_inputs

                expert_idx = group_id * self.group_size + local_id
                expert = expert_modules[expert_idx]

                use_checkpoint = self.training and (e_rounded > 128)
                if use_checkpoint:
                    expert_out = torch.utils.checkpoint.checkpoint(expert, expert_inputs_padded, use_reentrant=False)
                else:
                    expert_out = expert(expert_inputs_padded)

                if e_rounded > e_tokens:
                    expert_out = expert_out[:e_tokens]
                    expert_weights = expert_weights[:e_tokens]

                target_indices = local_sorted_buffer_indices[l_start:l_end]
                output_buffer[target_indices] += expert_out * expert_weights

        inverse_indices = torch.argsort(sorted_indices)
        final_flat_output = output_buffer[inverse_indices]
        final_hidden_states = final_flat_output.view(bt, top_k, hidden_dim).sum(dim=1)

        return final_hidden_states.to(dtype=dtype)

    def _calculate_tile_size(self) -> int:
        """Compute optimal TILE_SIZE using GCD."""
        ref_dim = self.config.moe_intermediate_size
        optimal_tile = math.gcd(ref_dim, 256)
        return max(32, optimal_tile)


class AevaMoE(nn.Module):
    """
    Hierarchical Grouped Sparse Computation (MoE) with Shared Capacity.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [AevaMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.n_routed_experts)]
        )
        self.gate = AevaTopkRouter(config)
        self.shared_experts = AevaMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        self.num_groups_for_anchors = config.num_local_groups
        self.group_size_for_anchors = self.n_routed_experts // self.num_groups_for_anchors
        self.anchors = nn.ModuleList(
            [
                AevaMLP(config, intermediate_size=config.anchor_intermediate_size)
                for _ in range(self.num_groups_for_anchors)
            ]
        )
        self.expert_kernel = AevaNaiveMoe(config)
        self.leak_factor = nn.Parameter(torch.tensor(0.9))
        self.firing_threshold = nn.Parameter(torch.ones(self.n_routed_experts) * 0.1)
        self.register_buffer("leak_min", torch.tensor(0.0))
        self.register_buffer("leak_max", torch.tensor(1.0))
        self.register_buffer("expert_potential", torch.zeros(self.n_routed_experts))

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implement brain-inspired, event-driven routing.
        Softmax activation (Lateral Inhibition).
        Leaky Integration of potentials (Temporal Memory).
        Threshold-based Firing (Event-Driven Sparsity).
        Hierarchical Group Selection.
        """
        scores = F.softmax(router_logits, dim=-1)
        leak = torch.clamp(self.leak_factor, self.leak_min, self.leak_max)
        threshold = F.softplus(self.firing_threshold)
        self.expert_potential = leak * self.expert_potential + scores.mean(dim=0)
        firing_mask = (self.expert_potential > threshold).float()
        if firing_mask.any():
            self.expert_potential = self.expert_potential - firing_mask * threshold
        masked_scores = scores * firing_mask.unsqueeze(0)
        scores_for_choice = masked_scores.view(-1, self.n_routed_experts)
        group_scores = scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group).sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HGSCMoE layer.
        Args:
            hidden_states: Input hidden states
        Returns:
            Tuple of (processed hidden states, router logits)
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.expert_kernel(
            flat_hidden_states,
            topk_indices,
            topk_weights,
            expert_modules=self.experts,
            anchor_modules=self.anchors,
        ).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states, router_logits


def sinkhorn_knopp(matrix: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Project matrix to doubly stochastic manifold."""
    matrix = torch.exp(matrix)
    for _ in range(iterations):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-8)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-8)
    return matrix


class AevaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: AevaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.attn_layer_types[layer_idx]
        self.num_streams = getattr(config, "num_streams", 4)
        stream_dim = self.hidden_size * self.num_streams
        use_moe = layer_idx >= config.first_k_dense_replace
        is_linear_attn = self.layer_type == "linear_attention"
        self.input_layernorm = AevaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        if is_linear_attn:
            self.self_attn = AevaLinearAttention(config, layer_idx)
        else:
            self.self_attn = AevaAttention(config, layer_idx)
        self.post_attention_layernorm = AevaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        if use_moe:
            self.mlp = AevaMoE(config)
        else:
            self.mlp = AevaMLP(config, intermediate_size=config.intermediate_size)
        self.rosa_norm = AevaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.rosa = RosaBase()
        self.rosa.init_rosa(config, layer_idx)
        self.mhc_alpha = nn.Parameter(torch.tensor([0.01, 0.01, 0.01]))
        self.mhc_norm = AevaRMSNorm(stream_dim, eps=config.rms_norm_eps)
        self.mhc_bias_pre = nn.Parameter(torch.zeros(self.num_streams))
        self.mhc_phi_pre = nn.Linear(stream_dim, self.num_streams, bias=False)
        self.mhc_bias_post = nn.Parameter(torch.zeros(self.num_streams))
        self.mhc_phi_post = nn.Linear(stream_dim, self.num_streams, bias=False)
        self.mhc_bias_res = nn.Parameter(torch.zeros(self.num_streams, self.num_streams))
        self.mhc_phi_res = nn.Linear(stream_dim, self.num_streams * self.num_streams, bias=False)
        self.deepemb = nn.Embedding(config.vocab_size, self.hidden_size)
        nn.init.ones_(self.deepemb.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        residual = hidden_states
        B, S, C = residual.shape
        n = self.num_streams
        stream_base = residual.unsqueeze(1).repeat(1, n, 1, 1)
        x_flat = stream_base.permute(0, 2, 1, 3).contiguous().view(B * S, -1)
        x_norm = self.mhc_norm(x_flat)
        dyn_pre = torch.tanh(self.mhc_phi_pre(x_norm))
        dyn_post = torch.tanh(self.mhc_phi_post(x_norm))
        dyn_res = torch.tanh(self.mhc_phi_res(x_norm)).view(B * S, n, n)
        H_pre = (self.mhc_alpha[0] * dyn_pre + self.mhc_bias_pre).view(B, S, n)
        H_post = (self.mhc_alpha[1] * dyn_post + self.mhc_bias_post).view(B, S, n)
        H_res = (self.mhc_alpha[2] * dyn_res + self.mhc_bias_res).view(B, S, n, n)
        H_pre = torch.sigmoid(H_pre)
        H_post = 2.0 * torch.sigmoid(H_post)
        H_res = sinkhorn_knopp(H_res)
        block_input = torch.einsum("bsn,bnsc->bsc", H_pre, stream_base)
        rosa_output = self.rosa.rosa_combine(
            states=self.rosa.rosa_dispatch(
                hidden_states=self.rosa_norm(block_input),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            ),
            inject_states=block_input,
        )
        attn_input = self.input_layernorm(rosa_output)
        if self.layer_type == "linear_attention":
            attn_output, _ = self.self_attn(
                hidden_states=attn_input,
                attention_mask=attention_mask,
                cache_params=past_key_values,
                **kwargs,
            )
        else:
            attn_output, _ = self.self_attn(
                hidden_states=attn_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = attn_input + attn_output
        mlp_input = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, AevaMoE):
            block_output, router_logits = self.mlp(mlp_input)
        else:
            block_output = self.mlp(mlp_input)
            router_logits = None
        if input_ids is not None:
            ids = input_ids[:, -S:] if input_ids.shape[1] != S else input_ids
            scale = self.deepemb(ids).to(block_output.dtype)
            block_output = block_output * scale
        block_output = mlp_input + block_output
        stream_update = block_output.unsqueeze(2) * H_post.unsqueeze(-1)
        stream_residual = torch.einsum("bsij,bjsc->bsic", H_res, stream_base)
        next_stream = stream_update + stream_residual
        output = next_stream.mean(dim=2)
        return output, router_logits


class AevaRotaryEmbedding(Glm4MoeRotaryEmbedding):
    pass


@auto_docstring
class AevaPreTrainedModel(PreTrainedModel):
    config: AevaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AevaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = is_grouped_mm_available()
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": AevaDecoderLayer,
        "attentions": [AevaAttention, AevaLinearAttention],
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, AevaTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class AevaModel(Glm4MoeModel):
    def __init__(self, config: AevaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [AevaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = AevaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AevaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _update_linear_attn_mask(
        self, attention_mask: torch.Tensor | None, cache_position: torch.LongTensor
    ) -> torch.Tensor | None:
        """
        Update linear attention mask.
        Note: Left-padding is used for linear attention mask.
        No need for zeroing states when:
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache:
            past_key_values = AevaDynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        all_router_logits = () if output_router_logits else None
        if self.config.sliding_window is None:
            mask_function = create_causal_mask
        else:
            mask_function = create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if decoder_layer.layer_type == "linear_attention":
                layer_mask = linear_attn_mask
                layer_pos_embeds = None
            else:
                layer_mask = causal_mask
                layer_pos_embeds = position_embeddings
            hidden_states, router_logits = decoder_layer(
                hidden_states,
                input_ids=input_ids,
                attention_mask=layer_mask,
                position_embeddings=layer_pos_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if output_router_logits and router_logits is not None:
                all_router_logits = all_router_logits + (router_logits,)
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=all_router_logits if output_router_logits else None,
        )


def load_balancing_loss_for_grouped_routing(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k: int = 2,
    attention_mask: torch.Tensor | None = None,
    num_groups: int = 1,
    topk_groups: int = 1,
) -> torch.Tensor | int:
    """
    Computes auxiliary load balancing loss for grouped routing.
    Args:
        gate_logits: Logits from the gate, tuple of tensors with shape
            [batch_size * sequence_length, num_experts].
        num_experts: Number of experts.
        top_k: Number of experts to route per-token.
        attention_mask: Attention mask with shape [batch_size, sequence_length].
        num_groups: Number of expert groups.
        topk_groups: Number of groups selected per token.
    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple) or num_experts is None:
        return 0
    compute_device = gate_logits[0].device
    routing_weights = F.softmax(
        torch.cat([g.to(compute_device, dtype=torch.float32) for g in gate_logits], dim=0),
        dim=-1,
    )
    batch_seq_len, _ = routing_weights.shape
    if num_groups > 1:
        experts_per_group = num_experts // num_groups
        group_weights = routing_weights.view(batch_seq_len, num_groups, experts_per_group)
        group_scores = group_weights.sum(dim=-1)
        _, selected_groups = torch.topk(group_scores, k=topk_groups, dim=-1, sorted=False)
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, selected_groups, 1)
        expert_mask_from_groups = (
            group_mask.unsqueeze(-1).expand(-1, num_groups, experts_per_group).reshape(batch_seq_len, num_experts)
        )
        masked_routing_weights = routing_weights * expert_mask_from_groups
        _, selected_experts = torch.topk(masked_routing_weights, top_k, dim=-1)
    else:
        selected_experts = torch.topk(routing_weights, top_k, dim=-1).indices
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        if num_groups > 1:
            effective_weights = routing_weights * expert_mask_from_groups
            group_norm = effective_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            normalized_weights = effective_weights / group_norm
            router_prob_per_expert = torch.mean(normalized_weights, dim=0)
        else:
            router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        num_hidden_layers = len(gate_logits)
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (
                    num_hidden_layers,
                    attention_mask.size(0),
                    attention_mask.size(1),
                    top_k,
                    num_experts,
                )
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / (
            torch.sum(expert_attention_mask, dim=0) + 1e-12
        )
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand(
                (
                    num_hidden_layers,
                    attention_mask.size(0),
                    attention_mask.size(1),
                    num_experts,
                )
            )
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        if num_groups > 1:
            effective_weights = routing_weights * expert_mask_from_groups
            group_norm = effective_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            normalized_weights = effective_weights / group_norm
            router_prob_per_expert = torch.sum(normalized_weights * router_per_expert_attention_mask, dim=0) / (
                torch.sum(router_per_expert_attention_mask, dim=0) + 1e-12
            )
        else:
            router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / (
                torch.sum(router_per_expert_attention_mask, dim=0) + 1e-12
            )
    overall_loss = torch.sum((tokens_per_expert + 1e-12) * (router_prob_per_expert.unsqueeze(0) + 1e-12))
    return overall_loss * num_experts


class AevaForCausalLM(Glm4MoeForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_router_logits: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        """
        Forward pass for the model.
        Example:
        ```python
        >>> from transformers import AutoTokenizer, AevaForCausalLM
        >>> model = AevaForCausalLM.from_pretrained("meta-glm4_moe/Aeva-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-glm4_moe/Aeva-2-7b-hf")
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
        >>>                        clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious,
        but I can talk to you."
        ```
        """
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        need_router_logits = self.training and (labels is not None) and output_router_logits
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_router_logits=need_router_logits,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        aux_loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
            if self.training and (outputs.router_logits is not None):
                aux_loss = load_balancing_loss_for_grouped_routing(
                    gate_logits=outputs.router_logits,
                    num_experts=self.config.n_routed_experts,
                    top_k=self.config.num_experts_per_tok,
                    attention_mask=attention_mask,
                    num_groups=self.config.n_group,
                    topk_groups=self.config.topk_group,
                )
                loss = loss + self.config.router_aux_loss_coef * aux_loss
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits if need_router_logits else None,
        )


__all__ = [
    "AevaConfig",
    "AevaPreTrainedModel",
    "AevaModel",
    "AevaForCausalLM",
]
