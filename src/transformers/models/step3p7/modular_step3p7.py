# Copyright 2026 The StepFun and HuggingFace Inc. teams. All rights reserved.
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
import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.dinov2.modeling_dinov2 import Dinov2LayerScale, Dinov2MLP
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm, repeat_kv
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging


STEP3P7_VISION_ENCODER_CONFIG_ARGS = r"""
    width (`int`, *optional*, defaults to 1536):
        Hidden size of the vision encoder.
    layers (`int`, *optional*, defaults to 47):
        Number of hidden layers in the vision encoder.
    heads (`int`, *optional*, defaults to 16):
        Number of attention heads in the vision encoder.
    num_channels (`int`, *optional*, defaults to 3):
        Number of input image channels.
    image_size (`int`, *optional*, defaults to 728):
        Size of the full image input expected by the vision encoder.
    mlp_ratio (`float`, *optional*, defaults to `8960 / 1536`):
        Ratio used to derive the intermediate size of the vision MLP layers.
    patch_size (`int`, *optional*, defaults to 14):
        Patch size used by the vision encoder.
    hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
        Activation function used by the vision encoder MLP layers.
    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
        Epsilon used by layer normalization in the vision encoder.
    use_cls_token (`bool`, *optional*, defaults to `False`):
        Whether to prepend a class token to the vision patch sequence.
    use_ln_pre (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization before the vision transformer.
    use_ln_post (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization after the vision transformer.
    use_abs_posemb (`bool`, *optional*, defaults to `True`):
        Whether to add absolute position embeddings in the vision encoder.
    use_rope2d (`bool`, *optional*, defaults to `True`):
        Whether to use 2D rotary position embeddings in the vision encoder.
    ls_init_value (`float`, *optional*, defaults to 0.1):
        Initial value for layer scale parameters in the vision encoder.
"""

STEP3P7_TEXT_CONFIG_ARGS = r"""
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the decoder hidden states.
    intermediate_size (`int`, *optional*, defaults to 11264):
        Dimensionality of dense MLP intermediate states.
    num_attention_heads (`int`, *optional*, defaults to 64):
        Number of attention heads for each attention layer.
    num_attention_groups (`int`, *optional*, defaults to 8):
        Number of key/value attention groups.
    num_hidden_layers (`int`, *optional*, defaults to 45):
        Number of decoder layers.
    vocab_size (`int`, *optional*, defaults to 128815):
        Vocabulary size of the text model.
    rms_norm_eps (`float`, *optional*, defaults to 1e-5):
        Epsilon used by RMS normalization layers.
    moe_intermediate_size (`int`, *optional*, defaults to 1280):
        Intermediate size of routed MoE experts.
    moe_num_experts (`int`, *optional*, defaults to 288):
        Number of routed MoE experts.
    moe_top_k (`int`, *optional*, defaults to 8):
        Number of experts selected per token.
    rope_theta (`float`, *optional*, defaults to 10000):
        Base period used by rotary position embeddings.
    rope_scaling (`dict[str, Any]`, *optional*):
        Rotary embedding scaling configuration.
    max_position_embeddings (`int`, *optional*, defaults to 128000):
        Maximum position embedding index supported by the model.
    share_expert_dim (`int`, *optional*, defaults to 1280):
        Intermediate size of shared experts.
    head_dim (`int`, *optional*, defaults to 128):
        Dimensionality of each attention head.
    layer_types (`list[str]`, *optional*):
        Attention type for each decoder layer.
    sliding_window (`int`, *optional*):
        Sliding window size for sliding attention layers.
    pad_token_id (`int`, *optional*, defaults to 1):
        Padding token id.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for attention weights.
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation used to initialize model weights.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether the model should return key/value caches.
    use_head_wise_attn_gate (`bool`, *optional*, defaults to `False`):
        Whether to use head-wise attention gates.
    use_moe_router_bias (`bool`, *optional*, defaults to `False`):
        Whether MoE router projections use a bias term.
    moe_router_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scaling factor applied to MoE router scores.
    need_fp32_gate (`bool`, *optional*, defaults to `False`):
        Whether to compute router gates in float32.
    attention_other_setting (`dict[str, Any]`, *optional*):
        Additional attention settings from original checkpoints.
    swiglu_limits (`list[float]`, *optional*):
        Clamp limits for routed expert SwiGLU activations.
    swiglu_limits_shared (`list[float]`, *optional*):
        Clamp limits for shared expert SwiGLU activations.
    yarn_only_types (`list[str]`, *optional*):
        Layer type names that should use YaRN-style RoPE settings only.
    moe_layers_enum (`tuple[int]`, *optional*):
        Indices of layers that use MoE blocks.
"""

STEP3P7_CONFIG_ARGS = r"""
    vision_config (`dict` or `Step3p7VisionEncoderConfig`, *optional*):
        Configuration of the Step3p7 vision encoder.
    text_config (`dict` or `Step3p7TextConfig`, *optional*):
        Configuration of the Step3p7 text decoder.
    projector_bias (`bool`, *optional*, defaults to `False`):
        Whether the multimodal projector uses a bias term.
    image_token_id (`int`, *optional*, defaults to 151679):
        Token id used as image placeholder in text inputs.
"""


@strict
@auto_docstring(custom_args=STEP3P7_VISION_ENCODER_CONFIG_ARGS, checkpoint="stepfun-ai/Step-3.7-Flash")
class Step3p7VisionEncoderConfig(PreTrainedConfig):
    r"""
    width (`int`, *optional*, defaults to 1536):
        Hidden size of the vision encoder.
    layers (`int`, *optional*, defaults to 47):
        Number of hidden layers in the vision encoder.
    heads (`int`, *optional*, defaults to 16):
        Number of attention heads in the vision encoder.
    num_channels (`int`, *optional*, defaults to 3):
        Number of input image channels.
    image_size (`int`, *optional*, defaults to 728):
        Size of the full image input expected by the vision encoder.
    mlp_ratio (`float`, *optional*, defaults to `8960 / 1536`):
        Ratio used to derive the intermediate size of the vision MLP layers.
    patch_size (`int`, *optional*, defaults to 14):
        Patch size used by the vision encoder.
    hidden_act (`str`, *optional*, defaults to `"quick_gelu"`):
        Activation function used by the vision encoder MLP layers.
    use_cls_token (`bool`, *optional*, defaults to `False`):
        Whether to prepend a class token to the vision patch sequence.
    use_ln_pre (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization before the vision transformer.
    use_ln_post (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization after the vision transformer.
    use_abs_posemb (`bool`, *optional*, defaults to `True`):
        Whether to add absolute position embeddings in the vision encoder.
    use_rope2d (`bool`, *optional*, defaults to `True`):
        Whether to use 2D rotary position embeddings in the vision encoder.
    ls_init_value (`float`, *optional*, defaults to 0.1):
        Initial value for layer scale parameters in the vision encoder.
    """

    model_type = "perception_encoder"

    width: int = 1536
    layers: int = 47
    heads: int = 16
    num_channels: int = 3
    image_size: int = 728
    mlp_ratio: float | int = 8960 / 1536
    patch_size: int = 14
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    use_cls_token: bool = False
    use_ln_pre: bool = True
    use_ln_post: bool = False
    use_abs_posemb: bool = True
    use_rope2d: bool = True
    ls_init_value: float = 0.1

    def __post_init__(self, **kwargs):
        # Common config/modeling tests look for standard aliases. Keep the original checkpoint names too.
        self.hidden_size = self.width
        self.num_hidden_layers = self.layers
        self.num_attention_heads = self.heads
        super().__post_init__(**kwargs)


@strict
@auto_docstring(custom_args=STEP3P7_TEXT_CONFIG_ARGS, checkpoint="stepfun-ai/Step-3.7-Flash")
class Step3p7TextConfig(PreTrainedConfig):
    r"""
    num_attention_groups (`int`, *optional*, defaults to 8):
        Number of key/value attention groups.
    moe_num_experts (`int`, *optional*, defaults to 288):
        Number of routed MoE experts.
    moe_top_k (`int`, *optional*, defaults to 8):
        Number of experts selected per token.
    rope_theta (`float` or `list[float]`, *optional*, defaults to 10000):
        Base period used by rotary position embeddings.
    rope_scaling (`dict`, *optional*):
        Rotary embedding scaling configuration.
    share_expert_dim (`int`, *optional*, defaults to 1280):
        Intermediate size of shared experts.
    use_head_wise_attn_gate (`bool`, *optional*, defaults to `False`):
        Whether to use head-wise attention gates.
    use_moe_router_bias (`bool`, *optional*, defaults to `False`):
        Whether MoE router projections use a bias term.
    moe_router_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scaling factor applied to MoE router scores.
    need_fp32_gate (`bool`, *optional*, defaults to `False`):
        Whether to compute router gates in float32.
    attention_other_setting (`dict`, *optional*):
        Additional attention settings from original checkpoints.
    swiglu_limits (`list`, *optional*):
        Clamp limits for routed expert SwiGLU activations.
    swiglu_limits_shared (`list`, *optional*):
        Clamp limits for shared expert SwiGLU activations.
    yarn_only_types (`list[str]`, *optional*):
        Layer type names that should use YaRN-style RoPE settings only.
    moe_layers_enum (`tuple[int]`, `list[int]` or `str`, *optional*):
        Indices of layers that use MoE blocks.
    """

    model_type = "step3p5"
    architectures = ["Step3p5ForCausalLM"]

    hidden_size: int = 4096
    intermediate_size: int = 11264
    num_attention_heads: int = 64
    num_attention_groups: int = 8
    num_hidden_layers: int = 45
    vocab_size: int = 128815
    rms_norm_eps: float = 1e-5
    moe_intermediate_size: int = 1280
    moe_num_experts: int = 288
    moe_top_k: int = 8
    rope_theta: float | int | list[float | int] = 10000
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 128000
    share_expert_dim: int = 1280
    head_dim: int = 128
    layer_types: list[str] | None = None
    sliding_window: int | None = None
    pad_token_id: int = 1
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    use_head_wise_attn_gate: bool = False
    use_moe_router_bias: bool = False
    moe_router_scaling_factor: float = 1.0
    need_fp32_gate: bool = False
    attention_other_setting: dict[str, Any] | None = None
    swiglu_limits: list[float | int | None] | None = None
    swiglu_limits_shared: list[float | int | None] | None = None
    yarn_only_types: list[str] | None = None
    moe_layers_enum: tuple[int, ...] | list[int] | str = (
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
    )

    def __post_init__(self, **kwargs):
        self.layer_types = _normalize_per_layer_values(self.layer_types, self.num_hidden_layers)
        if isinstance(self.rope_scaling, dict):
            self.rope_scaling = dict(self.rope_scaling)
        if isinstance(self.moe_layers_enum, str):
            self.moe_layers_enum = tuple(int(i) for i in self.moe_layers_enum.split(",") if i.strip())
        elif isinstance(self.moe_layers_enum, list):
            self.moe_layers_enum = tuple(self.moe_layers_enum)
        self.num_key_value_heads = self.num_attention_groups
        super().__post_init__(**kwargs)


def _normalize_per_layer_values(
    values: Sequence[Any] | None,
    num_hidden_layers: int,
) -> list[Any] | None:
    if values is None:
        return None
    normalized = list(values)
    if not normalized:
        return normalized
    if len(normalized) < num_hidden_layers:
        normalized.extend([normalized[-1]] * (num_hidden_layers - len(normalized)))
    # Some checkpoints keep MTP/spec layer entries after the decoder layers.
    # This config only builds num_hidden_layers decoder layers, and HF strict
    # validation requires per-layer fields to match that decoder count.
    return normalized[:num_hidden_layers]


@strict
@auto_docstring(custom_args=STEP3P7_CONFIG_ARGS, checkpoint="stepfun-ai/Step-3.7-Flash")
class Step3p7Config(PreTrainedConfig):
    r""" """

    sub_configs = {"vision_config": Step3p7VisionEncoderConfig, "text_config": Step3p7TextConfig}
    # This loader is a compatibility shim for original Step VL checkpoints
    # whose top-level config model_type is `step3p7`.
    model_type = "step3p7"

    vision_config: dict | Step3p7VisionEncoderConfig | None = None
    text_config: dict | Step3p7TextConfig | None = None
    projector_bias: bool = False
    image_token_id: int = 151679
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        shared_rope_scaling = kwargs.get("rope_scaling")
        if isinstance(shared_rope_scaling, dict):
            shared_rope_scaling = dict(shared_rope_scaling)

        if self.vision_config is None:
            self.vision_config = Step3p7VisionEncoderConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Step3p7VisionEncoderConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = Step3p7TextConfig(rope_scaling=shared_rope_scaling)
        elif isinstance(self.text_config, dict):
            text_config = dict(self.text_config)
            if shared_rope_scaling is not None and "rope_scaling" not in text_config:
                text_config["rope_scaling"] = shared_rope_scaling
            self.text_config = Step3p7TextConfig(**text_config)
        elif shared_rope_scaling is not None and self.text_config.rope_scaling is None:
            self.text_config.rope_scaling = dict(shared_rope_scaling)

        self.hidden_size = self.text_config.hidden_size
        self.intermediate_size = self.text_config.intermediate_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_attention_groups = self.text_config.num_attention_groups
        self.num_key_value_heads = self.text_config.num_attention_groups
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.max_position_embeddings = self.text_config.max_position_embeddings
        super().__post_init__(**kwargs)


def rotate_half_vision(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension halves (used by RoPE)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotary embeddings to queries / keys."""
    dtype = t.dtype
    if t.ndim == 3:
        freqs = freqs[-t.shape[-2] :]
    rot_dim = freqs.shape[-1]
    assert rot_dim <= t.shape[-1], f"feature dimension {t.shape[-1]} is too small for rot_dim {rot_dim}"
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos()) + (rotate_half_vision(t_rot) * freqs.sin())
    return torch.cat((t_rot, t_pass), dim=-1).to(dtype)


class Step3p7VisionEncoderRope2D(nn.Module):
    """Cacheable 2D rotary positional embedding."""

    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta: int | float = 10000,
        theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        # Lazy-init the cache on first forward call. This avoids two pitfalls:
        # (1) accelerate's `init_empty_weights` context (used when device_map
        # is set in transformers >= 5.x) would put the cache on the meta device
        # and it never gets materialized because the checkpoint doesn't carry
        # this buffer; (2) `persistent=True` would then trigger
        # "newly initialized" warnings without actually populating it.
        # The placeholder is zero-sized and harmless on meta.
        self.register_buffer("freqs_cache", torch.zeros(0), persistent=False)

    def _compute_inv_freq(self, base: int | float, dim: int) -> torch.Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        return freqs

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor):
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)
        return freqs

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h_range = torch.arange(self.max_grid_height, dtype=torch.float)
        grid_w_range = torch.arange(self.max_grid_width, dtype=torch.float)
        if self.use_cls_token:
            grid_h_range += 1
            grid_w_range += 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(self.max_grid_height * self.max_grid_width, -1)
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        freqs = freqs[None, None, ...]
        return freqs

    def forward(self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int]):
        # Materialize the cache on the actual device the first time we run
        # (or whenever the device changes). See note in __init__.
        if self.freqs_cache.numel() == 0 or self.freqs_cache.is_meta or self.freqs_cache.device != q.device:
            self.freqs_cache = self._compute_2d_freqs().to(device=q.device)

        # If grid matches cached shape we reuse directly to avoid recomputation.
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat([torch.zeros(1, device=q.device), positions + 1], dim=0)
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        return q, k


class Step3p7VisionEncoderLayerScale(Dinov2LayerScale):
    """Per-channel residual scaling used when ls_init_value is set."""

    def __init__(self, dim: int, init_values: float):
        nn.Module.__init__(self)
        self.lambda1 = nn.Parameter(torch.full((dim,), init_values))


class Step3p7VisionEncoderMLP(Dinov2MLP):
    """Feed-forward network used inside each transformer block."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.activation = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)


class Step3p7VisionEncoderAttention(nn.Module):
    """Multi-head self attention with optional 2D RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: int | float = 10000,
        rope_theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj_weight = nn.Parameter(torch.zeros(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.rope = None
        if use_rope2d:
            self.rope = Step3p7VisionEncoderRope2D(
                dim=self.head_dim,
                max_grid_height=max_grid_height,
                max_grid_width=max_grid_width,
                use_cls_token=use_cls_token,
                theta=rope_theta,
                theta_rescale_factor=rope_theta_rescale_factor,
            )

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(
            hidden_states,
            self.in_proj_weight,
            self.in_proj_bias,
        )
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw=grid_hw)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class Step3p7VisionEncoderBlock(nn.Module):
    """A single Vision Transformer block (self-attention + MLP)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: float | None = None,
        max_grid_height: int | None = None,
        max_grid_width: int | None = None,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
    ):
        super().__init__()
        self.attn = Step3p7VisionEncoderAttention(
            hidden_size,
            num_heads,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
            use_rope2d=use_rope2d,
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        intermediate = int(hidden_size * mlp_ratio)
        self.mlp = Step3p7VisionEncoderMLP(hidden_size, intermediate, hidden_act)

        self.ls_1 = Step3p7VisionEncoderLayerScale(hidden_size, ls_init_value)
        self.ls_2 = Step3p7VisionEncoderLayerScale(hidden_size, ls_init_value)

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, grid_hw=grid_hw)
        hidden_states = residual + self.ls_1(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.ls_2(hidden_states)
        return hidden_states


class Step3p7VisionEncoderTransformer(nn.Module):
    """Stack of encoder blocks parameterised by Step35VisionEncoderConfig."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: float | None = None,
        max_grid_height: int | None = None,
        max_grid_width: int | None = None,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
    ):
        super().__init__()
        self.layers = depth
        self.resblocks = nn.ModuleList(
            [
                Step3p7VisionEncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    hidden_act,
                    layer_norm_eps,
                    max_grid_height=max_grid_height,
                    max_grid_width=max_grid_width,
                    use_cls_token=use_cls_token,
                    use_rope2d=use_rope2d,
                    ls_init_value=ls_init_value,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        for block in self.resblocks:
            hidden_states = block(hidden_states, grid_hw=grid_hw)
        return hidden_states


class Step3p7VisionEncoder(nn.Module):
    """
    Vision encoder built from Step3p7VisionEncoderConfig.

    The encoder performs patch embedding followed by a stack of transformer
    blocks. Only the config fields defined in Step3p7VisionEncoderConfig (and
    Step3p7Config.vision_config) are expected.
    """

    def __init__(self, config: Step3p7VisionEncoderConfig):
        super().__init__()
        self.config = config

        self.hidden_size = config.width
        self.num_heads = config.heads
        self.num_hidden_layers = config.layers
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.use_cls_token = config.use_cls_token
        self.use_abs_posemb = config.use_abs_posemb
        self.use_ln_post = config.use_ln_post

        self.conv1 = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.ln_pre = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps) if config.use_ln_pre else nn.Identity()
        self.ln_post = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps) if self.use_ln_post else nn.Identity()

        grid_size = self.image_size // self.patch_size
        self.base_grid = (grid_size, grid_size)

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(torch.randn(self.hidden_size) * (self.hidden_size**-0.5))
        else:
            self.class_embedding = None

        if self.use_abs_posemb:
            self.posemb_grid_size = grid_size
            self.positional_embedding = nn.Parameter(
                (self.hidden_size**-0.5) * torch.randn(int(self.use_cls_token) + grid_size**2, self.hidden_size)
            )

        self.transformer = Step3p7VisionEncoderTransformer(
            embed_dim=self.hidden_size,
            depth=self.num_hidden_layers,
            num_heads=self.num_heads,
            mlp_ratio=config.mlp_ratio,
            hidden_act=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            ls_init_value=config.ls_init_value,
            max_grid_height=grid_size,
            max_grid_width=grid_size,
            use_cls_token=self.use_cls_token,
            use_rope2d=config.use_rope2d,
        )
        self.vit_downsampler1 = nn.Conv2d(self.hidden_size, self.hidden_size * 2, kernel_size=3, stride=2, padding=1)
        self.vit_downsampler2 = nn.Conv2d(
            self.hidden_size * 2, self.hidden_size * 4, kernel_size=3, stride=2, padding=1
        )

    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Args: pixel_values: image tensor of shape (B, C, H, W)."""
        bsz, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        hidden_state = self.conv1(pixel_values)  # (B, D, Gh, Gw)
        hidden_state = hidden_state.flatten(2).transpose(1, 2)  # (B, Gh*Gw, D)

        if self.use_cls_token:
            cls_token = self.class_embedding.view(1, 1, -1).expand(bsz, -1, -1)
            hidden_state = torch.cat([cls_token, hidden_state], dim=1)

        if self.use_abs_posemb:
            pos_emb = self.sample_abs_posemb(grid_h, grid_w)
            hidden_state = hidden_state + pos_emb
        hidden_state = self.ln_pre(hidden_state)
        hidden_state = self.transformer(hidden_state, grid_hw=(grid_h, grid_w))

        if self.use_ln_post:
            hidden_state = self.ln_post(hidden_state)

        if self.use_cls_token:
            hidden_state = hidden_state[:, 1:, :]

        return hidden_state


logger = logging.get_logger(__name__)


class Step3p7PreTrainedModel(PreTrainedModel):
    config_class = Step3p7Config
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.45\.*",
        r"model\.layers\.46\.*",
        r"model\.layers\.47\.*",
    ]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_static_cache = True
    _supports_attention_backend = True


class Step3p7RotaryEmbedding(nn.Module):
    """RoPE with per-layer overrides for `rope_theta` (a list in step3p7 checkpoints) and
    `rope_parameters` (yarn-only layers disable the global `rope_scaling`)."""

    def __init__(
        self,
        config: Step3p7TextConfig,
        rope_theta: float | None = None,
        rope_parameters: dict | None = None,
        device=None,
    ):
        super().__init__()
        # Shallow-copy the shared config so per-layer overrides don't leak between siblings.
        # `rope_parameters` is deep-copied because `standardize_rope_params` mutates it.
        self.config = copy.copy(config)
        if rope_theta is not None:
            self.config.rope_theta = rope_theta
        elif isinstance(self.config.rope_theta, list):
            self.config.rope_theta = self.config.rope_theta[0]
        # `rope_parameters` from PreTrainedConfig.convert_rope_params_to_dict already inherits
        # `rope_theta` from the parent config — which may have been a list — so propagate the
        # per-layer scalar here too.
        baked = copy.deepcopy(rope_parameters) if rope_parameters else None
        if baked is not None:
            baked["rope_theta"] = self.config.rope_theta
        self.config.rope_parameters = baked

        self.rope_theta = self.config.rope_theta
        self.partial_rotary_factor = getattr(self.config, "partial_rotary_factor", None) or 1.0

        if baked:
            self.rope_type = baked.get("rope_type", baked.get("type", "default"))
        else:
            self.rope_type = "default"

        rope_init_fn = (
            self.compute_default_rope_parameters
            if self.rope_type == "default"
            else ROPE_INIT_FUNCTIONS[self.rope_type]
        )
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float().to(x.device)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def compute_default_rope_parameters(
        config: Step3p7TextConfig | None = None,
        device: "torch.device | None" = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to (q, k) on the first `cos.shape[-1]` dims."""
    if cos.ndim == q.ndim - 1:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@auto_docstring(
    custom_intro="""
    Base class for Step3p7 model outputs.
    """
)
@dataclass
class Step3p7ModelOutputWithPast(ModelOutput):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Hidden states produced by the vision path when image inputs are provided.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for Step3p7 causal language model outputs.
    """
)
@dataclass
class Step3p7CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor`, *optional*):
        Language modeling loss when `labels` are provided.
    logits (`torch.FloatTensor`, *optional*):
        Prediction scores of the language modeling head.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Hidden states produced by the vision path when image inputs are provided.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class Step3p7MLP(nn.Module):
    def __init__(self, config, intermediate_size=None, swiglu_limit=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up)


class Step3p7MoELinear(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(num_experts, out_features, in_features))

    def forward(self, x, expert_id):
        x = F.linear(x.float(), self.weight[expert_id].float())
        return x


class Step3p7MoEMLP(nn.Module):
    def __init__(self, config, swiglu_limit=None):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.use_moe_router_bias = config.use_moe_router_bias
        if self.use_moe_router_bias:
            self.router_bias = nn.Parameter(
                torch.zeros(config.moe_num_experts, dtype=torch.float32), requires_grad=False
            )
            self.custom_routing_function = self.router_bias_func
        else:
            self.custom_routing_function = None
        self.need_fp32_gate = config.need_fp32_gate
        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit

        self.up_proj = Step3p7MoELinear(self.num_experts, self.hidden_size, self.moe_intermediate_size)
        self.gate_proj = Step3p7MoELinear(self.num_experts, self.hidden_size, self.moe_intermediate_size)
        self.down_proj = Step3p7MoELinear(self.num_experts, self.moe_intermediate_size, self.hidden_size)

    def router_bias_func(self, gating_output: torch.Tensor, topk: int, renormalize: bool):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20)
        return expert_topk_weight, indices

    def get_expert_output(self, inputs: torch.Tensor, expert_id):
        # if self.limit is None:
        up = self.up_proj(inputs, expert_id)
        gate = self.act_fn(self.gate_proj(inputs, expert_id))
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)

        return self.down_proj(gate * up, expert_id)

    def forward(self, hidden_states):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.need_fp32_gate:
            router_logits = torch.matmul(
                hidden_states.to(torch.float32),
                self.gate.weight.t().to(torch.float32),
            )
        else:
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

        if self.custom_routing_function:
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights = routing_weights * self.routed_scaling_factor

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                self.get_expert_output(current_state, expert_idx) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Step3p7RMSNorm(Gemma3RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


class Step3p7Attention(nn.Module):
    def __init__(self, config: Step3p7TextConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_groups

        layer_types = getattr(config, "layer_types", []) or []
        if layer_types:
            enable_sliding_window = layer_types[layer_idx] == "sliding_attention"
        else:
            enable_sliding_window = layer_idx % 2 == 0

        # Per-layer rope parameters: yarn-only layers disable rope scaling.
        yarn_only_types = getattr(config, "yarn_only_types", None)
        if yarn_only_types and layer_types[layer_idx] not in yarn_only_types:
            rope_parameters = None
        else:
            rope_parameters = getattr(config, "rope_scaling", None)

        # `rope_theta` may be a per-layer list; pick the entry for this layer.
        rope_theta = config.rope_theta
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[layer_idx]

        if enable_sliding_window:
            self.num_attention_heads = config.attention_other_setting["num_attention_heads"]
            self.num_key_value_heads = config.attention_other_setting["num_attention_groups"]
        self.sliding_window = config.sliding_window if enable_sliding_window else None

        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.rotary_emb = Step3p7RotaryEmbedding(config, rope_theta=rope_theta, rope_parameters=rope_parameters)

        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.q_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_attention_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if self.use_head_wise_attn_gate:
            gate_states = self.g_proj(hidden_states)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1)
        if self.use_head_wise_attn_gate:
            output = (
                attn_output.view(*attn_output.shape[:-1], self.num_attention_heads, self.head_dim)
                * gate_states.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Step3p7DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Step3p7Attention(config, layer_idx)
        layer_types = getattr(config, "layer_types", None) or []
        if layer_types:
            self.attention_type = layer_types[layer_idx]
        else:
            self.attention_type = "sliding_attention" if layer_idx % 2 == 0 else "full_attention"

        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            if isinstance(moe_layers_enum, str):
                moe_layers_idx = [int(i) for i in moe_layers_enum.split(",") if i.strip()]
            else:
                moe_layers_idx = [int(i) for i in moe_layers_enum]
        else:
            moe_layers_idx = list(range(1, config.num_hidden_layers))
        self.is_moe_layer = layer_idx in moe_layers_idx
        self.use_moe = False

        if (
            config.swiglu_limits_shared
            and config.swiglu_limits_shared[layer_idx] is not None
            and config.swiglu_limits_shared[layer_idx] != 0
        ):
            swiglu_limit_shared = config.swiglu_limits_shared[layer_idx]
        else:
            swiglu_limit_shared = None
        if (
            config.swiglu_limits
            and config.swiglu_limits[layer_idx] is not None
            and config.swiglu_limits[layer_idx] != 0
        ):
            swiglu_limit = config.swiglu_limits[layer_idx]
        else:
            swiglu_limit = None
        if self.is_moe_layer:
            self.moe = Step3p7MoEMLP(config, swiglu_limit=swiglu_limit)  #
            self.share_expert = Step3p7MLP(
                config, intermediate_size=config.share_expert_dim, swiglu_limit=swiglu_limit_shared
            )
            self.use_moe = True
        else:
            self.mlp = Step3p7MLP(config, intermediate_size=config.intermediate_size, swiglu_limit=swiglu_limit_shared)

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_moe:
            share_output = self.share_expert(hidden_states)
            moe_output = self.moe(hidden_states)
            ffn_output = moe_output + share_output
        else:
            ffn_output = self.mlp(hidden_states)
        if isinstance(ffn_output, tuple):
            hidden_states, _ = ffn_output
        else:
            hidden_states = ffn_output

        hidden_states = residual + hidden_states
        return hidden_states


class Step3p7TextPreTrainedModel(Step3p7PreTrainedModel):
    config_class = Step3p7TextConfig


@auto_docstring
class Step3p7TextModel(Step3p7TextPreTrainedModel):
    _no_split_modules = ["Step3p7DecoderLayer"]
    base_model_prefix = "model"
    config: Step3p7TextConfig

    def __init__(self, config: Step3p7TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Step3p7DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        layer_types = self.config.layer_types or []
        self.has_sliding_layers = not layer_types or "sliding_attention" in layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_kwargs["inputs_embeds"] = inputs_embeds
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # # create position embeddings to be shared across the decoder layers
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring(
    custom_intro="""
    The bare Step3p7 vision-language model outputting raw hidden states without any specific head on top.
    """
)
class Step3p7Model(Step3p7PreTrainedModel):
    config: Step3p7Config
    base_model_prefix = "model"

    def __init__(self, config: Step3p7Config):
        super().__init__(config)
        self.vision_model = Step3p7VisionEncoder(config.vision_config)
        self.language_model = Step3p7TextModel(config.text_config)
        self.vit_large_projector = nn.Linear(
            config.vision_config.width * 4, config.text_config.hidden_size, bias=config.projector_bias
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def _project_vision_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Run vit_downsampler1 + vit_downsampler2 + vit_large_projector on a (B, P, D) tensor."""
        bsz, num_patches, _ = image_features.shape
        side = int(num_patches**0.5)
        image_features = image_features.permute(0, 2, 1).view(bsz, -1, side, side)
        image_features = self.vision_model.vit_downsampler1(image_features)
        image_features = self.vision_model.vit_downsampler2(image_features)
        bsz, _, side, _ = image_features.shape
        image_features = image_features.view(bsz, -1, side * side).permute(0, 2, 1)
        return self.vit_large_projector(image_features)

    def _get_image_features_tensor(
        self,
        pixel_values: torch.FloatTensor | None,
        patch_pixel_values: torch.FloatTensor | None = None,
        num_patches: torch.Tensor | None = None,
        image_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if image_embeds is not None:
            return image_embeds.view(-1, image_embeds.shape[-1]).to(self.dtype).to(self.device)

        pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:]).to(self.dtype).to(self.device)
        base = self._project_vision_features(self.vision_model(pixel_values))

        patch = None
        if patch_pixel_values is not None:
            patch_pixel_values = patch_pixel_values.view(-1, *patch_pixel_values.shape[-3:])
            if patch_pixel_values.shape[0] > 0:
                patch_pixel_values = patch_pixel_values.to(self.dtype).to(self.device)
                patch = self._project_vision_features(self.vision_model(patch_pixel_values))

        if num_patches is None:
            num_patches = [0] * base.shape[0]

        merged: list[torch.Tensor] = []
        cur = 0
        for i, n in enumerate(num_patches):
            n = int(n)
            if n > 0 and patch is not None:
                merged.append(patch[cur : cur + n].reshape(-1, patch.shape[-1]))
                cur += n
            merged.append(base[i].reshape(-1, base.shape[-1]))
        return torch.cat(merged, dim=0)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None,
        patch_pixel_values: torch.FloatTensor | None = None,
        num_patches: torch.Tensor | None = None,
        image_embeds: torch.FloatTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor | BaseModelOutputWithPooling:
        """Returns image features ready to be scattered into `inputs_embeds`."""
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
            or getattr(self.config.vision_config, "output_hidden_states", False)
        )
        image_features = self._get_image_features_tensor(
            pixel_values=pixel_values,
            patch_pixel_values=patch_pixel_values,
            num_patches=num_patches,
            image_embeds=image_embeds,
        )
        hidden_states = (image_features, image_features) if output_hidden_states else None
        if not return_dict:
            return image_features, image_features, hidden_states
        return BaseModelOutputWithPooling(
            last_hidden_state=image_features,
            pooler_output=image_features,
            hidden_states=hidden_states,
        )

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor | None, inputs_embeds: torch.FloatTensor
    ) -> torch.Tensor:
        if input_ids is None:
            placeholder = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            mask = (inputs_embeds == placeholder).all(-1)
        else:
            mask = input_ids == self.config.image_token_id
        return mask.unsqueeze(-1).expand_as(inputs_embeds)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        patch_pixel_values: torch.FloatTensor | None = None,
        num_patches: torch.Tensor | None = None,
        image_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Step3p7ModelOutputWithPast:
        r"""
        patch_pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values for cropped high-resolution image patches.
        num_patches (`torch.Tensor`, *optional*):
            Number of cropped image patches associated with each input image.
        image_embeds (`torch.FloatTensor`, *optional*):
            Precomputed image embeddings to use instead of running the vision encoder.
        cache_position (`torch.LongTensor`, *optional*):
            Positions of the input sequence tokens in the cache.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features: torch.Tensor | None = None
        if pixel_values is not None or image_embeds is not None:
            image_features = self._get_image_features_tensor(
                pixel_values=pixel_values,
                patch_pixel_values=patch_pixel_values,
                num_patches=num_patches,
                image_embeds=image_embeds,
            ).to(inputs_embeds.device, inputs_embeds.dtype)
            mask = self.get_placeholder_mask(input_ids, inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)

        kwargs.pop("return_dict", None)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        return Step3p7ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


@auto_docstring(
    custom_intro="""
    Step3p7 model for conditional generation from text and image inputs.
    """
)
class Step3p7ForConditionalGeneration(Step3p7PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    config: Step3p7Config
    base_model_prefix = "model"

    def __init__(self, config: Step3p7Config):
        super().__init__(config)
        self.model = Step3p7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @property
    def language_model(self):
        return self.model.language_model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        patch_pixel_values: torch.FloatTensor | None = None,
        num_patches: torch.Tensor | None = None,
        image_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Step3p7CausalLMOutputWithPast:
        r"""
        patch_pixel_values (`torch.FloatTensor`, *optional*):
            Pixel values for cropped high-resolution image patches.
        num_patches (`torch.Tensor`, *optional*):
            Number of cropped image patches associated with each input image.
        image_embeds (`torch.FloatTensor`, *optional*):
            Precomputed image embeddings to use instead of running the vision encoder.
        cache_position (`torch.LongTensor`, *optional*):
            Positions of the input sequence tokens in the cache.
        """
        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            patch_pixel_values=patch_pixel_values,
            num_patches=num_patches,
            image_embeds=image_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Step3p7CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        patch_pixel_values=None,
        num_patches=None,
        image_embeds=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        is_first_iteration: bool = False,
        **kwargs,
    ):
        is_prefill = is_first_iteration
        if inputs_embeds is not None and not is_prefill:
            if past_key_values is None:
                is_prefill = True
            elif hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
                is_prefill = True

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_prefill,
            **kwargs,
        )

        if is_prefill or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["patch_pixel_values"] = patch_pixel_values
            model_inputs["num_patches"] = num_patches
            model_inputs["image_embeds"] = image_embeds

        return model_inputs

    def _fix_state_dict_key_on_load(self, key: str) -> tuple[str, bool]:
        if key.startswith("language_model."):
            return key[len("language_model.") :], True
        return key, False


__all__ = [
    "Step3p7Config",
    "Step3p7ForConditionalGeneration",
    "Step3p7Model",
    "Step3p7PreTrainedModel",
    "Step3p7TextConfig",
    "Step3p7TextModel",
    "Step3p7TextPreTrainedModel",
    "Step3p7VisionEncoder",
    "Step3p7VisionEncoderConfig",
]
