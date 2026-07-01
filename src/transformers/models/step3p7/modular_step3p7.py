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

import transformers.models.ernie4_5.modeling_ernie4_5 as ernie4_5_modeling
import transformers.models.llama.modeling_llama as llama_modeling
import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, RotaryEmbeddingConfigMixin, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts, DeepseekV4MLP
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Embeddings,
    Dinov2LayerScale,
    Dinov2MLP,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaModelOutputWithPast,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.siglip.modeling_siglip import SiglipAttention
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging

from ... import initialization as init
from ...backbone_utils import BackboneConfigMixin
from ...integrations import use_kernelized_func
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs


logger = logging.get_logger(__name__)


@strict
@auto_docstring(checkpoint="stepfun-ai/Step-3.7-Flash")
class Step3p7VisionEncoderConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    hidden_size (`int`, *optional*, defaults to 1536):
        Hidden size of the vision encoder.
    num_hidden_layers (`int`, *optional*, defaults to 47):
        Number of hidden layers in the vision encoder.
    num_attention_heads (`int`, *optional*, defaults to 16):
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
    layerscale_value (`float`, *optional*, defaults to 0.1):
        Initial value for layer scale parameters in the vision encoder.
    """

    model_type = "stepfun3p7_vision"

    hidden_size: int = 1536
    num_hidden_layers: int = 47
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 728
    mlp_ratio: float | int = 8960 / 1536
    patch_size: int = 14
    hidden_act: str = "quick_gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    qkv_bias: bool = True
    layerscale_value: float = 0.1
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


@strict
@auto_docstring(checkpoint="stepfun-ai/Step-3.7-Flash")
class Step3p7TextConfig(Qwen3MoeConfig, RotaryEmbeddingConfigMixin):
    r"""
    vocab_size (`int`, *optional*, defaults to 128815):
        Vocabulary size of the Step3p7 text decoder.
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the decoder hidden states.
    intermediate_size (`int`, *optional*, defaults to 11264):
        Intermediate size of dense MLP layers.
    moe_intermediate_size (`int`, *optional*, defaults to 1280):
        Intermediate size of routed MoE experts.
    num_hidden_layers (`int`, *optional*, defaults to 45):
        Number of decoder layers.
    num_attention_heads (`int`, *optional*, defaults to 64):
        Number of query attention heads.
    num_key_value_heads (`int`, *optional*, defaults to 8):
        Number of key/value attention heads. Defaults to `num_attention_groups`.
    head_dim (`int`, *optional*, defaults to 128):
        Dimension of each attention head.
    layer_types (`list[str]`, *optional*):
        Per-layer attention pattern, using `"full_attention"` or `"sliding_attention"`.
    sliding_window (`int`, *optional*):
        Window size used by layers whose `layer_types` entry is `"sliding_attention"`.
    rope_parameters (`dict`, *optional*):
        Rotary embedding parameters, optionally keyed by layer type.
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
    partial_rotary_factors (`list`, *optional*):
        Per-layer partial rotary factors from original Step3p7 checkpoints.
    yarn_only_types (`list[str]`, *optional*):
        Layer type names that should use YaRN-style RoPE settings only.
    moe_layers_enum (`tuple[int]`, `list[int]` or `str`, *optional*):
        Indices of layers that use MoE blocks.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type pattern. Uses `"dense"` for dense MLP layers and `"sparse"` for MoE layers.
    """

    model_type = "step3p5"
    architectures = ["Step3p5ForCausalLM"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.share_expert.gate_proj": "colwise",
        "layers.*.share_expert.up_proj": "colwise",
        "layers.*.share_expert.down_proj": "rowwise",
        "layers.*.moe.experts.gate_up_proj": "packed_colwise",
        "layers.*.moe.experts.down_proj": "rowwise",
        "layers.*.moe.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        "layers.*.moe.gate": "ep_router",
        "layers.*.moe.experts.gate_up_proj": "grouped_gemm",
        "layers.*.moe.experts.down_proj": "grouped_gemm",
        "layers.*.moe.experts": "moe_tp_experts",
    }

    hidden_size: int = 4096
    intermediate_size: int = 11264
    num_attention_heads: int = 64
    num_attention_groups: int = 8
    num_key_value_heads: int = 8
    num_hidden_layers: int = 45
    vocab_size: int = 128815
    rms_norm_eps: float = 1e-5
    moe_intermediate_size: int = 1280
    moe_num_experts: int = 288
    num_experts: int = 288
    moe_top_k: int = 8
    num_experts_per_tok: int = 8
    rope_theta: float | int | list[float | int] = 10000
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 128000
    share_expert_dim: int = 1280
    head_dim: int = 128
    layer_types: list[str] | None = None
    sliding_window: int | None = None
    pad_token_id: int = 1
    use_head_wise_attn_gate: bool = False
    use_moe_router_bias: bool = False
    moe_router_scaling_factor: float = 1.0
    need_fp32_gate: bool = False
    attention_other_setting: dict[str, Any] | None = None
    swiglu_limits: list[float | int | None] | None = None
    swiglu_limits_shared: list[float | int | None] | None = None
    partial_rotary_factors: list[float | int | None] | None = None
    yarn_only_types: list[str] | None = None
    moe_layers_enum: tuple[int, ...] | list[int] | str | None = None
    mlp_layer_types: list[str] | None = None

    use_sliding_window = AttributeError()
    decoder_sparse_step = AttributeError()
    mlp_only_layers = AttributeError()
    norm_topk_prob = AttributeError()
    output_router_logits = AttributeError()
    router_aux_loss_coef = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            has_sliding_metadata = self.attention_other_setting is not None and self.sliding_window is not None
            self.layer_types = [
                "sliding_attention" if has_sliding_metadata and layer_idx % 2 == 0 else "full_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = _normalize_per_layer_values(self.layer_types, self.num_hidden_layers)

        self.swiglu_limits = _normalize_per_layer_values(self.swiglu_limits, self.num_hidden_layers)
        self.swiglu_limits_shared = _normalize_per_layer_values(self.swiglu_limits_shared, self.num_hidden_layers)
        self.partial_rotary_factors = _normalize_per_layer_values(
            self.partial_rotary_factors, self.num_hidden_layers
        )

        if isinstance(self.rope_scaling, dict):
            self.rope_scaling = dict(self.rope_scaling)
        if self.rope_parameters is not None:
            self.rope_parameters = copy.deepcopy(self.rope_parameters)

        if isinstance(self.moe_layers_enum, str):
            self.moe_layers_enum = tuple(int(i) for i in self.moe_layers_enum.split(",") if i.strip())
        elif isinstance(self.moe_layers_enum, list):
            self.moe_layers_enum = tuple(self.moe_layers_enum)
        elif self.moe_layers_enum is None:
            self.moe_layers_enum = tuple(range(3, self.num_hidden_layers))

        if self.mlp_layer_types is None:
            moe_layers = set(self.moe_layers_enum)
            self.mlp_layer_types = [
                "sparse" if layer_idx in moe_layers else "dense" for layer_idx in range(self.num_hidden_layers)
            ]
        else:
            self.mlp_layer_types = _normalize_per_layer_values(self.mlp_layer_types, self.num_hidden_layers)

        self.num_experts = self.moe_num_experts
        self.num_experts_per_tok = self.moe_top_k
        self.num_key_value_heads = self.num_attention_groups
        PreTrainedConfig.__post_init__(self, **kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_scaling = rope_scaling if rope_scaling is not None else self.rope_scaling

        if self.rope_parameters is None:
            if self.layer_types is None:
                self.rope_parameters = dict(rope_scaling or {})
            else:
                self.rope_parameters = {}
                for layer_type in set(self.layer_types):
                    uses_scaled_rope = not self.yarn_only_types or layer_type in self.yarn_only_types
                    self.rope_parameters[layer_type] = dict(rope_scaling or {}) if uses_scaled_rope else {}
        elif rope_scaling is not None:
            if self.layer_types is not None and set(self.rope_parameters.keys()).issubset(self.layer_types):
                for layer_type in set(self.layer_types):
                    if not self.yarn_only_types or layer_type in self.yarn_only_types:
                        self.rope_parameters.setdefault(layer_type, {}).update(rope_scaling)
            else:
                self.rope_parameters.update(rope_scaling)

        self.standardize_rope_params()
        return kwargs


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


def _get_layer_swiglu_limit(values: list[float | int | None] | None, layer_idx: int):
    if not values:
        return None
    value = values[layer_idx]
    return value if value is not None and value != 0 else None


@strict
@auto_docstring(checkpoint="stepfun-ai/Step-3.7-Flash")
class Step3p7Config(PreTrainedConfig):
    r"""
    vision_config (`dict` or `Step3p7VisionEncoderConfig`, *optional*):
        Configuration of the Step3p7 vision encoder.
    text_config (`dict` or `Step3p7TextConfig`, *optional*):
        Configuration of the Step3p7 text decoder.
    projector_bias (`bool`, *optional*, defaults to `False`):
        Whether the multimodal projector uses a bias term.
    image_token_id (`int`, *optional*, defaults to 151679):
        Token id used as image placeholder in text inputs.
    """

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
        if self.vision_config is None:
            self.vision_config = Step3p7VisionEncoderConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Step3p7VisionEncoderConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = Step3p7TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Step3p7TextConfig(**self.text_config)

        super().__post_init__(**kwargs)


def apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply 2D rotary embeddings to queries / keys."""
    dtype = t.dtype
    if t.ndim == 3:
        freqs = freqs[-t.shape[-2] :]
    rot_dim = freqs.shape[-1]
    assert rot_dim <= t.shape[-1], f"feature dimension {t.shape[-1]} is too small for rot_dim {rot_dim}"
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos()) + (ernie4_5_modeling.rotate_half(t_rot) * freqs.sin())
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
    """Per-channel residual scaling."""

    def __init__(self, config: Step3p7VisionEncoderConfig):
        super().__init__(config)


class Step3p7VisionEncoderMLP(Dinov2MLP):
    """Feed-forward network used inside each transformer block."""

    def __init__(self, config: Step3p7VisionEncoderConfig):
        super().__init__(config)


class Step3p7VisionEncoderAttention(SiglipAttention):
    """Multi-head self attention with optional 2D RoPE."""

    def __init__(self, config: Step3p7VisionEncoderConfig):
        nn.Module.__init__(self)
        self.config = config
        hidden_size = config.hidden_size
        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({config.num_attention_heads})."
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj_weight = nn.Parameter(torch.zeros(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        grid_size = config.image_size // config.patch_size
        self.rope = Step3p7VisionEncoderRope2D(
            dim=self.head_dim,
            max_grid_height=grid_size,
            max_grid_width=grid_size,
            use_cls_token=False,
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
        q, k = self.rope(q, k, grid_hw=grid_hw)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class Step3p7VisionEncoderBlock(GradientCheckpointingLayer):
    """A single Vision Transformer block (self-attention + MLP)."""

    def __init__(self, config: Step3p7VisionEncoderConfig):
        GradientCheckpointingLayer.__init__(self)
        self.attn = Step3p7VisionEncoderAttention(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Step3p7VisionEncoderMLP(config)
        self.ls_1 = Step3p7VisionEncoderLayerScale(config)
        self.ls_2 = Step3p7VisionEncoderLayerScale(config)

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


class Step3p7VisionEncoderEmbeddings(Dinov2Embeddings):
    """Patch and positional embeddings for the Step3p7 vision tower."""

    def __init__(self, config: Step3p7VisionEncoderConfig):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.posemb_grid_size = self.image_size // self.patch_size

        self.conv1 = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.ln_pre = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.positional_embedding = nn.Parameter(
            (self.hidden_size**-0.5) * torch.randn(self.posemb_grid_size**2, self.hidden_size)
        )

    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)

        return pos_embed[None, ...]

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Interpolate Step3p7 absolute position embeddings for a target image size."""
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        return self.sample_abs_posemb(grid_h, grid_w)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        bsz, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        hidden_state = self.conv1(pixel_values)
        hidden_state = hidden_state.flatten(2).transpose(1, 2)
        hidden_state = hidden_state + self.sample_abs_posemb(grid_h, grid_w)

        return self.ln_pre(hidden_state), (grid_h, grid_w)


class Step3p7VisionEncoderTransformer(nn.Module):
    """Stack of encoder blocks parameterised by Step3p7VisionEncoderConfig."""

    def __init__(self, config: Step3p7VisionEncoderConfig):
        nn.Module.__init__(self)
        self.config = config
        self.layers = config.num_hidden_layers
        self.resblocks = nn.ModuleList([Step3p7VisionEncoderBlock(config) for _ in range(config.num_hidden_layers)])

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

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.patch_size = config.patch_size
        self.image_size = config.image_size

        grid_size = self.image_size // self.patch_size
        self.base_grid = (grid_size, grid_size)
        self.embeddings = Step3p7VisionEncoderEmbeddings(config)
        self.transformer = Step3p7VisionEncoderTransformer(config)
        self.vit_downsampler1 = nn.Conv2d(self.hidden_size, self.hidden_size * 2, kernel_size=3, stride=2, padding=1)
        self.vit_downsampler2 = nn.Conv2d(
            self.hidden_size * 2, self.hidden_size * 4, kernel_size=3, stride=2, padding=1
        )

    @property
    def conv1(self):
        return self.embeddings.conv1

    @property
    def ln_pre(self):
        return self.embeddings.ln_pre

    @property
    def positional_embedding(self):
        return self.embeddings.positional_embedding

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Args: pixel_values: image tensor of shape (B, C, H, W)."""
        hidden_state, grid_hw = self.embeddings(pixel_values)
        hidden_state = self.transformer(hidden_state, grid_hw=grid_hw)

        return hidden_state


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

    def post_init(self):
        super().post_init()
        # Legacy Step3p7 checkpoint conversions are load-only compatibility shims.
        # Freshly initialized models should save the canonical packed expert keys.
        if getattr(self, "_weight_conversions", None) is None:
            self._weight_conversions = []

    def _init_weights(self, module):
        super()._init_weights(module)
        text_config = getattr(self.config, "text_config", self.config)
        std = getattr(text_config, "initializer_range", 0.02)
        if isinstance(module, Step3p7MoEExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Step3p7RotaryEmbedding):
            for layer_type in dict.fromkeys(module.rope_layer_types):
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)


class Step3p7RotaryEmbedding(nn.Module):
    """RoPE with Step3p7 layer-type parameters while using standardized ``rope_parameters``."""

    def __init__(self, config: Step3p7TextConfig, device=None):
        super().__init__()
        self.config = copy.copy(config)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = {}
        self.rope_layer_types = []

        rope_parameters_by_key = {}
        signature_to_key = {}
        layer_type_counts = {}
        for layer_idx, layer_type in enumerate(config.layer_types):
            rope_parameters = self._get_layer_rope_parameters(config, layer_idx)
            signature = (layer_type, tuple(sorted((key, repr(value)) for key, value in rope_parameters.items())))
            if signature not in signature_to_key:
                suffix = layer_type_counts.get(layer_type, 0)
                rope_layer_type = layer_type if suffix == 0 else f"{layer_type}_{suffix}"
                layer_type_counts[layer_type] = suffix + 1
                signature_to_key[signature] = rope_layer_type
                rope_parameters_by_key[rope_layer_type] = rope_parameters
            self.rope_layer_types.append(signature_to_key[signature])

        self.config.layer_types = list(rope_parameters_by_key)
        self.config.rope_parameters = rope_parameters_by_key

        for rope_layer_type, rope_parameters in rope_parameters_by_key.items():
            self.rope_type[rope_layer_type] = rope_parameters["rope_type"]
            rope_init_fn = (
                self.compute_default_rope_parameters
                if self.rope_type[rope_layer_type] == "default"
                else ROPE_INIT_FUNCTIONS[self.rope_type[rope_layer_type]]
            )
            inv_freq, attention_scaling = rope_init_fn(self.config, device, layer_type=rope_layer_type)
            self.register_buffer(f"{rope_layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{rope_layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{rope_layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def _get_layer_rope_parameters(config: Step3p7TextConfig, layer_idx: int) -> dict[str, Any]:
        layer_type = config.layer_types[layer_idx]
        if config.rope_parameters:
            if set(config.rope_parameters.keys()).issubset(config.layer_types):
                rope_parameters = copy.deepcopy(config.rope_parameters[layer_type])
            else:
                rope_parameters = copy.deepcopy(config.rope_parameters)
        else:
            rope_parameters = {"rope_type": "default"}

        if isinstance(config.rope_theta, list):
            rope_theta = config.rope_theta[min(layer_idx, len(config.rope_theta) - 1)]
        else:
            rope_theta = rope_parameters.get("rope_theta", config.rope_theta)
        rope_parameters["rope_theta"] = rope_theta
        rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))

        if config.partial_rotary_factors is not None:
            partial_rotary_factor = config.partial_rotary_factors[layer_idx]
        else:
            partial_rotary_factor = rope_parameters.get(
                "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
            )
        if partial_rotary_factor is not None:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        return rope_parameters

    @staticmethod
    def compute_default_rope_parameters(
        config: Step3p7TextConfig | None = None,
        device: "torch.device | None" = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation.
        """
        rope_parameters = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
        base = rope_parameters["rope_theta"]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = config.head_dim
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type: str):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float().to(x.device)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

@auto_docstring(
    custom_intro="""
    Base class for Step3p7 model outputs.
    """
)
@dataclass
class Step3p7ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    Base class for Step3p7 causal language model outputs.
    """
)
@dataclass
class Step3p7CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor`, *optional*):
        Language modeling loss when `labels` are provided.
    logits (`torch.FloatTensor`, *optional*):
        Prediction scores of the language modeling head.
    """


class Step3p7MLP(DeepseekV4MLP):
    def __init__(self, config, intermediate_size=None, swiglu_limit=None):
        config = copy.copy(config)
        config.hidden_act = "silu"
        config.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        config.mlp_bias = False
        config.swiglu_limit = swiglu_limit if swiglu_limit is not None else float("inf")
        super().__init__(config)


class Step3p7MoEExperts(DeepseekV4Experts):
    def __init__(self, config: Step3p7TextConfig, swiglu_limit=None):
        config = copy.copy(config)
        config.num_local_experts = config.moe_num_experts
        config.intermediate_size = config.moe_intermediate_size
        config.hidden_act = "silu"
        config.swiglu_limit = swiglu_limit if swiglu_limit is not None else float("inf")
        super().__init__(config)


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
        self.routed_scaling_factor = config.moe_router_scaling_factor

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        self.experts = Step3p7MoEExperts(config, swiglu_limit=swiglu_limit)

    def router_bias_func(self, gating_output: torch.Tensor, topk: int, renormalize: bool):
        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_with_bias, k=topk, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        expert_topk_weight = topk_prob
        if renormalize:
            expert_topk_weight = expert_topk_weight / (torch.sum(expert_topk_weight, dim=-1, keepdim=True) + 1e-20)
        return expert_topk_weight, indices

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

        final_hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Step3p7RMSNorm(Gemma3RMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        nn.Module.__init__(self)
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    if q.shape[-1] == cos.shape[-1]:
        return llama_modeling.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    query_rot, query_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    key_rot, key_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    query_rot, key_rot = llama_modeling.apply_rotary_pos_emb(
        query_rot, key_rot, cos, sin, unsqueeze_dim=unsqueeze_dim
    )
    return torch.cat((query_rot, query_pass), dim=-1), torch.cat((key_rot, key_pass), dim=-1)


@use_kernelized_func(apply_rotary_pos_emb)
class Step3p7Attention(qwen3_modeling.Qwen3Attention):
    def __init__(self, config: Step3p7TextConfig, layer_idx):
        attention_config = copy.copy(config)
        attention_config.attention_bias = False
        enable_sliding_window = config.layer_types[layer_idx] == "sliding_attention"
        if enable_sliding_window:
            attention_config.num_attention_heads = config.attention_other_setting["num_attention_heads"]
            attention_config.num_key_value_heads = config.attention_other_setting["num_attention_groups"]
        else:
            attention_config.num_attention_heads = config.num_attention_heads
            attention_config.num_key_value_heads = config.num_attention_groups

        nn.Module.__init__(self)
        self.layer_type = config.layer_types[layer_idx]
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = attention_config.num_attention_heads
        self.num_key_value_heads = attention_config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)
        self.q_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if enable_sliding_window else None

        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_attention_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if self.use_head_wise_attn_gate:
            gate_states = self.g_proj(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, qwen3_modeling.eager_attention_forward
        )

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
        self.attention_type = config.layer_types[layer_idx]
        self.is_moe_layer = config.mlp_layer_types[layer_idx] == "sparse"

        swiglu_limit_shared = _get_layer_swiglu_limit(config.swiglu_limits_shared, layer_idx)
        swiglu_limit = _get_layer_swiglu_limit(config.swiglu_limits, layer_idx)
        if self.is_moe_layer:
            self.moe = Step3p7MoEMLP(config, swiglu_limit=swiglu_limit)  #
            self.share_expert = Step3p7MLP(
                config, intermediate_size=config.share_expert_dim, swiglu_limit=swiglu_limit_shared
            )
        else:
            self.mlp = Step3p7MLP(config, intermediate_size=config.intermediate_size, swiglu_limit=swiglu_limit_shared)

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe_layer:
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


Step3p7PreTrainedModel._can_record_outputs = {
    "hidden_states": Step3p7DecoderLayer,
    "attentions": Step3p7Attention,
}


@auto_docstring
class Step3p7TextModel(Step3p7PreTrainedModel):
    _no_split_modules = ["Step3p7DecoderLayer"]
    base_model_prefix = "model"
    config_class = Step3p7TextConfig
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
        self.rotary_emb = Step3p7RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "inputs_embeds": inputs_embeds,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = {
            rope_layer_type: self.rotary_emb(hidden_states, position_ids, rope_layer_type)
            for rope_layer_type in dict.fromkeys(self.rotary_emb.rope_layer_types)
        }

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings[self.rotary_emb.rope_layer_types[layer_idx]],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
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
            config.vision_config.hidden_size * 4, config.text_config.hidden_size, bias=config.projector_bias
        )
        self.post_init()

    def _project_vision_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Run vit_downsampler1 + vit_downsampler2 + vit_large_projector on a (B, P, D) tensor."""
        bsz, num_patches, _ = image_features.shape
        side = int(num_patches**0.5)
        image_features = image_features.permute(0, 2, 1).view(bsz, -1, side, side)
        image_features = self.vision_model.vit_downsampler1(image_features)
        image_features = self.vision_model.vit_downsampler2(image_features)
        bsz, _, side, _ = image_features.shape
        image_features = image_features.view(bsz, -1, side * side).permute(0, 2, 1)
        image_features = image_features.to(
            device=self.vit_large_projector.weight.device, dtype=self.vit_large_projector.weight.dtype
        )
        return self.vit_large_projector(image_features)

    def _get_image_features_tensor(
        self,
        pixel_values: torch.FloatTensor | None,
        patch_pixel_values: torch.FloatTensor | None = None,
        num_patches: torch.Tensor | None = None,
        image_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if image_embeds is not None:
            image_embeds = image_embeds.view(-1, image_embeds.shape[-1])
            return image_embeds.to(
                device=self.vit_large_projector.weight.device, dtype=self.vit_large_projector.weight.dtype
            )

        vision_reference = self.vision_model.conv1.weight
        pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:]).to(
            device=vision_reference.device, dtype=vision_reference.dtype
        )
        base = self._project_vision_features(self.vision_model(pixel_values))

        patch = None
        if patch_pixel_values is not None:
            patch_pixel_values = patch_pixel_values.view(-1, *patch_pixel_values.shape[-3:])
            if patch_pixel_values.shape[0] > 0:
                patch_pixel_values = patch_pixel_values.to(device=vision_reference.device, dtype=vision_reference.dtype)
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
        r"""
        Extract image features in the same order as Step3p7 image placeholders.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(num_images, num_channels, height, width)`, *optional*):
                Base image pixel values returned by [`Step3VisionProcessor`].
            patch_pixel_values (`torch.FloatTensor` of shape `(num_patches, num_channels, height, width)`, *optional*):
                Pixel values for cropped high-resolution image patches.
            num_patches (`torch.Tensor` or `list[int]`, *optional*):
                Number of cropped image patches associated with each base image. Patch features are placed before
                the corresponding base-image features.
            image_embeds (`torch.FloatTensor` of shape `(num_image_tokens, hidden_size)`, *optional*):
                Precomputed image features. If passed, the vision encoder is skipped.
            output_hidden_states (`bool`, *optional*):
                Whether to return image hidden states in a [`~modeling_outputs.BaseModelOutputWithPooling`].
            return_dict (`bool`, *optional*):
                Whether to return a [`~modeling_outputs.BaseModelOutputWithPooling`] instead of a tuple.

        Returns:
            `torch.Tensor` or [`~modeling_outputs.BaseModelOutputWithPooling`]: Flattened image features ready to
            be scattered into `inputs_embeds`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
            or self.config.vision_config.output_hidden_states
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Step3p7ModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_images, num_channels, height, width)`, *optional*):
            Base image pixel values returned by [`Step3VisionProcessor`].
        patch_pixel_values (`torch.FloatTensor` of shape `(num_patches, num_channels, height, width)`, *optional*):
            Pixel values for cropped high-resolution image patches returned by [`Step3VisionProcessor`].
        num_patches (`torch.Tensor` or `list[int]`, *optional*):
            Number of cropped image patches associated with each base image. Patch features are inserted before
            the corresponding base-image features.
        image_embeds (`torch.FloatTensor`, *optional*):
            Precomputed image features to use instead of running the vision encoder.
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
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

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
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Step3p7CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_images, num_channels, height, width)`, *optional*):
            Base image pixel values returned by [`Step3VisionProcessor`].
        patch_pixel_values (`torch.FloatTensor` of shape `(num_patches, num_channels, height, width)`, *optional*):
            Pixel values for cropped high-resolution image patches returned by [`Step3VisionProcessor`].
        num_patches (`torch.Tensor` or `list[int]`, *optional*):
            Number of cropped image patches associated with each base image. Patch features are inserted before
            the corresponding base-image features.
        image_embeds (`torch.FloatTensor`, *optional*):
            Precomputed image features to use instead of running the vision encoder.
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
            return_dict=True,
            **kwargs,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
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
        next_sequence_length=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        patch_pixel_values=None,
        num_patches=None,
        image_embeds=None,
        attention_mask=None,
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
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
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
    "Step3p7VisionEncoder",
    "Step3p7VisionEncoderConfig",
]
