# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HunYuanVL model."""

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..hunyuan_v1_dense.configuration_hunyuan_v1_dense import HunYuanDenseV1Config
from ..hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1Attention,
    HunYuanDenseV1DecoderLayer,
    HunYuanDenseV1Model,
    HunYuanDenseV1PreTrainedModel,
    HunYuanDenseV1RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,  # noqa: F401  - re-exported for downstream tooling
    rotate_half,
)
from ..llama.modeling_llama import LlamaRMSNorm


@auto_docstring(checkpoint="tencent/HunyuanOCR")
@strict
class HunYuanVLVisionConfig(PreTrainedConfig):
    r"""
    Vision backbone configuration for the dense-only, image-text HunYuanVL open-source variant.

    interpolate_mode (`str`, *optional*, defaults to `"bilinear"`):
        Interpolation mode used when resizing learned patch positional embeddings to match the current image grid.
    learnable_mlp_pooling_size (`int`, *optional*, defaults to 0):
        Optional learnable pooling size for the vision tower.
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Output hidden size produced by the vision tower before it is consumed by the text backbone.
    remove_prenorm (`bool`, *optional*, defaults to `True`):
        Whether to remove the pre-normalization behavior used by some internal vision variants.
    resize_resolution (`int`, *optional*, defaults to 2048):
        Reference resolution used when deriving image resizing and tokenization behavior.
    img_max_token_num (`int`, *optional*, defaults to 4096):
        Maximum image token count expected by the vision stack.
    max_image_size (`int`, *optional*, defaults to 2048):
        Maximum supported image size for the current open-source vision configuration.
    min_image_size (`int`, *optional*, defaults to 512):
        Minimum supported image size for the current open-source vision configuration.
    anyres_vit_max_image_size (`int`, *optional*, defaults to 2048):
        Maximum image size supported by the any-resolution vision preprocessing path.
    max_vit_seq_len (`int`, *optional*, defaults to 16384):
        Maximum sequence length produced by the vision transformer.
    text_hidden_size (`int`, *optional*, defaults to 3072):
        Hidden size expected by the text backbone when consuming visual embeddings.

    Example:

    ```python
    >>> from transformers import HunYuanVLVisionConfig
    >>>
    >>> configuration = HunYuanVLVisionConfig()
    >>> configuration.hidden_size
    1152
    ```"""

    model_type = "hunyuan_vl_vision"
    base_config_key = "vision_config"

    hidden_act: str = "gelu"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    interpolate_mode: str = "bilinear"
    rms_norm_eps: float = 1e-5
    learnable_mlp_pooling_size: int = 0
    attention_dropout: float = 0.0
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_channels: int = 3
    num_hidden_layers: int = 27
    out_hidden_size: int = 4096
    patch_size: int = 16
    remove_prenorm: bool = True
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    resize_resolution: int = 2048
    img_max_token_num: int = 4096
    max_image_size: int = 2048
    min_image_size: int = 512
    anyres_vit_max_image_size: int = 2048
    max_vit_seq_len: int = 16384
    text_hidden_size: int = 3072

    def __post_init__(self, **kwargs):
        if not self.num_key_value_heads:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="tencent/HunyuanOCR")
@strict
class HunYuanVLTextConfig(HunYuanDenseV1Config):
    r"""
    Text backbone configuration for the dense-only, image-text HunYuanVL open-source variant.

    Inherits the standard fields from [`HunYuanDenseV1Config`] and adds a few legacy aliases that some Tencent
    checkpoints persist on disk (`pad_id`, `attention_head_dim`, `rope_scaling`, `rope_theta`). Those legacy fields are
    normalized into the canonical `pad_token_id` / `head_dim` / `rope_parameters` slots in `__post_init__` so the rest
    of the model only ever needs to read the canonical fields.

    eod_token_id (`int`, *optional*, defaults to 3):
        Token id representing the end-of-document marker. Inherited from [`HunYuanDenseV1Config`] and re-documented
        here so the auto-generated docstring stays in sync.
    sep_token_id (`int`, *optional*, defaults to 4):
        Token id used as a separator marker by HunYuan tokenizers.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Legacy alias preserved for compatibility with checkpoints that persist a top-level rope theta. The value is
        merged into `rope_parameters` during normalization.
    rope_scaling (`dict`, *optional*):
        Legacy RoPE scaling payload from Tencent checkpoints. When provided, it is normalized into `rope_parameters`
        (and the equivalent `xdrope` rope type is rewritten to `dynamic`).
    pad_id (`int`, *optional*):
        Legacy padding token field. When `pad_token_id` is unset or `-1`, this value is normalized into `pad_token_id`.
    attention_head_dim (`int`, *optional*):
        Legacy alias for `head_dim`. When `head_dim` is not provided, this value is used as the per-head hidden size.
    org_vocab_size (`int`, *optional*):
        Original vocabulary size recorded in exported checkpoints for compatibility with Tencent tooling.
    use_qk_norm (`bool`, *optional*, defaults to `False`):
        Legacy flag preserved for checkpoint compatibility. Has no runtime effect in the open-source variant.
    use_cla (`bool`, *optional*, defaults to `False`):
        Legacy flag preserved for checkpoint compatibility. Has no runtime effect in the open-source variant.
    enable_lm_head_fp32 (`bool`, *optional*, defaults to `False`):
        Legacy flag preserved for checkpoint compatibility. Has no runtime effect in the open-source variant.
    """

    model_type = "hunyuan_vl_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    ignore_keys_at_rope_validation = {
        "alpha",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
        "xdrope_section",
    }

    sep_token_id: int | None = 4
    rope_scaling: dict | None = None
    rope_theta: float = 10000.0
    pad_id: int | None = None
    attention_head_dim: int | None = None
    org_vocab_size: int | None = None
    use_qk_norm: bool = False
    use_cla: bool = False
    enable_lm_head_fp32: bool = False

    def __post_init__(self, **kwargs):
        # Translate legacy aliases into canonical fields before invoking the standard validation.
        if self.head_dim is None and self.attention_head_dim is not None:
            self.head_dim = self.attention_head_dim
        if self.pad_token_id == -1 and self.pad_id not in (None, -1):
            self.pad_token_id = self.pad_id

        # Only normalize rope payloads when a legacy ``rope_scaling`` blob was provided. Otherwise let the parent
        # ``HunYuanDenseV1Config.__post_init__`` (and its ``standardize_rope_params`` helper) populate
        # ``rope_parameters`` itself, which keeps the canonical ``{rope_theta, rope_type}`` shape consistent across
        # save / reload cycles.
        if self.rope_scaling is not None:
            rope_parameters = self._normalize_rope_parameters(
                getattr(self, "rope_parameters", None),
                self.rope_scaling,
                self.rope_theta,
            )
            if rope_parameters is not None:
                self.rope_parameters = rope_parameters
                self.rope_scaling = rope_parameters
                self.rope_theta = rope_parameters["rope_theta"]

        super().__post_init__(**kwargs)

    @staticmethod
    def _normalize_rope_parameters(
        rope_parameters: dict | None,
        rope_scaling: dict | None,
        rope_theta: float,
    ) -> dict | None:
        if rope_parameters is None and rope_scaling is None:
            return None
        if rope_parameters is None:
            rope_parameters = dict(rope_scaling)
        else:
            rope_parameters = dict(rope_parameters)

        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        if rope_type == "xdrope":
            rope_type = "dynamic"
        rope_parameters["rope_type"] = rope_type
        # Mirror the rope type under the legacy ``type`` key so checkpoints exported with either spelling continue
        # to round-trip without losing information.
        rope_parameters["type"] = rope_type
        rope_parameters.setdefault("rope_theta", rope_theta)
        return rope_parameters

    def _rope_parameters_validation(self):
        # Skip rope validation when no rope payload was provided so minimal configs continue to work.
        if getattr(self, "rope_parameters", None) is None and getattr(self, "rope_scaling", None) is None:
            return
        self.standardize_rope_params()
        self.validate_rope()


@auto_docstring(checkpoint="tencent/HunyuanOCR")
class HunYuanVLConfig(PreTrainedConfig):
    r"""
    Top-level configuration for the open-source HunYuanVL integration.

    This configuration describes the dense-only, image-text-only variant used for OCR and document-understanding style
    workloads. It mirrors the `Qwen2_5_VL` / `Qwen3_VL` family layout: the top-level config simply composes a
    [`HunYuanVLTextConfig`] (text backbone) and a [`HunYuanVLVisionConfig`] (vision tower) plus a few token ids that
    delimit image spans in multimodal prompts.

    text_config (`HunYuanVLTextConfig` or `dict`, *optional*):
        Configuration of the text backbone. When `None`, default values are used.
    vision_config (`HunYuanVLVisionConfig` or `dict`, *optional*):
        Configuration of the vision tower. When `None`, default values are used.
    image_token_id (`int`, *optional*, defaults to 120120):
        Token id used as the visual placeholder in multimodal prompts.
    im_start_id (`int`, *optional*, defaults to 120118):
        Token id marking the beginning of an image span in multimodal prompts.
    im_end_id (`int`, *optional*, defaults to 120119):
        Token id marking the end of an image span in multimodal prompts.
    im_newline_id (`int`, *optional*, defaults to 120121):
        Token id used for newline-style separators inserted inside serialized image regions.

    Example:

    ```python
    >>> from transformers import HunYuanVLConfig, HunYuanVLForConditionalGeneration
    >>>
    >>> configuration = HunYuanVLConfig()
    >>> model = HunYuanVLForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "hunyuan_vl"
    sub_configs = {"vision_config": HunYuanVLVisionConfig, "text_config": HunYuanVLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 120120,
        im_start_id: int = 120118,
        im_end_id: int = 120119,
        im_newline_id: int = 120121,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        # When loading legacy "flat" Tencent checkpoints (where text fields live at the top level instead of inside a
        # nested `text_config` block) we fold the recognized text-side keys into the text config payload. This keeps
        # ``HunYuanVLConfig.from_pretrained(...)`` working with both the upstream nested layout and the existing
        # public OCR checkpoints.
        text_kwargs = self._extract_text_kwargs(kwargs)

        if text_config is None:
            self.text_config = self.sub_configs["text_config"](**text_kwargs)
        elif isinstance(text_config, dict):
            text_config = {**text_config, **text_kwargs}
            self.text_config = self.sub_configs["text_config"](**text_config)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.im_newline_id = im_newline_id

        # Keep the vision tower in sync with the consuming text backbone size.
        self.vision_config.text_hidden_size = self.text_config.hidden_size

        # Propagate text-side identifiers to the top-level config so generic generation utilities can read them.
        kwargs.setdefault("pad_token_id", self.text_config.pad_token_id)
        kwargs.setdefault("bos_token_id", self.text_config.bos_token_id)
        kwargs.setdefault("eos_token_id", self.text_config.eos_token_id)
        kwargs.setdefault("tie_word_embeddings", self.text_config.tie_word_embeddings)

        super().__init__(**kwargs)

    @classmethod
    def _extract_text_kwargs(cls, kwargs: dict) -> dict:
        """
        Pop and return the subset of ``kwargs`` that should be forwarded to [`HunYuanVLTextConfig`].

        Required to support legacy Tencent checkpoints whose ``config.json`` stores the text-backbone fields at the
        top level instead of inside a nested ``text_config`` block.
        """
        text_keys = set(cls.sub_configs["text_config"].__dataclass_fields__) | {"rope_scaling", "rope_theta"}
        return {key: kwargs.pop(key) for key in list(kwargs) if key in text_keys}


HUNYUAN_VL_TEXT_FORWARD_CUSTOM_ARGS = r"""
    cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
        Indices describing the absolute position of each input token. Used to derive default `position_ids` and to
        update the key-value cache during generation.
"""


def _get_past_seq_length(past_key_values: Cache | None, layer_idx: int | None = None) -> int:
    """Best-effort helper that supports both ``Cache`` instances and lightweight test fakes."""
    if past_key_values is None:
        return 0
    try:
        seq_len = past_key_values.get_seq_length(layer_idx)
    except TypeError:
        seq_len = past_key_values.get_seq_length()
    if isinstance(seq_len, torch.Tensor):
        return int(seq_len.max().item())
    return int(seq_len)


def _build_rotary_cache_from_inv_freq(
    rotary_emb: nn.Module, x: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct ``(cos, sin)`` from ``inv_freq`` for the xdrope path that needs an explicit cache size."""
    t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    freqs = torch.outer(t, rotary_emb.inv_freq.to(x.device).float())
    emb = torch.cat((freqs, freqs), dim=-1).float()
    attention_scaling = getattr(rotary_emb, "attention_scaling", 1.0)
    cos = (emb.cos() * attention_scaling).to(dtype=x.dtype)
    sin = (emb.sin() * attention_scaling).to(dtype=x.dtype)
    return cos, sin


def _normalize_xdrope_section(xdrope_section, head_dim: int) -> list[int] | None:
    """
    Normalize the ``xdrope_section`` config field to a list of integer half-head sizes.

    Real Tencent checkpoints store absolute half-head partition sizes (e.g. ``[16, 16, 16, 16]`` for ``head_dim=128``).
    Lightweight tests sometimes use ratio-style sections that sum to ``1.0``; both forms are accepted.
    """
    if xdrope_section is None:
        return None

    section_values = [float(section) for section in xdrope_section]
    section_ints = [int(section) for section in section_values]

    if all(value.is_integer() for value in section_values) and sum(section_ints) * 2 == head_dim:
        return section_ints

    if all(section <= 1.0 for section in section_values):
        return [int(section * head_dim / 2) for section in section_values]

    return section_ints


def apply_rotary_pos_emb_xdrope(q, k, cos, sin, position_ids, xdrope_section, output_size):
    """Apply HunYuan's xdrope rotary embedding to ``q`` and ``k``."""
    x_dim = len(xdrope_section)
    cos = cos[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()
    sin = sin[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()

    xdrope_section = [int(section) * 2 for section in xdrope_section]
    if sum(xdrope_section) != cos.shape[-1]:
        raise ValueError(
            f"Illegal partition for xd rope: expected {cos.shape[-1]} rotary dims, got {sum(xdrope_section)}"
        )

    cos = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(cos.split(xdrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[:, :, i % x_dim, :] for i, m in enumerate(sin.split(xdrope_section, dim=-1))], dim=-1)
    cos = cos.view(output_size[0], 1, output_size[2], -1)
    sin = sin.view(output_size[0], 1, output_size[2], -1)

    origin_dtype = q.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.float(), sin.float()
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(origin_dtype), k_out.to(origin_dtype)


class HunYuanVLRMSNorm(LlamaRMSNorm):
    pass


class HunYuanVLRotaryEmbedding(HunYuanDenseV1RotaryEmbedding):
    pass


class HunYuanVLVisionMLP(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.dense_h_to_4h = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense_4h_to_h(self.act_fn(self.dense_h_to_4h(hidden_states)))


class HunYuanVLVisionPatchEmbed(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.spatial_merge_size = config.spatial_merge_size
        self.interpolate_mode = config.interpolate_mode

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.max_num_patches = (config.max_image_size // self.patch_size) ** 2
        self.num_positions = self.max_num_patches + 1
        self.position_edge = int(self.num_positions**0.5)
        # The first token is the cls token; the remaining tokens form the learnable patch positional grid.
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.patch_pos_embed = None

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        num_patches, _ = pixel_values.shape
        pixel_values = pixel_values.reshape(num_patches, self.num_channels, self.patch_size, self.patch_size)

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.squeeze(-1).squeeze(-1).unsqueeze(0)

        if self.patch_pos_embed is None:
            patch_pos_shape = (1, self.position_edge, self.position_edge, self.embed_dim)
            self.patch_pos_embed = (
                self.position_embedding.weight[1:, :].reshape(patch_pos_shape).permute(0, 3, 1, 2).float()
            )

        patch_pos_embed_list = []
        for grid in grid_thw:
            _, h0, w0 = grid
            # Add a tiny epsilon to avoid floating point error in the interpolation.
            # See https://github.com/facebookresearch/dino/issues/8.
            h0, w0 = h0 + 0.1, w0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                self.patch_pos_embed,
                scale_factor=((h0 / self.position_edge).item(), (w0 / self.position_edge).item()),
                mode=self.interpolate_mode,
                align_corners=False,
            )
            patch_pos_embed = (
                patch_pos_embed.reshape(self.embed_dim, -1).transpose(0, 1).unsqueeze(0).to(patch_embeds.dtype)
            )
            patch_pos_embed_list.append(patch_pos_embed)

        patch_pos_embed = torch.cat(patch_pos_embed_list, dim=1)
        return patch_embeds + patch_pos_embed


class HunYuanVLVisionPatchMerger(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_merge_size: int, rms_norm_eps: float):
        super().__init__()

        embed_std = out_channels**-0.5
        self.spatial_merge_size = spatial_merge_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=spatial_merge_size, stride=spatial_merge_size),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1),
        )
        self.mlp = nn.Linear(in_channels * 4, out_channels)
        self.image_newline = nn.Parameter(torch.randn(in_channels * 4) * embed_std)
        self.image_begin = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_end = nn.Parameter(torch.randn(out_channels) * embed_std)
        self.image_sep = nn.Parameter(torch.randn(out_channels) * embed_std)

        self.before_rms = HunYuanVLRMSNorm(in_channels, eps=rms_norm_eps)
        self.after_rms = HunYuanVLRMSNorm(out_channels, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, size: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = self.before_rms(hidden_states)
        h, w = size
        dtype = hidden_states.dtype
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            hidden_states.shape[0], -1, int(h.item()), int(w.item())
        )
        hidden_states = self.proj(hidden_states)
        b, c, h, w = hidden_states.shape
        hidden_states = torch.cat(
            [
                hidden_states,
                self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype, non_blocking=True),
            ],
            dim=-1,
        )
        hidden_states = hidden_states.reshape(b, c, -1).permute(0, 2, 1)
        hidden_states = self.mlp(hidden_states)

        begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, hidden_states.shape[-1]).to(dtype, non_blocking=True)
        end = self.image_end.reshape(1, 1, -1).expand(b, 1, hidden_states.shape[-1]).to(dtype, non_blocking=True)
        hidden_states = torch.cat([begin, hidden_states, end], dim=1)

        return self.after_rms(hidden_states)


class HunYuanVLVisionAttention(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.is_causal = False
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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


class HunYuanVLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanVLVisionAttention(config)
        self.mlp = HunYuanVLVisionMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HunYuanVLVisionTransformer(nn.Module):
    """
    HunYuanVL vision tower: patch embedding -> transformer blocks -> per-image patch merger.

    Inputs are flat per-patch pixel tensors plus an ``image_grid_thw`` tensor describing the spatial layout of every
    image in the batch. The output is the concatenation of merged image embeddings, ready to be scattered into the
    language-model embedding stream.
    """

    config: HunYuanVLVisionConfig
    _no_split_modules = ["HunYuanVLVisionBlock"]

    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = HunYuanVLVisionPatchEmbed(config)
        self.layers = nn.ModuleList([HunYuanVLVisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.perceive = HunYuanVLVisionPatchMerger(
            self.config.hidden_size,
            self.config.text_hidden_size,
            self.config.spatial_merge_size,
            self.config.rms_norm_eps,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.LongTensor) -> torch.Tensor:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_patches, num_channels * patch_size * patch_size)`):
            Flat per-patch pixel features produced by the image processor.
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions for each image. Each row contains `[t, h, w]` patch counts.
        """
        hidden_states = self.embeddings(pixel_values, grid_thw)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        cu_seqlens: list = [0]
        for _, h, w in grid_thw:
            cu_seqlens.append((h * w).item())

        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        split_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        split_items = torch.split(hidden_states, split_lengths, dim=1)

        processed_items = []
        for grid, item in zip(grid_thw, split_items):
            _, h, w = grid
            processed_items.append(self.perceive(item, size=(h, w)))

        return torch.cat(processed_items, dim=1)


@auto_docstring
class HunYuanVLPreTrainedModel(HunYuanDenseV1PreTrainedModel):
    config: HunYuanVLConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True


class HunYuanVLDenseV1Attention(HunYuanDenseV1Attention):
    """
    HunYuan dense attention with optional xdrope rotary embeddings.

    On prefill, when ``rope_parameters['xdrope_section']`` is set and ``position_ids`` carries a 4-channel
    multimodal layout, queries and keys are rotated through [`apply_rotary_pos_emb_xdrope`]. Otherwise the layer
    behaves exactly like [`HunYuanDenseV1Attention`].
    """

    def __init__(self, config: HunYuanVLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.xdrope_section = _normalize_xdrope_section(rope_parameters.get("xdrope_section"), self.head_dim)
        self.rotary_emb: HunYuanVLRotaryEmbedding | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        position_ids = kwargs.get("position_ids")
        use_xdrope_prefill = (
            self.xdrope_section is not None
            and position_ids is not None
            and position_ids.dim() == 3
            and _get_past_seq_length(past_key_values, self.layer_idx) == 0
        )

        if use_xdrope_prefill:
            rotary_seq_len = max(key_states.shape[-2], int(position_ids.max().item()) + 1)
            cos, sin = _build_rotary_cache_from_inv_freq(self.rotary_emb, value_states, rotary_seq_len)
            output_size = (
                query_states.size(0),
                query_states.size(1),
                query_states.size(2),
                key_states.size(2),
            )
            query_states, key_states = apply_rotary_pos_emb_xdrope(
                query_states, key_states, cos, sin, position_ids, self.xdrope_section, output_size
            )
        else:
            if position_embeddings is None:
                rotary_position_ids = position_ids
                if rotary_position_ids is not None and rotary_position_ids.dim() == 3:
                    rotary_position_ids = rotary_position_ids[:, 0, :]
                cos, sin = self.rotary_emb(value_states, rotary_position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            try:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            except TypeError:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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


class HunYuanVLDenseV1DecoderLayer(HunYuanDenseV1DecoderLayer):
    def __init__(self, config: HunYuanVLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = HunYuanVLDenseV1Attention(config=config, layer_idx=layer_idx)


class HunYuanVLTextModel(HunYuanDenseV1Model):
    """Dense text backbone used inside [`HunYuanVLModel`]."""

    config: HunYuanVLTextConfig
    _no_split_modules = ["HunYuanVLDenseV1DecoderLayer"]
    _can_record_outputs = {
        "hidden_states": HunYuanVLDenseV1DecoderLayer,
        "attentions": HunYuanVLDenseV1Attention,
    }

    def __init__(self, config: HunYuanVLTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HunYuanVLDenseV1DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = HunYuanVLRotaryEmbedding(config=config)
        for layer in self.layers:
            layer.self_attn.rotary_emb = self.rotary_emb
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring(custom_args=HUNYUAN_VL_TEXT_FORWARD_CUSTOM_ARGS)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = _get_past_seq_length(past_key_values)
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_position_ids = position_ids[:, 0, :] if position_ids.dim() >= 3 else position_ids
        causal_mask = create_causal_mask(
            self.config,
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values=past_key_values,
            position_ids=causal_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = None
        if position_ids is not None and position_ids.dim() == 2:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class HunYuanVLForCausalLM(HunYuanVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        text_config = config.text_config
        self.model = HunYuanVLTextModel(text_config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.vocab_size = text_config.vocab_size
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    @can_return_tuple
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, HunYuanVLForCausalLM

        >>> model = HunYuanVLForCausalLM.from_pretrained("tencent/HunyuanOCR")
        >>> tokenizer = AutoTokenizer.from_pretrained("tencent/HunyuanOCR")

        >>> prompt = "The capital of France is"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> generate_ids = model.generate(inputs.input_ids, max_new_tokens=10)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class HunYuanVLForConditionalGeneration(HunYuanVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: HunYuanVLConfig

    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        text_config = config.text_config
        self.model = HunYuanVLTextModel(text_config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.vocab_size = text_config.vocab_size
        self.vit = HunYuanVLVisionTransformer(config.vision_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None
    ) -> torch.FloatTensor:
        r"""
        Encode images into continuous embeddings that can be scattered into the language-model token stream.

        pixel_values (`torch.FloatTensor`):
            Flat per-patch pixel features produced by the image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        vit_dtype = next(self.vit.parameters()).dtype
        pixel_values = pixel_values.to(vit_dtype)
        return self.vit(pixel_values, grid_thw=image_grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        token_id: int | None = None,
    ) -> torch.BoolTensor:
        """
        Compute a boolean mask over ``inputs_embeds`` selecting the positions that hold the visual placeholder
        token, and validate that the placeholder count matches the number of provided image features.
        """
        if token_id is None:
            token_id = self.config.image_token_id

        if input_ids is None:
            placeholder_token_embed = self.get_input_embeddings()(
                torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == placeholder_token_embed).all(-1)
        else:
            special_image_mask = input_ids == token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                f"features {image_features.shape[0]}"
            )
        return special_image_mask

    @can_return_tuple
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
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        >>> from PIL import Image
        >>> import torch

        >>> model_id = "tencent/HunyuanOCR"
        >>> processor = AutoProcessor.from_pretrained(model_id, backend="pil")
        >>> model = HunYuanVLForConditionalGeneration.from_pretrained(
        ...     model_id, attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto"
        ... )

        >>> image = Image.open("path/to/your/image.jpg").convert("RGB")
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "path/to/your/image.jpg"},
        ...             {"type": "text", "text": "Extract the text from the image."},
        ...         ],
        ...     }
        ... ]
        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

        >>> with torch.no_grad():
        ...     generated_ids = model.generate(**inputs, max_new_tokens=128)
        >>> generated_trimmed = generated_ids[0][inputs["input_ids"].shape[-1]:]
        >>> print(processor.decode(generated_trimmed, skip_special_tokens=True))
        ```"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            target_device = input_ids.device if input_ids is not None else inputs_embeds.device
            image_embeds = image_embeds.to(target_device, dtype=inputs_embeds.dtype, non_blocking=True)
            image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
                token_id=self.config.image_token_id,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        token_type_ids=None,
        imgs_pos=None,
        **kwargs,
    ):
        kwargs.pop("imgs", None)
        kwargs.pop("imgs_pos", None)

        cache_position = kwargs.get("cache_position")
        is_decode_step = _get_past_seq_length(past_key_values) > 0
        if is_decode_step:
            kwargs.pop("pixel_values", None)
            kwargs.pop("image_grid_thw", None)

        position_ids = kwargs.get("position_ids")

        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        prepared_cache_position = inputs.get("cache_position", cache_position)

        if position_ids is not None and position_ids.ndim == 3:
            if is_decode_step:
                if prepared_cache_position is not None:
                    pos = prepared_cache_position[-1:]
                    inputs["position_ids"] = (
                        pos.view(1, 1, 1).expand(position_ids.shape[0], position_ids.shape[1], 1).clone()
                    )
                else:
                    inputs["position_ids"] = position_ids[:, :, -1:].clone()
            else:
                inputs["position_ids"] = position_ids

        return inputs


__all__ = [
    "HunYuanVLConfig",
    "HunYuanVLVisionConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLPreTrainedModel",
    "HunYuanVLTextModel",
    "HunYuanVLForCausalLM",
    "HunYuanVLForConditionalGeneration",
]
