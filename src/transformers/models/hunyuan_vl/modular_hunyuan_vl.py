# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
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
from ..mllama.modeling_mllama import MllamaVisionAttention
from ..paddleocr_vl.modeling_paddleocr_vl import PaddleOCRVisionEmbeddings
from ..qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from ..qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorKwargs,
    Qwen2VLImageProcessorPil,
    smart_resize,
)
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from ..siglip.modeling_siglip import SiglipEncoderLayer, SiglipMLP


@dataclass
class HunYuanVLModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Image features produced by the vision tower and scattered into the language-model token stream.
    """

    image_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Vision backbone configuration for the dense-only, image-text HunYuanVL open-source variant.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLVisionConfig(PreTrainedConfig):
    r"""
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
    attribute_map = {"layer_norm_eps": "rms_norm_eps"}

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


@auto_docstring(
    custom_intro="""
    Text backbone configuration for the dense-only, image-text HunYuanVL open-source variant.

    Inherits the standard fields from [`HunYuanDenseV1Config`] and declares the canonical field names
    (`pad_token_id`, `head_dim`, `vocab_size`) as the only public attributes. Legacy aliases that some Tencent
    checkpoints persist on disk (`pad_id`, `attention_head_dim`, `org_vocab_size`) are mapped onto those canonical
    fields via `attribute_map`, so the rest of the model only ever needs to read the canonical fields. Legacy RoPE
    payloads persisted as `rope_scaling` / `rope_theta` are normalized by the base configuration class into
    `rope_parameters`.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLTextConfig(HunYuanDenseV1Config):
    r"""
    eod_token_id (`int`, *optional*, defaults to 3):
        Token id representing the end-of-document marker. Inherited from [`HunYuanDenseV1Config`] and re-documented
        here so the auto-generated docstring stays in sync.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie the input and output word embeddings.
    rope_parameters (`dict`, *optional*):
        RoPE configuration payload. Legacy `rope_scaling` / `rope_theta` checkpoint fields are normalized into this
        canonical field by the base configuration class.
    sep_token_id (`int`, *optional*, defaults to 4):
        Token id used as a separator marker by HunYuan tokenizers.
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

    # Legacy checkpoint fields that map onto canonical Transformers config names. The base class redirects every read
    # and write of the legacy name to the canonical slot, so the model only ever holds the canonical attribute. See
    # `problems.md` for the compatibility assumption behind `pad_id`.
    attribute_map = {
        "pad_id": "pad_token_id",
        "attention_head_dim": "head_dim",
        "org_vocab_size": "vocab_size",
    }

    sep_token_id: int | None = 4
    tie_word_embeddings: bool = True
    use_qk_norm: bool = False
    use_cla: bool = False
    enable_lm_head_fp32: bool = False

    def __post_init__(self, **kwargs):
        # Legacy aliases (`pad_id`, `attention_head_dim`, `org_vocab_size`) are normalized onto canonical fields by the
        # base `__setattr__` via `attribute_map`, so no manual translation is needed here.
        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        kwargs = PreTrainedConfig.convert_rope_params_to_dict(self, **kwargs)

        rope_parameters = getattr(self, "rope_parameters", None)
        if not rope_parameters:
            return kwargs

        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        if rope_type == "xdrope":
            rope_type = "dynamic"
        rope_parameters["rope_type"] = rope_type
        if "type" in rope_parameters:
            rope_parameters["type"] = rope_type
        return kwargs


@auto_docstring(
    custom_intro="""
    Top-level configuration for the open-source HunYuanVL integration.

    This configuration describes the dense-only, image-text-only variant used for OCR and document-understanding style
    workloads. It mirrors the `Qwen2_5_VL` / `Qwen3_VL` family layout: the top-level config simply composes a
    [`HunYuanVLTextConfig`] (text backbone) and a [`HunYuanVLVisionConfig`] (vision tower) plus a few token ids that
    delimit image spans in multimodal prompts.
    """,
    checkpoint="tencent/HunyuanOCR",
)
@strict
class HunYuanVLConfig(Qwen2VLConfig):
    r"""
    text_config (`HunYuanVLTextConfig` or `dict`, *optional*):
        Configuration of the text backbone. When `None`, default values are used.
    vision_config (`HunYuanVLVisionConfig` or `dict`, *optional*):
        Configuration of the vision tower. When `None`, default values are used.
    image_token_id (`int`, *optional*, defaults to 120120):
        Token id used as the visual placeholder in multimodal prompts.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie the input and output word embeddings.
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

    image_token_id: int = 120120
    im_start_id: int = 120118
    im_end_id: int = 120119
    im_newline_id: int = 120121
    tie_word_embeddings: bool = True

    # HunYuanVL delimits image spans with `im_start_id` / `im_end_id` instead of the Qwen2VL
    # `vision_start_token_id` / `vision_end_token_id` markers, and it has no video modality, so drop the
    # inherited Qwen2VL placeholder token ids that would otherwise leak into the generated config.
    vision_start_token_id = AttributeError()
    vision_end_token_id = AttributeError()
    video_token_id = AttributeError()

    def __post_init__(self, **kwargs):
        # When loading legacy "flat" Tencent checkpoints (where text fields live at the top level instead of inside a
        # nested `text_config` block) we fold the recognized text-side keys into the text config payload. This keeps
        # ``HunYuanVLConfig.from_pretrained(...)`` working with both the upstream nested layout and the existing
        # public OCR checkpoints.
        text_kwargs = self._extract_text_kwargs(kwargs)

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**{**self.text_config, **text_kwargs})
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"](**text_kwargs)

        # Keep the vision tower in sync with the consuming text backbone size.
        self.vision_config.text_hidden_size = self.text_config.hidden_size

        # `tie_word_embeddings` is read directly off the top-level config by the generic weight-tying path
        # (see `modeling_utils.PreTrainedModel._get_tied_criteria`), which does NOT go through
        # `get_text_config()`, so it must be mirrored here from the text config to stay consistent.
        # The text-side token ids (pad/bos/eos) are intentionally NOT mirrored: generic generation and
        # validation utilities always reach them via `config.get_text_config()`, and a top-level copy
        # would only create a second source of truth that can drift from the text config.
        kwargs.setdefault("tie_word_embeddings", self.text_config.tie_word_embeddings)

        # Call `PreTrainedConfig.__post_init__` directly rather than `super().__post_init__`. `super()` would
        # resolve to `Qwen2VLConfig`, whose `__post_init__` re-runs sub-config normalization and flat-field
        # folding; the modular converter would inline that body here, producing redundant dead code (the
        # branches are no-ops because we already normalized above) and an `inspect.signature` fold that would
        # silently drop a top-level `rope_parameters`. We already perform the equivalent work ourselves above,
        # so go straight to the base class.
        PreTrainedConfig.__post_init__(self, **kwargs)

    @classmethod
    def _extract_text_kwargs(cls, kwargs: dict) -> dict:
        """
        Pop and return the subset of ``kwargs`` that should be forwarded to [`HunYuanVLTextConfig`].

        Required to support legacy Tencent checkpoints whose ``config.json`` stores the text-backbone fields at the
        top level instead of inside a nested ``text_config`` block.
        """
        text_keys = set(cls.sub_configs["text_config"].__dataclass_fields__) | {"rope_scaling", "rope_theta"}
        return {key: kwargs.pop(key) for key in list(kwargs) if key in text_keys}


class HunYuanVLImageProcessorKwargs(Qwen2VLImageProcessorKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `512 * 512`):
        The min pixels of the image to resize the image.
    max_pixels (`int`, *optional*, defaults to `2048 * 2048`):
        The max pixels of the image to resize the image.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """


class HunYuanVLImageProcessor(Qwen2VLImageProcessor):
    size = {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    spatial_patch_size = 1
    valid_kwargs = HunYuanVLImageProcessorKwargs

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> tuple[int, int]:
        """Return the `(grid_h, grid_w)` patch counts used by HunYuanVL token accounting."""
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs["min_pixels"] if "min_pixels" in images_kwargs else self.size["shortest_edge"]
        max_pixels = images_kwargs["max_pixels"] if "max_pixels" in images_kwargs else self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return resized_height // patch_size, resized_width // patch_size


class HunYuanVLImageProcessorPil(Qwen2VLImageProcessorPil):
    size = {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    spatial_patch_size = 1
    valid_kwargs = HunYuanVLImageProcessorKwargs

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> tuple[int, int]:
        """Return the `(grid_h, grid_w)` patch counts used by HunYuanVL token accounting."""
        images_kwargs = images_kwargs or {}
        min_pixels = images_kwargs["min_pixels"] if "min_pixels" in images_kwargs else self.size["shortest_edge"]
        max_pixels = images_kwargs["max_pixels"] if "max_pixels" in images_kwargs else self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return resized_height // patch_size, resized_width // patch_size


def _get_past_seq_length(past_key_values: Cache | None, layer_idx: int | None = None) -> int:
    """Return the cached sequence length from a standard Transformers ``Cache``."""
    if past_key_values is None:
        return 0

    if layer_idx is None:
        seq_len = past_key_values.get_seq_length()
    else:
        seq_len = past_key_values.get_seq_length(layer_idx)

    if isinstance(seq_len, torch.Tensor):
        return int(seq_len.max().item())
    return int(seq_len)


def _normalize_xdrope_section(xdrope_section, head_dim: int) -> list[int] | None:
    """
    Normalize the ``xdrope_section`` config field to integer half-head partition sizes.

    Tencent checkpoints store absolute half-head partition sizes, e.g. ``[16, 16, 16, 16]`` for ``head_dim=128``.
    """
    if xdrope_section is None:
        return None

    section_values = [float(section) for section in xdrope_section]
    section_ints = [int(section) for section in section_values]
    expected_sum = head_dim // 2
    if not all(value.is_integer() for value in section_values) or sum(section_ints) != expected_sum:
        raise ValueError(
            f"Illegal xdrope partition: expected half-head sections summing to {expected_sum}, got {section_ints}"
        )
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
    def __init__(self, config: HunYuanVLTextConfig, device=None):
        super().__init__(config, device=device)
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.xdrope_section = _normalize_xdrope_section(rope_parameters.get("xdrope_section"), head_dim)

    def _build_rotary_cache(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a full ``(cos, sin)`` cache for xdrope position-id indexing."""
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        cos, sin = self(x, position_ids)
        return cos.squeeze(0), sin.squeeze(0)


class HunYuanVLVisionMLP(SiglipMLP):
    pass


class HunYuanVLVisionPatchEmbed(PaddleOCRVisionEmbeddings):
    def __init__(self, config: HunYuanVLVisionConfig):
        nn.Module.__init__(self)
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.max_num_patches = (config.max_image_size // self.patch_size) ** 2
        self.num_positions = self.max_num_patches + 1
        self.position_edge = config.max_image_size // self.patch_size
        # The first token is the cls token; the remaining tokens form the learnable patch positional grid.
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        patch_pos_embed = self.position_embedding.weight[1:, :].reshape(
            1, self.position_edge, self.position_edge, self.embed_dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2).float()
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(height, width),
            mode=self.config.interpolate_mode,
            align_corners=False,
        )
        return patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim).to(embeddings.dtype)

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        num_patches, _ = pixel_values.shape
        pixel_values = pixel_values.reshape(1, num_patches, self.config.num_channels, self.patch_size, self.patch_size)
        batch_size, sequence_len, channel, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = pixel_values.reshape(batch_size * sequence_len, channel, height, width)
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = embeddings.reshape(batch_size, sequence_len, -1).squeeze(0)

        start = 0
        image_embeddings_list = []
        for t, h, w in grid_thw:
            end = start + t * h * w
            image_embeddings = embeddings[start:end, :]
            position_embedding = self.interpolate_pos_encoding(image_embeddings, h, w).squeeze(0).repeat(t, 1)
            image_embeddings_list.append(image_embeddings + position_embedding)
            start = end

        return torch.concat(image_embeddings_list, dim=0).unsqueeze(0)


class HunYuanVLVisionPatchMerger(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()

        self.config = config
        in_channels = config.hidden_size
        out_channels = config.text_hidden_size
        spatial_merge_size = config.spatial_merge_size
        rms_norm_eps = config.rms_norm_eps
        embed_std = out_channels**-0.5
        self.spatial_merge_size = spatial_merge_size
        self.proj_conv = nn.Conv2d(
            in_channels, in_channels * 2, kernel_size=spatial_merge_size, stride=spatial_merge_size
        )
        self.proj_act = nn.GELU()
        self.proj_out = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1)
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
        hidden_states = self.proj_conv(hidden_states)
        hidden_states = self.proj_act(hidden_states)
        hidden_states = self.proj_out(hidden_states)
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


class HunYuanVLVisionAttention(MllamaVisionAttention):
    def __init__(self, config: HunYuanVLVisionConfig):
        nn.Module.__init__(self)
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.is_causal = False
        self.head_dim = getattr(config, "head_dim", self.embed_dim // self.num_heads)
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, q_seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class HunYuanVLVisionBlock(SiglipEncoderLayer):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__(config)
        self.self_attn = HunYuanVLVisionAttention(config)
        self.mlp = HunYuanVLVisionMLP(config)


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
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

        if position_embeddings is None:
            raise ValueError("HunYuanVLDenseV1Attention requires precomputed rotary embeddings.")

        if use_xdrope_prefill:
            cos, sin = position_embeddings
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
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
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


@auto_docstring
class HunYuanVLPreTrainedModel(HunYuanDenseV1PreTrainedModel):
    config: HunYuanVLConfig
    input_modalities = ("image", "text")
    _no_split_modules = ["HunYuanVLDenseV1DecoderLayer", "HunYuanVLVisionBlock"]
    _can_record_outputs = {
        "hidden_states": HunYuanVLDenseV1DecoderLayer,
        "attentions": HunYuanVLDenseV1Attention,
    }


class HunYuanVLVisionTransformer(HunYuanVLPreTrainedModel):
    """
    HunYuanVL vision tower: patch embedding -> transformer blocks -> per-image patch merger.

    Inputs are flat per-patch pixel tensors plus an ``image_grid_thw`` tensor describing the spatial layout of every
    image in the batch. The output is the concatenation of merged image embeddings, ready to be scattered into the
    language-model embedding stream.
    """

    config: HunYuanVLVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["HunYuanVLVisionBlock"]
    _can_record_outputs = {
        "hidden_states": HunYuanVLVisionBlock,
        "attentions": HunYuanVLVisionAttention,
    }

    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__(config)
        self.embeddings = HunYuanVLVisionPatchEmbed(config)
        self.layers = nn.ModuleList([HunYuanVLVisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.perceive = HunYuanVLVisionPatchMerger(config)
        self.gradient_checkpointing = False

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_patches, num_channels * patch_size * patch_size)`):
            Flat per-patch pixel features produced by the image processor.
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions for each image. Each row contains `[t, h, w]` patch counts.
        """
        hidden_states = self.embeddings(pixel_values, grid_thw)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=None, **kwargs)

        split_sizes = grid_thw.prod(dim=-1).tolist()
        split_items = torch.split(hidden_states, split_sizes, dim=1)

        processed_items = []
        for grid, item in zip(grid_thw, split_items):
            _, h, w = grid
            processed_items.append(self.perceive(item, size=(h, w)))

        image_features = torch.cat(processed_items, dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=image_features,
        )


class HunYuanVLTextModel(HunYuanVLPreTrainedModel, HunYuanDenseV1Model):
    """Dense text backbone used inside [`HunYuanVLModel`]."""

    config: HunYuanVLTextConfig

    def __init__(self, config: HunYuanVLTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HunYuanVLDenseV1DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = HunYuanVLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
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
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = _get_past_seq_length(past_key_values)
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)

        causal_position_ids = position_ids[:, 0, :] if position_ids.dim() >= 3 else position_ids
        causal_mask = create_causal_mask(
            self.config,
            inputs_embeds,
            attention_mask,
            past_key_values=past_key_values,
            position_ids=causal_position_ids,
        )

        hidden_states = inputs_embeds
        use_xdrope_prefill = (
            self.rotary_emb.xdrope_section is not None
            and position_ids is not None
            and position_ids.dim() == 3
            and _get_past_seq_length(past_key_values) == 0
        )
        if use_xdrope_prefill:
            rotary_seq_len = max(hidden_states.shape[1], int(position_ids.max().item()) + 1)
            position_embeddings = self.rotary_emb._build_rotary_cache(hidden_states, rotary_seq_len)
        else:
            rotary_position_ids = position_ids
            if rotary_position_ids is not None and rotary_position_ids.dim() == 3:
                rotary_position_ids = rotary_position_ids[:, 0, :]
            position_embeddings = self.rotary_emb(hidden_states, rotary_position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The HunYuanVL model which consists of a vision backbone and a language model, without a language modeling head.
    """
)
class HunYuanVLModel(HunYuanVLPreTrainedModel):
    config: HunYuanVLConfig
    base_model_prefix = "model"

    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        self.language_model = HunYuanVLTextModel(config.text_config)
        self.vit = HunYuanVLVisionTransformer(config.vision_config)
        self.post_init()

    def get_image_position_ids(self, grid_thw: torch.LongTensor, device: str | torch.device | None = None):
        """
        Compute HunYuanVL xdrope spatial indices for the pooled image-token grid of a single image.

        The vision merger appends one newline-style token per image row, so the width channel spans
        `patch_w + 1` positions while the height channel repeats each row id over that extra column.
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        _, grid_h, grid_w = (int(value) for value in grid_thw)
        patch_h = grid_h // spatial_merge_size
        patch_w = grid_w // spatial_merge_size

        position_width = torch.arange(patch_w + 1, dtype=torch.long, device=device).repeat(patch_h)
        position_height = torch.arange(patch_h, dtype=torch.long, device=device).repeat_interleave(patch_w + 1)
        return position_width, position_height

    def get_image_placeholder_spans(self, input_ids: torch.LongTensor) -> list[tuple[int, int]]:
        """
        Locate expanded image placeholder spans in one unpadded token sequence.

        Processor-produced inputs are wrapped as `im_start image* im_end`; manually constructed test inputs can also
        provide a bare contiguous run of image placeholder tokens.
        """
        image_token_id = self.config.image_token_id
        image_start_positions = torch.where(input_ids == self.config.im_start_id)[0]
        spans = []

        if len(image_start_positions) > 0:
            for start_pos in image_start_positions.tolist():
                end_candidates = torch.where(input_ids[start_pos + 1 :] == self.config.im_end_id)[0]
                if len(end_candidates) == 0:
                    raise ValueError("Found an image start token without a matching image end token.")

                end_pos = start_pos + 1 + int(end_candidates[0].item())
                image_positions = torch.where(input_ids[start_pos + 1 : end_pos] == image_token_id)[0]
                if len(image_positions) == 0:
                    continue

                span_start = start_pos + 1 + int(image_positions[0].item())
                span_end = start_pos + 1 + int(image_positions[-1].item()) + 1
                spans.append((span_start, span_end))
            return spans

        image_positions = torch.where(input_ids == image_token_id)[0].tolist()
        if not image_positions:
            return spans

        span_start = image_positions[0]
        previous_position = image_positions[0]
        for position in image_positions[1:]:
            if position != previous_position + 1:
                spans.append((span_start, previous_position + 1))
                span_start = position
            previous_position = position
        spans.append((span_start, previous_position + 1))
        return spans

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """
        Build HunYuanVL's 4-channel `(text_pos, width, height, temporal)` xdrope position ids.

        Text and non-grid image wrapper tokens use the flat sequence index in every channel. The image grid tokens
        inside each placeholder span overwrite the width and height channels with per-image 2D coordinates.
        """
        position_ids = torch.zeros(
            input_ids.shape[0],
            4,
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iter = iter(image_grid_thw) if image_grid_thw is not None else None
        image_index = 0

        for batch_idx, current_input_ids in enumerate(input_ids):
            valid_token_mask = None
            if attention_mask is not None:
                valid_token_mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[valid_token_mask]

            current_position_ids = torch.arange(
                current_input_ids.shape[-1], dtype=input_ids.dtype, device=input_ids.device
            )
            current_position_ids = current_position_ids.view(1, -1).expand(4, -1).clone()

            if grid_iter is not None:
                for span_start, span_end in self.get_image_placeholder_spans(current_input_ids):
                    try:
                        grid_thw = next(grid_iter)
                    except StopIteration as error:
                        raise ValueError(
                            "Found more image placeholder spans than entries in `image_grid_thw`."
                        ) from error

                    position_width, position_height = self.get_image_position_ids(grid_thw, device=input_ids.device)
                    grid_tokens = position_width.shape[0]
                    span_length = span_end - span_start
                    if span_length == grid_tokens + 2:
                        grid_start = span_start + 1
                    elif span_length == grid_tokens:
                        grid_start = span_start
                    else:
                        raise ValueError(
                            "Image placeholder span length does not match `image_grid_thw`: "
                            f"span_length={span_length}, expected {grid_tokens} or {grid_tokens + 2}."
                        )

                    grid_end = grid_start + grid_tokens
                    current_position_ids[1, grid_start:grid_end] = position_width.to(dtype=input_ids.dtype)
                    current_position_ids[2, grid_start:grid_end] = position_height.to(dtype=input_ids.dtype)
                    image_index += 1

            if valid_token_mask is not None:
                position_ids[batch_idx, :, valid_token_mask] = current_position_ids
            else:
                position_ids[batch_idx] = current_position_ids

        if image_grid_thw is not None and image_index != len(image_grid_thw):
            raise ValueError(
                "Found fewer image placeholder spans than entries in `image_grid_thw`: "
                f"spans={image_index}, images={len(image_grid_thw)}."
            )
        return position_ids

    def compute_xdrope_position_ids(
        self,
        input_ids: torch.LongTensor | None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.LongTensor | None:
        if input_ids is None or image_grid_thw is None or _get_past_seq_length(past_key_values) != 0:
            return None
        return self.get_rope_index(input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)

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
        return self.vit(pixel_values, grid_thw=image_grid_thw).pooler_output

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
    ) -> torch.BoolTensor:
        """
        Compute a boolean mask over ``inputs_embeds`` selecting the positions that hold the visual placeholder
        token, and validate that the placeholder count matches the number of provided image features.
        """
        if input_ids is None:
            placeholder_token_embed = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == placeholder_token_embed).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                f"features {image_features.shape[0]}"
            )
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> HunYuanVLModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Flat per-patch pixel features produced by the image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None
        if pixel_values is not None and image_grid_thw is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype, non_blocking=True)
            image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if position_ids is None:
            position_ids = self.compute_xdrope_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        outputs: BaseModelOutputWithPast = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return HunYuanVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_embeds if pixel_values is not None else None,
        )


@auto_docstring
class HunYuanVLForConditionalGeneration(HunYuanVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: HunYuanVLConfig

    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        self.model = HunYuanVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size
        self.post_init()

    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None
    ) -> torch.FloatTensor:
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
    ) -> torch.BoolTensor:
        return self.model.get_placeholder_mask(input_ids, inputs_embeds, image_features)

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
        logits_to_keep: int | torch.Tensor = 0,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
        >>> import torch

        >>> model_id = "tencent/HunyuanOCR"
        >>> processor = AutoProcessor.from_pretrained(model_id, backend="pil")
        >>> model = HunYuanVLForConditionalGeneration.from_pretrained(
        ...     model_id, attn_implementation="eager", torch_dtype=torch.bfloat16, device_map="auto"
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "path/to/your/image.jpg"},
        ...             {"type": "text", "text": "Extract the text from the image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenize=True,
        ...     add_generation_prompt=True,
        ...     return_tensors="pt",
        ...     return_dict=True,
        ...     processor_kwargs={"padding": True},
        ... )

        >>> with torch.no_grad():
        ...     generated_ids = model.generate(**inputs, max_new_tokens=128)
        >>> generated_trimmed = generated_ids[0][inputs["input_ids"].shape[-1]:]
        >>> print(processor.decode(generated_trimmed, skip_special_tokens=True))
        ```"""
        outputs: HunYuanVLModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
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
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        if position_ids is None:
            if is_first_iteration and image_grid_thw is not None:
                position_ids = self.model.compute_xdrope_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
            if position_ids is None:
                text_position_ids = super()._prepare_position_ids_for_generation(
                    input_ids,
                    {"attention_mask": attention_mask, "past_key_values": past_key_values},
                )
                position_ids = text_position_ids[:, None, :].expand(-1, 4, -1)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text_position_ids = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        past_key_values = model_kwargs.get("past_key_values")
        if _get_past_seq_length(past_key_values) != 0:
            return text_position_ids[:, None, :].expand(-1, 4, -1)

        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if is_input_ids and model_kwargs.get("image_grid_thw") is not None:
            position_ids = self.model.compute_xdrope_position_ids(
                input_ids=inputs_tensor,
                image_grid_thw=model_kwargs.get("image_grid_thw"),
                attention_mask=model_kwargs.get("attention_mask"),
                past_key_values=past_key_values,
            )
            if position_ids is not None:
                return position_ids

        return text_position_ids[:, None, :].expand(-1, 4, -1)


__all__ = [
    "HunYuanVLConfig",
    "HunYuanVLVisionConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLImageProcessorKwargs",
    "HunYuanVLImageProcessor",
    "HunYuanVLImageProcessorPil",
    "HunYuanVLPreTrainedModel",
    "HunYuanVLModel",
    "HunYuanVLTextModel",
    "HunYuanVLForConditionalGeneration",
]
