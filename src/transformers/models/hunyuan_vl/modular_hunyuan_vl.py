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

import inspect
from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..hunyuan_v1_dense.configuration_hunyuan_v1_dense import HunYuanDenseV1Config
from ..hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1Attention,
    HunYuanDenseV1DecoderLayer,
    HunYuanDenseV1Model,
    HunYuanDenseV1RotaryEmbedding,
    apply_rotary_pos_emb,
)
from ..llama.modeling_llama import LlamaRMSNorm, rotate_half


HUNYUAN_VL_TEXT_FORWARD_CUSTOM_ARGS = r"""
    cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
        Indices describing the absolute position of each input token. Used to derive default `position_ids` and to
        update the key-value cache during generation.
"""


@auto_docstring(checkpoint="tencent/HunyuanOCR")
@strict
class HunYuanVLVisionConfig(PretrainedConfig):
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

    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=1152,
        intermediate_size=4304,
        interpolate_mode="bilinear",
        rms_norm_eps=1e-05,
        learnable_mlp_pooling_size=0,
        attention_dropout=0.0,
        num_attention_heads=16,
        num_key_value_heads=None,
        num_channels=3,
        num_hidden_layers=27,
        out_hidden_size=4096,
        patch_size=16,
        remove_prenorm=True,
        spatial_merge_size=2,
        temporal_patch_size=1,
        resize_resolution=2048,
        img_max_token_num=4096,
        max_image_size=2048,
        min_image_size=512,
        anyres_vit_max_image_size=2048,
        max_vit_seq_len=16384,
        text_hidden_size=3072,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.interpolate_mode = interpolate_mode
        self.learnable_mlp_pooling_size = learnable_mlp_pooling_size
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads
        if not num_key_value_heads:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.remove_prenorm = remove_prenorm
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps

        self.resize_resolution = resize_resolution
        self.img_max_token_num = img_max_token_num
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.anyres_vit_max_image_size = anyres_vit_max_image_size
        self.max_vit_seq_len = max_vit_seq_len
        self.text_hidden_size = text_hidden_size


@auto_docstring(checkpoint="tencent/Hunyuan-A13B-Instruct")
@strict
class HunYuanVLTextConfig(HunYuanDenseV1Config):
    r"""
    eod_token_id (int, *optional*, defaults to 3):
        Token ID representing the end-of-document marker. Used to indicate the termination of a text sequence.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base period used by RoPE. Preserved explicitly so Tencent checkpoints that save legacy top-level rope fields
        can be normalized into the standard `rope_parameters` structure without losing the original theta value.
    rope_scaling (`dict`, *optional*):
        Legacy RoPE scaling payload from Tencent checkpoints. When provided, it is normalized into
        `rope_parameters` and kept in sync for backward compatibility during config loading.
    pad_id (`int`, *optional*):
        Legacy padding token field from Tencent checkpoints. When `pad_token_id` is unset or `-1`, this value is
        normalized into `pad_token_id`.
    attention_head_dim (`int`, *optional*):
        Legacy alias for `head_dim`. When `head_dim` is not provided, this value is used as the per-head hidden size.
    org_vocab_size (`int`, *optional*):
        Original vocabulary size recorded in exported checkpoints for compatibility with Tencent tooling.
    routed_scaling_factor (`float`, *optional*, defaults to 1.0):
        Legacy routing scaling field kept only for checkpoint compatibility in the dense-only open-source variant.
    use_qk_norm (`bool`, *optional*, defaults to `False`):
        Whether to enable query/key normalization in the text backbone.
    use_cla (`bool`, *optional*, defaults to `False`):
        Whether to enable CLA-specific behavior present in some checkpoints.
    enable_lm_head_fp32 (`bool`, *optional*, defaults to `False`):
        Whether to execute the LM head in float32 for numerical stability.
    num_experts (`int | list[int] | None`, *optional*, defaults to 1):
        Legacy MoE field kept only for checkpoint compatibility. The open-source `hunyuan_vl` implementation is dense-only.
    moe_topk (`int | list[int] | None`, *optional*, defaults to 1):
        Legacy MoE field kept only for checkpoint compatibility. The open-source `hunyuan_vl` implementation is dense-only.
    num_shared_expert (`int | list[int] | None`, *optional*):
        Legacy MoE field kept only for checkpoint compatibility. It is preserved for loading Tencent checkpoints but does
        not enable shared-expert execution in the dense-only open-source variant.
    moe_layer_num_skipped (`int`, *optional*, defaults to 0):
        Legacy checkpoint field indicating how many initial layers skipped MoE routing in internal training variants.
        Preserved for checkpoint compatibility only.
    enable_moe_fp32_combine (`bool`, *optional*, defaults to `True`):
        Legacy checkpoint flag preserved for compatibility. It has no effect on runtime execution in the dense-only
        open-source implementation.
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

    pad_id: int | None = None
    attention_head_dim: int | None = None
    org_vocab_size: int | None = None
    routed_scaling_factor: float = 1.0
    use_qk_norm: bool = False
    use_cla: bool = False
    enable_lm_head_fp32: bool = False
    rope_scaling: dict | None = None
    rope_theta: float = 10000.0
    sep_token_id: int | None = 4
    num_experts: int | list[int] | None = 1
    moe_topk: int | list[int] | None = 1
    moe_intermediate_size: int | list[int] | None = None
    num_shared_expert: int | list[int] | None = None
    moe_layer_num_skipped: int = 0
    enable_moe_fp32_combine: bool = True

    def __init__(
        self,
        vocab_size=290943,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eod_token_id=3,
        sep_token_id=4,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=None,
        pad_id=None,
        attention_head_dim=None,
        org_vocab_size=None,
        routed_scaling_factor=1.0,
        use_qk_norm=False,
        use_cla=False,
        enable_lm_head_fp32=False,
        num_experts=1,
        moe_topk=1,
        moe_intermediate_size=None,
        num_shared_expert=None,
        moe_layer_num_skipped=0,
        enable_moe_fp32_combine=True,
        **kwargs,
    ):
        if head_dim is None:
            head_dim = attention_head_dim
        if pad_token_id == -1 and pad_id not in (None, -1):
            pad_token_id = pad_id

        rope_parameters = self._normalize_rope_parameters(kwargs.pop("rope_parameters", None), rope_scaling, rope_theta)
        self.rope_parameters = rope_parameters

        _ = super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            eod_token_id=eod_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            head_dim=head_dim,
            **kwargs,
        )

        self.sep_token_id = sep_token_id
        self.pad_id = pad_id
        self.attention_head_dim = attention_head_dim
        self.org_vocab_size = org_vocab_size
        self.routed_scaling_factor = routed_scaling_factor
        self.use_qk_norm = use_qk_norm
        self.use_cla = use_cla
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.rope_scaling = rope_parameters
        self.rope_theta = rope_theta if rope_parameters is None else rope_parameters["rope_theta"]
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.moe_intermediate_size = moe_intermediate_size
        self.num_shared_expert = num_shared_expert
        self.moe_layer_num_skipped = moe_layer_num_skipped
        self.enable_moe_fp32_combine = enable_moe_fp32_combine

    @staticmethod
    def _normalize_rope_parameters(
        rope_parameters: dict | None, rope_scaling: dict | None, rope_theta: float
    ) -> dict | None:
        if rope_parameters is None and rope_scaling is not None:
            rope_parameters = dict(rope_scaling)
        elif rope_parameters is not None:
            rope_parameters = dict(rope_parameters)

        if rope_parameters is None:
            rope_parameters = {}

        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        if rope_type == "xdrope":
            rope_type = "dynamic"
        rope_parameters["rope_type"] = rope_type
        rope_parameters["type"] = rope_type
        rope_parameters.setdefault("rope_theta", rope_theta)
        return rope_parameters

    def _apply_compat_fields(self) -> "HunYuanVLTextConfig":
        if self.head_dim is None:
            self.head_dim = getattr(self, "attention_head_dim", None)
        if self.pad_token_id == -1 and getattr(self, "pad_id", None) not in (None, -1):
            self.pad_token_id = self.pad_id

        rope_parameters = self._normalize_rope_parameters(
            getattr(self, "rope_parameters", None),
            getattr(self, "rope_scaling", None),
            getattr(self, "rope_theta", 10000.0),
        )
        self.rope_parameters = rope_parameters
        self.rope_scaling = rope_parameters
        if rope_parameters is not None:
            self.rope_theta = rope_parameters["rope_theta"]

        return self

    def _rope_parameters_validation(self):
        if getattr(self, "rope_parameters", None) is None and getattr(self, "rope_scaling", None) is None:
            return

        self.standardize_rope_params()
        self.validate_rope()


@auto_docstring(checkpoint="tencent/HunyuanOCR")
@strict
class HunYuanVLConfig(PretrainedConfig):
    r"""
    Top-level configuration for the open-source HunYuanVL integration.

    This configuration describes the dense-only, image-text-only variant used for OCR and document-understanding style
    workloads. Legacy MoE- and Tencent-export-related fields may still appear in nested text configs for checkpoint
    compatibility, but they do not enable MoE runtime behavior in this open-source implementation.

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
        im_start_id=120118,
        im_end_id=120119,
        image_token_id=120120,
        im_newline_id=120121,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        text_kwargs = self._extract_text_kwargs(kwargs)
        self.text_config = self._build_text_config(text_config, text_kwargs, kwargs)

        self.image_token_id = image_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.im_newline_id = im_newline_id

        self.vision_config.text_hidden_size = self.text_config.hidden_size

        attn_implementation = kwargs.pop(
            "attn_implementation", getattr(self.text_config, "_attn_implementation_internal", None)
        )
        experts_implementation = kwargs.pop(
            "experts_implementation", getattr(self.text_config, "_experts_implementation_internal", None)
        )

        super().__init__(
            pad_token_id=self.text_config.pad_token_id,
            bos_token_id=self.text_config.bos_token_id,
            eos_token_id=self.text_config.eos_token_id,
            tie_word_embeddings=self.text_config.tie_word_embeddings,
            attn_implementation=attn_implementation,
            experts_implementation=experts_implementation,
            **kwargs,
        )

    @classmethod
    def _extract_text_kwargs(cls, kwargs: dict) -> dict:
        text_signature = inspect.signature(cls.sub_configs["text_config"].__init__).parameters
        text_keys = set(text_signature) | {"rope_scaling", "rope_theta"}
        return {key: kwargs.pop(key) for key in list(kwargs) if key in text_keys}

    @classmethod
    def _build_text_config(cls, text_config, text_kwargs: dict, kwargs: dict) -> HunYuanVLTextConfig:
        text_config_class = cls.sub_configs["text_config"]

        if text_config is None:
            normalized_text_config = dict(text_kwargs)
            normalized_text_config.setdefault("dtype", kwargs.get("torch_dtype", kwargs.get("dtype")))
            return text_config_class(**normalized_text_config)

        if isinstance(text_config, dict):
            normalized_text_config = dict(text_config)
            normalized_text_config.update(text_kwargs)
            return text_config_class(**normalized_text_config)

        if text_kwargs:
            text_config_dict = text_config.to_dict()
            text_config_dict.update(text_kwargs)
            return text_config_class(**text_config_dict)

        return text_config._apply_compat_fields()


class HunYuanVLVisionMLP(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.dense_h_to_4h = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.dense_4h_to_h = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, x):
        intermediate = self.dense_h_to_4h(x)
        intermediate = self.act_fn(intermediate)
        output = self.dense_4h_to_h(intermediate)
        return output


class HunYuanVLRMSNorm(LlamaRMSNorm):
    pass


class HunYuanVLRotaryEmbedding(HunYuanDenseV1RotaryEmbedding):
    pass


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
        # first token is cls token, skip it
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.patch_pos_embed = None

    def forward(self, pixel_values: torch.Tensor, grid_thw: list[list[int]]) -> torch.Tensor:
        num_patches, hidden_size = pixel_values.shape
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
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
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
        embeddings = patch_embeds + patch_pos_embed

        return embeddings


class HunYuanVLVisionPatchMerger(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_merge_size,
        rms_norm_eps,
        **kwargs,
    ):
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

    def forward(self, x, size=(16, 16)):
        x = self.before_rms(x)
        h, w = size
        dtype = x.dtype
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, int(h.item()), int(w.item()))
        x = self.proj(x)  # b,c,h,w
        b, c, h, w = x.shape
        x = torch.cat(
            [x, self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype, non_blocking=True)], dim=-1
        )
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        x = self.mlp(x)

        begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype, non_blocking=True)
        end = self.image_end.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype, non_blocking=True)
        x = torch.cat([begin, x, end], dim=1)

        return self.after_rms(x)


class HunYuanVLVisionAttention(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
        self.config = config
        self.is_causal = False  # used in flash_attention
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface = HunYuanVLPreTrainedModel.get_attention_interface(self.config)

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
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HunYuanVLVisionTransformer(nn.Module):
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

    def get_activation_function(self, act_name: str):
        act_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        return act_map.get(act_name.lower(), nn.GELU())  # default GELU

    # @auto_docstring
    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        #
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
        """
        hidden_states = self.embeddings(x, grid_thw)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        cu_seqlens: list = [0]
        for t, h, w in grid_thw:
            cu_seqlens.append((h * w).item())

        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        split_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        split_items = torch.split(hidden_states, split_lengths, dim=1)

        processed_items = []
        for grid, item in zip(grid_thw, split_items):
            t, h, w = grid
            processed = self.perceive(item, size=(h, w))
            processed_items.append(processed)

        hidden_states = torch.cat(processed_items, dim=1)

        return hidden_states


@auto_docstring
class HunYuanVLPreTrainedModel(PreTrainedModel):
    config_class = HunYuanVLConfig
    config: HunYuanVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    @staticmethod
    def build_rotary_cache_from_inv_freq(
        rotary_emb, x: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, rotary_emb.inv_freq.to(x.device).float())
        emb = torch.cat((freqs, freqs), dim=-1).float()
        attention_scaling = getattr(rotary_emb, "attention_scaling", 1.0)
        cos = (emb.cos() * attention_scaling).to(dtype=x.dtype)
        sin = (emb.sin() * attention_scaling).to(dtype=x.dtype)
        return cos, sin

    @staticmethod
    def get_past_seq_length(past_key_values: Cache | None, layer_idx: int | None = None) -> int:
        if past_key_values is None:
            return 0
        try:
            seq_len = past_key_values.get_seq_length(layer_idx)
        except TypeError:
            seq_len = past_key_values.get_seq_length()
        if isinstance(seq_len, torch.Tensor):
            return int(seq_len.max().item())
        return int(seq_len)

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states

        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Keep the eager fallback local to HunYuanVL so vision attention does not depend on the dense text helper.
        key_states = HunYuanVLPreTrainedModel.repeat_kv(key, module.num_key_value_groups)
        value_states = HunYuanVLPreTrainedModel.repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    @staticmethod
    def get_attention_interface(config) -> Callable:
        if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
            return ALL_ATTENTION_FUNCTIONS.get_interface(
                config._attn_implementation, HunYuanVLPreTrainedModel.eager_attention_forward
            )
        if config._attn_implementation == "eager":
            return HunYuanVLPreTrainedModel.eager_attention_forward
        return ALL_ATTENTION_FUNCTIONS[config._attn_implementation]

    @staticmethod
    def update_past_key_values(
        past_key_values: Cache,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_position: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = {"sin": sin, "cos": cos}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position
        try:
            return past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)
        except TypeError:
            return past_key_values.update(key_states, value_states, layer_idx)

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)

        # `inv_freq`/`original_inv_freq` are non-persistent buffers, so they are recreated
        # during `from_pretrained` finalization and must be re-derived from the RoPE config.
        if "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
            if module.rope_type == "dynamic" and module.config.rope_parameters.get("alpha"):
                dim = module.config.head_dim
                rope_theta = module.config.rope_parameters["rope_theta"]
                alpha = module.config.rope_parameters["alpha"]

                base = rope_theta * alpha ** (dim / (dim - 2))
                buffer_value = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            else:
                rope_fn = (
                    ROPE_INIT_FUNCTIONS[module.rope_type]
                    if module.rope_type != "default"
                    else module.compute_default_rope_parameters
                )
                buffer_value, _ = rope_fn(module.config)

            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)


class HunYuanVLDenseV1Attention(HunYuanDenseV1Attention):
    def __init__(self, config: HunYuanVLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.xdrope_section = self._normalize_xdrope_section(rope_parameters.get("xdrope_section"), self.head_dim)
        self.rotary_emb = None

    @staticmethod
    def _normalize_xdrope_section(xdrope_section, head_dim: int) -> list[int] | None:
        if xdrope_section is None:
            return None

        section_values = [float(section) for section in xdrope_section]
        section_ints = [int(section) for section in section_values]

        # Real HunYuan checkpoints store absolute half-head partition sizes, e.g. [16, 16, 16, 16] for head_dim=128.
        if all(value.is_integer() for value in section_values) and sum(section_ints) * 2 == head_dim:
            return section_ints

        # Lightweight tests may use ratio-style sections that sum to 1.0, e.g. [0.25, 0.25, 0.25, 0.25].
        if all(section <= 1.0 for section in section_values):
            return [int(section * head_dim / 2) for section in section_values]

        return section_ints

    @staticmethod
    def _apply_rotary_pos_emb_xdrope(q, k, cos, sin, position_ids, xdrope_section, output_size=None):
        x_dim = len(xdrope_section)
        cos = (
            cos[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()
        )
        sin = (
            sin[position_ids, ...].permute(0, 2, 1, 3).reshape(output_size[0], output_size[2], x_dim, -1).contiguous()
        )

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
        q_out, k_out = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

        return q_out.to(origin_dtype), k_out.to(origin_dtype)

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
            and HunYuanVLPreTrainedModel.get_past_seq_length(past_key_values, self.layer_idx) == 0
        )

        if use_xdrope_prefill:
            rotary_seq_len = max(key_states.shape[-2], int(position_ids.max().item()) + 1)
            cos, sin = HunYuanVLPreTrainedModel.build_rotary_cache_from_inv_freq(
                self.rotary_emb, value_states, rotary_seq_len
            )
            output_size = (
                query_states.size(0),
                query_states.size(1),
                query_states.size(2),
                key_states.size(2),
            )
            query_states, key_states = self._apply_rotary_pos_emb_xdrope(
                query_states, key_states, cos, sin, position_ids, self.xdrope_section, output_size
            )
        else:
            if position_embeddings is None:
                if position_ids is not None and position_ids.dim() == 3:
                    position_ids = position_ids[:, 0, :]
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if past_key_values is not None:
            key_states, value_states = HunYuanVLPreTrainedModel.update_past_key_values(
                past_key_values, key_states, value_states, self.layer_idx, cos, sin, cache_position
            )

        attention_interface = HunYuanVLPreTrainedModel.get_attention_interface(self.config)
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


class HunYuanVLDenseV1Model(HunYuanDenseV1Model):
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
            past_seen_tokens = HunYuanVLPreTrainedModel.get_past_seq_length(past_key_values)
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
        self.model = HunYuanVLDenseV1Model(text_config)
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

        >>> model = HunYuanVLForCausalLM.from_pretrained("meta-hunyuan_vl/HunYuanVL-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-hunyuan_vl/HunYuanVL-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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


class HunYuanVLForConditionalGeneration(HunYuanVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config: HunYuanVLConfig

    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        text_config = config.text_config
        self.model = HunYuanVLDenseV1Model(text_config)
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

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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

        >>> model_name_or_path = "tencent/HunYuanOCR"
        >>> processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
        >>> model = HunYuanVLForConditionalGeneration.from_pretrained(
        ...     model_name_or_path,
        ...     attn_implementation="eager",
        ...     torch_dtype=torch.bfloat16,
        ...     device_map="auto",
        ... )

        >>> img_path = "path/to/your/image.jpg"
        >>> image = Image.open(img_path).convert("RGB")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": img_path},
        ...             {"type": "text", "text": "Extract the text from the image."},
        ...         ],
        ...     }
        ... ]
        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        >>> with torch.no_grad():
        ...     generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = generated_ids[0][len(inputs["input_ids"][0]):]
        >>> output = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

        >>> print(output)

        ```"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        orig_input_ids = input_ids
        target_device = input_ids.device if input_ids is not None else inputs_embeds.device

        if self.vit is not None and pixel_values is not None and image_grid_thw is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(target_device, dtype=self.dtype, non_blocking=True)
            image_mask = self.get_placeholder_mask(
                orig_input_ids,
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor | None = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        """
        vit_dtype = next(self.vit.parameters()).dtype
        pixel_values = pixel_values.to(vit_dtype)
        return self.vit(pixel_values, grid_thw=image_grid_thw)

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
        is_decode_step = HunYuanVLPreTrainedModel.get_past_seq_length(past_key_values) > 0
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

    # Copied from transformers.models.llava.modeling_llava.LlavaModel.get_placeholder_mask
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        token_id: int | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if token_id is None:
            token_id = self.config.image_token_id

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        return special_image_mask


__all__ = [
    "HunYuanVLConfig",
    "HunYuanVLVisionConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLForConditionalGeneration",
    "HunYuanVLForCausalLM",
    "HunYuanVLPreTrainedModel",
]
