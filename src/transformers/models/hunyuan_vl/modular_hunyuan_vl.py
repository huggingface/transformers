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

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from PIL import Image
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling, SizeDict
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import (
    is_flash_attention_requested,
    maybe_autocast,
)
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_cu_seqlens
from ..hunyuan_v1_dense.configuration_hunyuan_v1_dense import HunYuanDenseV1Config
from ..hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1Attention,
    HunYuanDenseV1DecoderLayer,
    HunYuanDenseV1Model,
    HunYuanDenseV1PreTrainedModel,
    HunYuanDenseV1RotaryEmbedding,
    eager_attention_forward,
    repeat_kv,  # noqa: F401  - re-exported for downstream tooling
    rotate_half,
)
from ..llama.modeling_llama import LlamaRMSNorm
from ..mllama.modeling_mllama import MllamaVisionAttention
from ..qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from ..qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorKwargs,
    Qwen2VLImageProcessorPil,
    smart_resize,
)
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from ..qwen2_vl.modeling_qwen2_vl import Qwen2VLModel
from ..siglip.modeling_siglip import SiglipEncoderLayer, SiglipMLP


@dataclass
class HunYuanVLModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Last image features produced by the vision tower and scattered into the language-model token stream.
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
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Output hidden size produced by the vision tower before it is consumed by the text backbone.
    img_max_token_num (`int`, *optional*, defaults to 4096):
        Maximum image token count expected by the vision stack.
    max_image_size (`int`, *optional*, defaults to 2048):
        Maximum supported image size for the current open-source vision configuration.
    min_image_size (`int`, *optional*, defaults to 512):
        Minimum supported image size for the current open-source vision configuration.
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
    attribute_map = {"attention_heads": "num_attention_heads", "layer_norm_eps": "rms_norm_eps"}

    hidden_act: str = "gelu"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    interpolate_mode: str = "bilinear"
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    num_channels: int = 3
    num_hidden_layers: int = 27
    out_hidden_size: int = 4096
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    img_max_token_num: int = 4096
    max_image_size: int = 2048
    min_image_size: int = 512
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
    rope_parameters (`dict`, *optional*):
        RoPE configuration inherited from [`HunYuanDenseV1Config`]. When `mrope_section` is present, it partitions
        half of each attention head across HunYuanVL's multimodal RoPE axes. The expected order is `(width, height,
        image_index)` for 3-axis multimodal RoPE and `(position, width, height, image_index)` for 4-axis multimodal RoPE. The
        `image_index` axis is the ordinal of the image/frame in the input sequence; all visual tokens from one image
        share the same value on that axis.
    sep_token_id (`int`, *optional*, defaults to 4):
        Token id used as a separator marker by HunYuan tokenizers.
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
        "mrope_section",
    }

    attribute_map = {
        "pad_id": "pad_token_id",
    }

    sep_token_id: int | None = 4
    tie_word_embeddings: bool = True

    @staticmethod
    def _normalize_mrope_section_alias(rope_parameters):
        if not rope_parameters:
            return

        legacy_section = rope_parameters.pop("xdrope_section", None)
        if legacy_section is None:
            return

        mrope_section = rope_parameters.get("mrope_section")
        if mrope_section is not None:
            legacy_values = [float(section) for section in legacy_section]
            mrope_values = [float(section) for section in mrope_section]
            if legacy_values != mrope_values:
                raise ValueError(
                    "`rope_parameters` contains both `mrope_section` and legacy `xdrope_section`, but they differ: "
                    f"mrope_section={mrope_section}, xdrope_section={legacy_section}."
                )
            return

        rope_parameters["mrope_section"] = legacy_section

    @staticmethod
    def _validate_mrope_section(rope_parameters, head_dim):
        if not rope_parameters or rope_parameters.get("mrope_section") is None:
            return

        mrope_section = rope_parameters["mrope_section"]
        section_values = [float(section) for section in mrope_section]
        section_ints = [int(section) for section in section_values]
        expected_sum = head_dim // 2
        if not all(value.is_integer() for value in section_values) or sum(section_ints) != expected_sum:
            raise ValueError(
                f"Illegal mrope partition: expected half-head sections summing to {expected_sum}, got {section_ints}"
            )
        rope_parameters["mrope_section"] = section_ints

    def __post_init__(self, **kwargs):
        # Legacy aliases (`pad_id`, `attention_head_dim`, `org_vocab_size`) are normalized onto canonical fields by the
        # base `__setattr__` via `attribute_map`, so no manual translation is needed here.
        super().__post_init__(**kwargs)
        rope_parameters = getattr(self, "rope_parameters", None)
        head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        self._validate_mrope_section(rope_parameters, head_dim)

    def convert_rope_params_to_dict(self, **kwargs):
        kwargs = PreTrainedConfig.convert_rope_params_to_dict(self, **kwargs)

        rope_parameters = getattr(self, "rope_parameters", None)
        if not rope_parameters:
            return kwargs

        self._normalize_mrope_section_alias(rope_parameters)
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

    image_token_id: int = 120120
    im_start_id: int = 120118
    im_end_id: int = 120119
    im_newline_id: int = 120121
    tie_word_embeddings: bool = True

    vision_start_token_id = AttributeError()
    vision_end_token_id = AttributeError()
    video_token_id = AttributeError()

    def __post_init__(self, **kwargs):
        # When loading legacy "flat" Tencent checkpoints (where text fields live at the top level instead of inside a
        # nested `text_config` block) we fold the recognized text-side keys into the text config payload. This keeps
        # ``HunYuanVLConfig.from_pretrained(...)`` working with both the upstream nested layout and the existing
        # public OCR checkpoints.
        text_keys = set(self.sub_configs["text_config"].__dataclass_fields__) | {"rope_scaling", "rope_theta"}
        text_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in text_keys}

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

        # The attr is saved inside `text_config` on most VLMs, use it if available
        kwargs.setdefault("tie_word_embeddings", self.text_config.tie_word_embeddings)
        PreTrainedConfig.__post_init__(self, **kwargs)


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


@requires(backends=("vision", "torchvision"))
class HunYuanVLImageProcessorPil(Qwen2VLImageProcessorPil):
    size = {"shortest_edge": 512 * 512, "longest_edge": 2048 * 2048}
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    spatial_patch_size = 1
    valid_kwargs = HunYuanVLImageProcessorKwargs

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        all_patches = []
        all_grids = []

        for image in images:
            height, width = image.shape[-2:]
            # Match the original HunyuanOCR preprocessing with PIL.Image.resize
            # FIXME: raushan, investiagte why the quality degrafes with our np-based transforms
            if image.ndim == 3:
                pil_image = Image.fromarray(np.transpose(image, (1, 2, 0)).astype(np.uint8))
            else:
                pil_image = Image.fromarray(image.astype(np.uint8))

            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                # Intentionally do not pass `resample`: the baseline implementation
                # calls `image.resize((width, height))` directly. FIXME raushan
                pil_image = pil_image.resize((resized_width, resized_height))
            else:
                resized_height, resized_width = height, width

            image = np.array(pil_image)
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))

            image = image.astype(np.float32)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            patches = np.expand_dims(image, axis=0)
            if patches.ndim == 4:
                patches = np.expand_dims(patches, axis=1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = np.repeat(
                    patches[:, -1:], temporal_patch_size - (patches.shape[1] % temporal_patch_size), axis=1
                )
                patches = np.concatenate([patches, repeats], axis=1)

            batch_size, grid_t, channel = patches.shape[0], patches.shape[1] // temporal_patch_size, patches.shape[2]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.reshape(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )

            patches = patches.transpose(0, 1, 4, 5, 7, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size * grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            all_patches.append(flatten_patches)
            all_grids.append([grid_t, grid_h, grid_w])

        pixel_values = np.concatenate(all_patches, axis=0)
        image_grid_thw = np.array(all_grids, dtype=np.int64)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

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


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Apply HunYuan's multimodal rotary embedding to ``q`` and ``k``.

    `mrope_section` partitions half of the attention head dimension across the multimodal axes produced by
    `HunYuanVLModel.get_rope_index`. The section order matches the position-id channel order: `(width, height,
    image_index)` for 3-axis multimodal RoPE and `(position, width, height, image_index)` for 4-axis multimodal RoPE.
    """
    x_dim = len(mrope_section)
    mrope_section = [int(section) * 2 for section in mrope_section]
    if sum(mrope_section) != cos.shape[-1]:
        raise ValueError(
            f"Illegal partition for multimodal RoPE: expected {cos.shape[-1]} rotary dims, got {sum(mrope_section)}"
        )

    cos = torch.cat([m[i % x_dim] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
    sin = torch.cat([m[i % x_dim] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

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
        self.mrope_section = rope_parameters.get("mrope_section")

    def forward(self, x, position_ids):
        # In contrast to other models, model has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(len(self.mrope_section), position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (mrope_section, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class HunYuanVLVisionMLP(SiglipMLP):
    pass


class HunYuanVLVisionPatchEmbed(nn.Module):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__()
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
        """Interpolate the learned patch positional grid to the current image grid size."""
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
        self.spatial_merge_size = config.spatial_merge_size
        self.proj_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size * 2,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.proj_act = ACT2FN[config.hidden_act]
        self.proj_out = nn.Conv2d(config.hidden_size * 2, config.hidden_size * 4, kernel_size=1)
        self.mlp = nn.Linear(config.hidden_size * 4, config.text_hidden_size)
        self.image_newline = nn.Parameter(torch.empty(config.hidden_size * 4))
        self.image_begin = nn.Parameter(torch.empty(config.text_hidden_size))
        self.image_end = nn.Parameter(torch.empty(config.text_hidden_size))
        self.image_sep = nn.Parameter(torch.empty(config.text_hidden_size))

        self.before_rms = HunYuanVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.after_rms = HunYuanVLRMSNorm(config.text_hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        hidden_states = self.before_rms(hidden_states)
        dtype = hidden_states.dtype
        hidden_states = hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -1, *size)
        hidden_states = self.proj_conv(hidden_states)
        hidden_states = self.proj_act(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = torch.cat(
            [
                hidden_states,
                self.image_newline.reshape(1, channels, 1, 1)
                .expand(batch_size, channels, height, 1)
                .to(dtype, non_blocking=True),
            ],
            dim=-1,
        )
        hidden_states = hidden_states.reshape(batch_size, channels, -1).permute(0, 2, 1)
        hidden_states = self.mlp(hidden_states)

        begin = (
            self.image_begin.reshape(1, 1, -1)
            .expand(batch_size, 1, hidden_states.shape[-1])
            .to(device=hidden_states.device, dtype=dtype, non_blocking=True)
        )
        end = (
            self.image_end.reshape(1, 1, -1)
            .expand(batch_size, 1, hidden_states.shape[-1])
            .to(device=hidden_states.device, dtype=dtype, non_blocking=True)
        )
        hidden_states = torch.cat([begin, hidden_states, end], dim=1)

        return self.after_rms(hidden_states)


class HunYuanVLVisionAttention(MllamaVisionAttention):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__(config)
        self.is_causal = False
        self.head_dim = getattr(config, "head_dim", self.embed_dim // self.num_heads)
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        batch_size, seq_len, _ = query.shape

        query_states = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                scaling=self.scaling,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    scaling=self.scaling,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
            attn_weights = None

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class HunYuanVLVisionBlock(SiglipEncoderLayer):
    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__(config)
        self.self_attn = HunYuanVLVisionAttention(config)
        self.mlp = HunYuanVLVisionMLP(config)


class HunYuanVLDenseV1Attention(HunYuanDenseV1Attention):
    """
    HunYuan dense attention with optional multimodal rotary embeddings.

    When ``rope_parameters['mrope_section']`` is set, queries and keys are rotated with HunYuan's multimodal axes.
    During decoding all axes carry the same text-only positions, reducing the operation to standard 1D RoPE.
    """

    def __init__(self, config: HunYuanVLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.mrope_section = rope_parameters.get("mrope_section")

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

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.mrope_section
        )

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

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, HunYuanVLVisionPatchMerger):
            embed_std = module.config.text_hidden_size**-0.5
            init.normal_(module.image_newline, mean=0.0, std=embed_std)
            init.normal_(module.image_begin, mean=0.0, std=embed_std)
            init.normal_(module.image_end, mean=0.0, std=embed_std)
            init.normal_(module.image_sep, mean=0.0, std=embed_std)


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
    _can_record_outputs = {
        "hidden_states": HunYuanVLVisionBlock,
        "attentions": HunYuanVLVisionAttention,
    }

    def __init__(self, config: HunYuanVLVisionConfig):
        super().__init__(config)
        self.embeddings = HunYuanVLVisionPatchEmbed(config)
        self.layers = nn.ModuleList([HunYuanVLVisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.patch_merger = HunYuanVLVisionPatchMerger(config)
        self.gradient_checkpointing = False

        self.post_init()

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
        cu_seqlens = get_vision_cu_seqlens(grid_thw, kwargs=kwargs)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens, attention_mask=None, **kwargs)

        split_sizes = grid_thw.prod(dim=-1).tolist()
        split_items = torch.split(hidden_states, split_sizes, dim=1)

        processed_items = []
        for grid, item in zip(grid_thw.tolist(), split_items):
            _, h, w = grid
            processed_items.append(self.patch_merger(item, size=(h, w)))

        image_features = torch.cat(processed_items, dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=image_features,
        )


class HunYuanVLTextModel(HunYuanVLPreTrainedModel, HunYuanDenseV1Model):
    """Dense text backbone used inside [`HunYuanVLModel`]."""

    config: HunYuanVLTextConfig
    input_modalities = ("text",)

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

        num_mrope_axes = len(self.rotary_emb.mrope_section or [])

        # Expand to 3D with `num_mrope_axes` as first dim, if not yet expanded
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(num_mrope_axes, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(num_mrope_axes, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == num_mrope_axes + 1:
            text_position_ids = position_ids[0]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        if num_mrope_axes and position_ids.dim() == 3:
            if position_ids.shape[0] != num_mrope_axes:
                raise ValueError(
                    f"Expected {num_mrope_axes} multimodal RoPE channels, got "
                    f"position_ids with shape {tuple(position_ids.shape)}."
                )

        causal_mask = create_causal_mask(
            self.config,
            inputs_embeds,
            attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
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
class HunYuanVLModel(Qwen2VLModel):
    def __init__(self, config: HunYuanVLConfig):
        super().__init__(config)
        self.vision_tower = HunYuanVLVisionTransformer(config.vision_config)
        del self.visual

    def get_vision_position_ids(
        self,
        grid_hw: list[int, int] | torch.Tensor,
        spatial_merge_size: int = 1,
        device: str | torch.device | None = None,
    ):
        """
        Compute HunYuanVL multimodal RoPE spatial indices for the pooled image-token grid of a single image.

        The vision merger appends one newline-style token per image row, so the width channel spans
        `patch_w + 1` positions while the height channel repeats each row id over that extra column.
        """
        grid_h, grid_w = (int(value) for value in grid_hw)
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size

        position_height, position_width = torch.meshgrid(
            torch.arange(llm_grid_h, dtype=torch.long, device=device),
            torch.arange(llm_grid_w + 1, dtype=torch.long, device=device),
            indexing="ij",
        )
        return torch.stack([position_width.flatten(), position_height.flatten()], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Build HunYuanVL multimodal RoPE position ids.

        `rope_parameters["mrope_section"]` controls both the number and order of multimodal RoPE axes. Each section
        value is the number of half-rotary dimensions assigned to the corresponding axis, and the sections must sum to
        `head_dim // 2`.

        This method returns only the multimodal rotary axes consumed by the text backbone. The last three axes are
        `(width, height, image_index)`; any preceding axes keep their default 1D sequence positions.

        `width` and `height` index the pooled visual grid inside one image. `image_index` is the ordinal of the
        image/frame in the input sequence, so all visual tokens from the first image get `0`, the second image get
        `1`, and so on. Text-only 1D positions for the causal mask are inferred by the text backbone and are not part
        of this return value.
        """
        rope_parameters = self.config.text_config.rope_parameters or {}
        num_mrope_axes = len(rope_parameters.get("mrope_section", []))
        if num_mrope_axes < 3:
            raise ValueError(f"HunYuanVL expects at least 3 multimodal RoPE axes, got {num_mrope_axes}.")
        position_ids = torch.zeros(
            num_mrope_axes,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iter = iter(image_grid_thw) if image_grid_thw is not None else None
        rope_deltas = []
        image_index = 0

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            valid_token_mask = None
            if attention_mask is not None:
                valid_token_mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[valid_token_mask]
                input_token_type = input_token_type[valid_token_mask]

            current_position_ids = torch.arange(
                current_input_ids.shape[-1], dtype=input_ids.dtype, device=input_ids.device
            )
            current_position_ids = current_position_ids.view(1, -1).expand(num_mrope_axes, -1).clone()

            if grid_iter is not None:
                for modality_type, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                    if modality_type != 1:
                        continue
                    group = list(group)
                    span_start = group[0][0]
                    span_end = group[-1][0] + 1
                    try:
                        grid_thw = next(grid_iter)
                    except StopIteration as error:
                        raise ValueError(
                            "Found more image placeholder spans than entries in `image_grid_thw`."
                        ) from error

                    vision_position_ids = self.get_vision_position_ids(
                        grid_thw[1:],
                        spatial_merge_size=self.config.vision_config.spatial_merge_size,
                        device=input_ids.device,
                    )
                    grid_tokens = vision_position_ids.shape[1]
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
                    offset = num_mrope_axes - 3
                    current_position_ids[offset : offset + 2, grid_start:grid_end] = vision_position_ids.to(
                        dtype=input_ids.dtype
                    )
                    current_position_ids[offset + 2, grid_start:grid_end] = image_index
                    image_index += 1

            if valid_token_mask is not None:
                position_ids[:, batch_idx, valid_token_mask] = current_position_ids
            else:
                position_ids[:, batch_idx] = current_position_ids
            rope_deltas.append(current_position_ids.max() + 1 - len(current_input_ids))

        if image_grid_thw is not None and image_index != len(image_grid_thw):
            raise ValueError(
                "Found fewer image placeholder spans than entries in `image_grid_thw`: "
                f"spans={image_index}, images={len(image_grid_thw)}."
            )
        rope_deltas = torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, rope_deltas

    def compute_3d_position_ids(
        self,
        input_ids: torch.LongTensor | None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.LongTensor | None:
        past_seen_tokens = 0 if past_key_values is None else past_key_values.get_seq_length()
        if image_grid_thw is not None and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed via `image_grid_thw` but `mm_token_type_ids` is missing. Please pass "
                "`mm_token_type_ids` to the model so that multimodal RoPE can be computed correctly. "
                "`mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )
        if input_ids is None or image_grid_thw is None or past_seen_tokens != 0:
            return None
        rope_positions, rope_deltas = self.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        self.rope_deltas = rope_deltas
        return rope_positions

    def get_video_features(self, **super_kwargs):
        raise AttributeError("Model doesn't support videos")

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Encode images and return the complete vision tower outputs.

        pixel_values (`torch.FloatTensor`):
            Flat per-patch pixel features produced by the image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        vision_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(vision_dtype)
        return self.vision_tower(pixel_values, grid_thw=image_grid_thw, **kwargs)

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
        mm_token_type_ids: torch.IntTensor | None = None,
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
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = image_outputs.pooler_output
            image_embeds = image_embeds.to(inputs_embeds.device, dtype=inputs_embeds.dtype, non_blocking=True)
            image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
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
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
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
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Flat per-patch pixel features produced by the image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.

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
            mm_token_type_ids=mm_token_type_ids,
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
        mm_token_type_ids=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        position_ids = model_inputs.get("position_ids")
        if position_ids is not None and position_ids.dim() == 3:
            rope_parameters = self.config.text_config.rope_parameters or {}
            num_mrope_axes = len(rope_parameters.get("mrope_section", []))
            if num_mrope_axes and position_ids.shape[0] != num_mrope_axes:
                batch_size_source = model_inputs.get("input_ids")
                if batch_size_source is None:
                    batch_size_source = model_inputs.get("inputs_embeds")
                if batch_size_source is None:
                    batch_size_source = attention_mask
                batch_size = batch_size_source.shape[0]
                if position_ids.shape[0] % num_mrope_axes == 0 and position_ids.shape[1] != batch_size:
                    expand_size = position_ids.shape[0] // num_mrope_axes
                    position_ids = position_ids.view(expand_size, num_mrope_axes, *position_ids.shape[1:])
                    position_ids = position_ids.permute(1, 0, 2, 3).reshape(num_mrope_axes, -1, position_ids.shape[-1])
                    model_inputs["position_ids"] = position_ids

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Same as qwen-vl with variable `num_mrope_axes` based on config values
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        rope_parameters = self.config.text_config.rope_parameters or {}
        num_mrope_axes = len(rope_parameters.get("mrope_section", []))

        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            return (text_positions + self.model.rope_deltas).unsqueeze(0).expand(num_mrope_axes, -1, -1)

        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if (
            is_input_ids
            and model_kwargs.get("mm_token_type_ids") is not None
            and model_kwargs.get("image_grid_thw") is not None
        ):
            rope_positions, rope_deltas = self.model.get_rope_index(
                inputs_tensor,
                mm_token_type_ids=model_kwargs.get("mm_token_type_ids"),
                image_grid_thw=model_kwargs.get("image_grid_thw"),
                attention_mask=model_kwargs.get("attention_mask"),
            )
            self.model.rope_deltas = rope_deltas
        else:
            rope_positions = text_positions.unsqueeze(0).expand(num_mrope_axes, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        return rope_positions


__all__ = [
    "HunYuanVLConfig",
    "HunYuanVLVisionConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLImageProcessor",
    "HunYuanVLImageProcessorPil",
    "HunYuanVLPreTrainedModel",
    "HunYuanVLModel",
    "HunYuanVLTextModel",
    "HunYuanVLVisionTransformer",
    "HunYuanVLForConditionalGeneration",
]
