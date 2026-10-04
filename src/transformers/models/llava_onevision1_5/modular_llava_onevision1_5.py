# Copyright 2026 The LLaVA-OneVision team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LLaVA-OneVision-1.5 model."""

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, torch_compilable_check
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..glm_image.modeling_glm_image import GlmImageVisionPatchEmbed
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLVisionRotaryEmbedding,
)
from ..siglip.modeling_siglip import SiglipMLP


@auto_docstring(checkpoint="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct")
@strict
class LlavaOnevision1_5VisionConfig(Qwen2_5_VLVisionConfig):
    r"""
    window_size (`int`, *optional*, defaults to 112):
        Size of the local attention windows. All layers use full attention in this model.
    out_hidden_size (`int`, *optional*, defaults to 2560):
        Output size of the vision encoder.
    fullatt_block_indexes (`list[int]`, *optional*):
        Indices of layers using full attention.
    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
        The epsilon used by the vision encoder layer normalization layers.
    """

    model_type = "llava_onevision1_5_vision"
    base_config_key = "vision_config"

    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "gelu"
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size = AttributeError()
    tokens_per_second = AttributeError()
    window_size: int = 112
    out_hidden_size: int = 2560
    fullatt_block_indexes: list[int] | tuple[int, ...] = tuple(range(24))
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    rope_parameters: dict | None = None


@auto_docstring(checkpoint="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct")
@strict
class LlavaOnevision1_5Config(PreTrainedConfig):
    model_type = "llava_onevision1_5"
    sub_configs = {"vision_config": AutoConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = dict(self.vision_config)
            self.vision_config["model_type"] = "llava_onevision1_5_vision"
            self.vision_config["out_hidden_size"] = self.vision_config.pop(
                "text_hidden_size", self.vision_config.get("out_hidden_size", 2560)
            )
            self.vision_config.pop("temporal_patch_size", None)
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["llava_onevision1_5_vision"]()

        if isinstance(self.text_config, dict):
            self.text_config = dict(self.text_config)
            self.text_config["model_type"] = "qwen3"
            self.text_config = CONFIG_MAPPING["qwen3"](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        super().__post_init__(**kwargs)


class LlavaOnevision1_5VisionPatchEmbed(GlmImageVisionPatchEmbed):
    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        super().__init__(config)
        kernel_size = [self.patch_size, self.patch_size]
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )


class LlavaOnevision1_5VisionRotaryEmbedding(Qwen2_5_VLVisionRotaryEmbedding):
    pass


class LlavaOnevision1_5VisionPatchMerger(Qwen2_5_VLPatchMerger):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int, layer_norm_eps: float) -> None:
        super().__init__(dim, context_dim, spatial_merge_size)
        self.ln_q = nn.LayerNorm(context_dim, eps=layer_norm_eps)


class LlavaOnevision1_5VisionAttention(Qwen2_5_VLVisionAttention):
    pass


class LlavaOnevision1_5VisionMLP(SiglipMLP):
    pass


class LlavaOnevision1_5VisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        super().__init__(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = LlavaOnevision1_5VisionAttention(config)
        self.mlp = LlavaOnevision1_5VisionMLP(config)


@auto_docstring
class LlavaOnevision1_5PreTrainedModel(Qwen2_5_VLPreTrainedModel):
    config: LlavaOnevision1_5Config
    _no_split_modules = ["LlavaOnevision1_5VisionBlock"]


@auto_docstring
class LlavaOnevision1_5VisionModel(LlavaOnevision1_5PreTrainedModel, Qwen2_5_VisionTransformerPretrainedModel):
    config: LlavaOnevision1_5VisionConfig

    def __init__(self, config: LlavaOnevision1_5VisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.patch_embed = LlavaOnevision1_5VisionPatchEmbed(config)
        self.rotary_pos_emb = LlavaOnevision1_5VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([LlavaOnevision1_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = LlavaOnevision1_5VisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
        )

        head_dim = config.hidden_size // config.num_heads
        self.class_embedding = nn.Embedding(1, config.hidden_size)
        self.class_pos_emb = nn.Embedding(1, head_dim // 2)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        expected_patches = grid_thw.prod(-1).sum()
        torch_compilable_check(
            expected_patches == hidden_states.shape[0],
            f"Vision features and grid do not match, expected {expected_patches} patches but got "
            f"{hidden_states.shape[0]}",
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        num_segments = cu_seqlens.shape[0] - 1
        cls_indices = cu_seqlens.to(torch.long)[:-1] + torch.arange(num_segments, device=hidden_states.device)
        cls_mask = torch.zeros(hidden_states.shape[0] + num_segments, dtype=torch.bool, device=hidden_states.device)
        cls_mask[cls_indices] = True

        expanded_hidden_states = hidden_states.new_empty((cls_mask.shape[0], hidden_states.shape[-1]))
        expanded_hidden_states[cls_mask] = self.class_embedding.weight.to(hidden_states.dtype)
        expanded_hidden_states[~cls_mask] = hidden_states

        class_angles = torch.cat((self.class_pos_emb.weight, self.class_pos_emb.weight), dim=-1)
        expanded_position_embeddings = []
        for patch_embeddings, class_embeddings in zip(position_embeddings, (class_angles.cos(), class_angles.sin())):
            expanded_embeddings = patch_embeddings.new_empty((cls_mask.shape[0], patch_embeddings.shape[-1]))
            expanded_embeddings[cls_mask] = class_embeddings.to(patch_embeddings.dtype)
            expanded_embeddings[~cls_mask] = patch_embeddings
            expanded_position_embeddings.append(expanded_embeddings)

        hidden_states = self.pre_layernorm(expanded_hidden_states)
        position_embeddings = tuple(expanded_position_embeddings)
        cu_seqlens = cu_seqlens + torch.arange(cu_seqlens.shape[0], device=cu_seqlens.device)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen + 1 if max_seqlen is not None else None,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = hidden_states[~cls_mask]
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
        )


@auto_docstring
class LlavaOnevision1_5Model(Qwen2_5_VLModel):
    def __init__(self, config: LlavaOnevision1_5Config):
        super().__init__(config)
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def compute_3d_position_ids(self, **kwargs):
        return None


@auto_docstring
class LlavaOnevision1_5ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        model_inputs["position_ids"] = None
        return model_inputs

    def _prepare_position_ids_for_generation(self, **kwargs):
        raise AttributeError("LLaVA-OneVision-1.5 doesn't use 3D positions")


__all__ = [
    "LlavaOnevision1_5Config",
    "LlavaOnevision1_5ForConditionalGeneration",
    "LlavaOnevision1_5Model",
    "LlavaOnevision1_5PreTrainedModel",
    "LlavaOnevision1_5VisionModel",
    "LlavaOnevision1_5VisionConfig",
]
