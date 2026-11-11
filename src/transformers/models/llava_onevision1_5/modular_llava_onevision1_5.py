# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..auto import AutoConfig, AutoModel
from ..exaone4_5.modeling_exaone4_5 import Exaone4_5_Model
from ..glm_image.modeling_glm_image import GlmImageVisionBlock, GlmImageVisionMLP
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionRotaryEmbedding,
)
from ..qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger


@auto_docstring(checkpoint="Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf")
@strict
class LlavaOnevision15VisionConfig(Qwen2_5_VLVisionConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Size of the projected visual features, matching the text backbone's hidden size.
    """

    model_type = "llava_onevision1_5_vision"
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "depth",
        "text_hidden_size": "out_hidden_size",
    }

    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "gelu"
    intermediate_size: int = 4096
    temporal_patch_size: int = 1
    out_hidden_size: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_factor: float = 1.0
    tokens_per_second = AttributeError()
    window_size = AttributeError()
    fullatt_block_indexes = AttributeError()

    def validate_architecture(self):
        if self.temporal_patch_size != 1:
            raise ValueError("LLaVA-OneVision-1.5 requires vision_config.temporal_patch_size=1")


@auto_docstring(checkpoint="Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf")
@strict
class LlavaOnevision15Config(Qwen2_5_VLConfig):
    r"""
    Example:

    ```python
    >>> from transformers import LlavaOnevision15Config, LlavaOnevision15ForConditionalGeneration

    >>> config = LlavaOnevision15Config()
    >>> model = LlavaOnevision15ForConditionalGeneration(config)
    ```
    """

    model_type = "llava_onevision1_5"
    sub_configs_defaults = {
        "vision_config": LlavaOnevision15VisionConfig,
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen3"),
    }

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict) and self.vision_config.get("model_type") in (
            "rice_vit",
            "llava_onevision1_5",
        ):
            # Published checkpoints predate the separate vision model type.
            self.vision_config = {**self.vision_config, "model_type": LlavaOnevision15VisionConfig.model_type}
        if isinstance(self.text_config, dict) and self.text_config.get("model_type") == "LLaVAOneVision1_5_text":
            self.text_config = {**self.text_config, "model_type": "qwen3"}
        PreTrainedConfig.__post_init__(self, **kwargs)
        # Published checkpoints use the text backbone's weight-tying setting.
        self.tie_word_embeddings = self.text_config.tie_word_embeddings

    def validate_architecture(self):
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError("vision_config.out_hidden_size must match text_config.hidden_size")


def get_vision_frame_lengths(grid_thw: torch.Tensor, kwargs: dict | None = None) -> list[int]:
    """Get patch counts per frame, or consume geometry prepared before graph capture."""
    if kwargs is not None and (frame_lengths := kwargs.pop("frame_lengths", None)) is not None:
        return frame_lengths
    return torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()


class LlavaOnevision15VisionPatchEmbed(nn.Module):
    def __init__(self, config: LlavaOnevision15VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels * config.temporal_patch_size
        self.embed_dim = config.hidden_size
        self.proj = nn.Conv2d(
            self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class LlavaOnevision15VisionPatchMerger(Qwen3VLVisionPatchMerger):
    def __init__(self, config: LlavaOnevision15VisionConfig):
        super().__init__(config, use_postshuffle_norm=False)
        self.use_postshuffle_norm = False
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class LlavaOnevision15VisionMLP(GlmImageVisionMLP):
    pass


class LlavaOnevision15VisionRotaryEmbedding(Qwen2_5_VLVisionRotaryEmbedding):
    pass


class LlavaOnevision15VisionAttention(Qwen2_5_VLVisionAttention):
    pass


class LlavaOnevision15VisionBlock(GlmImageVisionBlock):
    def forward(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
        kwargs["position_embeddings"] = position_embeddings
        return super().forward(hidden_states, cu_seqlens, **kwargs)


@auto_docstring
class LlavaOnevision15PreTrainedModel(Qwen2_5_VLPreTrainedModel):
    _no_split_modules = ["LlavaOnevision15VisionBlock"]
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, LlavaOnevision15VisionModel):
            std = module.config.hidden_size**-0.5 * module.config.initializer_factor
            init.normal_(module.class_embedding, mean=0.0, std=std)
            init.normal_(module.class_pos_emb, mean=0.0, std=std)


@auto_docstring
class LlavaOnevision15VisionModel(LlavaOnevision15PreTrainedModel):
    config: LlavaOnevision15VisionConfig
    input_modalities = ("image", "video")
    _input_embed_layer = "patch_embed"
    _can_record_outputs = {
        "hidden_states": LlavaOnevision15VisionBlock,
        "attentions": LlavaOnevision15VisionAttention,
    }

    def __init__(self, config: LlavaOnevision15VisionConfig):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = LlavaOnevision15VisionPatchEmbed(config)
        self.rotary_pos_emb = LlavaOnevision15VisionRotaryEmbedding(config)
        self.class_embedding = nn.Parameter(torch.empty(config.hidden_size))
        self.class_pos_emb = nn.Parameter(torch.empty(1, config.hidden_size // config.num_heads // 2))
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.blocks = nn.ModuleList([LlavaOnevision15VisionBlock(config) for _ in range(config.depth)])
        self.merger = LlavaOnevision15VisionPatchMerger(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_patches, num_channels * patch_size * patch_size)`):
            Flattened image or video patches in spatial merge order.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            Temporal, height and width dimensions of the patch grid for each image or video.
        """
        hidden_states = self.patch_embed(pixel_values)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cos, sin = self.rotary_pos_emb(hidden_states, position_ids)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        frame_lengths = get_vision_frame_lengths(grid_thw, kwargs=kwargs)
        cu_seqlens = cu_seqlens + torch.arange(cu_seqlens.shape[0], device=cu_seqlens.device, dtype=cu_seqlens.dtype)
        if max_seqlen is not None:
            max_seqlen += 1
        cls_freqs = torch.cat((self.class_pos_emb, self.class_pos_emb), dim=-1).to(cos.dtype)
        hidden_states = self.prepend_cls(hidden_states, self.class_embedding[None, :], frame_lengths)
        cos = self.prepend_cls(cos, cls_freqs.cos(), frame_lengths)
        sin = self.prepend_cls(sin, cls_freqs.sin(), frame_lengths)
        position_embeddings = (cos, sin)
        if self.training and self.is_activation_checkpointing:
            # Reentrant checkpointing only tracks tensors passed as positional inputs, not nested tuples.
            position_embeddings = torch.stack(position_embeddings)
        hidden_states = self.pre_layernorm(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings, max_seqlen=max_seqlen, **kwargs)

        hidden_states = torch.cat(
            [frame[1:] for frame in torch.split(hidden_states, [length + 1 for length in frame_lengths])], dim=0
        )
        pooled_states = self.merger(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooled_states)

    @staticmethod
    def prepend_cls(hidden_states: torch.Tensor, cls_states: torch.Tensor, frame_lengths: list[int]) -> torch.Tensor:
        """Prepend a class token, cast to the input dtype, to each frame's patches."""
        cls_states = cls_states.to(hidden_states.dtype)
        return torch.cat([torch.cat((cls_states, frame)) for frame in hidden_states.split(frame_lengths)])


@auto_docstring
class LlavaOnevision15Model(LlavaOnevision15PreTrainedModel, Exaone4_5_Model):
    def __init__(self, config: LlavaOnevision15Config):
        PreTrainedModel.__init__(self, config)
        self.visual = LlavaOnevision15VisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()


@auto_docstring
class LlavaOnevision15ForConditionalGeneration(Qwen2VLForConditionalGeneration):
    def forward(self, second_per_grid_ts: torch.Tensor | None = None, **super_kwargs):
        r"""
        second_per_grid_ts (`torch.Tensor`, *optional*):
            Video frame intervals supplied by the processor. Text positions remain one-dimensional.

        Example:

        ```python
        >>> from transformers import AutoProcessor, LlavaOnevision15ForConditionalGeneration

        >>> model_id = "Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = LlavaOnevision15ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        >>> messages = [{"role": "user", "content": [
        ...     {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...     {"type": "text", "text": "Describe the image."},
        ... ]}]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)
        >>> output = model.generate(**inputs, max_new_tokens=64)
        >>> processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ```
        """
        return super().forward(**super_kwargs)

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        raise AttributeError("LLaVA-OneVision-1.5 uses the default 1D generation positions.")


__all__ = [
    "LlavaOnevision15Config",
    "LlavaOnevision15VisionConfig",
    "LlavaOnevision15PreTrainedModel",
    "LlavaOnevision15VisionModel",
    "LlavaOnevision15Model",
    "LlavaOnevision15ForConditionalGeneration",
]
