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
from packaging import version
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..auto import AutoConfig, AutoModel
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLVisionRotaryEmbedding,
)
from ..qwen2_vl.modeling_qwen2_vl import VisionMlp


@auto_docstring(checkpoint="Deep-VLM/LLaVA-OneVision-1.5-4B-Instruct-hf")
@strict
class LlavaOnevision15VisionConfig(Qwen2_5_VLVisionConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Size of the projected visual features, matching the text backbone's hidden size.
    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
        Epsilon used by the vision encoder's layer normalization layers.
    initializer_factor (`float`, *optional*, defaults to 1.0):
        Scale factor for initializing the learned class token and its rotary position embedding.
    """

    model_type = "llava_onevision1_5_vision"

    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "gelu"
    intermediate_size: int = 4096
    num_heads: int = 16
    temporal_patch_size: int = 1
    out_hidden_size: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_factor: float = 1.0
    tokens_per_second = AttributeError()
    window_size = AttributeError()
    fullatt_block_indexes = AttributeError()


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
        "vision_config": SubConfigSpec(config_class=LlavaOnevision15VisionConfig),
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen3"),
    }

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = dict(self.vision_config)
            self.vision_config["model_type"] = LlavaOnevision15VisionConfig.model_type
            if "text_hidden_size" in self.vision_config:
                self.vision_config.setdefault("out_hidden_size", self.vision_config.pop("text_hidden_size"))
        if isinstance(self.text_config, dict) and self.text_config.get("model_type") == "LLaVAOneVision1_5_text":
            self.text_config = {**self.text_config, "model_type": "qwen3"}
        # Before v5 the nested text setting controlled tying; the serialized root setting could differ.
        if (
            self.transformers_version is not None
            and version.parse(self.transformers_version) < version.parse("5.0.0")
            and self.text_config is not None
        ):
            self.tie_word_embeddings = (
                self.text_config.get("tie_word_embeddings", False)
                if isinstance(self.text_config, dict)
                else self.text_config.tie_word_embeddings
            )
        PreTrainedConfig.__post_init__(self, **kwargs)
        self.tie_word_embeddings = self.tie_word_embeddings or self.text_config.tie_word_embeddings
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ValueError("vision_config.out_hidden_size must match text_config.hidden_size")


def get_vision_frame_lengths(cu_seqlens: torch.Tensor, kwargs: dict | None = None) -> list[int]:
    """Get patch counts per frame, or consume geometry prepared before graph capture."""
    if kwargs is not None and (frame_lengths := kwargs.pop("frame_lengths", None)) is not None:
        return frame_lengths
    return (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()


class LlavaOnevision15VisionPatchEmbed(nn.Module):
    def __init__(self, config: LlavaOnevision15VisionConfig):
        super().__init__()
        if config.temporal_patch_size != 1:
            raise ValueError("LLaVA-OneVision-1.5 requires vision_config.temporal_patch_size=1")
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.proj = nn.Conv2d(
            self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class LlavaOnevision15VisionPatchMerger(Qwen2_5_VLPatchMerger):
    def __init__(self, config: LlavaOnevision15VisionConfig):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.ln_q(hidden_states).reshape(-1, self.hidden_size)
        return self.fc2(self.act(self.fc1(hidden_states)))


class LlavaOnevision15VisionMlp(VisionMlp):
    def __init__(self, config: LlavaOnevision15VisionConfig):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)


class LlavaOnevision15VisionRotaryEmbedding(Qwen2_5_VLVisionRotaryEmbedding):
    pass


class LlavaOnevision15VisionAttention(Qwen2_5_VLVisionAttention):
    pass


class LlavaOnevision15VisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config: LlavaOnevision15VisionConfig):
        GradientCheckpointingLayer.__init__(self)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = LlavaOnevision15VisionAttention(config)
        self.mlp = LlavaOnevision15VisionMlp(config)

    def forward(self, hidden_states, cu_seqlens, cos, sin, **kwargs):
        r"""
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence lengths, including one class token for each image or frame.
        cos (`torch.Tensor`):
            Cosines of rotary positions, including the learned class position.
        sin (`torch.Tensor`):
            Sines of rotary positions, including the learned class position.
        """
        # Learned class positions must be explicit checkpoint inputs, including with reentrant checkpointing.
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=(cos, sin), **kwargs
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


@auto_docstring
class LlavaOnevision15PreTrainedModel(PreTrainedModel):
    config: LlavaOnevision15Config
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaOnevision15VisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LlavaOnevision15VisionModel):
            std = module.config.hidden_size**-0.5 * module.config.initializer_factor
            init.normal_(module.class_embedding, mean=0.0, std=std)
            init.normal_(module.class_pos_emb, mean=0.0, std=std)


@auto_docstring
class LlavaOnevision15VisionModel(LlavaOnevision15PreTrainedModel):
    config: LlavaOnevision15VisionConfig
    base_model_prefix = "visual"
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
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> BaseModelOutputWithPooling:
        r"""
        hidden_states (`torch.Tensor` of shape `(num_patches, num_channels * patch_size * patch_size)`):
            Flattened image or video patches in spatial merge order.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            Temporal, height and width dimensions of the patch grid for each image or video.
        """
        hidden_states = self.patch_embed(hidden_states)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cos, sin = self.rotary_pos_emb(hidden_states, position_ids)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        frame_lengths = get_vision_frame_lengths(cu_seqlens, kwargs=kwargs)
        cu_seqlens = cu_seqlens + torch.arange(cu_seqlens.shape[0], device=cu_seqlens.device, dtype=cu_seqlens.dtype)
        if max_seqlen is not None:
            max_seqlen += 1
        cls_hidden = self.class_embedding.to(hidden_states.dtype).unsqueeze(0)
        cls_freqs = torch.cat((self.class_pos_emb, self.class_pos_emb), dim=-1).to(cos.dtype)
        cls_cos, cls_sin = cls_freqs.cos().to(cos.dtype), cls_freqs.sin().to(sin.dtype)
        hidden_states, cos, sin = [
            torch.cat([torch.cat((cls, frame), dim=0) for frame in torch.split(value, frame_lengths)], dim=0)
            for value, cls in ((hidden_states, cls_hidden), (cos, cls_cos), (sin, cls_sin))
        ]
        hidden_states = self.pre_layernorm(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, cos, sin, max_seqlen=max_seqlen, **kwargs)

        hidden_states = torch.cat(
            [frame[1:] for frame in torch.split(hidden_states, [length + 1 for length in frame_lengths])], dim=0
        )
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=self.merger(hidden_states))


@auto_docstring
class LlavaOnevision15Model(Qwen2_5_VLModel):
    _no_split_modules = ["LlavaOnevision15VisionBlock"]

    def __init__(self, config: LlavaOnevision15Config):
        PreTrainedModel.__init__(self, config)
        self.visual = LlavaOnevision15VisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def compute_3d_position_ids(self, **kwargs) -> None:
        # The Qwen3 backbone computes ordinary 1D positions from its sequence length and cache.
        return None

    def get_rope_index(self, *args, **kwargs):
        raise AttributeError("The Qwen3 backbone uses 1D position IDs.")

    def get_vision_position_ids(self, *args, **kwargs):
        raise AttributeError("The Qwen3 backbone uses 1D position IDs.")


@auto_docstring
class LlavaOnevision15ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def forward(self, **super_kwargs):
        r"""
        second_per_grid_ts (`torch.Tensor`, *optional*):
            Accepted for processor compatibility. The Qwen3 backbone uses 1D positions independently of timestamps.

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
        return GenerationMixin._prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs)


__all__ = [
    "LlavaOnevision15Config",
    "LlavaOnevision15VisionConfig",
    "LlavaOnevision15PreTrainedModel",
    "LlavaOnevision15VisionModel",
    "LlavaOnevision15Model",
    "LlavaOnevision15ForConditionalGeneration",
]
