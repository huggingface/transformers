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
"""PyTorch Ovis2.5 model."""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_utils import PILImageResampling
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from ...utils.generic import accepts_precomputed_kwargs, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ...vision_utils import (
    get_vision_attention_seqlens,
    get_vision_position_ids,
    get_vision_window_index,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs
from ..qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding
from ..video_llama_3.modeling_video_llama_3 import VideoLlama3VisionAttention, VideoLlama3VisionMLP


class Ovis2_5ImageProcessorKwargs(Qwen2VLImageProcessorKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `448 * 448`):
        The minimum number of pixels in the resized image.
    max_pixels (`int`, *optional*, defaults to `1344 * 1792`):
        The maximum number of pixels in the resized image.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The spatial merge size used by the visual tokenizer.
    """


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 448 * 448,
    max_pixels: int = 1344 * 1792,
) -> tuple[int, int]:
    """Resize an image according to the native Ovis2.5 preprocessing policy."""
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


class Ovis2_5ImageProcessor(Qwen2VLImageProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = Ovis2_5ImageProcessorKwargs


class Ovis2_5ImageProcessorPil(Qwen2VLImageProcessorPil):
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = Ovis2_5ImageProcessorKwargs


@auto_docstring(checkpoint="AIDC-AI/Ovis2.5-2B")
@strict
class Ovis2_5VisionConfig(PreTrainedConfig):
    r"""
    num_patches (`int`, *optional*, defaults to -1):
        Number of patches used by the original fixed-resolution position table. A negative value selects convolutional
        patch embedding, which is the layout used by the released Ovis2.5 checkpoints.
    hidden_stride (`int`, *optional*, defaults to 2):
        Spatial grouping factor applied before the visual-tokenizer head.
    window_size (`int`, *optional*, defaults to 112):
        Window size, in input pixels, used by windowed vision-attention layers.
    fullatt_block_indexes (`tuple[int, ...]`, *optional*):
        Indices of vision layers that use full attention. `None` makes every layer use full attention.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        Number of consecutive video frames represented by one temporal patch.
    preserve_original_pe (`bool`, *optional*, defaults to `True`):
        Whether to interpolate and add the learned 32 by 32 positional embedding.
    use_rope (`bool`, *optional*, defaults to `True`):
        Whether to apply two-dimensional rotary position embeddings in vision attention.
    vocab_size (`int`, *optional*, defaults to 65536):
        Size of the visual-token vocabulary, including four learned visual-boundary indicators.
    num_visual_indicator_tokens (`int`, *optional*, defaults to 4):
        Number of visual-vocabulary rows reserved for image/video boundary indicators.
    """

    model_type = "ovis2_5_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    num_patches: int = -1
    image_size: int = 512
    patch_size: int = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    hidden_stride: int = 2
    window_size: int = 112
    fullatt_block_indexes: tuple[int, ...] | list[int] | str | None = None
    temporal_patch_size: int = 1
    preserve_original_pe: bool = True
    use_rope: bool = True
    vocab_size: int = 65536
    num_visual_indicator_tokens: int = 4
    initializer_range: float = 0.02

    # Ignore copy
    def __post_init__(self, **kwargs):
        if isinstance(self.fullatt_block_indexes, str):
            self.fullatt_block_indexes = tuple(
                int(layer_index) for layer_index in self.fullatt_block_indexes.split("|") if layer_index
            )
        elif isinstance(self.fullatt_block_indexes, list):
            self.fullatt_block_indexes = tuple(self.fullatt_block_indexes)

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="AIDC-AI/Ovis2.5-2B")
@strict
class Ovis2_5Config(PreTrainedConfig):
    r"""
    visual_vocab_size (`int`, *optional*, defaults to 65536):
        Size of the visual-token vocabulary shared by the visual tokenizer and visual embedding table.
    visual_atom_token_id (`int`, *optional*, defaults to 151669):
        Text-vocabulary token used as the placeholder for one visual atom.
    image_start_token_id (`int`, *optional*, defaults to 151670):
        Text-vocabulary token replaced by the learned image-begin visual embedding.
    image_end_token_id (`int`, *optional*, defaults to 151671):
        Text-vocabulary token replaced by the learned image-end visual embedding.
    video_start_token_id (`int`, *optional*, defaults to 151672):
        Text-vocabulary token replaced by the learned video-begin visual embedding.
    video_end_token_id (`int`, *optional*, defaults to 151673):
        Text-vocabulary token replaced by the learned video-end visual embedding.
    """

    model_type = "ovis2_5"
    sub_configs = {"vision_config": Ovis2_5VisionConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    visual_vocab_size: int = 65536
    visual_atom_token_id: int = 151669
    image_start_token_id: int = 151670
    image_end_token_id: int = 151671
    video_start_token_id: int = 151672
    video_end_token_id: int = 151673
    image_token_id: int = 151669
    video_token_id: int = 151669
    tie_word_embeddings: bool = False

    # Ignore copy
    def __post_init__(self, **kwargs):
        # The original remote-code checkpoints use `llm_config` and `vit_config`. Normalize those aliases before
        # constructing native Transformers sub-configs so the public 2B and 9B configs load without editing.
        legacy_text_config = kwargs.pop("llm_config", None)
        legacy_vision_config = kwargs.pop("vit_config", None)
        if self.text_config is None:
            self.text_config = legacy_text_config
        if self.vision_config is None:
            self.vision_config = legacy_vision_config

        if isinstance(self.vision_config, dict):
            vision_config = dict(self.vision_config)
            vision_config.pop("model_type", None)
            self.vision_config = Ovis2_5VisionConfig(**vision_config)
        elif self.vision_config is None:
            self.vision_config = Ovis2_5VisionConfig()

        if isinstance(self.text_config, dict):
            text_config = dict(self.text_config)
            text_config.pop("model_type", None)
            self.text_config = CONFIG_MAPPING["qwen3"](**text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        self.vision_config.vocab_size = self.visual_vocab_size
        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = self.text_config.tie_word_embeddings
        # These aliases let generic multimodal utilities identify the repeated
        # placeholder token while preserving Ovis2.5's distinct boundary IDs.
        self.image_token_id = self.visual_atom_token_id
        self.video_token_id = self.visual_atom_token_id
        super().__post_init__(**kwargs)

    @property
    def llm_config(self) -> PreTrainedConfig:
        """Alias used by the original Ovis2.5 remote-code configuration."""
        return self.text_config

    @property
    def vit_config(self) -> Ovis2_5VisionConfig:
        """Alias used by the original Ovis2.5 remote-code configuration."""
        return self.vision_config


class Ovis2_5VisionRotaryEmbedding(VisionRotaryEmbedding):
    pass


class Ovis2_5VisionEmbeddings(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        if config.num_patches > 0:
            self.patch_embedding = nn.Linear(
                config.num_channels * config.patch_size**2,
                config.hidden_size,
            )
            if config.preserve_original_pe:
                self.position_embedding_size = int(config.num_patches**0.5)
                self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)
        else:
            self.patch_embedding = nn.Conv2d(
                in_channels=config.num_channels,
                out_channels=config.hidden_size,
                kernel_size=config.patch_size,
                stride=config.patch_size,
                padding="valid",
                bias=True,
            )
            if config.preserve_original_pe:
                self.position_embedding_size = config.image_size // config.patch_size
                self.position_embedding = nn.Embedding(self.position_embedding_size**2, config.hidden_size)

    def forward(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor) -> torch.Tensor:
        grid_values = grid_thw.tolist()
        target_dtype = self.patch_embedding.weight.dtype
        if isinstance(self.patch_embedding, nn.Linear):
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        else:
            pixel_values = pixel_values.view(
                -1,
                self.config.num_channels * self.config.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype)).reshape(-1, self.embed_dim)

        if not self.config.preserve_original_pe:
            return patch_embeds

        position_embeddings = self.position_embedding.weight.reshape(
            1,
            self.position_embedding_size,
            self.position_embedding_size,
            self.embed_dim,
        ).permute(0, 3, 1, 2)
        interpolated_positions = torch.zeros_like(patch_embeds)
        offset = 0
        hidden_stride = self.config.hidden_stride
        for grid_t, grid_h, grid_w in grid_values:
            num_tokens = grid_t * grid_h * grid_w
            position_embedding = nn.functional.interpolate(
                position_embeddings,
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False,
            )
            position_embedding = position_embedding.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
            position_embedding = position_embedding[0].repeat(grid_t, 1)
            position_embedding = position_embedding.reshape(
                grid_t,
                grid_h // hidden_stride,
                hidden_stride,
                grid_w // hidden_stride,
                hidden_stride,
                self.embed_dim,
            )
            position_embedding = position_embedding.permute(0, 1, 3, 2, 4, 5).reshape(num_tokens, -1)
            interpolated_positions[offset : offset + num_tokens] = position_embedding
            offset += num_tokens

        return patch_embeds + interpolated_positions


class Ovis2_5VisionMLP(VideoLlama3VisionMLP):
    pass


class Ovis2_5VisionAttention(VideoLlama3VisionAttention):
    pass


class Ovis2_5VisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Ovis2_5VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Ovis2_5VisionMLP(config)

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence boundaries for packed variable-length attention.
        max_seqlen (`int`, *optional*):
            Maximum packed sequence length, used by Flash Attention kernels.
        """
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class Ovis2_5VisionHiddenStateRecorder(nn.Module):
    """Restore encoder states from window order before output hooks record them."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        reverse_indices: torch.LongTensor,
        spatial_merge_unit: int,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        return hidden_states[reverse_indices].reshape(sequence_length, -1)


class Ovis2_5VisionEncoder(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Ovis2_5VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.hidden_state_recorder = Ovis2_5VisionHiddenStateRecorder()

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_hidden_states: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        spatial_merge_size = self.config.hidden_stride
        spatial_merge_unit = spatial_merge_size**2
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            spatial_merge_size=spatial_merge_size,
            window_size=self.config.window_size,
            patch_size=self.config.patch_size,
            kwargs=kwargs,
        )
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        sequence_length = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index].reshape(sequence_length, -1)
        rotary_cos, rotary_sin = position_embeddings
        rotary_cos = rotary_cos.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        rotary_sin = rotary_sin.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        position_embeddings = (
            rotary_cos[window_index].reshape(sequence_length, -1),
            rotary_sin[window_index].reshape(sequence_length, -1),
        )
        if not self.config.use_rope:
            position_embeddings = (
                torch.ones_like(position_embeddings[0]),
                torch.zeros_like(position_embeddings[1]),
            )
        reverse_indices = torch.argsort(window_index)

        recorded_hidden_states = None
        for layer_index, encoder_layer in enumerate(self.layers):
            use_full_attention = (
                self.config.fullatt_block_indexes is None or layer_index in self.config.fullatt_block_indexes
            )
            layer_cu_seqlens = cu_seqlens if use_full_attention else cu_window_seqlens
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=layer_cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen if use_full_attention else None,
                **kwargs,
            )
            if output_hidden_states:
                recorded_hidden_states = self.hidden_state_recorder(hidden_states, reverse_indices, spatial_merge_unit)
                if layer_index + 1 < len(self.layers):
                    hidden_states = recorded_hidden_states.reshape(
                        sequence_length // spatial_merge_unit, spatial_merge_unit, -1
                    )
                    hidden_states = hidden_states[window_index].reshape(sequence_length, -1)

        if recorded_hidden_states is not None:
            hidden_states = recorded_hidden_states
        else:
            hidden_states = hidden_states.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
            hidden_states = hidden_states[reverse_indices].reshape(sequence_length, -1)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Ovis2_5VisionTransformer(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Ovis2_5VisionEmbeddings(config)
        self.encoder = Ovis2_5VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = Ovis2_5VisionRotaryEmbedding(head_dim // 2)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        output_hidden_states: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(pixel_values, grid_thw)
        position_ids = get_vision_position_ids(grid_thw, self.config.hidden_stride, kwargs=kwargs)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary_pos_emb.cos(), rotary_pos_emb.sin())

        encoder_outputs = self.encoder(
            hidden_states,
            grid_thw=grid_thw,
            position_embeddings=position_embeddings,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        pre_layernorm_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(pre_layernorm_hidden_state)
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pre_layernorm_hidden_state,
        )


class Ovis2_5VisualEmbeddingTable(nn.Embedding):
    """Embedding table accepting either discrete visual ids or soft visual-token distributions."""

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.is_floating_point():
            return torch.matmul(visual_tokens, self.weight)
        return super().forward(visual_tokens)


@auto_docstring
@dataclass
class Ovis2_5VisualFeaturesOutput(BaseModelOutputWithPooling):
    r"""
    visual_indicator_features (`torch.FloatTensor` of shape `(4, hidden_size)`, *optional*):
        Learned image-begin, image-end, video-begin, and video-end embeddings, in that order.
    """

    visual_indicator_features: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class Ovis2_5ModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_hidden_states: torch.FloatTensor | None = None
    video_hidden_states: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class Ovis2_5CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor`, *optional*):
        Causal language modeling loss.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`, *optional*):
        Language modeling logits.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_hidden_states: torch.FloatTensor | None = None
    video_hidden_states: torch.FloatTensor | None = None


@auto_docstring
class Ovis2_5PreTrainedModel(PreTrainedModel):
    config: Ovis2_5Config
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Ovis2_5VisionEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Ovis2_5VisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


@auto_docstring(
    custom_intro="The Ovis2.5 vision tower and visual tokenizer, without the visual embedding table or language model."
)
class Ovis2_5VisionModel(Ovis2_5PreTrainedModel):
    config: Ovis2_5VisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image", "video")
    _can_record_outputs = {
        "hidden_states": OutputRecorder(
            Ovis2_5VisionHiddenStateRecorder,
            capture_initial_hidden_state=False,
        ),
        "attentions": Ovis2_5VisionAttention,
    }

    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__(config)
        self.transformer = Ovis2_5VisionTransformer(config)
        visual_token_vocab_size = config.vocab_size - config.num_visual_indicator_tokens
        self.head_linear = nn.Linear(
            config.hidden_size * config.hidden_stride**2,
            visual_token_vocab_size,
            bias=False,
        )
        self.head_norm = nn.LayerNorm(visual_token_vocab_size)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.embeddings.patch_embedding

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            Temporal, height, and width patch-grid dimensions for each packed image or video.
        """
        transformer_outputs = self.transformer(
            pixel_values,
            grid_thw=grid_thw,
            output_hidden_states=(
                self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
            ),
            **kwargs,
        )
        # The released visual-tokenizer head consumes the final encoder block before `post_layernorm`.
        hidden_states = transformer_outputs.pooler_output
        spatial_merge_unit = self.config.hidden_stride**2
        hidden_states = hidden_states.reshape(hidden_states.shape[0] // spatial_merge_unit, -1)
        logits = self.head_norm(self.head_linear(hidden_states))
        visual_tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        indicator_padding = torch.zeros(
            (visual_tokens.shape[0], self.config.num_visual_indicator_tokens),
            dtype=visual_tokens.dtype,
            device=visual_tokens.device,
        )
        visual_tokens = torch.cat((visual_tokens, indicator_padding), dim=-1)

        return BaseModelOutputWithPooling(
            last_hidden_state=transformer_outputs.last_hidden_state,
            pooler_output=visual_tokens,
        )


@auto_docstring(custom_intro="The bare Ovis2.5 multimodal model, without the language modeling head.")
class Ovis2_5Model(Ovis2_5PreTrainedModel):
    def __init__(self, config: Ovis2_5Config):
        super().__init__(config)
        vision_config = cast(Ovis2_5VisionConfig, config.vision_config)
        # Ovis2_5Config.__post_init__ constructs the Qwen3 sub-config.
        text_config = cast(Any, config.text_config)
        self.vision_tower = Ovis2_5VisionModel(vision_config)
        self.visual_embeddings_table = Ovis2_5VisualEmbeddingTable(
            config.visual_vocab_size,
            text_config.hidden_size,
        )
        self.language_model = AutoModel.from_config(text_config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.language_model.set_input_embeddings(value)

    def _get_token_mask(
        self,
        token_id: int,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
    ) -> torch.BoolTensor:
        if input_ids is not None:
            return input_ids == token_id
        token_embedding = self.get_input_embeddings()(
            torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        return (inputs_embeds == token_embedding).all(dim=-1)

    def _get_visual_features(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Ovis2_5VisualFeaturesOutput:
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            return_dict=True,
            **kwargs,
        )
        visual_features = self.visual_embeddings_table(vision_outputs.pooler_output)
        indicator_start = self.config.visual_vocab_size - self.vision_tower.config.num_visual_indicator_tokens
        indicator_token_ids = torch.arange(
            indicator_start,
            self.config.visual_vocab_size,
            dtype=torch.long,
            device=visual_features.device,
        )
        visual_indicator_features = self.visual_embeddings_table(indicator_token_ids)
        return Ovis2_5VisualFeaturesOutput(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=visual_features,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            visual_indicator_features=visual_indicator_features,
        )

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring(custom_intro="Encodes images into Ovis2.5 visual embeddings.")
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Ovis2_5VisualFeaturesOutput:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height, and width patch-grid dimensions for each packed image.
        """
        outputs = self._get_visual_features(pixel_values, image_grid_thw, **kwargs)
        split_sizes = (image_grid_thw.prod(dim=1) // self.vision_tower.config.hidden_stride**2).tolist()
        outputs.pooler_output = torch.split(outputs.pooler_output, split_sizes)
        return outputs

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring(custom_intro="Encodes videos into Ovis2.5 visual embeddings.")
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Ovis2_5VisualFeaturesOutput:
        r"""
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            Temporal, height, and width patch-grid dimensions for each packed video.
        """
        outputs = self._get_visual_features(pixel_values_videos, video_grid_thw, **kwargs)
        split_sizes = (video_grid_thw.prod(dim=1) // self.vision_tower.config.hidden_stride**2).tolist()
        outputs.pooler_output = torch.split(outputs.pooler_output, split_sizes)
        return outputs

    def _merge_visual_features(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor | None,
        visual_features: torch.FloatTensor,
        visual_indicator_features: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        is_video: bool,
    ) -> torch.FloatTensor:
        atom_mask = self._get_token_mask(self.config.visual_atom_token_id, input_ids, inputs_embeds)
        torch_compilable_check(
            atom_mask.sum() * inputs_embeds.shape[-1] == visual_features.numel(),
            lambda: (
                f"Visual features and visual atom tokens do not match: found {atom_mask.sum().item()} tokens and "
                f"{visual_features.shape[0]} features."
            ),
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            atom_mask.unsqueeze(-1).to(inputs_embeds.device),
            visual_features.to(inputs_embeds.device, inputs_embeds.dtype),
        )

        if is_video:
            boundary_token_ids = (self.config.video_start_token_id, self.config.video_end_token_id)
            indicator_indexes = (2, 3)
        else:
            boundary_token_ids = (self.config.image_start_token_id, self.config.image_end_token_id)
            indicator_indexes = (0, 1)

        for boundary_token_id, indicator_index in zip(boundary_token_ids, indicator_indexes):
            boundary_mask = self._get_token_mask(boundary_token_id, input_ids, inputs_embeds)
            torch_compilable_check(
                boundary_mask.sum() == grid_thw.shape[0],
                lambda: (
                    f"Expected {grid_thw.shape[0]} visual boundary tokens with id {boundary_token_id}, but found "
                    f"{boundary_mask.sum().item()}."
                ),
            )
            boundary_features = visual_indicator_features[indicator_index].to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            inputs_embeds = torch.where(
                boundary_mask.unsqueeze(-1),
                boundary_features.expand_as(inputs_embeds),
                inputs_embeds,
            )
        return inputs_embeds

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
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Ovis2_5ModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width patch-grid dimensions for each packed image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width patch-grid dimensions for each packed video.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("Ovis2.5 supports images or video in one request, but not both.")
        if pixel_values is not None and image_grid_thw is None:
            raise ValueError("`image_grid_thw` is required when `pixel_values` is provided.")
        if pixel_values_videos is not None and video_grid_thw is None:
            raise ValueError("`video_grid_thw` is required when `pixel_values_videos` is provided.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_hidden_states = None
        video_hidden_states = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
                **kwargs,
            )
            image_hidden_states = torch.cat(image_outputs.pooler_output, dim=0)
            inputs_embeds = self._merge_visual_features(
                inputs_embeds,
                input_ids,
                image_hidden_states,
                image_outputs.visual_indicator_features,
                image_grid_thw,
                is_video=False,
            )
        elif pixel_values_videos is not None:
            video_outputs = self.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_dict=True,
                **kwargs,
            )
            video_hidden_states = torch.cat(video_outputs.pooler_output, dim=0)
            inputs_embeds = self._merge_visual_features(
                inputs_embeds,
                input_ids,
                video_hidden_states,
                video_outputs.visual_indicator_features,
                video_grid_thw,
                is_video=True,
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        return Ovis2_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
            video_hidden_states=video_hidden_states,
        )


@auto_docstring(custom_intro="The Ovis2.5 multimodal model with a language modeling head.")
class Ovis2_5ForConditionalGeneration(Ovis2_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Ovis2_5Config):
        super().__init__(config)
        self.model = Ovis2_5Model(config)
        # Ovis2_5Config.__post_init__ constructs the Qwen3 sub-config.
        text_config = cast(Any, config.text_config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Ovis2_5VisualFeaturesOutput:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height, and width patch-grid dimensions for each packed image.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Ovis2_5VisualFeaturesOutput:
        r"""
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            Temporal, height, and width patch-grid dimensions for each packed video.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

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
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Ovis2_5CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the causal language modeling loss. Token indices set to `-100` are ignored.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height, and width patch-grid dimensions for each packed image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Temporal, height, and width patch-grid dimensions for the packed video.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.lm_head.out_features,
                **kwargs,
            )

        return Ovis2_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Generation creates placeholder `input_ids` when the caller supplies
        # only `inputs_embeds`. Prefer the real embeddings whenever they are
        # available so beam expansion still sees the multimodal boundaries.
        if inputs_embeds is not None:
            image_start_embedding = self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_start_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            video_start_embedding = self.get_input_embeddings()(
                torch.tensor(
                    self.config.video_start_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            image_start_mask = (inputs_embeds == image_start_embedding).all(dim=-1)
            video_start_mask = (inputs_embeds == video_start_embedding).all(dim=-1)
        elif input_ids is not None:
            image_start_mask = input_ids == self.config.image_start_token_id
            video_start_mask = input_ids == self.config.video_start_token_id
        else:
            raise ValueError("Either `input_ids` or `inputs_embeds` is required to expand multimodal inputs.")
        return image_start_mask.sum(dim=1), video_start_mask.sum(dim=1)

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs: Any,
    ) -> tuple[torch.LongTensor | None, dict[str, Any]]:
        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        image_grid_thw = model_kwargs.get("image_grid_thw")
        video_grid_thw = model_kwargs.get("video_grid_thw")
        image_nums, video_nums = self._get_image_nums_and_video_nums(
            input_ids,
            inputs_embeds=model_kwargs.get("inputs_embeds"),
        )

        def repeat_packed_samples(tensor, lengths):
            samples = torch.split(tensor, [int(length) for length in lengths])
            repeats = [expand_size] + [1] * (tensor.ndim - 1)
            return torch.cat([sample.repeat(*repeats) for sample in samples], dim=0)

        if model_kwargs.get("pixel_values") is not None:
            grid_samples = torch.split(image_grid_thw, [int(length) for length in image_nums])
            patch_lengths = [sample.prod(dim=1).sum().item() for sample in grid_samples]
            model_kwargs["pixel_values"] = repeat_packed_samples(model_kwargs["pixel_values"], patch_lengths)
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = repeat_packed_samples(image_grid_thw, image_nums)
        if model_kwargs.get("pixel_values_videos") is not None:
            grid_samples = torch.split(video_grid_thw, [int(length) for length in video_nums])
            patch_lengths = [sample.prod(dim=1).sum().item() for sample in grid_samples]
            model_kwargs["pixel_values_videos"] = repeat_packed_samples(
                model_kwargs["pixel_values_videos"], patch_lengths
            )
        if video_grid_thw is not None:
            model_kwargs["video_grid_thw"] = repeat_packed_samples(video_grid_thw, video_nums)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        for key, value in model_kwargs.items():
            if value is not None and isinstance(value, torch.Tensor) and key not in visual_keys:
                model_kwargs[key] = value.repeat_interleave(expand_size, dim=0)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("`encoder_outputs` must be provided for an encoder-decoder model.")
            model_kwargs["encoder_outputs"] = {
                key: value.repeat_interleave(expand_size, dim=0)
                for key, value in model_kwargs["encoder_outputs"].items()
            }
        return input_ids, model_kwargs


__all__ = [
    "Ovis2_5Config",
    "Ovis2_5ImageProcessor",
    "Ovis2_5ImageProcessorPil",
    "Ovis2_5VisionConfig",
    "Ovis2_5VisionModel",
    "Ovis2_5PreTrainedModel",
    "Ovis2_5Model",
    "Ovis2_5ForConditionalGeneration",
]
