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
from typing import Any

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
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
from ..auto import AutoConfig, AutoModel
from ..ovis2.modeling_ovis2 import Ovis2ForConditionalGeneration, Ovis2Model
from ..qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs
from ..qwen3 import Qwen3Config
from ..video_llama_3.modeling_video_llama_3 import (
    VideoLlama3CausalLMOutputWithPast,
    VideoLlama3ModelOutputWithPast,
    VideoLlama3PreTrainedModel,
    VideoLlama3VisionAttention,
    VideoLlama3VisionEncoderLayer,
    VideoLlama3VisionMLP,
    VideoLlama3VisionRotaryEmbedding,
)


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
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = Ovis2_5ImageProcessorKwargs


class Ovis2_5ImageProcessorPil(Qwen2VLImageProcessorPil):
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = Ovis2_5ImageProcessorKwargs


@auto_docstring(checkpoint="AIDC-AI/Ovis2.5-2B")
@strict
class Ovis2_5VisionConfig(PreTrainedConfig):
    r"""
    hidden_stride (`int`, *optional*, defaults to 2):
        Spatial grouping factor applied before the visual-tokenizer head.
    window_size (`int`, *optional*, defaults to 112):
        Window size, in input pixels, used by windowed vision-attention layers.
    layer_types (`list[str]`, *optional*):
        Per-layer attention type. Values are `"full_attention"` or `"sliding_attention"`. When omitted, every layer
        uses full attention.
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
    image_size: int = 512
    patch_size: int = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    hidden_stride: int = 2
    window_size: int = 112
    layer_types: list[str] | tuple[str, ...] | None = None
    temporal_patch_size: int = 1
    preserve_original_pe: bool = True
    use_rope: bool = True
    vocab_size: int = 65536
    num_visual_indicator_tokens: int = 4
    initializer_range: float = 0.02

    # Ignore copy
    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        else:
            self.layer_types = list(self.layer_types)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"Expected one vision attention type per layer, but got {len(self.layer_types)} entries for "
                f"{self.num_hidden_layers} layers."
            )
        if invalid_layer_types := set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported Ovis2.5 vision attention types: {sorted(invalid_layer_types)}")

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="AIDC-AI/Ovis2.5-2B")
@strict
class Ovis2_5Config(PreTrainedConfig):
    r"""
    visual_vocab_size (`int`, *optional*, defaults to 65536):
        Size of the visual-token vocabulary shared by the visual tokenizer and visual embedding table.
    image_token_id (`int`, *optional*, defaults to 151669):
        Text-vocabulary token used as the placeholder for one image atom.
    video_token_id (`int`, *optional*, defaults to 151669):
        Text-vocabulary token used as the placeholder for one video atom.
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

    text_config: Qwen3Config | dict | None = None
    vision_config: Ovis2_5VisionConfig | dict | None = None
    visual_vocab_size: int = 65536
    image_token_id: int = 151669
    video_token_id: int = 151669
    image_start_token_id: int = 151670
    image_end_token_id: int = 151671
    video_start_token_id: int = 151672
    video_end_token_id: int = 151673
    tie_word_embeddings: bool = False

    # Ignore copy
    def __post_init__(self, **kwargs):
        vision_config = self.vision_config
        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)
            vision_config = Ovis2_5VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Ovis2_5VisionConfig()
        self.vision_config = vision_config

        text_config = self.text_config
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            text_config = Qwen3Config(**text_config)
        elif text_config is None:
            text_config = Qwen3Config()
        self.text_config = text_config

        vision_config.vocab_size = self.visual_vocab_size
        if not self.tie_word_embeddings and text_config.tie_word_embeddings:
            self.tie_word_embeddings = text_config.tie_word_embeddings
        super().__post_init__(**kwargs)


class Ovis2_5VisionRotaryEmbedding(VideoLlama3VisionRotaryEmbedding):
    pass


class Ovis2_5VisionEmbeddings(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
            bias=True,
        )
        # CODEPATH: AIDC-AI/Ovis2.5-2B and Ovis2.5-9B set `preserve_original_pe=True`; `False` supports
        # custom configs without the learned position embedding.
        if config.preserve_original_pe:
            self.position_embedding_size = config.image_size // config.patch_size
            self.position_embedding = nn.Embedding(self.position_embedding_size**2, config.hidden_size)

    def forward(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor) -> torch.Tensor:
        grid_values = grid_thw.tolist()
        target_dtype = self.patch_embedding.weight.dtype
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


class Ovis2_5VisionEncoderLayer(VideoLlama3VisionEncoderLayer):
    pass


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
            use_full_attention = self.config.layer_types[layer_index] == "full_attention"
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

        if recorded_hidden_states is not None:
            hidden_states = recorded_hidden_states
        else:
            hidden_states = hidden_states.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
            hidden_states = hidden_states[reverse_indices].reshape(sequence_length, -1)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Ovis2_5VisualTokenProjector(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.spatial_merge_unit = config.hidden_stride**2
        self.num_visual_indicator_tokens = config.num_visual_indicator_tokens
        visual_token_vocab_size = config.vocab_size - config.num_visual_indicator_tokens
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size * self.spatial_merge_unit, visual_token_vocab_size, bias=False),
            nn.LayerNorm(visual_token_vocab_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(hidden_states.shape[0] // self.spatial_merge_unit, -1)
        logits = self.head(hidden_states)
        visual_tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        indicator_padding = torch.zeros(
            (visual_tokens.shape[0], self.num_visual_indicator_tokens),
            dtype=visual_tokens.dtype,
            device=visual_tokens.device,
        )
        return torch.cat((visual_tokens, indicator_padding), dim=-1)


@auto_docstring
@dataclass
class Ovis2_5VisualFeaturesOutput(BaseModelOutputWithPooling):
    r"""
    visual_indicator_features (`torch.FloatTensor` of shape `(4, hidden_size)`, *optional*):
        Learned image-begin, image-end, video-begin, and video-end embeddings, in that order.
    """

    visual_indicator_features: torch.FloatTensor | None = None


class Ovis2_5ModelOutputWithPast(VideoLlama3ModelOutputWithPast):
    pass


class Ovis2_5CausalLMOutputWithPast(VideoLlama3CausalLMOutputWithPast):
    pass


@auto_docstring
class Ovis2_5PreTrainedModel(VideoLlama3PreTrainedModel):
    config: Ovis2_5Config
    _no_split_modules = ["Ovis2_5VisionEncoderLayer"]
    _supports_cache_class = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False


@auto_docstring(custom_intro="The Ovis2.5 vision tower, without the visual tokenizer or language model.")
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
        self.embeddings = Ovis2_5VisionEmbeddings(config)
        self.encoder = Ovis2_5VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = Ovis2_5VisionRotaryEmbedding(head_dim // 2)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

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
        hidden_states = self.embeddings(pixel_values, grid_thw)
        position_ids = get_vision_position_ids(grid_thw, self.config.hidden_stride, kwargs=kwargs)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary_pos_emb.cos(), rotary_pos_emb.sin())
        encoder_outputs = self.encoder(
            hidden_states,
            grid_thw=grid_thw,
            position_embeddings=position_embeddings,
            output_hidden_states=(
                self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
            ),
            **kwargs,
        )
        pre_layernorm_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(pre_layernorm_hidden_state)
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pre_layernorm_hidden_state,
        )


@auto_docstring(custom_intro="The bare Ovis2.5 multimodal model, without the language modeling head.")
class Ovis2_5Model(Ovis2Model):
    def __init__(self, config: Ovis2_5Config):
        PreTrainedModel.__init__(self, config)
        vision_config: Ovis2_5VisionConfig = config.vision_config
        text_config: Qwen3Config = config.text_config
        self.vision_tower = Ovis2_5VisionModel(vision_config)
        self.visual_tokenizer = Ovis2_5VisualTokenProjector(vision_config)
        self.visual_embeddings_table = nn.Embedding(
            config.visual_vocab_size,
            text_config.hidden_size,
        )
        self.language_model = AutoModel.from_config(text_config)
        self.post_init()

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
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        )
        visual_tokens = self.visual_tokenizer(vision_outputs.pooler_output)
        visual_features = torch.matmul(visual_tokens, self.visual_embeddings_table.weight)
        indicator_start = self.config.visual_vocab_size - self.vision_tower.config.num_visual_indicator_tokens
        indicator_token_ids = torch.arange(
            indicator_start,
            self.config.visual_vocab_size,
            dtype=torch.long,
            device=visual_features.device,
        )
        visual_indicator_features = self.visual_embeddings_table(indicator_token_ids)
        split_sizes = (image_grid_thw.prod(dim=1) // self.vision_tower.config.hidden_stride**2).tolist()
        return Ovis2_5VisualFeaturesOutput(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=torch.split(visual_features, split_sizes),
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            visual_indicator_features=visual_indicator_features,
        )

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
        return self.get_image_features(
            pixel_values=pixel_values_videos,
            image_grid_thw=video_grid_thw,
            **kwargs,
        )

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
        merged_inputs_embeds: torch.Tensor = (
            self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        )

        image_hidden_states = None
        video_hidden_states = None
        visual_features = None
        visual_indicator_features = None
        boundary_token_ids = ()
        indicator_indexes = ()
        num_visual_inputs = 0
        if pixel_values is not None:
            image_outputs = self.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
                **kwargs,
            )
            image_hidden_states = torch.cat(image_outputs.pooler_output, dim=0)
            visual_features = image_hidden_states
            visual_indicator_features = image_outputs.visual_indicator_features
            boundary_token_ids = (self.config.image_start_token_id, self.config.image_end_token_id)
            indicator_indexes = (0, 1)
            num_visual_inputs = len(image_outputs.pooler_output)
        elif pixel_values_videos is not None:
            video_outputs = self.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_dict=True,
                **kwargs,
            )
            video_hidden_states = torch.cat(video_outputs.pooler_output, dim=0)
            visual_features = video_hidden_states
            visual_indicator_features = video_outputs.visual_indicator_features
            boundary_token_ids = (self.config.video_start_token_id, self.config.video_end_token_id)
            indicator_indexes = (2, 3)
            num_visual_inputs = len(video_outputs.pooler_output)

        if visual_features is not None and visual_indicator_features is not None:
            atom_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=merged_inputs_embeds,
                image_features=visual_features,
            )
            merged_inputs_embeds = merged_inputs_embeds.masked_scatter(
                atom_mask,
                visual_features.to(merged_inputs_embeds.device, merged_inputs_embeds.dtype),
            )

            for boundary_token_id, indicator_index in zip(boundary_token_ids, indicator_indexes):
                if input_ids is None:
                    boundary_embedding = self.get_input_embeddings()(
                        torch.tensor(boundary_token_id, dtype=torch.long, device=merged_inputs_embeds.device)
                    )
                    boundary_mask = (merged_inputs_embeds == boundary_embedding).all(dim=-1)
                else:
                    boundary_mask = input_ids == boundary_token_id
                torch_compilable_check(
                    boundary_mask.sum() == num_visual_inputs,
                    lambda: (
                        f"Expected {num_visual_inputs} visual boundary tokens with id {boundary_token_id}, but found "
                        f"{boundary_mask.sum().item()}."
                    ),
                )
                boundary_features = visual_indicator_features[indicator_index].to(
                    merged_inputs_embeds.device,
                    merged_inputs_embeds.dtype,
                )
                merged_inputs_embeds = torch.where(
                    boundary_mask.unsqueeze(-1),
                    boundary_features.expand_as(merged_inputs_embeds),
                    merged_inputs_embeds,
                )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=merged_inputs_embeds,
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
class Ovis2_5ForConditionalGeneration(Ovis2ForConditionalGeneration):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Ovis2_5Config):
        PreTrainedModel.__init__(self, config)
        self.model = Ovis2_5Model(config)
        text_config: Qwen3Config = config.text_config
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

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
