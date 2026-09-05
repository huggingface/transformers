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
from ...image_processing_backends import PilBackend, TorchvisionBackend
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling, SizeDict
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from ...utils.generic import accepts_precomputed_kwargs, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    get_vision_window_index,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..cosmos3_edge.configuration_cosmos3_edge import Cosmos3EdgeVisionConfig
from ..exaone4_5.modeling_exaone4_5 import Exaone4_5_ForConditionalGeneration
from ..glm4v.image_processing_glm4v import Glm4vImageProcessor, Glm4vImageProcessorKwargs
from ..glm4v.image_processing_pil_glm4v import Glm4vImageProcessorPil
from ..glm4v.video_processing_glm4v import Glm4vVideoProcessor
from ..muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel, MuseGlimmerVisionRotaryEmbedding
from ..ovis2.modeling_ovis2 import Ovis2Model
from ..paddleocr_vl.modeling_paddleocr_vl import PaddleOCRVisionEmbeddings
from ..video_llama_3.modeling_video_llama_3 import (
    VideoLlama3ModelOutputWithPast,
    VideoLlama3PreTrainedModel,
    VideoLlama3VisionAttention,
    VideoLlama3VisionEncoderLayer,
    VideoLlama3VisionMLP,
)


class Ovis2_5ImageProcessorKwargs(Glm4vImageProcessorKwargs, total=False):
    r"""
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
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    # Unlike Qwen, Ovis expands dimensions below the factor and clamps aspect ratios above 200.
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

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Ovis2_5ImageProcessor(Glm4vImageProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = Ovis2_5ImageProcessorKwargs

    def resize(
        self,
        images: torch.Tensor,
        size: SizeDict,
        resample: PILImageResampling | int | None,
        factor: int,
        temporal_factor: int,
        **kwargs,
    ) -> torch.Tensor:
        height, width = images.shape[-2:]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        return TorchvisionBackend.resize(
            self,
            image=images,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        size = images_kwargs.get("size", self.size)
        min_pixels = size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
        max_pixels = size["longest_edge"] if isinstance(size, dict) else size.longest_edge
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)


class Ovis2_5ImageProcessorPil(Glm4vImageProcessorPil):
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = Ovis2_5ImageProcessorKwargs

    def resize(
        self,
        image: Any,
        size: SizeDict,
        resample: PILImageResampling | int | None,
        factor: int,
        temporal_factor: int,
        **kwargs,
    ) -> Any:
        height, width = image.shape[-2:]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        return PilBackend.resize(
            self,
            image=image,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        size = images_kwargs.get("size", self.size)
        min_pixels = size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
        max_pixels = size["longest_edge"] if isinstance(size, dict) else size.longest_edge
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)


class Ovis2_5VideoProcessorInitKwargs(VideosKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size used by the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size used by the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The spatial merge size between the vision encoder and language model.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int


class Ovis2_5VideoProcessor(Glm4vVideoProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_sample_frames = False
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    max_image_size = AttributeError()
    max_duration = AttributeError()
    num_frames = AttributeError()
    fps = AttributeError()
    valid_kwargs = Ovis2_5VideoProcessorInitKwargs

    def sample_frames(self, **super_kwargs):
        raise AttributeError()

    def resize(
        self,
        videos: torch.Tensor,
        size: SizeDict,
        resample: PILImageResampling | int | None,
        factor: int,
        temporal_factor: int,
        **kwargs,
    ) -> torch.Tensor:
        """Resize each frame with the released Ovis2.5 spatial policy."""
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(f"`size` dict must contain 'shortest_edge' and 'longest_edge' keys but got {size}.")

        height, width = videos.shape[-2:]
        resized_height, resized_width = smart_resize(
            height=height,
            width=width,
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        return TorchvisionBackend.resize(
            self,
            image=videos,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
        )

    def get_number_of_video_patches(
        self,
        num_frames: int,
        height: int,
        width: int,
        videos_kwargs: dict | None = None,
    ) -> int:
        """
        A utility that returns the number of video patches for a given video size.

        Args:
            num_frames (`int`):
                Number of frames in the input video.
            height (`int`):
                Height of the input video frames.
            width (`int`):
                Width of the input video frames.
            videos_kwargs (`dict`, *optional*):
                Any kwargs to override defaults of the video processor.
        Returns:
            `int`: Number of video patches per video.
        """
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
        videos_kwargs = videos_kwargs or {}
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        do_resize = videos_kwargs.get("do_resize", self.do_resize)
        resized_height, resized_width = height, width
        if do_resize:
            size = videos_kwargs.get("size", self.size)
            min_pixels = size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
            max_pixels = size["longest_edge"] if isinstance(size, dict) else size.longest_edge
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        factor = patch_size * merge_size
        if resized_height % factor != 0 or resized_width % factor != 0:
            raise ValueError(
                "Ovis2.5 videos must have height and width divisible by "
                f"`patch_size * merge_size` ({factor}), got ({resized_height}, {resized_width})."
            )

        num_temporal_patches = math.ceil(num_frames / temporal_patch_size)
        return num_temporal_patches * (resized_height // patch_size) * (resized_width // patch_size)


@auto_docstring(checkpoint="AIDC-AI/Ovis2.5-2B")
@strict
class Ovis2_5VisionConfig(Cosmos3EdgeVisionConfig):
    r"""
    window_size (`int`, *optional*, defaults to 112):
        Window size, in input pixels, used by windowed vision-attention layers.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        Number of consecutive video frames represented by one temporal patch.
    vocab_size (`int`, *optional*, defaults to 65536):
        Size of the visual-token vocabulary, including four learned visual-boundary indicators.
    num_visual_indicator_tokens (`int`, *optional*, defaults to 4):
        Number of visual-vocabulary rows reserved for image/video boundary indicators.
    """

    model_type = "ovis2_5_vision"

    image_size: int = 512
    patch_size: int = 16
    window_size: int = 112
    layer_types: list[str] | tuple[str, ...] | None = None
    temporal_patch_size: int = 1
    vocab_size: int = 65536
    num_visual_indicator_tokens: int = 4
    initializer_range: float = 0.02
    max_position_embeddings: int = 1024
    rope_parameters: dict | None = None
    num_patches = AttributeError()

    def __post_init__(self, **kwargs):
        # Released configs use `hidden_stride=2`, matching the inherited spatial merge default.
        kwargs.pop("hidden_stride", None)
        full_attention_indexes = kwargs.pop("fullatt_block_indexes", None)
        if self.layer_types is None:
            if full_attention_indexes is None:
                self.layer_types = ["full_attention"] * self.num_hidden_layers
            else:
                if isinstance(full_attention_indexes, str):
                    full_attention_indexes = [int(index) for index in full_attention_indexes.split("|") if index]
                full_attention_indexes = set(full_attention_indexes)
                self.layer_types = [
                    "full_attention" if layer_index in full_attention_indexes else "sliding_attention"
                    for layer_index in range(self.num_hidden_layers)
                ]
        else:
            self.layer_types = list(self.layer_types)
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="AIDC-AI/Ovis2.5-2B")
@strict
class Ovis2_5Config(PreTrainedConfig):
    r"""
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

    text_config: PreTrainedConfig | dict | None = None
    vision_config: Ovis2_5VisionConfig | dict | None = None
    image_token_id: int = 151669
    video_token_id: int = 151669
    image_start_token_id: int = 151670
    image_end_token_id: int = 151671
    video_start_token_id: int = 151672
    video_end_token_id: int = 151673
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        # Released Hub checkpoints still use the remote-code names. Pop them here until their configs can be updated.
        legacy_text_config = kwargs.pop("llm_config", None)
        legacy_vision_config = kwargs.pop("vit_config", None)
        legacy_visual_vocab_size = kwargs.pop("visual_vocab_size", None)
        if self.text_config is None and legacy_text_config is not None:
            self.text_config = legacy_text_config
        if self.vision_config is None and legacy_vision_config is not None:
            self.vision_config = legacy_vision_config

        if isinstance(self.vision_config, dict):
            # Released configs label this remote-code tower as `siglip2_navit`; the native tower type is fixed here.
            self.vision_config.pop("model_type", None)
            self.vision_config.pop("preserve_original_pe", None)
            self.vision_config.pop("use_rope", None)
            self.vision_config.pop("num_patches", None)
            if legacy_visual_vocab_size is not None:
                self.vision_config.setdefault("vocab_size", legacy_visual_vocab_size)
            self.vision_config = Ovis2_5VisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Ovis2_5VisionConfig()

        if isinstance(self.text_config, dict):
            model_type = self.text_config.pop("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = self.text_config.tie_word_embeddings
        super().__post_init__(**kwargs)


class Ovis2_5VisionRotaryEmbedding(MuseGlimmerVisionRotaryEmbedding):
    pass


class Ovis2_5VisionEmbeddings(PaddleOCRVisionEmbeddings):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__(config)
        del self.position_ids
        self.spatial_merge_size = config.spatial_merge_size
        self.interpolation_mode = "bicubic"
        self.interpolation_align_corners = False

    def interpolate_pos_encoding(self):
        raise AttributeError("Not needed for Ovis2.5")

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.spatial_merge_size,
            kwargs=kwargs,
        )
        pixel_values = pixel_values.view(
            -1,
            1,
            self.config.num_channels * self.config.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        kwargs["interp_indices"] = interp_indices
        kwargs["interp_weights"] = interp_weights.to(self.position_embedding.weight.dtype)
        return super().forward(pixel_values, grid_thw, **kwargs)


class Ovis2_5VisionMLP(VideoLlama3VisionMLP):
    pass


class Ovis2_5VisionAttention(VideoLlama3VisionAttention):
    pass


class Ovis2_5VisionEncoderLayer(VideoLlama3VisionEncoderLayer, GradientCheckpointingLayer):
    pass


class Ovis2_5VisionEncoder(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Ovis2_5VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class Ovis2_5VisualTokenProjector(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.spatial_merge_unit = config.spatial_merge_size**2
        visual_token_vocab_size = config.vocab_size - config.num_visual_indicator_tokens
        self.head_linear = nn.Linear(
            config.hidden_size * self.spatial_merge_unit,
            visual_token_vocab_size,
            bias=False,
        )
        self.head_norm = nn.LayerNorm(visual_token_vocab_size)
        self.indicator_padding = nn.Buffer(torch.zeros(config.num_visual_indicator_tokens), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(hidden_states.shape[0] // self.spatial_merge_unit, -1)
        logits = self.head_norm(self.head_linear(hidden_states))
        visual_tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        indicator_padding = self.indicator_padding.expand(visual_tokens.shape[0], -1)
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


@auto_docstring
class Ovis2_5PreTrainedModel(VideoLlama3PreTrainedModel):
    config: Ovis2_5Config
    _no_split_modules = ["Ovis2_5VisionEncoderLayer"]
    _supports_cache_class = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Ovis2_5VisualTokenProjector):
            nn.init.zeros_(module.indicator_padding)


@auto_docstring(custom_intro="The Ovis2.5 vision tower, without the visual tokenizer or language model.")
class Ovis2_5VisionModel(Ovis2_5PreTrainedModel, MuseGlimmerVisionModel):
    config: Ovis2_5VisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image", "video")
    _input_embed_layer = "patch_embedding"
    _can_record_outputs = {
        "hidden_states": Ovis2_5VisionEncoderLayer,
        "attentions": Ovis2_5VisionAttention,
    }

    def __init__(self, config: Ovis2_5VisionConfig):
        Ovis2_5PreTrainedModel.__init__(self, config)
        self.spatial_merge_size = config.spatial_merge_size
        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.embeddings = Ovis2_5VisionEmbeddings(config)
        self.encoder = Ovis2_5VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Ovis2_5VisionRotaryEmbedding(config)
        self.post_init()

    def pixel_shuffle(self):
        raise AttributeError("Not needed for Ovis2.5")

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            Temporal, height, and width patch-grid dimensions for each packed image or video.
        """
        hidden_states = self.embeddings(pixel_values, grid_thw, **kwargs)
        spatial_merge_unit = self.spatial_merge_size**2
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            window_size=self.window_size,
            patch_size=self.patch_size,
            kwargs=kwargs,
        )
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        sequence_length = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index].reshape(sequence_length, -1)

        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        position_ids = position_ids.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        position_ids = position_ids[window_index].reshape(sequence_length, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens_mapping = {
            "full_attention": (cu_seqlens, max_seqlen),
            "sliding_attention": (cu_window_seqlens, None),
        }
        for layer_index, encoder_layer in enumerate(self.encoder.layers):
            layer_cu_seqlens, layer_max_seqlen = cu_seqlens_mapping[self.config.layer_types[layer_index]]
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=layer_cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=layer_max_seqlen,
                **kwargs,
            )

        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        pre_layernorm_hidden_state = hidden_states[reverse_indices].reshape(sequence_length, -1)
        last_hidden_state = self.post_layernorm(pre_layernorm_hidden_state)
        # The released visual tokenizer consumes the final encoder state before this output normalization.
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pre_layernorm_hidden_state,
        )


@auto_docstring(custom_intro="The bare Ovis2.5 multimodal model, without the language modeling head.")
class Ovis2_5Model(Ovis2Model):
    def __init__(self, config: Ovis2_5Config):
        PreTrainedModel.__init__(self, config)
        self.vision_tower = Ovis2_5VisionModel(config.vision_config)
        self.visual_tokenizer = Ovis2_5VisualTokenProjector(config.vision_config)
        self.visual_embeddings_table = nn.Embedding(
            config.vision_config.vocab_size,
            config.text_config.hidden_size,
        )
        self.language_model = AutoModel.from_config(config.text_config)
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
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        )
        visual_tokens = self.visual_tokenizer(vision_outputs.pooler_output)
        visual_features = torch.matmul(visual_tokens, self.visual_embeddings_table.weight)
        indicator_start = self.vision_tower.config.vocab_size - self.vision_tower.config.num_visual_indicator_tokens
        indicator_token_ids = torch.arange(
            indicator_start,
            self.vision_tower.config.vocab_size,
            dtype=torch.long,
            device=visual_features.device,
        )
        visual_indicator_features = self.visual_embeddings_table(indicator_token_ids)
        split_sizes = (image_grid_thw.prod(dim=1) // self.vision_tower.config.spatial_merge_size**2).tolist()
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
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("Ovis2.5 supports images or video in one request, but not both.")
        merged_inputs_embeds: torch.Tensor = (
            self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        )

        if pixel_values is not None:
            image_outputs = self.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
                **kwargs,
            )
            image_hidden_states = torch.cat(image_outputs.pooler_output, dim=0)
            video_hidden_states = None
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
            image_hidden_states = None
            visual_features = video_hidden_states
            visual_indicator_features = video_outputs.visual_indicator_features
            boundary_token_ids = (self.config.video_start_token_id, self.config.video_end_token_id)
            indicator_indexes = (2, 3)
            num_visual_inputs = len(video_outputs.pooler_output)
        else:
            image_hidden_states = None
            video_hidden_states = None

        if pixel_values is not None or pixel_values_videos is not None:
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
                num_boundary_tokens = boundary_mask.sum()
                torch_compilable_check(
                    num_boundary_tokens == num_visual_inputs,
                    lambda: (
                        f"Expected {num_visual_inputs} visual boundary tokens with id {boundary_token_id}, but found "
                        f"{num_boundary_tokens}."
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
class Ovis2_5ForConditionalGeneration(Ovis2_5PreTrainedModel, Exaone4_5_ForConditionalGeneration):
    """Ovis2.5 multimodal conditional generation model."""

    model: Ovis2_5Model

    def __init__(self, config: Ovis2_5Config):
        super().__init__(config)
        # Exaone uses an underscored model class, while Ovis exposes `Ovis2_5Model`.
        self.model = Ovis2_5Model(config)
        self.post_init()

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
                torch.full(
                    (),
                    self.config.image_start_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            video_start_embedding = self.get_input_embeddings()(
                torch.full(
                    (),
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


__all__ = [
    "Ovis2_5Config",
    "Ovis2_5ImageProcessor",
    "Ovis2_5ImageProcessorPil",
    "Ovis2_5VideoProcessor",
    "Ovis2_5VisionConfig",
    "Ovis2_5VisionModel",
    "Ovis2_5PreTrainedModel",
    "Ovis2_5Model",
    "Ovis2_5ForConditionalGeneration",
]
