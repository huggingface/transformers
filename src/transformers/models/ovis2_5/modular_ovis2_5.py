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

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_multimodal_utils import merge_mm_encoder_outputs
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndSplitSizes,
    MultimodalBaseModelOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import accepts_precomputed_kwargs, get_max_seqlen, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    get_vision_window_index,
)
from ..auto import AutoConfig
from ..cosmos3_edge.configuration_cosmos3_edge import Cosmos3EdgeVisionConfig
from ..exaone4_5.modeling_exaone4_5 import Exaone4_5_ForConditionalGeneration
from ..glm4v.image_processing_glm4v import Glm4vImageProcessor, Glm4vImageProcessorKwargs
from ..glm4v.image_processing_pil_glm4v import Glm4vImageProcessorPil
from ..glm4v.video_processing_glm4v import Glm4vVideoProcessor
from ..minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6VisionEmbeddings
from ..muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel, MuseGlimmerVisionRotaryEmbedding
from ..perception_lm.modeling_perception_lm import PerceptionLMModel
from ..video_llama_3.modeling_video_llama_3 import (
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
    *,
    num_frames: int = 1,
    temporal_factor: int = 1,
) -> tuple[int, int]:
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    The temporal arguments are accepted for the inherited resize wrappers; the pixel budget is per frame.

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


class Ovis2_5ImageProcessorPil(Glm4vImageProcessorPil):
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = Ovis2_5ImageProcessorKwargs


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

    def get_num_of_video_patches(self, **super_kwargs):
        raise AttributeError()

    def get_number_of_video_patches(
        self,
        num_frames: int,
        height: int,
        width: int,
        videos_kwargs: dict | None = None,
    ) -> int:
        """
        A utility that returns the number of video patches for a given video size.

        Used by `Ovis2_5Processor._get_num_multimodal_tokens` to estimate visual token counts from video dimensions.

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

            # Mirror `resize` so the estimate matches the processed video grid.
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

        # Released checkpoints use temporal_patch_size=1; match `patchify` for explicit processor overrides too.
        num_temporal_patches = math.ceil(num_frames / temporal_patch_size)
        return num_temporal_patches * (resized_height // patch_size) * (resized_width // patch_size)


@auto_docstring(checkpoint="ovis-community/Ovis2.5-2B")
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
    default_rope_type = "axial"

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
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        else:
            self.layer_types = list(self.layer_types)
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="ovis-community/Ovis2.5-2B")
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
    sub_configs_defaults = {
        "vision_config": Ovis2_5VisionConfig,
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen3"),
    }

    text_config: PreTrainedConfig | dict | None = None
    vision_config: PreTrainedConfig | dict | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    image_start_token_id: int = 151670
    image_end_token_id: int = 151671
    video_start_token_id: int = 151672
    video_end_token_id: int = 151673
    tie_word_embeddings: bool = False


class Ovis2_5VisionRotaryEmbedding(MuseGlimmerVisionRotaryEmbedding):
    pass


class Ovis2_5VisionEmbeddings(MiniCPMV4_6VisionEmbeddings):
    """Patch embeddings and interpolated learned positions for packed Ovis2.5 inputs."""

    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__(config)
        self.config = config
        # How the (square) learned position grid is resampled to each image's grid.
        self.num_grid_per_side = int(self.num_positions**0.5)
        self.interpolation_align_corners = False
        self.interpolation_mode = "bicubic"
        self.spatial_merge_size = config.spatial_merge_size

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(total_patches, num_channels * temporal_patch_size * patch_size**2)`):
                Packed flattened image or video patches.
            grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
                Temporal, height, and width patch-grid dimensions for each packed image or video.
        """
        # MiniCPM uses nearest-position lookup; Ovis needs bicubic positions in spatial-merge order.
        # Casting weights to the position table's dtype preserves the checkpoint's low-precision numerics.
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
            self.config.num_channels * self.config.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        target_dtype = self.patch_embedding.weight.dtype
        embeddings = self.patch_embedding(pixel_values.to(dtype=target_dtype)).flatten(1)

        interp_weights = interp_weights.to(self.position_embedding.weight.dtype)
        pos_embeds = (self.position_embedding(interp_indices) * interp_weights[:, :, None]).sum(1)
        embeddings = embeddings + pos_embeds.to(embeddings.dtype)

        return embeddings


class Ovis2_5VisionMLP(VideoLlama3VisionMLP):
    pass


class Ovis2_5VisionAttention(VideoLlama3VisionAttention):
    pass


class Ovis2_5VisionEncoderLayer(VideoLlama3VisionEncoderLayer, GradientCheckpointingLayer):
    pass


class Ovis2_5VisualTokenProjector(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig):
        super().__init__()
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.num_visual_indicator_tokens = config.num_visual_indicator_tokens
        self.vocab_size = config.vocab_size
        self.visual_token_vocab_size = config.vocab_size - config.num_visual_indicator_tokens
        self.linear = nn.Linear(
            config.hidden_size * self.spatial_merge_unit,
            self.visual_token_vocab_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(self.visual_token_vocab_size)
        self.indicator_padding = nn.Buffer(torch.zeros(config.num_visual_indicator_tokens), persistent=False)
        self.visual_indicator_token_ids = nn.Buffer(
            torch.arange(self.visual_token_vocab_size, self.vocab_size, dtype=torch.long), persistent=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(hidden_states.shape[0] // self.spatial_merge_unit, -1)
        logits = self.norm(self.linear(hidden_states))
        visual_tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        indicator_padding = self.indicator_padding.expand(visual_tokens.shape[0], -1)
        return torch.cat((visual_tokens, indicator_padding), dim=-1)

    def get_visual_indicator_token_ids(self, device: torch.device) -> torch.Tensor:
        return self.visual_indicator_token_ids.to(device)


class Ovis2_5VisualEmbeddingProjection(nn.Linear):
    def forward(self, inputs: torch.Tensor, *, lookup: bool = False) -> torch.Tensor:
        if lookup:
            return nn.functional.embedding(inputs, self.weight.T)
        return super().forward(inputs)


class Ovis2_5MultiModalProjector(nn.Module):
    def __init__(self, config: Ovis2_5VisionConfig, text_hidden_size: int):
        super().__init__()
        self.visual_tokenizer = Ovis2_5VisualTokenProjector(config)
        self.visual_embeddings_table = Ovis2_5VisualEmbeddingProjection(
            config.vocab_size, text_hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.visual_tokenizer(hidden_states)
        return self.visual_embeddings_table(visual_tokens)

    def get_visual_indicator_features(self, device: torch.device) -> torch.Tensor:
        indicator_token_ids = self.visual_tokenizer.get_visual_indicator_token_ids(device)
        return self.visual_embeddings_table(indicator_token_ids, lookup=True)


@auto_docstring
class Ovis2_5PreTrainedModel(VideoLlama3PreTrainedModel):
    config: Ovis2_5Config
    _no_split_modules = ["Ovis2_5VisionEncoderLayer"]
    _supports_cache_class = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Ovis2_5VisualTokenProjector):
            init.zeros_(module.indicator_padding)
            init.copy_(
                module.visual_indicator_token_ids,
                torch.arange(
                    module.visual_token_vocab_size,
                    module.vocab_size,
                    dtype=torch.long,
                    device=module.visual_indicator_token_ids.device,
                ),
            )


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
        self.layers = nn.ModuleList([Ovis2_5VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
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
        # Ovis feeds embeddings to the encoder without Muse's input LayerNorm.
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
        max_window_seqlen = get_max_seqlen(
            cu_window_seqlens, self.config, kwargs=kwargs, kwarg_name="max_window_seqlen"
        )

        sequence_length = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index].reshape(sequence_length, -1)

        # Ovis uses these RoPE coordinates without Muse's axis flip or offset.
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        position_ids = position_ids.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        position_ids = position_ids[window_index].reshape(sequence_length, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens_mapping = {
            "full_attention": (cu_seqlens, max_seqlen),
            "window_attention": (cu_window_seqlens, max_window_seqlen),
        }
        for layer_index, encoder_layer in enumerate(self.layers):
            layer_cu_seqlens, layer_max_seqlen = cu_seqlens_mapping[self.config.layer_types[layer_index]]
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=layer_cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=layer_max_seqlen,
                **kwargs,
            )

        # Keep one feature row per patch; the visual tokenizer performs spatial merging later.
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states.reshape(sequence_length // spatial_merge_unit, spatial_merge_unit, -1)
        pre_layernorm_hidden_state = hidden_states[reverse_indices].reshape(sequence_length, -1)
        last_hidden_state = self.post_layernorm(pre_layernorm_hidden_state)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)


@auto_docstring(custom_intro="The bare Ovis2.5 multimodal model, without the language modeling head.")
class Ovis2_5Model(PerceptionLMModel):
    def __init__(self, config: Ovis2_5Config):
        super().__init__(config)
        self.vision_tower = Ovis2_5VisionModel(config.vision_config)
        self.multi_modal_projector = Ovis2_5MultiModalProjector(
            config.vision_config,
            config.text_config.hidden_size,
        )

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring(custom_intro="Encodes images into Ovis2.5 visual embeddings.")
    def get_image_embeddings(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPoolingAndSplitSizes:
        kwargs["output_hidden_states"] = kwargs.get("output_hidden_states") or -1
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        )
        visual_features = self.multi_modal_projector(vision_outputs.hidden_states[-1])
        split_sizes = image_grid_thw.prod(dim=1) // self.vision_tower.config.spatial_merge_size**2
        vision_outputs.pooler_output = visual_features
        return BaseModelOutputWithPoolingAndSplitSizes(**vision_outputs, split_sizes=split_sizes)

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring(custom_intro="Encodes videos into Ovis2.5 visual embeddings.")
    def get_video_embeddings(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPoolingAndSplitSizes:
        return self.get_image_embeddings(
            pixel_values=pixel_values_videos,
            image_grid_thw=video_grid_thw,
            return_dict=True,
            **kwargs,
        )

    def _merge_visual_features(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor,
        visual_outputs: BaseModelOutputWithPoolingAndSplitSizes,
        is_video: bool,
    ) -> torch.Tensor:
        """Replace visual placeholders and start/end tokens with their visual embeddings."""
        if is_video:
            boundary_token_ids = (self.config.video_start_token_id, self.config.video_end_token_id)
            first_indicator_index = 2
        else:
            boundary_token_ids = (self.config.image_start_token_id, self.config.image_end_token_id)
            first_indicator_index = 0

        num_visual_inputs = visual_outputs.split_sizes.shape[0]
        visual_features = visual_outputs.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, video_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=None if is_video else visual_features,
            video_features=visual_features if is_video else None,
        )
        visual_mask = video_mask if is_video else image_mask
        merged_embeds = inputs_embeds.masked_scatter(visual_mask, visual_features)

        visual_indicator_features = self.multi_modal_projector.get_visual_indicator_features(visual_features.device)
        visual_indicator_features = visual_indicator_features.to(inputs_embeds.device, inputs_embeds.dtype)

        for indicator_index, boundary_token_id in enumerate(boundary_token_ids, start=first_indicator_index):
            if input_ids is None:
                boundary_embedding = self.get_input_embeddings()(
                    torch.full((), boundary_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                boundary_mask = (inputs_embeds == boundary_embedding).all(dim=-1)
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
            merged_embeds = torch.where(
                boundary_mask.unsqueeze(-1),
                visual_indicator_features[indicator_index],
                merged_embeds,
            )

        return merged_embeds

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    @merge_mm_encoder_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalBaseModelOutputWithPast:
        r"""
        mm_token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Modality labels returned by the processor with `return_mm_token_type_ids=True`. Pass them with
            `inputs_embeds` to locate visual placeholders when token embeddings are shared.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_embeddings(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                return_dict=True,
                **kwargs,
            )
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_embeddings(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                return_dict=True,
                **kwargs,
            )

        if inputs_embeds is not None and (
            mm_encoder_outputs.get("video") is not None and mm_encoder_outputs.get("image") is not None
        ):
            raise ValueError("The model doesn't support mixed-inputs with `inputs_embeds` as the main input.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if mm_encoder_outputs.get("image") is not None:
            inputs_embeds = self._merge_visual_features(
                input_ids,
                inputs_embeds,
                mm_encoder_outputs["image"],
                is_video=False,
            )

        if mm_encoder_outputs.get("video") is not None:
            inputs_embeds = self._merge_visual_features(
                input_ids,
                inputs_embeds,
                mm_encoder_outputs["video"],
                is_video=True,
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        return MultimodalBaseModelOutputWithPast(
            **outputs, image_outputs=mm_encoder_outputs.get("image"), video_outputs=mm_encoder_outputs.get("video")
        )


@auto_docstring(custom_intro="The Ovis2.5 multimodal model with a language modeling head.")
class Ovis2_5ForConditionalGeneration(Ovis2_5PreTrainedModel, Exaone4_5_ForConditionalGeneration):
    """Ovis2.5 multimodal conditional generation model."""

    def __init__(self, config: Ovis2_5Config):
        super().__init__(config)
        # Exaone uses an underscored model class, while Ovis exposes `Ovis2_5Model`.
        self.model = Ovis2_5Model(config)

    def prepare_inputs_for_generation(self, input_ids, mm_token_type_ids=None, **kwargs):
        if mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = mm_token_type_ids
        return super().prepare_inputs_for_generation(input_ids, **kwargs)


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
