# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
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

import math

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel, slice_logits_to_keep
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from ...utils.deprecation import deprecate_kwarg
from ...utils.output_capturing import select_hidden_states
from ..audioflamingo3.configuration_audioflamingo3 import AudioFlamingo3Config
from ..audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3MultiModalProjector
from ..auto import AutoConfig, AutoModel
from ..got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
from ..internvl.modeling_internvl import InternVLMultiModalProjector
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaModel, LlavaPreTrainedModel
from ..musicflamingo.modeling_musicflamingo import apply_rotary_time_emb


@auto_docstring(checkpoint="nvidia/nemotron-labs-audio-visual-flamingo-hf")
@strict
class AudioVisualFlamingoConfig(AudioFlamingo3Config):
    r"""
    video_token_id (`int`, *optional*, defaults to 151650):
        The video token index used as a placeholder for video embeddings.
    s2_scales (`list[int]`, *optional*, defaults to `(448, 896, 1344)`):
        Image sizes of the multi-scale (S2) vision features. The first one is the vision tower input size, and the
        features of all scales are concatenated along the channel dimension.
    video_temporal_pool_size (`int`, *optional*, defaults to 8):
        Number of consecutive video frames whose visual tokens are averaged together.
    time_embedding_dim (`int`, *optional*, defaults to 128):
        Number of channels rotated per axis by the constrained rotary time embedding of video and audio tokens.
    video_time_embedding_theta (`float`, *optional*, defaults to 50000.0):
        Base period of the time embedding of video tokens.
    audio_time_embedding_theta (`float`, *optional*, defaults to 10000.0):
        Base period of the time embedding of audio tokens.

    Example:

    ```python
    >>> from transformers import AudioVisualFlamingoConfig, AudioVisualFlamingoForConditionalGeneration

    >>> # Initializing an AudioVisualFlamingo configuration
    >>> configuration = AudioVisualFlamingoConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = AudioVisualFlamingoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audiovisualflamingo"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen2"),
        "vision_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="siglip_vision_model",
            init_kwargs={
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "image_size": 448,
                "patch_size": 14,
            },
        ),
        "audio_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen2_audio_encoder"),
    }

    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151649
    video_token_id: int = 151650
    audio_token_id: int = 151652
    vision_feature_layer: int = -2
    s2_scales: list[int] | tuple[int, ...] = (448, 896, 1344)
    video_temporal_pool_size: int = 8
    time_embedding_dim: int = 128
    video_time_embedding_theta: float | int = 50000.0
    audio_time_embedding_theta: float | int = 10000.0
    tie_word_embeddings: bool = False


def get_s2_tile_grids(s2_scales: list[int], num_rows: int, num_cols: int) -> list[tuple[int, int]]:
    """Returns the tile grid of each S2 scale: square grids for all scales but the last one, which has its own grid."""
    return [(scale // s2_scales[0],) * 2 for scale in s2_scales[:-1]] + [(num_rows, num_cols)]


def get_time_dtype(device: torch.device) -> torch.dtype:
    """Times are kept in float64, as rotation angles grow large with time, or float32 on MPS, which lacks float64."""
    return torch.float32 if device.type == "mps" else torch.float64


class AudioVisualFlamingoImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    s2_scales (`list[int]`, *optional*, defaults to `[448, 896, 1344]`):
        Resolutions at which an image is tiled. Every scale but the last tiles a square resize of the image, while the
        last scale tiles the grid whose aspect ratio is the closest to the image one.
    max_tiles (`int`, *optional*, defaults to 12):
        Maximum number of tiles of the last scale.
    """

    s2_scales: list[int]
    max_tiles: int


@auto_docstring
class AudioVisualFlamingoImageProcessor(TorchvisionBackend):
    valid_kwargs = AudioVisualFlamingoImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_hw"]
    resample = PILImageResampling.BICUBIC
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    s2_scales = [448, 896, 1344]
    max_tiles = 12

    @auto_docstring
    def preprocess(
        self, images: ImageInput, **kwargs: Unpack[AudioVisualFlamingoImageProcessorKwargs]
    ) -> BatchFeature:
        # Images are only tiled in prompts holding a single image, the others are resized to a single tile
        if isinstance(images, (list, tuple)) and images and isinstance(images[0], (list, tuple)):
            images = [sample for sample in images if sample]
            batch_num_images = [len(sample) for sample in images]
        else:
            batch_num_images = [1] * (len(images) if isinstance(images, (list, tuple)) else 1)
        return super().preprocess(images, batch_num_images, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        batch_num_images: list[int],
        size: SizeDict,
        resample: PILImageResampling | None,
        s2_scales: list[int],
        max_tiles: int,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | None,
        **kwargs,
    ) -> BatchFeature:
        tiles, grids = [], []
        is_tiled = [num_images == 1 for num_images in batch_num_images for _ in range(num_images)]
        for image, tiled in zip(images, is_tiled):
            if not tiled:
                # A single tile, at the last scale, whose features are used for every scale by the model
                tiles.append(self.resize(image, size, resample))
                grids.append([(0, 0)] * (len(s2_scales) - 1) + [(1, 1)])
                continue
            num_cols, num_rows = get_optimal_tiled_canvas(
                image.shape[-2:], (size.height, size.width), (s2_scales[-1] // s2_scales[0]) ** 2, max_tiles
            )
            image_grids = get_s2_tile_grids(s2_scales, num_rows, num_cols)
            for rows, cols in image_grids:
                resized = self.resize(image, SizeDict(height=rows * size.height, width=cols * size.width), resample)
                # (channels, rows * height, cols * width) -> (rows * cols, channels, height, width)
                resized = resized.unflatten(-2, (rows, -1)).unflatten(-1, (cols, -1))
                tiles.extend(resized.permute(1, 3, 0, 2, 4).flatten(0, 1))
            grids.append(image_grids)

        pixel_values = self.rescale_and_normalize(
            torch.stack(tiles), do_rescale, rescale_factor, do_normalize, image_mean, image_std
        )
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_hw": torch.tensor(grids)}, tensor_type=return_tensors
        )


class AudioVisualFlamingoMultiModalProjector(InternVLMultiModalProjector):
    def __init__(self, config: AudioVisualFlamingoConfig):
        # Features of all S2 scales, merged 2x2 spatially
        input_size = 4 * len(config.s2_scales) * config.vision_config.hidden_size
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear_1 = nn.Linear(input_size, config.text_config.hidden_size, bias=config.projector_bias)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )


class AudioVisualFlamingoAudioProjector(AudioFlamingo3MultiModalProjector):
    """Projects the audio encoder features to the language model space."""

    # Only sets the type of `config`, that the modular converter would otherwise name `AudioVisualFlamingoAudioConfig`
    def __init__(self, config: AudioVisualFlamingoConfig):
        super().__init__(config)


class AudioVisualFlamingoRotaryEmbedding(nn.Module):
    """
    Constrained rotary time embedding of video and audio tokens. The first `2 * time_embedding_dim` channels of each
    embedding are rotated along two axes, its frame (or frame group) index and its token index, with angles scaled by
    its timestamp.
    """

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.inv_freq = nn.Buffer(self.compute_inv_freq(), persistent=False)

    def compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))

    @torch.no_grad()
    def forward(self, timestamps: torch.Tensor, num_frames: int, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            timestamps (`torch.Tensor`): Times in seconds, broadcastable with `(num_frames, num_tokens)`.
            num_frames (`int`): Number of frames, or frame groups.
            num_tokens (`int`): Number of tokens per frame.
        """
        device = timestamps.device
        inv_freq = self.inv_freq.to(device)
        frame_freqs = torch.arange(num_frames, device=device).float()[:, None] * inv_freq
        token_freqs = torch.arange(num_tokens, device=device).float()[:, None] * inv_freq
        freqs = torch.cat(torch.broadcast_tensors(frame_freqs[:, None], token_freqs[None]), dim=-1)
        freqs = freqs.repeat_interleave(2, dim=-1) * (-2 * math.pi * timestamps)[..., None]
        freqs = freqs.to(get_time_dtype(device))
        return freqs.cos(), freqs.sin()


def merge_tiles(tiles: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """Stitches `(num_rows * num_cols, channels, height, width)` tiles into a `(channels, num_rows * height, num_cols * width)` map."""
    tiles = tiles.unflatten(0, (num_rows, num_cols))  # (num_rows, num_cols, channels, height, width)
    return tiles.permute(2, 0, 3, 1, 4).flatten(3, 4).flatten(1, 2)


def temporal_pool(hidden_states: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Averages groups of `pool_size` frames along dim 1. An incomplete last group is padded with its mean value."""
    remainder = hidden_states.shape[1] % pool_size
    if remainder:
        last = hidden_states[:, -remainder:]
        padding = last.mean(dim=tuple(range(1, last.ndim)), keepdim=True)
        hidden_states = torch.cat([hidden_states, padding.expand(-1, pool_size - remainder, *last.shape[2:])], dim=1)
    return hidden_states.unflatten(1, (-1, pool_size)).mean(2)


class AudioVisualFlamingoPreTrainedModel(LlavaPreTrainedModel):
    input_modalities = ("image", "video", "audio", "text")

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, AudioVisualFlamingoRotaryEmbedding):
            init.copy_(module.inv_freq, module.compute_inv_freq())


@auto_docstring(
    custom_intro="""
    The AudioVisualFlamingo model, made of a SigLIP vision tower, a Whisper-style audio encoder, their projectors and a
    Qwen2 language model, without a language modeling head.
    """
)
class AudioVisualFlamingoModel(LlavaModel):
    def __init__(self, config: AudioVisualFlamingoConfig):
        super().__init__(config)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.audio_projector = AudioVisualFlamingoAudioProjector(config)
        self.video_rotary_emb = AudioVisualFlamingoRotaryEmbedding(
            config.time_embedding_dim, config.video_time_embedding_theta
        )
        self.audio_rotary_emb = AudioVisualFlamingoRotaryEmbedding(
            config.time_embedding_dim, config.audio_time_embedding_theta
        )

    def _get_vision_feature_maps(
        self, pixel_values: torch.Tensor, **kwargs
    ) -> tuple[BaseModelOutputWithPooling, torch.Tensor]:
        """Runs the vision tower and returns its outputs along with the selected features as `(batch, channels, height, width)` maps."""
        vision_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        features = vision_outputs.hidden_states[self.config.vision_feature_layer]
        side = math.isqrt(features.shape[1])
        return vision_outputs, features.transpose(1, 2).unflatten(2, (side, side))

    def _project_feature_maps(self, feature_maps: torch.Tensor) -> torch.Tensor:
        """Merges `(batch, channels, height, width)` feature maps 2x2 spatially, then projects them to the language model space."""
        feature_maps = feature_maps.unflatten(2, (-1, 2)).unflatten(4, (-1, 2))
        return self.multi_modal_projector(feature_maps.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

    @can_return_tuple
    @select_hidden_states(lambda self: self.config.vision_feature_layer)
    @auto_docstring(
        custom_intro="Encodes images given as multi-scale (S2) tiles and projects them to the language model space."
    )
    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_hw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_tiles, num_channels, height, width)`):
            Image tiles of all S2 scales, as returned by [`AudioVisualFlamingoImageProcessor`].
        image_grid_hw (`torch.LongTensor` of shape `(num_images, num_scales, 2)`):
            Number of tile rows and columns of each S2 scale of each image. Images encoded as a single tile only have
            a `(1, 1)` grid at the last scale, whose features are used for every scale.
        """
        vision_outputs, tiles = self._get_vision_feature_maps(pixel_values, **kwargs)
        num_tiles = image_grid_hw.prod(-1)
        tiles_per_image = tiles.split(num_tiles.sum(-1).tolist())

        image_features = []
        for image_tiles, grids, counts in zip(tiles_per_image, image_grid_hw.tolist(), num_tiles.tolist()):
            tiles_per_scale = image_tiles.split(counts)
            scale_maps = [
                merge_tiles(scale_tiles, *grid)
                for scale_tiles, grid in zip(tiles_per_scale, grids)
                if len(scale_tiles)
            ]
            # Scales without tiles use the features of the last scale
            scale_maps = scale_maps[-1:] * (len(grids) - len(scale_maps)) + scale_maps
            # The feature maps of all scales are resized to the size of the last one, and concatenated along channels
            size = scale_maps[-1].shape[-2:]
            scale_maps = [
                x
                if x.shape[-2:] == size
                else nn.functional.interpolate(x[None].float(), size=size, mode="area")[0].to(x.dtype)
                for x in scale_maps
            ]
            image_features.append(self._project_feature_maps(torch.cat(scale_maps)[None])[0])

        vision_outputs.pooler_output = image_features
        return vision_outputs

    @can_return_tuple
    @select_hidden_states(lambda self: self.config.vision_feature_layer)
    @auto_docstring(
        custom_intro="Encodes video frames, pools them over time, then projects them to the language model space with their time embedding."
    )
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_timestamps: list[list[float]] | torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_videos, num_frames, num_channels, height, width)`):
            Video frames, as returned by [`AudioVisualFlamingoVideoProcessor`].
        video_timestamps (`list[list[float]]` or `torch.Tensor` of shape `(num_videos, num_frames)`):
            Time in seconds of each frame. The processor returns them as Python floats, so that casting the inputs to
            a lower precision doesn't affect them.
        """
        num_videos = pixel_values_videos.shape[0]
        vision_outputs, frames = self._get_vision_feature_maps(pixel_values_videos.flatten(0, 1), **kwargs)
        # Frames are encoded as single tiles, whose features are used for every S2 scale
        frames = self._project_feature_maps(frames.repeat(1, len(self.config.s2_scales), 1, 1))
        frames = temporal_pool(frames.unflatten(0, (num_videos, -1)), self.config.video_temporal_pool_size)
        timestamps = torch.as_tensor(video_timestamps, dtype=get_time_dtype(frames.device), device=frames.device)
        timestamps = temporal_pool(timestamps, self.config.video_temporal_pool_size)
        cos, sin = self.video_rotary_emb(timestamps[..., None], *frames.shape[1:3])
        vision_outputs.pooler_output = list(apply_rotary_time_emb(frames, cos, sin).flatten(1, 2))
        return vision_outputs

    @can_return_tuple
    @auto_docstring(
        custom_intro="Encodes audio in 30 second windows, then projects it to the language model space with its time embedding."
    )
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        audio_timestamps: list[list[float]] | torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_audios, num_mel_bins, num_frames)`):
            Log-mel features of each audio, padded to a multiple of the audio encoder window.
        input_features_mask (`torch.Tensor` of shape `(num_audios, num_frames)`):
            Mask of the mel frames whose embeddings are passed to the language model.
        audio_timestamps (`list[list[float]]` or `torch.Tensor` of shape `(num_audios, num_audio_tokens)`):
            Time in seconds of each audio token, with one token every 4 mel frames. The processor returns them as
            Python floats, so that casting the inputs to a lower precision doesn't affect them.
        """
        # Mel frames of a 30 second window, downsampled by 2 by the audio encoder convolutions
        window_size = 2 * self.audio_tower.config.max_source_positions
        windows = input_features.unflatten(-1, (-1, window_size)).transpose(1, 2)
        window_mask = input_features_mask.unflatten(-1, (-1, window_size)).any(-1)
        audio_outputs = self.audio_tower(windows[window_mask], return_dict=True, **kwargs)
        window_embeds = self.audio_projector(audio_outputs.last_hidden_state)
        input_features_mask = input_features_mask.to(window_embeds.device, torch.bool)

        audio_embeds = window_embeds.new_zeros(*window_mask.shape, *window_embeds.shape[1:])
        audio_embeds[window_mask.to(window_embeds.device)] = window_embeds
        audio_embeds = audio_embeds.flatten(1, 2)
        # Token times are rounded to float32, as in training
        timestamps = torch.as_tensor(audio_timestamps, dtype=torch.float32, device=audio_embeds.device)
        cos, sin = self.audio_rotary_emb(timestamps, 1, audio_embeds.shape[1])
        audio_embeds = apply_rotary_time_emb(audio_embeds, cos, sin)

        token_mask = input_features_mask[:, :: window_size // window_embeds.shape[1]]
        return BaseModelOutputWithPooling(
            **audio_outputs, pooler_output=[embeds[mask] for embeds, mask in zip(audio_embeds, token_mask)]
        )

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor | None, inputs_embeds: torch.Tensor, features: torch.Tensor, token_id: int
    ) -> torch.BoolTensor:
        """Returns the mask of the `token_id` placeholders, after checking that there is one per feature vector."""
        if input_ids is None:
            special_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_mask = special_mask.all(-1)
        else:
            special_mask = input_ids == token_id

        num_tokens = special_mask.sum()
        torch_compilable_check(
            num_tokens * inputs_embeds.shape[-1] == features.numel(),
            lambda: f"Got {features.shape[0]} feature vectors for {num_tokens} placeholder tokens of id {token_id}",
        )
        return special_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_hw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_timestamps: list[list[float]] | torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_timestamps: list[list[float]] | torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        image_grid_hw (`torch.LongTensor` of shape `(num_images, num_scales, 2)`, *optional*):
            Number of tile rows and columns of each S2 scale of each image.
        video_timestamps (`list[list[float]]` or `torch.Tensor` of shape `(num_videos, num_frames)`, *optional*):
            Time in seconds of each video frame.
        input_features_mask (`torch.Tensor` of shape `(num_audios, num_frames)`, *optional*):
            Mask of the mel frames whose embeddings are passed to the language model.
        audio_timestamps (`list[list[float]]` or `torch.Tensor` of shape `(num_audios, num_audio_tokens)`, *optional*):
            Time in seconds of each audio token.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        mm_encoder_outputs = dict(mm_encoder_outputs) if mm_encoder_outputs is not None else {}
        raw_inputs = {"image": pixel_values, "video": pixel_values_videos, "audio": input_features}
        for modality, raw_input in raw_inputs.items():
            if raw_input is not None and mm_encoder_outputs.get(modality) is not None:
                raise ValueError(f"You cannot specify both the raw {modality} inputs and their mm_encoder_outputs")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(pixel_values, image_grid_hw, return_dict=True)
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_timestamps, return_dict=True
            )
        if mm_encoder_outputs.get("audio") is None and input_features is not None:
            mm_encoder_outputs["audio"] = self.get_audio_features(
                input_features, input_features_mask, audio_timestamps, return_dict=True
            )

        for modality in ("image", "video", "audio"):
            if mm_encoder_outputs.get(modality) is not None:
                features = torch.cat(mm_encoder_outputs[modality].pooler_output)
                features = features.to(inputs_embeds.device, inputs_embeds.dtype)
                token_id = getattr(self.config, f"{modality}_token_id")
                special_mask = self.get_placeholder_mask(input_ids, inputs_embeds, features, token_id)
                inputs_embeds = inputs_embeds.masked_scatter(special_mask, features)

        return self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


@auto_docstring(
    custom_intro="""
    The AudioVisualFlamingo model for joint understanding of images, videos and audio, made of a SigLIP vision tower,
    a Whisper-style audio encoder, their projectors and a Qwen2 language model.
    """
)
class AudioVisualFlamingoForConditionalGeneration(LlavaForConditionalGeneration):
    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_hw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        image_grid_hw (`torch.LongTensor` of shape `(num_images, num_scales, 2)`):
            Number of tile rows and columns of each S2 scale of each image.
        """
        return self.model.get_image_features(pixel_values, image_grid_hw, **kwargs)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_timestamps: list[list[float]] | torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        video_timestamps (`list[list[float]]` or `torch.Tensor` of shape `(num_videos, num_frames)`):
            Time in seconds of each frame.
        """
        return self.model.get_video_features(pixel_values_videos, video_timestamps, **kwargs)

    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
        audio_timestamps: list[list[float]] | torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features_mask (`torch.Tensor` of shape `(num_audios, num_frames)`):
            Mask of the mel frames whose embeddings are passed to the language model.
        audio_timestamps (`list[list[float]]` or `torch.Tensor` of shape `(num_audios, num_audio_tokens)`):
            Time in seconds of each audio token.
        """
        return self.model.get_audio_features(input_features, input_features_mask, audio_timestamps, **kwargs)

    def _prepare_multimodal_encoder_kwargs_for_generation(self, model_kwargs):
        # Audio is also encoded before generation, so that its features are expanded per prompt (e.g. for beam
        # search) like the image and video ones, rather than per audio
        model_kwargs = super()._prepare_multimodal_encoder_kwargs_for_generation(model_kwargs)
        if model_kwargs["mm_encoder_outputs"].get("audio") is None and model_kwargs.get("input_features") is not None:
            model_kwargs["mm_encoder_outputs"]["audio"] = self.get_audio_features(
                model_kwargs.pop("input_features"),
                model_kwargs.pop("input_features_mask"),
                model_kwargs.pop("audio_timestamps"),
                return_dict=True,
            )
        return model_kwargs

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_hw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_timestamps: list[list[float]] | torch.Tensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_timestamps: list[list[float]] | torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        image_grid_hw (`torch.LongTensor` of shape `(num_images, num_scales, 2)`, *optional*):
            Number of tile rows and columns of each S2 scale of each image.
        video_timestamps (`list[list[float]]` or `torch.Tensor` of shape `(num_videos, num_frames)`, *optional*):
            Time in seconds of each video frame.
        input_features_mask (`torch.Tensor` of shape `(num_audios, num_frames)`, *optional*):
            Mask of the mel frames whose embeddings are passed to the language model.
        audio_timestamps (`list[list[float]]` or `torch.Tensor` of shape `(num_audios, num_audio_tokens)`, *optional*):
            Time in seconds of each audio token.

        Example:

        ```python
        >>> from transformers import AudioVisualFlamingoForConditionalGeneration, AutoProcessor

        >>> model_id = "nvidia/nemotron-labs-audio-visual-flamingo-hf"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = AudioVisualFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

        >>> conversation = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"},
        ...             {"type": "text", "text": "Describe the video and its audio."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     conversation, add_generation_prompt=True, tokenize=True, return_dict=True, load_audio_from_video=True
        ... ).to(model.device)

        >>> outputs = model.generate(**inputs, max_new_tokens=64)
        >>> processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_hw=image_grid_hw,
            pixel_values_videos=pixel_values_videos,
            video_timestamps=video_timestamps,
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_timestamps=audio_timestamps,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        logits = self.lm_head(slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep))

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "AudioVisualFlamingoConfig",
    "AudioVisualFlamingoImageProcessor",
    "AudioVisualFlamingoForConditionalGeneration",
    "AudioVisualFlamingoModel",
    "AudioVisualFlamingoPreTrainedModel",
]
