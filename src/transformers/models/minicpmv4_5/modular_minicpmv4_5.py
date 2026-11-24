# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...configuration_utils import SubConfigSpec
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MultimodalBaseModelOutputWithPast,
    MultimodalCausalLMOutputWithPast,
)
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import ProcessorMixin, Unpack, VideosKwargs
from ...utils import TensorType, TransformersKwargs, auto_docstring, torch_compilable_check
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import can_return_tuple
from ...video_utils import VideoMetadata
from ..auto import AutoConfig
from ..minicpmv2_5.image_processing_minicpmv2_5 import find_best_resize
from ..minicpmv2_6.configuration_minicpmv2_6 import MiniCPMV2_6Config, MiniCPMV2_6VisionConfig
from ..minicpmv2_6.modeling_minicpmv2_6 import (
    MiniCPMV2_6ForConditionalGeneration,
    MiniCPMV2_6Model,
    MiniCPMV2_6PreTrainedModel,
    MiniCPMV2_6Resampler,
    MiniCPMV2_6VisionTransformer,
)
from ..minicpmv2_6.processing_minicpmv2_6 import MiniCPMV2_6Processor
from ..minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor, ensure_divide


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4_5")
@strict
class MiniCPMV4_5VisionConfig(MiniCPMV2_6VisionConfig):
    model_type = "minicpmv4_5_vision"


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4_5")
@strict
class MiniCPMV4_5Config(MiniCPMV2_6Config):
    r"""
    resampler_n_latents (`int`, *optional*, defaults to 64):
        Number of learned latent queries of the resampler, i.e. the number of tokens each image slice becomes.
    resampler_n_heads (`int`, *optional*, defaults to 32):
        Number of attention heads of the resampler's cross-attention.
    image_token_id (`int`, *optional*, defaults to 128244):
        Token id of the image placeholder (`<unk>`), repeated `resampler_n_latents` times per image slice.
    video_token_id (`int`, *optional*, defaults to 151656):
        Token id of the video placeholder (`<|video_pad|>`), repeated `resampler_n_latents` times per group of frames.

    Example:

    ```python
    >>> from transformers import MiniCPMV4_5Config, MiniCPMV4_5ForConditionalGeneration

    >>> configuration = MiniCPMV4_5Config()
    >>> model = MiniCPMV4_5ForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "minicpmv4_5"
    sub_configs_defaults = {
        "vision_config": MiniCPMV4_5VisionConfig,
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="qwen3",
            init_kwargs={
                "vocab_size": 151748,
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 40960,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
                "bos_token_id": 151643,
                "eos_token_id": 151645,
            },
        ),
    }

    resampler_n_heads: int = 32
    video_token_id: int = 151656


class MiniCPMV4_5VideoProcessorKwargs(VideosKwargs, total=False):
    r"""
    max_num_frames (`int`, *optional*, defaults to 180):
        Maximum number of frames sampled per video before packing.
    max_num_packing (`int`, *optional*, defaults to 3):
        Maximum number of consecutive frames packed into one group, which the resampler turns into one set of tokens.
    time_scale (`float`, *optional*, defaults to 0.1):
        Resolution, in seconds, of the frame timestamps given to the resampler.
    max_slice_nums (`int`, *optional*, defaults to 1):
        Maximum number of slices each frame is split into.
    scale_resolution (`int`, *optional*, defaults to 448):
        Target resolution for individual slices.
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    slice_mode (`bool`, *optional*, defaults to `True`):
        Whether to split frames into multiple slices for higher resolution.
    """

    max_num_frames: int
    max_num_packing: int
    time_scale: float
    max_slice_nums: int
    scale_resolution: int
    patch_size: int
    slice_mode: bool


@auto_docstring
class MiniCPMV4_5VideoProcessor(MiniCPMV4_6VideoProcessor):
    # Frames are sliced like MiniCPM-V 2.5 images (sizes rounded to the patch size); groups of consecutive frames are
    # packed so that the resampler compresses each group, with each frame's timestamp, into one set of tokens.
    max_slice_nums = 1
    fps = 5
    max_num_frames = 180
    max_num_packing = 3
    time_scale = 0.1
    downsample_mode = AttributeError()
    use_image_id = AttributeError()
    stack_frames = AttributeError()
    valid_kwargs = MiniCPMV4_5VideoProcessorKwargs
    model_input_names = [
        "pixel_values_videos",
        "pixel_attention_mask_videos",
        "patch_position_ids_videos",
        "temporal_ids",
    ]

    def __init__(self, **kwargs: Unpack[MiniCPMV4_5VideoProcessorKwargs]):
        super().__init__(**kwargs)

    def concat_frames_as_image(self, video):
        raise AttributeError("Not needed for MiniCPM-V 4.5")

    def find_best_resize(self, image_size, scale_resolution, patch_size, allow_upscale=False, **kwargs):
        return find_best_resize(image_size, scale_resolution, patch_size, allow_upscale)

    def get_refine_size(self, image_size, grid, scale_resolution, patch_size, allow_upscale=False, **kwargs):
        height, width = image_size
        grid_y, grid_x = grid
        best_height, best_width = self.find_best_resize(
            (ensure_divide(height, grid_y) / grid_y, ensure_divide(width, grid_x) / grid_x),
            scale_resolution,
            patch_size,
            allow_upscale,
        )
        return best_height * grid_y, best_width * grid_x

    def get_num_packing(self, duration: float, fps: float, max_num_frames: int, max_num_packing: int) -> int:
        """Number of consecutive frames packed into one group, as chosen by the original `encode_video`."""
        if fps * int(duration) <= max_num_frames:
            return 1
        return min(math.ceil(duration * fps / max_num_frames), max_num_packing)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: float | None = None,
        max_num_frames: int | None = None,
        max_num_packing: int | None = None,
        **kwargs,
    ):
        """
        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            fps (`float`, *optional*):
                Frames per second to sample.
            max_num_frames (`int`, *optional*):
                Maximum number of frames sampled before packing.
            max_num_packing (`int`, *optional*):
                Maximum number of frames packed into one group.
        Returns:
            list[int]: Indices to sample video frames.
        """
        if metadata is None or metadata.duration is None or metadata.fps is None:
            raise ValueError(
                "MiniCPMV4_5 requires video metadata with `duration` and `fps` to sample frames. "
                "Please pass a complete `VideoMetadata` object or set `do_sample_frames=False`."
            )
        fps = fps if fps is not None else self.fps
        max_num_frames = max_num_frames if max_num_frames is not None else self.max_num_frames
        max_num_packing = max_num_packing if max_num_packing is not None else self.max_num_packing

        num_packing = self.get_num_packing(metadata.duration, fps, max_num_frames, max_num_packing)
        if num_packing == 1:
            num_frames = round(min(fps, round(metadata.fps)) * min(max_num_frames, metadata.duration))
        elif metadata.duration * fps / max_num_frames <= max_num_packing:
            num_frames = round(metadata.duration * fps)
        else:
            num_frames = round(max_num_frames * max_num_packing)
        gap = metadata.total_num_frames / num_frames
        return [int(i * gap + gap / 2) for i in range(num_frames)]

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_slice_nums: int,
        scale_resolution: int,
        patch_size: int,
        slice_mode: bool,
        max_num_frames: int,
        max_num_packing: int,
        time_scale: float,
        video_metadata: list[VideoMetadata],
        fps: float | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        fps = fps if fps is not None else self.fps
        slices, temporal_ids, grids, num_groups = [], [], [], []
        for video, metadata in zip(videos, video_metadata):
            # Timestamps on the original `time_scale` grid (frames without metadata are taken as sampled at `fps`),
            # then groups of consecutive frames
            video_fps = metadata.fps or fps
            duration = metadata.duration or metadata.total_num_frames / video_fps
            timestamps = np.asarray(metadata.frames_indices) / video_fps
            scale = np.arange(0, duration, time_scale)
            frame_ids = (scale[np.abs(scale[None, :] - timestamps[:, None]).argmin(-1)] / time_scale).astype(np.int32)
            num_packing = self.get_num_packing(duration, fps, max_num_frames, max_num_packing)

            frames, grid, _ = self.resize_and_split_patches(
                video[None], resample, slice_mode, max_slice_nums, scale_resolution, patch_size
            )
            frames = [frame[0] for frame in frames[0]]  # num_frames * (overview + slices), frame-major
            num_slices = (grid[0][0][0] * grid[0][0][1]) + 1
            grouped, grouped_index = group_images_by_shape(frames, disable_grouping=False)
            processed = {
                shape: self.rescale_and_normalize(
                    stacked.float(), do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                for shape, stacked in grouped.items()
            }
            frames = reorder_images(processed, grouped_index)

            # Group-major order: for each group, each slice index, the group's frames
            for start in range(0, len(frame_ids), num_packing):
                group_ids = frame_ids[start : start + num_packing].tolist()
                for slice_idx in range(num_slices):
                    for frame_idx in range(start, start + len(group_ids)):
                        slices.append(frames[frame_idx * num_slices + slice_idx])
                    temporal_ids.append(group_ids)
            grids.append(list(grid[0][0]))
            num_groups.append(math.ceil(len(frame_ids) / num_packing))

        # One row of patches per frame slice, padded to the longest, as for images
        sizes = [(s.shape[-2] // patch_size, s.shape[-1] // patch_size) for s in slices]
        rows = [s.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size) for s in slices]
        rows = [r.permute(0, 3, 1, 2, 4).reshape(r.shape[0], patch_size, -1) for r in rows]
        num_patches = [height * width for height, width in sizes]
        max_patches = max(num_patches)
        pixel_values = torch.stack([F.pad(r, (0, (max_patches - n) * patch_size)) for r, n in zip(rows, num_patches)])
        patch_index = torch.arange(max_patches)[None, :]
        pixel_attention_mask = patch_index < torch.tensor(num_patches)[:, None]
        widths = torch.tensor([width for _, width in sizes])[:, None]
        patch_position_ids = torch.stack([patch_index // widths, patch_index % widths], dim=-1)
        max_frames = max(len(ids) for ids in temporal_ids)
        temporal_ids = torch.tensor([ids + [-1] * (max_frames - len(ids)) for ids in temporal_ids])
        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values,
                "pixel_attention_mask_videos": pixel_attention_mask,
                "patch_position_ids_videos": patch_position_ids * pixel_attention_mask[..., None],
                "temporal_ids": temporal_ids,
                "grids_videos": grids,
                "num_groups_per_video": num_groups,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["grids_videos", "num_groups_per_video"],
        )


@auto_docstring
class MiniCPMV4_5Processor(MiniCPMV2_6Processor):
    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token: str = "<unk>",
        video_token: str = "<|video_pad|>",
        image_start_token: str = "<image>",
        image_end_token: str = "</image>",
        slice_start_token: str = "<slice>",
        slice_end_token: str = "</slice>",
        num_image_tokens: int = 64,
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<unk>"`):
            Marks an image in the text, and is repeated `num_image_tokens` times per image slice.
        video_token (`str`, *optional*, defaults to `"<|video_pad|>"`):
            Marks a video in the text, and is repeated `num_image_tokens` times per slice of each group of frames.
        image_start_token (`str`, *optional*, defaults to `"<image>"`):
            Opens the image tokens of the overview image.
        image_end_token (`str`, *optional*, defaults to `"</image>"`):
            Closes the image tokens of the overview image.
        slice_start_token (`str`, *optional*, defaults to `"<slice>"`):
            Opens the image tokens of one slice.
        slice_end_token (`str`, *optional*, defaults to `"</slice>"`):
            Closes the image tokens of one slice.
        num_image_tokens (`int`, *optional*, defaults to 64):
            Number of image tokens per slice (the resampler's number of latents).
        """
        self.image_token = image_token
        self.video_token = video_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.slice_start_token = slice_start_token
        self.slice_end_token = slice_end_token
        self.num_image_tokens = num_image_tokens
        ProcessorMixin.__init__(self, image_processor, video_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        # One placeholder per group of frames, laid out like an image, one line per group
        video_tokens = self.video_token * self.num_image_tokens
        num_rows, num_cols = video_inputs["grids_videos"][video_idx]
        rows = [(self.slice_start_token + video_tokens + self.slice_end_token) * num_cols for _ in range(num_rows)]
        group = self.image_start_token + video_tokens + self.image_end_token + "\n".join(rows)
        return "\n".join([group] * video_inputs["num_groups_per_video"][video_idx])

    @property
    def unused_input_names(self) -> list[str]:
        return ["grids", "grids_videos", "num_groups_per_video"]


class MiniCPMV4_5PreTrainedModel(MiniCPMV2_6PreTrainedModel):
    pass


class MiniCPMV4_5VisionTransformer(MiniCPMV2_6VisionTransformer):
    pass


class MiniCPMV4_5Resampler(MiniCPMV2_6Resampler):
    def get_temporal_pos_embed(self, temporal_ids: torch.Tensor) -> torch.Tensor:
        """1D sin-cos embedding of each key's frame timestamp, zero where `temporal_ids` is -1 (padding frames)."""
        half = self.embed_dim // 2
        omega = 1.0 / 10000 ** (torch.arange(half, device=temporal_ids.device, dtype=torch.float32) / half)
        angles = temporal_ids.clamp(min=0)[..., None].float() * omega
        return torch.cat([angles.sin(), angles.cos()], dim=-1) * (temporal_ids >= 0)[..., None]

    def forward(
        self,
        vision_features: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        patch_position_ids: torch.Tensor,
        temporal_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        kv = self.ln_kv(self.kv_proj(vision_features))
        pos_embed = self.get_2d_pos_embed(patch_position_ids)
        if temporal_ids is not None:
            pos_embed = pos_embed + self.get_temporal_pos_embed(temporal_ids)
        pos_embed = (pos_embed * pixel_attention_mask[..., None]).to(kv.dtype)
        queries = self.ln_q(self.query).unsqueeze(0).expand(kv.shape[0], -1, -1)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=queries,
            attention_mask=pixel_attention_mask,
            encoder_hidden_states=kv,
        )
        attn_output, _ = self.attn(
            queries, kv, attention_mask=attention_mask, encoder_position_embeddings=pos_embed, **kwargs
        )
        return self.ln_post(attn_output) @ self.proj


class MiniCPMV4_5Model(MiniCPMV2_6Model):
    @can_return_tuple
    @auto_docstring(
        custom_intro="Encodes video frame slices and resamples each group of frames to `config.resampler_n_latents` tokens."
    )
    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        pixel_attention_mask_videos: torch.Tensor,
        patch_position_ids_videos: torch.Tensor,
        temporal_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_frame_slices, num_channels, patch_size, max_patches * patch_size)`):
            Every frame slice as a single row of patches, padded to the longest, as returned by
            [`MiniCPMV4_5VideoProcessor`]: for each group of frames, for each slice, the group's frames.
        pixel_attention_mask_videos (`torch.BoolTensor` of shape `(num_frame_slices, max_patches)`):
            Which patches are real (`True`) rather than padding.
        patch_position_ids_videos (`torch.LongTensor` of shape `(num_frame_slices, max_patches, 2)`):
            `(row, col)` of each patch in its slice's patch grid.
        temporal_ids (`torch.LongTensor` of shape `(num_groups, max_frames)`):
            Timestamp of each frame of each (group, slice), in `time_scale` steps; -1 pads shorter groups.
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values_videos.to(self.vision_tower.dtype),
            patch_attention_mask=pixel_attention_mask_videos,
            patch_position_ids=patch_position_ids_videos,
            return_dict=True,
            **kwargs,
        )
        # The resampler attends to all frames of a group at once: (num_groups, max_frames * max_patches, ...)
        is_frame = temporal_ids >= 0
        grouped = []
        for frame_values in (vision_outputs.last_hidden_state, pixel_attention_mask_videos, patch_position_ids_videos):
            values = frame_values.new_zeros(*is_frame.shape, *frame_values.shape[1:])
            values[is_frame] = frame_values
            grouped.append(values.flatten(1, 2))
        frame_ids = temporal_ids[..., None].expand(-1, -1, pixel_attention_mask_videos.shape[-1]).flatten(1)
        vision_outputs.pooler_output = self.resampler(*grouped, temporal_ids=frame_ids)
        return vision_outputs

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, features: torch.Tensor, token_id: int
    ) -> torch.Tensor:
        """
        Obtains the placeholder mask of `token_id` from `input_ids` or `inputs_embeds`, and checks that the placeholder
        token count matches the length of `features`.
        """
        if input_ids is None:
            special_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_mask = special_mask.all(-1)
        else:
            special_mask = input_ids == token_id
        n_tokens = special_mask.sum()
        special_mask = special_mask.unsqueeze(-1).to(inputs_embeds.device)
        torch_compilable_check(
            n_tokens * inputs_embeds.shape[-1] == features.numel(),
            f"Multimodal features and tokens do not match, tokens: {n_tokens}, features: {features.shape[0] * features.shape[1]}",
        )
        return special_mask

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        patch_position_ids: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        pixel_attention_mask_videos: torch.Tensor | None = None,
        patch_position_ids_videos: torch.Tensor | None = None,
        temporal_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalBaseModelOutputWithPast:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
            Which patches of `pixel_values` are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its slice's patch grid.
        pixel_attention_mask_videos (`torch.BoolTensor` of shape `(num_frame_slices, max_patches)`, *optional*):
            Which patches of `pixel_values_videos` are real (`True`) rather than padding.
        patch_position_ids_videos (`torch.LongTensor` of shape `(num_frame_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its frame slice's patch grid.
        temporal_ids (`torch.LongTensor` of shape `(num_groups, max_frames)`, *optional*):
            Timestamp of each frame of each (group, slice), in `time_scale` steps; -1 pads shorter groups.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                patch_position_ids=patch_position_ids,
                return_dict=True,
                **kwargs,
            )
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos=pixel_values_videos,
                pixel_attention_mask_videos=pixel_attention_mask_videos,
                patch_position_ids_videos=patch_position_ids_videos,
                temporal_ids=temporal_ids,
                return_dict=True,
                **kwargs,
            )

        for modality, token_id in (("image", self.config.image_token_id), ("video", self.config.video_token_id)):
            if mm_encoder_outputs.get(modality) is not None:
                features = mm_encoder_outputs[modality].pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
                mask = self.get_placeholder_mask(input_ids, inputs_embeds, features, token_id)
                inputs_embeds = inputs_embeds.masked_scatter(mask, features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return MultimodalBaseModelOutputWithPast(
            **outputs, image_outputs=mm_encoder_outputs.get("image"), video_outputs=mm_encoder_outputs.get("video")
        )


class MiniCPMV4_5ForConditionalGeneration(MiniCPMV2_6ForConditionalGeneration):
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        pixel_attention_mask_videos: torch.Tensor,
        patch_position_ids_videos: torch.Tensor,
        temporal_ids: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_frame_slices, num_channels, patch_size, max_patches * patch_size)`):
            Every frame slice as a single row of patches, padded to the longest.
        pixel_attention_mask_videos (`torch.BoolTensor` of shape `(num_frame_slices, max_patches)`):
            Which patches are real (`True`) rather than padding.
        patch_position_ids_videos (`torch.LongTensor` of shape `(num_frame_slices, max_patches, 2)`):
            `(row, col)` of each patch in its slice's patch grid.
        temporal_ids (`torch.LongTensor` of shape `(num_groups, max_frames)`):
            Timestamp of each frame of each (group, slice), in `time_scale` steps; -1 pads shorter groups.
        """
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            pixel_attention_mask_videos=pixel_attention_mask_videos,
            patch_position_ids_videos=patch_position_ids_videos,
            temporal_ids=temporal_ids,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        patch_position_ids: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        pixel_attention_mask_videos: torch.Tensor | None = None,
        patch_position_ids_videos: torch.Tensor | None = None,
        temporal_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalCausalLMOutputWithPast:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
            Which patches of `pixel_values` are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its slice's patch grid.
        pixel_attention_mask_videos (`torch.BoolTensor` of shape `(num_frame_slices, max_patches)`, *optional*):
            Which patches of `pixel_values_videos` are real (`True`) rather than padding.
        patch_position_ids_videos (`torch.LongTensor` of shape `(num_frame_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its frame slice's patch grid.
        temporal_ids (`torch.LongTensor` of shape `(num_groups, max_frames)`, *optional*):
            Timestamp of each frame of each (group, slice), in `time_scale` steps; -1 pads shorter groups.

        Example:

        ```python
        >>> from transformers import AutoModelForImageTextToText, AutoProcessor

        >>> model = AutoModelForImageTextToText.from_pretrained("openbmb/MiniCPM-V-4_5", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-4_5")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/sample_demo_1.mp4"},
        ...             {"type": "text", "text": "Describe the video."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     add_generation_prompt=True,
        ...     enable_thinking=False,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ... ).to(model.device)
        >>> generated = model.generate(**inputs, max_new_tokens=32)
        >>> processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0]
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            patch_position_ids=patch_position_ids,
            pixel_values_videos=pixel_values_videos,
            pixel_attention_mask_videos=pixel_attention_mask_videos,
            patch_position_ids_videos=patch_position_ids_videos,
            temporal_ids=temporal_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs[0], logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
        return MultimodalCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_outputs=outputs.image_outputs,
            video_outputs=outputs.video_outputs,
        )


__all__ = [
    "MiniCPMV4_5Config",
    "MiniCPMV4_5VisionConfig",
    "MiniCPMV4_5VideoProcessor",
    "MiniCPMV4_5Processor",
    "MiniCPMV4_5PreTrainedModel",
    "MiniCPMV4_5VisionTransformer",
    "MiniCPMV4_5Model",
    "MiniCPMV4_5ForConditionalGeneration",
]
