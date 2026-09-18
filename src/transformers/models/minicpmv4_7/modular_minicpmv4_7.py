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


import itertools
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import ProcessingKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ...video_utils import VideoInput, make_batched_videos
from ..auto import AutoConfig
from ..minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6Config, MiniCPMV4_6VisionConfig
from ..minicpmv4_6.image_processing_minicpmv4_6 import MiniCPMV4_6ImageProcessor, MiniCPMV4_6ImageProcessorKwargs
from ..minicpmv4_6.image_processing_pil_minicpmv4_6 import MiniCPMV4_6ImageProcessorPil
from ..minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6Model,
    MiniCPMV4_6PreTrainedModel,
    MiniCPMV4_6ViTWindowAttentionMerger,
)
from ..minicpmv4_6.processing_minicpmv4_6 import MiniCPMV4_6Processor, MiniCPMV4_6ProcessorKwargs
from ..minicpmv4_6.video_processing_minicpmv4_6 import MiniCPMV4_6VideoProcessor, MiniCPMV4_6VideoProcessorKwargs


logger = logging.get_logger(__name__)


def _crop_end(crop, input_ids: torch.LongTensor, structural_ids: set, limit: int) -> int:
    """End of a frame's span, including the ``</image>``/``</slice>`` markers that close it."""
    end = crop["slices"][-1][1] if crop["slices"] else crop["thumbnail"][1]
    while end < limit and input_ids[end].item() in structural_ids:
        end += 1
    return end


def _group_visual_frames(
    input_ids: torch.LongTensor, mm_token_type_ids: torch.IntTensor, special_token_ids: dict
) -> list[list[dict]]:
    """Split one sequence into visual groups, each a list of ``thumbnail + slices`` frames.

    ``mm_token_type_ids`` labels every token with its modality (``0`` text, ``1`` image, ``2``
    video), so each maximal non-text run is exactly one crop. A crop is a slice when ``<slice>``
    sits in front of it, otherwise it opens a new frame and adopts the slices that follow.

    Frames separated by nothing but markers and newlines -- how a clip emits its frames, and how a
    picture can end up sitting right against one -- share a group. Grouping them matters: the next
    frame then starts one step past the tokens in between, so its halo cannot land on the last of
    them.
    """
    slice_start_id = special_token_ids["slice_start_id"]
    gap_ids = {
        special_token_ids["im_start_id"],
        special_token_ids["im_end_id"],
        slice_start_id,
        special_token_ids["slice_end_id"],
        special_token_ids["newline_id"],
    }

    groups = []
    crop_index = 0
    previous_end = 0
    for modality, run in itertools.groupby(enumerate(mm_token_type_ids.tolist()), lambda item: item[1]):
        if modality == 0:
            continue
        run = list(run)
        crop = (run[0][0], run[-1][0] + 1, crop_index)
        crop_index += 1

        if groups and input_ids[crop[0] - 1].item() == slice_start_id:
            groups[-1][-1]["slices"].append(crop)
        else:
            frame = {"thumbnail": crop, "slices": []}
            adjacent = groups and all(
                input_ids[position].item() in gap_ids for position in range(previous_end, crop[0] - 1)
            )
            if adjacent:
                groups[-1].append(frame)
            else:
                groups.append([frame])
        previous_end = crop[1]
    return groups


def _compute_canvas(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    target_sizes: torch.IntTensor,
    special_token_ids: dict,
    merge_factor: int,
) -> torch.LongTensor:
    """Lay one unpadded sequence out on the canvas and return its ``(3, seq_len)`` positions.

    Text runs keep plain 1-D positions on all three channels. Every frame freezes the temporal
    channel and spends the height/width channels on its own canvas, which is the slice grid when
    the crop was sliced and the thumbnail grid otherwise. The next frame starts one step past the
    widest side of the previous canvas, so images, slices and video frames never overlap.
    """
    device = input_ids.device
    seq_len = input_ids.shape[0]
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).expand(3, -1).clone()
    structural_ids = {
        special_token_ids["im_start_id"],
        special_token_ids["im_end_id"],
        special_token_ids["slice_start_id"],
        special_token_ids["slice_end_id"],
    }

    pos = 0
    cursor = 0
    for frames in _group_visual_frames(input_ids, mm_token_type_ids, special_token_ids):
        group_start = frames[0]["thumbnail"][0] - 1
        group_end = _crop_end(frames[-1], input_ids, structural_ids, seq_len)

        # Text in front of the group is plain 1-D.
        if group_start > cursor:
            text_len = group_start - cursor
            position_ids[:, cursor:group_start] = torch.arange(text_len, device=device) + pos
            pos += text_len

        frame_cursor = group_start
        for frame in frames:
            thumb_start, thumb_end, thumb_index = frame["thumbnail"]
            slices = frame["slices"]
            frame_start = thumb_start - 1

            # Tokens between two frames of a clip are 1-D text; the extra step afterwards keeps the
            # next frame's halo from landing on the last of them.
            if frame_start > frame_cursor:
                gap_len = frame_start - frame_cursor
                position_ids[:, frame_cursor:frame_start] = torch.arange(gap_len, device=device) + pos
                pos += gap_len + 1

            frame_end = min(_crop_end(frame, input_ids, structural_ids, group_end), group_end)
            canvas_origin = pos
            halo_before_canvas = max(canvas_origin - 1, 0)

            llm_thumb_h = target_sizes[thumb_index, 0].item() // merge_factor
            llm_thumb_w = target_sizes[thumb_index, 1].item() // merge_factor

            if slices:
                # Slices are laid out row-major; a gap wider than the two `</slice><slice>` markers
                # is the newline that ends a row and therefore fixes the column count.
                num_cols = len(slices)
                for k in range(len(slices) - 1):
                    if slices[k + 1][0] - slices[k][1] > 2:
                        num_cols = k + 1
                        break
                num_rows = len(slices) // num_cols if num_cols > 0 else 1
                if num_rows * num_cols != len(slices):
                    num_rows, num_cols = 1, len(slices)
                llm_slice_h = target_sizes[slices[0][2], 0].item() // merge_factor
                llm_slice_w = target_sizes[slices[0][2], 1].item() // merge_factor
                canvas_height = num_rows * llm_slice_h
                canvas_width = num_cols * llm_slice_w
            else:
                llm_slice_h, llm_slice_w = 0, 0
                num_rows, num_cols = 0, 0
                canvas_height = llm_thumb_h
                canvas_width = llm_thumb_w

            # Base coat for the whole frame, then `<image>` -> halo just outside the canvas.
            position_ids[:, frame_start:frame_end] = canvas_origin
            position_ids[1:, frame_start] = halo_before_canvas

            # `</image>` closes the thumbnail on the far corner of the canvas.
            image_end_pos = thumb_end
            if image_end_pos < frame_end:
                position_ids[1, image_end_pos] = canvas_origin + canvas_height
                position_ids[2, image_end_pos] = canvas_origin + canvas_width

            # Thumbnail tokens. With slices around, the thumbnail is stretched over the full canvas
            # so that it stays aligned with the detail crops underneath it. Without slices the canvas
            # is the thumbnail grid itself and `linspace` degenerates to `arange`.
            h_coords = torch.linspace(0, canvas_height - 1, llm_thumb_h, device=device).round().long()
            w_coords = torch.linspace(0, canvas_width - 1, llm_thumb_w, device=device).round().long()
            h_idx = h_coords.view(-1, 1).expand(-1, llm_thumb_w).reshape(-1)
            w_idx = w_coords.view(1, -1).expand(llm_thumb_h, -1).reshape(-1)
            position_ids[0, thumb_start:thumb_end] = canvas_origin
            position_ids[1, thumb_start:thumb_end] = h_idx + canvas_origin
            position_ids[2, thumb_start:thumb_end] = w_idx + canvas_origin

            # Slice tokens plus the `<slice>`/`</slice>` markers that pin each crop's corners.
            for k, (slice_start, slice_end, slice_index) in enumerate(slices):
                slice_h = target_sizes[slice_index, 0].item() // merge_factor
                slice_w = target_sizes[slice_index, 1].item() // merge_factor
                h_off = (k // num_cols) * llm_slice_h
                w_off = (k % num_cols) * llm_slice_w

                slice_start_pos = slice_start - 1
                if slice_start_pos >= frame_start:
                    position_ids[1, slice_start_pos] = canvas_origin + h_off
                    position_ids[2, slice_start_pos] = canvas_origin + w_off

                slice_end_pos = slice_end
                if slice_end_pos < frame_end:
                    position_ids[1, slice_end_pos] = canvas_origin + h_off + slice_h - 1
                    position_ids[2, slice_end_pos] = canvas_origin + w_off + slice_w - 1

                h_idx = torch.arange(slice_h, device=device).view(-1, 1).expand(-1, slice_w).reshape(-1)
                w_idx = torch.arange(slice_w, device=device).view(1, -1).expand(slice_h, -1).reshape(-1)
                position_ids[0, slice_start:slice_end] = canvas_origin
                position_ids[1, slice_start:slice_end] = h_idx + h_off + canvas_origin
                position_ids[2, slice_start:slice_end] = w_idx + w_off + canvas_origin

            # The "\n" that ends a row of slices sits just past the right edge of that row.
            for k in range(len(slices) - 1):
                gap_start, gap_end = slices[k][1], slices[k + 1][0]
                if gap_end - gap_start <= 2:
                    continue
                boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                right_edge_w = num_cols * llm_slice_w
                for newline_pos in range(gap_start + 1, gap_end - 1):
                    position_ids[1, newline_pos] = canvas_origin + boundary_h
                    position_ids[2, newline_pos] = canvas_origin + right_edge_w

            pos = canvas_origin + max(canvas_height, canvas_width) + 1
            frame_cursor = frame_end

        if frame_cursor < group_end:
            trail_len = group_end - frame_cursor
            position_ids[:, frame_cursor:group_end] = torch.arange(trail_len, device=device) + pos
            pos += trail_len
        cursor = group_end

    if cursor < seq_len:
        position_ids[:, cursor:seq_len] = torch.arange(seq_len - cursor, device=device) + pos
    return position_ids


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4.7")
@strict
class MiniCPMV4_7VisionConfig(MiniCPMV4_6VisionConfig):
    r"""
    insert_layer_id (`int`, *optional*, defaults to 6):
        Vision encoder layer index after which the window-attention merger is applied.
    window_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`):
        Window size `(h, w)` for the intermediate window-attention merger.
    """

    model_type = "minicpmv4_7_vision"


class MiniCPMV4_7ViTWindowAttentionMerger(MiniCPMV4_6ViTWindowAttentionMerger):
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_sizes: torch.IntTensor,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        device = hidden_states.device

        window_index, window_cu_seqlens, window_max_seqlens = self.get_window_index(target_sizes, kwargs=kwargs)
        window_index = window_index.to(device)

        hidden_states = hidden_states[:, window_index, :]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=window_cu_seqlens.to(device),
            max_seqlen=window_max_seqlens,
        )
        hidden_states = hidden_states[:, torch.argsort(window_index), :]
        hidden_states = residual + hidden_states

        batch_size, _ = target_sizes.shape
        window_h, window_w = self.window_kernel_size
        cu_seqlens = F.pad(
            torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32).to(device), (1, 0)
        )
        all_patches = []
        for batch_idx in range(batch_size):
            height = int(target_sizes[batch_idx, 0])
            width = int(target_sizes[batch_idx, 1])
            patch = hidden_states[0, cu_seqlens[batch_idx] : cu_seqlens[batch_idx + 1], :]

            embed_dim = patch.shape[-1]
            merged_h, merged_w = height // window_h, width // window_w
            patch_5d = patch.view(merged_h, window_h, merged_w, window_w, embed_dim).permute(0, 2, 1, 3, 4)
            hidden_state = patch_5d.reshape(merged_h * merged_w, window_h * window_w * embed_dim)
            patch_residual = patch_5d.reshape(merged_h * merged_w, window_h * window_w, embed_dim).mean(dim=1)

            hidden_state = self.pre_norm(hidden_state)
            hidden_state = self.linear_1(hidden_state)
            hidden_state = self.act(hidden_state)
            hidden_state = self.linear_2(hidden_state)

            all_patches.append(hidden_state + patch_residual)

        return torch.concat(all_patches, dim=0).unsqueeze(0)


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4.7")
@strict
class MiniCPMV4_7Config(MiniCPMV4_6Config):
    r"""
    insert_layer_id (`int`, *optional*, defaults to 6):
        Vision encoder layer index after which the window-attention merger is applied.
    image_size (`int`, *optional*, defaults to 448):
        Base resolution for image preprocessing.
    drop_vision_last_layer (`bool`, *optional*, defaults to `False`):
        Whether to drop the last layer of the vision encoder.
    image_token_id (`int`, *optional*):
        Token id used as the image placeholder.
    video_token_id (`int`, *optional*):
        Token id used as the video placeholder.
    downsample_mode (`str`, *optional*, defaults to `"16x"`):
        Visual token downsampling ratio. `"4x"` keeps 4× more tokens.
    merge_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`):
        Kernel size `(h, w)` for merging adjacent visual patches in the Merger.
    merger_times (`int`, *optional*, defaults to 1):
        Number of iterative merge rounds in the Merger.
    image_start_id (`int`, *optional*):
        Token id of the image-start marker (`<image>`) used by canvas M-RoPE. Resolved from the
        tokenizer by the conversion script and stored in `config.json`. Required for any
        checkpoint that is used with images or videos.
    image_end_id (`int`, *optional*):
        Token id of the image-end marker (`</image>`) used by canvas M-RoPE. See `image_start_id`.
    slice_start_id (`int`, *optional*):
        Token id of the slice-start marker (`<slice>`) used by canvas M-RoPE. See `image_start_id`.
    slice_end_id (`int`, *optional*):
        Token id of the slice-end marker (`</slice>`) used by canvas M-RoPE. See `image_start_id`.
    newline_id (`int`, *optional*):
        Token id of the newline (`"\n"`) separating slice rows for canvas M-RoPE. See
        `image_start_id`.
    """

    model_type = "minicpmv4_7"
    sub_configs = {"text_config": AutoConfig, "vision_config": MiniCPMV4_7VisionConfig}

    image_start_id: int | None = None
    image_end_id: int | None = None
    slice_start_id: int | None = None
    slice_end_id: int | None = None
    newline_id: int | None = None

    def get_mrope_special_token_ids(self) -> dict:
        return {
            "im_start_id": self.image_start_id,
            "im_end_id": self.image_end_id,
            "slice_start_id": self.slice_start_id,
            "slice_end_id": self.slice_end_id,
            "newline_id": self.newline_id,
        }


@auto_docstring
class MiniCPMV4_7PreTrainedModel(MiniCPMV4_6PreTrainedModel):
    config_class = MiniCPMV4_7Config


class MiniCPMV4_7Model(MiniCPMV4_6Model):
    def __init__(self, config: MiniCPMV4_7Config):
        super().__init__(config)
        self.rope_deltas = None

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        target_sizes_mrope: torch.IntTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        downsample_mode: str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canvas M-RoPE indices ``(3, B, S)`` plus ``rope_deltas``.

        ``mm_token_type_ids`` marks the crops, so the canvas is laid out per sequence on the
        tokens ``attention_mask`` keeps. Padding stays at zero and never shifts the canvas.
        """
        del kwargs
        special_token_ids = self.config.get_mrope_special_token_ids()
        # `16x` merges 4x4 patches into one LLM token (window merger 2x2, then merger 2x2), `4x`
        # skips the window merger and merges 2x2. Same divisor the processor counts placeholders
        # with, so it has to follow the per-call override the vision tower is given, not the config.
        downsample_mode = downsample_mode if downsample_mode else self.config.downsample_mode
        merge_factor = 2 if downsample_mode == "4x" else 4

        batch_size, seq_len = input_ids.shape
        position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=input_ids.device)

        for batch_idx in range(batch_size):
            if attention_mask is not None:
                valid_mask = attention_mask[batch_idx].bool()
            else:
                valid_mask = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)

            position_ids[:, batch_idx, valid_mask] = _compute_canvas(
                input_ids[batch_idx][valid_mask],
                mm_token_type_ids[batch_idx][valid_mask],
                target_sizes_mrope[batch_idx],
                special_token_ids,
                merge_factor,
            )

        if attention_mask is not None:
            seq_lens = attention_mask.sum(-1)
        else:
            seq_lens = torch.full((input_ids.shape[0],), input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        deltas = position_ids.amax(dim=(0, 2)).unsqueeze(1) + 1 - seq_lens.unsqueeze(1)
        return position_ids, deltas.long()

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        target_sizes_mrope: torch.IntTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        downsample_mode: str | None = None,
    ) -> torch.Tensor | None:
        """Return the ``(3, batch, seq)`` canvas M-RoPE positions ``[T, H, W]`` (Qwen-style entrypoint).

        Generation prepends the text channel in ``_prepare_position_ids_for_generation``; the text
        model reads that fourth row only when it is there.
        """
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()

        if (
            input_ids is not None
            and target_sizes_mrope is not None
            and (self.rope_deltas is None or past_key_values_length == 0)
        ):
            mrope_position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                attention_mask=attention_mask,
                target_sizes_mrope=target_sizes_mrope,
                mm_token_type_ids=mm_token_type_ids,
                downsample_mode=downsample_mode,
            )
            self.rope_deltas = rope_deltas
            return mrope_position_ids
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            # Can't build correct 3D positions. Let the model infer it as 1D
            if target_sizes_mrope is not None:
                logger.warning_once(
                    "Canvas M-RoPE needs `input_ids` to locate the visual spans, but only `inputs_embeds` was "
                    "given and no cached `rope_deltas` are available. Falling back to 1-D positions, which "
                    "degrades multimodal quality. Pass `input_ids` for the first forward pass."
                )
            position_ids = None
        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        target_sizes: torch.IntTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        target_sizes_videos: torch.IntTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        downsample_mode: str | None = None,
        target_sizes_mrope: torch.IntTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel value patches for images, NaViT-packed.
        target_sizes (`torch.IntTensor`, *optional*):
            Height and width (in patches) for each image.
        pixel_values_videos (`torch.FloatTensor`, *optional*):
            Pixel value patches for video frames, NaViT-packed.
        target_sizes_videos (`torch.IntTensor`, *optional*):
            Height and width (in patches) for each video frame.
        downsample_mode (`str`, *optional*):
            `"4x"` keeps 4x more visual tokens; default `"16x"` applies full merge.
        target_sizes_mrope (`torch.IntTensor` of shape `(batch_size, num_visuals, 2)`, *optional*):
            Spatial grid sizes (height, width in patches) per visual crop for canvas M-RoPE.
        mm_token_type_ids (`torch.IntTensor`, *optional*):
            Modality type ids (`0` text, `1` image, `2` video), matching the Qwen processor
            contract. Required together with `target_sizes_mrope`: canvas M-RoPE reads the crops
            off it.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            num_beams = pixel_values.shape[0]
            vision_output = self.get_image_features(pixel_values[:1], target_sizes, downsample_mode=downsample_mode)
            image_features = (
                torch.cat(vision_output.pooler_output, dim=0)
                .to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                .repeat(num_beams, 1)
            )
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features, self.config.image_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)

        if pixel_values_videos is not None:
            num_beams = pixel_values_videos.shape[0]
            vision_output = self.get_video_features(
                pixel_values_videos[:1], target_sizes_videos, downsample_mode=downsample_mode
            )
            video_features = (
                torch.cat(vision_output.pooler_output, dim=0)
                .to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                .repeat(num_beams, 1)
            )
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features, self.config.video_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(mask, video_features)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                target_sizes_mrope=target_sizes_mrope,
                mm_token_type_ids=mm_token_type_ids,
                downsample_mode=downsample_mode,
            )

        output = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return output


@auto_docstring
class MiniCPMV4_7ForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
    def __init__(self, config: MiniCPMV4_7Config):
        # Parent would build a MiniCPMV4_6Model; bypass it to build the 4.7 model instead.
        MiniCPMV4_7PreTrainedModel.__init__(self, config)
        self.model = MiniCPMV4_7Model(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        target_sizes: torch.IntTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        target_sizes_videos: torch.IntTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        downsample_mode: str | None = None,
        target_sizes_mrope: torch.IntTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor`, *optional*):
            Pixel value patches for images, NaViT-packed.
        target_sizes (`torch.IntTensor`, *optional*):
            Height and width (in patches) for each image.
        pixel_values_videos (`torch.FloatTensor`, *optional*):
            Pixel value patches for video frames, NaViT-packed.
        target_sizes_videos (`torch.IntTensor`, *optional*):
            Height and width (in patches) for each video frame.
        downsample_mode (`str`, *optional*):
            `"4x"` keeps 4x more visual tokens; default `"16x"` applies full merge.
        target_sizes_mrope (`torch.IntTensor` of shape `(batch_size, num_visuals, 2)`, *optional*):
            Spatial grid sizes per visual crop for canvas M-RoPE.
        mm_token_type_ids (`torch.IntTensor`, *optional*):
            Modality type ids (`0` text, `1` image, `2` video) from the processor. Required
            together with `target_sizes_mrope` for canvas M-RoPE.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            target_sizes=target_sizes,
            pixel_values_videos=pixel_values_videos,
            target_sizes_videos=target_sizes_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            downsample_mode=downsample_mode,
            target_sizes_mrope=target_sizes_mrope,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Overwritten -- canvas M-RoPE needs 4D position ids [text, T, H, W].
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()

        if past_length != 0 and self.model.rope_deltas is not None:
            batch_size = text_positions.shape[0]
            mrope_positions = text_positions.unsqueeze(0).expand(3, batch_size, -1)
            delta = self.model.rope_deltas.repeat_interleave(batch_size // self.model.rope_deltas.shape[0], dim=0)
            mrope_positions = mrope_positions + delta.to(device=text_positions.device)
            return torch.cat([text_positions.unsqueeze(0), mrope_positions], dim=0)

        if model_kwargs.get("target_sizes_mrope") is not None and model_kwargs.get("mm_token_type_ids") is not None:
            input_ids = model_kwargs.get("input_ids", inputs_tensor)
            mrope_positions, self.model.rope_deltas = self.model.get_rope_index(
                input_ids,
                attention_mask=model_kwargs.get("attention_mask"),
                target_sizes_mrope=model_kwargs.get("target_sizes_mrope"),
                mm_token_type_ids=model_kwargs.get("mm_token_type_ids"),
                downsample_mode=model_kwargs.get("downsample_mode"),
            )
            return torch.cat([text_positions.unsqueeze(0), mrope_positions], dim=0)

        return text_positions

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        ts_keys = ("target_sizes", "target_sizes_videos")
        mrope_keys = ("target_sizes_mrope", "mm_token_type_ids")
        saved = {k: model_kwargs.pop(k) for k in (*ts_keys, *mrope_keys) if model_kwargs.get(k) is not None}

        expanded_position_ids = None
        if (pos := model_kwargs.get("position_ids")) is not None and pos.ndim == 3:
            expanded_position_ids = model_kwargs.pop("position_ids").repeat_interleave(expand_size, dim=1)

        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )

        if expanded_position_ids is not None:
            model_kwargs["position_ids"] = expanded_position_ids
        model_kwargs.update(saved)
        return input_ids, model_kwargs


class MiniCPMV4_7ImageProcessorKwargs(MiniCPMV4_6ImageProcessorKwargs, total=False):
    pass


@auto_docstring
class MiniCPMV4_7ImageProcessor(MiniCPMV4_6ImageProcessor):
    def get_sliced_grid(
        self,
        image_size: tuple[int, int],
        max_slice_nums: int,
        scale_resolution: int,
    ) -> list[int] | None:
        original_height, original_width = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
            return None

        best_grid = [1, 1]
        min_error = float("inf")
        for num_slices in [multiple - 1, multiple, multiple + 1]:
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows == 0:
                    num_cols = num_slices // num_rows
                    error = abs(log_ratio - math.log(num_cols / num_rows))
                    if error < min_error:
                        best_grid = [num_rows, num_cols]
                        min_error = error
                    elif error == min_error and num_rows > best_grid[0]:
                        best_grid = [num_rows, num_cols]
        return best_grid


@auto_docstring
class MiniCPMV4_7ImageProcessorPil(MiniCPMV4_6ImageProcessorPil):
    # Upstream convention (glm4v, smolvlm, ...) is for the PIL backend to share the
    # `<Model>ImageProcessorKwargs` of the torchvision backend rather than declaring a
    # separate `...ImageProcessorPilKwargs`.
    valid_kwargs = MiniCPMV4_7ImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[MiniCPMV4_7ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[MiniCPMV4_7ImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def get_sliced_grid(
        self,
        image_size: tuple[int, int],
        max_slice_nums: int,
        scale_resolution: int,
    ) -> list[int] | None:
        original_height, original_width = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
            return None

        best_grid = [1, 1]
        min_error = float("inf")
        for num_slices in [multiple - 1, multiple, multiple + 1]:
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows == 0:
                    num_cols = num_slices // num_rows
                    error = abs(log_ratio - math.log(num_cols / num_rows))
                    if error < min_error:
                        best_grid = [num_rows, num_cols]
                        min_error = error
                    elif error == min_error and num_rows > best_grid[0]:
                        best_grid = [num_rows, num_cols]
        return best_grid


class MiniCPMV4_7VideoProcessorKwargs(MiniCPMV4_6VideoProcessorKwargs):
    pass


@auto_docstring
class MiniCPMV4_7VideoProcessor(MiniCPMV4_6VideoProcessor):
    # Video frames form a single temporal sequence, so they are not numbered with local image ids.
    use_image_id = False

    def get_sliced_grid(
        self,
        video_size: tuple[int, int],
        max_slice_nums: int,
        scale_resolution: int,
    ) -> list[int] | None:
        original_height, original_width = video_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
            return None

        best_grid = [1, 1]
        min_error = float("inf")
        for num_slices in [multiple - 1, multiple, multiple + 1]:
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows == 0:
                    num_cols = num_slices // num_rows
                    error = abs(log_ratio - math.log(num_cols / num_rows))
                    if error < min_error:
                        best_grid = [num_rows, num_cols]
                        min_error = error
                    elif error == min_error and num_rows > best_grid[0]:
                        best_grid = [num_rows, num_cols]
        return best_grid


class MiniCPMV4_7ProcessorKwargs(MiniCPMV4_6ProcessorKwargs, total=False):
    _defaults = {
        "common_kwargs": {
            "return_tensors": "pt",
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "return_mm_token_type_ids": True,
            "return_text_replacement_offsets": False,
        },
    }


@auto_docstring
class MiniCPMV4_7Processor(MiniCPMV4_6Processor):
    valid_processor_kwargs = MiniCPMV4_7ProcessorKwargs

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids", "target_sizes_mrope"]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        # MiniCPM needs to override `__call__` due to `_prepend_local_ids`, i.e. we add local image id inside text
        # Current `replace_image_tokens` API assumes that each image-placeholder doesn't depend on the other!
        images, text, videos, _ = self.prepare_inputs_layout(images=images, text=text, videos=videos, **kwargs)
        self.validate_inputs(images=images, text=text, videos=videos, **kwargs)

        merged_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )
        use_image_id = merged_kwargs["images_kwargs"].pop("use_image_id", None)
        use_image_id = use_image_id if use_image_id is not None else self.default_use_image_id
        # `use_image_id` is an image-only setting, so it must not leak into the video branch.
        merged_kwargs["videos_kwargs"].pop("use_image_id", None)

        processed_images = processed_videos = {}
        images_replacements = videos_replacements = []
        # Per-visual patch grids (not recoverable from config alone), one flat entry per visual
        # input, in the same order as the per-modality replacement strings.
        images_mrope_grids: list[list[list[int]]] = []
        videos_mrope_grids: list[list[list[int]]] = []
        if images is not None:
            processed_images, images_replacements = self._process_images(
                images,
                **merged_kwargs["images_kwargs"],
            )
            images_mrope_grids = self._image_mrope_grids(processed_images, images)
        if videos is not None:
            processed_videos, videos_replacements = self._process_videos(
                videos,
                **merged_kwargs["videos_kwargs"],
            )
            videos_mrope_grids = self._video_mrope_grids(processed_videos, videos)

        text_inputs = {}
        text_replacement_offsets = []
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors", None)
        if text is not None:
            return_mm_token_type_ids = merged_kwargs["text_kwargs"].pop("return_mm_token_type_ids", True)
            return_text_replacement_offsets = merged_kwargs["text_kwargs"].pop(
                "return_text_replacement_offsets", False
            )

            if images_replacements and use_image_id:
                images_replacements = self._prepend_local_ids(text, images_replacements, self.image_token)

            text, text_replacement_offsets = self.get_text_with_replacements(
                text,
                images_replacements,
                videos_replacements,
            )
            text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "audio"])

            if return_text_replacement_offsets:
                text_inputs["text_replacement_offsets"] = text_replacement_offsets

            if return_mm_token_type_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        # `target_sizes_mrope` must follow the order of the visual spans inside `input_ids`, which is
        # the order the placeholders appear in `text` — not the order the modalities are processed in.
        # A sample may interleave modalities (e.g. "<video>...</video><image>...</image>"), so the
        # per-modality grids collected above are re-sequenced through the text replacement offsets.
        mrope_inputs = {}
        if offsets_per_sample := text_replacement_offsets:
            mrope_tgt_sizes_per_sample = self._assemble_mrope_target_sizes(
                offsets_per_sample, images_mrope_grids, videos_mrope_grids
            )
            # Samples in a batch need not carry the same number of visuals — a text-only sample
            # carries none — so the per-sample grids are right-padded into one tensor. Canvas
            # M-RoPE walks the rows in visual-span order, so the padding rows are never read.
            max_visuals = max((len(grids) for grids in mrope_tgt_sizes_per_sample), default=0)
            target_sizes_mrope = torch.zeros(len(mrope_tgt_sizes_per_sample), max_visuals, 2, dtype=torch.int32)
            for idx, sample_grids in enumerate(mrope_tgt_sizes_per_sample):
                if sample_grids:
                    target_sizes_mrope[idx, : len(sample_grids)] = torch.tensor(sample_grids, dtype=torch.int32)
            # Do not return special_token_ids (available on model config) or image_bounds
            # (model recomputes bounds on compact/unpadded ids for left-padding safety).
            mrope_inputs = {"target_sizes_mrope": target_sizes_mrope}

        data = {**text_inputs, **processed_images, **processed_videos, **mrope_inputs}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}

        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    def _image_mrope_grids(self, image_inputs: dict, images: ImageInput) -> list[list[list[int]]]:
        """Return one flat list of patch grids per image, aligned with the image replacement strings."""
        images = make_flat_list_of_images(images)
        return [self._image_target_sizes(image_inputs, idx).tolist() for idx in range(len(images))]

    @staticmethod
    def _image_target_sizes(image_inputs: dict, image_idx: int):
        """Return the patch target sizes belonging to one image of the flattened batch."""
        cum_patches = np.cumsum(image_inputs["num_patches_per_image"])
        start_idx = cum_patches[image_idx - 1] if image_idx > 0 else 0
        end_idx = cum_patches[image_idx]
        return image_inputs["target_sizes"][start_idx:end_idx]

    def _video_mrope_grids(self, video_inputs: dict, videos: VideoInput) -> list[list[list[int]]]:
        """Return one flat list of patch grids per video, aligned with the video replacement strings.

        Frames are concatenated in order, and a frame may itself span several patches.
        """
        videos = make_batched_videos(videos)
        mrope_grids = []
        for idx in range(len(videos)):
            video_grids = []
            for frame_ts, _, _ in self._iter_video_frames(video_inputs, idx):
                video_grids.extend(frame_ts.tolist())
            mrope_grids.append(video_grids)
        return mrope_grids

    @staticmethod
    def _assemble_mrope_target_sizes(
        offsets_per_sample: list[list[dict]],
        images_mrope_grids: list[list[list[int]]],
        videos_mrope_grids: list[list[list[int]]],
    ) -> list[list[list[int]]]:
        """Flatten per-visual patch grids into per-sample lists, following the text order of the visuals.

        `get_text_with_replacements` walks each sample left to right and yields one offset entry per
        placeholder occurrence, tagged with its modality. Consuming the per-modality grid lists in that
        same order is what makes `target_sizes_mrope[b]` line up with the visual spans inside
        `input_ids[b]` even when a sample interleaves image and video placeholders.

        Grids are counted per visual input, not per frame: one video placeholder expands to all the
        frames of that video. For a batch of a single modality this reproduces the plain per-modality
        concatenation in use before, so single-modality behaviour is unchanged.
        """
        available = {
            "image": len(images_mrope_grids),
            "video": len(videos_mrope_grids),
        }
        num_placeholders = dict.fromkeys(available, 0)
        for sample_offsets in offsets_per_sample:
            for offset in sample_offsets:
                num_placeholders[offset["type"]] += 1

        for modality, expected in available.items():
            if num_placeholders[modality] != expected:
                raise ValueError(
                    f"Number of `{modality}` placeholders does not match the number of `{modality}` inputs: "
                    f"found {num_placeholders[modality]} placeholder(s) in `text` but received {expected} input(s). "
                    "Every placeholder must have a matching input."
                )

        image_iter = iter(images_mrope_grids)
        video_iter = iter(videos_mrope_grids)
        grids_per_sample = []
        for sample_offsets in offsets_per_sample:
            sample_grids = []
            for offset in sample_offsets:
                # One video input expands to one grid per frame, so all of its grids are appended here
                # in frame order; they then line up with that video's tokens from left to right.
                for grid in next(image_iter if offset["type"] == "image" else video_iter):
                    sample_grids.append(grid)
            grids_per_sample.append(sample_grids)
        return grids_per_sample

    def _iter_video_frames(self, video_inputs: dict, video_idx: int):
        """Yield `(frame_target_sizes, grid_rows, grid_cols)` per frame of one video, in text order."""
        video_target_sizes = video_inputs["target_sizes_videos"]
        num_frames_per_video = video_inputs["num_frames_per_video"]
        video_grids = video_inputs["grids_videos"]
        num_patches_per_frame = video_grids.prod(-1) + 1

        num_frames = num_frames_per_video[video_idx]
        cum_patches_per_frame = np.cumsum(num_patches_per_frame)
        num_past_frames = np.cumsum(num_frames_per_video)[video_idx] - num_frames

        for frame_idx in range(num_frames):
            frame_start_idx = num_past_frames + frame_idx

            start_idx = cum_patches_per_frame[frame_start_idx - 1] if frame_start_idx > 0 else 0
            end_idx = cum_patches_per_frame[frame_start_idx]

            grid_rows, grid_cols = video_grids[frame_start_idx]
            yield video_target_sizes[start_idx:end_idx], grid_rows, grid_cols


__all__ = [
    "MiniCPMV4_7Config",
    "MiniCPMV4_7VisionConfig",
    "MiniCPMV4_7PreTrainedModel",
    "MiniCPMV4_7Model",
    "MiniCPMV4_7ForConditionalGeneration",
    "MiniCPMV4_7ImageProcessor",
    "MiniCPMV4_7ImageProcessorPil",
    "MiniCPMV4_7VideoProcessor",
    "MiniCPMV4_7Processor",
]
