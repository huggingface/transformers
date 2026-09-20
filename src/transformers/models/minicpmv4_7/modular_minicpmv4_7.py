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
from typing import Any

import torch
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...image_utils import ImageInput, make_flat_list_of_images
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, logging
from ...video_utils import VideoInput, make_batched_videos
from ..minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6Config, MiniCPMV4_6VisionConfig
from ..minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6Model,
    MiniCPMV4_6ViTWindowAttentionMerger,
)
from ..minicpmv4_6.processing_minicpmv4_6 import MiniCPMV4_6Processor, MiniCPMV4_6ProcessorKwargs


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4.7")
@strict
class MiniCPMV4_7VisionConfig(MiniCPMV4_6VisionConfig):
    pass


@auto_docstring(checkpoint="openbmb/MiniCPM-V-4.7")
@strict
class MiniCPMV4_7Config(MiniCPMV4_6Config):
    r"""
    insert_layer_id (`int`, *optional*, defaults to 6):
        Vision encoder layer index after which the window-attention merger is applied.
    drop_vision_last_layer (`bool`, *optional*, defaults to `False`):
        Whether to drop the last layer of the vision encoder.
    downsample_mode (`str`, *optional*, defaults to `"16x"`):
        Visual token downsampling ratio. `"4x"` keeps 4× more tokens.
    merge_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`):
        Kernel size `(h, w)` for merging adjacent visual patches in the Merger.
    merger_times (`int`, *optional*, defaults to 1):
        Number of iterative merge rounds in the Merger.
    image_start_id (`int`, *optional*):
        Token id of the image-start marker (`<image>`). Canvas M-RoPE pins it to the halo just
        outside the top-left corner of the image canvas. Resolved from the tokenizer by the
        conversion script and stored in `config.json`; required for image or video inputs.
    image_end_id (`int`, *optional*):
        Token id of the image-end marker (`</image>`). Canvas M-RoPE pins it to the far corner of
        the image canvas. Resolved from the tokenizer by the conversion script and stored in
        `config.json`; required for image or video inputs.
    slice_start_id (`int`, *optional*):
        Token id of the slice-start marker (`<slice>`). Canvas M-RoPE pins it to the top-left
        corner of the slice it opens. Resolved from the tokenizer by the conversion script and
        stored in `config.json`; required for image or video inputs.
    slice_end_id (`int`, *optional*):
        Token id of the slice-end marker (`</slice>`). Canvas M-RoPE pins it to the bottom-right
        corner of the slice it closes. Resolved from the tokenizer by the conversion script and
        stored in `config.json`; required for image or video inputs.
    newline_id (`int`, *optional*):
        Token id of the newline (`"\n"`) that separates slice rows. Canvas M-RoPE pins it just
        past the right edge of the row it ends. Resolved from the tokenizer by the conversion
        script and stored in `config.json`; required for image or video inputs.
    """

    image_start_id: int | None = None
    image_end_id: int | None = None
    slice_start_id: int | None = None
    slice_end_id: int | None = None
    newline_id: int | None = None


def _crop_end(crop, input_ids: torch.LongTensor, structural_ids: set, limit: int) -> int:
    """End of a frame's span, including the ``</image>``/``</slice>`` markers that close it."""
    end = crop["slices"][-1][1] if crop["slices"] else crop["thumbnail"][1]
    while end < limit and input_ids[end].item() in structural_ids:
        end += 1
    return end


def _group_visual_frames(
    input_ids: torch.LongTensor, mm_token_type_ids: torch.IntTensor, special_token_ids: dict
) -> list[dict]:
    """Split one sequence into visual groups, each a list of ``thumbnail + slices`` frames.

    ``mm_token_type_ids`` labels every token with its modality (``0`` text, ``1`` image, ``2``
    video), so each maximal non-text run is exactly one crop. A crop is a slice when ``<slice>``
    sits in front of it, otherwise it opens a new frame and adopts the slices that follow.

    Frames separated by nothing but markers and newlines -- how a clip emits its frames, and how a
    picture can end up sitting right against one -- share a group. Grouping them matters: the next
    frame then starts one step past the tokens in between, so its halo cannot land on the last of
    them.

    Args:
        input_ids: `(seq_len,)` token ids of one unpadded sequence.
        mm_token_type_ids: `(seq_len,)` modality label per token (0 text, 1 image, 2 video).
        special_token_ids: The five canvas marker ids, keyed by `im_start_id`, `im_end_id`,
            `slice_start_id`, `slice_end_id` and `newline_id`.

    Returns:
        One dict per visual group, in sequence order, each holding the span indices the caller
        needs so that it never has to rescan `input_ids` itself:
        - `start`: index of the `<image>` marker opening the group.
        - `end`: index one past the last marker closing the group.
        - `frames`: the group's frames, each `{"thumbnail", "slices", "modality", "end"}` where
          `thumbnail` and every entry of `slices` is a `(start, end, crop_index)` triple and
          `end` is one past the markers that close that frame.
    """
    slice_start_id = special_token_ids["slice_start_id"]
    structural_ids = set(special_token_ids.values())
    gap_ids = torch.tensor(
        [
            special_token_ids["im_start_id"],
            special_token_ids["im_end_id"],
            slice_start_id,
            special_token_ids["slice_end_id"],
            special_token_ids["newline_id"],
        ],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    seq_len = input_ids.shape[0]
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
            groups[-1]["frames"][-1]["slices"].append(crop)
        else:
            frame = {"thumbnail": crop, "slices": [], "modality": modality}
            gap = input_ids[previous_end : crop[0] - 1]
            adjacent = bool(groups) and bool(torch.isin(gap, gap_ids).all())
            if adjacent:
                groups[-1]["frames"].append(frame)
            else:
                groups.append({"start": crop[0] - 1, "frames": [frame]})
        previous_end = crop[1]

    for group in groups:
        for frame in group["frames"]:
            frame["end"] = _crop_end(frame, input_ids, structural_ids, seq_len)
        group["end"] = group["frames"][-1]["end"]
    return groups


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
        hidden_states = residual[:, window_index, :] + hidden_states

        # This is where 4.7 parts ways with 4.6. 4.6 un-permutes the stream with `argsort(window_index)`
        # and then walks `target_sizes` image by image to reshape each one on its own. Here the stream is
        # left in window order, where `window_index` already puts the patches of one window next to each
        # other, so a single reshape merges every window at once and a batch whose images have different
        # patch grids no longer needs a Python loop. `target_sizes` is only read for the guard below.
        window_h, window_w = self.window_kernel_size
        window_size = window_h * window_w
        embed_dim = hidden_states.shape[-1]
        if window_cu_seqlens.numel() - 1 != hidden_states.shape[1] // window_size:
            raise ValueError(
                f"Patch grids {target_sizes} must be divisible by window kernel size {self.window_kernel_size}"
            )

        patch = hidden_states.reshape(-1, window_size, embed_dim)
        flat = patch.flatten(1)
        patch_residual = patch.mean(dim=1)

        hidden_state = self.pre_norm(flat)
        hidden_state = self.linear_1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.linear_2(hidden_state)

        return (hidden_state + patch_residual).unsqueeze(0)


class MiniCPMV4_7Model(MiniCPMV4_6Model):
    def __init__(self, config: MiniCPMV4_7Config):
        super().__init__(config)
        self.rope_deltas = None

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int, int, int] | torch.Tensor,
        canvas_height: int = 0,
        canvas_width: int = 0,
        h_offset: int = 0,
        w_offset: int = 0,
        spatial_merge_size: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """3D (t, h, w) position ids for one grid of patches placed on a canvas.

        canvas_height/canvas_width == 0 means "no canvas given" -> canvas defaults to
        this grid's own size, so the linspace resample degenerates to a plain arange.
        That single default covers three cases with the same math:
        - thumbnail *with* slices:  canvas = full slice canvas, offsets = 0
        - thumbnail *without* slices: canvas defaults to own grid -> arange, offsets = 0
        - a slice's interior grid: canvas defaults to own grid -> arange, offsets = its (h, w) placement

        Args:
            start_position (`int`):
                Canvas origin in the position-id sequence. The `t` channel is frozen to it and the
                `h`/`w` channels are counted from it.
            grid_thw (`list[int]` or `torch.Tensor`):
                Patch grid of this crop; only `[0]` (height) and `[1]` (width) are read.
            canvas_height (`int`, *optional*, defaults to 0):
                Height of the canvas in LLM tokens. 0 falls back to this grid's own height.
            canvas_width (`int`, *optional*, defaults to 0):
                Width of the canvas in LLM tokens. 0 falls back to this grid's own width.
            h_offset (`int`, *optional*, defaults to 0):
                Row of the canvas where this grid's top-left corner sits, in LLM tokens.
            w_offset (`int`, *optional*, defaults to 0):
                Column of the canvas where this grid's top-left corner sits, in LLM tokens.
            spatial_merge_size (`int`, *optional*, defaults to 1):
                Patch-to-LLM-token merge factor (4 in `"16x"` mode, 2 in `"4x"` mode).
            device (`str` or `torch.device`, *optional*):
                Device the returned tensor is allocated on.

        Returns:
            `torch.Tensor`: `(3, llm_grid_h * llm_grid_w)` position ids `[T, H, W]`, row-major over
            this crop's LLM tokens.
        """
        llm_grid_h = grid_thw[0].item() // spatial_merge_size
        llm_grid_w = grid_thw[1].item() // spatial_merge_size

        canvas_height = canvas_height or llm_grid_h
        canvas_width = canvas_width or llm_grid_w

        h_coords = torch.linspace(0, canvas_height - 1, llm_grid_h, device=device).round().long() + h_offset
        w_coords = torch.linspace(0, canvas_width - 1, llm_grid_w, device=device).round().long() + w_offset

        H_grid, W_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")
        T_grid = torch.zeros_like(H_grid)
        position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
        position_ids += start_position
        return position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor | None = None,
        target_sizes: torch.LongTensor | None = None,
        target_sizes_videos: torch.LongTensor | None = None,
        downsample_mode: str | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canvas M-RoPE indices ``(3, B, S)`` plus ``rope_deltas``.

        ``mm_token_type_ids`` marks the crops, so the canvas is laid out per sequence on the
        tokens ``attention_mask`` keeps. Padding stays at zero and never shifts the canvas.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                Token ids of the prompt. The canvas is anchored on the markers around each crop,
                so it cannot be derived from `inputs_embeds` alone.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, seq_len)`, *optional*):
                Modality label per token (0 text, 1 image, 2 video), as emitted by the processor.
            target_sizes (`torch.LongTensor` of shape `(num_image_crops, 2)`, *optional*):
                Patch grid `(h, w)` of every image crop, in the order the crops appear.
            target_sizes_videos (`torch.LongTensor` of shape `(num_video_crops, 2)`, *optional*):
                Patch grid `(h, w)` of every video crop, in the order the crops appear.
            downsample_mode (`str`, *optional*):
                Per-call override of `config.downsample_mode`. It has to match the mode the vision
                tower runs with, otherwise the placeholder count and the canvas disagree.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Padding mask. The canvas is laid out over the kept tokens only.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`:
            - **position_ids** of shape `(3, batch_size, seq_len)` holding `[T, H, W]`.
            - **rope_deltas** of shape `(batch_size, 1)`, the offset generation adds to the 1-D
              positions of the tokens appended after the prompt.
        """
        # `16x` merges 4x4 patches into one LLM token (window merger 2x2, then merger 2x2), `4x`
        # skips the window merger and merges 2x2. Same divisor the processor counts placeholders
        # with, so it has to follow the per-call override the vision tower is given, not the config.
        downsample_mode = downsample_mode or self.config.downsample_mode
        merge_factor = 2 if downsample_mode == "4x" else 4
        special_token_ids = {
            "im_start_id": self.config.image_start_id,
            "im_end_id": self.config.image_end_id,
            "slice_start_id": self.config.slice_start_id,
            "slice_end_id": self.config.slice_end_id,
            "newline_id": self.config.newline_id,
        }

        device = input_ids.device
        position_ids = torch.zeros(3, *input_ids.size(), dtype=torch.long, device=input_ids.device)

        grid_iters = {
            1: iter(target_sizes) if target_sizes is not None else None,
            2: iter(target_sizes_videos) if target_sizes_videos is not None else None,
        }

        for batch_idx in range(input_ids.shape[0]):
            current_input_ids = input_ids[batch_idx]
            current_mm_token_type_ids = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                valid_mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[valid_mask]
                current_mm_token_type_ids = current_mm_token_type_ids[valid_mask]
            else:
                # Nothing to unpad, so the canvas is scattered back over the whole row.
                valid_mask = slice(None)

            seq_len = current_input_ids.shape[0]
            curr_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).expand(3, -1).clone()

            current_pos = 0
            current_cursor = 0
            for group in _group_visual_frames(current_input_ids, current_mm_token_type_ids, special_token_ids):
                frames = group["frames"]
                group_start = group["start"]
                group_end = group["end"]

                # Text in front of the group is plain 1-D.
                if group_start > current_cursor:
                    text_len = group_start - current_cursor
                    curr_position_ids[:, current_cursor:group_start] = (
                        torch.arange(text_len, device=device) + current_pos
                    )
                    current_pos += text_len

                frame_cursor = group_start
                for frame in frames:
                    thumb_start, thumb_end, thumb_index = frame["thumbnail"]
                    slices = frame["slices"]
                    target_sizes_thumb = next(grid_iters[frame["modality"]])
                    frame_start = thumb_start - 1

                    # Tokens between two frames of a clip are 1-D text; the extra step afterwards keeps the
                    # next frame's halo from landing on the last of them.
                    if frame_start > frame_cursor:
                        gap_len = frame_start - frame_cursor
                        curr_position_ids[:, frame_cursor:frame_start] = (
                            torch.arange(gap_len, device=device) + current_pos
                        )
                        current_pos += gap_len + 1

                    frame_end = min(frame["end"], group_end)
                    canvas_origin = current_pos
                    halo_before_canvas = max(canvas_origin - 1, 0)

                    llm_slice_h, llm_slice_w = 0, 0
                    num_rows, num_cols = 0, 0
                    canvas_height = target_sizes_thumb[0].item() // merge_factor
                    canvas_width = target_sizes_thumb[1].item() // merge_factor
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
                        target_sizes_first_slice = next(grid_iters[frame["modality"]])
                        llm_slice_h = target_sizes_first_slice[0].item() // merge_factor
                        llm_slice_w = target_sizes_first_slice[1].item() // merge_factor
                        canvas_height = num_rows * llm_slice_h
                        canvas_width = num_cols * llm_slice_w

                    # Base coat for the whole frame, then `<image>` -> halo just outside the canvas.
                    curr_position_ids[:, frame_start:frame_end] = canvas_origin
                    curr_position_ids[1:, frame_start] = halo_before_canvas

                    # `</image>` closes the thumbnail on the far corner of the canvas.
                    image_end_pos = thumb_end
                    if image_end_pos < frame_end:
                        curr_position_ids[1, image_end_pos] = canvas_origin + canvas_height
                        curr_position_ids[2, image_end_pos] = canvas_origin + canvas_width

                    # Thumbnail tokens. With slices around, the thumbnail is stretched over the full canvas
                    # so that it stays aligned with the detail crops underneath it. Without slices the canvas
                    # is the thumbnail grid itself and `linspace` degenerates to `arange`.
                    curr_position_ids[:, thumb_start:thumb_end] = self.get_vision_position_ids(
                        start_position=canvas_origin,
                        grid_thw=target_sizes_thumb,
                        canvas_height=canvas_height,
                        canvas_width=canvas_width,
                        spatial_merge_size=merge_factor,
                        device=device,
                    )

                    # Slice tokens plus the `<slice>`/`</slice>` markers that pin each crop's corners.
                    for k, (slice_start, slice_end, slice_index) in enumerate(slices):
                        h_off = (k // num_cols) * llm_slice_h
                        w_off = (k % num_cols) * llm_slice_w

                        if k > 0:
                            target_sizes = next(grid_iters[frame["modality"]])
                        else:
                            target_sizes = target_sizes_first_slice

                        slice_h = target_sizes[0].item() // merge_factor
                        slice_w = target_sizes[1].item() // merge_factor
                        slice_start_pos = slice_start - 1
                        if slice_start_pos >= frame_start:
                            curr_position_ids[1, slice_start_pos] = canvas_origin + h_off
                            curr_position_ids[2, slice_start_pos] = canvas_origin + w_off
                        slice_end_pos = slice_end
                        if slice_end_pos < frame_end:
                            curr_position_ids[1, slice_end_pos] = canvas_origin + h_off + slice_h - 1
                            curr_position_ids[2, slice_end_pos] = canvas_origin + w_off + slice_w - 1

                        curr_position_ids[:, slice_start:slice_end] = self.get_vision_position_ids(
                            start_position=canvas_origin,
                            grid_thw=target_sizes,
                            h_offset=h_off,
                            w_offset=w_off,
                            spatial_merge_size=merge_factor,
                            device=device,
                        )

                    # The "\n" that ends a row of slices sits just past the right edge of that row.
                    for k in range(len(slices) - 1):
                        gap_start, gap_end = slices[k][1], slices[k + 1][0]
                        if gap_end - gap_start <= 2:
                            continue
                        boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                        right_edge_w = num_cols * llm_slice_w
                        for newline_pos in range(gap_start + 1, gap_end - 1):
                            curr_position_ids[1, newline_pos] = canvas_origin + boundary_h
                            curr_position_ids[2, newline_pos] = canvas_origin + right_edge_w

                    current_pos = canvas_origin + max(canvas_height, canvas_width) + 1
                    frame_cursor = frame_end

                if frame_cursor < group_end:
                    trail_len = group_end - frame_cursor
                    curr_position_ids[:, frame_cursor:group_end] = torch.arange(trail_len, device=device) + current_pos
                    current_pos += trail_len
                current_cursor = group_end

            if current_cursor < seq_len:
                curr_position_ids[:, current_cursor:seq_len] = (
                    torch.arange(seq_len - current_cursor, device=device) + current_pos
                )
            position_ids[:, batch_idx, valid_mask] = curr_position_ids

        if attention_mask is not None:
            seq_lens = attention_mask.sum(-1)
        else:
            seq_lens = torch.full((input_ids.shape[0],), input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        rope_deltas = position_ids.amax(dim=(0, 2)).unsqueeze(1) + 1 - seq_lens.unsqueeze(1)
        return position_ids, rope_deltas.long()

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        target_sizes: torch.LongTensor | None = None,
        target_sizes_videos: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        downsample_mode: str | None = None,
    ) -> torch.Tensor | None:
        """Return the ``(3, batch, seq)`` canvas M-RoPE positions ``[T, H, W]`` (Qwen-style entrypoint).

        Generation prepends the text channel in ``_prepare_position_ids_for_generation``; the text
        model reads that fourth row only when it is there.
        """
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = target_sizes is not None or target_sizes_videos is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed (via `target_sizes` or `target_sizes_videos`) but `mm_token_type_ids` "
                "is missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
                "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            mrope_position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                attention_mask=attention_mask,
                target_sizes=target_sizes,
                target_sizes_videos=target_sizes_videos,
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
            if target_sizes_videos is not None or target_sizes is not None:
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
                target_sizes=target_sizes,
                target_sizes_videos=target_sizes_videos,
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


class MiniCPMV4_7ForConditionalGeneration(MiniCPMV4_6ForConditionalGeneration):
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
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

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

        # `input_ids` is empty when generating from `inputs_embeds`, and canvas M-RoPE needs the ids to locate
        # the visual spans, so fall back to plain 1-D text positions whenever they are unavailable.
        input_ids = model_kwargs.get("input_ids", inputs_tensor)
        has_input_ids = input_ids.dim() == 2 and input_ids.dtype in (torch.int, torch.long) and input_ids.shape[1] > 0

        if (
            has_input_ids
            and (model_kwargs.get("target_sizes") is not None or model_kwargs.get("target_sizes_videos") is not None)
            and model_kwargs.get("mm_token_type_ids") is not None
        ):
            mrope_positions, self.model.rope_deltas = self.model.get_rope_index(
                input_ids,
                attention_mask=model_kwargs.get("attention_mask"),
                target_sizes=model_kwargs.get("target_sizes"),
                target_sizes_videos=model_kwargs.get("target_sizes_videos"),
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
        # `target_sizes*` are indexed by crop, not by batch item, so they must sit out the dim-0
        # `repeat_interleave` that `super()` applies to every tensor in `model_kwargs`. Note that
        # `mm_token_type_ids` is deliberately *not* saved here: it is `(batch, seq)`, so the default
        # dim-0 expansion is exactly what it needs.
        ts_keys = ("target_sizes", "target_sizes_videos")
        saved = {k: model_kwargs.pop(k) for k in ts_keys if model_kwargs.get(k) is not None}

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


# Different from MiniCPM4-6, we need `mm_token_type_ids` returned by default
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


class MiniCPMV4_7Processor(MiniCPMV4_6Processor):
    valid_processor_kwargs = MiniCPMV4_7ProcessorKwargs

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[MiniCPMV4_7ProcessorKwargs],
    ):
        kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )
        # `use_image_id` is an image-only setting, so it must not leak into the video branch.
        kwargs["videos_kwargs"].pop("use_image_id", None)
        return super().__call__(images=images, text=text, videos=videos, **kwargs)

    def validate_inputs(self, images=None, text=None, videos=None, audio=None, **kwargs):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)

        # `get_text_with_replacements` consumes one replacement per placeholder occurrence, so a visual
        # without a matching placeholder is dropped from the prompt while its patches still reach
        # `pixel_values`/`target_sizes`. That mismatch is invisible in `input_ids`, so reject it here.
        num_inputs = {
            "image": 0 if images is None else len(make_flat_list_of_images(images)),
            "video": 0 if videos is None else len(make_batched_videos(videos)),
        }
        placeholders = {"image": self.image_token, "video": self.video_token}
        for modality, expected in num_inputs.items():
            num_placeholders = sum(sample.count(placeholders[modality]) for sample in text)
            if num_placeholders != expected:
                raise ValueError(
                    f"Number of `{modality}` placeholders does not match the number of `{modality}` inputs: "
                    f"found {num_placeholders} placeholder(s) in `text` but received {expected} input(s). "
                    "Every placeholder must have a matching input."
                )

    def _prepend_local_ids(self, text, replacements, token):
        """Prepend local (per-sample) image/video ID tokens to each replacement string."""
        if token == self.video_token:
            return replacements
        return super()._prepend_local_ids(text, replacements, token)

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]


__all__ = [
    "MiniCPMV4_7Config",
    "MiniCPMV4_7VisionConfig",
    "MiniCPMV4_7PreTrainedModel",  # noqa: F822
    "MiniCPMV4_7Model",
    "MiniCPMV4_7ForConditionalGeneration",
    "MiniCPMV4_7Processor",
]
