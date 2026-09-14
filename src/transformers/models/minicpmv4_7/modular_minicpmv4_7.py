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
from typing import Any

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..auto import CONFIG_MAPPING, AutoConfig
from ..minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6Config, MiniCPMV4_6VisionConfig
from ..minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6Model,
    MiniCPMV4_6PreTrainedModel,
)

logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# Canvas M-RoPE helpers
# ---------------------------------------------------------------------------


def _compute_llm_grid(num_tokens: int, vision_grid_height: int, vision_grid_width: int) -> tuple[int, int]:
    """Infer the (height, width) LLM-token grid for a visual patch of ``num_tokens`` tokens."""
    if num_tokens <= 0 or vision_grid_height <= 0 or vision_grid_width <= 0:
        return 0, 0
    vit_total = vision_grid_height * vision_grid_width
    if vit_total == num_tokens:
        return vision_grid_height, vision_grid_width
    if vit_total < num_tokens:
        return 0, 0
    factor = int(round(math.sqrt(vit_total / num_tokens)))
    if factor > 0 and vision_grid_height % factor == 0 and vision_grid_width % factor == 0:
        height, width = vision_grid_height // factor, vision_grid_width // factor
        if height * width == num_tokens:
            return height, width
    ratio = vision_grid_height / vision_grid_width
    width = max(1, int(round(math.sqrt(num_tokens / ratio))))
    height = num_tokens // width
    if height * width == num_tokens and height > 0:
        return height, width
    return 0, 0


def _build_image_bounds(input_ids: torch.LongTensor, special_token_ids: dict) -> torch.LongTensor:
    """Locate ``(start, end)`` visual-token spans (thumbnail + slices) in a 1-D sequence.

    ``start`` is the first visual token (right after ``<image_start>``/``<slice_start>``) and ``end``
    is the matching ``<image_end>``/``<slice_end>`` position. Indices are relative to ``input_ids``,
    so when the model recomputes bounds on a left-unpadded (compact) sequence the offsets stay valid.
    """
    start_ids = [
        x for x in (special_token_ids.get("im_start_id"), special_token_ids.get("slice_start_id")) if x is not None
    ]
    end_ids = [x for x in (special_token_ids.get("im_end_id"), special_token_ids.get("slice_end_id")) if x is not None]
    if not start_ids or not end_ids:
        return torch.zeros(0, 2, dtype=torch.long, device=input_ids.device)
    start_cond = torch.zeros_like(input_ids, dtype=torch.bool)
    end_cond = torch.zeros_like(input_ids, dtype=torch.bool)
    for start_id in start_ids:
        start_cond |= input_ids == start_id
    for end_id in end_ids:
        end_cond |= input_ids == end_id
    starts = torch.where(start_cond)[0] + 1
    ends = torch.where(end_cond)[0]
    if len(starts) != len(ends):
        raise ValueError(
            f"Malformed visual markup: found {len(starts)} start marker(s) but {len(ends)} end marker(s). "
            "Every `<image>`/`<slice>` must be closed by a matching `</image>`/`</slice>`."
        )
    if len(starts) == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=input_ids.device)
    if not bool((ends >= starts).all()):
        raise ValueError(
            "Malformed visual markup: start/end markers are nested or out of order; canvas M-RoPE "
            "expects flat, sequentially closed visual spans."
        )
    return torch.stack([starts, ends], dim=-1)


def _group_sequence_images(
    sequence_spans, flat_input_ids, sequence_start, sequence_end, structural_ids, special_token_ids
):
    """Group visual spans of one sequence into ``thumbnail + slices`` groups, merging video frames.

    A group is a thumbnail span plus the slice spans that follow it. Consecutive groups whose gap
    holds only structural or newline tokens belong to the same video and are merged into a single
    ``{"video_frames": [...]}`` entry so each frame later receives its own temporal index.
    """
    slice_start_id = special_token_ids.get("slice_start_id")

    raw_groups = []
    current_group = None
    for span_start, span_end, span_index in sequence_spans:
        marker_position = span_start - 1
        is_slice = (
            slice_start_id is not None
            and marker_position >= sequence_start
            and flat_input_ids[marker_position].item() == slice_start_id
        )
        if is_slice and current_group is not None:
            current_group["slices"].append((span_start, span_end, span_index))
        else:
            if current_group is not None:
                raw_groups.append(current_group)
            current_group = {"thumbnail": (span_start, span_end, span_index), "slices": []}
    if current_group is not None:
        raw_groups.append(current_group)

    # Merge consecutive tightly-adjacent groups into video groups. Two groups are "tightly adjacent"
    # if the gap between them contains only structural tokens and "\n" (no real text). This handles
    # both no-slice frames and frames-with-slices.
    newline_id = special_token_ids.get("newline_id")
    gap_ok_ids = structural_ids | ({newline_id} if newline_id is not None else set())

    def _group_end_position(group):
        if group["slices"]:
            last_visual_end = group["slices"][-1][1]
        else:
            last_visual_end = group["thumbnail"][1]
        group_end = last_visual_end
        while group_end < sequence_end and flat_input_ids[group_end].item() in structural_ids:
            group_end += 1
        return min(group_end, sequence_end)

    def _is_tight_gap(previous_group, next_group):
        group_end = _group_end_position(previous_group)
        next_group_marker = next_group["thumbnail"][0] - 1  # image_start of next group
        for position in range(group_end, next_group_marker):
            if flat_input_ids[position].item() not in gap_ok_ids:
                return False
        return True

    groups = []
    group_index = 0
    while group_index < len(raw_groups):
        video_frames = [raw_groups[group_index]]
        next_index = group_index + 1
        while next_index < len(raw_groups) and _is_tight_gap(raw_groups[next_index - 1], raw_groups[next_index]):
            video_frames.append(raw_groups[next_index])
            next_index += 1
        if len(video_frames) == 1:
            groups.append(raw_groups[group_index])
        else:
            groups.append({"video_frames": video_frames})
        group_index = next_index
    return groups


def _compute_canvas(
    position_ids_2d,
    cu_seqlens,
    image_bound,
    target_sizes,
    input_ids,
    special_token_ids,
):
    """Assign spatial ``(3, batch, seq)`` canvas positions to visual tokens of a packed sequence."""
    if position_ids_2d.ndim == 1:
        position_ids_2d = position_ids_2d.unsqueeze(0)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    has_images = (
        isinstance(image_bound, torch.Tensor)
        and image_bound.numel() > 0
        and isinstance(target_sizes, torch.Tensor)
        and target_sizes.numel() > 0
    )
    if not has_images:
        return position_ids_2d.unsqueeze(0).expand(3, -1, -1).contiguous()

    device = position_ids_2d.device
    position_ids_3d = position_ids_2d.unsqueeze(0).expand(3, -1, -1).clone()
    ids_flat = input_ids.view(-1)

    im_start_id = special_token_ids.get("im_start_id")
    im_end_id = special_token_ids.get("im_end_id")
    slice_start_id = special_token_ids.get("slice_start_id")
    slice_end_id = special_token_ids.get("slice_end_id")

    structural_ids = {v for v in (im_start_id, im_end_id, slice_start_id, slice_end_id) if v is not None}

    num_seqs = len(cu_seqlens) - 1
    image_pointer = 0
    num_images = len(image_bound)

    for seq_index in range(num_seqs):
        seq_start = cu_seqlens[seq_index].item()
        seq_end = cu_seqlens[seq_index + 1].item()
        if seq_start >= seq_end:
            continue

        sequence_spans = []
        while image_pointer < num_images:
            bound_start = image_bound[image_pointer, 0].item()
            bound_end = image_bound[image_pointer, 1].item()
            if bound_start >= seq_start and bound_end <= seq_end:
                sequence_spans.append((bound_start, bound_end, image_pointer))
                image_pointer += 1
            else:
                break
        if not sequence_spans:
            continue

        # Group images: thumbnail + following slices; merge consecutive frames into one video group.
        groups = _group_sequence_images(
            sequence_spans, ids_flat, seq_start, seq_end, structural_ids, special_token_ids
        )

        # Assign 3-D positions.
        pos = 0
        cursor = seq_start

        for group in groups:
            is_video_group = "video_frames" in group

            if is_video_group:
                # Video group: multiple frames merged. Each frame gets its own base position so
                # different frames have distinct spatial positions. Inter-frame gap tokens ("\n"
                # etc.) are pure 1-D text; after each gap, pos advances by 1 extra so the next
                # frame's halo does not collide with the last gap token.
                frames = group["video_frames"]
                first_frame = frames[0]
                last_frame = frames[-1]

                group_start = first_frame["thumbnail"][0] - 1
                if last_frame["slices"]:
                    last_visual_end = last_frame["slices"][-1][1]
                else:
                    last_visual_end = last_frame["thumbnail"][1]
                group_end = last_visual_end
                while group_end < seq_end and ids_flat[group_end].item() in structural_ids:
                    group_end += 1
                group_end = min(group_end, seq_end)

                # Text before the video group.
                text_len = group_start - cursor
                if text_len > 0:
                    text_pos = torch.arange(text_len, device=device, dtype=torch.long) + pos
                    for dim in range(3):
                        position_ids_3d[dim, 0, cursor:group_start] = text_pos
                    pos += text_len

                frame_cursor = group_start
                for frame in frames:
                    thumb_start, thumb_end, thumb_index = frame["thumbnail"]
                    frame_slices = frame["slices"]
                    image_start_pos = thumb_start - 1

                    # Inter-frame gap: pure 1-D text T=H=W=pos.
                    gap_len = image_start_pos - frame_cursor
                    if gap_len > 0:
                        text_pos = torch.arange(gap_len, device=device, dtype=torch.long) + pos
                        for dim in range(3):
                            position_ids_3d[dim, 0, frame_cursor:image_start_pos] = text_pos
                        pos += gap_len
                        pos += 1  # buffer: prevents halo collision

                    base = pos
                    halo_lo = max(base - 1, 0)

                    num_thumb = thumb_end - thumb_start
                    llm_thumb_h, llm_thumb_w = _compute_llm_grid(
                        num_thumb, target_sizes[thumb_index, 0].item(), target_sizes[thumb_index, 1].item()
                    )

                    if frame_slices:
                        num_cols = len(frame_slices)
                        for k in range(len(frame_slices) - 1):
                            if frame_slices[k + 1][0] - frame_slices[k][1] > 2:
                                num_cols = k + 1
                                break
                        num_rows = len(frame_slices) // num_cols if num_cols > 0 else 1
                        if num_rows * num_cols != len(frame_slices):
                            num_rows, num_cols = 1, len(frame_slices)
                        slice0_start, slice0_end, slice0_index = frame_slices[0]
                        llm_slice_h, llm_slice_w = _compute_llm_grid(
                            slice0_end - slice0_start,
                            target_sizes[slice0_index, 0].item(),
                            target_sizes[slice0_index, 1].item(),
                        )
                        canvas_height = num_rows * llm_slice_h
                        canvas_width = num_cols * llm_slice_w
                    else:
                        llm_slice_h, llm_slice_w = 0, 0
                        num_rows, num_cols = 0, 0
                        canvas_height = max(llm_thumb_h, 1)
                        canvas_width = max(llm_thumb_w, 1)

                    if frame_slices:
                        frame_visual_end = frame_slices[-1][1]
                    else:
                        frame_visual_end = thumb_end
                    frame_end = frame_visual_end
                    while frame_end < group_end and ids_flat[frame_end].item() in structural_ids:
                        frame_end += 1
                    frame_end = min(frame_end, group_end)

                    # Base coat for this frame's span.
                    position_ids_3d[0, 0, image_start_pos:frame_end] = base
                    position_ids_3d[1, 0, image_start_pos:frame_end] = base
                    position_ids_3d[2, 0, image_start_pos:frame_end] = base

                    # <image_start> -> halo.
                    position_ids_3d[1, 0, image_start_pos] = halo_lo
                    position_ids_3d[2, 0, image_start_pos] = halo_lo

                    # <image_end> -> halo (right after thumbnail).
                    image_end_pos = thumb_end
                    if image_end_pos < frame_end:
                        position_ids_3d[1, 0, image_end_pos] = base + canvas_height
                        position_ids_3d[2, 0, image_end_pos] = base + canvas_width

                    # Thumbnail visual tokens.
                    if llm_thumb_h > 0 and llm_thumb_w > 0 and llm_thumb_h * llm_thumb_w == num_thumb:
                        if frame_slices and canvas_height > 0 and canvas_width > 0:
                            h_coords = torch.linspace(0, canvas_height - 1, llm_thumb_h, device=device).round().long()
                            w_coords = torch.linspace(0, canvas_width - 1, llm_thumb_w, device=device).round().long()
                        else:
                            h_coords = torch.arange(llm_thumb_h, device=device)
                            w_coords = torch.arange(llm_thumb_w, device=device)
                        h_idx = h_coords.view(-1, 1).expand(-1, llm_thumb_w).reshape(-1)
                        w_idx = w_coords.view(1, -1).expand(llm_thumb_h, -1).reshape(-1)
                        position_ids_3d[0, 0, thumb_start:thumb_end] = base
                        position_ids_3d[1, 0, thumb_start:thumb_end] = h_idx + base
                        position_ids_3d[2, 0, thumb_start:thumb_end] = w_idx + base
                    else:
                        fallback = torch.arange(num_thumb, device=device, dtype=torch.long) + base
                        for dim in range(3):
                            position_ids_3d[dim, 0, thumb_start:thumb_end] = fallback

                    # Slice visual tokens + slice specials.
                    for k, (slice_start, slice_end, slice_index) in enumerate(frame_slices):
                        num_slice = slice_end - slice_start
                        slice_h, slice_w = _compute_llm_grid(
                            num_slice, target_sizes[slice_index, 0].item(), target_sizes[slice_index, 1].item()
                        )
                        row = k // num_cols
                        col = k % num_cols
                        h_off = row * llm_slice_h
                        w_off = col * llm_slice_w

                        slice_start_pos = slice_start - 1
                        if slice_start_pos >= image_start_pos:
                            position_ids_3d[1, 0, slice_start_pos] = base + h_off
                            position_ids_3d[2, 0, slice_start_pos] = base + w_off

                        slice_end_pos = slice_end
                        if slice_h > 0 and slice_w > 0:
                            end_h = h_off + slice_h - 1
                            end_w = w_off + slice_w - 1
                        else:
                            end_h = h_off
                            end_w = w_off
                        if slice_end_pos < frame_end:
                            position_ids_3d[1, 0, slice_end_pos] = base + end_h
                            position_ids_3d[2, 0, slice_end_pos] = base + end_w

                        if slice_h > 0 and slice_w > 0 and slice_h * slice_w == num_slice:
                            h_idx = torch.arange(slice_h, device=device).view(-1, 1).expand(-1, slice_w).reshape(-1)
                            w_idx = torch.arange(slice_w, device=device).view(1, -1).expand(slice_h, -1).reshape(-1)
                            position_ids_3d[0, 0, slice_start:slice_end] = base
                            position_ids_3d[1, 0, slice_start:slice_end] = h_idx + h_off + base
                            position_ids_3d[2, 0, slice_start:slice_end] = w_idx + w_off + base
                        else:
                            fallback = torch.arange(num_slice, device=device, dtype=torch.long) + base
                            for dim in range(3):
                                position_ids_3d[dim, 0, slice_start:slice_end] = fallback

                    # "\n" between slice rows -> W = right edge + 1; H stays within the ended row.
                    if frame_slices:
                        for k in range(len(frame_slices) - 1):
                            gap_start = frame_slices[k][1]
                            gap_end = frame_slices[k + 1][0]
                            if gap_end - gap_start <= 2:
                                continue
                            row_ended = k // num_cols
                            boundary_h = (row_ended + 1) * llm_slice_h - 1
                            right_edge_w = num_cols * llm_slice_w
                            for newline_pos in range(gap_start + 1, gap_end - 1):
                                position_ids_3d[1, 0, newline_pos] = base + boundary_h
                                position_ids_3d[2, 0, newline_pos] = base + right_edge_w

                    pos = base + max(canvas_height, canvas_width) + 1
                    frame_cursor = frame_end

                # Trailing tokens after the last frame (if any).
                if frame_cursor < group_end:
                    trail_len = group_end - frame_cursor
                    text_pos = torch.arange(trail_len, device=device, dtype=torch.long) + pos
                    for dim in range(3):
                        position_ids_3d[dim, 0, frame_cursor:group_end] = text_pos
                    pos += trail_len

                cursor = group_end

            else:
                # Single image group (thumbnail + optional slices).
                thumb_start, thumb_end, thumb_index = group["thumbnail"]
                slices = group["slices"]

                group_start = thumb_start - 1
                if slices:
                    last_visual_end = slices[-1][1]
                else:
                    last_visual_end = thumb_end
                group_end = last_visual_end
                while group_end < seq_end and ids_flat[group_end].item() in structural_ids:
                    group_end += 1
                group_end = min(group_end, seq_end)

                # Text before the group.
                text_len = group_start - cursor
                if text_len > 0:
                    text_pos = torch.arange(text_len, device=device, dtype=torch.long) + pos
                    for dim in range(3):
                        position_ids_3d[dim, 0, cursor:group_start] = text_pos
                    pos += text_len

                base = pos
                halo_lo = max(base - 1, 0)

                num_thumb = thumb_end - thumb_start
                llm_thumb_h, llm_thumb_w = _compute_llm_grid(
                    num_thumb, target_sizes[thumb_index, 0].item(), target_sizes[thumb_index, 1].item()
                )

                if slices:
                    num_cols = len(slices)
                    for k in range(len(slices) - 1):
                        if slices[k + 1][0] - slices[k][1] > 2:
                            num_cols = k + 1
                            break
                    num_rows = len(slices) // num_cols if num_cols > 0 else 1
                    if num_rows * num_cols != len(slices):
                        num_rows, num_cols = 1, len(slices)

                    slice0_start, slice0_end, slice0_index = slices[0]
                    llm_slice_h, llm_slice_w = _compute_llm_grid(
                        slice0_end - slice0_start,
                        target_sizes[slice0_index, 0].item(),
                        target_sizes[slice0_index, 1].item(),
                    )

                    canvas_height = num_rows * llm_slice_h
                    canvas_width = num_cols * llm_slice_w
                else:
                    llm_slice_h, llm_slice_w = 0, 0
                    canvas_height = max(llm_thumb_h, 1)
                    canvas_width = max(llm_thumb_w, 1)

                # Base coat.
                position_ids_3d[0, 0, group_start:group_end] = base
                position_ids_3d[1, 0, group_start:group_end] = base
                position_ids_3d[2, 0, group_start:group_end] = base

                # <image_start> -> halo.
                position_ids_3d[1, 0, group_start] = halo_lo
                position_ids_3d[2, 0, group_start] = halo_lo

                # <image_end> -> halo.
                image_end_pos = thumb_end
                if image_end_pos < group_end:
                    position_ids_3d[1, 0, image_end_pos] = base + canvas_height
                    position_ids_3d[2, 0, image_end_pos] = base + canvas_width

                # Thumbnail visual tokens.
                if llm_thumb_h > 0 and llm_thumb_w > 0 and llm_thumb_h * llm_thumb_w == num_thumb:
                    if slices and canvas_height > 0 and canvas_width > 0:
                        h_coords = torch.linspace(0, canvas_height - 1, llm_thumb_h, device=device).round().long()
                        w_coords = torch.linspace(0, canvas_width - 1, llm_thumb_w, device=device).round().long()
                    else:
                        h_coords = torch.arange(llm_thumb_h, device=device)
                        w_coords = torch.arange(llm_thumb_w, device=device)

                    h_idx = h_coords.view(-1, 1).expand(-1, llm_thumb_w).reshape(-1)
                    w_idx = w_coords.view(1, -1).expand(llm_thumb_h, -1).reshape(-1)

                    position_ids_3d[0, 0, thumb_start:thumb_end] = base
                    position_ids_3d[1, 0, thumb_start:thumb_end] = h_idx + base
                    position_ids_3d[2, 0, thumb_start:thumb_end] = w_idx + base
                else:
                    fallback = torch.arange(num_thumb, device=device, dtype=torch.long) + base
                    for dim in range(3):
                        position_ids_3d[dim, 0, thumb_start:thumb_end] = fallback

                # Slice visual tokens + slice specials.
                for k, (slice_start, slice_end, slice_index) in enumerate(slices):
                    num_slice = slice_end - slice_start
                    slice_h, slice_w = _compute_llm_grid(
                        num_slice, target_sizes[slice_index, 0].item(), target_sizes[slice_index, 1].item()
                    )

                    row = k // num_cols
                    col = k % num_cols
                    h_off = row * llm_slice_h
                    w_off = col * llm_slice_w

                    slice_start_pos = slice_start - 1
                    if slice_start_pos >= group_start:
                        position_ids_3d[1, 0, slice_start_pos] = base + h_off
                        position_ids_3d[2, 0, slice_start_pos] = base + w_off

                    slice_end_pos = slice_end
                    if slice_h > 0 and slice_w > 0:
                        end_h = h_off + slice_h - 1
                        end_w = w_off + slice_w - 1
                    else:
                        end_h = h_off
                        end_w = w_off
                    if slice_end_pos < group_end:
                        position_ids_3d[1, 0, slice_end_pos] = base + end_h
                        position_ids_3d[2, 0, slice_end_pos] = base + end_w

                    if slice_h > 0 and slice_w > 0 and slice_h * slice_w == num_slice:
                        h_idx = torch.arange(slice_h, device=device).view(-1, 1).expand(-1, slice_w).reshape(-1)
                        w_idx = torch.arange(slice_w, device=device).view(1, -1).expand(slice_h, -1).reshape(-1)

                        position_ids_3d[0, 0, slice_start:slice_end] = base
                        position_ids_3d[1, 0, slice_start:slice_end] = h_idx + h_off + base
                        position_ids_3d[2, 0, slice_start:slice_end] = w_idx + w_off + base
                    else:
                        fallback = torch.arange(num_slice, device=device, dtype=torch.long) + base
                        for dim in range(3):
                            position_ids_3d[dim, 0, slice_start:slice_end] = fallback

                # "\n" between slice rows -> W = right edge + 1; H stays within the ended row.
                if slices:
                    for k in range(len(slices) - 1):
                        gap_start = slices[k][1]
                        gap_end = slices[k + 1][0]
                        if gap_end - gap_start <= 2:
                            continue
                        row_ended = k // num_cols
                        boundary_h = (row_ended + 1) * llm_slice_h - 1
                        right_edge_w = num_cols * llm_slice_w
                        for newline_pos in range(gap_start + 1, gap_end - 1):
                            position_ids_3d[1, 0, newline_pos] = base + boundary_h
                            position_ids_3d[2, 0, newline_pos] = base + right_edge_w

                pos = base + max(canvas_height, canvas_width) + 1
                cursor = group_end

        # Remaining text.
        remaining = seq_end - cursor
        if remaining > 0:
            text_pos = torch.arange(remaining, device=device, dtype=torch.long) + pos
            for dim in range(3):
                position_ids_3d[dim, 0, cursor:seq_end] = text_pos

    return position_ids_3d


def _compute_canvas_single(input_ids, position_ids_2d, image_bound, target_sizes, special_token_ids):
    """Canvas positions for a single (unpacked, unpadded) sequence; returns ``(3, seq_len)``."""
    seq_len = input_ids.shape[0]
    cu_seqlens = torch.tensor([0, seq_len], device=input_ids.device, dtype=torch.long)
    return _compute_canvas(
        position_ids_2d.unsqueeze(0),
        cu_seqlens,
        image_bound,
        target_sizes,
        input_ids.unsqueeze(0),
        special_token_ids,
    )[:, 0, :]


def _normalize_target_sizes(target_sizes_mrope, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    """Coerce processor-provided per-sample grids into a list of ``(num_visuals, 2)`` long tensors."""
    empty = torch.zeros(0, 2, dtype=torch.long, device=device)
    if target_sizes_mrope is None:
        return [empty for _ in range(batch_size)]
    if isinstance(target_sizes_mrope, torch.Tensor) and target_sizes_mrope.ndim == 2:
        target_sizes_mrope = [target_sizes_mrope for _ in range(batch_size)]
    result = []
    for batch_idx in range(batch_size):
        item = target_sizes_mrope[batch_idx] if batch_idx < len(target_sizes_mrope) else None
        if item is None:
            result.append(empty)
            continue
        tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
        if tensor.numel() == 0:
            result.append(empty)
        else:
            result.append(tensor.to(device=device, dtype=torch.long).reshape(-1, 2))
    return result


def _has_visual_grids(target_sizes_mrope) -> bool:
    """Whether the processor actually reported at least one visual crop."""
    if target_sizes_mrope is None:
        return False
    if isinstance(target_sizes_mrope, torch.Tensor):
        return target_sizes_mrope.numel() > 0
    for item in target_sizes_mrope:
        if item is None:
            continue
        tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
        if tensor.numel() > 0:
            return True
    return False


def make_packed_text_position_ids(cu_seqlens, device=None) -> torch.Tensor:
    """Per-document 1-D text positions for a packed sequence; returns ``(1, total_len)``.

    Each document inside the packed sequence restarts at 0, so a document never inherits
    positions from the one packed before it.
    """
    if not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.as_tensor(cu_seqlens, device=device)
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    total_len = int(cu_seqlens[-1].item())
    flat = torch.arange(total_len, device=device, dtype=torch.long)
    starts = torch.repeat_interleave(cu_seqlens[:-1], lengths)
    return (flat - starts).unsqueeze(0)


def expand_1d_position_ids_to_3d(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand sequential 1-D positions to ``(3, batch, seq)`` for the Qwen3.5 text backbone."""
    if attention_mask is not None:
        return attention_mask.long().cumsum(-1).sub(1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)

    if input_ids.ndim == 1:
        pos = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        return pos.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

    batch_size, seq_len = input_ids.shape
    pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
    return pos.view(1, 1, -1).expand(3, batch_size, -1)


def compute_canvas_position_ids(input_ids, attention_mask, target_sizes_mrope, special_token_ids):
    """Build canvas ``(3, B, S)`` position ids with left-padding-safe mask compaction.

    Each batch row is compacted to valid tokens (Qwen-style), bounds are recomputed on the
    compact ids via ``_build_image_bounds``, then results are scattered back. This keeps the
    default processor ``padding_side="left"`` correct for ``batch > 1``.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    batch_size, seq_len = input_ids.shape
    out = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=input_ids.device)
    target_sizes_list = _normalize_target_sizes(target_sizes_mrope, batch_size, input_ids.device)

    for batch_idx in range(batch_size):
        if attention_mask is not None:
            valid_mask = attention_mask[batch_idx].bool()
        else:
            valid_mask = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)

        compact_ids = input_ids[batch_idx][valid_mask]
        compact_len = int(compact_ids.shape[0])
        compact_pos2d = torch.arange(compact_len, device=input_ids.device, dtype=torch.long)
        bounds = _build_image_bounds(compact_ids, special_token_ids)
        grids = target_sizes_list[batch_idx]

        if bounds.numel() == 0 or grids.numel() == 0:
            compact_pos3d = compact_pos2d.unsqueeze(0).expand(3, -1)
        else:
            if bounds.shape[0] != grids.shape[0]:
                raise ValueError(
                    f"Sample {batch_idx}: `input_ids` contains {bounds.shape[0]} visual span(s) but "
                    f"`target_sizes_mrope` provides {grids.shape[0]} grid(s). The processor and the model "
                    "disagree about the number of visual crops; canvas positions cannot be built."
                )
            compact_pos3d = _compute_canvas_single(compact_ids, compact_pos2d, bounds, grids, special_token_ids)

        out[:, batch_idx, valid_mask] = compact_pos3d
    return out


def compute_canvas_rope_index(
    input_ids,
    attention_mask,
    target_sizes_mrope,
    special_token_ids,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build canvas ``(3, batch, seq)`` position ids and decode ``rope_deltas``.

    ``rope_deltas`` uses ``amax + 1 - seq_len`` (Qwen convention) so continuing generation with
    an existing cache does not collide with the last prefill spatial position.
    """
    position_ids = compute_canvas_position_ids(input_ids, attention_mask, target_sizes_mrope, special_token_ids)
    if attention_mask is not None:
        seq_lens = attention_mask.sum(-1)
    else:
        seq_lens = torch.full(
            (input_ids.shape[0],), input_ids.shape[1], device=input_ids.device, dtype=torch.long
        )
    deltas = position_ids.amax(dim=(0, 2)).unsqueeze(1) + 1 - seq_lens.unsqueeze(1)
    return position_ids, deltas.long()


def compute_canvas_position_ids_packed(input_ids, cu_seqlens, target_sizes_mrope, special_token_ids):
    """Canvas ``(3, 1, total_len)`` positions for a packed (``batch=1``) sequence.

    ``input_ids`` holds several documents concatenated end to end, with boundaries given by
    ``cu_seqlens``. Visual spans are located per document so a crop never spills across a
    document boundary, and each document's canvas restarts from its own origin.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError(
            f"Packed canvas M-RoPE expects `batch=1` with documents concatenated along the sequence "
            f"axis, but `input_ids` has batch size {input_ids.shape[0]}. Pass one packed row, or drop "
            "`cu_seqlens` to use the regular padded path."
        )
    if not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.as_tensor(cu_seqlens, device=input_ids.device)
    cu_seqlens = cu_seqlens.to(device=input_ids.device, dtype=torch.long)

    total_len = input_ids.shape[1]
    if int(cu_seqlens[-1].item()) != total_len:
        raise ValueError(
            f"`cu_seqlens` ends at {int(cu_seqlens[-1].item())} but the packed sequence has {total_len} "
            "tokens; the two must describe the same sequence."
        )

    num_docs = cu_seqlens.numel() - 1
    grids_per_doc = _normalize_target_sizes(target_sizes_mrope, num_docs, input_ids.device)

    out = torch.zeros(3, 1, total_len, dtype=torch.long, device=input_ids.device)
    ids_flat = input_ids[0]

    for doc_idx in range(num_docs):
        doc_start = int(cu_seqlens[doc_idx].item())
        doc_end = int(cu_seqlens[doc_idx + 1].item())
        if doc_start >= doc_end:
            continue

        doc_ids = ids_flat[doc_start:doc_end]
        doc_pos2d = torch.arange(doc_end - doc_start, device=input_ids.device, dtype=torch.long)
        bounds = _build_image_bounds(doc_ids, special_token_ids)
        grids = grids_per_doc[doc_idx]

        if bounds.numel() == 0 or grids.numel() == 0:
            doc_pos3d = doc_pos2d.unsqueeze(0).expand(3, -1)
        else:
            if bounds.shape[0] != grids.shape[0]:
                raise ValueError(
                    f"Packed document {doc_idx}: `input_ids` contains {bounds.shape[0]} visual span(s) but "
                    f"`target_sizes_mrope` provides {grids.shape[0]} grid(s). The processor and the model "
                    "disagree about the number of visual crops; canvas positions cannot be built."
                )
            doc_pos3d = _compute_canvas_single(doc_ids, doc_pos2d, bounds, grids, special_token_ids)

        out[:, 0, doc_start:doc_end] = doc_pos3d
    return out


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

    # Structural token ids consumed by canvas M-RoPE. They are resolved from the tokenizer at
    # conversion time and persisted in `config.json`; a checkpoint that ships without them cannot
    # build canvas positions and `get_rope_index` raises instead of silently falling back to 1-D.
    image_start_id: int | None = None
    image_end_id: int | None = None
    slice_start_id: int | None = None
    slice_end_id: int | None = None
    newline_id: int | None = None

    # No tp/ep plan rewriting here: the text config declares `base_model_tp_plan` /
    # `base_model_ep_plan` as class attributes, and `init_parallel_plans()` already merges every
    # child module's plan under its own attribute name (`language_model.*`).

    def get_mrope_special_token_ids(self) -> dict:
        return {
            "im_start_id": self.image_start_id,
            "im_end_id": self.image_end_id,
            "slice_start_id": self.slice_start_id,
            "slice_end_id": self.slice_end_id,
            "newline_id": self.newline_id,
        }


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
        target_sizes_mrope: list[torch.Tensor] | None = None,
        special_token_ids: dict | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Canvas M-RoPE indices ``(3, B, S)`` plus ``rope_deltas``.

        ``mm_token_type_ids`` is accepted for Qwen-API compatibility; the canvas layout is
        always derived by scanning ``input_ids`` for the structural tokens, so no caller
        supplied bounds are needed. When ``attention_mask`` is set, that scan runs on the
        unpadded tokens.
        """
        del mm_token_type_ids, kwargs  # API compat; the scan below is the single source of truth
        token_ids = special_token_ids if special_token_ids is not None else self.config.get_mrope_special_token_ids()
        if _has_visual_grids(target_sizes_mrope) and not any(v is not None for v in token_ids.values()):
            raise ValueError(
                "Canvas M-RoPE needs the structural token ids but none are set. Populate "
                "`image_start_id` / `image_end_id` / `slice_start_id` / `slice_end_id` / `newline_id` "
                "on the model config (the conversion script resolves them from the tokenizer), or pass "
                "`special_token_ids=` explicitly. Continuing would silently fall back to 1-D positions "
                "and degrade the model on every image and video input."
            )
        return compute_canvas_rope_index(
            input_ids,
            attention_mask,
            target_sizes_mrope=target_sizes_mrope if target_sizes_mrope is not None else [],
            special_token_ids=token_ids,
        )

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        target_sizes_mrope: list[torch.Tensor] | None = None,
        special_token_ids: dict | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Return 4D position ids ``[text, T, H, W]`` for canvas M-RoPE (Qwen-style entrypoint).

        When ``cu_seqlens`` is provided the sequence is treated as packed (``batch=1``,
        multiple documents concatenated). Per-document canvas positions are built and each
        document restarts its own 1-D text counter from zero.
        """
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = target_sizes_mrope is not None

        # Packed training: batch=1, documents delimited by cu_seqlens.
        if cu_seqlens is not None and input_ids is not None and past_key_values_length == 0:
            token_ids = special_token_ids if special_token_ids is not None else self.config.get_mrope_special_token_ids()
            text_position_ids = make_packed_text_position_ids(cu_seqlens, device=input_ids.device)
            if has_multimodal:
                if _has_visual_grids(target_sizes_mrope) and not any(v is not None for v in token_ids.values()):
                    raise ValueError(
                        "Canvas M-RoPE needs the structural token ids but none are set. "
                        "Populate `image_start_id` / `image_end_id` / `slice_start_id` / "
                        "`slice_end_id` / `newline_id` on the model config."
                    )
                mrope_position_ids = compute_canvas_position_ids_packed(
                    input_ids,
                    cu_seqlens=cu_seqlens,
                    target_sizes_mrope=target_sizes_mrope if target_sizes_mrope is not None else [],
                    special_token_ids=token_ids,
                )
                self.rope_deltas = mrope_position_ids.amax(dim=(0, 2)).unsqueeze(1) + 1 - input_ids.shape[1]
                return torch.cat([text_position_ids.unsqueeze(0), mrope_position_ids], dim=0)
            return text_position_ids

        if input_ids is not None and has_multimodal and (self.rope_deltas is None or past_key_values_length == 0):
            mrope_position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                attention_mask=attention_mask,
                target_sizes_mrope=target_sizes_mrope,
                special_token_ids=special_token_ids,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
            text_position_ids = self._text_position_ids(input_ids, attention_mask, past_key_values_length)
            return torch.cat([text_position_ids.unsqueeze(0), mrope_position_ids], dim=0)

        if self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            if input_ids is None:
                if inputs_embeds is None:
                    return None
                batch_size, seq_length = inputs_embeds.shape[:2]
                device = inputs_embeds.device
                text_position_ids = (
                    torch.arange(past_key_values_length, past_key_values_length + seq_length, device=device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            else:
                batch_size = input_ids.shape[0]
                text_position_ids = self._text_position_ids(input_ids, attention_mask, past_key_values_length)
            mrope_position_ids = text_position_ids.unsqueeze(0).expand(3, -1, -1)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            mrope_position_ids = mrope_position_ids + delta.to(device=mrope_position_ids.device)
            return torch.cat([text_position_ids.unsqueeze(0), mrope_position_ids], dim=0)

        if input_ids is not None:
            return expand_1d_position_ids_to_3d(input_ids, attention_mask)

        if has_multimodal:
            logger.warning_once(
                "Canvas M-RoPE needs `input_ids` to locate the visual spans, but only `inputs_embeds` was "
                "given and no cached `rope_deltas` are available. Falling back to 1-D positions, which "
                "degrades multimodal quality. Pass `input_ids` for the first forward pass."
            )
        return None

    @staticmethod
    def _prepare_packed_attention_kwargs(
        kwargs: dict,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | torch.Tensor | None = None,
    ) -> dict:
        """Map packed ``cu_seqlens`` to FlashAttention / FLA ``cu_seq_lens_*`` kwargs.

        Uses the shared ``get_max_seqlen`` utility (passing ``config=self.config``) rather
        than rebuilding the logic inline, as suggested by the reviewer.
        """
        if cu_seqlens is None:
            return kwargs
        cu = cu_seqlens if isinstance(cu_seqlens, torch.Tensor) else torch.as_tensor(cu_seqlens)
        cu = cu.to(dtype=torch.int32)
        if max_seqlen is None:
            max_seqlen_val = int((cu[1:] - cu[:-1]).max().item())
        elif isinstance(max_seqlen, torch.Tensor):
            max_seqlen_val = int(max_seqlen.item())
        else:
            max_seqlen_val = int(max_seqlen)
        kwargs = dict(kwargs)
        kwargs["cu_seq_lens_q"] = cu
        kwargs["cu_seq_lens_k"] = cu
        kwargs["max_length_q"] = max_seqlen_val
        kwargs["max_length_k"] = max_seqlen_val
        return kwargs

    def _text_position_ids(self, input_ids, attention_mask, past_key_values_length=0):
        """Standard 1-D text position ids of shape ``(batch, seq)``."""
        if attention_mask is not None:
            text_positions = attention_mask.long().cumsum(-1) - 1
            return text_positions.masked_fill(attention_mask == 0, 0)
        batch_size, seq_length = input_ids.shape
        return (
            torch.arange(past_key_values_length, past_key_values_length + seq_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

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
        use_cache: bool | None = None,
        downsample_mode: str | None = None,
        target_sizes_mrope: list[torch.Tensor] | None = None,
        special_token_ids: dict | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | torch.Tensor | None = None,
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
        target_sizes_mrope (`list[torch.Tensor]`, *optional*):
            Spatial grid sizes (height, width in patches) per visual crop for canvas M-RoPE.
        special_token_ids (`dict`, *optional*):
            Override for canvas structural token ids; defaults to values on `config`.
        mm_token_type_ids (`torch.IntTensor`, *optional*):
            Modality type ids (text/image/video), matching the Qwen processor contract.
        cu_seqlens (`torch.Tensor` of shape `(num_seqs + 1,)`, *optional*):
            Cumulative sequence lengths for packed-sequence training (batch size must be 1).
            Canvas M-RoPE restarts per document, and FlashAttention uses these boundaries
            for variable-length attention so it does not rebuild lengths from 3D position IDs.
        max_seqlen (`int` or `torch.Tensor`, *optional*):
            Max document length inside the packed sequence; derived from `cu_seqlens` when omitted.
        """
        if cu_seqlens is None:
            cu_seqlens = kwargs.pop("cu_seqlens", None)
        if max_seqlen is None:
            max_seqlen = kwargs.pop("max_seqlen", None)
        kwargs = self._prepare_packed_attention_kwargs(kwargs, cu_seqlens, max_seqlen)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and self.config.image_token_id is not None:
            num_beams = pixel_values.shape[0]
            vision_output = self.get_image_features(pixel_values[:1], target_sizes, downsample_mode=downsample_mode)
            image_features = (
                torch.cat(vision_output.pooler_output, dim=0)
                .to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                .repeat(num_beams, 1)
            )
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features, self.config.image_token_id)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)

        if pixel_values_videos is not None and self.config.video_token_id is not None:
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
                special_token_ids=special_token_ids,
                mm_token_type_ids=mm_token_type_ids,
                cu_seqlens=cu_seqlens,
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
        target_sizes_mrope: list[torch.Tensor] | None = None,
        special_token_ids: dict | None = None,
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
        target_sizes_mrope (`list[torch.Tensor]`, *optional*):
            Spatial grid sizes per visual crop for canvas M-RoPE.
        special_token_ids (`dict`, *optional*):
            Override for canvas structural token ids.
        mm_token_type_ids (`torch.IntTensor`, *optional*):
            Modality type ids from the processor (Qwen-compatible).
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
            special_token_ids=special_token_ids,
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        target_sizes=None,
        pixel_values_videos=None,
        target_sizes_videos=None,
        downsample_mode=None,
        target_sizes_mrope=None,
        special_token_ids=None,
        mm_token_type_ids=None,
        cu_seqlens=None,
        position_ids=None,
        use_cache=True,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            downsample_mode=downsample_mode,
            **kwargs,
        )
        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["target_sizes"] = target_sizes
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["target_sizes_videos"] = target_sizes_videos
            model_inputs["target_sizes_mrope"] = target_sizes_mrope
            model_inputs["special_token_ids"] = special_token_ids
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
            model_inputs["cu_seqlens"] = cu_seqlens
        return model_inputs

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

        if model_kwargs.get("target_sizes_mrope") is not None or model_kwargs.get("mm_token_type_ids") is not None:
            input_ids = model_kwargs.get("input_ids", inputs_tensor)
            mrope_positions, self.model.rope_deltas = self.model.get_rope_index(
                input_ids,
                attention_mask=model_kwargs.get("attention_mask"),
                target_sizes_mrope=model_kwargs.get("target_sizes_mrope"),
                special_token_ids=model_kwargs.get("special_token_ids"),
                mm_token_type_ids=model_kwargs.get("mm_token_type_ids"),
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
        mrope_keys = ("target_sizes_mrope", "special_token_ids", "mm_token_type_ids")
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


__all__ = [
    "MiniCPMV4_7Config",
    "MiniCPMV4_7VisionConfig",
    "MiniCPMV4_7PreTrainedModel",
    "MiniCPMV4_7Model",
    "MiniCPMV4_7ForConditionalGeneration",
]
