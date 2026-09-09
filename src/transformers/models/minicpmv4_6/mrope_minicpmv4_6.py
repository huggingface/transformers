# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
"""Canvas M-RoPE position IDs for MiniCPM-V 4.6."""

import math

import torch


def compute_llm_grid(n_tokens: int, vision_grid_height: int, vision_grid_width: int) -> tuple[int, int]:
    if n_tokens <= 0 or vision_grid_height <= 0 or vision_grid_width <= 0:
        return 0, 0
    vit_total = vision_grid_height * vision_grid_width
    if vit_total == n_tokens:
        return vision_grid_height, vision_grid_width
    if vit_total < n_tokens:
        return 0, 0
    factor = int(round(math.sqrt(vit_total / n_tokens)))
    if factor > 0 and vision_grid_height % factor == 0 and vision_grid_width % factor == 0:
        height, width = vision_grid_height // factor, vision_grid_width // factor
        if height * width == n_tokens:
            return height, width
    ratio = vision_grid_height / vision_grid_width
    width = max(1, int(round(math.sqrt(n_tokens / ratio))))
    height = n_tokens // width
    if height * width == n_tokens and height > 0:
        return height, width
    return 0, 0


def build_image_bounds(input_ids: torch.LongTensor, special_token_ids: dict) -> torch.LongTensor:
    start_ids = [
        x for x in (special_token_ids.get("im_start_id"), special_token_ids.get("slice_start_id")) if x is not None
    ]
    end_ids = [x for x in (special_token_ids.get("im_end_id"), special_token_ids.get("slice_end_id")) if x is not None]
    if not start_ids or not end_ids:
        return torch.zeros(0, 2, dtype=torch.long, device=input_ids.device)
    start_cond = torch.zeros_like(input_ids, dtype=torch.bool)
    end_cond = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in start_ids:
        start_cond |= input_ids == sid
    for eid in end_ids:
        end_cond |= input_ids == eid
    starts = torch.where(start_cond)[0] + 1
    ends = torch.where(end_cond)[0]
    n = min(len(starts), len(ends))
    if n == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=input_ids.device)
    return torch.stack([starts[:n], ends[:n]], dim=-1)


def _compute_canvas_single(input_ids, position_ids_2d, image_bound, target_sizes, special_token_ids):
    seq_len = input_ids.shape[0]
    cu_seqlens = torch.tensor([0, seq_len], device=input_ids.device, dtype=torch.long)
    return _compute_canvas_packed(
        position_ids_2d.unsqueeze(0),
        cu_seqlens,
        image_bound,
        target_sizes,
        input_ids.unsqueeze(0),
        special_token_ids,
    )[:, 0, :]


def _group_sequence_images(
    sequence_spans, flat_input_ids, sequence_start, sequence_end, structural_ids, special_token_ids
):
    """
    Group the visual spans of one packed sequence into canvas groups.

    A group is a thumbnail span plus the slice spans that follow it. Consecutive groups whose gap holds only
    structural or newline tokens belong to the same video and are merged into a single ``{"video_frames": [...]}``
    entry, so that every frame later receives its own temporal index.
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

    # merge consecutive tightly-adjacent groups into video groups.
    # Two groups are "tightly adjacent" if the gap between them
    # contains only structural tokens and \n (no real text).
    # This handles both no-slice frames and frames-with-slices.
    newline_id = special_token_ids.get("newline_id")
    gap_ok_ids = structural_ids | ({newline_id} if newline_id is not None else set())

    def _group_end_position(group):
        """Compute the group_end for a raw_group (same logic as main loop)."""
        if group["slices"]:
            last_visual_end = group["slices"][-1][1]
        else:
            last_visual_end = group["thumbnail"][1]
        group_end = last_visual_end
        while group_end < sequence_end and flat_input_ids[group_end].item() in structural_ids:
            group_end += 1
        return min(group_end, sequence_end)

    def _is_tight_gap(previous_group, next_group):
        """Check if only structural/nl tokens sit between two groups."""
        group_end = _group_end_position(previous_group)
        next_group_marker = next_group["thumbnail"][0] - 1  # im_start of next group
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


def _compute_canvas_packed(
    position_ids_2d,
    cu_seqlens,
    image_bound,
    target_sizes,
    input_ids,
    special_token_ids,
):
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
    pos3d = position_ids_2d.unsqueeze(0).expand(3, -1, -1).clone()
    ids_flat = input_ids.view(-1)

    im_start_id = special_token_ids.get("im_start_id")
    im_end_id = special_token_ids.get("im_end_id")
    slice_start_id = special_token_ids.get("slice_start_id")
    slice_end_id = special_token_ids.get("slice_end_id")

    structural_ids = {v for v in (im_start_id, im_end_id, slice_start_id, slice_end_id) if v is not None}

    num_seqs = len(cu_seqlens) - 1
    img_ptr = 0
    num_imgs = len(image_bound)

    for si in range(num_seqs):
        s_start = cu_seqlens[si].item()
        s_end = cu_seqlens[si + 1].item()
        if s_start >= s_end:
            continue

        seq_imgs = []
        while img_ptr < num_imgs:
            bs_ = image_bound[img_ptr, 0].item()
            be_ = image_bound[img_ptr, 1].item()
            if bs_ >= s_start and be_ <= s_end:
                seq_imgs.append((bs_, be_, img_ptr))
                img_ptr += 1
            else:
                break
        if not seq_imgs:
            continue

        # ---- group images: thumbnail + following slices ----
        # merge consecutive no-slice frames into one video group
        groups = _group_sequence_images(seq_imgs, ids_flat, s_start, s_end, structural_ids, special_token_ids)

        # ---- assign 3-D positions ----
        pos = 0
        cursor = s_start

        for group in groups:
            is_video_group = "video_frames" in group

            if is_video_group:
                # === Video group: multiple frames merged ===
                # Each frame gets its own base_f so different frames have
                # distinct spatial positions.  Inter-frame gap tokens (\n
                # etc.) are pure 1D text.  After each gap, pos advances
                # by 1 extra so halo_lo of the next frame does not
                # collide with the last gap token.
                frames = group["video_frames"]
                first_frame = frames[0]
                last_frame = frames[-1]

                # group_start = <im_start> of first frame
                group_start = first_frame["thumbnail"][0] - 1
                # group_end = after last structural token of last frame
                if last_frame["slices"]:
                    last_visual_end = last_frame["slices"][-1][1]
                else:
                    last_visual_end = last_frame["thumbnail"][1]
                group_end = last_visual_end
                while group_end < s_end and ids_flat[group_end].item() in structural_ids:
                    group_end += 1
                group_end = min(group_end, s_end)

                # text before video group
                text_len = group_start - cursor
                if text_len > 0:
                    t_pos = torch.arange(text_len, device=device, dtype=torch.long) + pos
                    for d in range(3):
                        pos3d[d, 0, cursor:group_start] = t_pos
                    pos += text_len

                # process each frame with its own per-frame base
                frame_cursor = group_start
                for fi, vf in enumerate(frames):
                    t_bs, t_be, t_gi = vf["thumbnail"]
                    vf_slices = vf["slices"]
                    im_start_pos = t_bs - 1

                    # --- inter-frame gap: pure 1D text T=H=W=pos ---
                    gap_len = im_start_pos - frame_cursor
                    if gap_len > 0:
                        t_pos = torch.arange(gap_len, device=device, dtype=torch.long) + pos
                        for d in range(3):
                            pos3d[d, 0, frame_cursor:im_start_pos] = t_pos
                        pos += gap_len
                        pos += 1  # buffer: prevents halo_lo collision

                    # --- per-frame base ---
                    base_f = pos
                    halo_lo_f = max(base_f - 1, 0)

                    # per-frame canvas dimensions
                    n_thumb = t_be - t_bs
                    llm_th, llm_tw = compute_llm_grid(
                        n_thumb, target_sizes[t_gi, 0].item(), target_sizes[t_gi, 1].item()
                    )

                    if vf_slices:
                        f_cols = len(vf_slices)
                        for kk in range(len(vf_slices) - 1):
                            if vf_slices[kk + 1][0] - vf_slices[kk][1] > 2:
                                f_cols = kk + 1
                                break
                        f_rows = len(vf_slices) // f_cols if f_cols > 0 else 1
                        if f_rows * f_cols != len(vf_slices):
                            f_rows, f_cols = 1, len(vf_slices)
                        s0_bs, s0_be, s0_gi = vf_slices[0]
                        llm_sh, llm_sw = compute_llm_grid(
                            s0_be - s0_bs, target_sizes[s0_gi, 0].item(), target_sizes[s0_gi, 1].item()
                        )
                        canvas_H = f_rows * llm_sh
                        canvas_W = f_cols * llm_sw
                    else:
                        llm_sh, llm_sw = 0, 0
                        f_rows, f_cols = 0, 0
                        canvas_H = max(llm_th, 1)
                        canvas_W = max(llm_tw, 1)

                    # find per-frame end (after structural tokens)
                    if vf_slices:
                        f_vis_end = vf_slices[-1][1]
                    else:
                        f_vis_end = t_be
                    frame_end = f_vis_end
                    while frame_end < group_end and ids_flat[frame_end].item() in structural_ids:
                        frame_end += 1
                    frame_end = min(frame_end, group_end)

                    # base coat for this frame's span
                    pos3d[0, 0, im_start_pos:frame_end] = base_f
                    pos3d[1, 0, im_start_pos:frame_end] = base_f
                    pos3d[2, 0, im_start_pos:frame_end] = base_f

                    # <im_start> → halo
                    pos3d[1, 0, im_start_pos] = halo_lo_f
                    pos3d[2, 0, im_start_pos] = halo_lo_f

                    # <im_end> → halo (right after thumbnail)
                    im_end_pos = t_be
                    if im_end_pos < frame_end:
                        pos3d[1, 0, im_end_pos] = base_f + canvas_H
                        pos3d[2, 0, im_end_pos] = base_f + canvas_W

                    # thumbnail visual tokens
                    if llm_th > 0 and llm_tw > 0 and llm_th * llm_tw == n_thumb:
                        if vf_slices and canvas_H > 0 and canvas_W > 0:
                            h_c = torch.linspace(0, canvas_H - 1, llm_th, device=device).round().long()
                            w_c = torch.linspace(0, canvas_W - 1, llm_tw, device=device).round().long()
                        else:
                            h_c = torch.arange(llm_th, device=device)
                            w_c = torch.arange(llm_tw, device=device)
                        h_idx = h_c.view(-1, 1).expand(-1, llm_tw).reshape(-1)
                        w_idx = w_c.view(1, -1).expand(llm_th, -1).reshape(-1)
                        pos3d[0, 0, t_bs:t_be] = base_f
                        pos3d[1, 0, t_bs:t_be] = h_idx + base_f
                        pos3d[2, 0, t_bs:t_be] = w_idx + base_f
                    else:
                        fallback = torch.arange(n_thumb, device=device, dtype=torch.long) + base_f
                        for d in range(3):
                            pos3d[d, 0, t_bs:t_be] = fallback

                    # slice visual tokens + slice specials (same as image path)
                    for k, (s_bs, s_be, s_gi) in enumerate(vf_slices):
                        n_s = s_be - s_bs
                        sh_k, sw_k = compute_llm_grid(n_s, target_sizes[s_gi, 0].item(), target_sizes[s_gi, 1].item())
                        gr = k // f_cols
                        gc = k % f_cols
                        h_off = gr * llm_sh
                        w_off = gc * llm_sw

                        slice_start_pos = s_bs - 1
                        if slice_start_pos >= im_start_pos:
                            pos3d[1, 0, slice_start_pos] = base_f + h_off
                            pos3d[2, 0, slice_start_pos] = base_f + w_off

                        slice_end_pos = s_be
                        if sh_k > 0 and sw_k > 0:
                            end_h = h_off + sh_k - 1
                            end_w = w_off + sw_k - 1
                        else:
                            end_h = h_off
                            end_w = w_off
                        if slice_end_pos < frame_end:
                            pos3d[1, 0, slice_end_pos] = base_f + end_h
                            pos3d[2, 0, slice_end_pos] = base_f + end_w

                        if sh_k > 0 and sw_k > 0 and sh_k * sw_k == n_s:
                            h_idx = torch.arange(sh_k, device=device).view(-1, 1).expand(-1, sw_k).reshape(-1)
                            w_idx = torch.arange(sw_k, device=device).view(1, -1).expand(sh_k, -1).reshape(-1)
                            pos3d[0, 0, s_bs:s_be] = base_f
                            pos3d[1, 0, s_bs:s_be] = h_idx + h_off + base_f
                            pos3d[2, 0, s_bs:s_be] = w_idx + w_off + base_f
                        else:
                            fallback = torch.arange(n_s, device=device, dtype=torch.long) + base_f
                            for d in range(3):
                                pos3d[d, 0, s_bs:s_be] = fallback

                    # \n between slice rows → W = right edge + 1
                    # H stays within the ended row (does NOT cross to next row).
                    # Geometry: \n sits at the right-outside of the row that just ended,
                    # at H = the last row of that row's slices, W = canvas_W (one column
                    # past the rightmost slice).
                    if vf_slices:
                        for k in range(len(vf_slices) - 1):
                            gap_s = vf_slices[k][1]
                            gap_e = vf_slices[k + 1][0]
                            if gap_e - gap_s <= 2:
                                continue
                            row_ended = k // f_cols
                            boundary_h = (row_ended + 1) * llm_sh - 1
                            right_edge_w = f_cols * llm_sw
                            for nl_p in range(gap_s + 1, gap_e - 1):
                                pos3d[1, 0, nl_p] = base_f + boundary_h
                                pos3d[2, 0, nl_p] = base_f + right_edge_w

                    pos = base_f + max(canvas_H, canvas_W) + 1
                    frame_cursor = frame_end

                # trailing tokens after last frame (if any)
                if frame_cursor < group_end:
                    trail_len = group_end - frame_cursor
                    t_pos = torch.arange(trail_len, device=device, dtype=torch.long) + pos
                    for d in range(3):
                        pos3d[d, 0, frame_cursor:group_end] = t_pos
                    pos += trail_len

                cursor = group_end

            else:
                # === Single image group (thumbnail + optional slices) ===
                t_bs, t_be, t_gi = group["thumbnail"]
                slices = group["slices"]

                group_start = t_bs - 1
                if slices:
                    last_visual_end = slices[-1][1]
                else:
                    last_visual_end = t_be
                group_end = last_visual_end
                while group_end < s_end and ids_flat[group_end].item() in structural_ids:
                    group_end += 1
                group_end = min(group_end, s_end)

                # text before group
                text_len = group_start - cursor
                if text_len > 0:
                    t_pos = torch.arange(text_len, device=device, dtype=torch.long) + pos
                    for d in range(3):
                        pos3d[d, 0, cursor:group_start] = t_pos
                    pos += text_len

                base = pos
                halo_lo = max(base - 1, 0)

                # compute canvas dimensions
                n_thumb = t_be - t_bs
                llm_th, llm_tw = compute_llm_grid(n_thumb, target_sizes[t_gi, 0].item(), target_sizes[t_gi, 1].item())

                if slices:
                    cols = len(slices)
                    for k in range(len(slices) - 1):
                        gap = slices[k + 1][0] - slices[k][1]
                        if gap > 2:
                            cols = k + 1
                            break
                    rows = len(slices) // cols if cols > 0 else 1
                    if rows * cols != len(slices):
                        rows, cols = 1, len(slices)

                    s0_bs, s0_be, s0_gi = slices[0]
                    llm_sh, llm_sw = compute_llm_grid(
                        s0_be - s0_bs, target_sizes[s0_gi, 0].item(), target_sizes[s0_gi, 1].item()
                    )

                    canvas_H = rows * llm_sh
                    canvas_W = cols * llm_sw
                else:
                    llm_sh, llm_sw = 0, 0
                    canvas_H = max(llm_th, 1)
                    canvas_W = max(llm_tw, 1)

                # base coat
                pos3d[0, 0, group_start:group_end] = base
                pos3d[1, 0, group_start:group_end] = base
                pos3d[2, 0, group_start:group_end] = base

                # <im_start> → halo
                pos3d[1, 0, group_start] = halo_lo
                pos3d[2, 0, group_start] = halo_lo

                # <im_end> → halo
                im_end_pos = t_be
                if im_end_pos < group_end:
                    pos3d[1, 0, im_end_pos] = base + canvas_H
                    pos3d[2, 0, im_end_pos] = base + canvas_W

                # thumbnail visual tokens
                if llm_th > 0 and llm_tw > 0 and llm_th * llm_tw == n_thumb:
                    if slices and canvas_H > 0 and canvas_W > 0:
                        h_c = torch.linspace(0, canvas_H - 1, llm_th, device=device).round().long()
                        w_c = torch.linspace(0, canvas_W - 1, llm_tw, device=device).round().long()
                    else:
                        h_c = torch.arange(llm_th, device=device)
                        w_c = torch.arange(llm_tw, device=device)

                    h_idx = h_c.view(-1, 1).expand(-1, llm_tw).reshape(-1)
                    w_idx = w_c.view(1, -1).expand(llm_th, -1).reshape(-1)

                    pos3d[0, 0, t_bs:t_be] = base
                    pos3d[1, 0, t_bs:t_be] = h_idx + base
                    pos3d[2, 0, t_bs:t_be] = w_idx + base
                else:
                    fallback = torch.arange(n_thumb, device=device, dtype=torch.long) + base
                    for d in range(3):
                        pos3d[d, 0, t_bs:t_be] = fallback

                # slice visual tokens + slice specials
                for k, (s_bs, s_be, s_gi) in enumerate(slices):
                    n_s = s_be - s_bs
                    sh_k, sw_k = compute_llm_grid(n_s, target_sizes[s_gi, 0].item(), target_sizes[s_gi, 1].item())

                    gr = k // cols
                    gc = k % cols
                    h_off = gr * llm_sh
                    w_off = gc * llm_sw

                    # <slice_start_k>
                    slice_start_pos = s_bs - 1
                    if slice_start_pos >= group_start:
                        pos3d[1, 0, slice_start_pos] = base + h_off
                        pos3d[2, 0, slice_start_pos] = base + w_off

                    # <slice_end_k>
                    slice_end_pos = s_be
                    if sh_k > 0 and sw_k > 0:
                        end_h = h_off + sh_k - 1
                        end_w = w_off + sw_k - 1
                    else:
                        end_h = h_off
                        end_w = w_off
                    if slice_end_pos < group_end:
                        pos3d[1, 0, slice_end_pos] = base + end_h
                        pos3d[2, 0, slice_end_pos] = base + end_w

                    # visual tokens
                    if sh_k > 0 and sw_k > 0 and sh_k * sw_k == n_s:
                        h_idx = torch.arange(sh_k, device=device).view(-1, 1).expand(-1, sw_k).reshape(-1)
                        w_idx = torch.arange(sw_k, device=device).view(1, -1).expand(sh_k, -1).reshape(-1)

                        pos3d[0, 0, s_bs:s_be] = base
                        pos3d[1, 0, s_bs:s_be] = h_idx + h_off + base
                        pos3d[2, 0, s_bs:s_be] = w_idx + w_off + base
                    else:
                        fallback = torch.arange(n_s, device=device, dtype=torch.long) + base
                        for d in range(3):
                            pos3d[d, 0, s_bs:s_be] = fallback

                # \n between slice rows → W = right edge + 1
                # H stays within the ended row (does NOT cross to next row).
                # Geometry: \n sits at the right-outside of the row that just ended,
                # at H = the last row of that row's slices, W = canvas_W (one column
                # past the rightmost slice).
                if slices:
                    for k in range(len(slices) - 1):
                        gap_s = slices[k][1]
                        gap_e = slices[k + 1][0]
                        if gap_e - gap_s <= 2:
                            continue
                        row_ended = k // cols
                        boundary_h = (row_ended + 1) * llm_sh - 1
                        right_edge_w = cols * llm_sw
                        for nl_pos in range(gap_s + 1, gap_e - 1):
                            pos3d[1, 0, nl_pos] = base + boundary_h
                            pos3d[2, 0, nl_pos] = base + right_edge_w

                pos = base + max(canvas_H, canvas_W) + 1
                cursor = group_end

        # remaining text
        rem = s_end - cursor
        if rem > 0:
            t_pos = torch.arange(rem, device=device, dtype=torch.long) + pos
            for d in range(3):
                pos3d[d, 0, cursor:s_end] = t_pos

    return pos3d


def uses_mrope_canvas(mrope_mode: str | None) -> bool:
    return mrope_mode is not None and mrope_mode.strip().lower() == "canvas"


def expand_1d_position_ids_to_3d(
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand sequential 1-D positions to ``(3, batch, seq)`` for the Qwen3.5 backbone."""
    if attention_mask is not None:
        return attention_mask.long().cumsum(-1).sub(1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)

    if input_ids.ndim == 1:
        pos = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        return pos.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

    batch_size, seq_len = input_ids.shape
    pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
    return pos.view(1, 1, -1).expand(3, batch_size, -1)


def compute_canvas_position_ids(input_ids, attention_mask, image_bounds, target_sizes, special_token_ids):
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    bsz, seqlen = input_ids.shape
    out = torch.zeros(3, bsz, seqlen, dtype=torch.long, device=input_ids.device)
    if isinstance(image_bounds, torch.Tensor):
        if image_bounds.ndim == 3:
            image_bounds = [image_bounds[b] for b in range(image_bounds.shape[0])]
        else:
            image_bounds = [image_bounds] * bsz
    if isinstance(target_sizes, torch.Tensor):
        if target_sizes.ndim == 3:
            target_sizes = [target_sizes[b] for b in range(target_sizes.shape[0])]
        elif target_sizes.ndim == 2:
            target_sizes = [target_sizes] * bsz
    for b in range(bsz):
        if attention_mask is not None:
            valid_len = int(attention_mask[b].sum().item())
        else:
            valid_len = seqlen
        ids = input_ids[b, :valid_len]
        pos2d = torch.arange(valid_len, device=input_ids.device, dtype=torch.long)
        bounds = (
            image_bounds[b] if b < len(image_bounds) else torch.zeros(0, 2, dtype=torch.long, device=input_ids.device)
        )
        ts = target_sizes[b] if b < len(target_sizes) else torch.zeros(0, 2, dtype=torch.long, device=input_ids.device)
        if bounds.numel() == 0 or ts.numel() == 0:
            out[:, b, :valid_len] = pos2d.unsqueeze(0).expand(3, -1)
        else:
            out[:, b, :valid_len] = _compute_canvas_single(ids, pos2d, bounds, ts, special_token_ids)
    return out


def compute_canvas_rope_index(
    input_ids,
    attention_mask,
    image_bounds,
    target_sizes,
    special_token_ids,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build canvas ``(3, batch, seq)`` position ids and decode ``rope_deltas``."""
    position_ids = compute_canvas_position_ids(
        input_ids, attention_mask, image_bounds, target_sizes, special_token_ids
    )
    if attention_mask is not None:
        seq_lens = attention_mask.sum(-1)
    else:
        seq_lens = torch.full((input_ids.shape[0],), input_ids.shape[1], device=input_ids.device)
    deltas = position_ids.amax(dim=(0, 2)).unsqueeze(1) - seq_lens.unsqueeze(1)
    return position_ids, deltas.long()


def make_packed_text_position_ids(cu_seqlens: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Per-document ``arange`` text positions for a packed sequence, shape ``(1, L)``."""
    cu = cu_seqlens.to(dtype=torch.long)
    if device is None:
        device = cu.device
    else:
        cu = cu.to(device=device)
    total = int(cu[-1].item())
    text_pos = torch.empty(total, dtype=torch.long, device=device)
    for i in range(cu.numel() - 1):
        start, end = int(cu[i].item()), int(cu[i + 1].item())
        text_pos[start:end] = torch.arange(end - start, device=device, dtype=torch.long)
    return text_pos.unsqueeze(0)


def _coerce_flat_hw_tensor(value, device: torch.device) -> torch.Tensor:
    """Flatten processor list/tensor bounds or grids into ``(N, 2)``."""
    if value is None:
        return torch.zeros(0, 2, dtype=torch.long, device=device)
    if isinstance(value, (list, tuple)):
        parts: list[torch.Tensor] = []
        for item in value:
            if item is None:
                continue
            tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
            if tensor.numel() == 0:
                continue
            parts.append(tensor.to(device=device, dtype=torch.long).reshape(-1, 2))
        if not parts:
            return torch.zeros(0, 2, dtype=torch.long, device=device)
        return torch.cat(parts, dim=0)
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() == 0:
        return torch.zeros(0, 2, dtype=torch.long, device=device)
    return tensor.to(device=device, dtype=torch.long).reshape(-1, 2)


def compute_canvas_rope_index_packed(
    input_ids: torch.LongTensor,
    cu_seqlens: torch.Tensor,
    image_bounds,
    target_sizes,
    special_token_ids: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canvas M-RoPE for a packed batch (``input_ids`` shape ``(1, L)``).

    ``image_bounds`` / ``target_sizes`` must use absolute offsets on the packed axis
    (same convention as training-side ``modeling_minicpmv``). Returns
    ``(mrope_position_ids, rope_deltas)`` with mrope shape ``(3, 1, L)``.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError(f"Packed canvas M-RoPE expects batch size 1, got input_ids.shape={tuple(input_ids.shape)}")

    device = input_ids.device
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    if int(cu[-1].item()) != input_ids.shape[1]:
        raise ValueError(f"cu_seqlens[-1]={int(cu[-1].item())} != packed length {input_ids.shape[1]}")

    text_pos = make_packed_text_position_ids(cu, device=device)  # (1, L)
    bounds = _coerce_flat_hw_tensor(image_bounds, device)
    grids = _coerce_flat_hw_tensor(target_sizes, device)
    special_token_ids = special_token_ids if special_token_ids is not None else {}

    mrope = _compute_canvas_packed(
        text_pos,
        cu,
        bounds,
        grids,
        input_ids,
        special_token_ids,
    )
    # rope_deltas: match padded API shape (batch, 1); packed uses batch=1 and total length L.
    seq_lens = torch.tensor([input_ids.shape[1]], device=device, dtype=torch.long)
    deltas = mrope.amax(dim=(0, 2)).unsqueeze(1) - seq_lens.unsqueeze(1)
    return mrope, deltas.long()
