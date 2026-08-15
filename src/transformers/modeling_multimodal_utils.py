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
"""Helpers shared by multimodal models, in the two places a multimodal model needs them.

Multimodal models repeat a handful of methods almost verbatim: locating each modality's placeholder span,
turning image/video grids into decoder position ids, and threading those positions through generation. The
mixins here hold the shared implementations so a model only writes what is actually specific to it, and
override them where a family genuinely differs.
"""

from __future__ import annotations

import itertools

import torch


class MultiModalPreTrainedModelMixin:
    """Shared helpers for a multimodal **base** model (the `<X>Model` that owns the vision/audio towers).

    Mixed into the model class alongside its pretrained base:

    ```python
    class MyVLMModel(MyVLMPreTrainedModel, MultiModalPreTrainedModelMixin): ...
    ```

    Every method is a default, not a contract: a family whose behaviour differs overrides it (and may call
    `super()`), exactly as it would for any inherited method.
    """

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """3D positions (temporal, height, width) for one image/video grid in the decoder sequence.

        Thin method form of [`~modeling_rope_utils.get_mrope_vision_positions`] — see it for the semantics.

        Args:
            start_position (`int`):
                Offset added to all computed positional indices.
            grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
                The (T, H, W) grid representing the feature layout of the current image or video after patch
                embedding.
            temp_merge_size (`int`, *optional*):
                Factor by which the temporal dimension is reduced in the backbone. Defaults to 1.
            spatial_merge_size (`int`, *optional*):
                Factor by which the spatial dimensions (H and W) are reduced in the backbone. Defaults to 1.
            time_interval (`int`, *optional*):
                Spacing factor applied between consecutive temporal position indices. Defaults to 1.
            device (`str` or `torch.device`, *optional*):
                Device on which the resulting tensor is allocated.

        Returns:
            `torch.LongTensor` of shape `(3, sequence_length)`: temporal/height/width positions, flattened
            into sequence form and offset by `start_position`.
        """
        return get_mrope_vision_positions(
            start_position,
            grid_thw,
            temporal_merge_size=temp_merge_size,
            spatial_merge_size=spatial_merge_size,
            time_interval=time_interval,
            device=device,
        )

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """M-RoPE decoder position ids for a `vision + text` sequence: `(position_ids, rope_deltas)`.

        Runs the layout the model's config declares (`config.mrope_layout`) via
        [`~modeling_rope_utils.get_mrope_index`], which is also where the layout's knobs are read off the
        config. Expects a mixed sequence and errors out otherwise; for a pure text sequence rely on the
        model's auto-inferred position ids.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Per-token modality marker (0 for text, 1 for image, 2 for video), returned by the processor.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices; positions are laid out on the
                unpadded tokens.
            kwargs:
                Ignored, so that callers which forward the whole set of model kwargs (generation does) do not
                have to filter it. A family whose layout takes extra inputs — `second_per_grid_ts`,
                `audio_seqlens` — names them in its own override and passes them on.

        Returns:
            position_ids (`torch.LongTensor` of shape `(num_axes, batch_size, sequence_length)`)
            rope_deltas (`torch.Tensor` of shape `(batch_size, 1)`)
        """
        return get_mrope_index(
            self.config,
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        """Position ids for one forward pass, laying out vision spans when there are any.

        Computes them from scratch on a prefill (and caches `rope_deltas` on the model), shifts the cached
        deltas onto plain 1D positions while decoding, and returns `None` when neither is possible so the
        text model infers positions itself.
        """
        if not uses_mrope(self.config):
            # Multimodal, but keeping 1D text positions (no `mrope_section` in its rope parameters): there is
            # nothing multi-axis to lay out, so let the text model infer its own positions.
            return None

        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
                "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
                "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        # Use pre-calculated rope-deltas to infer correct 3D position ids during incremental
        # generation (past_key_values_length > 0) or when only inputs_embeds is provided (no input_ids
        # to recompute from). Skip when input_ids is provided without past_key_values to avoid shape
        # mismatches from stale rope_deltas (e.g., training forward pass after generation).
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
            # Can't build correct 3D positions. Let the model infer it
            position_ids = None
        return position_ids


class MultiModalGenerationMixin:
    """Shared generation-side helpers for a multimodal `<X>ForConditionalGeneration`.

    Mixed in **before** [`GenerationMixin`], so that the overrides here take precedence while still being
    able to call `super()` for the text-only behaviour:

    ```python
    class MyVLMForConditionalGeneration(MyVLMPreTrainedModel, MultiModalGenerationMixin, GenerationMixin): ...
    ```
    """

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Overwritten -- requires multi-axis position ids

        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        if not uses_mrope(self.config):
            # Multimodal, but keeping 1D text positions: the text path already produced what this model wants.
            return text_positions
        return self._prepare_mrope_position_ids_for_generation(text_positions, inputs_tensor, model_kwargs)

    def _prepare_mrope_position_ids_for_generation(self, text_positions, inputs_tensor, model_kwargs):
        """Multi-axis position ids for one generation step, given the 1D `text_positions`.

        Override this rather than [`_prepare_position_ids_for_generation`] when a family lays its axes out
        differently: the text positions arrive as an argument, so an override never has to reason about where
        `super()` lands once this mixin is in the MRO.
        """
        # Early exit in case we are continuing generation from past kv
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            position_ids = text_positions[None, ...] + self.model.rope_deltas
            return position_ids

        # Otherwise compute 3d position ids for vision tokens and concat with text position ids
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if (
            is_input_ids
            and model_kwargs.get("mm_token_type_ids") is not None
            and (model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None)
        ):
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        # Concatenate "text + vision" positions into [4, bs, seq-len]
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        return position_ids

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.full((), vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.full((), image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.full((), video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums


# ── Multimodal (M-RoPE) decoder positions ────────────────────────────────────
# Where a multimodal model's decoder position ids come from: text tokens keep 1D positions while each
# image/video span gets its own axes. `get_mrope_index` is the entry point, `_MROPE_LAYOUT_FUNCTIONS` the
# layouts; a model declares which one it uses (and the knobs it needs) on its config.


def get_mrope_vision_positions(
    start_position: int,
    grid_thw: list[int] | torch.Tensor,
    temporal_merge_size: int = 1,
    spatial_merge_size: int = 1,
    time_interval: float = 1,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """3D M-RoPE positions (temporal, height, width) for one image/video grid, in the *decoder* sequence.

    This is the decoder-side counterpart to [`get_vision_position_ids`] (which produces the vision
    *encoder*'s rotary positions): it lays out a single grid's `(T, H, W)` patches as three position axes,
    each offset by `start_position` (the running position in the token sequence). The temporal axis is
    additionally scaled by `time_interval` (video temporal spacing). Merge sizes downscale the grid the way
    the vision backbone does. Pure function of its arguments — the model passes its config-derived spec.

    Args:
        start_position: running position offset of this grid in the decoder sequence.
        grid_thw: `(3,)` `(T, H, W)` grid of the image/video after patch embedding.
        temporal_merge_size: temporal backbone merge factor (`T // temporal_merge_size`).
        spatial_merge_size: spatial backbone merge factor (`H, W // spatial_merge_size`).
        time_interval: spacing between consecutive temporal position ids. May be fractional, in which case
            the scaled temporal ids are truncated to `dtype`.
        dtype: dtype of the returned positions.
        device: device for the returned tensor.

    Returns:
        `(3, T'*H'*W')` — temporal/height/width position ids, offset by `start_position`.
    """
    dtype = dtype if dtype is not None else torch.long
    llm_grid_t = grid_thw[0].item() // temporal_merge_size
    llm_grid_h = grid_thw[1].item() // spatial_merge_size
    llm_grid_w = grid_thw[2].item() // spatial_merge_size
    position_temporal = (torch.arange(llm_grid_t, device=device) * time_interval).to(dtype)
    position_height = torch.arange(llm_grid_h, dtype=dtype, device=device) + start_position
    position_width = torch.arange(llm_grid_w, dtype=dtype, device=device) + start_position
    t_grid, h_grid, w_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
    vision_position_ids = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
    vision_position_ids[0] += start_position  # temporal offset, after the time_interval scaling
    return vision_position_ids


def uses_mrope(config) -> bool:
    """Whether `config` declares M-RoPE at all — its text rope parameters carry an `mrope_section` (the
    per-axis head split only multi-axis models have). `False` for plain decoders and for VLMs that keep 1D
    text positions (Llava & co.), which is the signal to leave `position_ids` alone."""
    text_config = config.get_text_config()
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "mrope_section" in rope_parameters:
        return True
    return "mrope_section" in (getattr(text_config, "ignore_keys_at_rope_validation", None) or ())


def get_mrope_text_positions(
    attention_mask: torch.Tensor,
    num_axes: int = 3,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain 1D positions broadcast over `num_axes` M-RoPE axes: `(position_ids, rope_deltas)`.

    What a multi-axis model falls back to when a sequence has no vision span to lay out — every token
    (including audio ones, which carry no spatial axes of their own) just counts up, padded slots keeping
    position 1.
    """
    dtype = dtype if dtype is not None else torch.long
    position_ids = attention_mask.to(dtype).cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    position_ids = position_ids.unsqueeze(0).expand(num_axes, -1, -1)
    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    return position_ids, max_position_ids + 1 - attention_mask.sum(dim=-1, keepdim=True)


def _mrope_place_positions(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None,
    num_axes: int,
    attention_mask: torch.Tensor | None,
    positions_for_sequence,
    dtype: torch.dtype | None = None,
    padded_position: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared frame every M-RoPE layout sits in: `(position_ids, rope_deltas)`.

    Per batch row it drops the padded tokens, asks `positions_for_sequence(token_ids, token_types)` for that
    sequence's `(num_axes, seq)` positions — which is where the layouts differ — scatters them back onto the
    unpadded slots, and records the delta `generate` advances decode positions by. `dtype` defaults to
    `input_ids`' (a layout that lays out fractional temporal positions asks for a float one) and
    `padded_position` is what the masked-out slots keep.
    """
    position_ids = torch.full(
        (num_axes, input_ids.shape[0], input_ids.shape[1]),
        padded_position,
        dtype=dtype or input_ids.dtype,
        device=input_ids.device,
    )
    rope_deltas = []
    for batch_idx, token_ids in enumerate(input_ids):
        token_types = mm_token_type_ids[batch_idx] if mm_token_type_ids is not None else None
        valid_tokens = None
        if attention_mask is not None:
            valid_tokens = attention_mask[batch_idx].bool()
            token_ids = token_ids[valid_tokens]
            token_types = token_types[valid_tokens] if token_types is not None else None

        positions = positions_for_sequence(token_ids, token_types)
        positions = positions.to(device=position_ids.device, dtype=position_ids.dtype)

        if valid_tokens is not None:
            position_ids[:, batch_idx, valid_tokens] = positions
        else:
            position_ids[:, batch_idx] = positions
        rope_deltas.append(positions.max() + 1 - len(token_ids))
    return position_ids, torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)


def _mrope_index_interleaved_runs(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    *,
    spatial_merge_size: int,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    tokens_per_second: float | None = None,
    split_video_frames: bool = False,
    video_temporal_merge_size: int = 1,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-axis M-RoPE laid out run by run: `(position_ids, mrope_deltas)`.

    Text runs use standard 1D positions (broadcast across the 3 axes); image/video runs use 3D
    (temporal, height, width) positions from [`get_mrope_vision_positions`]. Runs are delimited by
    `mm_token_type_ids` (0=text, 1=image, 2=video). A pure function driven entirely by the caller's spec —
    each VLM's `get_rope_index` becomes a thin wrapper passing its config-derived values, so the layout
    lives here instead of being copied per model.

    Args:
        input_ids: `(batch, seq)` token ids.
        mm_token_type_ids: `(batch, seq)` per-token modality marker (0=text, 1=image, 2=video).
        spatial_merge_size: spatial backbone merge factor.
        attention_mask: `(batch, seq)` padding mask; positions are placed on the unpadded tokens.
        image_grid_thw: `(num_images, 3)` grids, consumed in order for image runs.
        video_grid_thw: `(num_videos, 3)` grids, consumed in order for video runs.
        second_per_grid_ts: `(num_videos,)` per-video seconds-per-temporal-grid.
        tokens_per_second: when set, video temporal spacing scales by `tokens_per_second *
            second_per_grid_ts`; otherwise consecutive temporal ids are 1 apart.
        split_video_frames: expand each video grid to one `T=1` row per frame before placement, for
            processors that separate frames with timestamps.
        video_temporal_merge_size: temporal backbone merge factor, applied to video grids only.

    Returns:
        position_ids: `(3, batch, seq)` long.
        mrope_position_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row.
    """
    if split_video_frames and video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }
    second_per_grid_iter = (
        iter(second_per_grid_ts) if second_per_grid_ts is not None else iter([1] * input_ids.shape[1])
    )

    def positions_for_sequence(token_ids, token_types):
        input_type_group = []
        for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(_mrope_positions_block(text_len, current_pos, device=token_ids.device))
                current_pos += text_len
            else:
                grid_thw = next(grid_iters[modality_type])
                if modality_type == 2 and tokens_per_second is not None:
                    time_interval = tokens_per_second * int(next(second_per_grid_iter))
                else:
                    time_interval = 1
                vision_position_ids = get_mrope_vision_positions(
                    current_pos,
                    grid_thw,
                    temporal_merge_size=video_temporal_merge_size if modality_type == 2 else 1,
                    spatial_merge_size=spatial_merge_size,
                    time_interval=time_interval,
                    device=token_ids.device,
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        return torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

    return _mrope_place_positions(input_ids, mm_token_type_ids, 3, attention_mask, positions_for_sequence)


def _mrope_index_indexed_images(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    *,
    num_axes: int,
    spatial_merge_size: int,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """M-RoPE over a per-image ordinal instead of a temporal axis: `(position_ids, rope_deltas)`.

    HunYuanVL's layout. Every token starts from plain 1D positions (a text baseline that is *overwritten*,
    not accumulated) and each image span replaces its slice with `(width, height, image_index)` on the last
    three of `mrope_section`'s axes — earlier axes keep the 1D positions. The pooled image grid carries one
    newline-style token per row, so the width channel spans `w + 1` (see `get_mrope_image_positions`), and a
    span may include the two image-boundary tokens.

    Returns:
        position_ids: `(num_axes, batch, seq)` long — unlike the other layouts this is not fixed at 3 axes,
            since `mrope_section` decides how many there are and only the last 3 carry the image.
        rope_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row.
    """
    if num_axes < 3:
        raise ValueError(f"An indexed-image M-RoPE layout needs at least 3 axes, got {num_axes}.")

    grid_iter = iter(image_grid_thw) if image_grid_thw is not None else None
    image_index = 0

    def positions_for_sequence(token_ids, token_types):
        nonlocal image_index
        current_position_ids = torch.arange(token_ids.shape[-1], dtype=input_ids.dtype, device=token_ids.device)
        current_position_ids = current_position_ids.view(1, -1).expand(num_axes, -1).clone()
        if grid_iter is None:
            return current_position_ids

        for modality_type, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
            if modality_type != 1:  # image == 1; this layout has no video/audio modality
                continue
            group = list(group)
            span_start, span_end = group[0][0], group[-1][0] + 1
            try:
                grid_thw = next(grid_iter)
            except StopIteration as error:
                raise ValueError("Found more image placeholder spans than entries in `image_grid_thw`.") from error

            vision_position_ids = get_mrope_image_positions(
                grid_thw[1:], spatial_merge_size=spatial_merge_size, device=token_ids.device
            )
            grid_tokens = vision_position_ids.shape[1]
            span_length = span_end - span_start
            if span_length == grid_tokens + 2:  # span includes the image-boundary tokens
                grid_start = span_start + 1
            elif span_length == grid_tokens:
                grid_start = span_start
            else:
                raise ValueError(
                    "Image placeholder span length does not match `image_grid_thw`: "
                    f"span_length={span_length}, expected {grid_tokens} or {grid_tokens + 2}."
                )

            grid_end = grid_start + grid_tokens
            offset = num_axes - 3
            current_position_ids[offset : offset + 2, grid_start:grid_end] = vision_position_ids.to(
                dtype=input_ids.dtype
            )
            current_position_ids[offset + 2, grid_start:grid_end] = image_index
            image_index += 1
        return current_position_ids

    position_ids, rope_deltas = _mrope_place_positions(
        input_ids, mm_token_type_ids, num_axes, attention_mask, positions_for_sequence
    )
    if image_grid_thw is not None and image_index != len(image_grid_thw):
        raise ValueError(
            "Number of image placeholder spans does not match `image_grid_thw`: "
            f"spans={image_index}, images={len(image_grid_thw)}."
        )
    return position_ids, rope_deltas


def get_mrope_image_positions(
    grid_hw: list[int] | torch.Tensor,
    spatial_merge_size: int = 1,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """`(2, tokens)` `(width, height)` indices for one pooled image grid, for the indexed-image layout.

    The vision merger appends one newline-style token per image row, so the width channel spans `w + 1`
    positions while the height channel repeats each row id over that extra column.
    """
    grid_h, grid_w = (int(value) for value in grid_hw)
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    position_height, position_width = torch.meshgrid(
        torch.arange(llm_grid_h, dtype=torch.long, device=device),
        torch.arange(llm_grid_w + 1, dtype=torch.long, device=device),
        indexing="ij",
    )
    return torch.stack([position_width.flatten(), position_height.flatten()], dim=0)


def _mrope_positions_block(
    length: int | torch.Tensor,
    start_position: int | torch.Tensor,
    num_axes: int = 3,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """`(num_axes, length)` block of consecutive positions from `start_position`, the same on every axis.

    What a text run — or an audio span, which is 1D in time — contributes to a multi-axis layout.
    """
    dtype = dtype if dtype is not None else torch.long
    return torch.arange(int(length), dtype=dtype, device=device).view(1, -1).expand(num_axes, -1) + start_position


def _mrope_temporal_chunks(
    temporal_positions: torch.Tensor, positions_per_chunk: int, start_position: int | torch.Tensor
) -> list[tuple[int, int]]:
    """`(start, end)` slices cutting a monotonic temporal axis into successive `positions_per_chunk` ranges.

    Used to interleave a video and its own audio track in time: both are cut at the same chunk boundaries
    (`position_id_per_seconds * seconds_per_chunk` positions ≈ one chunk of wall-clock time), then emitted
    chunk by chunk. `start_position` is the span's own origin, subtracted before chunking.
    """
    chunks = []
    chunk_start, chunk_index = 0, 1
    for index in range(len(temporal_positions)):
        if temporal_positions[index] - start_position >= chunk_index * positions_per_chunk:
            chunks.append((chunk_start, index))
            chunk_start = index
            chunk_index += 1
    chunks.append((chunk_start, len(temporal_positions)))
    return chunks


def _mrope_index_audio_chunked(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None = None,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    audio_start_token_id: int,
    position_id_per_seconds: float,
    seconds_per_chunk: float,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    audio_seqlens: torch.LongTensor | None = None,
    use_audio_in_video: bool = False,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-axis M-RoPE over image, video **and audio** spans, audio-in-video interleaved in time chunks.

    Qwen2.5-Omni's layout. Spans are located by scanning `input_ids` for placeholder tokens rather than by
    grouping `mm_token_type_ids` (which this family does not produce): at each step the *nearest* upcoming
    image/video/audio placeholder wins, and its span contributes a one-position `bos` block, the modality's
    own positions, and a one-position `eos` block, each starting one past the previous block's maximum.
    Audio is 1D in time, images and videos are 3D (see [`get_mrope_vision_positions`]), and the temporal axis
    counts `position_id_per_seconds` positions per second of media. With `use_audio_in_video`, a video and
    its soundtrack share one span: both are cut into `seconds_per_chunk` chunks by
    [`_mrope_temporal_chunks`] and emitted video-chunk-then-audio-chunk, wrapped in doubled bos/eos blocks.

    Args:
        input_ids: `(batch, seq)` token ids.
        mm_token_type_ids: unused — spans come from the placeholder token ids.
        spatial_merge_size: spatial backbone merge factor.
        image_token_id, video_token_id, audio_token_id: placeholder token ids marking each modality's span.
        vision_start_token_id, audio_start_token_id: the tokens opening a vision/audio span, used to count
            the spans in the sequence.
        position_id_per_seconds: temporal positions per second of media.
        seconds_per_chunk: audio-in-video interleaving granularity, in seconds.
        attention_mask: `(batch, seq)` padding mask; positions are placed on the unpadded tokens.
        image_grid_thw: `(num_images, 3)` grids, consumed in order for image spans.
        video_grid_thw: `(num_videos, 3)` grids, consumed in order for video spans.
        second_per_grid_ts: `(num_videos,)` per-video seconds-per-temporal-grid.
        audio_seqlens: `(num_audios,)` mel-frame lengths, consumed in order for audio spans.
        use_audio_in_video: lay a video out interleaved with its own audio track.

    Returns:
        position_ids: `(3, batch, seq)` long, masked-out slots left at 1.
        mrope_position_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row.
    """
    image_idx, video_idx, audio_idx = 0, 0, 0

    def positions_for_sequence(token_ids, _token_types):
        nonlocal image_idx, video_idx, audio_idx
        device = token_ids.device
        input_tokens = token_ids.tolist()
        blocks = []

        def next_position():
            return blocks[-1].max() + 1 if blocks else 0

        def block(length, start_position=None):
            start_position = next_position() if start_position is None else start_position
            return _mrope_positions_block(length, start_position, device=device)

        vision_start_indices = torch.argwhere(token_ids == vision_start_token_id).squeeze(1)
        vision_tokens = token_ids[vision_start_indices + 1]
        num_audios = (token_ids == audio_start_token_id).sum()
        num_images = (vision_tokens == image_token_id).sum()
        num_videos = (
            (vision_tokens == audio_start_token_id).sum()
            if use_audio_in_video
            else (vision_tokens == video_token_id).sum()
        )
        remain_images, remain_videos, remain_audios = num_images, num_videos, num_audios
        num_spans = num_images + num_audios if use_audio_in_video else num_images + num_videos + num_audios

        st = 0
        for _ in range(num_spans):
            unreachable = len(input_tokens) + 1
            ed_image = (
                input_tokens.index(image_token_id, st)
                if image_token_id in input_tokens and remain_images > 0
                else unreachable
            )
            ed_video = (
                input_tokens.index(video_token_id, st)
                if video_token_id in input_tokens and remain_videos > 0
                else unreachable
            )
            ed_audio = (
                input_tokens.index(audio_token_id, st)
                if audio_token_id in input_tokens and remain_audios > 0
                else unreachable
            )
            min_ed = min(ed_image, ed_video, ed_audio)
            # the span's own opening token sits between the text run and the placeholders (two of them for
            # audio-in-video, which opens with both a vision and an audio start token)
            text_len = min_ed - st - (2 if min_ed == ed_video and use_audio_in_video else 1)
            if text_len != 0:
                blocks.append(block(text_len))
            bos_len = eos_len = 1

            if min_ed == ed_audio:
                blocks.append(block(bos_len))
                audio_len = _mrope_audio_length_pooled(audio_seqlens[audio_idx])
                blocks.append(block(audio_len))
                blocks.append(block(eos_len))
                st += int(text_len + bos_len + audio_len + eos_len)
                audio_idx += 1
                remain_audios -= 1

            elif min_ed == ed_image:
                blocks.append(block(bos_len))
                grid_thw = image_grid_thw[image_idx]
                blocks.append(
                    get_mrope_vision_positions(
                        next_position(),
                        grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        time_interval=position_id_per_seconds,
                        device=device,
                    )
                )
                image_len = grid_thw.prod() // (spatial_merge_size**2)
                blocks.append(block(eos_len))
                st += int(text_len + bos_len + image_len + eos_len)
                image_idx += 1
                remain_images -= 1

            elif min_ed == ed_video and not use_audio_in_video:
                blocks.append(block(bos_len))
                grid_thw = video_grid_thw[video_idx]
                blocks.append(
                    get_mrope_vision_positions(
                        next_position(),
                        grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        time_interval=float(second_per_grid_ts[video_idx]) * position_id_per_seconds,
                        device=device,
                    )
                )
                video_len = grid_thw.prod() // (spatial_merge_size**2)
                blocks.append(block(eos_len))
                st += int(text_len + bos_len + video_len + eos_len)
                video_idx += 1
                remain_videos -= 1

            elif min_ed == ed_video and use_audio_in_video:
                bos_position = next_position()
                blocks.append(block(bos_len, bos_position))
                blocks.append(block(bos_len, bos_position))

                span_start = next_position()
                grid_thw = video_grid_thw[video_idx]
                audio_len = _mrope_audio_length_pooled(audio_seqlens[audio_idx])
                audio_positions = block(audio_len, span_start)
                video_positions = get_mrope_vision_positions(
                    span_start,
                    grid_thw,
                    spatial_merge_size=spatial_merge_size,
                    time_interval=float(second_per_grid_ts[video_idx]) * position_id_per_seconds,
                    device=device,
                )
                positions_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                video_chunks = _mrope_temporal_chunks(video_positions[0], positions_per_chunk, span_start)
                audio_chunks = _mrope_temporal_chunks(audio_positions[0], positions_per_chunk, span_start)
                for video_chunk, audio_chunk in itertools.zip_longest(video_chunks, audio_chunks):
                    if video_chunk is not None:
                        blocks.append(video_positions[:, video_chunk[0] : video_chunk[1]])
                    if audio_chunk is not None:
                        blocks.append(audio_positions[:, audio_chunk[0] : audio_chunk[1]])
                video_len = grid_thw.prod() // (spatial_merge_size**2)

                eos_position = next_position()
                blocks.append(block(eos_len, eos_position))
                blocks.append(block(eos_len, eos_position))
                st += int(text_len + 2 * bos_len + audio_len + video_len + 2 * eos_len)
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

        if st < len(input_tokens):
            blocks.append(block(len(input_tokens) - st))
        return torch.cat(blocks, dim=1).reshape(3, -1)

    return _mrope_place_positions(
        input_ids, mm_token_type_ids, 3, attention_mask, positions_for_sequence, padded_position=1
    )


def _mrope_index_audio_merged(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None = None,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    audio_start_token_id: int,
    position_id_per_seconds: float,
    audio_window_size: int,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    audio_seqlens: torch.LongTensor | None = None,
    use_audio_in_video: bool = False,
    **unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-axis M-RoPE over image, video and audio spans, audio-in-video merged position by position.

    Qwen3-Omni's layout. Like [`_mrope_index_audio_chunked`] it locates spans by scanning `input_ids`, but
    it scans for the *opening* tokens (`vision_start_token_id` / `audio_start_token_id`) — so the text run
    ends where the opening token starts, and positions advance by counting emitted tokens instead of
    re-reading the previous block's maximum. Audio-in-video is merged position by position (whichever of the
    two streams is earlier in time goes next) rather than in chunks, and its span is wrapped in two-position
    bos/eos blocks. Positions are float, since a fractional `second_per_grid_ts` leaves the temporal axis
    fractional here.

    Args:
        input_ids: `(batch, seq)` token ids.
        mm_token_type_ids: unused — spans come from the opening token ids.
        spatial_merge_size: spatial backbone merge factor.
        image_token_id, video_token_id, audio_token_id: placeholder token ids marking each modality's span.
        vision_start_token_id, audio_start_token_id: the tokens opening a vision/audio span.
        position_id_per_seconds: temporal positions per second of media.
        audio_window_size: audio encoder window (`audio_config.n_window`), which sets how many decoder
            tokens an audio of a given mel length takes.
        attention_mask: `(batch, seq)` padding mask; positions are placed on the unpadded tokens.
        image_grid_thw: `(num_images, 3)` grids, consumed in order for image spans.
        video_grid_thw: `(num_videos, 3)` grids, consumed in order for video spans.
        second_per_grid_ts: `(num_videos,)` per-video seconds-per-temporal-grid.
        audio_seqlens: `(num_audios,)` mel-frame lengths, consumed in order for audio spans.
        use_audio_in_video: lay a video out interleaved with its own audio track.

    Returns:
        position_ids: `(3, batch, seq)` float.
        mrope_position_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row.
    """
    image_idx, video_idx, audio_idx = 0, 0, 0

    def positions_for_sequence(token_ids, _token_types):
        nonlocal image_idx, video_idx, audio_idx
        device = token_ids.device
        input_tokens = token_ids.tolist()
        blocks = []

        def block(length, start_position):
            return _mrope_positions_block(length, start_position, dtype=torch.float, device=device)

        vision_start_indices = torch.argwhere(token_ids == vision_start_token_id).squeeze(1)
        vision_tokens = token_ids[vision_start_indices + 1]
        num_audios = (token_ids == audio_start_token_id).sum()
        num_images = (vision_tokens == image_token_id).sum()
        num_videos = (
            (vision_tokens == audio_start_token_id).sum()
            if use_audio_in_video
            else (vision_tokens == video_token_id).sum()
        )
        remain_images, remain_videos, remain_audios = num_images, num_videos, num_audios
        num_spans = num_images + num_audios if use_audio_in_video else num_images + num_videos + num_audios

        st = 0
        for _ in range(num_spans):
            start_position = blocks[-1].max() + 1 if blocks else 0
            unreachable = len(input_tokens) + 1
            ed_vision_start = (
                input_tokens.index(vision_start_token_id, st)
                if (image_token_id in input_tokens or video_token_id in input_tokens)
                and (remain_videos > 0 or remain_images > 0)
                else unreachable
            )
            ed_audio_start = (
                input_tokens.index(audio_start_token_id, st)
                if audio_token_id in input_tokens and remain_audios > 0
                else unreachable
            )
            min_ed = min(ed_vision_start, ed_audio_start)

            text_len = min_ed - st
            if text_len != 0:
                blocks.append(block(text_len, start_position))
                start_position += text_len

            audio_in_video = min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start
            bos_len = eos_len = 2 if audio_in_video else 1
            blocks.append(block(bos_len, start_position))
            start_position += bos_len

            if min_ed == ed_audio_start:
                audio_len = _mrope_audio_length_windowed(audio_seqlens[audio_idx], audio_window_size)
                blocks.append(block(audio_len, start_position))
                st += int(text_len + bos_len + audio_len + eos_len)
                audio_idx += 1
                remain_audios -= 1

            elif min_ed == ed_vision_start and token_ids[ed_vision_start + 1] == image_token_id:
                grid_thw = image_grid_thw[image_idx]
                blocks.append(
                    get_mrope_vision_positions(
                        start_position,
                        grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        time_interval=position_id_per_seconds,
                        dtype=torch.float,
                        device=device,
                    )
                )
                st += int(text_len + bos_len + grid_thw.prod() // (spatial_merge_size**2) + eos_len)
                image_idx += 1
                remain_images -= 1

            elif min_ed == ed_vision_start and token_ids[ed_vision_start + 1] == video_token_id:
                grid_thw = video_grid_thw[video_idx]
                blocks.append(
                    get_mrope_vision_positions(
                        start_position,
                        grid_thw,
                        spatial_merge_size=spatial_merge_size,
                        time_interval=float(second_per_grid_ts[video_idx]) * position_id_per_seconds,
                        dtype=torch.float,
                        device=device,
                    )
                )
                st += int(text_len + bos_len + grid_thw.prod() // (spatial_merge_size**2) + eos_len)
                video_idx += 1
                remain_videos -= 1

            elif audio_in_video:
                grid_thw = video_grid_thw[video_idx]
                audio_len = _mrope_audio_length_windowed(audio_seqlens[audio_idx], audio_window_size)
                audio_positions = block(audio_len, start_position)
                video_positions = get_mrope_vision_positions(
                    start_position,
                    grid_thw,
                    spatial_merge_size=spatial_merge_size,
                    time_interval=float(second_per_grid_ts[video_idx]) * position_id_per_seconds,
                    dtype=torch.float,
                    device=device,
                )
                # merge the two streams by temporal position, one token at a time, video first on a tie
                video_pos, audio_pos = 0, 0
                while video_pos < video_positions.shape[-1] and audio_pos < audio_positions.shape[-1]:
                    if video_positions[0][video_pos] <= audio_positions[0][audio_pos]:
                        blocks.append(video_positions[:, video_pos : video_pos + 1])
                        video_pos += 1
                    else:
                        blocks.append(audio_positions[:, audio_pos : audio_pos + 1])
                        audio_pos += 1
                if video_pos < video_positions.shape[-1]:
                    blocks.append(video_positions[:, video_pos:])
                if audio_pos < audio_positions.shape[-1]:
                    blocks.append(audio_positions[:, audio_pos:])
                video_len = grid_thw.prod() // (spatial_merge_size**2)
                st += int(text_len + bos_len + audio_len + video_len + eos_len)
                audio_idx += 1
                video_idx += 1
                remain_videos -= 1
                remain_audios -= 1

            blocks.append(block(eos_len, blocks[-1].max() + 1))

        if st < len(input_tokens):
            blocks.append(block(len(input_tokens) - st, blocks[-1].max() + 1 if blocks else 0))
        return torch.cat(blocks, dim=1).reshape(3, -1)

    return _mrope_place_positions(
        input_ids, mm_token_type_ids, 3, attention_mask, positions_for_sequence, dtype=torch.float
    )


def _mrope_audio_length_pooled(audio_seqlen: torch.Tensor | int) -> torch.Tensor | int:
    """Decoder tokens one audio takes in the chunked layout: two stride-2 convs, then a stride-2 pooler."""
    return ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1


def _mrope_audio_length_windowed(audio_seqlen: torch.Tensor | int, window_size: int) -> torch.Tensor | int:
    """Decoder tokens one audio takes in the merged layout: three stride-2 convs over fixed-size windows,
    each full `2 * window_size`-frame window collapsing to 13 tokens."""
    chunk_len = window_size * 2
    feat_len = (audio_seqlen % chunk_len - 1) // 2 + 1
    return ((feat_len - 1) // 2 + 1 - 1) // 2 + 1 + (audio_seqlen // chunk_len) * 13


# Layout name -> implementation, the way `ACT2FN` maps an activation name. A model declares which one it
# uses with `config.mrope_layout`; a family whose positions are laid out differently adds its function here
# (and names it in its config) rather than growing an existing layout. Every layout is a pure function of
# `(config, inputs)`, which is what lets a model's `get_rope_index` be a one-line wrapper AND lets the
# exporters compute positions from a saved config with no model instance.
_MROPE_LAYOUT_FUNCTIONS: dict[str, callable] = {
    "interleaved_runs": _mrope_index_interleaved_runs,
    "indexed_images": _mrope_index_indexed_images,
    "audio_chunked": _mrope_index_audio_chunked,
    "audio_merged": _mrope_index_audio_merged,
}


def get_mrope_index(
    config,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None = None,
    *,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
    audio_seqlens: torch.LongTensor | None = None,
    use_audio_in_video: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """M-RoPE decoder positions for a `vision + text` sequence: `(position_ids, rope_deltas)`.

    Runs the layout `config.mrope_layout` declares, with that layout's knobs read off the config here — the
    one place config is turned into M-RoPE parameters, so it works from a saved config with no model
    instance (a VLM's `get_rope_index` is a one-line wrapper over this, and the exporters drive exported
    VLMs through the very same call). The layouts themselves (`_MROPE_LAYOUT_FUNCTIONS`) take plain values,
    so they stay usable and testable on their own.

    Args:
        config: the model config, declaring `mrope_layout` and its layout's knobs.
        input_ids: `(batch, seq)` token ids.
        mm_token_type_ids: `(batch, seq)` per-token modality marker (0=text, 1=image, 2=video); layouts that
            locate their spans by placeholder token id instead do not need it.
        image_grid_thw: `(num_images, 3)` grids, consumed in order for image spans.
        video_grid_thw: `(num_videos, 3)` grids, consumed in order for video spans.
        attention_mask: `(batch, seq)` padding mask; positions are placed on the unpadded tokens.
        second_per_grid_ts: `(num_videos,)` per-video seconds-per-temporal-grid.
        audio_seqlens: `(num_audios,)` mel-frame lengths, consumed in order for audio spans.
        use_audio_in_video: lay each video out interleaved with its own audio track.

    A layout ignores the tensors it has no use for (an image-only layout takes no video grids).

    Returns:
        position_ids: `(num_axes, batch, seq)` — 3 axes for every layout except `indexed_images`, which
            takes its axis count from `mrope_section`; long, except layouts whose temporal axis is
            fractional (`audio_merged`), which return float.
        rope_deltas: `(batch, 1)` — `max_position + 1 - unpadded_len` per batch row, in every layout.
    """
    layout_name = getattr(config, "mrope_layout", None)
    layout = _MROPE_LAYOUT_FUNCTIONS.get(layout_name)
    if layout is None:
        raise NotImplementedError(
            f"`{getattr(config, 'model_type', type(config).__name__)}` declares "
            f"`mrope_layout={layout_name!r}`, which is not implemented here (known layouts: "
            f"{sorted(_MROPE_LAYOUT_FUNCTIONS)}). Its own `get_rope_index` is the reference: port it to a "
            "`_MROPE_LAYOUT_FUNCTIONS` entry (a pure function of its inputs) and name it in the config."
        )
    vision_config = getattr(config, "vision_config", config)
    audio_config = getattr(config, "audio_config", config)
    rope_parameters = getattr(config.get_text_config(), "rope_parameters", None) or {}
    return layout(
        input_ids,
        mm_token_type_ids,
        spatial_merge_size=vision_config.spatial_merge_size,
        tokens_per_second=getattr(vision_config, "tokens_per_second", None),
        video_temporal_merge_size=getattr(vision_config, "temporal_merge_size", None) or 1,
        split_video_frames=getattr(vision_config, "timestamped_video_frames", False),
        num_axes=len(rope_parameters.get("mrope_section", [])),
        image_token_id=getattr(config, "image_token_id", None),
        video_token_id=getattr(config, "video_token_id", None),
        audio_token_id=getattr(config, "audio_token_id", None),
        vision_start_token_id=getattr(config, "vision_start_token_id", None),
        audio_start_token_id=getattr(config, "audio_start_token_id", None),
        position_id_per_seconds=getattr(config, "position_id_per_seconds", None),
        seconds_per_chunk=getattr(config, "seconds_per_chunk", None),
        audio_window_size=getattr(audio_config, "n_window", None),
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        audio_seqlens=audio_seqlens,
        use_audio_in_video=use_audio_in_video,
    )
