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
from collections import defaultdict

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
        second_per_grid_ts: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """M-RoPE decoder position ids for a `vision + text` sequence: `(position_ids, rope_deltas)`.

        The skeleton every M-RoPE family shares: per batch row, walk `mm_token_type_ids` span by span
        (0=text, 1=image, 2=video) and ask [`get_mrope_position_block`] for each span's `(num_axes, length)`
        positions — which is where families differ, so a model overrides the block (or this whole method for
        a layout that is not span-by-span at all) rather than copying the walk.

        Implementations (overrides included) must stay pure functions of `(self.config, the arguments)` —
        no weights, buffers or devices off `self` — so positions can be computed without a real model: a
        bare object carrying only `config` serves as `self` (how the unit tests drive this, and how the
        exporters rebuild positions for an exported model from its saved config).

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
                Padding mask; positions are placed on the unpadded tokens.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos,)`, *optional*):
                Per-video seconds covered by one temporal grid step, for time-scaled video positions.
            kwargs:
                Ignored, so that callers which forward the whole set of model kwargs (generation does) do not
                have to filter it. A family whose blocks take extra inputs names them in its own override.

        Returns:
            position_ids (`torch.LongTensor` of shape `(num_axes, batch_size, sequence_length)`)
            rope_deltas (`torch.Tensor` of shape `(batch_size, 1)`)
        """
        grids = {1: image_grid_thw, 2: video_grid_thw}

        def positions_for_sequence(token_ids, token_types):
            modality_counter = defaultdict(int)
            current_position = 0
            blocks = []
            for modality_type, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                group = list(group)
                start_idx, end_idx = group[0][0], group[-1][0] + 1
                grid_thw = None
                if modality_type != 0:
                    grid_thw = grids[modality_type][modality_counter[modality_type]]
                block, current_position = self.get_mrope_position_block(
                    modality_type,
                    start_idx,
                    end_idx,
                    current_position,
                    grid_thw=grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    modality_counter=modality_counter,
                    device=token_ids.device,
                )
                modality_counter[modality_type] += 1
                blocks.append(block)
            return torch.cat(blocks, dim=1)

        return _mrope_place_positions(input_ids, mm_token_type_ids, 3, attention_mask, positions_for_sequence)

    def get_mrope_position_block(
        self,
        modality_type: int,
        start_idx: int,
        end_idx: int,
        current_position: int,
        grid_thw: torch.Tensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        modality_counter: dict | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, int]:
        """One modality span's M-RoPE positions: `((num_axes, length) block, next start position)`.

        The default covers the common family: a text span counts 1D positions on every axis; an image or
        video span lays a 3D (temporal, height, width) grid, temporally scaled by
        `tokens_per_second * second_per_grid_ts` when the config declares a clock, and advances the text
        position by the span's largest spatial extent. A family whose spans differ overrides only this — the
        walk in [`get_rope_index`] stays shared.
        """
        if modality_type == 0:
            length = end_idx - start_idx
            return _mrope_positions_block(length, current_position, device=device), current_position + length
        vision_config = getattr(self.config, "vision_config", self.config)
        time_interval = 1
        if modality_type == 2:
            tokens_per_second = getattr(vision_config, "tokens_per_second", None)
            if tokens_per_second is not None and second_per_grid_ts is not None:
                time_interval = tokens_per_second * int(second_per_grid_ts[modality_counter[modality_type]])
        block = get_mrope_vision_positions(
            current_position,
            grid_thw,
            temporal_merge_size=(getattr(vision_config, "temporal_merge_size", None) or 1)
            if modality_type == 2
            else 1,
            spatial_merge_size=vision_config.spatial_merge_size,
            time_interval=time_interval,
            device=device,
        )
        spatial_merge_size = vision_config.spatial_merge_size
        return block, current_position + int(max(grid_thw[1], grid_thw[2])) // spatial_merge_size

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


# ── Multimodal (M-RoPE) decoder positions ────────────────────────────────────────────────────────────
# Shared building blocks for the mixin's `get_rope_index` walk and for models that lay their own spans
# out (`get_mrope_position_block` overrides, or a fully custom `get_rope_index`).


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
