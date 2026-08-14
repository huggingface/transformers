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

import torch

from .modeling_rope_utils import get_mrope_index, get_mrope_vision_positions, uses_mrope


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
