# Copyright 2026 H Company and the HuggingFace Inc. team. All rights reserved.
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
"""Processor class for NeoMME."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring
from ...utils.import_utils import requires


class NeoMMEProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": "longest",
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@requires(backends=("torch",))
@auto_docstring
class NeoMMEProcessor(ProcessorMixin):
    r"""
    Constructs a processor that prepares text and images for NeoMME models.

    Plain text is tokenized without retrieval markers. Use [`~NeoMMEProcessor.apply_chat_template`] to format retrieval
    inputs. Images receive a patch grid and two-axis positions.
    """

    valid_processor_kwargs = NeoMMEProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[NeoMMEProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Tokenize text or process images. Retrieval inputs must first be formatted with
        [`~NeoMMEProcessor.apply_chat_template`].

        Returns:
            A [`BatchFeature`] with `input_ids` and `attention_mask`. Image inputs also return `position_ids`,
            and `pixel_values`.
        """
        retrieval_task = kwargs.pop("_retrieval_task", None)
        text_batch = [text] if isinstance(text, str) else text
        expected_prefix_id = getattr(self.tokenizer, f"{retrieval_task}_token_id", None)
        query_mask_counts = (
            [value.count(self.tokenizer.mask_token) for value in text_batch]
            if retrieval_task == "query" and text_batch is not None
            else []
        )

        batch = super().__call__(images=images, text=text, **kwargs)
        input_ids = batch["input_ids"]
        input_ids_list = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        attention_mask = batch.get("attention_mask")
        attention_mask_list = attention_mask.tolist() if hasattr(attention_mask, "tolist") else attention_mask
        for index, input_ids_row in enumerate(input_ids_list):
            non_padded_ids = (
                [
                    token_id
                    for token_id, is_non_padding in zip(input_ids_row, attention_mask_list[index])
                    if is_non_padding
                ]
                if attention_mask_list is not None
                else input_ids_row
            )
            if expected_prefix_id is not None and (not non_padded_ids or non_padded_ids[0] != expected_prefix_id):
                raise ValueError("The NeoMME retrieval template must preserve its leading task marker.")
            if query_mask_counts and input_ids_row.count(self.tokenizer.mask_token_id) != query_mask_counts[index]:
                raise ValueError("Truncation removed NeoMME query expansion tokens; increase `max_length`.")

        image_grid_hw = batch.pop("image_grid_hw", None)
        if image_grid_hw is not None:
            if attention_mask is None:
                raise ValueError("NeoMME image inputs require `return_attention_mask=True`.")
            batch["position_ids"] = self._build_position_ids(
                input_ids,
                attention_mask,
                image_grid_hw,
            )
        return batch

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        videos=None,
        audio=None,
        **kwargs,
    ):
        images, text, videos, audio = super().prepare_inputs_layout(
            images=images,
            text=text,
            videos=videos,
            audio=audio,
            **kwargs,
        )
        if images is not None and text is None:
            images = make_flat_list_of_images(images)
            text = [self.tokenizer.document_token + self.image_token] * len(images)
        return images, text, videos, audio

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        grid_height, grid_width = image_inputs["image_grid_hw"][image_idx]
        row = self.tokenizer.image_token * int(grid_width) + self.tokenizer.row_token
        return self.tokenizer.image_token + row * int(grid_height)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        task: Literal["query", "document"] | None = None,
        processor_kwargs: dict | None = None,
        **kwargs,
    ):
        """Apply the configured retrieval template and optionally tokenize its output.

        When `tokenize=True`, image content must include an image, URL, path, or base64 value. Pass processing
        options such as `max_length` or `max_side` through `processor_kwargs`.
        """
        if kwargs.get("return_assistant_tokens_mask"):
            raise ValueError("NeoMME retrieval templates do not support `return_assistant_tokens_mask`.")

        processor_kwargs = {**(processor_kwargs or {}), "_retrieval_task": task}
        template_kwargs = {"task": task} if task is not None else {}
        return super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            processor_kwargs=processor_kwargs,
            **template_kwargs,
            **kwargs,
        )

    @property
    def model_input_names(self) -> list[str]:
        return [name for name in super().model_input_names if name != "image_grid_hw"] + ["position_ids"]

    def _build_position_ids(self, input_ids, attention_mask, image_grid_hw):
        """Build two-axis positions for batches containing document images."""
        input_ids_list = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        attention_mask = attention_mask.tolist() if hasattr(attention_mask, "tolist") else attention_mask
        image_grid_hw = image_grid_hw.tolist() if hasattr(image_grid_hw, "tolist") else image_grid_hw

        if len({len(input_ids_row) for input_ids_row in input_ids_list}) != 1:
            raise ValueError("NeoMME image batches require padding to a common sequence length.")
        position_ids = np.zeros((2, len(input_ids_list), len(input_ids_list[0])), dtype=np.int64)
        image_index = 0
        for batch_index, (input_ids_row, attention_mask_row) in enumerate(zip(input_ids_list, attention_mask)):
            non_padded_indices = np.flatnonzero(attention_mask_row)
            non_padded_ids = [input_ids_row[index] for index in non_padded_indices]
            if self.image_token_id in non_padded_ids and image_index < len(image_grid_hw):
                grid_height, grid_width = image_grid_hw[image_index]
                image_index += 1
                expected_input_ids = [
                    self.tokenizer.document_token_id,
                    self.image_token_id,
                    *([self.image_token_id] * grid_width + [self.tokenizer.row_token_id]) * grid_height,
                ]
                if non_padded_ids != expected_input_ids:
                    raise ValueError("NeoMME image inputs contain an invalid or truncated token layout.")
                sample_position_ids = self._image_positions(grid_height, grid_width)
            else:
                text_position_ids = np.arange(len(non_padded_indices), dtype=np.int64)
                sample_position_ids = np.stack((text_position_ids, text_position_ids), axis=-1)
            position_ids[:, batch_index, non_padded_indices] = sample_position_ids.T

        if image_index != len(image_grid_hw):
            raise ValueError(f"Got {image_index} image prompts for {len(image_grid_hw)} images.")

        if hasattr(input_ids, "new_tensor"):
            return input_ids.new_tensor(position_ids)
        return position_ids if isinstance(input_ids, np.ndarray) else position_ids.tolist()

    @staticmethod
    def _image_positions(grid_height: int, grid_width: int) -> np.ndarray:
        """Return positions for `<doc> <img>` followed by the patch grid and row markers."""
        position_ids = np.empty((2 + grid_height * (grid_width + 1), 2), dtype=np.int64)
        position_ids[0] = (0, 0)
        position_ids[1] = (1, 1)
        rows = np.broadcast_to(np.arange(grid_height)[:, None], (grid_height, grid_width + 1))
        columns = np.broadcast_to(np.arange(grid_width + 1)[None, :], (grid_height, grid_width + 1))
        position_ids[2:, 0] = 2 + rows.ravel()
        position_ids[2:, 1] = 2 + columns.ravel()
        return position_ids


__all__ = ["NeoMMEProcessor"]
