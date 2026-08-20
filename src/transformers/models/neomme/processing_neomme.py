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
from ...utils.chat_template_utils import _get_template_variables
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
        expected_prefix = getattr(self.tokenizer, f"{retrieval_task}_token_id", None)
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
        for index, ids in enumerate(input_ids_list):
            active_ids = (
                [token_id for token_id, keep in zip(ids, attention_mask_list[index]) if keep]
                if attention_mask_list is not None
                else ids
            )
            if expected_prefix is not None and (not active_ids or active_ids[0] != expected_prefix):
                raise ValueError("The NeoMME retrieval template must preserve its leading task marker.")
            if query_mask_counts and ids.count(self.tokenizer.mask_token_id) != query_mask_counts[index]:
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

        template = chat_template or self.chat_template
        if isinstance(self.chat_template, dict):
            template = self.chat_template.get(chat_template or "default", template)
        if isinstance(template, str) and "task" not in _get_template_variables(template):
            raise ValueError("NeoMME retrieval templates must use the `task` variable.")

        processor_kwargs = {**(processor_kwargs or {}), "_retrieval_task": task}
        return super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            processor_kwargs=processor_kwargs,
            task=task,
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

        if len({len(ids) for ids in input_ids_list}) != 1:
            raise ValueError("NeoMME image batches require padding to a common sequence length.")
        positions = np.zeros((2, len(input_ids_list), len(input_ids_list[0])), dtype=np.int64)
        image_index = 0
        for batch_index, (ids, mask) in enumerate(zip(input_ids_list, attention_mask)):
            active = np.flatnonzero(mask)
            active_ids = [ids[index] for index in active]
            if self.image_token_id in active_ids and image_index < len(image_grid_hw):
                grid_height, grid_width = image_grid_hw[image_index]
                image_index += 1
                expected_ids = [
                    self.tokenizer.document_token_id,
                    self.image_token_id,
                    *([self.image_token_id] * grid_width + [self.tokenizer.row_token_id]) * grid_height,
                ]
                if active_ids != expected_ids:
                    raise ValueError("NeoMME image inputs contain an invalid or truncated token layout.")
                sample_positions = self._image_positions(grid_height, grid_width)
            else:
                indices = np.arange(len(active), dtype=np.int64)
                sample_positions = np.stack((indices, indices), axis=-1)
            positions[:, batch_index, active] = sample_positions.T

        if image_index != len(image_grid_hw):
            raise ValueError(f"Got {image_index} image prompts for {len(image_grid_hw)} images.")

        if hasattr(input_ids, "new_tensor"):
            return input_ids.new_tensor(positions)
        return positions if isinstance(input_ids, np.ndarray) else positions.tolist()

    @staticmethod
    def _image_positions(grid_height: int, grid_width: int) -> np.ndarray:
        """Return positions for `<doc> <img>` followed by the patch grid and row markers."""
        positions = np.empty((2 + grid_height * (grid_width + 1), 2), dtype=np.int64)
        positions[0] = (0, 0)
        positions[1] = (1, 1)
        rows = np.broadcast_to(np.arange(grid_height)[:, None], (grid_height, grid_width + 1))
        columns = np.broadcast_to(np.arange(grid_width + 1)[None, :], (grid_height, grid_width + 1))
        positions[2:, 0] = 2 + rows.ravel()
        positions[2:, 1] = 2 + columns.ravel()
        return positions


__all__ = ["NeoMMEProcessor"]
