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

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, is_numpy_array, is_torch_available


if is_torch_available():
    import torch


@auto_docstring
class NeoMMEProcessor(ProcessorMixin):
    r"""
    Constructs a processor that prepares text and images for NeoMME models.

    Plain text is tokenized without retrieval markers. Use [`~NeoMMEProcessor.apply_chat_template`] to format retrieval
    inputs. Images receive a patch grid and two-axis positions.
    """

    valid_processor_kwargs = ProcessingKwargs

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
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        r"""
        Tokenize text or process images. Depending on the downstream task, inputs must first be formatted with
        [`~NeoMMEProcessor.apply_chat_template`].

        - Text inputs are tokenized as provided. For masked language modeling, prepend the document token
          (`tokenizer.document_token`) manually. For retrieval, use [`~NeoMMEProcessor.apply_chat_template`]
          instead.
        - Image inputs always receive the complete document-page token formatting (the `<doc>` prefix followed
          by the `<img>` patch grid with `<row>` markers). For retrieval, use
          `apply_chat_template(..., task="document")` instead.

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
        **kwargs,
    ):
        images, text, videos, audio = super().prepare_inputs_layout(
            images=images,
            text=text,
            **kwargs,
        )
        if images is not None and text is None:
            images = make_flat_list_of_images(images)
            text = [self.tokenizer.document_token + self.image_token] * len(images)
        return images, text, None, None

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
        """
        Build two-axis positions for batches containing document images. This method
        is stored in `processor` since it requires knowledge about special MM tokens
        """
        input_ids_list = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        if len({len(input_ids_row) for input_ids_row in input_ids_list}) != 1:
            raise ValueError("NeoMME image batches require padding to a common sequence length.")

        input_ids_is_tensor = isinstance(input_ids, torch.Tensor)
        input_ids_is_numpy = is_numpy_array(input_ids)
        input_ids = torch.as_tensor(input_ids)
        attention_mask = torch.as_tensor(attention_mask, device=input_ids.device)
        image_grid_hw = torch.as_tensor(image_grid_hw, device=input_ids.device)
        # position_ids: (axes, batch, seq_len)
        position_ids = torch.zeros((2, *input_ids.shape), dtype=torch.long, device=input_ids.device)
        image_index = 0

        for batch_index, (input_ids_row, attention_mask_row) in enumerate(zip(input_ids, attention_mask)):
            non_padded_indices = torch.nonzero(attention_mask_row, as_tuple=True)[0]
            non_padded_ids = input_ids_row[non_padded_indices]
            if self.image_token_id in non_padded_ids and image_index < len(image_grid_hw):
                grid_height, grid_width = image_grid_hw[image_index].tolist()
                image_index += 1
                expected_input_ids = torch.tensor(
                    [
                        self.tokenizer.document_token_id,
                        self.image_token_id,
                        *([self.image_token_id] * grid_width + [self.tokenizer.row_token_id]) * grid_height,
                    ],
                    device=input_ids.device,
                )
                if not torch.equal(non_padded_ids, expected_input_ids):
                    raise ValueError("NeoMME image inputs contain an invalid or truncated token layout.")
                sample_position_ids = self._image_positions(grid_height, grid_width, device=input_ids.device)
            else:
                text_position_ids = torch.arange(len(non_padded_indices), device=input_ids.device)
                sample_position_ids = torch.stack((text_position_ids, text_position_ids), dim=-1)
            # sample_position_ids: (seq_len, axes)
            position_ids[:, batch_index, non_padded_indices] = sample_position_ids.mT

        if image_index != len(image_grid_hw):
            raise ValueError(f"Got {image_index} image prompts for {len(image_grid_hw)} images.")

        if input_ids_is_tensor:
            return position_ids
        return position_ids.numpy() if input_ids_is_numpy else position_ids.tolist()

    @staticmethod
    def _image_positions(grid_height: int, grid_width: int, device: torch.device | None = None) -> torch.Tensor:
        """Return positions for `<doc> <img>` followed by the patch grid and row markers."""
        position_ids = torch.empty((2 + grid_height * (grid_width + 1), 2), dtype=torch.long, device=device)
        position_ids[:2] = torch.arange(2, device=device).unsqueeze(-1)
        rows = torch.arange(grid_height, device=device).unsqueeze(-1).expand(-1, grid_width + 1)
        columns = torch.arange(grid_width + 1, device=device).unsqueeze(0).expand(grid_height, -1)
        position_ids[2:, 0] = 2 + rows.ravel()
        position_ids[2:, 1] = 2 + columns.ravel()
        return position_ids


__all__ = ["NeoMMEProcessor"]
