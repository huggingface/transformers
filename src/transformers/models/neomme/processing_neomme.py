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
from ...tokenization_utils_base import TextInput, TruncationStrategy
from ...utils import auto_docstring, is_numpy_array, is_torch_available


if is_torch_available():
    import torch


# Per-token outputs that `_call_with_literal_texts` cannot keep aligned after splicing in the user text.
_UNSUPPORTED_RETRIEVAL_OUTPUTS = (
    "return_offsets_mapping",
    "return_token_type_ids",
    "return_special_tokens_mask",
    "return_overflowing_tokens",
    "return_length",
    "return_mm_token_type_ids",
)


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

        - Text inputs are tokenized as provided, so special-token strings in the text become special tokens. For
          masked language modeling, prepend the document token (`tokenizer.document_token`) manually. For
          retrieval, use [`~NeoMMEProcessor.apply_chat_template`] instead.
        - Image inputs always receive the complete document-page token formatting (the `<doc>` prefix followed
          by the `<img>` patch grid with `<row>` markers). For retrieval, use
          `apply_chat_template(..., task="document")` instead.

        Returns:
            A [`BatchFeature`] with `input_ids` and `attention_mask`. Image inputs also return `position_ids`,
            and `pixel_values`.
        """
        retrieval_task = kwargs.pop("_retrieval_task", None)
        retrieval_texts = kwargs.pop("_retrieval_texts", None)
        if retrieval_texts is None:
            batch = super().__call__(images=images, text=text, **kwargs)
        else:
            batch = self._call_with_literal_texts(images, text, retrieval_task, retrieval_texts, **kwargs)

        image_grid_hw = batch.pop("image_grid_hw", None)
        if image_grid_hw is not None:
            if "attention_mask" not in batch:
                raise ValueError("NeoMME image inputs require `return_attention_mask=True`.")
            batch["position_ids"] = self._build_position_ids(
                batch["input_ids"],
                batch["attention_mask"],
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
        """Apply the NeoMME retrieval template and optionally tokenize the result.

        With `tokenize=True`, user text is tokenized as plain text: a string such as `<img>` or `<mask>` in the text
        never becomes a special token. This does not hold for `tokenize=False`: the template rejects `<img>` in
        text, and tokenizing the rendered string yourself turns strings such as `<mask>` into special tokens.

        Image content must include an image, URL, path, or base64 value. Pass processing options such as
        `max_length` or `max_side` through `processor_kwargs`.

        A custom template used with `tokenize=True` must insert the text as is (no filters) and join text items with
        no separator, like the default template.
        """
        if kwargs.get("return_assistant_tokens_mask"):
            raise ValueError("NeoMME retrieval templates do not support `return_assistant_tokens_mask`.")

        processor_kwargs = {**(processor_kwargs or {}), "_retrieval_task": task}
        if kwargs.get("tokenize"):
            conversation, processor_kwargs["_retrieval_texts"] = self._replace_texts_with_placeholder(conversation)
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

    def _replace_texts_with_placeholder(self, conversation: list) -> tuple[list, list[str | None]]:
        """
        Put one `unk` placeholder where each conversation's text goes, so the template never sees user text. Returns
        the new conversation and each conversation's joined text (`None` when it has no text).

        Example:

        ```python
        >>> processor._replace_texts_with_placeholder(
        ...     [
        ...         {"role": "system", "content": "find "},
        ...         {"role": "user", "content": [{"type": "text", "text": "<mask>"}]},
        ...     ]
        ... )
        (
            [
                {"role": "system", "content": "<unk>"},  # first text: the placeholder
                {"role": "user", "content": [{"type": "text", "text": ""}]},  # later texts: emptied
            ],
            ["find <mask>"],  # all texts joined, tokenized later with `split_special_tokens=True`
        )
        ```
        """
        if self.tokenizer.unk_token is None:
            raise ValueError("NeoMME retrieval templates need `tokenizer.unk_token` to mark where text goes.")
        is_batched = isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        conversations = conversation if is_batched else [conversation]
        placeholder = self.tokenizer.unk_token
        placeholder_conversations, retrieval_texts = [], []
        for messages in conversations:
            text_parts, placeholder_messages = [], []
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    placeholder_messages.append({**message, "content": "" if text_parts else placeholder})
                    text_parts.append(content)
                elif isinstance(content, list):
                    placeholder_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            placeholder_content.append({**item, "text": "" if text_parts else placeholder})
                            text_parts.append(item["text"])
                        else:
                            placeholder_content.append(item)
                    placeholder_messages.append({**message, "content": placeholder_content})
                else:
                    placeholder_messages.append(message)
            placeholder_conversations.append(placeholder_messages)
            retrieval_texts.append("".join(text_parts) if text_parts else None)
        return (placeholder_conversations if is_batched else placeholder_conversations[0]), retrieval_texts

    def _call_with_literal_texts(
        self,
        images: ImageInput | None,
        text: TextInput | list[TextInput],
        retrieval_task: str | None,
        retrieval_texts: list[str | None],
        **kwargs,
    ) -> BatchFeature:
        """
        Process the template output, then put each text's ids where its placeholder landed. The text is tokenized
        with `split_special_tokens=True` and truncated on its own, so markers and query expansion stay.
        """
        merged_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        text_kwargs = merged_kwargs["text_kwargs"]
        unsupported_outputs = [name for name in _UNSUPPORTED_RETRIEVAL_OUTPUTS if text_kwargs.get(name)]
        if unsupported_outputs:
            raise ValueError(f"NeoMME retrieval templates do not support {', '.join(unsupported_outputs)}.")
        padding, truncation, max_length, _ = self.tokenizer._get_padding_truncation_strategies(
            padding=text_kwargs.get("padding", False),
            truncation=text_kwargs.get("truncation"),
            max_length=text_kwargs.get("max_length"),
            pad_to_multiple_of=text_kwargs.get("pad_to_multiple_of"),
        )
        truncate = truncation != TruncationStrategy.DO_NOT_TRUNCATE
        batch = super().__call__(
            images=images,
            text=text,
            text_kwargs={
                **text_kwargs,
                "padding": False,
                "truncation": False,
                "return_tensors": None,
            },
            images_kwargs=merged_kwargs["images_kwargs"],
        )
        # Image rows have no text; an empty string keeps one tokenizer call aligned with the batch.
        text_ids_batch = self.tokenizer(
            [retrieval_text or "" for retrieval_text in retrieval_texts],
            add_special_tokens=False,
            split_special_tokens=True,
        )["input_ids"]

        task_token_id = getattr(self.tokenizer, f"{retrieval_task}_token_id", None)
        placeholder_id = self.tokenizer.unk_token_id
        input_ids = []
        for template_ids, retrieval_text, text_ids in zip(batch["input_ids"], retrieval_texts, text_ids_batch):
            # Check the marker first: a template that drops the text also drops the placeholder.
            if task_token_id is not None and template_ids[:1] != [task_token_id]:
                raise ValueError("The NeoMME retrieval template must preserve its leading task marker.")
            has_text = retrieval_text is not None
            if has_text and template_ids.count(placeholder_id) != 1:
                raise ValueError("The NeoMME retrieval template must render the text content exactly once.")
            if truncate:
                # Only the text is cut: cutting the template would drop query expansion tokens or break an image layout.
                max_text_length = max_length - (len(template_ids) - has_text)
                if max_text_length < 0:
                    raise ValueError("`max_length` is too short for the NeoMME retrieval template.")
                if self.tokenizer.truncation_side == "left":
                    text_ids = text_ids[max(len(text_ids) - max_text_length, 0) :]
                else:
                    text_ids = text_ids[:max_text_length]
            if has_text:
                placeholder_index = template_ids.index(placeholder_id)
                template_ids = template_ids[:placeholder_index] + text_ids + template_ids[placeholder_index + 1 :]
            input_ids.append(template_ids)

        padded_text = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=text_kwargs.get("pad_to_multiple_of"),
            padding_side=text_kwargs.get("padding_side"),
            return_attention_mask=text_kwargs.get("return_attention_mask"),
        )
        return BatchFeature(
            {**batch, **padded_text},
            tensor_type=text_kwargs.get("return_tensors"),
            skip_tensor_conversion=self.skip_tensor_conversion,
        )

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
