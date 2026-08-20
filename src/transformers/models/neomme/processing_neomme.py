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

from typing import Any, Literal

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
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

    Plain text is tokenized without retrieval markers. When `task` is `"query"` or `"document"`, text and images are
    formatted with the checkpoint's retrieval template. Images receive a patch grid and two-axis positions.
    """

    valid_processor_kwargs = NeoMMEProcessorKwargs
    unused_input_names = ["image_grid_hw"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        query_expand: int = 0,
        **kwargs,
    ):
        r"""
        query_expand (`int`, *optional*, defaults to 0):
            Number of `<mask>` buffer tokens appended to every query.
        """
        if not isinstance(query_expand, int) or isinstance(query_expand, bool) or query_expand < 0:
            raise ValueError(f"query_expand must be a non-negative integer, got {query_expand!r}.")

        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.image_token = tokenizer.image_token
        self.query_expand = query_expand

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        task: Literal["query", "document"] | None = None,
        _chat_template_applied: bool = False,
        **kwargs: Unpack[NeoMMEProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Tokenize plain text, format retrieval text, or process images.

        _chat_template_applied (`bool`, *optional*, defaults to `False`):
            Internal flag set after the retrieval markers are rendered.
        task (`str`, *optional*):
            Set to `"query"` or `"document"` to apply the checkpoint's retrieval template. Leave unset for generic
            NeoMME text or image processing.

        Returns:
            A [`BatchFeature`] with `input_ids` and `attention_mask`. Image inputs also return `position_ids`,
            and `pixel_values`.
        """
        if not _chat_template_applied:
            return self._process_direct_inputs(images=images, text=text, task=task, **kwargs)

        images, text, _, _ = self.prepare_inputs_layout(images=images, text=text, **kwargs)
        self.validate_inputs(images=images, text=text, **kwargs)
        if "task" in kwargs.get("text_kwargs", {}):
            raise ValueError("Pass `task` as a top-level processor argument, not inside `text_kwargs`.")

        output_kwargs = self._merge_kwargs(
            NeoMMEProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        if isinstance(text, str):
            text = [text]

        # What the caller actually named, flat or nested, as opposed to what `_merge_kwargs` injected.
        requested = set(kwargs) | set(kwargs.get("text_kwargs", {}))
        text_kwargs = self._supported_text_kwargs(output_kwargs["text_kwargs"], requested)
        image_inputs, replacements = ({}, [])
        if images is not None:
            image_inputs, replacements = self._process_images(images, **output_kwargs["images_kwargs"])
        return self._tokenize_rendered_inputs(
            text,
            task=task,
            image_inputs=image_inputs,
            image_replacements=replacements,
            **text_kwargs,
        )

    def validate_inputs(self, images: ImageInput | None = None, text: TextInput | None = None, **kwargs):
        """Validate text before tokenization."""
        super().validate_inputs(images=images, text=text, **kwargs)
        if text is not None:
            text = [text] if isinstance(text, str) else text
            if not text:
                raise ValueError("text must contain at least one string.")
            if any(not isinstance(value, str) for value in text):
                raise ValueError("Pretokenized text is not supported.")

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

        processor_kwargs = {**(processor_kwargs or {}), "task": task, "_chat_template_applied": True}
        return super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            processor_kwargs=processor_kwargs,
            task=task,
            **kwargs,
        )

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names + ["position_ids"]

    def _process_direct_inputs(
        self,
        images: ImageInput | None,
        text: TextInput | list[TextInput] | None,
        task: Literal["query", "document"] | None,
        **kwargs,
    ) -> BatchFeature:
        """Process generic inputs directly or format retrieval inputs with the checkpoint's chat template."""
        if (text is None) == (images is None):
            raise ValueError("Pass exactly one of `text` or `images`.")

        if task is None:
            if text is not None:
                return super().__call__(text=text, **kwargs)
            return self._process_generic_images(images, **kwargs)

        if images is not None:
            image_list = [images] if is_valid_image(images) else list(images)
            conversations = [
                [{"role": "user", "content": [{"type": "image", "image": image}]}] for image in image_list
            ]
        else:
            text_list = [text] if isinstance(text, str) else text
            assert text_list is not None
            if not text_list:
                raise ValueError("text must contain at least one string.")
            if any(not isinstance(value, str) for value in text_list):
                raise ValueError("Pretokenized text is not supported.")
            conversations = [[{"role": "user", "content": value}] for value in text_list]

        return self.apply_chat_template(
            conversations,
            task=task,
            tokenize=True,
            return_dict=True,
            processor_kwargs=kwargs,
        )

    def _process_generic_images(self, images: ImageInput, **kwargs) -> BatchFeature:
        """Build NeoMME's architecture-level image layout without applying the retrieval template."""
        image_list = [images] if is_valid_image(images) else list(images)
        output_kwargs = self._merge_kwargs(
            NeoMMEProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        requested = set(kwargs) | set(kwargs.get("text_kwargs", {}))
        text_kwargs = self._supported_text_kwargs(output_kwargs["text_kwargs"], requested)
        image_inputs, replacements = self._process_images(image_list, **output_kwargs["images_kwargs"])
        rendered_images = [self.tokenizer.document_token + self.tokenizer.image_token] * len(replacements)
        return self._tokenize_rendered_inputs(
            rendered_images,
            task="document",
            image_inputs=image_inputs,
            image_replacements=replacements,
            **text_kwargs,
        )

    def _tokenize_rendered_inputs(
        self,
        text: list[str],
        task: Literal["query", "document"],
        image_inputs: dict[str, Any],
        image_replacements: list[str],
        max_length: int | None = None,
        padding: bool | str = "longest",
        return_tensors: str | None = "pt",
    ) -> BatchFeature:
        marker_ids = self._marker_ids()
        image_grid_hw = image_inputs.pop("image_grid_hw", None)
        text, _ = self.get_text_with_replacements(text, images_replacements=image_replacements)
        sequences = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if image_replacements:
            self._check_special_mm_tokens(text, {"input_ids": sequences}, modalities=["image"])

        finalized_sequences = []
        positions: list[np.ndarray] | None = [] if image_grid_hw is not None else None
        image_index = 0
        for ids in sequences:
            is_image_document = (
                image_grid_hw is not None
                and len(ids) > 1
                and ids[:2]
                == [
                    marker_ids["document"],
                    marker_ids["image"],
                ]
            )
            if not is_image_document:
                ids = self._finalize_text_sequence(ids, task, marker_ids, max_length)
                finalized_sequences.append(ids)
                if positions is not None:
                    positions.append(self._text_positions(len(ids)))
                continue
            if image_index >= len(image_grid_hw):
                raise ValueError("NeoMME rendered more image documents than the processor received.")

            grid_height, grid_width = image_grid_hw[image_index].tolist()
            expected_ids, position_ids = self._encode_image_grid(grid_height, grid_width, marker_ids)
            if ids != expected_ids:
                raise ValueError("NeoMME image template and placeholder replacement produced an invalid token layout.")
            if max_length is not None and len(ids) > max_length:
                raise ValueError(
                    f"NeoMME image document length {len(ids)} exceeds max_length={max_length}; image grids cannot "
                    "be truncated."
                )
            finalized_sequences.append(ids)
            assert positions is not None
            positions.append(position_ids)
            image_index += 1

        if image_grid_hw is not None and image_index != len(image_grid_hw):
            raise ValueError(f"Got {image_index} image prompts for {len(image_grid_hw)} images.")

        batch = self._pad_sequences(
            finalized_sequences,
            positions,
            padding=padding,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        batch.update(image_inputs)
        return batch

    def _finalize_text_sequence(
        self,
        ids: list[int],
        task: Literal["query", "document"],
        marker_ids: dict[str, int],
        max_length: int | None,
    ) -> list[int]:
        prefix_id = marker_ids["query"] if task == "query" else marker_ids["document"]
        if not ids or ids[0] != prefix_id or ids.count(prefix_id) != 1:
            raise ValueError(f"NeoMME chat template must render exactly one leading {task} token.")

        expansion_length = self.query_expand if task == "query" else 0
        if expansion_length:
            expansion = [marker_ids["mask"]] * expansion_length
            if ids[-expansion_length:] != expansion or ids.count(marker_ids["mask"]) != expansion_length:
                raise ValueError(f"NeoMME query template must render exactly {expansion_length} trailing mask tokens.")
            content = ids[1:-expansion_length]
        else:
            expansion = []
            content = ids[1:]

        content_limit = None if max_length is None else max_length - 1 - expansion_length
        if content_limit is not None and content_limit < 0:
            raise ValueError(
                f"query_expand={expansion_length} leaves no room for content inside max_length={max_length}"
            )
        return [prefix_id, *content[:content_limit], *expansion]

    @staticmethod
    def _text_positions(length: int) -> np.ndarray:
        positions = np.arange(length, dtype=np.int64)
        return np.stack((positions, positions), axis=-1)

    def _marker_ids(self) -> dict[str, int]:
        """Resolve marker token IDs and validate that the tokenizer defines them."""
        ids = {
            "query": getattr(self.tokenizer, "query_token_id", None),
            "document": getattr(self.tokenizer, "document_token_id", None),
            "image": getattr(self.tokenizer, "image_token_id", None),
            "row": getattr(self.tokenizer, "row_token_id", None),
            "mask": self.tokenizer.mask_token_id,
        }
        unknown_id = self.tokenizer.unk_token_id
        missing = [name for name, token_id in ids.items() if token_id is None or token_id == unknown_id]
        if missing:
            raise ValueError(f"The tokenizer is missing NeoMME marker tokens: {missing}")

        if len(set(ids.values())) != len(ids):
            raise ValueError("NeoMME query, document, image, row, and mask markers must use distinct token IDs.")

        pad_token_id = self.tokenizer.pad_token_id
        if ids["image"] == (pad_token_id if pad_token_id is not None else 0):
            raise ValueError("The NeoMME image marker must not use the padding token ID.")
        return ids

    def _encode_image_grid(
        self, grid_height: int, grid_width: int, marker_ids: dict[str, int]
    ) -> tuple[list[int], np.ndarray]:
        """`<doc> <img>` + `grid_height` rows of `grid_width` patch tokens each closed by a `<row>` break."""
        grid = np.full(
            (grid_height, grid_width + 1), marker_ids["image"], dtype=np.int64
        )  # (grid_height, grid_width + 1)
        grid[:, grid_width] = marker_ids["row"]
        ids = [marker_ids["document"], marker_ids["image"], *grid.ravel().tolist()]

        positions = np.empty((len(ids), 2), dtype=np.int64)  # (sequence_length, 2)
        positions[0] = (0, 0)
        positions[1] = (1, 1)
        rows = np.broadcast_to(np.arange(grid_height)[:, None], grid.shape)
        columns = np.broadcast_to(np.arange(grid_width + 1)[None, :], grid.shape)
        positions[2:, 0] = 2 + rows.ravel()
        positions[2:, 1] = 2 + columns.ravel()
        return ids, positions

    def _pad_sequences(
        self,
        sequences: list[list[int]],
        positions: list[np.ndarray] | None = None,
        padding: bool | str = "longest",
        max_length: int | None = None,
        return_tensors: str | None = "pt",
    ) -> BatchFeature:
        """Right-pad `sequences` (and their two-axis positions) into `(batch, length)` tensors."""
        if padding in (False, "do_not_pad") and return_tensors is None and positions is None:
            return BatchFeature(
                data={
                    "input_ids": [list(ids) for ids in sequences],
                    "attention_mask": [[1] * len(ids) for ids in sequences],
                }
            )

        length = self._padded_length([len(ids) for ids in sequences], padding, max_length)
        pad_token_id = self.tokenizer.pad_token_id or 0

        data: dict[str, Any] = {
            "input_ids": [ids + [pad_token_id] * (length - len(ids)) for ids in sequences],
            "attention_mask": [[1] * len(ids) + [0] * (length - len(ids)) for ids in sequences],
        }
        if positions is not None:
            # Index 0 is the M-RoPE row axis, index 1 the column axis.
            grid = np.zeros((2, len(sequences), length), dtype=np.int64)  # (2, batch_size, sequence_length)
            for index, image_positions in enumerate(positions):
                grid[:, index, : image_positions.shape[0]] = image_positions.T
            data["position_ids"] = grid.tolist()
        return BatchFeature(data=data, tensor_type=return_tensors)

    def _padded_length(self, lengths: list[int], padding: bool | str, max_length: int | None) -> int:
        """The width every row is padded to, following the tokenizer's `padding` vocabulary."""
        longest = max(lengths)
        if padding == "max_length":
            if max_length is None:
                raise ValueError("padding='max_length' needs a `max_length`.")
            if max_length < longest:
                raise ValueError(
                    f"max_length={max_length} is shorter than the longest encoded row ({longest}); padding to it "
                    "would drop tokens."
                )
            return max_length

        if padding in (False, "do_not_pad") and longest != min(lengths):
            raise ValueError(
                "padding=False cannot return a single tensor for rows of different lengths. Pass "
                "padding='longest', or encode one sequence at a time."
            )
        return longest

    def _supported_text_kwargs(self, text_kwargs: dict[str, Any], requested: set[str]) -> dict[str, Any]:
        """Filter text kwargs to the subset supported by this processor."""
        supported = {
            name: text_kwargs[name] for name in ("max_length", "padding", "return_tensors") if name in text_kwargs
        }
        if text_kwargs.get("truncation") not in (None, False, "do_not_truncate") and "max_length" not in supported:
            supported["max_length"] = self.tokenizer.model_max_length

        if text_kwargs.get("truncation") is False and text_kwargs.get("max_length") is not None:
            raise ValueError(
                "truncation=False with a max_length is not supported: the marker and query-expansion layout "
                "is fixed, so content past max_length is always dropped."
            )

        unsupported = sorted((set(text_kwargs) & requested) - set(supported) - {"truncation"})
        if unsupported:
            raise ValueError(f"NeoMMEProcessor does not implement these text kwargs: {unsupported}.")
        return supported


__all__ = ["NeoMMEProcessor"]
