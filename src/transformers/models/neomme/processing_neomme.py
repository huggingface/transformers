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
from ...utils import auto_docstring, is_torch_available, logging
from ...utils.chat_template_utils import _get_template_variables
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


def _pad_grids(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Variable-length `(length, dim)` token grids -> padded `(batch, max_length, dim)` plus a bool mask."""
    device = embeddings[0].device
    lengths = torch.tensor([grid.shape[0] for grid in embeddings], device=device)  # (batch_size,)
    mask = torch.arange(int(lengths.max()), device=device)[None, :] < lengths[:, None]  # (batch_size, max_length)
    padded = torch.zeros(
        *mask.shape, embeddings[0].shape[-1], dtype=embeddings[0].dtype, device=device
    )  # (batch_size, max_length, dim)
    for index, grid in enumerate(embeddings):
        padded[index, : grid.shape[0]] = grid
    return padded, mask


def _as_padded_grids(embeddings: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept either a padded 3-D tensor or a list of token grids."""
    if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
        return embeddings, embeddings.abs().sum(-1) > 0  # mask: (batch_size, max_length)
    return _pad_grids(list(embeddings))


def maxsim_scores(
    query_grids: torch.Tensor,
    passage_grids: torch.Tensor,
    query_mask: torch.Tensor,
    passage_mask: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute MaxSim from padded token grids, optionally normalized by query length."""
    query_grids = (
        torch.nn.functional.normalize(query_grids.float(), dim=-1) * query_mask[..., None]
    )  # (num_queries, query_length, dim)
    passage_grids = torch.nn.functional.normalize(passage_grids.float(), dim=-1)  # (num_passages, passage_length, dim)
    similarity = torch.einsum(
        "qid,pjd->qpij", query_grids, passage_grids
    )  # (num_queries, num_passages, query_length, passage_length)
    similarity = similarity.masked_fill(~passage_mask[None, :, None, :], torch.finfo(similarity.dtype).min)
    scores = similarity.max(dim=-1).values.sum(dim=-1)  # (num_queries, num_passages)

    if normalize:
        scores = scores / query_mask.sum(-1, keepdim=True).clamp_min(1).to(scores.dtype)
    return scores.masked_fill(~passage_mask.any(dim=-1)[None, :], -1.0)


class NeoMMEProcessorKwargs(ProcessingKwargs, total=False):
    # `_merge_kwargs` reads this attribute directly; TypedDict subclasses do not inherit it.
    _defaults = {}


@requires(backends=("torch",))
@auto_docstring
class NeoMMEProcessor(ProcessorMixin):
    r"""
    Constructs a processor that prepares text and document images for NeoMME retrieval models.

    Queries, text documents, and image documents are encoded in separate forward passes. Pass exactly one of `text`
    or `images` to each call. Queries receive a `<query>` prefix and `<mask>` expansion tokens. Text and image
    documents receive a `<doc>` prefix, and image documents also receive a patch grid and two-axis positions.
    """

    valid_processor_kwargs = NeoMMEProcessorKwargs
    unused_input_names = ["image_grid_hw"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        query_expand: int = 10,
        image_placeholder: str = "<|image_placeholder|>",
        **kwargs,
    ):
        r"""
        query_expand (`int`, *optional*, defaults to 10):
            Number of `<mask>` buffer tokens appended to every query.
        image_placeholder (`str`, *optional*, defaults to `"<|image_placeholder|>"`):
            Temporary string emitted by the chat template for one image. It is replaced before tokenization.
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.query_expand = query_expand
        self.image_placeholder = image_placeholder

    @property
    def image_token(self) -> str | None:
        return getattr(self.tokenizer, "image_token", None)

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names + ["position_ids"]

    def validate_inputs(self, images: ImageInput | None = None, text: TextInput | None = None, **kwargs):
        """Validate that each call contains exactly one NeoMME input modality."""
        super().validate_inputs(images=images, text=text, **kwargs)
        if text is not None and images is not None:
            raise ValueError(
                "Pass exactly one of `text` or `images`: they are opposite retrieval sides, encoded in "
                "separate forward passes."
            )

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        task: Literal["query", "document"] = "query",
        processor_kwargs: dict | None = None,
        **kwargs,
    ):
        """Apply the configured template and prepare the selected retrieval side when tokenizing."""
        template = self._resolve_chat_template(chat_template)
        template_variables = _get_template_variables(template)
        if "task" not in template_variables:
            raise ValueError("NeoMME chat templates must use the `task` variable to distinguish retrieval sides.")

        modality = self._validate_chat_conversation(conversation, task)
        if modality == "image" and "image_placeholder" not in template_variables:
            raise ValueError("NeoMME image chat templates must use the `image_placeholder` variable.")
        if modality == "image" and kwargs.get("return_assistant_tokens_mask"):
            raise ValueError("Image document templates do not support `return_assistant_tokens_mask`.")

        processor_kwargs = {**(processor_kwargs or {}), "task": task, "_chat_template_applied": True}
        if "image_placeholder" in template_variables:
            kwargs["image_placeholder"] = self.image_placeholder
        return super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            processor_kwargs=processor_kwargs,
            task=task,
            **kwargs,
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        task: Literal["query", "document"] = "query",
        _chat_template_applied: bool = False,
        **kwargs: Unpack[NeoMMEProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Pass exactly one of `text` or `images`. Pretokenized text is not supported.

        task (`str`, *optional*, defaults to `"query"`):
            Whether `text` is a retrieval query or a text document. Queries receive `<query>` and `<mask>` tokens.
            Text documents receive `<doc>`. Ignored for `images`, which are always document inputs.
        _chat_template_applied (`bool`, *optional*, defaults to `False`):
            Internal flag set by [`~NeoMMEProcessor.apply_chat_template`] to avoid adding special tokens a second time.

        Returns:
            A [`BatchFeature`] with `input_ids` and `attention_mask`. Image inputs also return `position_ids`,
            and `pixel_values`.
        """
        rendered_text = text if _chat_template_applied else None
        if _chat_template_applied and images is not None:
            text = None
        images, text, _, _ = self.prepare_inputs_layout(images=images, text=text, **kwargs)
        self.validate_inputs(images=images, text=text, **kwargs)
        if "task" in kwargs.get("text_kwargs", {}):
            raise ValueError("Pass `task` as a top-level processor argument, not inside `text_kwargs`.")

        output_kwargs = self._merge_kwargs(
            NeoMMEProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        if images is not None:
            return self.process_images(images, rendered_text=rendered_text, **output_kwargs["images_kwargs"])

        if isinstance(text, str):
            text = [text]
        if text and not isinstance(text[0], str):
            raise ValueError("Pretokenized text is not supported")

        # What the caller actually named, flat or nested, as opposed to what `_merge_kwargs` injected.
        requested = set(kwargs) | set(kwargs.get("text_kwargs", {}))
        text_kwargs = self._supported_text_kwargs(output_kwargs["text_kwargs"], requested)
        if _chat_template_applied:
            return self._tokenize_rendered_text(text, task=task, **text_kwargs)
        if task == "query":
            return self.process_queries(text, **text_kwargs)
        if task == "document":
            return self.process_text_documents(text, **text_kwargs)
        raise ValueError(f"task={task!r} is not supported: expected 'query' or 'document'.")

    def _resolve_chat_template(self, chat_template: str | None) -> str:
        """Resolve a template name or the configured default before validating its variables."""
        if chat_template is None:
            if isinstance(self.chat_template, dict):
                chat_template = self.chat_template.get("default")
            else:
                chat_template = self.chat_template
        elif isinstance(self.chat_template, dict) and chat_template in self.chat_template:
            chat_template = self.chat_template[chat_template]
        if not isinstance(chat_template, str):
            raise ValueError("NeoMMEProcessor requires one default chat template or an explicit template.")
        return chat_template

    def _validate_chat_conversation(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        task: Literal["query", "document"],
    ) -> Literal["text", "image"]:
        """Validate that each chat item represents one supported NeoMME retrieval side."""
        is_batched = (
            isinstance(conversation, (list, tuple)) and conversation and isinstance(conversation[0], (list, tuple))
        )
        conversations = conversation if is_batched else [conversation]
        modalities = set()

        for messages in conversations:
            has_text = False
            image_count = 0
            for message in messages:
                content = message.get("content") or []
                if isinstance(content, str):
                    self._validate_user_text(content)
                    has_text |= bool(content)
                    continue
                for item in content:
                    content_type = item.get("type")
                    if content_type == "text":
                        item_text = item.get("text") or ""
                        self._validate_user_text(item_text)
                        has_text |= bool(item_text)
                    elif content_type in {"image", "image_url"}:
                        image_count += 1
                    else:
                        raise ValueError(f"NeoMME chat templates do not support content type {content_type!r}.")

            if image_count and has_text:
                raise ValueError("NeoMME cannot encode text and images in the same chat item.")
            if image_count > 1:
                raise ValueError("NeoMME accepts one image document per chat item.")
            if image_count and task != "document":
                raise ValueError("Images in chat messages must use task='document'.")
            modalities.add("image" if image_count else "text")

        if len(modalities) != 1:
            raise ValueError("A NeoMME chat template batch cannot mix text and image documents.")
        return modalities.pop()

    def _validate_user_text(self, text: str) -> None:
        if self.image_placeholder in text:
            raise ValueError(
                f"{self.image_placeholder!r} is reserved for internal image processing and cannot appear "
                "in user text."
            )

    def _render_template(self, conversations: list[list[dict]], task: Literal["query", "document"]) -> list[str]:
        template = self._resolve_chat_template(None)
        if "task" not in _get_template_variables(template):
            raise ValueError("NeoMME chat templates must use the `task` variable to distinguish retrieval sides.")
        return self.tokenizer.apply_chat_template(
            conversations,
            chat_template=template,
            task=task,
            image_placeholder=self.image_placeholder,
            tokenize=False,
        )

    def _render_text(self, text: list[str], task: Literal["query", "document"]) -> list[str]:
        for value in text:
            self._validate_user_text(value)
        conversations = [[{"role": "user", "content": value}] for value in text]
        return self._render_template(conversations, task)

    def _tokenize_rendered_text(
        self,
        text: list[str],
        task: Literal["query", "document"],
        max_length: int | None = None,
        padding: bool | str = "longest",
        return_tensors: str | None = "pt",
    ) -> BatchFeature:
        marker_ids = self._marker_ids()
        encodings = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        sequences = []

        for ids in encodings:
            prefix_id = marker_ids["query"] if task == "query" else marker_ids["document"]
            if not ids or ids[0] != prefix_id or ids.count(prefix_id) != 1:
                raise ValueError(f"NeoMME chat template must render exactly one leading {task} token.")

            expansion_length = self.query_expand if task == "query" else 0
            if expansion_length:
                expansion = [marker_ids["mask"]] * expansion_length
                if ids[-expansion_length:] != expansion or ids.count(marker_ids["mask"]) != expansion_length:
                    raise ValueError(
                        f"NeoMME query template must render exactly {expansion_length} trailing mask tokens."
                    )
                content = ids[1:-expansion_length]
            else:
                expansion = []
                content = ids[1:]

            content_limit = None if max_length is None else max_length - 1 - expansion_length
            if content_limit is not None and content_limit < 0:
                raise ValueError(
                    f"query_expand={expansion_length} leaves no room for content inside max_length={max_length}"
                )
            sequences.append([prefix_id, *content[:content_limit], *expansion])

        return self._pad_sequences(sequences, padding=padding, max_length=max_length, return_tensors=return_tensors)

    def process_queries(
        self,
        text: list[str] | str,
        max_length: int | None = None,
        padding: bool | str = "longest",
        return_tensors: str | None = "pt",
    ) -> BatchFeature:
        """
        Prepare text queries with a `<query>` prefix and `<mask>` expansion tokens.

        `max_length` includes the query marker and expansion tokens. The returned [`BatchFeature`] contains
        `input_ids` and `attention_mask`.
        """
        if isinstance(text, str):
            text = [text]
        rendered = self._render_text(list(text), task="query")
        return self._tokenize_rendered_text(rendered, "query", max_length, padding, return_tensors)

    def process_text_documents(
        self,
        text: list[str] | str,
        max_length: int | None = None,
        padding: bool | str = "longest",
        return_tensors: str | None = "pt",
    ) -> BatchFeature:
        """
        Prepare text documents with a `<doc>` prefix.

        `max_length` includes the document marker. The returned [`BatchFeature`] contains `input_ids` and
        `attention_mask`.
        """
        if isinstance(text, str):
            text = [text]
        rendered = self._render_text([passage or " " for passage in text], task="document")
        return self._tokenize_rendered_text(rendered, "document", max_length, padding, return_tensors)

    def process_images(self, images: ImageInput, rendered_text: list[str] | str | None = None, **kwargs) -> BatchFeature:
        """
        Prepare document images as image patch grids with two-axis positions.

        Each image starts with `<doc> <img>`, followed by row-major patch placeholders and one `<row>` marker per
        patch row. Image arguments such as `max_side` are forwarded to the image processor. The returned
        [`BatchFeature`] contains `input_ids`, `attention_mask`, `position_ids`, and `pixel_values`.
        """
        if is_valid_image(images):
            images = [images]
        return_tensors = kwargs.setdefault("return_tensors", "pt")
        image_inputs = self.image_processor(images=images, **kwargs)
        image_grid_hw = image_inputs.pop("image_grid_hw")
        marker_ids = self._marker_ids()

        if rendered_text is None:
            conversations = [
                [{"role": "user", "content": [{"type": "image"}]}] for _ in range(len(image_grid_hw))
            ]
            rendered_text = self._render_template(conversations, task="document")
        elif isinstance(rendered_text, str):
            rendered_text = [rendered_text]

        replacements = [self.replace_image_token({"image_grid_hw": image_grid_hw}, index) for index in range(len(image_grid_hw))]
        rendered_text, _ = self.get_text_with_replacements(list(rendered_text), images_replacements=replacements)
        sequences = self.tokenizer(rendered_text, add_special_tokens=False)["input_ids"]
        if len(sequences) != len(image_grid_hw):
            raise ValueError(f"Got {len(sequences)} image prompts for {len(image_grid_hw)} images.")

        positions: list[np.ndarray] = []
        for ids, (grid_height, grid_width) in zip(sequences, image_grid_hw.tolist()):
            expected_ids, position_ids = self._encode_image_grid(grid_height, grid_width, marker_ids)
            if ids != expected_ids:
                raise ValueError("NeoMME image template and placeholder replacement produced an invalid token layout.")
            positions.append(position_ids)

        batch = self._pad_sequences(sequences, positions, return_tensors=return_tensors)
        batch["pixel_values"] = image_inputs["pixel_values"]  # (num_patches, 3 * patch_size ** 2)
        return batch

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        grid_height, grid_width = image_inputs["image_grid_hw"][image_idx]
        row = self.tokenizer.image_token * int(grid_width) + self.tokenizer.row_token
        return self.tokenizer.image_token + row * int(grid_height)

    def get_text_with_replacements(
        self,
        text: list[str],
        images_replacements: list[str] | None = None,
        videos_replacements: list[str] | None = None,
        audio_replacements: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Replace temporary image placeholders with grid-dependent trained image tokens."""
        images_replacements = images_replacements or []
        if videos_replacements or audio_replacements:
            raise ValueError("NeoMME supports image replacements only.")
        if sum(sample.count(self.image_placeholder) for sample in text) != len(images_replacements):
            raise ValueError("The number of image placeholders must match the number of images.")

        replacements = iter(images_replacements)
        for index, sample in enumerate(text):
            while self.image_placeholder in sample:
                sample = sample.replace(self.image_placeholder, next(replacements), 1)
            text[index] = sample
        if any(self.image_placeholder in sample for sample in text):
            raise ValueError("An image placeholder remained after replacement.")
        return text, []

    def score_retrieval(
        self,
        query_embeddings: torch.Tensor | list[torch.Tensor],
        passage_embeddings: torch.Tensor | list[torch.Tensor],
        batch_size: int = 128,
        output_dtype: torch.dtype | None = None,
        output_device: str | torch.device = "cpu",
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Score query and document embeddings with MaxSim or cosine similarity.

        The method infers the representation from tensor rank. Token embeddings use MaxSim, and dense embeddings use
        cosine similarity. Both arguments must use the same representation.

        Args:
            query_embeddings (`torch.Tensor` or `list[torch.Tensor]`):
                Multi-vector grids of shape `(num_queries, query_length, dim)` / list of
                `(query_length_i, dim)`, or dense vectors of shape `(num_queries, dim)`.
            passage_embeddings (`torch.Tensor` or `list[torch.Tensor]`):
                Same conventions as `query_embeddings`, for passages.
            batch_size (`int`, *optional*, defaults to 128):
                Chunk size over queries and passages when computing MaxSim (ignored for dense).
            output_dtype (`torch.dtype`, *optional*):
                Dtype of the returned score tensor. Defaults to the dtype of the computed scores.
            output_device (`str` or `torch.device`, *optional*, defaults to `"cpu"`):
                Device of the returned score tensor.
            normalize (`bool`, *optional*, defaults to `True`):
                Applies only to MaxSim scores. If `True`, divide each score by the number of non-padding query
                tokens. If `False`, return the raw sum of maximum similarities. This argument does not affect dense
                cosine scores or embedding normalization.

        Returns:
            `torch.Tensor` of shape `(num_queries, num_passages)`.
        """
        if len(query_embeddings) == 0 or len(passage_embeddings) == 0:
            raise ValueError("Both `query_embeddings` and `passage_embeddings` must be non-empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        query_kind, query_dim = self._embedding_kind(query_embeddings, "query_embeddings")
        passage_kind, passage_dim = self._embedding_kind(passage_embeddings, "passage_embeddings")
        if query_kind != passage_kind:
            raise ValueError(
                "`query_embeddings` and `passage_embeddings` must both be dense or both be multi-vector, "
                f"but got {query_kind} queries and {passage_kind} passages."
            )
        if query_dim != passage_dim:
            raise ValueError(
                "`query_embeddings` and `passage_embeddings` must have the same embedding dimension, "
                f"but got {query_dim} and {passage_dim}."
            )

        if query_kind == "multi-vector":
            scores = self._maxsim_in_blocks(query_embeddings, passage_embeddings, batch_size, normalize, output_device)
        else:
            queries = self._as_dense(query_embeddings)  # (num_queries, dim)
            passages = self._as_dense(passage_embeddings)  # (num_passages, dim)
            scores = (
                torch.nn.functional.normalize(queries.float(), dim=-1)
                @ torch.nn.functional.normalize(passages.float(), dim=-1).t()
            )  # (num_queries, num_passages)

        return scores.to(output_dtype or scores.dtype).to(output_device)

    def _marker_ids(self) -> dict[str, int]:
        """Resolve marker token ids and validate that the tokenizer defines them."""
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
        if padding in ("max_length",):
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

    def _maxsim_in_blocks(
        self,
        query_embeddings: torch.Tensor | list[torch.Tensor],
        passage_embeddings: torch.Tensor | list[torch.Tensor],
        batch_size: int,
        normalize: bool,
        output_device: str | torch.device,
    ) -> torch.Tensor:
        """Compute MaxSim scores in query-passage blocks."""
        rows: list[torch.Tensor] = []
        for query_start in range(0, len(query_embeddings), batch_size):
            query_block = query_embeddings[query_start : query_start + batch_size]
            query_grids, query_mask = _as_padded_grids(query_block)
            columns: list[torch.Tensor] = []
            for passage_start in range(0, len(passage_embeddings), batch_size):
                passage_block = passage_embeddings[passage_start : passage_start + batch_size]
                passage_grids, passage_mask = _as_padded_grids(passage_block)
                scores = maxsim_scores(
                    query_grids,
                    passage_grids,
                    query_mask,
                    passage_mask,
                    normalize=normalize,
                )
                columns.append(scores.to(output_device))
            rows.append(torch.cat(columns, dim=1))
        return torch.cat(rows, dim=0)  # (num_queries, num_passages)

    def _embedding_kind(self, embeddings: torch.Tensor | list[torch.Tensor], name: str) -> tuple[str, int]:
        """Validate one embedding collection and return its representation kind and dimension."""
        if isinstance(embeddings, torch.Tensor):
            if embeddings.dim() == 2:
                return "dense", embeddings.shape[-1]
            if embeddings.dim() == 3:
                return "multi-vector", embeddings.shape[-1]
            raise ValueError(
                f"`{name}` must be a 2-D dense tensor or a 3-D multi-vector tensor, got {embeddings.dim()}-D."
            )

        if any(not isinstance(embedding, torch.Tensor) for embedding in embeddings):
            raise ValueError(f"`{name}` must contain tensors.")
        ranks = {embedding.dim() for embedding in embeddings}
        if len(ranks) != 1:
            raise ValueError(f"`{name}` must contain tensors of one consistent rank, got {sorted(ranks)}.")
        rank = ranks.pop()
        if rank == 1:
            kind = "dense"
        elif rank == 2:
            kind = "multi-vector"
        else:
            raise ValueError(
                f"`{name}` must contain 1-D dense vectors or 2-D multi-vector grids, got {rank}-D entries."
            )

        dimensions = {embedding.shape[-1] for embedding in embeddings}
        if len(dimensions) != 1:
            raise ValueError(f"`{name}` must use one consistent embedding dimension, got {sorted(dimensions)}.")
        return kind, dimensions.pop()

    def _as_dense(self, embeddings: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return (
            embeddings if isinstance(embeddings, torch.Tensor) else torch.stack(list(embeddings))
        )  # (batch_size, dim)


__all__ = ["NeoMMEProcessor"]
