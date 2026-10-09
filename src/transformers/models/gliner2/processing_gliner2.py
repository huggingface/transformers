# Copyright 2026 The HuggingFace Inc. team.
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

from __future__ import annotations

import bisect
import copy
import logging
import re
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring


logger = logging.getLogger(__name__)

TextInput = str | Sequence[str]
RawSpan = tuple[str, float, int, int]

SEP_STRUCT = "[SEP_STRUCT]"
SEP_TEXT = "[SEP_TEXT]"
P_TOKEN = "[P]"
C_TOKEN = "[C]"
E_TOKEN = "[E]"
R_TOKEN = "[R]"
L_TOKEN = "[L]"
EXAMPLE_TOKEN = "[EXAMPLE]"
OUTPUT_TOKEN = "[OUTPUT]"
DESC_TOKEN = "[DESCRIPTION]"
SPECIAL_TOKENS = (
    SEP_STRUCT,
    SEP_TEXT,
    P_TOKEN,
    C_TOKEN,
    E_TOKEN,
    R_TOKEN,
    L_TOKEN,
    EXAMPLE_TOKEN,
    OUTPUT_TOKEN,
    DESC_TOKEN,
)
TASK_TYPE_TO_ID = {
    "entities": 1,
    "json_structures": 2,
    "relations": 3,
    "classifications": 4,
}
RECORD_MODE_TO_ID = {"natural": 1, "latent": 2, "anchorless": 3}
CARDINALITY_TO_ID = {
    "optional_one": 1,
    "required_one": 2,
    "zero_or_more": 3,
    "one_or_more": 4,
}
VALID_RECORD_MODES = ("natural", "latent", "anchorless")
RECORD_TASK_TYPES = ("json_structures",)
_SPAN_RESERVED = frozenset({"text", "confidence", "start", "end"})
_OVERLAP_ALIASES = {
    "allow": "allow",
    "all": "allow",
    "none": "allow",
    "nested": "nested",
    "allow_nested": "nested",
    "flat": "disallow",
    "disallow": "disallow",
    "no_overlap": "disallow",
    "non_overlapping": "disallow",
    "longest": "longest",
    "keep_longest": "longest",
}


@dataclass
class SamplingConfig:
    """Training draws. Call order matches the gliner2 schema transformer."""

    remove_json_structure_prob: float = 0.2
    shuffle_json_fields: bool = True
    remove_json_field_prob: float = 0.2
    remove_entities_prob: float = 0.0
    shuffle_entities: bool = False
    remove_entity_prob: float = 0.0
    synthetic_entity_label_prob: float = 0.2
    remove_relations_prob: float = 0.2
    swap_head_tail_prob: float = 0.2
    remove_classification_prob: float = 0.0
    shuffle_classification_labels: bool = True
    remove_classification_label_prob: float = 0.5
    synthetic_label_prob: float = 0.5
    include_true_label_prob: float = 0.5
    max_num_labels: int = 1000


@dataclass
class SchemaField:
    """One schema field after normalization."""

    name: str
    dtype: str = "list"
    threshold: float | None = None
    description: str | None = None
    choices: tuple[str, ...] = ()
    validators: tuple[Any, ...] = ()


@dataclass
class SchemaGroup:
    """One encoded schema group."""

    task: str
    name: str
    prompt: str
    fields: tuple[SchemaField, ...] = ()
    tokens: tuple[str, ...] = ()
    examples: tuple[tuple[str, str], ...] = ()
    label_descriptions: dict[str, str] = field(default_factory=dict)
    example_mode: str = "none"
    choices: dict[str, tuple[str, ...]] = field(default_factory=dict)
    true_labels: tuple[str, ...] | None = None


_WS_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)
_CHAR_PATTERN = re.compile(r"[A-Za-z0-9@._\-+]+|\S")


def _whitespace_words(text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
    """Yield ``(token, start, end)`` on the original string."""
    for match in _WS_PATTERN.finditer(text):
        token = match.group()
        yield (token.lower() if lower else token), match.start(), match.end()


def _char_words(text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
    """Yield Latin runs intact and every other non-space as a token."""
    for match in _CHAR_PATTERN.finditer(text):
        token = match.group()
        yield (token.lower() if lower else token), match.start(), match.end()


_SPLITTERS = {"whitespace": _whitespace_words, "char": _char_words}


def _resolve_word_splitter(word_splitter: Any) -> Callable[..., Iterator[tuple[str, int, int]]]:
    """Resolve a splitter name or callable."""
    if word_splitter is None:
        word_splitter = "whitespace"
    if isinstance(word_splitter, str):
        try:
            return _SPLITTERS[word_splitter]
        except KeyError:
            supported = ", ".join(repr(name) for name in sorted(_SPLITTERS))
            raise ValueError(f"Unknown word_splitter {word_splitter!r}. Supported names: {supported}.") from None
    if callable(word_splitter):
        return word_splitter
    raise TypeError(f"word_splitter must be a name or callable, got {type(word_splitter).__name__}")


def _task_type_id(task_type: str) -> int:
    try:
        return TASK_TYPE_TO_ID[task_type]
    except KeyError:
        raise ValueError(f"unknown task type {task_type!r}") from None


def _normalize_text(text: str) -> str:
    """Ensure text ends with sentence punctuation."""
    if not text:
        return "."
    if not text.endswith((".", "!", "?")):
        return text + "."
    return text


def _resolve_schema(schema: Any) -> Any:
    """Unwrap a schema builder into its dict."""
    if hasattr(schema, "build"):
        return schema.build()
    if hasattr(schema, "schema"):
        return schema.schema
    return schema


def _normalize_overlap_policy(policy: str | None, default: str | None = None) -> str:
    """Return a canonical overlap policy."""
    selected = default if policy is None else policy
    if selected is None:
        raise ValueError("overlap_policy=None requires an architecture default")
    if not isinstance(selected, str):
        raise TypeError("overlap_policy must be a string or None")
    key = selected.strip().lower().replace("-", "_")
    try:
        return _OVERLAP_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown overlap_policy {selected!r}; expected one of: allow, nested, flat/disallow, longest"
        ) from None


def _resolve_overlaps(
    items: Sequence[Any],
    policy: str | None,
    *,
    score: Callable[[Any], float],
    start: Callable[[Any], int],
    end: Callable[[Any], int],
    default: str | None = None,
) -> list[Any]:
    """Resolve half-open spans with shared overlap semantics."""
    canonical = _normalize_overlap_policy(policy, default=default)
    if not items:
        return []
    indexed = list(enumerate(items))

    def rank_key(row):
        index, item = row
        return (-float(score(item)), int(start(item)), int(end(item)), index)

    ranked = sorted(indexed, key=rank_key)
    distinct = []
    seen_boundaries = set()
    for row in ranked:
        item = row[1]
        boundaries = (int(start(item)), int(end(item)))
        if boundaries in seen_boundaries:
            continue
        seen_boundaries.add(boundaries)
        distinct.append(row)
    if canonical == "allow":
        return [item for _, item in distinct]
    if canonical == "nested":
        kept = []
        for row in distinct:
            candidate = row[1]
            candidate_start = int(start(candidate))
            candidate_end = int(end(candidate))
            crossing = False
            for _, existing in kept:
                existing_start = int(start(existing))
                existing_end = int(end(existing))
                overlaps = candidate_start < existing_end and existing_start < candidate_end
                contains = (candidate_start <= existing_start and existing_end <= candidate_end) or (
                    existing_start <= candidate_start and candidate_end <= existing_end
                )
                if overlaps and not contains:
                    crossing = True
                    break
            if not crossing:
                kept.append(row)
        return [item for _, item in kept]
    if canonical == "longest":
        kept = []
        for row in distinct:
            candidate = row[1]
            candidate_start = int(start(candidate))
            candidate_end = int(end(candidate))
            strictly_contained = any(
                int(start(other)) <= candidate_start
                and candidate_end <= int(end(other))
                and (int(start(other)) < candidate_start or candidate_end < int(end(other)))
                for _, other in distinct
            )
            if not strictly_contained:
                kept.append(row)
        return [item for _, item in kept]
    by_end = sorted(
        distinct,
        key=lambda row: (int(end(row[1])), int(start(row[1])), -float(score(row[1])), row[0]),
    )
    ends = [int(end(item)) for _, item in by_end]
    predecessors = [
        bisect.bisect_right(ends, int(start(item)), 0, index) - 1 for index, (_, item) in enumerate(by_end)
    ]
    best: list[tuple] = [(0.0, ())]

    def selection_key(selection: tuple):
        rows = [by_end[index] for index in selection]
        return tuple(rank_key(row) for row in sorted(rows, key=rank_key))

    for index, (_, item) in enumerate(by_end):
        previous_score, previous_selection = best[predecessors[index] + 1]
        with_item = (previous_score + float(score(item)), previous_selection + (index,))
        without_item = best[index]
        if with_item[0] > without_item[0] or (
            with_item[0] == without_item[0]
            and (
                len(with_item[1]) > len(without_item[1])
                or (
                    len(with_item[1]) == len(without_item[1])
                    and selection_key(with_item[1]) < selection_key(without_item[1])
                )
            )
        ):
            best.append(with_item)
        else:
            best.append(without_item)
    selected = [by_end[index] for index in best[-1][1]]
    selected.sort(key=rank_key)
    return [item for _, item in selected]


def _finalize_spans(
    raw_spans: Sequence[RawSpan],
    *,
    dtype: str = "list",
    overlap_policy: str | None = None,
) -> list[RawSpan]:
    """Keep non-overlapping spans, highest confidence first when policy is unset."""
    if overlap_policy is None:
        ranked = sorted(raw_spans, key=lambda span: span[1], reverse=True)
        spans: list[RawSpan] = []
        for candidate in ranked:
            candidate_start, candidate_end = candidate[2], candidate[3]
            if any(candidate_start < existing[3] and existing[2] < candidate_end for existing in spans):
                continue
            spans.append(candidate)
    else:
        spans = _resolve_overlaps(
            list(raw_spans),
            overlap_policy,
            score=lambda span: span[1],
            start=lambda span: span[2],
            end=lambda span: span[3],
        )
    return spans if dtype == "list" else spans[:1]


def _default_cardinality(dtype: str | None, is_anchor: bool) -> str:
    if is_anchor:
        return "required_one"
    if dtype == "str":
        return "optional_one"
    return "zero_or_more"


@dataclass(frozen=True)
class TextChunk:
    """One word window and its offsets into the original text."""

    text: str
    start_char: int
    end_char: int
    start_word: int
    end_word: int


def chunk_words(
    words: str | Sequence[Any],
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    word_splitter: Any = None,
) -> list[TextChunk]:
    """Split text or a word sequence into overlapping windows.

    A string is split with the whitespace (or configured) splitter. Offsets
    index that original string. A sequence of strings, or of
    ``(token, start, end)`` tuples, is windowed directly.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = _resolve_word_splitter(word_splitter)
    if isinstance(words, str):
        tokens = list(splitter(words, False))
        source = words
    else:
        tokens = []
        pieces = []
        for index, word in enumerate(words):
            if isinstance(word, (tuple, list)) and len(word) >= 3:
                tokens.append((str(word[0]), int(word[1]), int(word[2])))
            else:
                tokens.append((str(word), index, index + 1))
            pieces.append(str(word if not isinstance(word, (tuple, list)) else word[0]))
        source = None
    if not tokens:
        text = words if isinstance(words, str) else ""
        return [TextChunk(text=text, start_char=0, end_char=len(text), start_word=0, end_word=0)]
    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    start_word = 0
    while start_word < len(tokens):
        end_word = min(start_word + chunk_size, len(tokens))
        start_char = tokens[start_word][1]
        end_char = tokens[end_word - 1][2]
        if source is None:
            window = " ".join(token for token, _, _ in tokens[start_word:end_word])
            chunks.append(TextChunk(window, start_char, end_char, start_word, end_word))
        else:
            chunks.append(TextChunk(source[start_char:end_char], start_char, end_char, start_word, end_word))
        if end_word == len(tokens):
            break
        start_word += step
    return chunks


def _is_span_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "text" in value
        and "start" in value
        and "end" in value
        and isinstance(value["start"], int)
        and isinstance(value["end"], int)
    )


def _is_classification_dict(value: Any) -> bool:
    return isinstance(value, dict) and "label" in value and "confidence" in value


def _chunk_field(chunk: Any, name: str) -> Any:
    if isinstance(chunk, dict):
        return chunk[name]
    return getattr(chunk, name)


def remap_result_spans(result: Any, original_text: str, chunk: Any) -> Any:
    """Add a chunk's character offset onto span dicts."""
    if isinstance(result, list):
        return [remap_result_spans(item, original_text, chunk) for item in result]
    if isinstance(result, dict):
        remapped = {key: remap_result_spans(value, original_text, chunk) for key, value in result.items()}
        if _is_span_dict(remapped):
            start = int(remapped["start"]) + int(_chunk_field(chunk, "start_char"))
            end = int(remapped["end"]) + int(_chunk_field(chunk, "start_char"))
            remapped["start"] = start
            remapped["end"] = end
            if 0 <= start <= end <= len(original_text):
                remapped["text"] = original_text[start:end]
        return remapped
    return result


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _canonical_key(value: Any) -> str:
    if isinstance(value, dict):
        return repr(sorted((key, _canonical_key(item)) for key, item in value.items() if key != "confidence"))
    if isinstance(value, list):
        return repr([_canonical_key(item) for item in value])
    return repr(value)


def _representative_confidence(value: Any) -> float:
    if isinstance(value, dict):
        if isinstance(value.get("confidence"), (int, float)) and not isinstance(value.get("confidence"), bool):
            return float(value["confidence"])
        nested = [_representative_confidence(item) for item in value.values()]
        return max(nested) if nested else 0.0
    if isinstance(value, list):
        nested = [_representative_confidence(item) for item in value]
        return max(nested) if nested else 0.0
    return 0.0


def _dedupe_items(items: list[Any], overlap_policy: str | None = None) -> list[Any]:
    span_items = [item for item in items if _is_span_dict(item)]
    other_items = [item for item in items if not _is_span_dict(item)]
    deduped: list[Any] = []
    if span_items:
        selected = _resolve_overlaps(
            span_items,
            overlap_policy,
            default="allow",
            score=lambda item: float(item.get("confidence", 0.0)),
            start=lambda item: int(item["start"]),
            end=lambda item: int(item["end"]),
        )
        deduped.extend(sorted(selected, key=lambda item: (item["start"], item["end"], item.get("text", ""))))
    seen_other: dict[str, int] = {}
    other_deduped: list[Any] = []
    for item in other_items:
        key = _canonical_key(item)
        if key not in seen_other:
            seen_other[key] = len(other_deduped)
            other_deduped.append(item)
        elif _representative_confidence(item) > _representative_confidence(other_deduped[seen_other[key]]):
            other_deduped[seen_other[key]] = item
    deduped.extend(other_deduped)
    return deduped


def _merge_values(values: list[Any], overlap_policy: str = "disallow") -> Any:
    non_empty = [value for value in values if value not in (None, {}, [])]
    if not non_empty:
        return values[0] if values else None
    if all(_is_classification_dict(value) for value in non_empty):
        return max(non_empty, key=lambda value: value.get("confidence", 0.0))
    if all(isinstance(value, str) for value in non_empty):
        counts = Counter(non_empty)
        return max(non_empty, key=lambda value: (counts[value], -non_empty.index(value)))
    if all(isinstance(value, list) for value in non_empty):
        items: list[Any] = []
        for value in non_empty:
            items.extend(value)
        return _dedupe_items(items, overlap_policy=overlap_policy)
    if all(isinstance(value, dict) for value in non_empty):
        return _merge_nested_dicts(non_empty, overlap_policy)
    return non_empty[0]


def _merge_nested_dicts(values: list[dict[str, Any]], overlap_policy: str = "disallow") -> dict[str, Any]:
    merged: dict[str, Any] = {}
    keys: list[str] = []
    seen = set()
    for value in values:
        for key in value:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    for key in keys:
        merged[key] = _merge_values([value.get(key) for value in values if key in value], overlap_policy)
    return merged


def _merge_entity_maps(values: list[Any], scalar_labels: set, overlap_policy: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    labels: list[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        for label in value:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    for label in labels:
        items: list[Any] = []
        for value in values:
            if isinstance(value, dict) and label in value:
                items.extend(_as_list(value[label]))
        deduped = _dedupe_items(items, overlap_policy=overlap_policy)
        merged[label] = (deduped[0] if deduped else None) if label in scalar_labels else deduped
    return merged


def _merge_relation_maps(values: list[Any]) -> dict[str, list[Any]]:
    merged: dict[str, list[Any]] = {}
    labels: list[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        for label in value:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    for label in labels:
        items: list[Any] = []
        for value in values:
            if isinstance(value, dict) and label in value:
                items.extend(_as_list(value[label]))
        merged[label] = _dedupe_items(items)
    return merged


def _merge_result_dicts(
    results: list[dict[str, Any]],
    scalar_entity_labels: set | None = None,
    overlap_policy: str = "disallow",
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    keys: list[str] = []
    seen = set()
    for result in results:
        for key in result:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    for key in keys:
        values = [result.get(key) for result in results if key in result]
        if key == "entities":
            merged[key] = _merge_entity_maps(values, scalar_entity_labels or set(), overlap_policy)
        elif key == "relation_extraction":
            merged[key] = _merge_relation_maps(values)
        else:
            merged[key] = _merge_values(values, overlap_policy)
    return merged


def _strip_span_metadata(value: Any, include_confidence: bool, include_spans: bool) -> Any:
    if isinstance(value, list):
        return [_strip_span_metadata(item, include_confidence, include_spans) for item in value]
    if isinstance(value, dict):
        if _is_span_dict(value):
            extras = {key: item for key, item in value.items() if key not in _SPAN_RESERVED}
            if not include_confidence and not include_spans and not extras:
                return value.get("text", "")
            stripped: dict[str, Any] = {"text": value.get("text", "")}
            if include_confidence and "confidence" in value:
                stripped["confidence"] = value["confidence"]
            if include_spans:
                stripped["start"] = value["start"]
                stripped["end"] = value["end"]
            stripped.update(extras)
            return stripped
        if _is_classification_dict(value):
            if include_confidence:
                return {"label": value["label"], "confidence": value["confidence"]}
            return value["label"]
        if "text" in value and "confidence" in value and "start" not in value and "end" not in value:
            if include_confidence:
                return {"text": value["text"], "confidence": value["confidence"]}
            return value["text"]
        return {key: _strip_span_metadata(item, include_confidence, include_spans) for key, item in value.items()}
    return value


def merge_chunk_results(
    original_text: str,
    chunks: Sequence[Any],
    chunk_results: Sequence[dict[str, Any]],
    include_confidence: bool = False,
    include_spans: bool = False,
    scalar_entity_labels: Iterable[str] | None = None,
    overlap_policy: str | None = None,
) -> dict[str, Any]:
    """Merge formatted chunk results onto document character offsets."""
    if len(chunks) != len(chunk_results):
        raise ValueError("chunks and chunk_results must have the same length")
    policy = _normalize_overlap_policy(overlap_policy, default="disallow")
    remapped = [remap_result_spans(result, original_text, chunk) for chunk, result in zip(chunks, chunk_results)]
    merged = _merge_result_dicts(remapped, set(scalar_entity_labels or ()), policy)
    return _strip_span_metadata(merged, include_confidence, include_spans)


def _is_score(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _format_classification(value: object, include_confidence: bool) -> object:
    """Format a label pair or a list of label pairs."""
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, (list, tuple)):
            if include_confidence:
                return [{"label": item[0], "confidence": item[1]} for item in value]
            return [item[0] for item in value]
        if len(value) == 2 and isinstance(first, str) and _is_score(value[1]):
            label, confidence = value
            return {"label": label, "confidence": confidence} if include_confidence else label
    return value


def format_entity_dict(entities: dict, include_confidence: bool) -> dict:
    """Deduplicate an entity-type map."""
    formatted = {}
    for name, spans in entities.items():
        if isinstance(spans, list):
            unique = []
            seen = set()
            for span in spans:
                if isinstance(span, tuple):
                    text, confidence, start, end = span
                    if text and (text.lower(), start, end) not in seen:
                        seen.add((text.lower(), start, end))
                        unique.append({"text": text, "confidence": confidence} if include_confidence else text)
                elif isinstance(span, dict):
                    text = span.get("text", "")
                    if "start" in span and "end" in span:
                        key = (text.lower(), span["start"], span["end"])
                    else:
                        key = (text.lower(), None, None)
                    if text and key not in seen:
                        seen.add(key)
                        unique.append(span)
                elif span and span.lower() not in seen:
                    seen.add(span.lower())
                    unique.append(span)
            formatted[name] = unique
        elif isinstance(spans, tuple):
            text, confidence, _, _ = spans
            formatted[name] = {"text": text, "confidence": confidence} if include_confidence and text else text
        else:
            formatted[name] = spans or None
    return formatted


def format_struct(struct: dict, include_confidence: bool) -> dict:
    """Deduplicate one structure instance."""
    formatted = {}
    for field_name, value in struct.items():
        if isinstance(value, list):
            unique = []
            seen = set()
            for item in value:
                if isinstance(item, tuple):
                    text, confidence, start, end = item
                    if text and (text.lower(), start, end) not in seen:
                        seen.add((text.lower(), start, end))
                        unique.append({"text": text, "confidence": confidence} if include_confidence else text)
                elif isinstance(item, dict):
                    text = item.get("text", "")
                    if "start" in item and "end" in item:
                        key = (text.lower(), item["start"], item["end"])
                    else:
                        key = (text.lower(), None, None)
                    if text and key not in seen:
                        seen.add(key)
                        unique.append(item)
                elif item and item.lower() not in seen:
                    seen.add(item.lower())
                    unique.append(item)
            formatted[field_name] = unique
        elif isinstance(value, tuple):
            text, confidence, _, _ = value
            formatted[field_name] = {"text": text, "confidence": confidence} if include_confidence and text else text
        elif value:
            formatted[field_name] = value
        else:
            formatted[field_name] = None
    return formatted


def format_results(
    results: dict,
    include_confidence: bool = False,
    requested_relations: list[str] | None = None,
    classification_tasks: list[str] | None = None,
) -> dict[str, Any]:
    """Format raw extraction results into the public payload."""
    formatted: dict[str, Any] = {}
    relations: dict[str, Any] = {}
    requested_relations = requested_relations or []
    classification_tasks = classification_tasks or []
    for key, value in results.items():
        is_classification = key in classification_tasks
        is_relation = False
        if not is_classification:
            if key in requested_relations:
                is_relation = True
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], tuple) and len(value[0]) == 2:
                    is_relation = True
                elif isinstance(value[0], dict) and "head" in value[0] and "tail" in value[0]:
                    is_relation = True
        if is_classification:
            formatted[key] = _format_classification(value, include_confidence)
        elif is_relation:
            relations[key] = value if isinstance(value, list) else []
        elif isinstance(value, list):
            if len(value) == 0:
                formatted[key] = {} if key == "entities" else value
            elif isinstance(value[0], dict):
                if key == "entities":
                    formatted[key] = format_entity_dict(value[0], include_confidence)
                else:
                    formatted[key] = [format_struct(item, include_confidence) for item in value]
            elif isinstance(value[0], tuple):
                if include_confidence:
                    formatted[key] = [{"label": label, "confidence": confidence} for label, confidence in value]
                else:
                    formatted[key] = [label for label, _ in value]
            else:
                formatted[key] = value
        elif isinstance(value, tuple):
            label, confidence = value
            formatted[key] = {"label": label, "confidence": confidence} if include_confidence else label
        elif isinstance(value, dict):
            formatted[key] = format_struct(value, include_confidence)
        else:
            formatted[key] = value
    for relation in requested_relations:
        if relation not in relations:
            relations[relation] = []
    if relations:
        formatted["relation_extraction"] = relations
    return formatted


def _optional_decoding():
    """Return ``decoding_gliner2`` when that module is importable."""
    try:
        from . import decoding_gliner2
    except ImportError:
        return None
    return decoding_gliner2


def _call_decoder(name: str, *args, **kwargs):
    module = _optional_decoding()
    function = getattr(module, name, None) if module is not None else None
    if function is None:
        raise NotImplementedError(f"{name} is not available; expected decoding_gliner2.{name}")
    return function(*args, **kwargs)


def _transform_schema(
    parent: str,
    fields: list[str],
    child_prefix: str,
    prompt: str | None = None,
    examples: list[tuple[str, str]] | None = None,
    label_descriptions: dict[str, str] | None = None,
    example_mode: str = "both",
    rng: Any = None,
) -> list[str]:
    """Turn one schema group into structural word tokens."""
    prompt_str = parent
    if prompt:
        prompt_str = f"{parent}: {prompt}"
    if example_mode in ("descriptions", "both") and label_descriptions:
        described = [(label, desc) for label, desc in label_descriptions.items() if label in fields]
        if rng is not None:
            rng.shuffle(described)
        for label, desc in described:
            prompt_str += f" {DESC_TOKEN} {label}: {desc}"
    if example_mode in ("few_shot", "both") and examples:
        ordered = list(examples)
        if rng is not None:
            rng.shuffle(ordered)
        for inp, out in ordered:
            if out in fields:
                out_str = out if isinstance(out, str) else ", ".join(out)
                prompt_str += f" {EXAMPLE_TOKEN} {inp} {OUTPUT_TOKEN} {out_str}"
    tokens = ["(", P_TOKEN, prompt_str, "("]
    for field_name in fields:
        tokens.extend([child_prefix, field_name])
    tokens.extend([")", ")"])
    return tokens


def _choice_field_pairs(fields: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Choice fields, including specs that omit an embedded gold value."""
    pairs = []
    for fname, fval in fields.items():
        if isinstance(fval, dict) and "choices" in fval and ("value" in fval or fval.get("choices")):
            pairs.append((fname, fval))
    return pairs


def _classification_prefix(schema: Mapping[str, Any], rng: Any = None) -> list[str]:
    """Word tokens prepended for JSON choice fields."""
    prefix_tokens: list[str] = []
    for struct in schema.get("json_structures", []) or []:
        for parent, fields in struct.items():
            cls_fields = _choice_field_pairs(fields)
            if rng is not None:
                rng.shuffle(cls_fields)
            inner: list[str] = []
            for fname, fval in cls_fields:
                choices = list(fval["choices"])
                if rng is not None:
                    rng.shuffle(choices)
                choice_tokens: list[str] = []
                for index, choice in enumerate(choices):
                    if index > 0:
                        choice_tokens.append("|")
                    choice_tokens.append(str(choice))
                inner.extend([fname, "("] + choice_tokens + [")", ","])
            if inner:
                inner = inner[:-1]
                prefix_tokens.extend(["(", f"{parent}:", *inner, ")"])
    return prefix_tokens


def _schema_token_groups(schema: Mapping[str, Any]) -> tuple[list[list[str]], list[str]]:
    """Inference schema order: structures, entities, relations, classifications."""
    schemas: list[list[str]] = []
    types: list[str] = []
    if "json_structures" in schema:
        json_descs = schema.get("json_descriptions", {}) or {}
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for item in schema["json_structures"] or []:
            for parent, fields in item.items():
                groups.setdefault(parent, []).append(fields)
        for parent, occurrences in groups.items():
            common: list[str] = []
            seen_fields = set()
            for occ in occurrences:
                for field_name in occ:
                    if field_name not in seen_fields:
                        common.append(field_name)
                        seen_fields.add(field_name)
            if not common:
                continue
            descs = json_descs.get(parent, {}) or {}
            mode = "descriptions" if descs else "none"
            schemas.append(_transform_schema(parent, common, C_TOKEN, label_descriptions=descs, example_mode=mode))
            types.append("json_structures")
    if "entities" in schema:
        entity_fields = list((schema.get("entities") or {}).keys())
        descs = schema.get("entity_descriptions", {}) or {}
        if entity_fields:
            mode = "descriptions" if descs else "none"
            schemas.append(
                _transform_schema("entities", entity_fields, E_TOKEN, label_descriptions=descs, example_mode=mode)
            )
            types.append("entities")
    if "relations" in schema:
        relation_descriptions = schema.get("relation_descriptions", {}) or {}
        groups = {}
        for item in schema["relations"] or []:
            for parent, fields in item.items():
                groups.setdefault(parent, []).append(fields)
        for parent, occurrences in groups.items():
            if not occurrences:
                continue
            field_names = list(occurrences[0].keys())
            if not any(all(field in occ for field in field_names) for occ in occurrences):
                continue
            schemas.append(_transform_schema(parent, field_names, R_TOKEN, prompt=relation_descriptions.get(parent)))
            types.append("relations")
    if "classifications" in schema:
        for item in schema["classifications"] or []:
            labels = list(item["labels"])
            schemas.append(
                _transform_schema(
                    item["task"],
                    labels,
                    L_TOKEN,
                    prompt=item.get("prompt"),
                    examples=item.get("examples", []) or [],
                    label_descriptions=item.get("label_descriptions", {}) or {},
                    example_mode="both",
                )
            )
            types.append("classifications")
    return schemas, types


def _fields_from_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(tokens[index + 1]) for index in range(4, len(tokens) - 2, 2)]


def _group_name(tokens: Sequence[str], task_type: str) -> str:
    if task_type == "entities":
        return "entities"
    if len(tokens) > 2:
        return str(tokens[2]).split(f" {DESC_TOKEN} ")[0]
    return task_type


def _public_metadata(source: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Metadata the span decoder needs to map ids back to strings."""
    if hasattr(source, "build") or hasattr(source, "_entity_metadata"):
        classifications = schema.get("classifications", []) or []
        return {
            "field_metadata": dict(getattr(source, "_field_metadata", {}) or {}),
            "entity_metadata": dict(getattr(source, "_entity_metadata", {}) or {}),
            "relation_metadata": dict(getattr(source, "_relation_metadata", {}) or {}),
            "relation_descriptions": schema.get("relation_descriptions", {}) or {},
            "field_orders": dict(getattr(source, "_field_orders", {}) or {}),
            "entity_order": list(getattr(source, "_entity_order", []) or []),
            "relation_order": list(getattr(source, "_relation_order", []) or []),
            "classification_tasks": [item["task"] for item in classifications],
            "entity_attribute_groups": getattr(source, "_entity_attribute_groups", {}) or {},
            "entity_attribute_prompt_labels": dict(getattr(source, "_entity_attribute_prompt_labels", {}) or {}),
            "entity_attribute_labels": set(getattr(source, "_entity_attribute_labels", ()) or ()),
        }
    entities = schema.get("entities")
    entity_order = list(entities.keys()) if isinstance(entities, dict) else []
    classifications = schema.get("classifications", []) or []
    relation_order = []
    for item in schema.get("relations") or []:
        if isinstance(item, Mapping):
            for name in item:
                key = str(name)
            if key not in relation_order:
                relation_order.append(key)
    field_metadata = {}
    for item in schema.get("json_structures") or []:
        if not isinstance(item, Mapping):
            continue
        for parent, fields in item.items():
            if not isinstance(fields, Mapping):
                continue
            for fname, spec in fields.items():
                if not isinstance(spec, Mapping):
                    continue
                entry = {}
                if spec.get("dtype"):
                    entry["dtype"] = spec["dtype"]
                if spec.get("threshold") is not None:
                    entry["threshold"] = spec["threshold"]
                if spec.get("choices"):
                    entry["choices"] = list(spec["choices"])
                if entry:
                    field_metadata[f"{parent}.{fname}"] = entry
    return {
        "field_metadata": field_metadata,
        "entity_metadata": {},
        "relation_metadata": {},
        "relation_descriptions": schema.get("relation_descriptions", {}) or {},
        "field_orders": {},
        "entity_order": entity_order,
        "relation_order": relation_order,
        "classification_tasks": [item["task"] for item in classifications],
        "entity_attribute_groups": {},
        "entity_attribute_prompt_labels": {},
        "entity_attribute_labels": set(),
    }


def _choice_fields(schema: Mapping[str, Any]) -> dict[str, list[str]]:
    choices = {}
    for struct in schema.get("json_structures", []) or []:
        for parent, fields in struct.items():
            for fname, fval in fields.items():
                if isinstance(fval, dict) and "choices" in fval:
                    choices[f"{parent}.{fname}"] = list(fval["choices"])
    return choices


def _field_dtypes(schema: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    dtypes: dict[str, dict[str, str]] = {}
    for struct in schema.get("json_structures", []) or []:
        for parent, fields in struct.items():
            parent_dtypes = dtypes.setdefault(str(parent), {})
            for fname, fval in fields.items():
                if isinstance(fval, dict) and fval.get("dtype"):
                    parent_dtypes[str(fname)] = str(fval["dtype"])
                elif isinstance(fval, str) and fval in ("str", "list"):
                    parent_dtypes[str(fname)] = fval
    return dtypes


def _normalize_record_metadata(
    raw: Mapping[str, Any] | None,
    field_dtypes: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate ``record_metadata`` into a JSON-ready dict."""
    if not raw:
        return {}
    field_dtypes = field_dtypes or {}
    normalized: dict[str, dict[str, Any]] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, Mapping):
            raise ValueError(f"record_metadata[{name!r}] must be a mapping")
        mode = cfg.get("mode")
        if mode is None:
            continue
        if mode not in VALID_RECORD_MODES:
            raise ValueError(f"record_metadata[{name!r}].mode must be one of {VALID_RECORD_MODES}, got {mode!r}")
        anchor = cfg.get("anchor")
        if mode == "natural" and not anchor:
            raise ValueError(f"record_metadata[{name!r}] mode='natural' requires 'anchor'")
        if mode != "natural" and anchor:
            raise ValueError(f"record_metadata[{name!r}] mode={mode!r} must not set 'anchor'")
        policy = cfg.get("occurrence_policy", "latent_all")
        if policy not in ("all", "first", "error_on_ambiguous", "latent_all"):
            raise ValueError(f"record_metadata[{name!r}].occurrence_policy invalid: {policy!r}")
        dtypes = field_dtypes.get(name, {})
        fields_out: dict[str, dict[str, Any]] = {}
        for fname, fcfg in (cfg.get("fields", {}) or {}).items():
            fcfg = fcfg or {}
            is_anchor = bool(anchor) and fname == anchor
            card = fcfg.get("cardinality")
            if card is None:
                card = _default_cardinality(dtypes.get(fname), is_anchor)
            elif card not in CARDINALITY_TO_ID:
                raise ValueError(f"record_metadata[{name!r}].fields[{fname!r}].cardinality invalid: {card!r}")
            fields_out[fname] = {"cardinality": card, "exclusive": bool(fcfg.get("exclusive", False))}
        normalized[str(name)] = {
            "mode": mode,
            "anchor": anchor,
            "occurrence_policy": policy,
            "fields": fields_out,
        }
    return normalized


def _compile_record_specs(
    groups: Sequence[Mapping[str, Any]], record_metadata: Mapping[str, Any] | None, field_dtypes
):
    """Compile record specs keyed by schema-group index."""
    normalized = _normalize_record_metadata(record_metadata, field_dtypes=field_dtypes)
    if not normalized:
        return []
    queries = []
    query_id = 0
    by_task: dict[int, list[dict]] = {}
    for task_index, group in enumerate(groups):
        if group["task_type"] == "classifications" or not group["fields"]:
            continue
        for role_index, role_name in enumerate(group["fields"]):
            query = {
                "query_id": query_id,
                "task_index": task_index,
                "task_type": group["task_type"],
                "task_name": group["name"],
                "role_index": role_index,
                "role_name": role_name,
            }
            queries.append(query)
            by_task.setdefault(task_index, []).append(query)
            query_id += 1
    specs = []
    for task_index, task_queries in by_task.items():
        name = task_queries[0]["task_name"]
        if task_queries[0]["task_type"] not in RECORD_TASK_TYPES or name not in normalized:
            continue
        cfg = normalized[name]
        mode = cfg["mode"]
        anchor_name = cfg.get("anchor")
        fields_cfg = cfg.get("fields", {})
        fields = []
        anchor_query_id = None
        for query in sorted(task_queries, key=lambda item: item["role_index"]):
            fcfg = fields_cfg.get(query["role_name"], {})
            is_anchor = mode == "natural" and query["role_name"] == anchor_name
            card = fcfg.get("cardinality")
            if card is None:
                dtype = (field_dtypes or {}).get(name, {}).get(query["role_name"])
                card = _default_cardinality(dtype, is_anchor)
            fields.append(
                {
                    "query_id": query["query_id"],
                    "name": query["role_name"],
                    "cardinality": card,
                    "is_anchor": is_anchor,
                }
            )
            if is_anchor:
                anchor_query_id = query["query_id"]
        if mode == "natural" and anchor_query_id is None:
            raise ValueError(f"record {name!r} declares anchor {anchor_name!r} but no matching field was found")
        specs.append(
            {
                "task_index": task_index,
                "task_name": name,
                "mode": mode,
                "fields": fields,
                "anchor_query_id": anchor_query_id,
            }
        )
    return specs


def _find_choice_idx(choice: str, tokens: Sequence[str]) -> int:
    choice_lower = choice.lower()
    for index, token in enumerate(tokens):
        if token.lower() == choice_lower:
            return index
    return -1


def _find_spans(
    scores: torch.Tensor,
    threshold: float,
    text: str,
    start_map: Sequence[int],
    end_map: Sequence[int],
) -> list[RawSpan]:
    """Return spans at or above ``threshold`` as ``(text, score, start, end)``."""
    text_len = len(start_map)
    valid = torch.where(scores >= threshold)
    spans: list[RawSpan] = []
    for start, width in zip(valid[0].tolist(), valid[1].tolist()):
        end = start + width + 1
        if not (0 <= start < text_len and end <= text_len):
            continue
        try:
            char_start = start_map[start]
            char_end = end_map[end - 1]
            span_text = text[char_start:char_end].strip()
        except (IndexError, KeyError):
            continue
        if span_text:
            spans.append((span_text, float(scores[start, width].item()), int(char_start), int(char_end)))
    return spans


def _format_spans(spans: Sequence[RawSpan], include_confidence: bool, include_spans: bool) -> list[Any]:
    if include_spans and include_confidence:
        return [{"text": span[0], "confidence": span[1], "start": span[2], "end": span[3]} for span in spans]
    if include_spans:
        return [{"text": span[0], "start": span[2], "end": span[3]} for span in spans]
    if include_confidence:
        return [{"text": span[0], "confidence": span[1]} for span in spans]
    return [span[0] for span in spans]


def _format_one_span(span: RawSpan, include_confidence: bool, include_spans: bool) -> Any:
    text, confidence, char_start, char_end = span
    if include_spans and include_confidence:
        return {"text": text, "confidence": confidence, "start": char_start, "end": char_end}
    if include_spans:
        return {"text": text, "start": char_start, "end": char_end}
    if include_confidence:
        return {"text": text, "confidence": confidence}
    return text


def _passes_validators(text: str, validators: Sequence[Any]) -> bool:
    for validator in validators or []:
        if hasattr(validator, "validate") and not validator.validate(text):
            return False
    return True


def _resolve_classification_config(
    prompt_str: str, classifications: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Find the classification config that owns ``prompt_str``."""
    best = None
    for config in classifications:
        task = config.get("task", "")
        if not task or not prompt_str.startswith(task):
            continue
        rest = prompt_str[len(task) :]
        if rest == "" or rest[0] in (":", " "):
            if best is None or len(task) > len(best.get("task", "")):
                best = config
    if best is None:
        best = next((config for config in classifications if prompt_str.startswith(config.get("task", ""))), None)
    return best


def _classification_probs(logits: torch.Tensor, multi_label: bool, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(logits)
    if activation == "softmax":
        return torch.softmax(logits, dim=-1)
    return torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=-1)


def _doc_axis(tensor: torch.Tensor, doc_len: int) -> torch.Tensor:
    """Take the document-word rows, leaving an empty axis when there are none."""
    if doc_len <= 0:
        return tensor[..., :0, :]
    return tensor[..., -doc_len:, :]


def _decode_classification_group(
    logits: torch.Tensor,
    config: Mapping[str, Any],
    threshold: float,
    temperature: float,
    activated: bool = False,
) -> Any:
    """Decode one classification group to a label pair or a list of pairs."""
    labels = list(config["labels"])
    flat = logits.detach().float().cpu().reshape(-1)
    if flat.numel() != len(labels):
        raise ValueError(f"classification logits ({flat.numel()}) do not match labels ({len(labels)})")
    multi_label = bool(config.get("multi_label", False))
    if activated:
        probs = flat
    else:
        if temperature <= 0:
            raise ValueError("classification temperature must be > 0")
        probs = _classification_probs(flat / temperature, multi_label, str(config.get("class_act", "auto")))
    cls_threshold = config.get("cls_threshold", threshold)
    if multi_label:
        chosen = [
            (labels[index], float(probs[index].item()))
            for index in range(len(labels))
            if float(probs[index]) >= cls_threshold
        ]
        if not chosen:
            best = int(torch.argmax(probs).item())
            chosen = [(labels[best], float(probs[best].item()))]
        return chosen
    best = int(torch.argmax(probs).item())
    return (labels[best], float(probs[best].item()))


def _group_attr(group: Any, name: str, default=None):
    if isinstance(group, dict):
        return group.get(name, default)
    return getattr(group, name, default)


def _assign_entity_attributes(
    logits: torch.Tensor, start: int, width: int, group_indices: dict[str, Any], entity_name: str
) -> dict[str, Any]:
    assigned: dict[str, Any] = {}
    for group_name, (labels, indices, group) in group_indices.items():
        applies_to = _group_attr(group, "applies_to")
        if applies_to is not None and entity_name not in applies_to:
            continue
        values = logits[indices, start, width]
        if _group_attr(group, "multi_label", False):
            probabilities = torch.sigmoid(values)
            threshold = float(_group_attr(group, "threshold", 0.5))
            assigned[group_name] = [
                {"label": labels[index], "confidence": float(probabilities[index].item())}
                for index in range(len(labels))
                if float(probabilities[index]) >= threshold
            ]
        else:
            probabilities = torch.softmax(values, dim=-1)
            best = int(probabilities.argmax())
            assigned[group_name] = {"label": labels[best], "confidence": float(probabilities[best].item())}
    return assigned


def _parse_extract_field(spec: str) -> dict[str, Any]:
    """Parse ``name::str::[a|b]::desc`` into a field dict."""
    parts = spec.split("::")
    dtype = "list"
    choices = None
    description = None
    dtype_set = False
    for part in parts[1:]:
        if part in ("str", "list"):
            dtype = part
            dtype_set = True
        elif part.startswith("[") and part.endswith("]"):
            choices = [item.strip() for item in part[1:-1].split("|")]
            if not dtype_set:
                dtype = "str"
        else:
            description = part
    parsed: dict[str, Any] = {"dtype": dtype, "value": ""}
    if choices:
        parsed["choices"] = choices
    if description:
        parsed["description"] = description
    return parsed


def _parse_field_value(value: Any) -> Any:
    """Normalize one JSON field value into a dict spec."""
    if isinstance(value, Mapping):
        return dict(value)
    if value in ("", None):
        return {"dtype": "list", "value": ""}
    if not isinstance(value, str):
        raise TypeError("json structure field values must be dicts")
    if value in ("str", "list"):
        return {"dtype": value, "value": ""}
    if "::" in value:
        return _parse_extract_field(value)
    raise TypeError("json structure field values must be dicts")


def _canonicalize_schema(schema: Any) -> dict[str, Any]:
    """Normalize entities, classifications, relations, and JSON structures."""
    schema = _resolve_schema(schema)
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a dict")
    schema = copy.deepcopy(dict(schema))
    entities = schema.get("entities", None)
    if isinstance(entities, list):
        schema["entities"] = {str(name): "" for name in entities}
    elif isinstance(entities, Mapping):
        descriptions = dict(schema.get("entity_descriptions") or {})
        normalized: dict[str, Any] = {}
        for name, value in entities.items():
            key = str(name)
            if isinstance(value, str):
                if value:
                    descriptions.setdefault(key, value)
                normalized[key] = ""
            elif isinstance(value, Mapping):
                desc = value.get("description")
                if isinstance(desc, str) and desc:
                    descriptions.setdefault(key, desc)
                normalized[key] = dict(value)
            elif value in ("", None):
                normalized[key] = ""
            else:
                raise TypeError(f"entities[{key!r}] must be a string or dict")
        schema["entities"] = normalized
        if descriptions:
            schema["entity_descriptions"] = descriptions
    elif entities is not None:
        raise TypeError("entities must be a list or dict")

    structures = schema.get("json_structures", None)
    if structures is not None:
        if not isinstance(structures, list):
            raise TypeError("json_structures must be a list of {parent: {field: value}}")
        parsed_structures = []
        for item in structures:
            if not isinstance(item, Mapping) or not item:
                raise ValueError("json_structures items must be {parent: {field: value}}")
            parsed_item = {}
            for parent, fields in item.items():
                if not isinstance(fields, Mapping):
                    raise TypeError(f"json_structures[{parent!r}] values must be dicts")
                parsed_item[str(parent)] = {str(fname): _parse_field_value(fval) for fname, fval in fields.items()}
            parsed_structures.append(parsed_item)
        schema["json_structures"] = parsed_structures

    relations = schema.get("relations", None)
    if relations is not None:
        if not isinstance(relations, list):
            raise TypeError("relations must be a list of {name: {head, tail}}")
        for item in relations:
            if not isinstance(item, Mapping) or not item:
                raise ValueError("relations items must be {name: {head, tail}}")
            for name, endpoints in item.items():
                if not isinstance(endpoints, Mapping):
                    raise TypeError(f"relations[{name!r}] must be a dict of endpoints")
                schema_fields = {str(key) for key in endpoints}
                if "head" not in schema_fields or "tail" not in schema_fields:
                    raise ValueError(f"relations[{name!r}] must include head and tail")

    classifications = schema.get("classifications", None)
    if classifications is not None:
        if not isinstance(classifications, list):
            raise TypeError("classifications must be a list of {task, labels}")
        for item in classifications:
            if not isinstance(item, Mapping) or "task" not in item or "labels" not in item:
                raise ValueError("classifications items must be {task, labels}")
            if not isinstance(item["labels"], (list, tuple)):
                raise TypeError(f"classifications[{item['task']!r}].labels must be a list")
    return schema


def _empty_labels() -> dict[str, dict[str, Any]]:
    return {"entities": {}, "classifications": {}, "relations": {}, "json_structures": {}}


def _as_instances(value: Any, kind: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        return [value]
    raise TypeError(f"{kind} labels must be a list of instances or one instance dict")


def _canonicalize_labels(labels: Any, schema: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Check label keys against the schema. Spans are resolved after tokenization."""
    if not isinstance(labels, Mapping):
        raise TypeError("labels must be a dict")
    unknown = set(labels) - {"entities", "classifications", "relations", "json_structures"}
    if unknown:
        raise ValueError(f"unknown label keys {sorted(unknown)}")
    normalized = _empty_labels()
    entity_names = set((schema.get("entities") or {}).keys())
    for name, value in (labels.get("entities") or {}).items():
        if name not in entity_names:
            raise ValueError(f"entity label {name!r} is not in the schema")
        normalized["entities"][str(name)] = value if isinstance(value, list) else [value]

    task_labels = {item["task"]: list(item["labels"]) for item in schema.get("classifications") or []}
    for task, value in (labels.get("classifications") or {}).items():
        if task not in task_labels:
            raise ValueError(f"classification label task {task!r} is not in the schema")
        true = value if isinstance(value, list) else [value]
        missing = [item for item in true if item not in task_labels[task]]
        if missing:
            raise ValueError(f"classification labels {missing} are not in task {task!r}")
        normalized["classifications"][str(task)] = [str(item) for item in true]

    relation_names = set()
    for item in schema.get("relations") or []:
        relation_names.update(item.keys())
    for name, value in (labels.get("relations") or {}).items():
        if name not in relation_names:
            raise ValueError(f"relation label {name!r} is not in the schema")
        normalized["relations"][str(name)] = _as_instances(value, "relation")

    structure_names = set()
    for item in schema.get("json_structures") or []:
        structure_names.update(item.keys())
    for name, value in (labels.get("json_structures") or {}).items():
        if name not in structure_names:
            raise ValueError(f"structure label {name!r} is not in the schema")
        normalized["json_structures"][str(name)] = _as_instances(value, "structure")
    return normalized


def _rename_map(rows: list[Any], real2syn: Mapping[str, str]) -> list[Any]:
    renamed = []
    for row in rows:
        if isinstance(row, Mapping):
            renamed.append({real2syn.get(str(key), str(key)): value for key, value in row.items()})
        else:
            renamed.append(row)
    return renamed


def _choice_map(occurrences: Sequence[Mapping[str, Any]], names: Sequence[str], real2syn: Mapping[str, str]) -> dict:
    choices: dict[str, tuple[str, ...]] = {}
    for occ in occurrences:
        for fname, fval in occ.items():
            key = real2syn.get(str(fname), str(fname))
            if key in names and isinstance(fval, Mapping) and fval.get("choices"):
                choices[key] = tuple(str(choice) for choice in fval["choices"])
    return choices


def _make_group(
    task: str,
    parent: str,
    fields: Sequence[str],
    tokens: Sequence[str],
    choices: Mapping[str, Sequence[str]] | None = None,
    true_labels: tuple[str, ...] | None = None,
) -> SchemaGroup:
    """Build a group whose field names are the encoded child tokens."""
    choice_map = {name: tuple(choices.get(name, ())) for name in fields} if choices else {}
    return SchemaGroup(
        task=task,
        name=parent,
        prompt=tokens[2] if len(tokens) > 2 else parent,
        fields=tuple(SchemaField(name=name, choices=tuple(choice_map.get(name, ()))) for name in fields),
        tokens=tuple(tokens),
        choices={name: tuple(values) for name, values in choice_map.items() if values},
        true_labels=None if true_labels is None else tuple(true_labels),
    )


def _groups_for_tokens(schema: Mapping[str, Any], token_groups: Sequence[Sequence[str]], task_types: Sequence[str]):
    """Pair compiled tokens with parents. Token text comes from ``_schema_token_groups``."""
    pending: list[dict[str, Any]] = []
    if "json_structures" in schema:
        grouped: dict[str, list] = {}
        for item in schema.get("json_structures") or []:
            for parent, fields in item.items():
                grouped.setdefault(parent, []).append(fields)
        for parent, occurrences in grouped.items():
            common: list[str] = []
            seen = set()
            for occ in occurrences:
                for field_name in occ:
                    if field_name not in seen:
                        common.append(field_name)
                        seen.add(field_name)
            if not common:
                continue
            pending.append(
                {
                    "task": "json_structures",
                    "name": parent,
                    "fields": common,
                    "choices": _choice_map(occurrences, common, {}),
                    "true_labels": None,
                }
            )
    if "entities" in schema:
        entity_fields = list((schema.get("entities") or {}).keys())
        if entity_fields:
            pending.append(
                {"task": "entities", "name": "entities", "fields": entity_fields, "choices": {}, "true_labels": None}
            )
    if "relations" in schema:
        grouped = {}
        for item in schema.get("relations") or []:
            for parent, fields in item.items():
                grouped.setdefault(parent, []).append(fields)
        for parent, occurrences in grouped.items():
            if not occurrences:
                continue
            field_names = list(occurrences[0].keys())
            if not any(all(field in occ for field in field_names) for occ in occurrences):
                continue
            pending.append(
                {"task": "relations", "name": parent, "fields": field_names, "choices": {}, "true_labels": None}
            )
    if "classifications" in schema:
        for item in schema.get("classifications") or []:
            pending.append(
                {
                    "task": "classifications",
                    "name": item["task"],
                    "fields": list(item["labels"]),
                    "choices": {},
                    "true_labels": None,
                }
            )
    if len(pending) != len(token_groups):
        raise RuntimeError("schema groups diverged from token groups")
    groups = []
    for meta, tokens, task in zip(pending, token_groups, task_types):
        if task != meta["task"]:
            raise RuntimeError("schema group order diverged from token groups")
        groups.append(
            _make_group(task, meta["name"], _fields_from_tokens(tokens), tokens, meta["choices"], meta["true_labels"])
        )
    return groups


def _sample_groups(schema: dict[str, Any], labels: dict[str, dict[str, Any]], rng: Any, config: SamplingConfig):
    """Apply SamplingConfig. RNG call order matches gliner2 training."""
    groups: list[SchemaGroup] = []
    record_meta = schema.get("record_metadata") or {}
    if "json_structures" in schema:
        json_descs = schema.get("json_descriptions") or {}
        grouped: dict[str, list] = {}
        for item in schema.get("json_structures") or []:
            for parent, fields in item.items():
                grouped.setdefault(parent, []).append(fields)
        for parent, occurrences in grouped.items():
            if rng.random() < config.remove_json_structure_prob:
                labels["json_structures"].pop(parent, None)
                continue
            is_record = bool((record_meta.get(parent) or {}).get("mode"))
            common: list[str] = []
            seen: set[str] = set()
            for occ in occurrences:
                for field_name in occ:
                    if field_name not in seen:
                        common.append(str(field_name))
                        seen.add(str(field_name))
            if config.shuffle_json_fields:
                rng.shuffle(common)
            if not is_record:
                chosen = [name for name in common if not (rng.random() < config.remove_json_field_prob)]
            else:
                chosen = list(common)
            if not chosen:
                labels["json_structures"].pop(parent, None)
                continue
            descs = dict(json_descs.get(parent) or {})
            example_modes = ["none", "descriptions"]
            real2syn: dict[str, str] = {}
            if (not is_record) and rng.random() < config.synthetic_entity_label_prob:
                example_modes.remove("none")
                synthetic = []
                for index, real in enumerate(chosen, 1):
                    syn = f"field {index}"
                    real2syn[real] = syn
                    synthetic.append(syn)
                descs = {real2syn.get(key, key): descs.get(key, key) for key in chosen}
                chosen = synthetic
                if parent in labels["json_structures"]:
                    labels["json_structures"][parent] = _rename_map(labels["json_structures"][parent], real2syn)
            kept = set(chosen)
            if parent in labels["json_structures"]:
                labels["json_structures"][parent] = [
                    {key: value for key, value in inst.items() if key in kept}
                    for inst in labels["json_structures"][parent]
                    if isinstance(inst, Mapping)
                ]
            mode = rng.choice(example_modes)
            tokens = _transform_schema(parent, chosen, C_TOKEN, label_descriptions=descs, example_mode=mode, rng=rng)
            groups.append(
                _make_group("json_structures", parent, chosen, tokens, _choice_map(occurrences, chosen, real2syn))
            )
    if "entities" in schema:
        if rng.random() < config.remove_entities_prob:
            schema["entities"] = {}
            labels["entities"] = {}
        else:
            entity_fields = list(schema["entities"].keys())
            descs = dict(schema.get("entity_descriptions") or {})
            example_modes = ["none", "descriptions"]
            if rng.random() < config.synthetic_entity_label_prob:
                example_modes.remove("none")
                real2syn = {}
                synthetic = []
                for index, real in enumerate(entity_fields, 1):
                    syn = f"entity {index}"
                    real2syn[real] = syn
                    synthetic.append(syn)
                descs = {real2syn.get(key, key): value for key, value in descs.items()}
                schema["entities"] = {real2syn.get(key, key): value for key, value in schema["entities"].items()}
                schema["entity_descriptions"] = descs
                labels["entities"] = {
                    real2syn[key]: value for key, value in labels["entities"].items() if key in real2syn
                }
                entity_fields = synthetic
            if config.shuffle_entities:
                rng.shuffle(entity_fields)
            chosen = [name for name in entity_fields if not (rng.random() < config.remove_entity_prob)]
            labels["entities"] = {key: value for key, value in labels["entities"].items() if key in chosen}
            if chosen:
                mode = rng.choice(example_modes)
                tokens = _transform_schema(
                    "entities", chosen, E_TOKEN, label_descriptions=descs, example_mode=mode, rng=rng
                )
                groups.append(_make_group("entities", "entities", chosen, tokens))
    if "relations" in schema:
        relation_descriptions = schema.get("relation_descriptions") or {}
        grouped = {}
        for item in list(schema.get("relations") or []):
            if rng.random() < config.remove_relations_prob:
                continue
            for parent, fields in item.items():
                grouped.setdefault(parent, []).append(dict(fields))
        kept_names = []
        for parent, occurrences in grouped.items():
            field_names = list(occurrences[0].keys())
            if "head" in field_names and "tail" in field_names and rng.random() < config.swap_head_tail_prob:
                head = field_names.index("head")
                tail = field_names.index("tail")
                field_names[head], field_names[tail] = field_names[tail], field_names[head]
            if not any(all(field in occ for field in field_names) for occ in occurrences):
                continue
            tokens = _transform_schema(parent, field_names, R_TOKEN, prompt=relation_descriptions.get(parent), rng=rng)
            groups.append(_make_group("relations", parent, field_names, tokens))
            kept_names.append(parent)
        labels["relations"] = {key: value for key, value in labels["relations"].items() if key in kept_names}
    if "classifications" in schema:
        rewritten = []
        for item in schema.get("classifications") or []:
            task = item["task"]
            if rng.random() < config.remove_classification_prob:
                labels["classifications"].pop(task, None)
                continue
            cls_labels = list(item["labels"])
            examples = list(item.get("examples") or [])
            descs = dict(item.get("label_descriptions") or {})
            real2syn = {}
            example_modes = ["few_shot", "descriptions", "both", "none"]
            if rng.random() < config.synthetic_label_prob:
                example_modes = [mode for mode in example_modes if mode != "none"]
                synthetic = []
                original = list(cls_labels)
                for index, real in enumerate(original, 1):
                    syn = f"label {index}"
                    real2syn[real] = syn
                    synthetic.append(syn)
                cls_labels = synthetic
                descs = {real2syn.get(key, key): descs.get(key, key) for key in original}
                examples = [(inp, real2syn.get(out, out)) for inp, out in examples]
            mode = rng.choice(example_modes) if example_modes else "none"
            drop_frac = rng.betavariate(1, 1) * config.remove_classification_label_prob
            num_remove = int(len(cls_labels) * drop_frac)
            if num_remove > 0:
                cls_labels = rng.sample(cls_labels, len(cls_labels) - num_remove)
            max_labels = (
                config.max_num_labels // 2 if mode in ("few_shot", "both", "descriptions") else config.max_num_labels
            )
            if len(cls_labels) > max_labels:
                cls_labels = cls_labels[:max_labels]
            if rng.random() < config.include_true_label_prob:
                for true_name in list(labels["classifications"].get(task, [])):
                    if true_name not in cls_labels:
                        cls_labels.append(true_name)
            if config.shuffle_classification_labels:
                rng.shuffle(cls_labels)
            tokens = _transform_schema(
                task,
                cls_labels,
                L_TOKEN,
                prompt=item.get("prompt"),
                examples=examples,
                label_descriptions=descs,
                example_mode=mode,
                rng=rng,
            )
            true = list(labels["classifications"].get(task, []))
            if real2syn:
                true = [real2syn.get(name, name) for name in true]
                labels["classifications"][task] = true
            groups.append(_make_group("classifications", task, cls_labels, tokens, true_labels=tuple(true)))
            updated = dict(item)
            updated["labels"] = cls_labels
            rewritten.append(updated)
        schema["classifications"] = rewritten
    order = list(range(len(groups)))
    rng.shuffle(order)
    return [groups[index] for index in order]


def _label_words(surface: str, splitter: Callable[..., Iterator[tuple[str, int, int]]]) -> list[str]:
    return [token for token, _, _ in splitter(surface, True)]


def _locate_mention(
    mention: Any,
    *,
    choices: Sequence[str] | None,
    doc_words: Sequence[str],
    prefix_len: int,
    prefix_tokens: Sequence[str],
    starts: Sequence[int],
    ends: Sequence[int],
    text: str,
    splitter: Callable[..., Iterator[tuple[str, int, int]]],
) -> list[tuple[int, int]]:
    """Inclusive word spans. Strings match every occurrence."""
    if isinstance(mention, Mapping):
        if not {"text", "start", "end"} <= set(mention):
            raise ValueError("span labels must be strings or {text, start, end}")
        surface = str(mention["text"])
        start = int(mention["start"])
        end = int(mention["end"])
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"span [{start}, {end}) for {surface!r} is outside the text")
        snippet = text[start:end]
        if snippet != surface and snippet.strip() != surface:
            raise ValueError(f"{surface!r} does not match text[{start}:{end}]")
        covered = [
            index
            for index, (word_start, word_end) in enumerate(zip(starts, ends))
            if word_start < end and start < word_end
        ]
        if not covered:
            raise ValueError(f"{surface!r} at [{start}, {end}) does not cover a word")
        pieces = _label_words(surface, splitter)
        got = list(doc_words[covered[0] : covered[-1] + 1])
        if got != pieces:
            raise ValueError(f"{surface!r} at [{start}, {end}) does not match the words {got}")
        return [(prefix_len + covered[0], prefix_len + covered[-1])]
    if not isinstance(mention, str):
        raise TypeError(f"span label must be a string or {{text, start, end}}, got {type(mention).__name__}")
    if choices is not None:
        if mention not in choices:
            raise ValueError(f"choice {mention!r} is not in {list(choices)}")
        found = [index for index, token in enumerate(prefix_tokens) if token.lower() == mention.lower()]
        if not found:
            raise ValueError(f"choice {mention!r} is not in the text")
        return [(index, index) for index in found]
    pieces = _label_words(mention, splitter)
    if not pieces:
        raise ValueError(f"label {mention!r} is not in the text")
    width = len(pieces)
    matches = [
        (prefix_len + index, prefix_len + index + width - 1)
        for index in range(len(doc_words) - width + 1)
        if list(doc_words[index : index + width]) == pieces
    ]
    if not matches:
        raise ValueError(f"label {mention!r} is not in the text")
    return matches


def _field_spans(raw: Any, **kwargs) -> list[tuple[int, int]]:
    mentions = raw if isinstance(raw, list) else [raw]
    spans: list[tuple[int, int]] = []
    seen = set()
    for mention in mentions:
        if mention in ("", None):
            continue
        for span in _locate_mention(mention, **kwargs):
            if span not in seen:
                seen.add(span)
                spans.append(span)
    return spans


def _bind_labels(
    groups: Sequence[SchemaGroup],
    labels: Mapping[str, Mapping[str, Any]],
    doc_words: Sequence[str],
    starts: Sequence[int],
    ends: Sequence[int],
    text: str,
    prefix_len: int,
    prefix_tokens: Sequence[str],
    splitter: Callable[..., Iterator[tuple[str, int, int]]],
) -> dict[str, Any]:
    """Map label strings onto inclusive word spans and half-open mentions."""
    structure_labels: list[Any] = []
    mentions: list[tuple[int, int, int]] = []
    relation_edges: list[list[tuple[int, int, int, int]]] = []
    query_id = 0
    locate_kwargs = {
        "doc_words": doc_words,
        "prefix_len": prefix_len,
        "prefix_tokens": prefix_tokens,
        "starts": starts,
        "ends": ends,
        "text": text,
        "splitter": splitter,
    }
    for group in groups:
        if group.task == "classifications":
            true = (
                set(group.true_labels)
                if group.true_labels is not None
                else set(labels["classifications"].get(group.name, []))
            )
            structure_labels.append([1 if field.name in true else 0 for field in group.fields])
            continue
        names = [field.name for field in group.fields]
        if group.task == "entities":
            instance = []
            for name in names:
                raw = labels["entities"].get(name, [])
                instance.append(_field_spans(raw, choices=None, **locate_kwargs))
            count = 1 if any(instance) else 0
            structure_labels.append([count, [instance] if count else []])
            for field_index, spans in enumerate(instance):
                for start, end in spans:
                    mentions.append((query_id + field_index, start, end + 1))
        else:
            bucket = labels["relations"] if group.task == "relations" else labels["json_structures"]
            built = []
            edges = []
            for raw_instance in bucket.get(group.name, []):
                if not isinstance(raw_instance, Mapping):
                    raise TypeError(f"{group.name} labels must be dicts of field values")
                unknown = set(raw_instance) - set(names)
                if unknown:
                    raise ValueError(f"unknown fields {sorted(unknown)} for {group.name}")
                instance = []
                for name in names:
                    choices = group.choices.get(name)
                    if name not in raw_instance:
                        if group.task == "relations":
                            raise ValueError(f"relation {group.name!r} is missing {name!r}")
                        instance.append([])
                        continue
                    instance.append(
                        _field_spans(raw_instance[name], choices=choices if choices else None, **locate_kwargs)
                    )
                if group.task == "relations":
                    # First two fields are the head and tail queries, in token order.
                    if len(instance) < 2 or not instance[0] or not instance[1]:
                        raise ValueError(f"relation {group.name!r} is not in the text")
                    for head_start, head_end in instance[0]:
                        for tail_start, tail_end in instance[1]:
                            edges.append((head_start, head_end + 1, tail_start, tail_end + 1))
                if any(instance):
                    built.append(instance)
            structure_labels.append([len(built), built])
            for instance in built:
                for field_index, spans in enumerate(instance):
                    for start, end in spans:
                        mentions.append((query_id + field_index, start, end + 1))
            if group.task == "relations":
                relation_edges.append(edges)
        query_id += len(group.fields)
    return {
        "structure_labels": structure_labels,
        "mentions": mentions,
        "query_count": query_id,
        "relation_edges": relation_edges,
    }


def _pack_targets(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad mention, classification, and relation targets."""
    batch = len(records)
    supervisions = [record["supervision"] for record in records]
    query_width = max((item["query_count"] for item in supervisions), default=0)
    gold_width = 1
    grouped: list[dict[int, list[tuple[int, int]]]] = []
    for item in supervisions:
        per_query: dict[int, list[tuple[int, int]]] = {}
        for query_id, start, end in item["mentions"]:
            pairs = per_query.setdefault(query_id, [])
            pair = (start, end)
            if pair not in pairs:
                pairs.append(pair)
        gold_width = max(gold_width, max((len(pairs) for pairs in per_query.values()), default=0))
        grouped.append(per_query)
    mention_pairs = torch.zeros((batch, query_width, gold_width, 2), dtype=torch.long)
    mention_mask = torch.zeros((batch, query_width, gold_width), dtype=torch.bool)
    for batch_index, per_query in enumerate(grouped):
        for query_id, pairs in per_query.items():
            count = len(pairs)
            if count:
                mention_pairs[batch_index, query_id, :count] = torch.tensor(pairs, dtype=torch.long)
                mention_mask[batch_index, query_id, :count] = True
    cls_rows = []
    for item in supervisions:
        row = []
        for labels in item["structure_labels"]:
            if labels and isinstance(labels[0], int) and not isinstance(labels[0], bool):
                if len(labels) == 2 and isinstance(labels[1], list):
                    continue
                row.extend(int(value) for value in labels)
        cls_rows.append(row)
    cls_width = max((len(row) for row in cls_rows), default=0)
    classification_targets = torch.zeros((batch, cls_width), dtype=torch.float)
    classification_target_mask = torch.zeros((batch, cls_width), dtype=torch.bool)
    for index, row in enumerate(cls_rows):
        if row:
            classification_targets[index, : len(row)] = torch.tensor(row, dtype=torch.float)
            classification_target_mask[index, : len(row)] = True
    relation_rows = [item["relation_edges"] for item in supervisions]
    relation_width = max((len(rows) for rows in relation_rows), default=0)
    edge_width = max((len(edges) for rows in relation_rows for edges in rows), default=0)
    relation_edges = torch.zeros((batch, relation_width, max(edge_width, 1), 4), dtype=torch.long)
    relation_edge_mask = torch.zeros((batch, relation_width, max(edge_width, 1)), dtype=torch.bool)
    for batch_index, rows in enumerate(relation_rows):
        for relation_index, edges in enumerate(rows):
            if not edges:
                continue
            relation_edges[batch_index, relation_index, : len(edges)] = torch.tensor(edges, dtype=torch.long)
            relation_edge_mask[batch_index, relation_index, : len(edges)] = True
    record_targets = []
    for record, item in zip(records, supervisions):
        sample = []
        for spec in record.get("record_specs") or []:
            task_index = spec["task_index"]
            structure = item["structure_labels"][task_index]
            sample.append(
                {
                    "task_index": task_index,
                    "task_name": spec["task_name"],
                    "count": structure[0],
                    "instances": structure[1],
                }
            )
        record_targets.append(sample)
    return {
        "structure_labels": [item["structure_labels"] for item in supervisions],
        "mention_pairs": mention_pairs,
        "mention_mask": mention_mask,
        "classification_targets": classification_targets,
        "classification_target_mask": classification_target_mask,
        "relation_edges": relation_edges,
        "relation_edge_mask": relation_edge_mask,
        "records": record_targets,
    }


@auto_docstring
class Gliner2Processor(ProcessorMixin):
    """Schema-conditioned processor for GLiNER2.

    Tokenization matches inference collation: lowercase whitespace words, each
    word tokenized alone, structural markers kept as single special tokens, no
    CLS or SEP wrapping, and first-subword indexes for token pooling.
    """

    model_input_names = [
        "input_ids",
        "attention_mask",
        "text_word_indices",
        "text_word_mask",
        "query_marker_indices",
        "query_marker_mask",
        "query_group_index",
        "cls_marker_indices",
        "cls_marker_mask",
        "cls_group_index",
        "prompt_marker_indices",
        "prompt_marker_mask",
        "prompt_group_index",
        "task_type_ids",
        "group_mask",
    ]

    def __init__(self, tokenizer=None, **kwargs):
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        token_pooling = kwargs.pop("token_pooling", "first")
        word_splitter = kwargs.pop("word_splitter", None)
        sampling_config = kwargs.pop("sampling_config", None)
        super().__init__(tokenizer, **kwargs)
        if token_pooling not in ("first", "mean", "max"):
            token_pooling = "first"
        self.token_pooling = token_pooling
        self.word_splitter = _resolve_word_splitter(word_splitter)
        self.sampling_config = sampling_config if sampling_config is not None else SamplingConfig()
        self._piece_ids: dict[str, tuple[int, ...]] = {}
        self._register_special_tokens()

    def _register_special_tokens(self) -> None:
        self.tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
        for token in list(SPECIAL_TOKENS) + ["(", ")", ",", "|"]:
            self._encode_piece(token)

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        """Cache one piece's ids. Pieces are encoded, then concatenated."""
        cached = self._piece_ids.get(piece)
        if cached is None:
            cached = tuple(int(item) for item in self.tokenizer.encode(piece, add_special_tokens=False))
            self._piece_ids[piece] = cached
        return cached

    def _split_words(self, text: str) -> tuple[list[str], list[int], list[int]]:
        words, starts, ends = [], [], []
        for token, start, end in self.word_splitter(text, True):
            words.append(token)
            starts.append(start)
            ends.append(end)
        return words, starts, ends

    def _format_input(self, schema_tokens_list: Sequence[Sequence[str]], text_tokens: Sequence[str]) -> dict[str, Any]:
        """Join schema groups and text, then map words onto first subwords."""
        combined: list[str] = []
        for struct in schema_tokens_list:
            combined.extend(struct)
            combined.append(SEP_STRUCT)
        if combined:
            combined.pop()
        combined.append(SEP_TEXT)
        combined.extend(text_tokens)
        # Offset counts a [SEP_STRUCT] after every schema, including the last.
        schema_marker_orig_indices = set()
        offset = 0
        for struct in schema_tokens_list:
            if len(struct) > 1:
                schema_marker_orig_indices.add(offset + 1)
            schema_marker_orig_indices.update(offset + index for index in range(4, len(struct) - 2, 2))
            offset += len(struct) + 1
        input_ids: list[int] = []
        text_word_first_positions: list[int] = []
        schema_special_positions: list[list[int]] = [[] for _ in schema_tokens_list]
        num_schemas = len(schema_tokens_list)
        current_schema = 0
        found_sep = False
        last_text_orig = None
        for orig_idx, token in enumerate(combined):
            if token == SEP_TEXT:
                seg_type = "sep"
                found_sep = True
                schema_idx = num_schemas
            elif not found_sep:
                seg_type = "schema"
                schema_idx = current_schema
                if token == SEP_STRUCT:
                    current_schema += 1
            else:
                seg_type = "text"
                schema_idx = num_schemas
            subword_pos = len(input_ids)
            piece_ids = self._encode_piece(token)
            input_ids.extend(piece_ids)
            if seg_type == "text" and orig_idx != last_text_orig:
                last_text_orig = orig_idx
                if not piece_ids:
                    logger.warning(
                        "text word %r (index %d) produced no subwords; inserting a placeholder",
                        token,
                        orig_idx,
                    )
                text_word_first_positions.append(subword_pos)
            elif seg_type == "schema" and orig_idx in schema_marker_orig_indices:
                schema_special_positions[schema_idx].append(subword_pos)
        return {
            "input_ids": input_ids,
            "text_word_first_positions": text_word_first_positions,
            "schema_special_positions": schema_special_positions,
        }

    def _transform_one(
        self,
        text: str,
        schema: Any,
        max_len: int | None,
        architecture: str,
        labels: Any = None,
        rng: Any = None,
        sampling_config: SamplingConfig | None = None,
    ) -> dict[str, Any]:
        source = schema
        schema = _canonicalize_schema(schema)
        label_spec = None if labels is None else _canonicalize_labels(labels, schema)
        config = sampling_config or self.sampling_config
        if rng is None:
            prefix = _classification_prefix(schema)
            schema_tokens, task_types = _schema_token_groups(schema)
            compiled = _groups_for_tokens(schema, schema_tokens, task_types)
        else:
            prefix = _classification_prefix(schema, rng)
            compiled = _sample_groups(schema, label_spec or _empty_labels(), rng, config)
            schema_tokens = [group.tokens for group in compiled]
            task_types = [group.task for group in compiled]
        text = _normalize_text(text)
        words, starts, ends = self._split_words(text)
        if max_len is not None:
            words, starts, ends = words[:max_len], starts[:max_len], ends[:max_len]
        text_tokens = list(prefix) + words
        formatted = self._format_input(schema_tokens, text_tokens)
        groups = []
        for group in compiled:
            groups.append(
                {
                    "task_type": group.task,
                    "name": _group_name(group.tokens, group.task),
                    "prompt": group.tokens[2] if len(group.tokens) > 2 else "",
                    "fields": [field.name for field in group.fields],
                }
            )
        meta = _public_metadata(source, schema)
        if rng is not None:
            for group in compiled:
                if group.task == "entities":
                    meta["entity_order"] = [field.name for field in group.fields]
        supervision = None
        if label_spec is not None:
            supervision = _bind_labels(
                compiled,
                label_spec,
                words,
                starts,
                ends,
                text,
                len(prefix),
                list(prefix),
                self.word_splitter,
            )
        meta.update(
            {
                "groups": groups,
                "text": text,
                "start": starts,
                "end": ends,
                "prefix_len": len(prefix),
                "task_types": task_types,
                "architecture": architecture,
                "classifications": list(schema.get("classifications", []) or []),
                "cls_fields": _choice_fields(schema),
                "schema": schema,
                "constraints": list(schema.get("constraints", []) or []),
                "record_specs": _compile_record_specs(groups, schema.get("record_metadata"), _field_dtypes(schema)),
            }
        )
        return {
            "input_ids": formatted["input_ids"],
            "text_word_first_positions": formatted["text_word_first_positions"],
            "schema_special_positions": formatted["schema_special_positions"],
            "task_types": list(task_types),
            "words": text_tokens,
            "schema_meta": meta,
            "record_specs": meta["record_specs"],
            "supervision": supervision,
        }

    def _relation_pairs(self, records: Sequence[Mapping[str, Any]]) -> list[list[tuple[int, int, int]]]:
        pairs = []
        for record in records:
            sample = []
            query_cursor = 0
            for group_index, positions in enumerate(record["schema_special_positions"]):
                field_count = max(len(positions) - 1, 0)
                task = record["task_types"][group_index]
                if task == "relations" and field_count >= 2:
                    sample.append((query_cursor, query_cursor + 1, group_index))
                if task != "classifications":
                    query_cursor += field_count
            pairs.append(sample)
        return pairs

    def _pack_routes(self, routes: Sequence[Sequence[tuple[int, int]]], batch_size: int):
        width = max((len(values) for values in routes), default=0)
        indices = torch.zeros((batch_size, width), dtype=torch.long)
        mask = torch.zeros((batch_size, width), dtype=torch.bool)
        groups = torch.zeros((batch_size, width), dtype=torch.long)
        for index, values in enumerate(routes):
            if not values:
                continue
            positions, group_ids = zip(*values)
            count = len(values)
            indices[index, :count] = torch.tensor(positions, dtype=torch.long)
            groups[index, :count] = torch.tensor(group_ids, dtype=torch.long)
            mask[index, :count] = True
        return indices, mask, groups

    def _pack_record_tensors(self, record_specs: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, torch.Tensor]:
        batch_size = len(record_specs)
        max_records = max((len(group) for group in record_specs), default=0)
        max_fields = max((len(spec["fields"]) for group in record_specs for spec in group), default=0)
        record_group_index = torch.zeros((batch_size, max_records), dtype=torch.long)
        record_mode_ids = torch.zeros((batch_size, max_records), dtype=torch.long)
        record_anchor_query = torch.full((batch_size, max_records), -1, dtype=torch.long)
        record_field_query = torch.zeros((batch_size, max_records, max_fields), dtype=torch.long)
        record_field_cardinality = torch.zeros((batch_size, max_records, max_fields), dtype=torch.long)
        record_field_anchor = torch.zeros((batch_size, max_records, max_fields), dtype=torch.bool)
        record_field_mask = torch.zeros((batch_size, max_records, max_fields), dtype=torch.bool)
        record_mask = torch.zeros((batch_size, max_records), dtype=torch.bool)
        for batch_index, group in enumerate(record_specs):
            for record_index, spec in enumerate(group):
                record_mask[batch_index, record_index] = True
                record_group_index[batch_index, record_index] = spec["task_index"]
                record_mode_ids[batch_index, record_index] = RECORD_MODE_TO_ID[spec["mode"]]
                if spec["anchor_query_id"] is not None:
                    record_anchor_query[batch_index, record_index] = spec["anchor_query_id"]
                for field_index, field_spec in enumerate(spec["fields"]):
                    record_field_mask[batch_index, record_index, field_index] = True
                    record_field_query[batch_index, record_index, field_index] = field_spec["query_id"]
                    record_field_cardinality[batch_index, record_index, field_index] = CARDINALITY_TO_ID[
                        field_spec["cardinality"]
                    ]
                    record_field_anchor[batch_index, record_index, field_index] = field_spec["is_anchor"]
        return {
            "record_group_index": record_group_index,
            "record_mode_ids": record_mode_ids,
            "record_anchor_query": record_anchor_query,
            "record_field_query": record_field_query,
            "record_field_cardinality": record_field_cardinality,
            "record_field_anchor": record_field_anchor,
            "record_field_mask": record_field_mask,
            "record_mask": record_mask,
        }

    def _collate(self, records: Sequence[Mapping[str, Any]], architecture: str) -> dict[str, Any]:
        batch_size = len(records)
        max_len = max((len(record["input_ids"]) for record in records), default=0)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for index, record in enumerate(records):
            length = len(record["input_ids"])
            if length:
                input_ids[index, :length] = torch.tensor(record["input_ids"], dtype=torch.long)
                attention_mask[index, :length] = 1
        word_counts = [len(record["text_word_first_positions"]) for record in records]
        max_words = max(word_counts) if word_counts else 0
        text_word_indices = torch.zeros((batch_size, max_words), dtype=torch.long)
        text_word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
        for index, record in enumerate(records):
            count = word_counts[index]
            if count:
                text_word_indices[index, :count] = torch.tensor(record["text_word_first_positions"], dtype=torch.long)
                text_word_mask[index, :count] = True
        query_routes = []
        cls_routes = []
        for record in records:
            query_row = []
            cls_row = []
            for group_index, positions in enumerate(record["schema_special_positions"]):
                routes = [(position, group_index) for position in positions[1:]]
                if record["task_types"][group_index] == "classifications":
                    cls_row.extend(routes)
                else:
                    query_row.extend(routes)
            query_routes.append(query_row)
            cls_routes.append(cls_row)
        prompt_routes = []
        for record in records:
            prompt_row = []
            for group_index, positions in enumerate(record["schema_special_positions"]):
                if record["task_types"][group_index] == "classifications" or not positions:
                    continue
                prompt_row.append((positions[0], group_index))
            prompt_routes.append(prompt_row)
        query_marker_indices, query_marker_mask, query_group_index = self._pack_routes(query_routes, batch_size)
        cls_marker_indices, cls_marker_mask, cls_group_index = self._pack_routes(cls_routes, batch_size)
        prompt_marker_indices, prompt_marker_mask, prompt_group_index = self._pack_routes(prompt_routes, batch_size)
        group_width = max((len(record["task_types"]) for record in records), default=0)
        task_type_ids = torch.zeros((batch_size, group_width), dtype=torch.long)
        group_mask = torch.zeros((batch_size, group_width), dtype=torch.bool)
        for index, record in enumerate(records):
            count = len(record["task_types"])
            if count:
                task_type_ids[index, :count] = torch.tensor(
                    [_task_type_id(task) for task in record["task_types"]], dtype=torch.long
                )
                group_mask[index, :count] = True
        relation_pairs = self._relation_pairs(records)
        relation_width = max((len(sample) for sample in relation_pairs), default=0)
        relation_head_index = torch.zeros((batch_size, relation_width), dtype=torch.long)
        relation_tail_index = torch.zeros((batch_size, relation_width), dtype=torch.long)
        relation_group_index = torch.zeros((batch_size, relation_width), dtype=torch.long)
        relation_mask = torch.zeros((batch_size, relation_width), dtype=torch.bool)
        for index, sample in enumerate(relation_pairs):
            if not sample:
                continue
            count = len(sample)
            relation_head_index[index, :count] = torch.tensor([item[0] for item in sample], dtype=torch.long)
            relation_tail_index[index, :count] = torch.tensor([item[1] for item in sample], dtype=torch.long)
            relation_group_index[index, :count] = torch.tensor([item[2] for item in sample], dtype=torch.long)
            relation_mask[index, :count] = True
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_word_indices": text_word_indices,
            "text_word_mask": text_word_mask,
            "query_marker_indices": query_marker_indices,
            "query_marker_mask": query_marker_mask,
            "query_group_index": query_group_index,
            "cls_marker_indices": cls_marker_indices,
            "cls_marker_mask": cls_marker_mask,
            "cls_group_index": cls_group_index,
            "prompt_marker_indices": prompt_marker_indices,
            "prompt_marker_mask": prompt_marker_mask,
            "prompt_group_index": prompt_group_index,
            "task_type_ids": task_type_ids,
            "group_mask": group_mask,
            "relation_head_index": relation_head_index,
            "relation_tail_index": relation_tail_index,
            "relation_group_index": relation_group_index,
            "relation_mask": relation_mask,
            "metadata": [{"words": record["words"], "schema_meta": record["schema_meta"]} for record in records],
        }
        if architecture == "boundary":
            data.update(self._pack_record_tensors([record["record_specs"] for record in records]))
        if any(record.get("supervision") is not None for record in records):
            if any(record.get("supervision") is None for record in records):
                raise ValueError("labels must be provided for every text in the batch")
            data["targets"] = _pack_targets(records)
        return data

    @auto_docstring
    def __call__(
        self,
        text: TextInput,
        schema: Any = None,
        return_tensors: str = "pt",
        max_len: int | None = None,
        architecture: str = "span",
        labels: Any = None,
        rng: Any = None,
        sampling_config: SamplingConfig | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Build model tensors from text plus schema.

        Args:
            text (`str` or `list[str]`):
                Input text.
            schema (`dict`):
                One schema covering entities, classifications, relations, and
                structures.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                Tensor format. Only `"pt"` is supported.
            max_len (`int`, *optional*):
                Maximum tokenized length.
            architecture (`str`, *optional*, defaults to `"span"`):
                `"span"` or `"boundary"`.
            labels (`dict`, *optional*):
                Supervision grouped by task. Strings match every occurrence.
                `{text, start, end}` selects one occurrence.
            rng (`random.Random`, *optional*):
                Draw source for `sampling_config`.
            sampling_config (`SamplingConfig`, *optional*):
                Task, field, and label sampling.
        """
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        if return_tensors != "pt":
            raise ValueError("Gliner2Processor only returns PyTorch tensors")
        if architecture not in ("span", "boundary"):
            raise ValueError("architecture must be 'span' or 'boundary'")
        if schema is None:
            raise ValueError("schema is required")
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        if not texts:
            raise ValueError("text must contain at least one string")
        if isinstance(schema, (list, tuple)):
            if len(schema) != len(texts):
                raise ValueError(f"Schema count ({len(schema)}) != text count ({len(texts)})")
            schemas = list(schema)
        else:
            schemas = [schema] * len(texts)
        if labels is None:
            label_rows: list[Any] = [None] * len(texts)
        elif isinstance(labels, (list, tuple)):
            if len(labels) != len(texts):
                raise ValueError(f"Label count ({len(labels)}) != text count ({len(texts)})")
            label_rows = list(labels)
        elif len(texts) != 1:
            raise ValueError("labels must be a list when text is a batch")
        else:
            label_rows = [labels]
        records = [
            self._transform_one(
                item,
                item_schema,
                max_len,
                architecture,
                label_rows[index],
                rng,
                sampling_config,
            )
            for index, (item, item_schema) in enumerate(zip(texts, schemas))
        ]
        skipped = ["metadata"]
        if any(record.get("supervision") is not None for record in records):
            skipped.append("targets")
        return BatchFeature(self._collate(records, architecture), tensor_type="pt", skip_tensor_conversion=skipped)

    def chunk_words(
        self,
        words: str | Sequence[Any],
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> list[TextChunk]:
        """Split text or words into overlapping windows."""
        return chunk_words(words, chunk_size=chunk_size, chunk_overlap=chunk_overlap, word_splitter=self.word_splitter)

    def merge_chunk_results(
        self,
        original_text: str,
        chunks: Sequence[Any],
        chunk_results: Sequence[dict[str, Any]],
        include_confidence: bool = False,
        include_spans: bool = False,
        scalar_entity_labels: Iterable[str] | None = None,
        overlap_policy: str | None = None,
    ) -> dict[str, Any]:
        """Merge chunk payloads onto document offsets."""
        return merge_chunk_results(
            original_text,
            chunks,
            chunk_results,
            include_confidence=include_confidence,
            include_spans=include_spans,
            scalar_entity_labels=scalar_entity_labels,
            overlap_policy=overlap_policy,
        )

    def _unpack_metadata(self, metadata: Any) -> list[dict[str, Any]]:
        if isinstance(metadata, Mapping):
            metadata = [metadata]
        rows = []
        for item in metadata:
            if "schema_meta" in item:
                meta = dict(item["schema_meta"])
                meta["words"] = list(item.get("words", meta.get("words", [])))
            else:
                meta = dict(item)
                meta["words"] = list(meta.get("words", []))
            rows.append(meta)
        return rows

    def _sample_outputs(self, outputs: Any, batch_size: int) -> list[Mapping[str, Any]]:
        if isinstance(outputs, (list, tuple)):
            if len(outputs) != batch_size:
                raise ValueError(f"outputs length ({len(outputs)}) != metadata length ({batch_size})")
            return list(outputs)
        boundary = getattr(outputs, "boundary", None)
        if not isinstance(outputs, Mapping):
            outputs = dict(outputs.items()) if hasattr(outputs, "items") else dict(outputs)
        if boundary is not None and getattr(boundary, "candidates", None) is not None:
            samples = []
            candidates = boundary.candidates
            for index in range(batch_size):
                sample = {
                    "candidates": type(candidates)(
                        indices=candidates.indices[index : index + 1],
                        proposal_logits=(
                            None
                            if candidates.proposal_logits is None
                            else candidates.proposal_logits[index : index + 1]
                        ),
                        pair_logits=candidates.pair_logits[index : index + 1],
                        valid_mask=candidates.valid_mask[index : index + 1],
                        query_mask=candidates.query_mask[index : index + 1],
                        candidate_states=(
                            None
                            if candidates.candidate_states is None
                            else candidates.candidate_states[index : index + 1]
                        ),
                    ),
                    "pair_logits": candidates.pair_logits[index],
                    "null_logits": None if boundary.null_logits is None else boundary.null_logits[index],
                    "count_log_rates": (None if boundary.count_log_rates is None else boundary.count_log_rates[index]),
                }
                if outputs.get("classification_logits") is not None:
                    sample["classification_logits"] = outputs["classification_logits"][index]
                samples.append(sample)
            return samples
        keys = (
            "span_scores",
            "span_logits",
            "counts",
            "classification_logits",
            "classification_probs",
            "raw_logits",
            "record_logits",
            "pair_logits",
            "grouped_candidates",
        )
        present = [key for key in keys if key in outputs and outputs[key] is not None]
        if not present:
            raise ValueError(
                "outputs need span_scores or span_logits (count, fields, words, width) "
                "and/or classification_logits (num_labels,). "
                "words includes the classification prefix. "
                "Boundary pair_logits require decoding_gliner2.decode_boundary."
            )
        samples = []
        for index in range(batch_size):
            samples.append({key: outputs[key][index] for key in present})
        return samples

    def _span_group_tensors(self, value: Any, num_groups: int, apply_sigmoid: bool) -> list[torch.Tensor]:
        if torch.is_tensor(value):
            tensors = [value]
        else:
            tensors = list(value)
        if len(tensors) != num_groups:
            raise ValueError(f"expected {num_groups} span tensors, got {len(tensors)}")
        prepared = []
        for tensor in tensors:
            if not torch.is_tensor(tensor):
                tensor = torch.tensor(tensor, dtype=torch.float)
            tensor = tensor.detach().float().cpu()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                raise ValueError("span scores must have shape (count, fields, words, width)")
            prepared.append(torch.sigmoid(tensor) if apply_sigmoid else tensor)
        return prepared

    def _classification_vectors(self, value: Any, num_groups: int) -> list[torch.Tensor]:
        if torch.is_tensor(value):
            vectors = [value]
        else:
            vectors = list(value)
        if len(vectors) != num_groups:
            raise ValueError(f"expected {num_groups} classification vectors, got {len(vectors)}")
        prepared = []
        for vector in vectors:
            if not torch.is_tensor(vector):
                vector = torch.tensor(vector, dtype=torch.float)
            prepared.append(vector.detach().float().cpu().reshape(-1))
        return prepared

    def _decode_entities(
        self,
        field_names: Sequence[str],
        scores: torch.Tensor,
        raw_logits: torch.Tensor | None,
        meta: Mapping[str, Any],
        text: str,
        start_map: Sequence[int],
        end_map: Sequence[int],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
        overlap_policy: str | None,
    ) -> list[dict[str, Any]]:
        doc_len = len(start_map)
        entity_results: dict[str, Any] = OrderedDict()
        attribute_labels = set(meta.get("entity_attribute_labels") or ())
        groups = meta.get("entity_attribute_groups") or {}
        use_attributes = bool(groups) and raw_logits is not None and attribute_labels
        if use_attributes:
            prompt_labels = meta.get("entity_attribute_prompt_labels") or {}
            group_indices = {}
            doc_logits = _doc_axis(raw_logits[0], doc_len)
            for group_name, group in groups.items():
                labels = list(_group_attr(group, "labels", []))
                present = [
                    (label, field_names.index(prompt_labels.get(label, label)))
                    for label in labels
                    if prompt_labels.get(label, label) in field_names
                ]
                if present:
                    chosen, indices = zip(*present)
                    group_indices[group_name] = (list(chosen), torch.tensor(indices, dtype=torch.long), group)
            content_names = [name for name in field_names if name not in attribute_labels]
            order = [name for name in meta.get("entity_order", content_names) if name in content_names]
            for name in order:
                index = field_names.index(name)
                entity_meta = (meta.get("entity_metadata") or {}).get(name, {})
                dtype = entity_meta.get("dtype", "list")
                ent_threshold = entity_meta.get("threshold")
                ent_threshold = float(ent_threshold) if ent_threshold is not None else threshold
                entity_scores = scores[index]
                found = []
                starts, widths = torch.where(entity_scores >= ent_threshold)
                for start, width in zip(starts.tolist(), widths.tolist()):
                    end = start + width + 1
                    if not (0 <= start < doc_len and end <= doc_len):
                        continue
                    try:
                        char_start, char_end = start_map[start], end_map[end - 1]
                        span_text = text[char_start:char_end].strip()
                    except (IndexError, KeyError):
                        continue
                    if not span_text or not _passes_validators(span_text, entity_meta.get("validators", [])):
                        continue
                    found.append(
                        {
                            "text": span_text,
                            "confidence": float(entity_scores[start, width].item()),
                            "start": int(char_start),
                            "end": int(char_end),
                            **_assign_entity_attributes(doc_logits, start, width, group_indices, name),
                        }
                    )
                surviving = _finalize_spans(
                    [(item["text"], item["confidence"], item["start"], item["end"]) for item in found],
                    dtype=dtype,
                    overlap_policy=overlap_policy,
                )
                found_by_span = {(item["start"], item["end"]): item for item in found}
                formatted = []
                for _, _, char_start, char_end in surviving:
                    entity = found_by_span[(char_start, char_end)]
                    result = {"text": entity["text"]}
                    if include_confidence:
                        result["confidence"] = entity["confidence"]
                    if include_spans:
                        result["start"] = entity["start"]
                        result["end"] = entity["end"]
                    result.update((key, value) for key, value in entity.items() if key not in _SPAN_RESERVED)
                    formatted.append(result)
                entity_results[name] = formatted if dtype == "list" else (formatted[0] if formatted else None)
            return [entity_results] if entity_results else []
        order = [name for name in meta.get("entity_order", field_names) if name in field_names]
        for name in order:
            index = list(field_names).index(name)
            entity_meta = (meta.get("entity_metadata") or {}).get(name, {})
            dtype = entity_meta.get("dtype", "list")
            ent_threshold = entity_meta.get("threshold")
            ent_threshold = float(ent_threshold) if ent_threshold is not None else threshold
            spans = _find_spans(scores[index], ent_threshold, text, start_map, end_map)
            spans = [span for span in spans if _passes_validators(span[0], entity_meta.get("validators", []))]
            spans = _finalize_spans(spans, dtype=dtype, overlap_policy=overlap_policy)
            if dtype == "list":
                entity_results[name] = _format_spans(spans, include_confidence, include_spans)
            elif spans:
                entity_results[name] = _format_one_span(spans[0], include_confidence, include_spans)
            else:
                entity_results[name] = "" if not include_spans and not include_confidence else None
        return [entity_results] if entity_results else []

    def _decode_relations(
        self,
        rel_name: str,
        field_names: Sequence[str],
        span_scores: torch.Tensor,
        count: int,
        meta: Mapping[str, Any],
        text: str,
        start_map: Sequence[int],
        end_map: Sequence[int],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
        overlap_policy: str | None,
        doc_len: int,
    ) -> list[Any]:
        rel_threshold = threshold
        configured = (meta.get("relation_metadata") or {}).get(rel_name, {}).get("threshold")
        if configured is not None:
            rel_threshold = configured
        ordered = (meta.get("field_orders") or {}).get(rel_name, field_names)
        instances = []
        for inst in range(count):
            values = []
            field_data = []
            for fname in ordered:
                if fname not in field_names:
                    continue
                fidx = list(field_names).index(fname)
                spans = _find_spans(
                    _doc_axis(span_scores[inst, fidx], doc_len), rel_threshold, text, start_map, end_map
                )
                spans = _finalize_spans(spans, overlap_policy=overlap_policy)
                if spans:
                    text_val, confidence, char_start, char_end = spans[0]
                    values.append(text_val)
                    field_data.append(
                        {"text": text_val, "confidence": confidence, "start": char_start, "end": char_end}
                    )
                else:
                    values.append(None)
                    field_data.append(None)
            if len(values) == 2 and values[0] and values[1]:
                if include_spans and include_confidence:
                    instances.append({"head": field_data[0], "tail": field_data[1]})
                elif include_spans:
                    instances.append(
                        {
                            "head": {
                                "text": field_data[0]["text"],
                                "start": field_data[0]["start"],
                                "end": field_data[0]["end"],
                            },
                            "tail": {
                                "text": field_data[1]["text"],
                                "start": field_data[1]["start"],
                                "end": field_data[1]["end"],
                            },
                        }
                    )
                elif include_confidence:
                    instances.append(
                        {
                            "head": {"text": field_data[0]["text"], "confidence": field_data[0]["confidence"]},
                            "tail": {"text": field_data[1]["text"], "confidence": field_data[1]["confidence"]},
                        }
                    )
                else:
                    instances.append((values[0], values[1]))
        return instances

    def _decode_structures(
        self,
        struct_name: str,
        field_names: Sequence[str],
        span_scores: torch.Tensor,
        count: int,
        meta: Mapping[str, Any],
        words: Sequence[str],
        text: str,
        start_map: Sequence[int],
        end_map: Sequence[int],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
        overlap_policy: str | None,
        doc_len: int,
        prefix_len: int,
    ) -> list[dict[str, Any]]:
        instances = []
        ordered = (meta.get("field_orders") or {}).get(struct_name, field_names)
        cls_fields = meta.get("cls_fields") or {}
        prefix_tokens = list(words[:prefix_len])
        for inst in range(count):
            instance: dict[str, Any] = OrderedDict()
            for fname in ordered:
                if fname not in field_names:
                    continue
                fidx = list(field_names).index(fname)
                field_key = f"{struct_name}.{fname}"
                field_meta = (meta.get("field_metadata") or {}).get(field_key, {})
                field_threshold = field_meta.get("threshold")
                field_threshold = field_threshold if field_threshold is not None else threshold
                dtype = field_meta.get("dtype", "list")
                validators = field_meta.get("validators", [])
                if field_key in cls_fields:
                    choices = cls_fields[field_key]
                    prefix_scores = span_scores[inst, fidx, :prefix_len]
                    if dtype == "list":
                        selected = []
                        seen = set()
                        for choice in choices:
                            if choice in seen:
                                continue
                            idx = _find_choice_idx(choice, prefix_tokens)
                            if idx >= 0 and idx < prefix_scores.shape[0]:
                                score = float(prefix_scores[idx, 0].item())
                                if score >= field_threshold:
                                    selected.append(
                                        {"text": choice, "confidence": score} if include_confidence else choice
                                    )
                                    seen.add(choice)
                        instance[fname] = selected
                    else:
                        best = None
                        best_score = -1.0
                        for choice in choices:
                            idx = _find_choice_idx(choice, prefix_tokens)
                            if idx >= 0 and idx < prefix_scores.shape[0]:
                                score = float(prefix_scores[idx, 0].item())
                                if score > best_score:
                                    best_score = score
                                    best = choice
                        if best and best_score >= field_threshold:
                            instance[fname] = {"text": best, "confidence": best_score} if include_confidence else best
                        else:
                            instance[fname] = None
                else:
                    spans = _find_spans(
                        _doc_axis(span_scores[inst, fidx], doc_len), field_threshold, text, start_map, end_map
                    )
                    spans = [span for span in spans if _passes_validators(span[0], validators)]
                    spans = _finalize_spans(spans, dtype=dtype, overlap_policy=overlap_policy)
                    if dtype == "list":
                        instance[fname] = _format_spans(spans, include_confidence, include_spans)
                    elif spans:
                        instance[fname] = _format_one_span(spans[0], include_confidence, include_spans)
                    else:
                        instance[fname] = None
            if any(value is not None and value != [] for value in instance.values()):
                instances.append(instance)
        return instances

    def _decode_sample(
        self,
        sample_out: Mapping[str, Any],
        meta: Mapping[str, Any],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
        overlap_policy: str | None,
        temperature: float,
    ) -> dict[str, Any]:
        if sample_out.get("pair_logits") is not None or sample_out.get("grouped_candidates") is not None:
            decoded = _call_decoder(
                "decode_boundary",
                sample_out,
                meta,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
                overlap_policy=overlap_policy,
                temperature=temperature,
            )
            cls_source = sample_out.get("classification_logits")
            cls_groups = [group for group in (meta.get("groups") or []) if group["task_type"] == "classifications"]
            if cls_source is not None and cls_groups:
                vectors = self._classification_vectors(cls_source, len(cls_groups))
                classifications = list(meta.get("classifications") or [])
                for group, vector in zip(cls_groups, vectors):
                    config = _resolve_classification_config(group["prompt"], classifications)
                    if config is None:
                        continue
                    decoded[config["task"]] = _decode_classification_group(
                        vector, config, threshold, temperature, activated=False
                    )
            return decoded
        if sample_out.get("record_logits") is not None:
            return _call_decoder(
                "decode_records",
                sample_out,
                meta,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
        groups = list(meta.get("groups") or [])
        span_groups = [group for group in groups if group["task_type"] != "classifications"]
        cls_groups = [group for group in groups if group["task_type"] == "classifications"]
        span_scores = None
        raw_logits = None
        if sample_out.get("span_scores") is not None:
            span_scores = self._span_group_tensors(sample_out["span_scores"], len(span_groups), apply_sigmoid=False)
        elif sample_out.get("span_logits") is not None:
            span_scores = self._span_group_tensors(sample_out["span_logits"], len(span_groups), apply_sigmoid=True)
        if sample_out.get("raw_logits") is not None:
            raw_logits = self._span_group_tensors(sample_out["raw_logits"], len(span_groups), apply_sigmoid=False)
        elif sample_out.get("span_logits") is not None:
            raw_logits = self._span_group_tensors(sample_out["span_logits"], len(span_groups), apply_sigmoid=False)
        counts = sample_out.get("counts")
        if counts is not None and not isinstance(counts, (list, tuple)):
            counts = counts.detach().cpu().tolist()
        results: dict[str, Any] = {}
        words = list(meta.get("words") or [])
        text = meta.get("text") or ""
        start_map = list(meta.get("start") or [])
        end_map = list(meta.get("end") or [])
        doc_len = len(start_map)
        prefix_len = int(meta.get("prefix_len") or 0)
        if span_scores is not None:
            for index, group in enumerate(span_groups):
                scores = span_scores[index]
                count = int(counts[index]) if counts is not None else int(scores.shape[0])
                count = max(count, 0)
                scores = scores[:count]
                if count <= 0:
                    if group["name"] == "entities" or group["task_type"] == "relations":
                        results[group["name"]] = []
                    else:
                        results[group["name"]] = {}
                    continue
                expected_words = len(words)
                if scores.shape[-2] != expected_words:
                    raise ValueError(
                        f"span word axis {scores.shape[-2]} != words {expected_words} "
                        "(include the classification prefix)"
                    )
                if group["task_type"] == "entities":
                    doc_scores = _doc_axis(scores[0], doc_len)
                    group_raw = None
                    if raw_logits is not None:
                        group_raw = raw_logits[index][:count]
                    results[group["name"]] = self._decode_entities(
                        group["fields"],
                        doc_scores,
                        group_raw,
                        meta,
                        text,
                        start_map,
                        end_map,
                        threshold,
                        include_confidence,
                        include_spans,
                        overlap_policy,
                    )
                elif group["task_type"] == "relations":
                    results[group["name"]] = self._decode_relations(
                        group["name"],
                        group["fields"],
                        scores,
                        count,
                        meta,
                        text,
                        start_map,
                        end_map,
                        threshold,
                        include_confidence,
                        include_spans,
                        overlap_policy,
                        doc_len,
                    )
                else:
                    results[group["name"]] = self._decode_structures(
                        group["name"],
                        group["fields"],
                        scores,
                        count,
                        meta,
                        words,
                        text,
                        start_map,
                        end_map,
                        threshold,
                        include_confidence,
                        include_spans,
                        overlap_policy,
                        doc_len,
                        prefix_len,
                    )
        cls_source = sample_out.get("classification_logits")
        cls_are_probs = False
        if cls_source is None and sample_out.get("classification_probs") is not None:
            cls_source = sample_out["classification_probs"]
            cls_are_probs = True
        if cls_source is not None and cls_groups:
            vectors = self._classification_vectors(cls_source, len(cls_groups))
            classifications = list(meta.get("classifications") or [])
            for group, vector in zip(cls_groups, vectors):
                config = _resolve_classification_config(group["prompt"], classifications)
                if config is None:
                    continue
                decoded = _decode_classification_group(vector, config, threshold, temperature, activated=cls_are_probs)
                results[config["task"]] = decoded
        return results

    def post_process_extraction(
        self,
        outputs: Any,
        metadata: Any,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        overlap_policy: str | None = None,
        temperature: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Decode logits into the public extraction payload.

        ``metadata`` is ``encoding["metadata"]``. Each sample needs ``words``
        and ``schema_meta`` (char ``start``/``end``, ``groups``, ``text``).

        Expected ``outputs`` keys, per sample or batched as lists:

        - ``span_scores``: probabilities shaped ``(count, fields, words, width)``.
          One tensor per non-classification group, in ``groups`` order.
          ``words`` includes the classification-choice prefix. A 3D tensor is
          treated as a single instance.
        - ``span_logits``: same layout. Sigmoid is applied when scores are absent.
        - ``counts``: optional instance counts. Defaults to ``span_scores`` size 0.
        - ``classification_logits``: one ``(num_labels,)`` vector per
          classification group, aligned with ``schema_meta["classifications"]``.
        - ``classification_probs``: used when logits are absent.
        - ``raw_logits``: optional pre-sigmoid span scores for entity attributes.

        Relations and JSON structures are decoded from the same span scores.
        ``record_logits`` and boundary ``pair_logits`` / ``grouped_candidates``
        are delegated to ``decoding_gliner2`` and raise ``NotImplementedError``
        when that module does not define ``decode_records`` or ``decode_boundary``.
        """
        rows = self._unpack_metadata(metadata)
        samples = self._sample_outputs(outputs, len(rows))
        formatted = []
        for sample_out, meta in zip(samples, rows):
            chosen = overlap_policy if overlap_policy is not None else meta.get("_overlap_policy")
            if chosen is None and meta.get("architecture") == "boundary":
                chosen = "disallow"
            policy = None if chosen is None else _normalize_overlap_policy(chosen)
            raw = self._decode_sample(
                sample_out,
                meta,
                threshold,
                include_confidence,
                include_spans,
                policy,
                temperature,
            )
            formatted.append(
                format_results(
                    raw,
                    include_confidence=include_confidence,
                    requested_relations=list(meta.get("relation_order") or []),
                    classification_tasks=list(meta.get("classification_tasks") or []),
                )
            )
        return formatted

    def post_process_constrained_classification(
        self,
        outputs: Any,
        metadata: Any,
        threshold: float = 0.5,
        include_confidence: bool = False,
        temperature: float = 1.0,
        decoder: str = "independent",
    ) -> list[dict[str, Any]]:
        """Decode classification logits.

        ``decoder="independent"`` (and ``"auto"`` without constraints) scores
        each task locally: softmax for single-label, sigmoid for multi-label.
        ``decoder="beam"`` or ``"exact"``, and non-empty schema constraints,
        call ``decoding_gliner2.decode_classification`` when it exists.
        """
        rows = self._unpack_metadata(metadata)
        needs_solver = decoder in ("beam", "exact") or any(row.get("constraints") for row in rows)
        if needs_solver and decoder != "independent":
            return _call_decoder(
                "decode_classification",
                outputs,
                rows,
                threshold=threshold,
                include_confidence=include_confidence,
                temperature=temperature,
                decoder=decoder,
            )
        samples = self._sample_outputs(outputs, len(rows))
        results = []
        for sample_out, meta in zip(samples, rows):
            raw = self._decode_sample(
                sample_out,
                meta,
                threshold,
                include_confidence,
                include_spans=False,
                overlap_policy=None,
                temperature=temperature,
            )
            results.append(
                format_results(
                    {key: value for key, value in raw.items() if key in set(meta.get("classification_tasks") or [])},
                    include_confidence=include_confidence,
                    classification_tasks=list(meta.get("classification_tasks") or []),
                )
            )
        return results

    def post_process_joint_extraction(
        self,
        outputs: Any,
        metadata: Any,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        optimizer: str = "independent",
        overlap_policy: str | None = None,
    ) -> list[dict[str, Any]]:
        """Decode a joint entity, structure, and relation payload.

        ``optimizer="independent"`` decodes each span group on its own scores.
        ``optimizer="beam"``, ``"greedy"``, and ``"exact"`` call
        ``decoding_gliner2.decode_joint``.
        """
        if optimizer in ("beam", "exact", "greedy"):
            return _call_decoder(
                "decode_joint",
                outputs,
                metadata,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
                optimizer=optimizer,
                overlap_policy=overlap_policy,
            )
        if optimizer not in ("independent", "auto", "none"):
            raise ValueError("optimizer must be 'independent', 'auto', 'beam', 'greedy', or 'exact'")
        return self.post_process_extraction(
            outputs,
            metadata,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap_policy,
        )


__all__ = ["Gliner2Processor"]
