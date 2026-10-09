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

import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs
from ...utils import auto_docstring
from .decoding_gliner2 import (
    decode_boundary,
    decode_classification,
    decode_joint,
    decode_records,
    decode_spans,
    format_span,
    normalize_overlap_policy,
    resolve_overlaps,
)


_AGGREGATIONS = ("max", "mean", "first")
_POOLING = ("first", "mean", "max")

logger = logging.getLogger(__name__)

TextInput = str | Sequence[str]

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
_SCALAR_CARDINALITY = frozenset({"optional_one", "required_one"})


class Gliner2TextKwargs(TextKwargs, total=False):
    """
    GLiNER2 call options merged through `ProcessingKwargs`.

    Args:
        max_len (`int`, *optional*):
            Maximum tokenized length of one window.
        architecture (`str`, *optional*):
            `"span"` or `"boundary"`.
        labels (`dict`, *optional*):
            Supervision grouped by task.
        max_gold_per_query (`int`, *optional*):
            Maximum gold spans packed per query.
    """

    max_len: int | None
    architecture: str
    labels: Any
    max_gold_per_query: int | None


class Gliner2ProcessorKwargs(ProcessingKwargs, total=False):
    """Call kwargs accepted by `Gliner2Processor`."""

    _defaults = {}
    text_kwargs: Gliner2TextKwargs


class FieldSpec(TypedDict, total=False):
    """
    One entity or structure field in a caller schema.

    Args:
        dtype (`str`, *optional*):
            Value kind, such as `"str"` or `"list"`.
        threshold (`float`, *optional*):
            Minimum score for this field.
        description (`str`, *optional*):
            Text added to the field prompt.
        choices (`list[str]`, *optional*):
            Closed set of values for a choice field.
        validators (`list`, *optional*):
            Caller-side checks stored with the field.
        value (`str`, *optional*):
            Fixed value stored with the field.
    """

    dtype: str
    threshold: float
    description: str
    choices: list[str]
    validators: list[Any]
    value: str


class ClassificationTask(TypedDict, total=False):
    """
    One classification task in a caller schema.

    Args:
        task (`str`, *optional*):
            Name of the classification task.
        labels (`list[str]`, *optional*):
            Class names.
        multi_label (`bool`, *optional*):
            Whether several classes may be selected.
        cls_threshold (`float`, *optional*):
            Minimum class score.
        class_act (`str`, *optional*):
            `"softmax"` or `"sigmoid"`.
        prompt (`str`, *optional*):
            Text prepended to the class list.
        examples (`list[tuple[str, str]]`, *optional*):
            Text and label pairs shown in the prompt.
        label_descriptions (`dict[str, str]`, *optional*):
            Extra text for each class name.
    """

    task: str
    labels: list[str]
    multi_label: bool
    cls_threshold: float
    class_act: str
    prompt: str
    examples: list[tuple[str, str]]
    label_descriptions: dict[str, str]


class Gliner2Schema(TypedDict, total=False):
    """
    Schema accepted by `Gliner2Processor.__call__`.

    Args:
        entities (`dict` or `list`, *optional*):
            Entity fields, as a name-to-spec map or a list of names.
        entity_descriptions (`dict[str, str]`, *optional*):
            Prompt text for each entity name.
        entity_attribute_groups (`dict`, *optional*):
            Attribute groups attached to entity types.
        entity_attribute_labels (`list[str]`, *optional*):
            Attribute names scored for every entity.
        entity_attribute_prompt_labels (`dict[str, str]`, *optional*):
            Prompt text for each attribute name.
        json_structures (`list[dict]`, *optional*):
            Record structures and their fields.
        json_descriptions (`dict`, *optional*):
            Prompt text for structure fields.
        relations (`list[dict]`, *optional*):
            Relation types and their head and tail fields.
        relation_descriptions (`dict[str, str]`, *optional*):
            Prompt text for each relation name.
        relation_metadata (`dict`, *optional*):
            Endpoint names and thresholds for each relation.
        classifications (`list`, *optional*):
            Classification tasks.
        constraints (`list[dict]`, *optional*):
            Constraints applied by the classification decoder.
        record_metadata (`dict`, *optional*):
            Record mode and anchor field for each structure.
    """

    entities: dict[str, str | FieldSpec] | list[str]
    entity_descriptions: dict[str, str]
    entity_attribute_groups: dict[str, dict[str, Any]]
    entity_attribute_labels: list[str]
    entity_attribute_prompt_labels: dict[str, str]
    json_structures: list[dict[str, dict[str, Any]]]
    json_descriptions: dict[str, dict[str, str]]
    relations: list[dict[str, dict[str, Any]]]
    relation_descriptions: dict[str, str]
    relation_metadata: dict[str, dict[str, Any]]
    classifications: list[ClassificationTask]
    constraints: list[dict[str, Any]]
    record_metadata: dict[str, dict[str, Any]]


class SpanLabel(TypedDict, total=True):
    """
    One labeled character span.

    Args:
        start (`int`):
            Inclusive character start.
        end (`int`):
            Exclusive character end.
    """

    text: str
    start: int
    end: int


class Gliner2Labels(TypedDict, total=False):
    """
    Supervision accepted by `Gliner2Processor.__call__`.

    Args:
        entities (`dict`, *optional*):
            Entity name to a span, a string, or a list of either.
        classifications (`dict`, *optional*):
            Task name to a class name or a list of class names.
        relations (`dict`, *optional*):
            Relation name to a head/tail pair or a list of pairs.
        json_structures (`dict`, *optional*):
            Structure name to a record or a list of records.
    """

    entities: dict[str, list[str | SpanLabel] | str | SpanLabel]
    classifications: dict[str, list[str] | str]
    relations: dict[str, list[dict[str, Any]] | dict[str, Any]]
    json_structures: dict[str, list[dict[str, Any]] | dict[str, Any]]


@auto_docstring(custom_intro="Schema field kind. Serializes to `span`, `choice`, or `label`.")
class FieldKind(str, Enum):
    """Schema field kind. Serializes to `span`, `choice`, or `label`."""

    SPAN = "span"
    CHOICE = "choice"
    LABEL = "label"

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Field:
    """One schema field after compilation."""

    kind: FieldKind = FieldKind.SPAN
    name: str = ""
    dtype: str = "list"
    threshold: float | None = None
    choices: tuple[str, ...] = ()
    validators: tuple[Any, ...] = ()
    description: str | None = None
    query_id: int | None = None
    is_scalar: bool | None = None
    start: int | None = None
    end: int | None = None

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, FieldKind) else FieldKind(self.kind)
        choices = self.choices if isinstance(self.choices, tuple) else tuple(self.choices or ())
        validators = self.validators if isinstance(self.validators, tuple) else tuple(self.validators or ())
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "validators", validators)


@dataclass(frozen=True)
class AttributeSpec:
    """Entity attribute group rescored from kept spans."""

    name: str
    labels: tuple[str, ...]
    applies_to: tuple[str, ...] | None
    multi_label: bool
    threshold: float


@dataclass(frozen=True)
class RecordField:
    """One record field and its boundary query id."""

    name: str
    query_id: int
    cardinality: str
    is_anchor: bool
    exclusive: bool


@dataclass(frozen=True)
class GroupRecord:
    """Record head configuration for one structure group."""

    mode: str
    anchor: str | None
    occurrence_policy: str
    fields: tuple[RecordField, ...]
    anchor_query_id: int | None
    task_index: int


@dataclass(frozen=True)
class FieldGroup:
    """One encoded schema group. Decoding and the pipeline read this object."""

    task: str
    name: str
    fields: tuple[Field, ...]
    tokens: tuple[str, ...] = ()
    endpoints: tuple[str, ...] = ()
    record: GroupRecord | None = None
    threshold: float | None = None
    multi_label: bool = False
    activation: str = "auto"
    prompt: str | None = None
    attributes: tuple[AttributeSpec, ...] = ()
    attribute_labels: tuple[str, ...] = ()
    attribute_prompts: tuple[tuple[str, str], ...] = ()
    cls_threshold: float | None = None
    class_act: str | None = None
    examples: tuple[Any, ...] | None = None
    label_descriptions: dict[str, str] | None = None
    options: dict[str, Any] | None = None
    choices: dict[str, Any] | None = None
    true_labels: Any = None
    record_mode: str | None = None
    record_anchor: str | None = None

    def __post_init__(self) -> None:
        if self.class_act is not None:
            object.__setattr__(self, "activation", self.class_act)
        options = self.options
        if options:
            record = options.get("record") if isinstance(options, Mapping) else None
            if isinstance(record, Mapping):
                if self.record_mode is None:
                    object.__setattr__(self, "record_mode", record.get("mode"))
                if self.record_anchor is None:
                    object.__setattr__(self, "record_anchor", record.get("anchor"))
            return
        object.__setattr__(self, "options", _field_group_options(self))


def _field_group_options(group: FieldGroup) -> dict[str, Any]:
    """Build the metadata dict `SchemaGroup.options` exposes."""
    fields = {
        field.name: {
            "dtype": field.dtype,
            "threshold": field.threshold,
            "validators": field.validators,
            "choices": field.choices,
        }
        for field in group.fields
    }
    options: dict[str, Any] = {"fields": fields}
    class_act = group.class_act if group.class_act is not None else group.activation
    descriptions = dict(group.label_descriptions or {})
    examples = list(group.examples or [])
    if (
        group.task == "classifications"
        or group.class_act is not None
        or descriptions
        or examples
        or group.cls_threshold is not None
    ):
        options["classification"] = {
            "multi_label": group.multi_label,
            "cls_threshold": group.cls_threshold if group.cls_threshold is not None else group.threshold,
            "class_act": class_act,
            "examples": examples,
            "label_descriptions": descriptions,
        }
    if group.threshold is not None and group.task != "classifications":
        options["threshold"] = group.threshold
    if group.endpoints:
        options["endpoints"] = tuple(group.endpoints)
    if group.record_mode is not None or group.record_anchor is not None:
        options["record"] = {"mode": group.record_mode, "anchor": group.record_anchor}
    return options


SchemaField = Field
SchemaGroup = FieldGroup


@dataclass(frozen=True)
class WordGrid:
    """Document words and the choice-prefix length they were aligned to."""

    words: tuple[str, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    text: str
    prefix_len: int
    prefix: tuple[str, ...]

    @property
    def encoded_length(self) -> int:
        return self.prefix_len + len(self.words)


@dataclass
class Cardinality:
    """Whether a record field keeps one span."""

    is_scalar: bool


@dataclass
class RecordFieldSpec:
    """Record field the loss reads by query id."""

    query_id: int
    cardinality: Cardinality | None = None
    name: str | None = None
    exclusive: bool = False
    allows_absent: bool = False
    values: Any = None


@dataclass
class RecordSpec:
    """Record schema object `RecordHead.forward_group` reads."""

    mode: str
    fields: list[RecordFieldSpec]
    anchor_query_id: int | None
    task_index: int = 0


FieldCardinality = Cardinality


@dataclass
class RecordFieldValue:
    """Gold spans for one record field."""

    query_id: int
    values: list


@dataclass
class RecordTarget:
    """Gold record with `field_for_query`."""

    task_index: int
    fields: list[RecordFieldValue]

    def field_for_query(self, query_id: int) -> RecordFieldValue | None:
        wanted = int(query_id)
        for item in self.fields:
            if item.query_id == wanted:
                return item
        return None


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
    if isinstance(schema, Mapping):
        return schema
    for cls in type(schema).__mro__:
        build = cls.__dict__.get("build")
        if callable(build):
            return schema.build()
    for cls in type(schema).__mro__:
        if "schema" in cls.__dict__:
            return schema.__getattribute__("schema")
    return schema


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

    Args:
        words (`str` or sequence):
            Document text or pre-split words.
        chunk_size (`int`, *optional*, defaults to 384):
            Maximum words in one window.
        chunk_overlap (`int`, *optional*, defaults to 64):
            Words shared by neighboring windows.
        word_splitter (`str` or `callable`, *optional*):
            Splitter name or callable used when `words` is a string.
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
        for index, word in enumerate(words):
            if isinstance(word, (tuple, list)) and len(word) >= 3:
                tokens.append((str(word[0]), int(word[1]), int(word[2])))
            else:
                tokens.append((str(word), index, index + 1))
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


def _chunk_start(chunk: Any) -> int:
    if isinstance(chunk, Mapping):
        return int(chunk["start_char"])
    return int(chunk.start_char)


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


def remap_result_spans(result: Any, original_text: str, chunk: Any) -> Any:
    """Add a chunk's character offset onto span dicts.

    Args:
        result:
            Nested decode payload.
        original_text (`str`):
            Document the chunk was cut from.
        chunk:
            Window with `start_char`.
    """
    if isinstance(result, list):
        return [remap_result_spans(item, original_text, chunk) for item in result]
    if isinstance(result, dict):
        remapped = {key: remap_result_spans(value, original_text, chunk) for key, value in result.items()}
        if _is_span_dict(remapped):
            start = int(remapped["start"]) + _chunk_start(chunk)
            end = int(remapped["end"]) + _chunk_start(chunk)
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
        selected = resolve_overlaps(
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


def _merge_map(
    values: list[Any],
    overlap_policy: str = "disallow",
    *,
    scalars: set | None = None,
    listed: bool = False,
) -> dict[str, Any]:
    """Merge dicts by key. `scalars` keeps one value. `listed` always returns lists."""
    merged: dict[str, Any] = {}
    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        for key in value:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    for key in keys:
        if listed or scalars is not None:
            items: list[Any] = []
            for value in values:
                if isinstance(value, dict) and key in value:
                    items.extend(_as_list(value[key]))
            if listed:
                merged[key] = _dedupe_items(items)
            else:
                deduped = _dedupe_items(items, overlap_policy=overlap_policy)
                merged[key] = (deduped[0] if deduped else None) if key in scalars else deduped
        else:
            nested = [value[key] for value in values if isinstance(value, dict) and key in value]
            merged[key] = _merge_values(nested, overlap_policy)
    return merged


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
        return _merge_map(non_empty, overlap_policy)
    kinds = {type(value).__name__ for value in non_empty}
    if len(kinds) > 1:
        raise ValueError(f"cannot merge values of types {sorted(kinds)}")
    return non_empty[0]


def _merge_result_dicts(
    results: list[dict[str, Any]],
    scalar_entity_labels: set | None = None,
    overlap_policy: str = "disallow",
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    keys: list[str] = []
    seen: set[str] = set()
    for result in results:
        for key in result:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    for key in keys:
        values = [result.get(key) for result in results if key in result]
        if key == "entities":
            merged[key] = _merge_map(values, overlap_policy, scalars=set(scalar_entity_labels or ()))
        elif key == "relation_extraction":
            merged[key] = _merge_map(values, overlap_policy, listed=True)
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


def _is_joint_result(result: Mapping[str, Any]) -> bool:
    entities = result.get("entities")
    if not isinstance(entities, list):
        return False
    return not entities or (isinstance(entities[0], Mapping) and "type" in entities[0])


def _merge_joint_documents(original_text, chunks, chunk_results, include_confidence, include_spans) -> dict[str, Any]:
    """Merge joint graphs by document offsets. Relations stay inside one chunk."""
    entity_by_key = {}
    relation_rows = {}
    for chunk, raw in zip(chunks, chunk_results):
        start_char = _chunk_start(chunk)
        entities = list(raw.get("entities") or [])
        relations = list(raw.get("relations") or [])
        local_keys = {}
        for entity in entities:
            start = int(entity.get("start", 0)) + start_char
            end = int(entity.get("end", 0)) + start_char
            key = (str(entity.get("type", entity.get("label", ""))), start, end)
            local_keys[str(entity.get("id", ""))] = key
            confidence = entity.get("confidence")
            previous = entity_by_key.get(key)
            previous_confidence = (
                float("-inf") if previous is None or previous.get("confidence") is None else previous["confidence"]
            )
            current_confidence = float("-inf") if confidence is None else confidence
            if previous is None or current_confidence > previous_confidence:
                entity_by_key[key] = {
                    "type": key[0],
                    "text": original_text[start:end],
                    "start": start,
                    "end": end,
                    "confidence": confidence,
                    "sentence_id": entity.get("sentence_id"),
                    "rescued": bool(entity.get("rescued", False)),
                }
        for relation in relations:
            head, tail = str(relation.get("head", "")), str(relation.get("tail", ""))
            if head not in local_keys or tail not in local_keys:
                continue
            head_key, tail_key = local_keys[head], local_keys[tail]
            key = (str(relation.get("type", relation.get("label", ""))), head_key, tail_key)
            confidence = relation.get("confidence")
            previous = relation_rows.get(key)
            previous_confidence = float("-inf") if previous is None or previous[3] is None else previous[3]
            current_confidence = float("-inf") if confidence is None else confidence
            if previous is None or current_confidence > previous_confidence:
                relation_rows[key] = (key[0], head_key, tail_key, confidence, bool(relation.get("derived", False)))
    ordered = sorted(entity_by_key, key=lambda key: (key[1], key[2], key[0]))
    key_to_id = {key: f"e{index + 1}" for index, key in enumerate(ordered)}
    entities_out = []
    for key in ordered:
        item = entity_by_key[key]
        payload = {"id": key_to_id[key], "type": item["type"], "text": item["text"]}
        if include_spans:
            payload.update(start=item["start"], end=item["end"])
            if item["sentence_id"] is not None:
                payload["sentence_id"] = item["sentence_id"]
        if include_confidence and item["confidence"] is not None:
            payload["confidence"] = item["confidence"]
        if item["rescued"]:
            payload["rescued"] = True
        entities_out.append(payload)
    relations_out = []
    for label, head, tail, confidence, derived in relation_rows.values():
        payload = {"type": label, "head": key_to_id[head], "tail": key_to_id[tail]}
        if include_confidence and confidence is not None:
            payload["confidence"] = confidence
        if derived:
            payload["derived"] = True
        relations_out.append(payload)
    relations_out.sort(key=lambda item: (item["type"], item["head"], item["tail"]))
    return {"entities": entities_out, "relations": relations_out}


def merge_chunk_results(
    original_text: str,
    chunks: Sequence[Any],
    chunk_results: Sequence[dict[str, Any]],
    include_confidence: bool = False,
    include_spans: bool = False,
    scalar_entity_labels: Iterable[str] | None = None,
    overlap_policy: str | None = None,
) -> dict[str, Any]:
    """Merge formatted chunk results onto document character offsets.

    Args:
        original_text (`str`):
            Document text.
        chunks:
            Windows aligned with `chunk_results`.
        chunk_results:
            One formatted decode per window.
        include_confidence (`bool`, *optional*, defaults to `False`):
            Keep scores after the merge.
        include_spans (`bool`, *optional*, defaults to `False`):
            Keep character offsets after the merge.
        scalar_entity_labels (`iterable`, *optional*):
            Entity names stored as one span instead of a list.
        overlap_policy (`str`, *optional*):
            Overlap rule. The default is `disallow`.
    """
    if len(chunks) != len(chunk_results):
        raise ValueError("chunks and chunk_results must have the same length")
    if chunk_results and all(isinstance(item, Mapping) and _is_joint_result(item) for item in chunk_results):
        return _merge_joint_documents(original_text, chunks, chunk_results, include_confidence, include_spans)
    policy = normalize_overlap_policy(overlap_policy, default="disallow")
    remapped = [remap_result_spans(result, original_text, chunk) for chunk, result in zip(chunks, chunk_results)]
    merged = _merge_result_dicts(remapped, set(scalar_entity_labels or ()), policy)
    return _strip_span_metadata(merged, include_confidence, include_spans)


def _classification_tasks(schema: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    tasks = schema.get("tasks")
    if isinstance(tasks, Mapping):
        rows = []
        for name, body in tasks.items():
            labels = body.get("labels") if isinstance(body, Mapping) else body
            names = list(labels) if isinstance(labels, Mapping) else list(labels or [])
            rows.append((str(name), [str(item) for item in names]))
        return rows
    rows = []
    for item in schema.get("classifications") or []:
        rows.append((str(item["task"]), [str(label) for label in item["labels"]]))
    if not rows:
        raise ValueError("schema has no classification tasks")
    return rows


def _logit_number(value: Any) -> float:
    if torch.is_tensor(value):
        flat = value.detach().float().cpu().reshape(-1)
        return float(flat[0].item()) if flat.numel() else float("-inf")
    return float(value)


def _labels_from_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(tokens[index + 1]) for index in range(len(tokens) - 1) if tokens[index] == L_TOKEN]


def _align_encoded_logits(
    payload: Mapping[str, Any], tasks: Sequence[tuple[str, list[str]]]
) -> dict[str, dict[str, float]]:
    """Align schema-token logit rows onto the schema's task and label order."""
    tokens = payload.get("schema_tokens_list", payload.get("schema_tokens"))
    raw = payload["logits"]
    known = [name for name, _ in tasks]
    label_of = dict(tasks)
    if tokens and isinstance(tokens[0], str):
        groups = [list(tokens)]
        rows = [raw]
    else:
        groups = [list(group) for group in tokens]
        if torch.is_tensor(raw) and raw.ndim == 2:
            rows = [raw[index] for index in range(raw.shape[0])]
        else:
            rows = list(raw)
    found = {name: {} for name in known}
    for group, row in zip(groups, rows):
        names = _labels_from_tokens(group)
        prompt = group[2] if len(group) > 2 else ""
        config = _resolve_classification_config(prompt, [{"task": name} for name in known])
        task = config["task"] if config is not None else str(prompt).split(f" {DESC_TOKEN} ")[0].split(":", 1)[0]
        if task not in label_of:
            raise ValueError(f"encoded tokens name unknown task {task!r}")
        values = row.detach().float().cpu().tolist() if torch.is_tensor(row) else list(row)
        for label, value in zip(names, values):
            if label in label_of[task]:
                found[task][label] = float(value)
    aligned = {}
    for name, labels in tasks:
        if not found[name]:
            raise ValueError(f"logits missing task {name!r}")
        aligned[name] = {label: found[name].get(label, float("-inf")) for label in labels}
    return aligned


def _align_label_rows(logits: Any, tasks: Sequence[tuple[str, list[str]]]) -> dict[str, dict[str, float]]:
    if isinstance(logits, Mapping) and ("schema_tokens" in logits or "schema_tokens_list" in logits):
        return _align_encoded_logits(logits, tasks)
    payload = logits
    if isinstance(payload, Mapping) and "tasks" in payload and not any(name in payload for name, _ in tasks):
        payload = payload["tasks"]
    aligned: dict[str, dict[str, float]] = {}
    if isinstance(payload, Mapping):
        for name, labels in tasks:
            row = payload[name]
            if isinstance(row, Mapping):
                aligned[name] = {label: _logit_number(row[label]) for label in labels}
            else:
                values = row.detach().float().cpu().tolist() if torch.is_tensor(row) else list(row)
                aligned[name] = {label: float(values[index]) for index, label in enumerate(labels)}
        return aligned
    rows = payload
    if torch.is_tensor(rows) and rows.ndim == 2:
        rows = [rows[index] for index in range(rows.shape[0])]
    for (name, labels), row in zip(tasks, rows):
        values = row.detach().float().cpu().tolist() if torch.is_tensor(row) else list(row)
        aligned[name] = {label: float(values[index]) for index, label in enumerate(labels)}
    return aligned


def aggregate_classification_logits(chunk_logits, schema, mode: str = "max") -> dict:
    """Aggregate per-chunk label logits, then the caller decodes once.

    Args:
        chunk_logits:
            One label-logit mapping or row per chunk.
        schema (`dict`):
            Classification schema (`tasks` or `classifications`).
        mode (`str`, *optional*, defaults to `"max"`):
            `max`, `mean`, or `first`.
    """
    if mode not in _AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {_AGGREGATIONS}")
    if not chunk_logits:
        raise ValueError("cannot aggregate an empty list of chunk scores")
    tasks = _classification_tasks(schema)
    aligned = [_align_label_rows(item, tasks) for item in chunk_logits]
    aggregated = {}
    for name, labels in tasks:
        aggregated[name] = {}
        for label in labels:
            column = [row[name][label] for row in aligned]
            if mode == "max":
                aggregated[name][label] = max(column)
            elif mode == "mean":
                aggregated[name][label] = sum(column) / len(column)
            else:
                aggregated[name][label] = column[0]
    return aggregated


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


def _dedupe_field_values(value: list, include_confidence: bool) -> list:
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
    return unique


def format_field(struct: Mapping[str, Any], include_confidence: bool) -> dict[str, Any]:
    """Deduplicate one entity map or one structure instance.

    Args:
        struct (`dict`):
            Field name to spans, a span tuple, or a nested value.
        include_confidence (`bool`):
            Keep scores on tuple spans.
    """
    formatted = {}
    for field_name, value in struct.items():
        if isinstance(value, list):
            formatted[field_name] = _dedupe_field_values(value, include_confidence)
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
    """Format raw extraction results into the public payload.

    Args:
        results (`dict`):
            Raw task outputs from decoding.
        include_confidence (`bool`, *optional*, defaults to `False`):
            Keep scores on labels and spans.
        requested_relations (`list[str]`, *optional*):
            Relation names that are always present, possibly empty.
        classification_tasks (`list[str]`, *optional*):
            Keys formatted as classification labels.
    """
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
                    formatted[key] = format_field(value[0], include_confidence)
                else:
                    formatted[key] = [format_field(item, include_confidence) for item in value]
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
            formatted[key] = format_field(value, include_confidence)
        else:
            formatted[key] = value
    for relation in requested_relations:
        if relation not in relations:
            relations[relation] = []
    if relations:
        formatted["relation_extraction"] = relations
    return formatted


def _transform_schema(
    parent: str,
    fields: list[str],
    child_prefix: str,
    prompt: str | None = None,
    examples: list[tuple[str, str]] | None = None,
    label_descriptions: dict[str, str] | None = None,
    example_mode: str = "both",
) -> list[str]:
    """Turn one schema group into structural word tokens."""
    prompt_str = parent
    if prompt:
        prompt_str = f"{parent}: {prompt}"
    if example_mode in ("descriptions", "both") and label_descriptions:
        described = [(label, desc) for label, desc in label_descriptions.items() if label in fields]
        for label, desc in described:
            prompt_str += f" {DESC_TOKEN} {label}: {desc}"
    if example_mode in ("few_shot", "both") and examples:
        for inp, out in examples:
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


def _choice_prefix(schema: Mapping[str, Any]) -> list[str]:
    """Word tokens prepended for JSON choice fields."""
    prefix_tokens: list[str] = []
    for struct in schema.get("json_structures", []) or []:
        for parent, fields in struct.items():
            cls_fields = _choice_field_pairs(fields)
            inner: list[str] = []
            for fname, fval in cls_fields:
                choice_tokens: list[str] = []
                for index, choice in enumerate(fval["choices"]):
                    if index > 0:
                        choice_tokens.append("|")
                    choice_tokens.append(str(choice))
                inner.extend([fname, "("] + choice_tokens + [")", ","])
            if inner:
                inner = inner[:-1]
                prefix_tokens.extend(["(", f"{parent}:", *inner, ")"])
    return prefix_tokens


def _field_spec_map(occurrences: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """First field dict for each name, in occurrence order."""
    specs: dict[str, dict[str, Any]] = {}
    for occurrence in occurrences:
        for name in names:
            if name in specs or name not in occurrence:
                continue
            value = occurrence[name]
            specs[name] = dict(value) if isinstance(value, Mapping) else {}
    return specs


def _make_field(task: str, name: str, spec: Mapping[str, Any] | None, choices: Sequence[str] = ()) -> Field:
    spec = spec or {}
    raw_choices = tuple(str(choice) for choice in (spec.get("choices") or choices or ()))
    if task == "classifications":
        kind = FieldKind.LABEL
    elif task == "json_structures" and raw_choices:
        kind = "choice"
    else:
        kind = "span"
    threshold = spec.get("threshold")
    return Field(
        kind=kind,
        name=name,
        dtype=str(spec.get("dtype") or "list"),
        threshold=None if threshold is None else float(threshold),
        choices=raw_choices,
        validators=tuple(spec.get("validators") or ()),
    )


def _make_fields(
    task: str,
    names: Sequence[str],
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    choices: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Field, ...]:
    choice_map = choices or {}
    spec_map = specs or {}
    return tuple(_make_field(task, name, spec_map.get(name), choice_map.get(name, ())) for name in names)


def _attribute_specs(raw: Mapping[str, Any] | None) -> tuple[AttributeSpec, ...]:
    if not raw:
        return ()
    if not isinstance(raw, Mapping):
        raise TypeError("entity_attribute_groups must be a mapping")
    specs = []
    for name, group in raw.items():
        if not isinstance(group, Mapping):
            raise TypeError(f"entity_attribute_groups[{name!r}] must be a mapping")
        applies = group.get("applies_to")
        raw_threshold = group.get("threshold", 0.5)
        specs.append(
            AttributeSpec(
                name=str(name),
                labels=tuple(str(label) for label in (group.get("labels") or ())),
                applies_to=None if applies is None else tuple(str(item) for item in applies),
                multi_label=bool(group.get("multi_label", False)),
                threshold=0.5 if raw_threshold is None else float(raw_threshold),
            )
        )
    return tuple(specs)


def _structure_groups(schema: Mapping[str, Any]) -> list[FieldGroup]:
    if "json_structures" not in schema:
        return []
    json_descs = schema.get("json_descriptions", {}) or {}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in schema["json_structures"] or []:
        for parent, fields in item.items():
            grouped.setdefault(parent, []).append(fields)
    compiled = []
    for parent, occurrences in grouped.items():
        common: list[str] = []
        seen_fields: set[str] = set()
        for occurrence in occurrences:
            for field_name in occurrence:
                if field_name not in seen_fields:
                    common.append(field_name)
                    seen_fields.add(field_name)
        if not common:
            continue
        descs = json_descs.get(parent, {}) or {}
        mode = "descriptions" if descs else "none"
        tokens = _transform_schema(parent, common, C_TOKEN, label_descriptions=descs, example_mode=mode)
        choices = {}
        for occurrence in occurrences:
            for fname, fval in occurrence.items():
                if fname in common and isinstance(fval, Mapping) and fval.get("choices"):
                    choices[str(fname)] = tuple(str(choice) for choice in fval["choices"])
        compiled.append(
            FieldGroup(
                task="json_structures",
                name=parent,
                fields=_make_fields("json_structures", common, _field_spec_map(occurrences, common), choices),
                tokens=tuple(tokens),
            )
        )
    return compiled


def _entity_group(schema: Mapping[str, Any]) -> FieldGroup | None:
    if "entities" not in schema:
        return None
    entity_values = schema.get("entities") or {}
    entity_fields = list(entity_values.keys())
    if not entity_fields:
        return None
    descs = schema.get("entity_descriptions", {}) or {}
    mode = "descriptions" if descs else "none"
    tokens = _transform_schema("entities", entity_fields, E_TOKEN, label_descriptions=descs, example_mode=mode)
    specs = {name: dict(value) if isinstance(value, Mapping) else {} for name, value in entity_values.items()}
    prompts = schema.get("entity_attribute_prompt_labels") or {}
    return FieldGroup(
        task="entities",
        name="entities",
        fields=_make_fields("entities", entity_fields, specs),
        tokens=tuple(tokens),
        attributes=_attribute_specs(schema.get("entity_attribute_groups") or {}),
        attribute_labels=tuple(str(label) for label in (schema.get("entity_attribute_labels") or ())),
        attribute_prompts=tuple((str(key), str(value)) for key, value in prompts.items()),
    )


def _relation_groups(schema: Mapping[str, Any]) -> list[FieldGroup]:
    if "relations" not in schema:
        return []
    relation_descriptions = schema.get("relation_descriptions", {}) or {}
    relation_metadata = schema.get("relation_metadata") or {}
    grouped: dict[str, list] = {}
    for item in schema["relations"] or []:
        for parent, fields in item.items():
            grouped.setdefault(parent, []).append(fields)
    compiled = []
    for parent, occurrences in grouped.items():
        if not occurrences:
            continue
        field_names = list(occurrences[0].keys())
        if not any(all(field in occurrence for field in field_names) for occurrence in occurrences):
            continue
        tokens = _transform_schema(parent, field_names, R_TOKEN, prompt=relation_descriptions.get(parent))
        threshold = (relation_metadata.get(parent) or {}).get("threshold")
        compiled.append(
            FieldGroup(
                task="relations",
                name=parent,
                fields=_make_fields("relations", field_names, _field_spec_map(occurrences, field_names)),
                tokens=tuple(tokens),
                endpoints=tuple(field_names[:2]),
                threshold=None if threshold is None else float(threshold),
                prompt=None if relation_descriptions.get(parent) is None else str(relation_descriptions.get(parent)),
            )
        )
    return compiled


def _label_groups(schema: Mapping[str, Any]) -> list[FieldGroup]:
    if "classifications" not in schema:
        return []
    compiled = []
    for item in schema["classifications"] or []:
        labels = list(item["labels"])
        tokens = _transform_schema(
            item["task"],
            labels,
            L_TOKEN,
            prompt=item.get("prompt"),
            examples=item.get("examples", []) or [],
            label_descriptions=item.get("label_descriptions", {}) or {},
            example_mode="both",
        )
        cls_threshold = item.get("cls_threshold")
        compiled.append(
            FieldGroup(
                task="classifications",
                name=str(item["task"]),
                fields=_make_fields("classifications", labels),
                tokens=tuple(tokens),
                threshold=None if cls_threshold is None else float(cls_threshold),
                multi_label=bool(item.get("multi_label", False)),
                activation=str(item.get("class_act", "auto")),
                prompt=None if item.get("prompt") is None else str(item.get("prompt")),
            )
        )
    return compiled


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


def _with_records(groups: Sequence[FieldGroup], schema: Mapping[str, Any]) -> list[FieldGroup]:
    """Attach compiled record specs. Query ids skip classification groups."""
    dtypes = {
        group.name: {field.name: field.dtype for field in group.fields}
        for group in groups
        if group.task == "json_structures"
    }
    normalized = _normalize_record_metadata(schema.get("record_metadata"), dtypes)
    if not normalized:
        return list(groups)
    by_task: dict[int, list[dict[str, Any]]] = {}
    query_id = 0
    for task_index, group in enumerate(groups):
        if group.task == "classifications" or not group.fields:
            continue
        for role_index, field in enumerate(group.fields):
            by_task.setdefault(task_index, []).append(
                {"query_id": query_id, "role_index": role_index, "role_name": field.name}
            )
            query_id += 1
    updated = list(groups)
    for task_index, task_queries in by_task.items():
        group = groups[task_index]
        if group.task not in RECORD_TASK_TYPES or group.name not in normalized:
            continue
        cfg = normalized[group.name]
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
                dtype = dtypes.get(group.name, {}).get(query["role_name"])
                card = _default_cardinality(dtype, is_anchor)
            fields.append(
                RecordField(
                    name=query["role_name"],
                    query_id=query["query_id"],
                    cardinality=card,
                    is_anchor=is_anchor,
                    exclusive=bool(fcfg.get("exclusive", False)),
                )
            )
            if is_anchor:
                anchor_query_id = query["query_id"]
        if mode == "natural" and anchor_query_id is None:
            raise ValueError(f"record {group.name!r} declares anchor {anchor_name!r} but no matching field was found")
        updated[task_index] = replace(
            group,
            record=GroupRecord(
                mode=mode,
                anchor=anchor_name,
                occurrence_policy=cfg["occurrence_policy"],
                fields=tuple(fields),
                anchor_query_id=anchor_query_id,
                task_index=task_index,
            ),
        )
    return updated


def compile_schema(schema: Any) -> tuple[list[FieldGroup], list[str]]:
    """Compile a schema into field groups and the choice-prefix words.

    Token words match `_transform_schema` plus the JSON choice prefix. Inference
    and `labels=` both use this compiler.

    Args:
        schema (`dict`):
            Entities, structures, relations, and classifications.

    Returns:
        `tuple[list[FieldGroup], list[str]]`: Groups in encode order, then the
        choice-prefix words prepended to the document.
    """
    schema = _canonicalize_schema(schema)
    groups: list[FieldGroup] = []
    groups.extend(_structure_groups(schema))
    entity = _entity_group(schema)
    if entity is not None:
        groups.append(entity)
    groups.extend(_relation_groups(schema))
    groups.extend(_label_groups(schema))
    return _with_records(groups, schema), _choice_prefix(schema)


def _result_key(group: FieldGroup) -> str:
    """Decode key. Relation prompts keep the `name: description` text."""
    if group.task == "entities":
        return "entities"
    if len(group.tokens) > 2:
        return str(group.tokens[2]).split(f" {DESC_TOKEN} ")[0]
    return group.task


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


def _canonicalize_entities(schema: dict[str, Any], entities: Any) -> None:
    if isinstance(entities, list):
        schema["entities"] = {str(name): "" for name in entities}
        return
    if isinstance(entities, Mapping):
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
        return
    if entities is not None:
        raise TypeError("entities must be a list or dict")


def _canonicalize_structures(structures: Any) -> list[dict[str, Any]]:
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
    return parsed_structures


def _canonicalize_schema(schema: Any) -> dict[str, Any]:
    """Normalize entities, classifications, relations, and JSON structures."""
    schema = _resolve_schema(schema)
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a dict")
    schema = dict(schema)
    _canonicalize_entities(schema, schema.get("entities"))
    structures = schema.get("json_structures")
    if structures is not None:
        schema["json_structures"] = _canonicalize_structures(structures)
    relations = schema.get("relations")
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
    classifications = schema.get("classifications")
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


def _bind_example(
    groups: Sequence[FieldGroup], labels: Mapping[str, Mapping[str, Any]], grid: WordGrid, splitter
) -> dict:
    """Map one example's labels onto inclusive word spans and half-open mentions."""
    structure_labels: list[Any] = []
    mentions: list[tuple[int, int, int]] = []
    relation_edges: list[list[tuple[int, int, int, int]]] = []
    relation_queries: list[tuple[int, int]] = []
    query_id = 0
    locate_kwargs = {
        "doc_words": grid.words,
        "prefix_len": grid.prefix_len,
        "prefix_tokens": grid.prefix,
        "starts": grid.starts,
        "ends": grid.ends,
        "text": grid.text,
        "splitter": splitter,
    }
    for group in groups:
        if group.task == "classifications":
            true = set(labels["classifications"].get(group.name, []))
            structure_labels.append([1 if field.name in true else 0 for field in group.fields])
            continue
        bound = _bind_span_group(group, labels, locate_kwargs)
        structure_labels.append(bound["structure"])
        for field_index, spans in bound["mentions_by_field"]:
            for start, end in spans:
                mentions.append((query_id + field_index, start, end + 1))
        if group.task == "relations":
            relation_edges.append(bound["edges"])
            relation_queries.append((query_id, query_id + 1))
        query_id += len(group.fields)
    return {
        "structure_labels": structure_labels,
        "mentions": mentions,
        "query_count": query_id,
        "relation_edges": relation_edges,
        "relation_queries": relation_queries,
    }


def _bind_span_group(
    group: FieldGroup, labels: Mapping[str, Mapping[str, Any]], locate_kwargs: dict
) -> dict[str, Any]:
    names = [field.name for field in group.fields]
    if group.task == "entities":
        instance = []
        for field in group.fields:
            raw = labels["entities"].get(field.name, [])
            instance.append(_field_spans(raw, choices=None, **locate_kwargs))
        count = 1 if any(instance) else 0
        pairs = list(enumerate(instance))
        return {"structure": [count, [instance] if count else []], "mentions_by_field": pairs, "edges": []}
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
        for field in group.fields:
            choices = field.choices if field.kind == "choice" else None
            if field.name not in raw_instance:
                if group.task == "relations":
                    raise ValueError(f"relation {group.name!r} is missing {field.name!r}")
                instance.append([])
                continue
            instance.append(
                _field_spans(raw_instance[field.name], choices=choices if choices else None, **locate_kwargs)
            )
        if group.task == "relations":
            if len(instance) < 2 or not instance[0] or not instance[1]:
                raise ValueError(f"relation {group.name!r} is not in the text")
            for head_start, head_end in instance[0]:
                for tail_start, tail_end in instance[1]:
                    edges.append((head_start, head_end + 1, tail_start, tail_end + 1))
        if any(instance):
            built.append(instance)
    pairs = []
    for instance in built:
        for field_index, spans in enumerate(instance):
            pairs.append((field_index, spans))
    return {"structure": [len(built), built], "mentions_by_field": pairs, "edges": edges}


def _span_block(labels: Any) -> bool:
    """Span supervision is `[count, instances]`, not a classification bit vector."""
    return (
        isinstance(labels, list)
        and len(labels) == 2
        and isinstance(labels[0], int)
        and not isinstance(labels[0], bool)
        and isinstance(labels[1], list)
    )


def dense_targets_from_pairs(pairs: torch.Tensor, mask: torch.Tensor, text_length: int):
    """Build start, end, and inside targets from half-open mention pairs.

    Args:
        pairs (`torch.Tensor`):
            Mention pairs shaped `[batch, queries, gold, 2]`.
        mask (`torch.Tensor`):
            True for real pairs. Same shape as `pairs` without the last axis.
        text_length (`int`):
            Word axis length, including the choice prefix.
    """
    if pairs.shape[:-1] != mask.shape or pairs.shape[-1] != 2:
        raise ValueError(f"pairs {tuple(pairs.shape)} and mask {tuple(mask.shape)} are incompatible")
    if text_length < 0:
        raise ValueError("text_length must be non-negative")
    batch, queries = pairs.shape[:2]
    valid = mask & (pairs[..., 0] >= 0) & (pairs[..., 1] > pairs[..., 0]) & (pairs[..., 1] <= text_length)
    weights = valid.to(torch.float32)
    starts = pairs[..., 0].masked_fill(~mask, 0).clamp(0, text_length)
    ends = pairs[..., 1].masked_fill(~mask, 0).clamp(0, text_length)
    start_targets = torch.zeros(batch, queries, text_length + 1, dtype=torch.float32, device=pairs.device)
    end_targets = torch.zeros_like(start_targets)
    start_targets.scatter_add_(2, starts, weights).clamp_(max=1.0)
    end_targets.scatter_add_(2, ends, weights).clamp_(max=1.0)
    difference = torch.zeros(batch, queries, text_length + 2, dtype=torch.float32, device=pairs.device)
    difference.scatter_add_(2, starts, weights)
    difference.scatter_add_(2, ends, -weights)
    inside_targets = (difference[..., : text_length + 1].cumsum(-1)[..., :text_length] > 0.5).to(torch.float32)
    return start_targets, end_targets, inside_targets


def build_record_spec(group: Mapping[str, Any]) -> RecordSpec:
    """Record schema object `RecordHead.forward_group` reads.

    Args:
        group (`dict`):
            Packed record group with `field_query_ids`, `field_scalar`, and `mode`.
    """
    query_ids = group.get("field_query_ids")
    scalars = group.get("field_scalar")
    if not query_ids or scalars is None or len(query_ids) != len(scalars):
        raise ValueError("record group requires field_query_ids and field_scalar of equal length")
    mode = group.get("mode")
    if mode not in ("natural", "latent", "anchorless"):
        raise ValueError(f"unknown record mode {mode!r}")
    fields = [
        RecordFieldSpec(query_id=int(query_id), cardinality=Cardinality(is_scalar=bool(scalar)))
        for query_id, scalar in zip(query_ids, scalars)
    ]
    return RecordSpec(
        mode=mode,
        fields=fields,
        anchor_query_id=group.get("anchor_query_id"),
        task_index=int(group.get("task_index", 0)),
    )


def build_record_targets(group: Mapping[str, Any]) -> list[RecordTarget]:
    """Gold records with `field_for_query`.

    Args:
        group (`dict`):
            Packed record group whose `records` list holds `fields` by query id.
    """
    records = []
    for record in group.get("records", ()):
        raw_fields = record.get("fields", {})
        fields = [RecordFieldValue(query_id=int(query_id), values=values) for query_id, values in raw_fields.items()]
        records.append(RecordTarget(task_index=int(group.get("task_index", 0)), fields=fields))
    return records


def _pack_mentions(supervisions: Sequence[Mapping[str, Any]], max_gold_per_query: int | None):
    batch = len(supervisions)
    query_width = max((item["query_count"] for item in supervisions), default=0)
    grouped: list[dict[int, list[tuple[int, int]]]] = []
    observed = 0
    for item in supervisions:
        per_query: dict[int, list[tuple[int, int]]] = {}
        for query_id, start, end in item["mentions"]:
            pairs = per_query.setdefault(query_id, [])
            pair = (start, end)
            if pair not in pairs:
                pairs.append(pair)
        observed = max(observed, max((len(pairs) for pairs in per_query.values()), default=0))
        grouped.append(per_query)
    if max_gold_per_query is None:
        gold_width = max(observed, 1)
    else:
        if max_gold_per_query <= 0:
            raise ValueError("max_gold_per_query must be > 0 or None")
        gold_width = max_gold_per_query
        for batch_index, per_query in enumerate(grouped):
            for query_id, pairs in per_query.items():
                if len(pairs) > gold_width:
                    raise ValueError(
                        f"sample={batch_index} query_id={query_id} contains {len(pairs)} gold spans, "
                        f"but max_gold_per_query={gold_width}."
                    )
    mention_pairs = torch.zeros((batch, query_width, gold_width, 2), dtype=torch.long)
    mention_mask = torch.zeros((batch, query_width, gold_width), dtype=torch.bool)
    for batch_index, per_query in enumerate(grouped):
        for query_id, pairs in per_query.items():
            count = len(pairs)
            if count:
                mention_pairs[batch_index, query_id, :count] = torch.tensor(pairs, dtype=torch.long)
                mention_mask[batch_index, query_id, :count] = True
    return mention_pairs, mention_mask, query_width


def _pack_routing(supervisions: Sequence[Mapping[str, Any]], query_width: int):
    batch = len(supervisions)
    queries = [item.get("relation_queries") or [] for item in supervisions]
    relation_width = max((len(rows) for rows in queries), default=0)
    head_member = torch.zeros((batch, relation_width, query_width), dtype=torch.bool)
    tail_member = torch.zeros_like(head_member)
    relation_valid = torch.zeros((batch, relation_width), dtype=torch.bool)
    allow_self = torch.zeros_like(relation_valid)
    for batch_index, rows in enumerate(queries):
        for relation_index, (head_query, tail_query) in enumerate(rows):
            if not (0 <= head_query < query_width and 0 <= tail_query < query_width):
                continue
            relation_valid[batch_index, relation_index] = True
            head_member[batch_index, relation_index, head_query] = True
            tail_member[batch_index, relation_index, tail_query] = True
    return head_member, tail_member, relation_valid, allow_self


def _record_supervision(group: FieldGroup, structure: list) -> dict[str, Any]:
    record = group.record
    instances = structure[1] if len(structure) > 1 else []
    gold = []
    for instance in instances:
        fields = {}
        for field_spec, spans in zip(record.fields, instance):
            half_open = [(int(start), int(end) + 1) for start, end in spans]
            fields[int(field_spec.query_id)] = [half_open]
        gold.append({"fields": fields})
    packed = {
        "task_index": record.task_index,
        "task_name": group.name,
        "mode": record.mode,
        "anchor_query_id": record.anchor_query_id,
        "field_query_ids": [int(field.query_id) for field in record.fields],
        "field_scalar": [field.cardinality in _SCALAR_CARDINALITY for field in record.fields],
        "records": gold,
    }
    packed["spec"] = build_record_spec(packed)
    packed["targets"] = build_record_targets(packed)
    return packed


def _pack_record_groups(batch_groups: Sequence[Sequence[FieldGroup]], supervisions: Sequence[Mapping[str, Any]]):
    grouped = []
    for groups, item in zip(batch_groups, supervisions):
        sample = []
        for group in groups:
            if group.record is None:
                continue
            sample.append(_record_supervision(group, item["structure_labels"][group.record.task_index]))
        grouped.append(sample)
    return grouped


def build_targets(
    groups: Sequence[FieldGroup] | Sequence[Sequence[FieldGroup]],
    labels: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    grid: WordGrid | Sequence[WordGrid],
    max_gold_per_query: int | None = None,
    splitter: Callable[..., Iterator[tuple[str, int, int]]] | None = None,
) -> dict[str, Any]:
    """Bind labels on the word grid and pack the loss targets.

    One example is a `FieldGroup` sequence. A batch is a sequence of those.
    The returned keys are the ones the loss reads.

    Args:
        groups:
            Compiled groups for one example or a batch.
        labels:
            Canonical labels aligned with `groups`.
        grid:
            Word grid aligned with `groups`.
        max_gold_per_query (`int`, *optional*):
            Cap on gold spans stored per query.
        splitter (`callable`, *optional*):
            Word splitter used to locate string mentions.
    """
    if groups and isinstance(groups[0], FieldGroup):
        batch_groups = [groups]
        batch_labels = [labels]
        batch_grids = [grid]
    else:
        batch_groups = list(groups)
        batch_labels = list(labels)
        batch_grids = list(grid)
    locate = splitter or _whitespace_words
    supervisions = [
        _bind_example(example_groups, example_labels, example_grid, locate)
        for example_groups, example_labels, example_grid in zip(batch_groups, batch_labels, batch_grids)
    ]
    mention_pairs, mention_mask, query_width = _pack_mentions(supervisions, max_gold_per_query)
    classification_targets = []
    span_structures = []
    for item in supervisions:
        classes = []
        spans = []
        for block in item["structure_labels"]:
            if _span_block(block):
                spans.append(block)
            elif block and isinstance(block[0], int) and not isinstance(block[0], bool):
                classes.append(torch.tensor([int(value) for value in block], dtype=torch.float))
        classification_targets.append(classes)
        span_structures.append(spans)
    relation_rows = [item["relation_edges"] for item in supervisions]
    relation_width = max((len(rows) for rows in relation_rows), default=0)
    edge_width = max((len(edges) for rows in relation_rows for edges in rows), default=0)
    batch = len(supervisions)
    relation_edges = torch.zeros((batch, relation_width, max(edge_width, 1), 4), dtype=torch.long)
    relation_edge_mask = torch.zeros((batch, relation_width, max(edge_width, 1)), dtype=torch.bool)
    for batch_index, rows in enumerate(relation_rows):
        for relation_index, edges in enumerate(rows):
            if not edges:
                continue
            relation_edges[batch_index, relation_index, : len(edges)] = torch.tensor(edges, dtype=torch.long)
            relation_edge_mask[batch_index, relation_index, : len(edges)] = True
    text_length = max((example.encoded_length for example in batch_grids), default=0)
    start_targets, end_targets, inside_targets = dense_targets_from_pairs(mention_pairs, mention_mask, text_length)
    return {
        "span_structures": span_structures,
        "mention_pairs": mention_pairs,
        "mention_mask": mention_mask,
        "start_targets": start_targets,
        "end_targets": end_targets,
        "inside_targets": inside_targets,
        "classification_targets": classification_targets,
        "relation_gold_pairs": relation_edges,
        "relation_gold_mask": relation_edge_mask,
        "relation_routing": _pack_routing(supervisions, query_width),
        "record_groups": _pack_record_groups(batch_groups, supervisions),
    }


def _doc_axis(tensor: torch.Tensor, doc_len: int) -> torch.Tensor:
    if doc_len <= 0:
        return tensor[..., :0, :]
    return tensor[..., -doc_len:, :]


def _find_choice_idx(choice: str, tokens: Sequence[str]) -> int:
    choice_lower = choice.lower()
    for index, token in enumerate(tokens):
        if token.lower() == choice_lower:
            return index
    return -1


def _resolve_classification_config(
    prompt_str: str, classifications: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Longest boundary-aware task name for a prompt, including descriptions."""
    prompt_str = str(prompt_str or "")
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
        bare = prompt_str.split(" [DESCRIPTION] ", 1)[0].split(":", 1)[0]
        best = next((config for config in classifications if config.get("task") == bare), None)
    if best is None:
        best = next((config for config in classifications if prompt_str.startswith(config.get("task", ""))), None)
    return best


def _classification_probs(logits: torch.Tensor, multi_label: bool, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(logits)
    if activation == "softmax":
        return torch.softmax(logits, dim=-1)
    return torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=-1)


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


def _classification_config(group: FieldGroup) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task": group.name,
        "labels": [field.name for field in group.fields],
        "multi_label": group.multi_label,
        "class_act": group.activation,
    }
    if group.threshold is not None:
        config["cls_threshold"] = group.threshold
    if group.prompt is not None:
        config["prompt"] = group.prompt
    return config


def _span_group_tensors(value: Any, num_groups: int, apply_sigmoid: bool) -> list[torch.Tensor]:
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
        if tensor.ndim != 4:
            raise ValueError("span scores must have shape (count, fields, words, width)")
        prepared.append(torch.sigmoid(tensor) if apply_sigmoid else tensor)
    return prepared


def _classification_vectors(value: Any, num_groups: int) -> list[torch.Tensor]:
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


def _decode_choice(
    field: Field, prefix_scores: torch.Tensor, prefix_tokens: Sequence[str], threshold: float, include_confidence: bool
):
    field_threshold = field.threshold if field.threshold is not None else threshold
    if field.dtype == "list":
        selected = []
        seen = set()
        for choice in field.choices:
            if choice in seen:
                continue
            idx = _find_choice_idx(choice, prefix_tokens)
            if idx >= 0 and idx < prefix_scores.shape[0]:
                score = float(prefix_scores[idx, 0].item())
                if score >= field_threshold:
                    selected.append(format_span(choice, score, 0, 0, include_confidence, False))
                    seen.add(choice)
        return selected
    best = None
    best_score = -1.0
    for choice in field.choices:
        idx = _find_choice_idx(choice, prefix_tokens)
        if idx >= 0 and idx < prefix_scores.shape[0]:
            score = float(prefix_scores[idx, 0].item())
            if score > best_score:
                best_score = score
                best = choice
    if best and best_score >= field_threshold:
        return format_span(best, best_score, 0, 0, include_confidence, False)
    return None


def _decode_entity_fields(
    group, scores, text, maps, threshold, include_confidence, include_spans, overlap
) -> dict[str, Any]:
    found = {}
    for index, field in enumerate(group.fields):
        ent_threshold = field.threshold if field.threshold is not None else threshold
        found[field.name] = decode_spans(
            scores[index],
            {
                "dtype": field.dtype or "list",
                "threshold": ent_threshold,
                "validators": list(field.validators),
                "include_confidence": include_confidence,
                "include_spans": include_spans,
                "empty": "blank",
            },
            text,
            maps,
            threshold,
            overlap,
        )
    return found


def _decode_relation_endpoints(
    group, scores, count, text, maps, doc_len, threshold, include_confidence, include_spans, overlap
):
    rel_threshold = group.threshold if group.threshold is not None else threshold
    instances = []
    for inst in range(count):
        sides = []
        for index, _field in enumerate(group.fields):
            sides.append(
                decode_spans(
                    _doc_axis(scores[inst, index], doc_len),
                    {
                        "dtype": "str",
                        "threshold": rel_threshold,
                        "include_confidence": include_confidence,
                        "include_spans": include_spans,
                    },
                    text,
                    maps,
                    threshold,
                    overlap,
                )
            )
        if len(sides) == 2 and sides[0] and sides[1]:
            if include_spans or include_confidence:
                instances.append({"head": sides[0], "tail": sides[1]})
            else:
                instances.append((sides[0], sides[1]))
    return instances


def _decode_structure_group(
    group, scores, count, words, text, maps, doc_len, prefix_len, threshold, include_confidence, include_spans, overlap
):
    instances = []
    prefix_tokens = list(words[:prefix_len])
    for inst in range(count):
        instance = {}
        for index, field in enumerate(group.fields):
            if field.kind == "choice":
                instance[field.name] = _decode_choice(
                    field, scores[inst, index, :prefix_len], prefix_tokens, threshold, include_confidence
                )
            else:
                field_threshold = field.threshold if field.threshold is not None else threshold
                instance[field.name] = decode_spans(
                    _doc_axis(scores[inst, index], doc_len),
                    {
                        "dtype": field.dtype or "list",
                        "threshold": field_threshold,
                        "validators": list(field.validators),
                        "include_confidence": include_confidence,
                        "include_spans": include_spans,
                    },
                    text,
                    maps,
                    threshold,
                    overlap,
                )
        if any(value is not None and value != [] for value in instance.values()):
            instances.append(instance)
    return instances


def _char_span_of(start, end, offset, start_map, end_map, text):
    """Map a half-open word span onto the normalized text."""
    token_start, token_end = start - offset, end - offset
    if not (0 <= token_start < token_end <= len(end_map)):
        return None
    char_start = int(start_map[token_start])
    char_end = int(end_map[token_end - 1])
    surface = text[char_start:char_end].strip()
    if not surface:
        return None
    return surface, char_start, char_end


def _dedupe_relation_edges(edges):
    """Collapse contained mentions and repeated head/tail text."""
    if len(edges) < 2:
        return edges

    def canonical_mentions(side):
        mentions = {(edge[side][1], edge[side][2]): edge[side] for edge in edges}
        canonical = {}
        for coordinates in mentions:
            start, end = coordinates
            containing = [
                candidate for candidate in mentions.values() if candidate[1] <= start and candidate[2] >= end
            ]
            canonical[coordinates] = max(
                containing, key=lambda candidate: (candidate[2] - candidate[1], -candidate[1])
            )
        return canonical

    head_canonical = canonical_mentions("head")
    tail_canonical = canonical_mentions("tail")
    exact = {}
    for edge in edges:
        head = head_canonical[(edge["head"][1], edge["head"][2])]
        tail = tail_canonical[(edge["tail"][1], edge["tail"][2])]
        normalized = {**edge, "head": head, "tail": tail}
        key = (head[1], head[2], tail[1], tail[2])
        previous = exact.get(key)
        if previous is None or edge["score"] > previous["score"]:
            exact[key] = normalized

    def semantic_text(value):
        return " ".join(value.casefold().split())

    def rank(candidate):
        _, head_start, head_end = candidate["head"]
        _, tail_start, tail_end = candidate["tail"]
        distance = max(head_start - tail_end, tail_start - head_end, 0)
        return (distance, -candidate["score"], head_start, tail_start)

    semantic = {}
    for edge in exact.values():
        key = (semantic_text(edge["head"][0]), semantic_text(edge["tail"][0]))
        previous = semantic.get(key)
        if previous is None or rank(edge) < rank(previous):
            semantic[key] = edge
    values = list(semantic.values())
    kept = []
    for edge in values:
        head_tokens = set(semantic_text(edge["head"][0]).split())
        tail_tokens = set(semantic_text(edge["tail"][0]).split())
        dominated = False
        for other in values:
            if other is edge:
                continue
            other_head = set(semantic_text(other["head"][0]).split())
            other_tail = set(semantic_text(other["tail"][0]).split())
            if (head_tokens < other_head and tail_tokens == other_tail) or (
                tail_tokens < other_tail and head_tokens == other_head
            ):
                dominated = True
                break
        if not dominated:
            kept.append(edge)
    return sorted(kept, key=lambda edge: (edge["head"][1], edge["tail"][1], -edge["score"]))


def _format_relation_edge(edge, include_confidence, include_spans):
    score = edge["score"]
    head, head_start, head_end = edge["head"]
    tail, tail_start, tail_end = edge["tail"]
    if include_spans:
        value = {
            "head": {"text": head, "start": head_start, "end": head_end},
            "tail": {"text": tail, "start": tail_start, "end": tail_end},
        }
        if include_confidence:
            value["head"]["confidence"] = score
            value["tail"]["confidence"] = score
        return value
    if include_confidence:
        return {"head": {"text": head, "confidence": score}, "tail": {"text": tail, "confidence": score}}
    return (head, tail)


def _decode_relation_pairs(
    groups,
    pairs,
    logits,
    text,
    start_map,
    end_map,
    prefix_len,
    threshold,
    include_confidence,
    include_spans,
    temperature,
):
    """Map relation pair logits onto character offsets."""
    if pairs is None or logits is None or not torch.is_tensor(pairs) or pairs.numel() == 0:
        return {}
    probabilities = torch.sigmoid(logits.detach().float().cpu() / temperature)
    relation_groups = [group for group in groups if group.task == "relations"]
    aliases = {f"{group.name}: {group.prompt}": group.name for group in relation_groups if group.prompt}
    edges: dict[str, list] = {}
    for pair_index, probability in enumerate(probabilities):
        relation_index = int(pairs[pair_index, 1])
        if relation_index < 0 or relation_index >= len(relation_groups):
            continue
        group = relation_groups[relation_index]
        relation_name = _result_key(group)
        relation_type = aliases.get(relation_name, relation_name)
        relation_threshold = group.threshold if group.threshold is not None else threshold
        score = float(probability.detach())
        if score < relation_threshold:
            continue
        head = _char_span_of(
            int(pairs[pair_index, 2]), int(pairs[pair_index, 3]), prefix_len, start_map, end_map, text
        )
        tail = _char_span_of(
            int(pairs[pair_index, 4]), int(pairs[pair_index, 5]), prefix_len, start_map, end_map, text
        )
        if head is None or tail is None:
            continue
        edges.setdefault(relation_type, []).append({"score": score, "head": head, "tail": tail})
    return {
        relation_type: [
            _format_relation_edge(edge, include_confidence, include_spans)
            for edge in _dedupe_relation_edges(relation_edges)
        ]
        for relation_type, relation_edges in edges.items()
    }


def _apply_relation_pairs(
    results,
    groups,
    span_groups,
    relation_pairs,
    relation_logits,
    text,
    start_map,
    end_map,
    prefix_len,
    threshold,
    include_confidence,
    include_spans,
    relation_temperature,
) -> None:
    """Write pair-decoded relations and keep empty lists for the rest."""
    paired = _decode_relation_pairs(
        groups,
        relation_pairs,
        relation_logits,
        text,
        start_map,
        end_map,
        prefix_len,
        threshold,
        include_confidence,
        include_spans,
        relation_temperature,
    )
    if paired or (torch.is_tensor(relation_pairs) and relation_pairs.numel()):
        results.update(paired)
        for group in span_groups:
            if group.task == "relations":
                results.setdefault(_result_key(group), [])


def _decode_span_groups(
    groups,
    tensors,
    counts,
    words,
    text,
    maps,
    doc_len,
    prefix_len,
    threshold,
    include_confidence,
    include_spans,
    overlap,
):
    results = {}
    for index, group in enumerate(groups):
        scores = tensors[index]
        count = int(counts[index]) if counts is not None else int(scores.shape[0])
        count = max(count, 0)
        scores = scores[:count]
        key = _result_key(group)
        if count <= 0:
            results[key] = [] if group.task in ("entities", "relations") else {}
            continue
        if scores.shape[-2] != len(words):
            raise ValueError(
                f"span word axis {scores.shape[-2]} != words {len(words)} (include the classification prefix)"
            )
        if group.task == "entities":
            decoded = _decode_entity_fields(
                group, _doc_axis(scores[0], doc_len), text, maps, threshold, include_confidence, include_spans, overlap
            )
            results[key] = [decoded] if decoded else []
        elif group.task == "relations":
            results[key] = _decode_relation_endpoints(
                group, scores, count, text, maps, doc_len, threshold, include_confidence, include_spans, overlap
            )
        else:
            results[key] = _decode_structure_group(
                group,
                scores,
                count,
                words,
                text,
                maps,
                doc_len,
                prefix_len,
                threshold,
                include_confidence,
                include_spans,
                overlap,
            )
    return results


def _decode_label_groups(groups, classification, threshold, temperature, words_prompt: bool = True) -> dict[str, Any]:
    if classification is None or not groups:
        return {}
    vectors = _classification_vectors(classification, len(groups))
    configs = [_classification_config(group) for group in groups]
    decoded = {}
    for group, vector, config in zip(groups, vectors, configs):
        prompt = group.tokens[2] if len(group.tokens) > 2 else group.name
        resolved = _resolve_classification_config(prompt, configs) if words_prompt else config
        if resolved is None:
            continue
        decoded[resolved["task"]] = _decode_classification_group(
            vector, resolved, threshold, temperature, activated=False
        )
    return decoded


def _apply_span_groups(
    results,
    span_groups,
    scores,
    counts,
    words,
    text,
    start_map,
    end_map,
    prefix_len,
    threshold,
    include_confidence,
    include_spans,
    overlap,
) -> None:
    """Decode span and choice fields into `results`."""
    tensors = _span_group_tensors(scores, len(span_groups), apply_sigmoid=True)
    if counts is not None and not isinstance(counts, (list, tuple)):
        counts = counts.detach().cpu().tolist()
    results.update(
        _decode_span_groups(
            span_groups,
            tensors,
            counts,
            words,
            text,
            {"start": start_map, "end": end_map},
            len(start_map),
            prefix_len,
            threshold,
            include_confidence,
            include_spans,
            overlap,
        )
    )


def decode_fields(
    groups: Sequence[FieldGroup],
    scores,
    text: str,
    maps: Mapping[str, Sequence[int]],
    *,
    words: Sequence[str] | None = None,
    prefix_len: int = 0,
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False,
    overlap: str | None = None,
    temperature: float = 1.0,
    counts=None,
    classification=None,
    relation_pairs=None,
    relation_logits=None,
    relation_temperature: float = 1.0,
) -> dict[str, Any]:
    """Decode one sample from field groups.

    Span fields call `decode_spans`. Choice fields call `format_span`. Label
    fields use the classification decoder. Relations use pair logits when they
    are present and per-endpoint `decode_spans` otherwise.

    Args:
        groups:
            Field groups for this sample, or only the label groups.
        scores:
            Per-group span logits aligned with the non-label groups, or `None`.
        text (`str`):
            Normalized document text.
        maps:
            `start` and `end` character offsets for document words.
        words (`Sequence[str]`, *optional*):
            Encoded words, including the choice prefix.
        prefix_len (`int`, *optional*, defaults to 0):
            Choice-prefix length inside `words`.
        threshold (`float`, *optional*, defaults to 0.5):
            Score cutoff.
        include_confidence (`bool`, *optional*, defaults to `False`):
            Keep scores.
        include_spans (`bool`, *optional*, defaults to `False`):
            Keep character offsets.
        overlap (`str`, *optional*):
            Overlap policy passed to `decode_spans`.
        temperature (`float`, *optional*, defaults to 1.0):
            Classification temperature.
        counts:
            Optional instance counts aligned with span groups.
        classification:
            Classification logits aligned with label groups.
        relation_pairs:
            Relation edges shaped `[pairs, 6]`.
        relation_logits:
            One score per relation pair.
        relation_temperature (`float`, *optional*, defaults to 1.0):
            Divisor applied before the relation sigmoid.
    """
    span_groups = [group for group in groups if group.task != "classifications"]
    label_groups = [group for group in groups if group.task == "classifications"]
    start_map = list(maps.get("start") or [])
    end_map = list(maps.get("end") or [])
    word_list = list(words or [])
    results: dict[str, Any] = {}
    if scores is not None and span_groups:
        _apply_span_groups(
            results,
            span_groups,
            scores,
            counts,
            word_list,
            text,
            start_map,
            end_map,
            prefix_len,
            threshold,
            include_confidence,
            include_spans,
            overlap,
        )
    if relation_pairs is not None and relation_logits is not None:
        _apply_relation_pairs(
            results,
            groups,
            span_groups,
            relation_pairs,
            relation_logits,
            text,
            start_map,
            end_map,
            prefix_len,
            threshold,
            include_confidence,
            include_spans,
            relation_temperature,
        )
    results.update(_decode_label_groups(label_groups, classification, threshold, temperature))
    return results


def _group_dict(group: FieldGroup) -> dict[str, Any]:
    options: dict[str, Any] = {
        "fields": {
            field.name: {
                "dtype": field.dtype,
                "threshold": field.threshold,
                "validators": list(field.validators),
                "choices": list(field.choices),
            }
            for field in group.fields
        }
    }
    if group.endpoints:
        options["endpoints"] = group.endpoints
    if group.threshold is not None and group.task == "relations":
        options["threshold"] = group.threshold
    if group.record is not None:
        options["record"] = {"mode": group.record.mode, "anchor": group.record.anchor}
    if group.task == "classifications":
        options["classification"] = {
            "multi_label": group.multi_label,
            "cls_threshold": group.threshold,
            "class_act": group.activation,
        }
    if group.attributes:
        options["attributes"] = {
            spec.name: {
                "labels": list(spec.labels),
                "applies_to": None if spec.applies_to is None else list(spec.applies_to),
                "multi_label": spec.multi_label,
                "threshold": spec.threshold,
            }
            for spec in group.attributes
        }
    return {
        "task_type": group.task,
        "name": _result_key(group),
        "prompt": group.tokens[2] if len(group.tokens) > 2 else "",
        "fields": [field.name for field in group.fields],
        "options": options,
    }


def _record_spec_dict(group: FieldGroup) -> dict[str, Any]:
    record = group.record
    return {
        "task_index": record.task_index,
        "task_name": group.name,
        "mode": record.mode,
        "fields": [
            {
                "query_id": field.query_id,
                "name": field.name,
                "cardinality": field.cardinality,
                "is_anchor": field.is_anchor,
                "exclusive": field.exclusive,
            }
            for field in record.fields
        ],
        "anchor_query_id": record.anchor_query_id,
    }


def _decoder_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Dict view current `decode_*` functions read. Built from `FieldGroup`s."""
    groups = tuple(row.get("groups") or ())
    if groups and not isinstance(groups[0], FieldGroup):
        return dict(row)
    entity = next((group for group in groups if group.task == "entities"), None)
    field_metadata = {}
    cls_fields = {}
    for group in groups:
        if group.task != "json_structures":
            continue
        for field in group.fields:
            entry = {}
            if field.dtype:
                entry["dtype"] = field.dtype
            if field.threshold is not None:
                entry["threshold"] = field.threshold
            if field.choices:
                entry["choices"] = list(field.choices)
            if field.validators:
                entry["validators"] = list(field.validators)
            if entry:
                field_metadata[f"{group.name}.{field.name}"] = entry
            if field.kind == "choice":
                cls_fields[f"{group.name}.{field.name}"] = list(field.choices)
    entity_metadata = {}
    if entity is not None:
        for field in entity.fields:
            entry = {"dtype": field.dtype}
            if field.threshold is not None:
                entry["threshold"] = field.threshold
            if field.validators:
                entry["validators"] = list(field.validators)
            if field.choices:
                entry["choices"] = list(field.choices)
            entity_metadata[field.name] = entry
    relation_metadata = {}
    relation_descriptions = {}
    for group in groups:
        if group.task != "relations":
            continue
        if group.threshold is not None:
            relation_metadata.setdefault(group.name, {})["threshold"] = group.threshold
        if group.prompt:
            relation_descriptions[group.name] = group.prompt
    attribute_groups = {}
    attribute_labels: set[str] = set()
    attribute_prompts = {}
    if entity is not None:
        attribute_labels = set(entity.attribute_labels)
        attribute_prompts = dict(entity.attribute_prompts)
        for spec in entity.attributes:
            attribute_groups[spec.name] = {
                "labels": list(spec.labels),
                "multi_label": spec.multi_label,
                "threshold": spec.threshold,
            }
            if spec.applies_to is not None:
                attribute_groups[spec.name]["applies_to"] = list(spec.applies_to)
    return {
        "groups": [_group_dict(group) for group in groups],
        "text": row.get("text") or "",
        "start": list(row.get("start") or []),
        "end": list(row.get("end") or []),
        "prefix_len": int(row.get("prefix_len") or 0),
        "words": list(row.get("words") or []),
        "architecture": row.get("architecture"),
        "schema": row.get("schema") or {},
        "constraints": list(row.get("constraints") or []),
        "classifications": [_classification_config(group) for group in groups if group.task == "classifications"],
        "record_specs": [_record_spec_dict(group) for group in groups if group.record is not None],
        "field_metadata": field_metadata,
        "entity_metadata": entity_metadata,
        "relation_metadata": relation_metadata,
        "relation_descriptions": relation_descriptions,
        "entity_order": [field.name for field in entity.fields] if entity is not None else [],
        "relation_order": [group.name for group in groups if group.task == "relations"],
        "classification_tasks": [group.name for group in groups if group.task == "classifications"],
        "cls_fields": cls_fields,
        "entity_attribute_groups": attribute_groups,
        "entity_attribute_labels": attribute_labels,
        "entity_attribute_prompt_labels": attribute_prompts,
        "field_orders": {},
    }


def _metadata_rows(metadata: Any) -> list[Mapping[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, Mapping):
        nested = "metadata" in metadata and "groups" not in metadata
        metadata = metadata["metadata"] if nested else [metadata]
    return [item for item in metadata if isinstance(item, Mapping)]


def _row_groups(metadata: Any) -> tuple[FieldGroup, ...]:
    rows = _metadata_rows(metadata)
    if not rows:
        return ()
    groups = rows[0].get("groups") or ()
    if groups and isinstance(groups[0], FieldGroup):
        return tuple(groups)
    return ()


def _split_boundary(outputs: Mapping[str, Any], boundary: Mapping[str, Any], batch_size: int) -> list[dict[str, Any]]:
    candidates = boundary.get("candidates")
    samples = []
    for index in range(batch_size):
        sample = {
            "candidates": type(candidates)(
                indices=candidates.indices[index : index + 1],
                proposal_logits=None
                if candidates.proposal_logits is None
                else candidates.proposal_logits[index : index + 1],
                pair_logits=candidates.pair_logits[index : index + 1],
                valid_mask=candidates.valid_mask[index : index + 1],
                query_mask=candidates.query_mask[index : index + 1],
                candidate_states=(
                    None if candidates.candidate_states is None else candidates.candidate_states[index : index + 1]
                ),
            ),
            "pair_logits": candidates.pair_logits[index],
            "null_logits": None if boundary.get("null_logits") is None else boundary["null_logits"][index],
            "count_log_rates": None if boundary.get("count_log_rates") is None else boundary["count_log_rates"][index],
        }
        if outputs.get("classification_logits") is not None:
            sample["classification_logits"] = outputs["classification_logits"][index]
        text_states = outputs.get("text_states")
        query_states = outputs.get("query_states")
        if text_states is not None:
            sample["text_states"] = text_states[index : index + 1]
        if query_states is not None:
            sample["query_states"] = query_states[index : index + 1]
        relation_pairs = outputs.get("relation_pairs")
        relation_logits = outputs.get("relation_logits")
        if relation_pairs is not None and relation_logits is not None and relation_pairs.numel():
            keep = relation_pairs[:, 0] == index
            sample["relation_pairs"] = relation_pairs[keep]
            sample["relation_logits"] = relation_logits[keep]
            sample["relation_temperature"] = outputs.get("relation_temperature")
        if outputs.get("record_logits") is not None:
            sample["record_logits"] = outputs["record_logits"][index]
        samples.append(sample)
    return samples


def _split_outputs(outputs: Any, batch_size: int) -> list[Any]:
    """One mapping per batch row. Boundary candidates stay on that row."""
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != batch_size:
            raise ValueError(f"outputs length ({len(outputs)}) != metadata length ({batch_size})")
        return list(outputs)
    if not isinstance(outputs, Mapping):
        try:
            outputs = dict(outputs.items())
        except (AttributeError, TypeError):
            raise TypeError("outputs must be a mapping or a sequence of sample outputs") from None
    boundary = outputs.get("boundary")
    if isinstance(boundary, Mapping) and boundary.get("candidates") is not None:
        return _split_boundary(outputs, boundary, batch_size)
    keys = (
        "span_logits",
        "counts",
        "classification_logits",
        "record_logits",
        "relation_pairs",
        "relation_logits",
        "pair_logits",
        "grouped_candidates",
    )
    present = [key for key in keys if key in outputs and outputs[key] is not None]
    if not present:
        raise ValueError(
            "outputs need span_logits (count, fields, words, width) "
            "and/or classification_logits (num_labels,). "
            "Boundary pair_logits require candidates on the forward output."
        )
    return [{key: outputs[key][index] for key in present} for index in range(batch_size)]


def _decode_sample(sample, row, threshold, include_confidence, include_spans, overlap, temperature) -> dict[str, Any]:
    groups = tuple(row.get("groups") or ())
    if groups and not isinstance(groups[0], FieldGroup):
        groups = ()
    view = _decoder_view(row)
    if sample.get("pair_logits") is not None or sample.get("grouped_candidates") is not None:
        decoded = decode_boundary(
            sample,
            view,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap,
            temperature=temperature,
        )
        labels = [group for group in groups if group.task == "classifications"]
        decoded.update(
            decode_fields(
                labels,
                None,
                view.get("text") or "",
                {"start": view.get("start") or [], "end": view.get("end") or []},
                words=view.get("words") or [],
                prefix_len=int(view.get("prefix_len") or 0),
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
                overlap=overlap,
                temperature=temperature,
                classification=sample.get("classification_logits"),
            )
        )
        return decoded
    if sample.get("record_logits") is not None:
        return decode_records(
            sample,
            view,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            overlap_policy=overlap,
        )
    temperature_rel = sample.get("relation_temperature") or 1.0
    return decode_fields(
        groups,
        sample.get("span_logits"),
        view.get("text") or "",
        {"start": view.get("start") or [], "end": view.get("end") or []},
        words=view.get("words") or [],
        prefix_len=int(view.get("prefix_len") or 0),
        threshold=threshold,
        include_confidence=include_confidence,
        include_spans=include_spans,
        overlap=overlap,
        temperature=temperature,
        counts=sample.get("counts"),
        classification=sample.get("classification_logits"),
        relation_pairs=sample.get("relation_pairs"),
        relation_logits=sample.get("relation_logits"),
        relation_temperature=float(temperature_rel),
    )


def _task_logit_map(sample: Mapping[str, Any], classifications: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    vectors = sample.get("classification_logits")
    if vectors is None:
        vectors = sample.get("classification_probs")
    if isinstance(vectors, Mapping):
        return vectors
    if torch.is_tensor(vectors) and vectors.ndim == 1:
        grouped = [vectors]
    elif torch.is_tensor(vectors) and vectors.ndim == 2:
        grouped = [vectors[index] for index in range(vectors.shape[0])]
    else:
        grouped = list(vectors or [])
    if len(grouped) != len(classifications):
        raise ValueError(f"expected {len(classifications)} classification rows, got {len(grouped)}")
    logits = {}
    for entry, vector in zip(classifications, grouped):
        flat = vector.detach().float().cpu().reshape(-1) if torch.is_tensor(vector) else vector
        values = flat.tolist() if torch.is_tensor(flat) else list(flat)
        logits[entry["task"]] = {label: float(values[index]) for index, label in enumerate(entry["labels"])}
    return logits


def _output_value(value: Any, name: str):
    if isinstance(value, Mapping) and name in value:
        return value[name]
    return None


def _states_from(value: Any):
    if value is None:
        return None
    text_states = _output_value(value, "text_states")
    text_mask = _output_value(value, "text_mask")
    if text_mask is None:
        text_mask = _output_value(value, "text_word_mask")
    query_states = _output_value(value, "query_states")
    query_mask = _output_value(value, "query_mask")
    if query_mask is None:
        query_mask = _output_value(value, "query_marker_mask")
    tensors = (text_states, text_mask, query_states, query_mask)
    if any(item is None or not torch.is_tensor(item) for item in tensors):
        return None
    return tensors


def _cached_states(outputs: Any):
    states = _states_from(outputs)
    if states is not None:
        return states
    return _states_from(_output_value(outputs, "boundary"))


def _batch_states(text_states, text_mask, query_states, query_mask):
    if text_states.dim() == 2:
        text_states = text_states.unsqueeze(0)
    if text_mask.dim() == 1:
        text_mask = text_mask.unsqueeze(0)
    if query_states.dim() == 2:
        query_states = query_states.unsqueeze(0)
    if query_mask.dim() == 1:
        query_mask = query_mask.unsqueeze(0)
    return text_states[:1], text_mask[:1], query_states[:1], query_mask[:1]


def _entity_query_index(groups: Sequence[FieldGroup]) -> dict[str, int]:
    index = {}
    cursor = 0
    for group in groups:
        if group.task == "classifications":
            continue
        for field in group.fields:
            if group.task == "entities":
                index[field.name] = cursor
            cursor += 1
    return index


def _entity_nodes(decoded: Mapping[str, Any]):
    entities = decoded.get("entities") if isinstance(decoded, Mapping) else None
    if isinstance(entities, Mapping):
        for name, value in entities.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield name, item
            elif isinstance(value, dict) and "text" in value:
                yield name, value
        return
    if isinstance(entities, list):
        for item in entities:
            if isinstance(item, dict):
                yield str(item.get("type", item.get("label", ""))), item


def _word_bounds(char_start: int, char_end: int, start_map, end_map, prefix: int):
    """Map a character span onto half-open word indices, including the choice prefix."""
    word_start = None
    word_end = None
    for index, (start, end) in enumerate(zip(start_map, end_map)):
        if int(end) <= char_start or int(start) >= char_end:
            continue
        if word_start is None:
            word_start = index
        word_end = index
    if word_start is None or word_end is None:
        return None
    return word_start + prefix, word_end + 1 + prefix


def _assign_entity_attributes(logits: torch.Tensor, column: int, rows: Sequence[tuple[str, int]], spec: AttributeSpec):
    labels, indices = zip(*rows)
    values = logits[list(indices), column]
    if spec.multi_label:
        probabilities = torch.sigmoid(values)
        return [
            {"label": label, "confidence": float(probabilities[index].item())}
            for index, label in enumerate(labels)
            if float(probabilities[index]) >= spec.threshold
        ]
    probabilities = torch.softmax(values, dim=-1)
    best = int(probabilities.argmax())
    return {"label": labels[best], "confidence": float(probabilities[best].item())}


@auto_docstring
class Gliner2Processor(ProcessorMixin):
    """Schema-conditioned processor for GLiNER2.

    Tokenization matches inference collation: lowercase whitespace words, each
    word tokenized alone, structural markers kept as single special tokens, no
    CLS or SEP wrapping, and first-subword indexes for token pooling.
    """

    valid_processor_kwargs = Gliner2ProcessorKwargs
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

    def __init__(self, tokenizer=None, word_splitter=None, token_pooling: str = "first", **kwargs):
        r"""
        word_splitter (`str` or `Callable`, *optional*, defaults to `"whitespace"`):
            How document text is split into words. `"whitespace"` or `"char"`, or a
            callable returning `(words, starts, ends)`. Saved with the processor.
        token_pooling (`str`, *optional*, defaults to `"first"`):
            Subword pooling mode, one of `"first"`, `"mean"`, or `"max"`. Saved with
            the processor. Any other value raises `ValueError`.
        """
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if token_pooling not in _POOLING:
            raise ValueError(f"token_pooling must be one of {_POOLING}, got {token_pooling!r}.")
        super().__init__(tokenizer, **kwargs)
        self.token_pooling = token_pooling
        self.word_splitter = "whitespace" if word_splitter is None else word_splitter
        self._splitter = _resolve_word_splitter(self.word_splitter)
        self._piece_ids: dict[str, tuple[int, ...]] = {}
        self._register_special_tokens()

    def _register_special_tokens(self) -> None:
        self.tokenizer.add_special_tokens({"extra_special_tokens": list(SPECIAL_TOKENS)})
        for token in list(SPECIAL_TOKENS) + ["(", ")", ",", "|"]:
            self._encode_piece(token)

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        """Cache one piece's ids for words the batched call drops."""
        cached = self._piece_ids.get(piece)
        if cached is None:
            cached = tuple(int(item) for item in self.tokenizer.encode(piece, add_special_tokens=False))
            self._piece_ids[piece] = cached
        return cached

    def _encode_words(self, words: Sequence[str]) -> tuple[list[int], list[int], set[int]]:
        """Encode pre-split words in one call and return first-subword indexes."""
        if not words:
            return [], [], set()
        encoding = self.tokenizer(list(words), is_split_into_words=True, add_special_tokens=False)
        ids = [int(item) for item in encoding["input_ids"]]
        first: list[int] = []
        empty: set[int] = set()
        cursor = 0
        shift = 0
        for index, word in enumerate(words):
            span = encoding.word_to_tokens(index)
            missing = span is None or int(span.start) == int(span.end)
            if not missing:
                first.append(int(span.start) + shift)
                cursor = int(span.end)
                continue
            piece = self._encode_piece(word)
            at = cursor + shift
            if piece:
                ids = ids[:at] + list(piece) + ids[at:]
                shift += len(piece)
            else:
                empty.add(index)
            first.append(at)
        return ids, first, empty

    def _split_words(self, text: str) -> tuple[list[str], list[int], list[int]]:
        words, starts, ends = [], [], []
        for token, start, end in self._splitter(text, True):
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
        input_ids, first_positions, empty_words = self._encode_words(combined)
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
            subword_pos = first_positions[orig_idx]
            if seg_type == "text" and orig_idx != last_text_orig:
                last_text_orig = orig_idx
                if orig_idx in empty_words:
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
        self, text: str, schema: Any, max_len: int | None, architecture: str, labels: Any = None
    ) -> dict:
        schema = _canonicalize_schema(schema)
        label_spec = None if labels is None else _canonicalize_labels(labels, schema)
        groups, prefix = compile_schema(schema)
        schema_tokens = [list(group.tokens) for group in groups]
        task_types = [group.task for group in groups]
        text = _normalize_text(text)
        words, starts, ends = self._split_words(text)
        if max_len is not None:
            words, starts, ends = words[:max_len], starts[:max_len], ends[:max_len]
        text_tokens = list(prefix) + words
        formatted = self._format_input(schema_tokens, text_tokens)
        grid = WordGrid(tuple(words), tuple(starts), tuple(ends), text, len(prefix), tuple(prefix))
        return {
            "input_ids": formatted["input_ids"],
            "text_word_first_positions": formatted["text_word_first_positions"],
            "schema_special_positions": formatted["schema_special_positions"],
            "task_types": list(task_types),
            "words": text_tokens,
            "groups": tuple(groups),
            "grid": grid,
            "labels": label_spec,
            "record_specs": [_record_spec_dict(group) for group in groups if group.record is not None],
            "schema_row": {
                "groups": tuple(groups),
                "text": text,
                "start": starts,
                "end": ends,
                "prefix_len": len(prefix),
                "words": text_tokens,
                "architecture": architecture,
                "schema": schema,
                "constraints": list(schema.get("constraints") or []),
            },
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

    def _pack_routes(self, routes: Sequence[Sequence[tuple[int, ...]]], batch_size: int, width_of: int = 2):
        """Pad route tuples into one column per coordinate plus a mask."""
        width = max((len(row) for row in routes), default=0)
        columns = [torch.zeros((batch_size, width), dtype=torch.long) for _ in range(width_of)]
        mask = torch.zeros((batch_size, width), dtype=torch.bool)
        for index, values in enumerate(routes):
            if not values:
                continue
            count = len(values)
            stacked = torch.tensor(list(values), dtype=torch.long)
            for column, tensor in enumerate(columns):
                tensor[index, :count] = stacked[:, column]
            mask[index, :count] = True
        return (*columns, mask)

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

    def _collate(self, records: Sequence[Mapping[str, Any]], architecture: str, max_gold_per_query: int | None = None):
        batch_size = len(records)
        id_rows = [torch.tensor(record["input_ids"], dtype=torch.long) for record in records]
        mask_rows = [torch.ones(len(record["input_ids"]), dtype=torch.long) for record in records]
        if not id_rows:
            input_ids = torch.zeros((0, 0), dtype=torch.long)
            attention_mask = torch.zeros((0, 0), dtype=torch.long)
        else:
            input_ids = pad_sequence(id_rows, batch_first=True, padding_value=0)
            attention_mask = pad_sequence(mask_rows, batch_first=True, padding_value=0)
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
        query_marker_indices, query_group_index, query_marker_mask = self._pack_routes(query_routes, batch_size)
        cls_marker_indices, cls_group_index, cls_marker_mask = self._pack_routes(cls_routes, batch_size)
        prompt_marker_indices, prompt_group_index, prompt_marker_mask = self._pack_routes(prompt_routes, batch_size)
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
        relation_head_index, relation_tail_index, relation_group_index, relation_mask = self._pack_routes(
            self._relation_pairs(records), batch_size, width_of=3
        )
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
            "metadata": [dict(record["schema_row"]) for record in records],
        }
        if architecture == "boundary":
            data.update(self._pack_record_tensors([record["record_specs"] for record in records]))
        if any(record.get("labels") is not None for record in records):
            if any(record.get("labels") is None for record in records):
                raise ValueError("labels must be provided for every text in the batch")
            data["targets"] = build_targets(
                [record["groups"] for record in records],
                [record["labels"] for record in records],
                [record["grid"] for record in records],
                max_gold_per_query=max_gold_per_query,
                splitter=self._splitter,
            )
        return data

    @auto_docstring
    def __call__(
        self,
        text: TextInput,
        schema: Gliner2Schema | Sequence[Gliner2Schema] | None = None,
        return_tensors: str = "pt",
        max_len: int | None = None,
        architecture: str = "span",
        labels: Gliner2Labels | Sequence[Gliner2Labels] | None = None,
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
        """
        merged = self._merge_kwargs(
            Gliner2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.__dict__.get("init_kwargs"),
            max_len=max_len,
            architecture=architecture,
            labels=labels,
            **kwargs,
        )
        text_kwargs = merged["text_kwargs"]
        max_len = text_kwargs.get("max_len", max_len)
        architecture = text_kwargs.get("architecture", architecture)
        labels = text_kwargs.get("labels", labels)
        max_gold_per_query = text_kwargs.get("max_gold_per_query")
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
            self._transform_one(item, item_schema, max_len, architecture, label_rows[index])
            for index, (item, item_schema) in enumerate(zip(texts, schemas))
        ]
        skipped = ["metadata"]
        if any(record.get("labels") is not None for record in records):
            skipped.append("targets")
        return BatchFeature(
            self._collate(records, architecture, max_gold_per_query),
            tensor_type="pt",
            skip_tensor_conversion=skipped,
        )

    def chunk_words(
        self, words: str | Sequence[Any], chunk_size: int = 384, chunk_overlap: int = 64
    ) -> list[TextChunk]:
        """Split text or words into overlapping windows."""
        return chunk_words(words, chunk_size=chunk_size, chunk_overlap=chunk_overlap, word_splitter=self._splitter)

    def windows(self, text: str, chunk_size: int | None, chunk_overlap: int | None) -> list[TextChunk]:
        """One chunk per word window. A short text stays a single window.

        Args:
            text (`str`):
                Document text.
            chunk_size (`int`, *optional*):
                Word window length. Non-positive values keep the whole document.
            chunk_overlap (`int`, *optional*):
                Words shared by neighboring windows.
        """
        whole = TextChunk(text=text, start_char=0, end_char=len(text), start_word=0, end_word=0)
        if chunk_size is None or chunk_size <= 0:
            return [whole]
        overlap = 0 if chunk_overlap is None or chunk_overlap < 0 else chunk_overlap
        if overlap >= chunk_size:
            overlap = chunk_size - 1
        chunks = self.chunk_words(text, chunk_size=chunk_size, chunk_overlap=overlap)
        if len(chunks) <= 1:
            return [whole]
        return list(chunks)

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

    def scalar_entity_labels(self, metadata: Any) -> set[str]:
        """Entity names declared with a non-list dtype.

        Args:
            metadata:
                Processor metadata for one window or a batch.
        """
        labels = set()
        for group in _row_groups(metadata):
            if group.task != "entities":
                continue
            for field in group.fields:
                if field.dtype != "list":
                    labels.add(field.name)
        return labels

    def can_rescore_attributes(self, outputs: Any, metadata: Any, model: Any) -> bool:
        """Return whether this window can rescore attributes without another encoder pass.

        Args:
            outputs:
                Forward output for the window.
            metadata:
                Processor metadata for the window.
            model:
                `Gliner2ForSchemaExtraction` instance.
        """
        try:
            head = model.boundary_head
        except AttributeError:
            return False
        if head is None:
            return False
        entity = next((group for group in _row_groups(metadata) if group.task == "entities"), None)
        return bool(entity and entity.attributes) and _cached_states(outputs) is not None

    def rescore_attributes(self, result: dict[str, Any], model: Any, outputs: Any, metadata: Any) -> dict[str, Any]:
        """Write attribute labels onto kept entity spans using `score_spans`.

        Args:
            result (`dict`):
                One window's formatted decode. Spans need character offsets.
            model:
                Model whose `score_spans` and `boundary_config.pair_temperature` are used.
            outputs:
                Forward output that cached word and query states.
            metadata:
                Processor metadata with entity attribute groups and word maps.
        """
        if not self.can_rescore_attributes(outputs, metadata, model):
            return result
        cached = _cached_states(outputs)
        if cached is None:
            return result
        groups = _row_groups(metadata)
        entity = next(group for group in groups if group.task == "entities")
        text_states, text_mask, query_states, query_mask = _batch_states(*cached)
        query_of = _entity_query_index(groups)
        prompts = dict(entity.attribute_prompts)
        attribute_rows = []
        for spec in entity.attributes:
            for label in spec.labels:
                query_id = query_of.get(prompts.get(label, label))
                if query_id is not None and query_id < query_states.shape[1]:
                    attribute_rows.append((label, query_id))
        attribute_rows = list(dict.fromkeys(attribute_rows))
        hidden = set(entity.attribute_labels)
        nodes = [(name, item) for name, item in _entity_nodes(result) if name not in hidden]
        row = _metadata_rows(metadata)[0]
        start_map = list(row.get("start") or [])
        end_map = list(row.get("end") or [])
        prefix = int(row.get("prefix_len") or 0)
        located = []
        for name, item in nodes:
            if "start" not in item or "end" not in item:
                continue
            bounds = _word_bounds(int(item["start"]), int(item["end"]), start_map, end_map, prefix)
            if bounds is not None:
                located.append((name, item, bounds))
        unique_pairs = list(dict.fromkeys(bounds for _, _, bounds in located))
        if not attribute_rows or not unique_pairs:
            return result
        device = model.device
        with torch.no_grad():
            text_states = text_states.to(device)
            text_mask = text_mask.to(device)
            query_ids = torch.tensor([query_id for _, query_id in attribute_rows], dtype=torch.long, device=device)
            query_states = query_states.to(device).index_select(1, query_ids)
            query_mask = query_mask.to(device).index_select(1, query_ids)
            pairs = torch.tensor(unique_pairs, dtype=torch.long, device=device)
            indices = pairs.view(1, 1, len(unique_pairs), 2).expand(1, len(attribute_rows), len(unique_pairs), 2)
            logits = model.score_spans(text_states, text_mask, query_states, query_mask, indices.contiguous())
        logits = logits[0].detach().float().cpu()
        boundary = model.config.boundary_config
        pair_temperature = _output_value(boundary, "pair_temperature") if isinstance(boundary, Mapping) else None
        if pair_temperature is None:
            try:
                pair_temperature = boundary.pair_temperature
            except AttributeError:
                pair_temperature = 1.0
        logits = logits / float(pair_temperature or 1.0)
        row_of = {label: row_index for row_index, (label, _) in enumerate(attribute_rows)}
        column_of = {pair: index for index, pair in enumerate(unique_pairs)}
        for name, item, bounds in located:
            column = column_of[bounds]
            for spec in entity.attributes:
                if spec.applies_to is not None and name not in spec.applies_to:
                    continue
                present = [(label, row_of[label]) for label in spec.labels if label in row_of]
                if not present:
                    continue
                item[spec.name] = _assign_entity_attributes(logits, column, present, spec)
        return result

    def _names(self, metadata: Any, task: str) -> list[str]:
        groups = _row_groups(metadata)
        if groups:
            return [group.name for group in groups if group.task == task]
        view = _decoder_view(_metadata_rows(metadata)[0]) if _metadata_rows(metadata) else {}
        if task == "relations":
            return list(view.get("relation_order") or [])
        return list(view.get("classification_tasks") or [])

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

        ``metadata`` is ``encoding["metadata"]``. Each sample carries `FieldGroup`
        objects plus char ``start``/``end``, ``words``, and ``text``.

        Expected ``outputs`` keys, per sample or batched as lists:

        - ``span_logits``: pre-sigmoid scores shaped ``(count, fields, words, width)``.
          One tensor per non-classification group, in group order.
          ``words`` includes the classification-choice prefix.
        - ``counts``: optional instance counts. Defaults to ``span_logits`` size 0.
        - ``classification_logits``: one ``(num_labels,)`` vector per
          classification group.
        - ``relation_pairs`` and ``relation_logits``: typed relation edges scored
          in `forward`.
        - ``record_logits``: record-head logits scored in `forward`.

        Relations and JSON structures are decoded from the same span scores.
        Boundary ``pair_logits`` / ``grouped_candidates`` call ``decode_boundary``.
        ``record_logits`` without boundary candidates call ``decode_records``.

        Args:
            outputs:
                Model output or a list of per-sample outputs.
            metadata:
                Processor metadata aligned with `outputs`.
            threshold (`float`, *optional*, defaults to 0.5):
                Score cutoff.
            include_confidence (`bool`, *optional*, defaults to `False`):
                Keep scores.
            include_spans (`bool`, *optional*, defaults to `False`):
                Keep character offsets.
            overlap_policy (`str`, *optional*):
                Overlap rule. Boundary defaults to `disallow` when unset.
            temperature (`float`, *optional*, defaults to 1.0):
                Classification temperature.
        """
        rows = _metadata_rows(metadata)
        samples = _split_outputs(outputs, len(rows))
        formatted = []
        for sample_out, meta in zip(samples, rows):
            chosen = overlap_policy if overlap_policy is not None else meta.get("_overlap_policy")
            if chosen is None and meta.get("architecture") == "boundary":
                chosen = "disallow"
            policy = None if chosen is None else normalize_overlap_policy(chosen)
            raw = _decode_sample(sample_out, meta, threshold, include_confidence, include_spans, policy, temperature)
            formatted.append(
                format_results(
                    raw,
                    include_confidence=include_confidence,
                    requested_relations=self._names(meta, "relations"),
                    classification_tasks=self._names(meta, "classifications"),
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

        ``decoder="beam"`` or ``"exact"``, and non-empty schema constraints, call
        ``decode_classification``. Other decoders score each label group locally
        and do not decode span fields.

        Args:
            outputs:
                Model output with classification logits.
            metadata:
                Processor metadata aligned with `outputs`.
            threshold (`float`, *optional*, defaults to 0.5):
                Candidate threshold for the constrained decoder.
            include_confidence (`bool`, *optional*, defaults to `False`):
                Keep scores.
            temperature (`float`, *optional*, defaults to 1.0):
                Classification temperature.
            decoder (`str`, *optional*, defaults to `"independent"`):
                `independent`, `auto`, `beam`, or `exact`.
        """
        rows = _metadata_rows(metadata)
        views = [_decoder_view(row) for row in rows]
        needs_solver = decoder in ("beam", "exact") or any(view.get("constraints") for view in views)
        if needs_solver and decoder != "independent":
            solved = []
            samples = _split_outputs(outputs, len(rows))
            for sample, view in zip(samples, views):
                classifications = list(view.get("classifications") or [])
                schema = {"classifications": classifications, "constraints": list(view.get("constraints") or [])}
                solved.append(
                    decode_classification(
                        _task_logit_map(sample, classifications),
                        schema,
                        temperature,
                        text=view.get("text") or "",
                        decoder=decoder,
                        include_confidence=include_confidence,
                        candidate_threshold=threshold,
                    )
                )
            return solved
        samples = _split_outputs(outputs, len(rows))
        results = []
        for sample_out, meta in zip(samples, rows):
            labels = [group for group in _row_groups(meta) if group.task == "classifications"]
            raw = decode_fields(
                labels,
                None,
                meta.get("text") or "",
                {"start": meta.get("start") or [], "end": meta.get("end") or []},
                threshold=threshold,
                include_confidence=include_confidence,
                temperature=temperature,
                classification=sample_out.get("classification_logits"),
            )
            results.append(
                format_results(
                    raw,
                    include_confidence=include_confidence,
                    classification_tasks=self._names(meta, "classifications"),
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
        """Decode a joint entity and relation payload.

        ``optimizer="beam"``, ``"greedy"``, and ``"exact"`` call ``decode_joint``.
        ``optimizer="independent"`` decodes each span group on its own scores.

        Args:
            outputs:
                Model output.
            metadata:
                Processor metadata aligned with `outputs`.
            threshold (`float`, *optional*, defaults to 0.5):
                Score cutoff.
            include_confidence (`bool`, *optional*, defaults to `False`):
                Keep scores.
            include_spans (`bool`, *optional*, defaults to `False`):
                Keep character offsets.
            optimizer (`str`, *optional*, defaults to `"independent"`):
                `independent`, `auto`, `beam`, `greedy`, or `exact`.
            overlap_policy (`str`, *optional*):
                Overlap rule forwarded to the decoder.
        """
        if optimizer in ("beam", "exact", "greedy"):
            rows = _metadata_rows(metadata)
            views = [_decoder_view(row) for row in rows]
            return decode_joint(
                outputs,
                views,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
                optimizer="beam" if optimizer == "exact" else optimizer,
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


__all__ = [
    "Gliner2Processor",
    "Gliner2Schema",
    "Gliner2Labels",
]
