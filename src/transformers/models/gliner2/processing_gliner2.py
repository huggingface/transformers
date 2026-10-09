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
from typing import Any, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs
from ...utils import auto_docstring
from .decoding_gliner2 import (
    DecodeOptions,
    GroupRecord,
    RecordField,
    _doc_axis,
    decode_boundary,
    decode_constrained_classification,
    decode_joint_sample,
    decode_relations,
    decode_spans,
    format_span,
    normalize_overlap_policy,
    resolve_overlaps,
)


logger = logging.getLogger(__name__)

SEP_STRUCT, SEP_TEXT, P_TOKEN, C_TOKEN, E_TOKEN, R_TOKEN, L_TOKEN = (
    "[SEP_STRUCT]",
    "[SEP_TEXT]",
    "[P]",
    "[C]",
    "[E]",
    "[R]",
    "[L]",
)
EXAMPLE_TOKEN, OUTPUT_TOKEN, DESC_TOKEN = "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]"
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
TASK_TYPE_TO_ID = {"entities": 1, "json_structures": 2, "relations": 3, "classifications": 4}
RECORD_MODE_TO_ID = {"natural": 1, "latent": 2, "anchorless": 3}
CARDINALITY_TO_ID = {"optional_one": 1, "required_one": 2, "zero_or_more": 3, "one_or_more": 4}
_POOLING = ("first", "mean", "max")
_SPAN_RESERVED = frozenset({"text", "confidence", "start", "end"})


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


class Gliner2Schema(TypedDict, total=False):
    """
    Schema accepted by `Gliner2Processor.__call__`.

    Args:
        entities (`dict` or `list`, *optional*):
            Entity names, or a map from name to a description or a field spec (`dtype`, `threshold`,
            `description`, `validators`).
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
            Classification tasks with `task`, `labels`, and optional `multi_label`, `cls_threshold`, `class_act`,
            `prompt`, `examples`, and `label_descriptions`.
        constraints (`list[dict]`, *optional*):
            Constraints applied by the classification decoder.
        record_metadata (`dict`, *optional*):
            Record mode and anchor field for each structure.
    """

    entities: dict[str, Any] | list[str]
    entity_descriptions: dict[str, str]
    entity_attribute_groups: dict[str, dict[str, Any]]
    entity_attribute_labels: list[str]
    entity_attribute_prompt_labels: dict[str, str]
    json_structures: list[dict[str, dict[str, Any]]]
    json_descriptions: dict[str, dict[str, str]]
    relations: list[dict[str, dict[str, Any]]]
    relation_descriptions: dict[str, str]
    relation_metadata: dict[str, dict[str, Any]]
    classifications: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    record_metadata: dict[str, dict[str, Any]]


class Gliner2Labels(TypedDict, total=False):
    """
    Supervision accepted by `Gliner2Processor.__call__`.

    Args:
        entities (`dict`, *optional*):
            Entity name to a string, a `{text, start, end}` span, or a list of either.
        classifications (`dict`, *optional*):
            Task name to a class name or a list of class names.
        relations (`dict`, *optional*):
            Relation name to a head/tail pair or a list of pairs.
        json_structures (`dict`, *optional*):
            Structure name to a record or a list of records.
    """

    entities: dict[str, Any]
    classifications: dict[str, list[str] | str]
    relations: dict[str, list[dict[str, Any]] | dict[str, Any]]
    json_structures: dict[str, list[dict[str, Any]] | dict[str, Any]]


@dataclass(frozen=True)
class Field:
    """One compiled schema field. `kind` is `span`, `choice`, or `label`."""

    kind: str = "span"
    name: str = ""
    dtype: str = "list"
    threshold: float | None = None
    choices: tuple[str, ...] = ()
    validators: tuple[Any, ...] = ()


@dataclass(frozen=True)
class AttributeSpec:
    """Entity attribute group rescored from kept spans."""

    name: str
    labels: tuple[str, ...]
    applies_to: tuple[str, ...] | None
    multi_label: bool
    threshold: float


@dataclass(frozen=True)
class FieldGroup:
    """One encoded schema group. Decoding and the pipeline read this object."""

    task: str
    name: str
    fields: tuple[Field, ...]
    tokens: tuple[str, ...] = ()
    record: GroupRecord | None = None
    threshold: float | None = None
    multi_label: bool = False
    activation: str = "auto"
    attributes: tuple[AttributeSpec, ...] = ()
    attribute_labels: tuple[str, ...] = ()
    attribute_prompts: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class WordGrid:
    """Document words and the choice-prefix length they were aligned to."""

    words: tuple[str, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    text: str
    prefix_len: int
    prefix: tuple[str, ...]


_SPLITTERS = {
    "whitespace": re.compile(
        r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |\w+(?:[-_]\w+)*
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    ),
    "char": re.compile(r"[A-Za-z0-9@._\-+]+|\S"),
}


def _resolve_word_splitter(word_splitter: Any) -> Callable[..., Iterator[tuple[str, int, int]]]:
    """Return a callable yielding `(token, start, end)` for a splitter name or callable."""
    if callable(word_splitter):
        return word_splitter
    pattern = _SPLITTERS.get(word_splitter or "whitespace")
    if pattern is None:
        raise ValueError(f"Unknown word_splitter {word_splitter!r}. Supported names: {sorted(_SPLITTERS)}.")

    def split(text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
        for match in pattern.finditer(text):
            yield (match.group().lower() if lower else match.group()), match.start(), match.end()

    return split


def _normalize_text(text: str) -> str:
    """Ensure text ends with sentence punctuation."""
    if not text:
        return "."
    return text if text.endswith((".", "!", "?")) else text + "."


@dataclass(frozen=True)
class TextChunk:
    """One word window and its offsets into the original text."""

    text: str
    start_char: int
    end_char: int
    start_word: int
    end_word: int


def chunk_words(
    words: str | Sequence[Any], chunk_size: int = 384, chunk_overlap: int = 64, word_splitter: Any = None
) -> list[TextChunk]:
    """Split text, words, or `(token, start, end)` tuples into overlapping windows.

    Args:
        words (`str` or sequence):
            Document text or pre-split words. String offsets index the original string.
        chunk_size (`int`, *optional*, defaults to 384):
            Maximum words in one window.
        chunk_overlap (`int`, *optional*, defaults to 64):
            Words shared by neighboring windows.
        word_splitter (`str` or `callable`, *optional*):
            Splitter name or callable used when `words` is a string.
    """
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size must be positive and larger than a non-negative chunk_overlap")
    source = words if isinstance(words, str) else None
    if source is not None:
        tokens = list(_resolve_word_splitter(word_splitter)(source, False))
    else:
        tokens = [
            (str(word[0]), int(word[1]), int(word[2]))
            if isinstance(word, (tuple, list)) and len(word) >= 3
            else (str(word), index, index + 1)
            for index, word in enumerate(words)
        ]
    if not tokens:
        return [TextChunk(source or "", 0, len(source or ""), 0, 0)]
    chunks = []
    for start_word in range(0, len(tokens), chunk_size - chunk_overlap):
        end_word = min(start_word + chunk_size, len(tokens))
        start_char, end_char = tokens[start_word][1], tokens[end_word - 1][2]
        if source is None:
            window = " ".join(token for token, _, _ in tokens[start_word:end_word])
        else:
            window = source[start_char:end_char]
        chunks.append(TextChunk(window, start_char, end_char, start_word, end_word))
        if end_word == len(tokens):
            break
    return chunks


def _chunk_start(chunk: Any) -> int:
    return int(chunk["start_char"] if isinstance(chunk, Mapping) else chunk.start_char)


def _is_span_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("start"), int)
        and isinstance(value.get("end"), int)
        and ("text" in value)
    )


def _is_classification_dict(value: Any) -> bool:
    return isinstance(value, dict) and "label" in value and "confidence" in value


def _is_score(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def remap_result_spans(result: Any, original_text: str, chunk: Any) -> Any:
    """Shift span dicts in a nested decode payload by the chunk's `start_char`."""
    if isinstance(result, list):
        return [remap_result_spans(item, original_text, chunk) for item in result]
    if not isinstance(result, dict):
        return result
    remapped = {key: remap_result_spans(value, original_text, chunk) for key, value in result.items()}
    if _is_span_dict(remapped):
        start = int(remapped["start"]) + _chunk_start(chunk)
        end = int(remapped["end"]) + _chunk_start(chunk)
        remapped.update(start=start, end=end)
        if 0 <= start <= end <= len(original_text):
            remapped["text"] = original_text[start:end]
    return remapped


def _as_list(value: Any) -> list[Any]:
    return [] if value is None else value if isinstance(value, list) else [value]


def _canonical_key(value: Any) -> str:
    if isinstance(value, dict):
        return repr(sorted((key, _canonical_key(item)) for key, item in value.items() if key != "confidence"))
    if isinstance(value, list):
        return repr([_canonical_key(item) for item in value])
    return repr(value)


def _representative_confidence(value: Any) -> float:
    if isinstance(value, dict) and _is_score(value.get("confidence")):
        return float(value["confidence"])
    nested = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return max((_representative_confidence(item) for item in nested), default=0.0)


def _dedupe_items(items: list[Any], overlap_policy: str | None = None) -> list[Any]:
    spans = [item for item in items if _is_span_dict(item)]
    deduped = []
    if spans:
        selected = resolve_overlaps(
            spans,
            overlap_policy,
            default="allow",
            score=lambda item: float(item.get("confidence", 0.0)),
            start=lambda item: int(item["start"]),
            end=lambda item: int(item["end"]),
        )
        deduped = sorted(selected, key=lambda item: (item["start"], item["end"], item.get("text", "")))
    others: dict[str, Any] = {}
    for item in items:
        if _is_span_dict(item):
            continue
        key = _canonical_key(item)
        if key not in others or _representative_confidence(item) > _representative_confidence(others[key]):
            others[key] = item
    return deduped + list(others.values())


def _ordered_keys(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys(key for value in values if isinstance(value, dict) for key in value))


def _merge_map(values: list[Any], overlap_policy: str = "disallow", *, scalars=None, listed: bool = False):
    """Merge dicts by key. `scalars` keeps one value. `listed` always returns lists."""
    merged: dict[str, Any] = {}
    for key in _ordered_keys(values):
        present = [value[key] for value in values if isinstance(value, dict) and key in value]
        if not listed and scalars is None:
            merged[key] = _merge_values(present, overlap_policy)
            continue
        items = [item for value in present for item in _as_list(value)]
        if listed:
            merged[key] = _dedupe_items(items)
        else:
            deduped = _dedupe_items(items, overlap_policy=overlap_policy)
            merged[key] = (deduped[0] if deduped else None) if key in scalars else deduped
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
        return _dedupe_items([item for value in non_empty for item in value], overlap_policy=overlap_policy)
    if all(isinstance(value, dict) for value in non_empty):
        return _merge_map(non_empty, overlap_policy)
    kinds = {type(value).__name__ for value in non_empty}
    if len(kinds) > 1:
        raise ValueError(f"cannot merge values of types {sorted(kinds)}")
    return non_empty[0]


def _strip_span_metadata(value: Any, include_confidence: bool, include_spans: bool) -> Any:
    if isinstance(value, list):
        return [_strip_span_metadata(item, include_confidence, include_spans) for item in value]
    if not isinstance(value, dict):
        return value
    if _is_span_dict(value):
        extras = {key: item for key, item in value.items() if key not in _SPAN_RESERVED}
        if not include_confidence and not include_spans and not extras:
            return value.get("text", "")
        stripped: dict[str, Any] = {"text": value.get("text", "")}
        if include_confidence and "confidence" in value:
            stripped["confidence"] = value["confidence"]
        if include_spans:
            stripped.update(start=value["start"], end=value["end"])
        return {**stripped, **extras}
    if _is_classification_dict(value):
        return {"label": value["label"], "confidence": value["confidence"]} if include_confidence else value["label"]
    if "text" in value and "confidence" in value and "start" not in value and "end" not in value:
        return {"text": value["text"], "confidence": value["confidence"]} if include_confidence else value["text"]
    return {key: _strip_span_metadata(item, include_confidence, include_spans) for key, item in value.items()}


def _is_joint_result(result: Any) -> bool:
    entities = result.get("entities") if isinstance(result, Mapping) else None
    return isinstance(entities, list) and (
        not entities or (isinstance(entities[0], Mapping) and "type" in entities[0])
    )


def _rank(confidence: float | None) -> float:
    return float("-inf") if confidence is None else confidence


def _merge_joint_documents(original_text, chunks, chunk_results, include_confidence, include_spans) -> dict[str, Any]:
    """Merge joint graphs by document offsets. Relations stay inside one chunk."""
    entity_by_key, relation_rows = {}, {}
    for chunk, raw in zip(chunks, chunk_results):
        offset = _chunk_start(chunk)
        local_keys = {}
        for entity in raw.get("entities") or []:
            start, end = int(entity.get("start", 0)) + offset, int(entity.get("end", 0)) + offset
            key = (str(entity.get("type", entity.get("label", ""))), start, end)
            local_keys[str(entity.get("id", ""))] = key
            previous = entity_by_key.get(key)
            if previous is None or _rank(entity.get("confidence")) > _rank(previous["confidence"]):
                entity_by_key[key] = {
                    "text": original_text[start:end],
                    "confidence": entity.get("confidence"),
                    "sentence_id": entity.get("sentence_id"),
                    "rescued": bool(entity.get("rescued", False)),
                }
        for relation in raw.get("relations") or []:
            head, tail = local_keys.get(str(relation.get("head", ""))), local_keys.get(str(relation.get("tail", "")))
            if head is None or tail is None:
                continue
            key = (str(relation.get("type", relation.get("label", ""))), head, tail)
            previous = relation_rows.get(key)
            if previous is None or _rank(relation.get("confidence")) > _rank(previous[0]):
                relation_rows[key] = (relation.get("confidence"), bool(relation.get("derived", False)))
    ordered = sorted(entity_by_key, key=lambda key: (key[1], key[2], key[0]))
    key_to_id = {key: f"e{index + 1}" for index, key in enumerate(ordered)}
    entities_out = []
    for key in ordered:
        item = entity_by_key[key]
        payload = {"id": key_to_id[key], "type": key[0], "text": item["text"]}
        if include_spans:
            payload.update(start=key[1], end=key[2])
            if item["sentence_id"] is not None:
                payload["sentence_id"] = item["sentence_id"]
        if include_confidence and item["confidence"] is not None:
            payload["confidence"] = item["confidence"]
        if item["rescued"]:
            payload["rescued"] = True
        entities_out.append(payload)
    relations_out = []
    for (label, head, tail), (confidence, derived) in relation_rows.items():
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
    if chunk_results and all(_is_joint_result(item) for item in chunk_results):
        return _merge_joint_documents(original_text, chunks, chunk_results, include_confidence, include_spans)
    policy = normalize_overlap_policy(overlap_policy, default="disallow")
    results = [remap_result_spans(result, original_text, chunk) for chunk, result in zip(chunks, chunk_results)]
    merged = {}
    for key in _ordered_keys(results):
        values = [result[key] for result in results if key in result]
        if key == "entities":
            merged[key] = _merge_map(values, policy, scalars=set(scalar_entity_labels or ()))
        elif key == "relation_extraction":
            merged[key] = _merge_map(values, policy, listed=True)
        else:
            merged[key] = _merge_values(values, policy)
    return _strip_span_metadata(merged, include_confidence, include_spans)


def _labelled(label: Any, confidence: Any, include_confidence: bool) -> Any:
    return {"label": label, "confidence": confidence} if include_confidence else label


def _format_classification(value: object, include_confidence: bool) -> object:
    """Format a label pair or a list of label pairs."""
    if isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], (list, tuple)):
            return [_labelled(item[0], item[1], include_confidence) for item in value]
        if len(value) == 2 and isinstance(value[0], str) and _is_score(value[1]):
            return _labelled(value[0], value[1], include_confidence)
    return value


def _dedupe_field_values(value: list, include_confidence: bool) -> list:
    unique, seen = [], set()
    for item in value:
        if isinstance(item, tuple):
            text, confidence, start, end = item
            key, kept = (
                (text.lower(), start, end),
                {"text": text, "confidence": confidence} if include_confidence else text,
            )
        elif isinstance(item, dict):
            text = item.get("text", "")
            key, kept = (text.lower(), item.get("start"), item.get("end")), item
            if "start" not in item or "end" not in item:
                key = (text.lower(), None, None)
        else:
            text, key, kept = item, item.lower() if item else None, item
        if text and key not in seen:
            seen.add(key)
            unique.append(kept)
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
        else:
            formatted[field_name] = value or None
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
    for key, value in results.items():
        first = value[0] if isinstance(value, list) and value else None
        if key in (classification_tasks or ()):
            formatted[key] = _format_classification(value, include_confidence)
        elif (
            key in (requested_relations or ())
            or (isinstance(first, tuple) and len(first) == 2)
            or (isinstance(first, dict) and "head" in first and "tail" in first)
        ):
            relations[key] = value if isinstance(value, list) else []
        elif isinstance(first, dict):
            entities = key == "entities"
            formatted[key] = (
                format_field(first, include_confidence)
                if entities
                else [format_field(item, include_confidence) for item in value]
            )
        elif isinstance(first, tuple):
            formatted[key] = [_labelled(label, confidence, include_confidence) for label, confidence in value]
        elif isinstance(value, tuple):
            formatted[key] = _labelled(value[0], value[1], include_confidence)
        elif isinstance(value, dict):
            formatted[key] = format_field(value, include_confidence)
        else:
            formatted[key] = {} if key == "entities" and value == [] else value
    for name in requested_relations or ():
        relations.setdefault(name, [])
    if relations:
        formatted["relation_extraction"] = relations
    return formatted


def _transform_schema(parent, fields, child_prefix, prompt=None, examples=None, label_descriptions=None) -> list[str]:
    """Turn one schema group into structural word tokens."""
    prompt_str = f"{parent}: {prompt}" if prompt else parent
    for label, desc in (label_descriptions or {}).items():
        if label in fields:
            prompt_str += f" {DESC_TOKEN} {label}: {desc}"
    for inp, out in examples or ():
        if out in fields:
            prompt_str += f" {EXAMPLE_TOKEN} {inp} {OUTPUT_TOKEN} {out if isinstance(out, str) else ', '.join(out)}"
    return ["(", P_TOKEN, prompt_str, "(", *[token for name in fields for token in (child_prefix, name)], ")", ")"]


def _choice_prefix(schema: Mapping[str, Any]) -> list[str]:
    """Word tokens prepended for JSON choice fields."""
    prefix: list[str] = []
    for struct in schema.get("json_structures") or []:
        for parent, fields in struct.items():
            inner: list[str] = []
            for name, spec in fields.items():
                if isinstance(spec, dict) and "choices" in spec and ("value" in spec or spec["choices"]):
                    choices = [str(choice) for choice in spec["choices"]]
                    separated = [token for index, item in enumerate(choices) for token in ("|", item)[index == 0 :]]
                    inner += [name, "(", *separated, ")", ","]
            if inner:
                prefix += ["(", f"{parent}:", *inner[:-1], ")"]
    return prefix


def _fields(task: str, names: Sequence[str], occurrences: Sequence[Mapping[str, Any]] = ()) -> tuple[Field, ...]:
    """Build fields from the first spec given for each name."""
    fields = []
    for name in names:
        first = next((occurrence[name] for occurrence in occurrences if name in occurrence), None)
        spec = dict(first) if isinstance(first, Mapping) else {}
        later = (
            occurrence[name]["choices"]
            for occurrence in reversed(occurrences)
            if isinstance(occurrence.get(name), Mapping) and occurrence[name].get("choices")
        )
        choices = tuple(str(choice) for choice in (spec.get("choices") or next(later, ())))
        kind = "label" if task == "classifications" else "choice" if task == "json_structures" and choices else "span"
        threshold = spec.get("threshold")
        fields.append(
            Field(
                kind=kind,
                name=name,
                dtype=str(spec.get("dtype") or "list"),
                threshold=None if threshold is None else float(threshold),
                choices=choices,
                validators=tuple(spec.get("validators") or ()),
            )
        )
    return tuple(fields)


def _attribute_specs(raw: Mapping[str, Any]) -> tuple[AttributeSpec, ...]:
    specs = []
    for name, group in raw.items():
        applies = group.get("applies_to")
        threshold = group.get("threshold", 0.5)
        specs.append(
            AttributeSpec(
                name=str(name),
                labels=tuple(str(label) for label in (group.get("labels") or ())),
                applies_to=None if applies is None else tuple(str(item) for item in applies),
                multi_label=bool(group.get("multi_label", False)),
                threshold=0.5 if threshold is None else float(threshold),
            )
        )
    return tuple(specs)


def _grouped(items: Sequence[Mapping[str, Any]] | None) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in items or []:
        for parent, fields in item.items():
            grouped.setdefault(parent, []).append(fields)
    return grouped


def _schema_groups(schema: Mapping[str, Any]) -> list[FieldGroup]:
    """Compile structures, entities, relations, then classifications."""
    groups = []
    for parent, occurrences in _grouped(schema.get("json_structures")).items():
        names = list(dict.fromkeys(name for occurrence in occurrences for name in occurrence))
        if names:
            descs = (schema.get("json_descriptions") or {}).get(parent) or {}
            tokens = _transform_schema(parent, names, C_TOKEN, label_descriptions=descs)
            groups.append(
                FieldGroup("json_structures", parent, _fields("json_structures", names, occurrences), tuple(tokens))
            )
    entities = schema.get("entities") or {}
    if entities:
        names = list(entities)
        tokens = _transform_schema("entities", names, E_TOKEN, label_descriptions=schema.get("entity_descriptions"))
        prompts = schema.get("entity_attribute_prompt_labels") or {}
        groups.append(
            FieldGroup(
                task="entities",
                name="entities",
                fields=_fields("entities", names, [entities]),
                tokens=tuple(tokens),
                attributes=_attribute_specs(schema.get("entity_attribute_groups") or {}),
                attribute_labels=tuple(str(label) for label in (schema.get("entity_attribute_labels") or ())),
                attribute_prompts=tuple((str(key), str(value)) for key, value in prompts.items()),
            )
        )
    descriptions = schema.get("relation_descriptions") or {}
    for parent, occurrences in _grouped(schema.get("relations")).items():
        names = list(occurrences[0])
        threshold = ((schema.get("relation_metadata") or {}).get(parent) or {}).get("threshold")
        groups.append(
            FieldGroup(
                task="relations",
                name=parent,
                fields=_fields("relations", names, occurrences),
                tokens=tuple(_transform_schema(parent, names, R_TOKEN, prompt=descriptions.get(parent))),
                threshold=None if threshold is None else float(threshold),
            )
        )
    for item in schema.get("classifications") or []:
        labels = list(item["labels"])
        tokens = _transform_schema(
            item["task"], labels, L_TOKEN, item.get("prompt"), item.get("examples"), item.get("label_descriptions")
        )
        threshold = item.get("cls_threshold")
        groups.append(
            FieldGroup(
                task="classifications",
                name=str(item["task"]),
                fields=_fields("classifications", labels),
                tokens=tuple(tokens),
                threshold=None if threshold is None else float(threshold),
                multi_label=bool(item.get("multi_label", False)),
                activation=str(item.get("class_act", "auto")),
            )
        )
    return groups


def _with_records(groups: Sequence[FieldGroup], schema: Mapping[str, Any]) -> list[FieldGroup]:
    """Attach record specs from `record_metadata`. Query ids skip classification groups."""
    metadata = schema.get("record_metadata") or {}
    offsets, cursor = [], 0
    for group in groups:
        offsets.append(cursor)
        cursor += 0 if group.task == "classifications" else len(group.fields)
    updated = list(groups)
    for task_index, group in enumerate(groups):
        cfg = metadata.get(group.name) or {}
        mode, anchor = cfg.get("mode"), cfg.get("anchor")
        if group.task != "json_structures" or mode is None:
            continue
        if mode not in RECORD_MODE_TO_ID or (mode == "natural") != bool(anchor):
            raise ValueError(f"record_metadata[{group.name!r}] needs a valid mode, with an anchor only when natural")
        policy = cfg.get("occurrence_policy", "latent_all")
        if policy not in ("all", "first", "error_on_ambiguous", "latent_all"):
            raise ValueError(f"record_metadata[{group.name!r}].occurrence_policy invalid: {policy!r}")
        fields = []
        for role_index, field in enumerate(group.fields):
            field_cfg = (cfg.get("fields") or {}).get(field.name) or {}
            is_anchor = mode == "natural" and field.name == anchor
            default = "required_one" if is_anchor else "optional_one" if field.dtype == "str" else "zero_or_more"
            cardinality = field_cfg.get("cardinality") or default
            if cardinality not in CARDINALITY_TO_ID:
                raise ValueError(f"record_metadata[{group.name!r}] cardinality invalid: {cardinality!r}")
            fields.append(
                RecordField(
                    name=field.name,
                    query_id=offsets[task_index] + role_index,
                    cardinality=cardinality,
                    is_anchor=is_anchor,
                    exclusive=bool(field_cfg.get("exclusive", False)),
                )
            )
        anchor_query_id = next((item.query_id for item in fields if item.is_anchor), None)
        if mode == "natural" and anchor_query_id is None:
            raise ValueError(f"record {group.name!r} declares anchor {anchor!r} but no matching field was found")
        record = GroupRecord(
            mode=mode,
            anchor=anchor,
            occurrence_policy=policy,
            fields=tuple(fields),
            anchor_query_id=anchor_query_id,
            task_index=task_index,
        )
        updated[task_index] = replace(group, record=record)
    return updated


def compile_schema(schema: Any) -> tuple[list[FieldGroup], list[str]]:
    """Compile a schema into field groups and the choice-prefix words.

    Args:
        schema (`dict`):
            Entities, structures, relations, and classifications.

    Returns:
        `tuple[list[FieldGroup], list[str]]`: Groups in encode order, then the
        choice-prefix words prepended to the document.
    """
    schema = _canonicalize_schema(schema)
    return _with_records(_schema_groups(schema), schema), _choice_prefix(schema)


def _result_key(group: FieldGroup) -> str:
    """Decode key. Relation prompts keep the `name: description` text."""
    if group.task == "entities":
        return "entities"
    if len(group.tokens) > 2:
        return str(group.tokens[2]).split(f" {DESC_TOKEN} ")[0]
    return group.task


def _parse_field_value(value: Any) -> dict[str, Any]:
    """Normalize a JSON field value, including `name::str::[a|b]::desc` strings."""
    if isinstance(value, Mapping):
        return dict(value)
    if value in ("", None, "str", "list"):
        return {"dtype": value or "list", "value": ""}
    if not isinstance(value, str) or "::" not in value:
        raise TypeError("json structure field values must be dicts")
    parsed: dict[str, Any] = {"dtype": "list", "value": ""}
    dtype_set = False
    for part in value.split("::")[1:]:
        if part in ("str", "list"):
            parsed["dtype"], dtype_set = part, True
        elif part.startswith("[") and part.endswith("]"):
            parsed["choices"] = [item.strip() for item in part[1:-1].split("|")]
            parsed["dtype"] = parsed["dtype"] if dtype_set else "str"
        elif part:
            parsed["description"] = part
    return parsed


def _canonicalize_schema(schema: Any) -> dict[str, Any]:
    """Normalize entities and JSON structures and check relations and classifications."""
    if not isinstance(schema, Mapping) and callable(getattr(type(schema), "build", None)):
        schema = schema.build()
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a dict")
    schema = dict(schema)
    entities = schema.get("entities")
    if isinstance(entities, list):
        schema["entities"] = {str(name): "" for name in entities}
    elif isinstance(entities, Mapping):
        descriptions = dict(schema.get("entity_descriptions") or {})
        normalized = {}
        for name, value in entities.items():
            description = value if isinstance(value, str) else value.get("description") if value else None
            if isinstance(description, str) and description:
                descriptions.setdefault(str(name), description)
            normalized[str(name)] = dict(value) if isinstance(value, Mapping) else ""
        schema["entities"] = normalized
        if descriptions:
            schema["entity_descriptions"] = descriptions
    elif entities is not None:
        raise TypeError("entities must be a list or dict")
    if schema.get("json_structures") is not None:
        schema["json_structures"] = [
            {
                str(parent): {str(name): _parse_field_value(value) for name, value in fields.items()}
                for parent, fields in item.items()
            }
            for item in schema["json_structures"]
        ]
    for item in schema.get("relations") or []:
        for name, endpoints in item.items():
            if not isinstance(endpoints, Mapping) or not {"head", "tail"} <= {str(key) for key in endpoints}:
                raise ValueError(f"relations[{name!r}] must include head and tail")
    for item in schema.get("classifications") or []:
        if not isinstance(item, Mapping) or not isinstance(item.get("labels"), (list, tuple)) or "task" not in item:
            raise ValueError("classifications items must be {task, labels}")
    return schema


def _canonicalize_labels(labels: Any, schema: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Check label keys against the schema. Spans are resolved after tokenization."""
    unknown = set(labels) - {"entities", "classifications", "relations", "json_structures"}
    if unknown:
        raise ValueError(f"unknown label keys {sorted(unknown)}")
    known = {
        "entities": set(schema.get("entities") or {}),
        "relations": {name for item in schema.get("relations") or [] for name in item},
        "json_structures": {name for item in schema.get("json_structures") or [] for name in item},
    }
    task_labels = {item["task"]: list(item["labels"]) for item in schema.get("classifications") or []}
    normalized: dict[str, dict[str, Any]] = {"classifications": {}}
    for kind, names in known.items():
        normalized[kind] = {}
        for name, value in (labels.get(kind) or {}).items():
            if name not in names:
                raise ValueError(f"{kind} label {name!r} is not in the schema")
            normalized[kind][str(name)] = value if isinstance(value, list) else [value]
    for task, value in (labels.get("classifications") or {}).items():
        true = value if isinstance(value, list) else [value]
        if task not in task_labels or any(item not in task_labels[task] for item in true):
            raise ValueError(f"classification labels {true} are not in task {task!r}")
        normalized["classifications"][str(task)] = [str(item) for item in true]
    return normalized


def _locate_mention(mention: Any, choices, grid: WordGrid, splitter) -> list[tuple[int, int]]:
    """Inclusive word spans. Strings match every occurrence."""
    offset = grid.prefix_len
    if isinstance(mention, Mapping):
        surface, start, end = str(mention["text"]), int(mention["start"]), int(mention["end"])
        snippet = grid.text[start:end] if 0 <= start < end <= len(grid.text) else None
        if snippet is None or surface not in (snippet, snippet.strip()):
            raise ValueError(f"{surface!r} does not match text[{start}:{end}]")
        covered = [
            index for index, bounds in enumerate(zip(grid.starts, grid.ends)) if bounds[0] < end and start < bounds[1]
        ]
        pieces = [token for token, _, _ in splitter(surface, True)]
        if not covered or list(grid.words[covered[0] : covered[-1] + 1]) != pieces:
            raise ValueError(f"{surface!r} at [{start}, {end}) does not match the document words")
        return [(offset + covered[0], offset + covered[-1])]
    if not isinstance(mention, str):
        raise TypeError(f"span label must be a string or {{text, start, end}}, got {type(mention).__name__}")
    if choices:
        found = [index for index, token in enumerate(grid.prefix) if token.lower() == mention.lower()]
        if mention not in choices or not found:
            raise ValueError(f"choice {mention!r} is not in {list(choices)}")
        return [(index, index) for index in found]
    pieces = [token for token, _, _ in splitter(mention, True)]
    width = len(pieces)
    matches = [
        (offset + index, offset + index + width - 1)
        for index in range(len(grid.words) - width + 1)
        if pieces and list(grid.words[index : index + width]) == pieces
    ]
    if not matches:
        raise ValueError(f"label {mention!r} is not in the text")
    return matches


def _field_spans(raw: Any, choices, grid: WordGrid, splitter) -> list[tuple[int, int]]:
    spans = (
        span
        for mention in (raw if isinstance(raw, list) else [raw])
        if mention not in ("", None)
        for span in _locate_mention(mention, choices, grid, splitter)
    )
    return list(dict.fromkeys(spans))


def _bind_instances(group: FieldGroup, labels, grid: WordGrid, splitter):
    """Inclusive word spans per field for each gold instance, plus half-open relation edges."""
    if group.task == "entities":
        instance = [_field_spans(labels["entities"].get(item.name, []), None, grid, splitter) for item in group.fields]
        return ([instance] if any(instance) else []), []
    names = {item.name for item in group.fields}
    instances, edges = [], []
    for raw in labels[group.task].get(group.name, []):
        if not isinstance(raw, Mapping) or set(raw) - names:
            raise ValueError(f"{group.name} labels must map its fields to values")
        if group.task == "relations" and not names <= set(raw):
            raise ValueError(f"relation {group.name!r} needs every endpoint")
        instance = [
            _field_spans(raw[item.name], item.choices if item.kind == "choice" else None, grid, splitter)
            if item.name in raw
            else []
            for item in group.fields
        ]
        if group.task == "relations":
            if len(instance) < 2 or not instance[0] or not instance[1]:
                raise ValueError(f"relation {group.name!r} is not in the text")
            edges += [(hs, he + 1, ts, te + 1) for hs, he in instance[0] for ts, te in instance[1]]
        if any(instance):
            instances.append(instance)
    return instances, edges


def _bind_example(groups: Sequence[FieldGroup], labels, grid: WordGrid, splitter) -> dict:
    """Map one example's labels onto inclusive word spans and half-open mentions."""
    bound = {"blocks": [], "classes": [], "spans": [], "mentions": [], "edges": [], "relation_queries": []}
    query_id = 0
    for group in groups:
        if group.task == "classifications":
            true = set(labels["classifications"].get(group.name, []))
            block = [1 if item.name in true else 0 for item in group.fields]
            bound["blocks"].append(block)
            bound["classes"].append(torch.tensor(block, dtype=torch.float))
            continue
        instances, edges = _bind_instances(group, labels, grid, splitter)
        block = [len(instances), instances]
        bound["blocks"].append(block)
        bound["spans"].append(block)
        for instance in instances:
            for field_index, spans in enumerate(instance):
                bound["mentions"] += [(query_id + field_index, start, end + 1) for start, end in spans]
        if group.task == "relations":
            bound["edges"].append(edges)
            bound["relation_queries"].append((query_id, query_id + 1))
        query_id += len(group.fields)
    bound["query_count"] = query_id
    return bound


def _pack_mentions(supervisions: Sequence[Mapping[str, Any]], max_gold_per_query: int | None):
    query_width = max((item["query_count"] for item in supervisions), default=0)
    grouped = []
    for item in supervisions:
        per_query: dict[int, dict[tuple[int, int], None]] = {}
        for query_id, start, end in item["mentions"]:
            per_query.setdefault(query_id, {})[(start, end)] = None
        grouped.append({query_id: list(pairs) for query_id, pairs in per_query.items()})
    observed = max((len(pairs) for per_query in grouped for pairs in per_query.values()), default=0)
    if max_gold_per_query is not None and (max_gold_per_query <= 0 or observed > max_gold_per_query):
        raise ValueError(f"max_gold_per_query={max_gold_per_query} must be positive and cover {observed} gold spans")
    gold_width = max(observed, 1) if max_gold_per_query is None else max_gold_per_query
    mention_pairs = torch.zeros((len(supervisions), query_width, gold_width, 2), dtype=torch.long)
    mention_mask = torch.zeros((len(supervisions), query_width, gold_width), dtype=torch.bool)
    for batch_index, per_query in enumerate(grouped):
        for query_id, pairs in per_query.items():
            mention_pairs[batch_index, query_id, : len(pairs)] = torch.tensor(pairs, dtype=torch.long)
            mention_mask[batch_index, query_id, : len(pairs)] = True
    return mention_pairs, mention_mask, query_width


def build_targets(
    batch_groups: Sequence[Sequence[FieldGroup]],
    batch_labels: Sequence[Mapping[str, Any]],
    batch_grids: Sequence[WordGrid],
    max_gold_per_query: int | None = None,
    splitter: Callable[..., Iterator[tuple[str, int, int]]] | None = None,
) -> dict[str, Any]:
    """Bind labels on each example's word grid and pack the loss targets.

    Args:
        batch_groups:
            Compiled groups per example.
        batch_labels:
            Canonical labels aligned with `batch_groups`.
        batch_grids:
            Word grids aligned with `batch_groups`.
        max_gold_per_query (`int`, *optional*):
            Cap on gold spans stored per query.
        splitter (`callable`, *optional*):
            Word splitter used to locate string mentions.
    """
    splitter = splitter or _resolve_word_splitter("whitespace")
    supervisions = [
        _bind_example(groups, labels, grid, splitter)
        for groups, labels, grid in zip(batch_groups, batch_labels, batch_grids)
    ]
    mention_pairs, mention_mask, query_width = _pack_mentions(supervisions, max_gold_per_query)
    batch = len(supervisions)
    relation_width = max((len(item["edges"]) for item in supervisions), default=0)
    edge_width = max((len(edges) for item in supervisions for edges in item["edges"]), default=0)
    relation_edges = torch.zeros((batch, relation_width, max(edge_width, 1), 4), dtype=torch.long)
    relation_edge_mask = torch.zeros((batch, relation_width, max(edge_width, 1)), dtype=torch.bool)
    head_member = torch.zeros((batch, relation_width, query_width), dtype=torch.bool)
    tail_member = torch.zeros_like(head_member)
    relation_valid = torch.zeros((batch, relation_width), dtype=torch.bool)
    for batch_index, item in enumerate(supervisions):
        for relation_index, edges in enumerate(item["edges"]):
            if edges:
                relation_edges[batch_index, relation_index, : len(edges)] = torch.tensor(edges, dtype=torch.long)
                relation_edge_mask[batch_index, relation_index, : len(edges)] = True
        for relation_index, (head_query, tail_query) in enumerate(item["relation_queries"]):
            if 0 <= head_query < query_width and 0 <= tail_query < query_width:
                relation_valid[batch_index, relation_index] = True
                head_member[batch_index, relation_index, head_query] = True
                tail_member[batch_index, relation_index, tail_query] = True
    record_groups = [
        [
            {
                "spec": group.record,
                "targets": [
                    {
                        field.query_id: [[(start, end + 1) for start, end in spans]]
                        for field, spans in zip(group.record.fields, instance)
                    }
                    for instance in item["blocks"][group.record.task_index][1]
                ],
            }
            for group in groups
            if group.record is not None
        ]
        for groups, item in zip(batch_groups, supervisions)
    ]
    return {
        "span_structures": [item["spans"] for item in supervisions],
        "mention_pairs": mention_pairs,
        "mention_mask": mention_mask,
        "classification_targets": [item["classes"] for item in supervisions],
        "relation_gold_pairs": relation_edges,
        "relation_gold_mask": relation_edge_mask,
        "relation_routing": (head_member, tail_member, relation_valid, torch.zeros_like(relation_valid)),
        "record_groups": record_groups,
    }


def _decode_label_group(group: FieldGroup, logits: Any, options: DecodeOptions) -> Any:
    """`(label, score)` for a single-label task, or the labels above the cutoff."""
    labels = [item.name for item in group.fields]
    flat = torch.as_tensor(logits, dtype=torch.float).detach().cpu().reshape(-1)
    if flat.numel() != len(labels):
        raise ValueError(f"classification logits ({flat.numel()}) do not match labels ({len(labels)})")
    if options.temperature <= 0:
        raise ValueError("classification temperature must be > 0")
    flat = flat / options.temperature
    sigmoid = group.activation == "sigmoid" or (group.activation != "softmax" and group.multi_label)
    probs = torch.sigmoid(flat) if sigmoid else torch.softmax(flat, dim=-1)
    best = int(torch.argmax(probs).item())
    if not group.multi_label:
        return (labels[best], float(probs[best].item()))
    cutoff = options.threshold if group.threshold is None else group.threshold
    chosen = [
        (label, float(probs[index].item())) for index, label in enumerate(labels) if float(probs[index]) >= cutoff
    ]
    return chosen or [(labels[best], float(probs[best].item()))]


def _classification_config(group: FieldGroup) -> dict[str, Any]:
    config = {
        "task": group.name,
        "labels": [item.name for item in group.fields],
        "multi_label": group.multi_label,
        "class_act": group.activation,
    }
    if group.threshold is not None:
        config["cls_threshold"] = group.threshold
    return config


def _decode_choice(field: Field, prefix_scores: torch.Tensor, prefix: Sequence[str], options: DecodeOptions):
    """Score each choice word in the prefix. A list keeps every passing choice."""
    cutoff = options.threshold if field.threshold is None else field.threshold
    lowered = [token.lower() for token in prefix]
    scored = [
        (choice, float(prefix_scores[lowered.index(choice.lower()), 0].item()))
        for choice in dict.fromkeys(field.choices)
        if choice.lower() in lowered and lowered.index(choice.lower()) < prefix_scores.shape[0]
    ]
    if field.dtype == "list":
        return [
            format_span(choice, score, 0, 0, options.include_confidence, False)
            for choice, score in scored
            if score >= cutoff
        ]
    best = max(scored, key=lambda item: item[1], default=None)
    if best and best[0] and best[1] >= cutoff:
        return format_span(best[0], best[1], 0, 0, options.include_confidence, False)
    return None


def _decode_span_group(group: FieldGroup, scores: torch.Tensor, row: Mapping[str, Any], options: DecodeOptions):
    """Decode `(count, fields, words, width)` probabilities for one non-label group."""
    doc_len = len(row["start"])
    prefix_len = row["prefix_len"]

    def spans(inst, index, item=None, **kwargs):
        if item is not None:
            kwargs.update(dtype=item.dtype or "list", threshold=item.threshold, validators=item.validators)
        return decode_spans(_doc_axis(scores[inst, index], doc_len), row, options, **kwargs)

    if group.task == "entities":
        decoded = {item.name: spans(0, index, item, blank=True) for index, item in enumerate(group.fields)}
        return [decoded] if decoded else []
    instances = []
    for inst in range(scores.shape[0]):
        if group.task == "relations":
            sides = [spans(inst, index, dtype="str", threshold=group.threshold) for index in range(len(group.fields))]
            if len(sides) == 2 and sides[0] and sides[1]:
                pair = {"head": sides[0], "tail": sides[1]}
                instances.append(pair if options.include_spans or options.include_confidence else (sides[0], sides[1]))
            continue
        instance = {
            item.name: _decode_choice(item, scores[inst, index, :prefix_len], row["words"][:prefix_len], options)
            if item.kind == "choice"
            else spans(inst, index, item)
            for index, item in enumerate(group.fields)
        }
        if any(value is not None and value != [] for value in instance.values()):
            instances.append(instance)
    return instances


def _per_group(value: Any, count: int, what: str) -> list:
    values = [value] if torch.is_tensor(value) else list(value)
    if len(values) != count:
        raise ValueError(f"expected {count} {what}, got {len(values)}")
    return values


def decode_fields(
    groups: Sequence[FieldGroup], sample: Mapping[str, Any], row: Mapping[str, Any], options: DecodeOptions
) -> dict[str, Any]:
    """Decode one span-architecture row from its field groups.

    Args:
        groups (`Sequence[FieldGroup]`):
            Groups to decode, in encode order.
        sample (`dict`):
            One row of model output: `span_logits` aligned with the non-label groups, optional `counts`,
            `classification_logits` aligned with the label groups, and optional typed relation pairs.
        row (`dict`):
            Processor metadata with `text`, `start`, `end`, `words`, and `prefix_len`.
        options (`DecodeOptions`):
            Cutoff, overlap policy, classification temperature, and payload switches.
    """
    span_groups = [group for group in groups if group.task != "classifications"]
    label_groups = [group for group in groups if group.task == "classifications"]
    results: dict[str, Any] = {}
    scores = sample.get("span_logits")
    if scores is not None and span_groups:
        counts = sample.get("counts")
        if torch.is_tensor(counts):
            counts = counts.detach().cpu().tolist()
        for index, (group, tensor) in enumerate(
            zip(span_groups, _per_group(scores, len(span_groups), "span tensors"))
        ):
            tensor = torch.sigmoid(torch.as_tensor(tensor, dtype=torch.float).detach().cpu())
            if tensor.ndim != 4 or tensor.shape[-2] != len(row["words"]):
                raise ValueError(
                    f"span scores {tuple(tensor.shape)} must be (count, fields, {len(row['words'])}, width)"
                )
            count = max(int(counts[index]) if counts is not None else int(tensor.shape[0]), 0)
            if count == 0:
                results[_result_key(group)] = [] if group.task in ("entities", "relations") else {}
            else:
                results[_result_key(group)] = _decode_span_group(group, tensor[:count], row, options)
    pairs = sample.get("relation_pairs")
    if pairs is not None and sample.get("relation_logits") is not None:
        paired = decode_relations(sample, groups, row, options)
        if paired or pairs.numel():
            results.update(paired)
            for group in span_groups:
                if group.task == "relations":
                    results.setdefault(_result_key(group), [])
    logits = sample.get("classification_logits")
    if logits is not None and label_groups:
        for group, vector in zip(label_groups, _per_group(logits, len(label_groups), "classification vectors")):
            results[group.name] = _decode_label_group(group, vector, options)
    return results


def _metadata_rows(metadata: Any) -> list[Mapping[str, Any]]:
    if isinstance(metadata, Mapping):
        return list(metadata["metadata"]) if "groups" not in metadata else [metadata]
    return list(metadata or [])


def _group_names(row: Mapping[str, Any], task: str) -> list[str]:
    return [group.name for group in row["groups"] if group.task == task]


_CANDIDATE_FIELDS = ("indices", "proposal_logits", "pair_logits", "valid_mask", "query_mask", "candidate_states")


def _split_outputs(outputs: Any, batch_size: int) -> list[Any]:
    """One mapping per batch row. Boundary candidate tensors keep a batch axis of one."""
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != batch_size:
            raise ValueError(f"outputs length ({len(outputs)}) != metadata length ({batch_size})")
        return list(outputs)
    if not isinstance(outputs, Mapping):
        outputs = dict(outputs.items())
    boundary = outputs.get("boundary")
    if not isinstance(boundary, Mapping) or boundary.get("candidates") is None:
        keys = ("span_logits", "counts", "classification_logits", "relation_pairs", "relation_logits")
        present = [key for key in keys if outputs.get(key) is not None]
        if not present:
            raise ValueError("outputs need span_logits, classification_logits, or boundary candidates")
        return [{key: outputs[key][index] for key in present} for index in range(batch_size)]
    candidates = boundary["candidates"]
    pairs, pair_logits = outputs.get("relation_pairs"), outputs.get("relation_logits")
    samples = []
    for index in range(batch_size):
        row = slice(index, index + 1)
        fields = {name: getattr(candidates, name) for name in _CANDIDATE_FIELDS}
        sample = {
            "candidates": replace(
                candidates, **{name: None if value is None else value[row] for name, value in fields.items()}
            ),
            "pair_logits": candidates.pair_logits[index],
        }
        for name in ("null_logits", "count_log_rates"):
            sample[name] = None if boundary.get(name) is None else boundary[name][index]
        for name in ("classification_logits", "record_logits"):
            if outputs.get(name) is not None:
                sample[name] = outputs[name][index]
        for name in ("text_states", "query_states"):
            if outputs.get(name) is not None:
                sample[name] = outputs[name][row]
        if pairs is not None and pair_logits is not None and pairs.numel():
            keep = pairs[:, 0] == index
            sample.update(
                relation_pairs=pairs[keep],
                relation_logits=pair_logits[keep],
                relation_temperature=outputs.get("relation_temperature"),
            )
        samples.append(sample)
    return samples


def _decode_sample(sample: Mapping[str, Any], row: Mapping[str, Any], options: DecodeOptions) -> dict[str, Any]:
    groups = row["groups"]
    if sample.get("pair_logits") is None:
        return decode_fields(groups, sample, row, options)
    decoded = decode_boundary(sample, groups, row, options)
    labels = [group for group in groups if group.task == "classifications"]
    decoded.update(decode_fields(labels, {"classification_logits": sample.get("classification_logits")}, row, options))
    return decoded


def _task_logit_map(sample: Mapping[str, Any], groups: Sequence[FieldGroup]) -> dict[str, Any]:
    """Label logits keyed by task, then label, for the constrained classifier."""
    vectors = sample.get("classification_logits")
    if torch.is_tensor(vectors) and vectors.ndim == 2:
        vectors = list(vectors)
    rows = _per_group(vectors, len(groups), "classification rows")
    return {
        group.name: dict(
            zip([item.name for item in group.fields], torch.as_tensor(row).detach().float().reshape(-1).tolist())
        )
        for group, row in zip(groups, rows)
    }


def _first_present(value: Mapping[str, Any], *names: str):
    return next((value[name] for name in names if value.get(name) is not None), None)


def _cached_states(outputs: Any):
    """Word and query states cached on a forward output or its boundary dict."""
    for value in (outputs, outputs.get("boundary") if isinstance(outputs, Mapping) else None):
        if not isinstance(value, Mapping):
            continue
        states = (
            value.get("text_states"),
            _first_present(value, "text_mask", "text_word_mask"),
            value.get("query_states"),
            _first_present(value, "query_mask", "query_marker_mask"),
        )
        if all(torch.is_tensor(state) for state in states):
            return states
    return None


def _entity_query_index(groups: Sequence[FieldGroup]) -> dict[str, int]:
    index, cursor = {}, 0
    for group in groups:
        if group.task == "classifications":
            continue
        for item in group.fields:
            if group.task == "entities":
                index[item.name] = cursor
            cursor += 1
    return index


def _entity_nodes(decoded: Mapping[str, Any]):
    entities = decoded.get("entities") if isinstance(decoded, Mapping) else None
    if isinstance(entities, Mapping):
        for name, value in entities.items():
            for item in (
                value if isinstance(value, list) else [value] if isinstance(value, dict) and "text" in value else []
            ):
                if isinstance(item, dict):
                    yield name, item
    elif isinstance(entities, list):
        for item in entities:
            if isinstance(item, dict):
                yield str(item.get("type", item.get("label", ""))), item


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
        self.tokenizer.add_special_tokens({"extra_special_tokens": list(SPECIAL_TOKENS)})
        for token in (*SPECIAL_TOKENS, "(", ")", ",", "|"):
            self._encode_piece(token)

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        """Cache one piece's ids for words the batched call drops."""
        if piece not in self._piece_ids:
            self._piece_ids[piece] = tuple(
                int(item) for item in self.tokenizer.encode(piece, add_special_tokens=False)
            )
        return self._piece_ids[piece]

    def _encode_words(self, words: Sequence[str]) -> tuple[list[int], list[int], set[int]]:
        """Encode pre-split words in one call and return first-subword indexes."""
        if not words:
            return [], [], set()
        encoding = self.tokenizer(list(words), is_split_into_words=True, add_special_tokens=False)
        ids = [int(item) for item in encoding["input_ids"]]
        first: list[int] = []
        empty: set[int] = set()
        cursor = shift = 0
        for index, word in enumerate(words):
            span = encoding.word_to_tokens(index)
            if span is not None and int(span.start) != int(span.end):
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

    def _format_input(self, schema_tokens: Sequence[Sequence[str]], text_tokens: Sequence[str]) -> dict[str, Any]:
        """Join schema groups and text, then map words onto first subwords."""
        combined: list[str] = []
        markers = []
        for struct in schema_tokens:
            offset = len(combined)
            markers.append(
                [offset + 1] * (len(struct) > 1) + [offset + index for index in range(4, len(struct) - 2, 2)]
            )
            combined += [*struct, SEP_STRUCT]
        if combined:
            combined.pop()
        text_start = len(combined) + 1
        combined += [SEP_TEXT, *text_tokens]
        input_ids, first_positions, empty_words = self._encode_words(combined)
        for index in sorted(empty_words):
            if index >= text_start:
                logger.warning(
                    "text word %r (index %d) produced no subwords; inserting a placeholder", combined[index], index
                )
        return {
            "input_ids": input_ids,
            "text_word_first_positions": first_positions[text_start:],
            "schema_special_positions": [[first_positions[index] for index in row] for row in markers],
        }

    def _transform_one(self, text: str, schema: Any, max_len: int | None, architecture: str, labels: Any) -> dict:
        schema = _canonicalize_schema(schema)
        groups, prefix = compile_schema(schema)
        text = _normalize_text(text)
        tokens = list(self._splitter(text, True))[:max_len]
        words, starts, ends = (
            [token[0] for token in tokens],
            [token[1] for token in tokens],
            [token[2] for token in tokens],
        )
        text_tokens = list(prefix) + words
        return {
            **self._format_input([group.tokens for group in groups], text_tokens),
            "task_types": [group.task for group in groups],
            "groups": tuple(groups),
            "grid": WordGrid(tuple(words), tuple(starts), tuple(ends), text, len(prefix), tuple(prefix)),
            "labels": None if labels is None else _canonicalize_labels(labels, schema),
            "schema_row": {
                "groups": tuple(groups),
                "text": text,
                "start": starts,
                "end": ends,
                "prefix_len": len(prefix),
                "words": text_tokens,
                "architecture": architecture,
                "schema": schema,
            },
        }

    @staticmethod
    def _pack_routes(routes: Sequence[Sequence[tuple[int, ...]]], width_of: int = 2):
        """Pad route tuples into one column per coordinate plus a mask."""
        width = max((len(row) for row in routes), default=0)
        columns = [torch.zeros((len(routes), width), dtype=torch.long) for _ in range(width_of)]
        mask = torch.zeros((len(routes), width), dtype=torch.bool)
        for index, values in enumerate(routes):
            if values:
                stacked = torch.tensor(list(values), dtype=torch.long)
                for column, tensor in enumerate(columns):
                    tensor[index, : len(values)] = stacked[:, column]
                mask[index, : len(values)] = True
        return (*columns, mask)

    @staticmethod
    def _pack_record_tensors(records: Sequence[Sequence[GroupRecord]]) -> dict[str, torch.Tensor]:
        batch, width = len(records), max((len(group) for group in records), default=0)
        fields = max((len(spec.fields) for group in records for spec in group), default=0)
        packed = {
            "record_group_index": torch.zeros((batch, width), dtype=torch.long),
            "record_mode_ids": torch.zeros((batch, width), dtype=torch.long),
            "record_anchor_query": torch.full((batch, width), -1, dtype=torch.long),
            "record_field_query": torch.zeros((batch, width, fields), dtype=torch.long),
            "record_field_cardinality": torch.zeros((batch, width, fields), dtype=torch.long),
            "record_field_anchor": torch.zeros((batch, width, fields), dtype=torch.bool),
            "record_field_mask": torch.zeros((batch, width, fields), dtype=torch.bool),
            "record_mask": torch.zeros((batch, width), dtype=torch.bool),
        }
        for row, group in enumerate(records):
            for column, spec in enumerate(group):
                packed["record_mask"][row, column] = True
                packed["record_group_index"][row, column] = spec.task_index
                packed["record_mode_ids"][row, column] = RECORD_MODE_TO_ID[spec.mode]
                if spec.anchor_query_id is not None:
                    packed["record_anchor_query"][row, column] = spec.anchor_query_id
                for index, item in enumerate(spec.fields):
                    packed["record_field_mask"][row, column, index] = True
                    packed["record_field_query"][row, column, index] = item.query_id
                    packed["record_field_cardinality"][row, column, index] = CARDINALITY_TO_ID[item.cardinality]
                    packed["record_field_anchor"][row, column, index] = item.is_anchor
        return packed

    def _collate(self, records: Sequence[Mapping[str, Any]], architecture: str, max_gold_per_query: int | None = None):
        input_ids = pad_sequence(
            [torch.tensor(record["input_ids"], dtype=torch.long) for record in records], batch_first=True
        )
        attention_mask = pad_sequence(
            [torch.ones(len(record["input_ids"]), dtype=torch.long) for record in records], batch_first=True
        )
        word_routes, query_routes, cls_routes, prompt_routes, relation_routes = [], [], [], [], []
        for record in records:
            query_row, cls_row, prompt_row, relation_row = [], [], [], []
            query_cursor = 0
            for group_index, positions in enumerate(record["schema_special_positions"]):
                task = record["task_types"][group_index]
                routes = [(position, group_index) for position in positions[1:]]
                if task == "classifications":
                    cls_row += routes
                    continue
                query_row += routes
                if positions:
                    prompt_row.append((positions[0], group_index))
                field_count = max(len(positions) - 1, 0)
                if task == "relations" and field_count >= 2:
                    relation_row.append((query_cursor, query_cursor + 1, group_index))
                query_cursor += field_count
            word_routes.append([(position, 0) for position in record["text_word_first_positions"]])
            query_routes.append(query_row)
            cls_routes.append(cls_row)
            prompt_routes.append(prompt_row)
            relation_routes.append(relation_row)
        text_word_indices, _, text_word_mask = self._pack_routes(word_routes)
        query_marker_indices, query_group_index, query_marker_mask = self._pack_routes(query_routes)
        cls_marker_indices, cls_group_index, cls_marker_mask = self._pack_routes(cls_routes)
        prompt_marker_indices, prompt_group_index, prompt_marker_mask = self._pack_routes(prompt_routes)
        task_rows = [[(TASK_TYPE_TO_ID[task],) for task in record["task_types"]] for record in records]
        task_type_ids, group_mask = self._pack_routes(task_rows, width_of=1)
        relation_head_index, relation_tail_index, relation_group_index, relation_mask = self._pack_routes(
            relation_routes, width_of=3
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
            data.update(
                self._pack_record_tensors(
                    [[group.record for group in record["groups"] if group.record is not None] for record in records]
                )
            )
        if any(record["labels"] is not None for record in records):
            if any(record["labels"] is None for record in records):
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
        text: str | Sequence[str],
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
        text_kwargs = self._merge_kwargs(
            Gliner2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.__dict__.get("init_kwargs"),
            max_len=max_len,
            architecture=architecture,
            labels=labels,
            **kwargs,
        )["text_kwargs"]
        max_len = text_kwargs.get("max_len", max_len)
        architecture = text_kwargs.get("architecture", architecture)
        labels = text_kwargs.get("labels", labels)
        if return_tensors != "pt" or architecture not in ("span", "boundary") or schema is None:
            raise ValueError(
                "Gliner2Processor needs a schema, `return_tensors='pt'`, and a span or boundary architecture"
            )
        texts = [text] if isinstance(text, str) else list(text)
        schemas = list(schema) if isinstance(schema, (list, tuple)) else [schema] * len(texts)
        if labels is None or isinstance(labels, (list, tuple)):
            label_rows = [None] * len(texts) if labels is None else list(labels)
        else:
            label_rows = [labels] * (len(texts) == 1)
        if not texts or len(schemas) != len(texts) or len(label_rows) != len(texts):
            raise ValueError("text, schema, and labels must be non-empty and have the same batch size")
        records = [
            self._transform_one(*row, max_len, architecture, row_labels)
            for row, row_labels in zip(zip(texts, schemas), label_rows)
        ]
        skipped = ["metadata"] + ["targets"] * any(record["labels"] is not None for record in records)
        return BatchFeature(
            self._collate(records, architecture, text_kwargs.get("max_gold_per_query")),
            tensor_type="pt",
            skip_tensor_conversion=skipped,
        )

    def chunk_words(
        self, words: str | Sequence[Any], chunk_size: int = 384, chunk_overlap: int = 64
    ) -> list[TextChunk]:
        """Split text or words into overlapping windows."""
        return chunk_words(words, chunk_size=chunk_size, chunk_overlap=chunk_overlap, word_splitter=self._splitter)

    def windows(self, text: str, chunk_size: int | None, chunk_overlap: int | None) -> list[TextChunk]:
        """One chunk per word window. A short text, or a non-positive `chunk_size`, stays a single window.

        Args:
            text (`str`):
                Document text.
            chunk_size (`int`, *optional*):
                Word window length.
            chunk_overlap (`int`, *optional*):
                Words shared by neighboring windows.
        """
        chunks = []
        if chunk_size is not None and chunk_size > 0:
            overlap = min(max(chunk_overlap or 0, 0), chunk_size - 1)
            chunks = self.chunk_words(text, chunk_size=chunk_size, chunk_overlap=overlap)
        return chunks if len(chunks) > 1 else [TextChunk(text, 0, len(text), 0, 0)]

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
            include_confidence,
            include_spans,
            scalar_entity_labels,
            overlap_policy,
        )

    def scalar_entity_labels(self, metadata: Any) -> set[str]:
        """Entity names declared with a non-list dtype.

        Args:
            metadata:
                Processor metadata for one window or a batch.
        """
        rows = _metadata_rows(metadata)
        groups = rows[0]["groups"] if rows else ()
        return {
            item.name for group in groups if group.task == "entities" for item in group.fields if item.dtype != "list"
        }

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
        rows = _metadata_rows(metadata)
        entity = next((group for group in rows[0]["groups"] if group.task == "entities"), None) if rows else None
        return (
            getattr(model, "boundary_head", None) is not None
            and bool(entity and entity.attributes)
            and _cached_states(outputs) is not None
        )

    def rescore_attributes(
        self, result: dict[str, Any], model: Any, outputs: Any, metadata: Any, index: int = 0
    ) -> dict[str, Any]:
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
            index (`int`, *optional*, defaults to 0):
                Batch row of `result` in `outputs` and `metadata`.
        """
        if not self.can_rescore_attributes(outputs, metadata, model):
            return result
        row = _metadata_rows(metadata)[index]
        entity = next(group for group in row["groups"] if group.task == "entities")
        states = [
            (state.unsqueeze(0) if state.dim() == dim else state)[index : index + 1]
            for state, dim in zip(_cached_states(outputs), (2, 1, 2, 1))
        ]
        text_states, text_mask, query_states, query_mask = states
        query_of = _entity_query_index(row["groups"])
        prompts = dict(entity.attribute_prompts)
        attribute_rows = []
        for spec in entity.attributes:
            for label in spec.labels:
                query_id = query_of.get(prompts.get(label, label))
                if query_id is not None and query_id < query_states.shape[1]:
                    attribute_rows.append((label, query_id))
        attribute_rows = list(dict.fromkeys(attribute_rows))
        located = []
        for name, item in _entity_nodes(result):
            if name in entity.attribute_labels or "start" not in item or "end" not in item:
                continue
            covered = [
                word
                for word, (start, end) in enumerate(zip(row["start"], row["end"]))
                if int(end) > int(item["start"]) and int(start) < int(item["end"])
            ]
            if covered:
                located.append((name, item, (covered[0] + row["prefix_len"], covered[-1] + 1 + row["prefix_len"])))
        unique_pairs = list(dict.fromkeys(bounds for _, _, bounds in located))
        if not attribute_rows or not unique_pairs:
            return result
        device = model.device
        query_ids = torch.tensor([query_id for _, query_id in attribute_rows], device=device)
        pairs = torch.tensor(unique_pairs, device=device)
        indices = pairs.view(1, 1, len(unique_pairs), 2).expand(1, len(attribute_rows), len(unique_pairs), 2)
        with torch.no_grad():
            logits = model.score_spans(
                text_states.to(device),
                text_mask.to(device),
                query_states.to(device).index_select(1, query_ids),
                query_mask.to(device).index_select(1, query_ids),
                indices.contiguous(),
            )
        pair_temperature = float(getattr(model.config.boundary_config, "pair_temperature", 1.0) or 1.0)
        logits = logits[0].detach().float().cpu() / pair_temperature
        row_of = {label: index for index, (label, _) in enumerate(attribute_rows)}
        column_of = {pair: index for index, pair in enumerate(unique_pairs)}
        for name, item, bounds in located:
            for spec in entity.attributes:
                present = [(label, row_of[label]) for label in spec.labels if label in row_of]
                if present and (spec.applies_to is None or name in spec.applies_to):
                    item[spec.name] = _assign_entity_attributes(logits, column_of[bounds], present, spec)
        return result

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
        """Decode a forward output into the public extraction payload, one dict per text.

        Boundary rows decode forward's candidates, typed relation pairs, and record
        groups. Span rows decode `span_logits` group by group. Both decode label
        groups with each task's activation.

        Args:
            outputs:
                `Gliner2SchemaExtractionOutput`, or one output mapping per text.
            metadata (`list[dict]`):
                `encoding["metadata"]`, aligned with `outputs`.
            threshold (`float`, *optional*, defaults to 0.5):
                Score cutoff. Fields, relations, and multi-label tasks may override it.
            include_confidence (`bool`, *optional*, defaults to `False`):
                Keep scores.
            include_spans (`bool`, *optional*, defaults to `False`):
                Keep character offsets.
            overlap_policy (`str`, *optional*):
                `allow`, `nested`, `disallow`, or `longest`. Boundary rows default to `disallow`.
            temperature (`float`, *optional*, defaults to 1.0):
                Classification temperature. Boundary rows also divide pair logits by it.
        """
        rows = _metadata_rows(metadata)
        formatted = []
        for sample, row in zip(_split_outputs(outputs, len(rows)), rows):
            policy = "disallow" if overlap_policy is None and row["architecture"] == "boundary" else overlap_policy
            options = DecodeOptions(
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
                overlap=None if policy is None else normalize_overlap_policy(policy),
                temperature=temperature,
            )
            formatted.append(
                format_results(
                    _decode_sample(sample, row, options),
                    include_confidence=include_confidence,
                    requested_relations=_group_names(row, "relations"),
                    classification_tasks=_group_names(row, "classifications"),
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
        """Decode classification logits, optionally under the schema's cross-task constraints.

        `decoder="beam"` or `"exact"`, or schema `constraints` with any decoder other
        than `"independent"`, run `decode_constrained_classification`. Otherwise each
        task is decoded on its own.

        Args:
            outputs:
                Model output with classification logits.
            metadata (`list[dict]`):
                `encoding["metadata"]`, aligned with `outputs`.
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
        constrained = decoder != "independent" and (
            decoder in ("beam", "exact") or any(row["schema"].get("constraints") for row in rows)
        )
        results = []
        for sample, row in zip(_split_outputs(outputs, len(rows)), rows):
            labels = [group for group in row["groups"] if group.task == "classifications"]
            if constrained:
                schema = {
                    "classifications": [_classification_config(group) for group in labels],
                    "constraints": list(row["schema"].get("constraints") or []),
                }
                results.append(
                    decode_constrained_classification(
                        _task_logit_map(sample, labels),
                        schema,
                        temperature,
                        text=row["text"],
                        decoder=decoder,
                        include_confidence=include_confidence,
                        candidate_threshold=threshold,
                    )
                )
                continue
            options = DecodeOptions(
                threshold=threshold, include_confidence=include_confidence, temperature=temperature
            )
            raw = decode_fields(labels, {"classification_logits": sample.get("classification_logits")}, row, options)
            results.append(
                format_results(raw, include_confidence, classification_tasks=_group_names(row, "classifications"))
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
        """Decode entities and relations together under the schema's joint constraints.

        `optimizer="beam"`, `"greedy"`, and `"exact"` run `decode_joint_sample`.
        `"independent"` and `"auto"` fall back to `post_process_extraction`.

        Args:
            outputs:
                Model output.
            metadata (`list[dict]`):
                `encoding["metadata"]`, aligned with `outputs`.
            threshold (`float`, *optional*, defaults to 0.5):
                Score cutoff.
            include_confidence (`bool`, *optional*, defaults to `False`):
                Keep scores.
            include_spans (`bool`, *optional*, defaults to `False`):
                Keep character offsets.
            optimizer (`str`, *optional*, defaults to `"independent"`):
                `independent`, `auto`, `beam`, `greedy`, or `exact`.
            overlap_policy (`str`, *optional*):
                Entity overlap rule.
        """
        if optimizer in ("beam", "exact", "greedy"):
            rows = _metadata_rows(metadata)
            return [
                decode_joint_sample(
                    sample,
                    row,
                    threshold=threshold,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                    optimizer=optimizer,
                    overlap_policy=overlap_policy,
                )
                for sample, row in zip(_split_outputs(outputs, len(rows)), rows)
            ]
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
