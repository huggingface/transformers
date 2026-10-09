# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from collections.abc import Mapping
from typing import Any

from ..utils import is_torch_available
from .base import ChunkPipeline


if is_torch_available():
    import torch


_SPAN_KEYS = ("text", "confidence", "start", "end")


class SchemaExtractionPipeline(ChunkPipeline):
    """Extract entities, labels, relations, and records from text and a schema.

    Uses [`AutoModelForSchemaExtraction`]. Every label in the schema is scored in one forward pass.
    Text longer than `chunk_size` words is split into overlapping windows. Attribute labels on a
    boundary model are rescored with [`~Gliner2ForSchemaExtraction.score_spans`] on the word states
    from that forward when the output already carries them.
    """

    _load_processor = True
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False

    def _sanitize_parameters(
        self,
        schema=None,
        threshold=None,
        include_confidence=None,
        include_spans=None,
        constraints=None,
        joint=None,
        max_len=None,
        chunk_size=None,
        chunk_overlap=None,
        **kwargs,
    ):
        preprocess_params = {}
        if schema is not None:
            preprocess_params["schema"] = schema
        if max_len is not None:
            preprocess_params["max_len"] = max_len
        if chunk_size is not None:
            preprocess_params["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            preprocess_params["chunk_overlap"] = chunk_overlap
        postprocess_params = {}
        if threshold is not None:
            postprocess_params["threshold"] = threshold
        if include_confidence is not None:
            postprocess_params["include_confidence"] = include_confidence
        if include_spans is not None:
            postprocess_params["include_spans"] = include_spans
        if constraints is not None:
            postprocess_params["constraints"] = constraints
        if joint is not None:
            postprocess_params["joint"] = joint
        postprocess_params.update(kwargs)
        return preprocess_params, {}, postprocess_params

    def __call__(self, inputs, schema=None, **kwargs):
        """
        Extract structured information from text.

        Args:
            inputs (`str` or `list[str]`):
                One text or a batch of texts.
            schema (`dict`):
                A GLiNER2 schema. Keys include `entities`, `classifications`, `relations`,
                `json_structures`, and `attributes`.
            threshold (`float`, *optional*, defaults to 0.5):
                Span score cutoff.
            include_confidence (`bool`, *optional*, defaults to `False`):
                Include scores in the result.
            include_spans (`bool`, *optional*, defaults to `False`):
                Include character offsets in the result.
            constraints (`dict`, *optional*):
                Classification constraints. Selects the constrained decoder.
            joint (`bool`, *optional*, defaults to `False`):
                Decode entities, structures, and relations together.
            max_len (`int`, *optional*):
                Maximum tokens in one window.
            chunk_size (`int`, *optional*, defaults to 384):
                Maximum words in one window. Shorter text stays a single window.
            chunk_overlap (`int`, *optional*, defaults to 64):
                Words shared by adjacent windows.

        Returns:
            `dict` or `list[dict]`: The same result dictionaries the GLiNER2 package returns.
        """
        if schema is None and "schema" not in kwargs:
            raise ValueError("schema-extraction requires a schema dict.")
        single = isinstance(inputs, str)
        outputs = super().__call__(inputs, schema=schema, **kwargs)
        if single and isinstance(outputs, list) and len(outputs) == 1:
            return outputs[0]
        return outputs

    def preprocess(self, inputs, schema=None, max_len=None, chunk_size=384, chunk_overlap=64):
        """Encode one window, or overlapping word windows when the text is longer than `chunk_size`."""
        architecture = getattr(self.model.config, "architecture", "span")
        document = inputs if isinstance(inputs, str) else str(inputs)
        for window_text, start_char in _iter_windows(self.processor, document, chunk_size, chunk_overlap):
            encoding = self.processor(
                window_text,
                schema=schema,
                return_tensors="pt",
                max_len=max_len,
                architecture=architecture,
            )
            metadata = encoding.pop("metadata")
            yield {
                "model_inputs": encoding,
                "metadata": metadata,
                "window_start": start_char,
                "document_text": document,
            }

    def get_iterator(self, inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params):
        """Score each text in one forward. The generic tokenizer collate does not pad these tensors."""
        del num_workers, batch_size
        for text in inputs:
            yield self.run_single(text, preprocess_params, forward_params, postprocess_params)

    def _forward(self, model_inputs):
        metadata = model_inputs.pop("metadata")
        window_start = int(model_inputs.pop("window_start", 0))
        document_text = model_inputs.pop("document_text", "")
        outputs = self.model(**model_inputs["model_inputs"])
        return {
            "outputs": outputs,
            "metadata": metadata,
            "window_start": window_start,
            "document_text": document_text,
        }

    def postprocess(
        self,
        model_outputs,
        threshold=0.5,
        include_confidence=False,
        include_spans=False,
        constraints=None,
        joint=False,
        **kwargs,
    ) -> dict[str, Any]:
        """Decode each window and merge long documents by task."""
        windows = model_outputs if isinstance(model_outputs, list) else [model_outputs]
        multiple = len(windows) > 1
        decoded = [
            self._decode_window(
                window,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
                constraints=constraints,
                joint=joint,
                retain_offsets=multiple,
                **kwargs,
            )
            for window in windows
        ]
        if not multiple:
            return decoded[0]
        document = windows[0].get("document_text", "") if isinstance(windows[0], Mapping) else ""
        shifted = [
            _shift_spans(result, int(window.get("window_start", 0)) if isinstance(window, Mapping) else 0, document)
            for result, window in zip(decoded, windows)
        ]
        tasks = _classification_tasks(windows[0].get("metadata") if isinstance(windows[0], Mapping) else None)
        if joint:
            merged = _merge_joint(shifted, tasks)
        else:
            merged = _merge_windows(shifted, tasks)
        return _strip_result(merged, include_confidence, include_spans)

    def _decode_window(
        self,
        model_outputs,
        threshold=0.5,
        include_confidence=False,
        include_spans=False,
        constraints=None,
        joint=False,
        retain_offsets=False,
        **kwargs,
    ) -> dict[str, Any]:
        """Decode one window and rescore boundary attributes on cached states."""
        outputs = model_outputs["outputs"]
        metadata = model_outputs["metadata"]
        rescore = self._can_rescore_attributes(outputs, metadata)
        use_spans = include_spans or retain_offsets or rescore
        use_confidence = include_confidence or retain_offsets or rescore
        temperature = float(getattr(self.model.config, "classification_temperature", 1.0))
        if constraints is not None:
            decoded = self.processor.post_process_constrained_classification(
                outputs,
                metadata,
                threshold=threshold,
                include_confidence=use_confidence,
                temperature=temperature,
                decoder=kwargs.get("decoder", "auto"),
            )
        elif joint:
            decoded = self.processor.post_process_joint_extraction(
                outputs,
                metadata,
                threshold=threshold,
                include_confidence=use_confidence,
                include_spans=use_spans,
                optimizer=kwargs.get("optimizer", "beam"),
            )
        else:
            decoded = self.processor.post_process_extraction(
                outputs,
                metadata,
                threshold=threshold,
                include_confidence=use_confidence,
                include_spans=use_spans,
                temperature=temperature,
            )
        decoded = decoded[0]
        if rescore:
            decoded = self._attach_attributes(decoded, outputs, metadata)
        if not retain_offsets and (use_spans != include_spans or use_confidence != include_confidence):
            decoded = _strip_result(decoded, include_confidence, include_spans)
        return decoded

    def _can_rescore_attributes(self, outputs, metadata) -> bool:
        """Return whether this window can rescore attributes without another encoder pass."""
        if not is_torch_available() or getattr(self.model, "boundary_head", None) is None:
            return False
        groups = _schema_meta(metadata).get("entity_attribute_groups") or {}
        return bool(groups) and _cached_states(outputs) is not None

    def _attach_attributes(self, decoded, outputs, metadata) -> dict[str, Any]:
        """Write attribute labels onto kept entity spans using `score_spans`."""
        cached = _cached_states(outputs)
        if cached is None or not is_torch_available():
            return decoded
        meta = _schema_meta(metadata)
        groups = meta.get("entity_attribute_groups") or {}
        if not isinstance(groups, Mapping) or not groups:
            return decoded
        text_states, text_mask, query_states, query_mask = _batch_states(*cached)
        query_of = _entity_query_index(meta)
        prompt_labels = meta.get("entity_attribute_prompt_labels") or {}
        if not isinstance(prompt_labels, Mapping):
            prompt_labels = {}
        attribute_rows = []
        for group in groups.values():
            for label in _field(group, "labels", []) or []:
                query_id = query_of.get(prompt_labels.get(label, label))
                if query_id is not None and query_id < query_states.shape[1]:
                    attribute_rows.append((label, query_id))
        attribute_rows = list(dict.fromkeys(attribute_rows))
        nodes = [
            (name, item)
            for name, item in _entity_nodes(decoded)
            if name not in set(meta.get("entity_attribute_labels") or ())
        ]
        start_map = list(meta.get("start") or [])
        end_map = list(meta.get("end") or [])
        prefix = int(meta.get("prefix_len") or 0)
        located = []
        for name, item in nodes:
            if "start" not in item or "end" not in item:
                continue
            bounds = _word_bounds(int(item["start"]), int(item["end"]), start_map, end_map, prefix)
            if bounds is not None:
                located.append((name, item, bounds))
        unique_pairs = list(dict.fromkeys(bounds for _, _, bounds in located))
        if not attribute_rows or not unique_pairs:
            return decoded
        device = self.model.device
        with torch.no_grad():
            text_states = text_states.to(device)
            text_mask = text_mask.to(device)
            query_ids = torch.tensor([query_id for _, query_id in attribute_rows], dtype=torch.long, device=device)
            query_states = query_states.to(device).index_select(1, query_ids)
            query_mask = query_mask.to(device).index_select(1, query_ids)
            pairs = torch.tensor(unique_pairs, dtype=torch.long, device=device)
            indices = pairs.view(1, 1, len(unique_pairs), 2).expand(1, len(attribute_rows), len(unique_pairs), 2)
            logits = self.model.score_spans(text_states, text_mask, query_states, query_mask, indices.contiguous())
        logits = logits[0].detach().float().cpu()
        temperature = float(getattr(self.model.config, "pair_temperature", 1.0) or 1.0)
        logits = logits / temperature
        row_of = {label: row for row, (label, _) in enumerate(attribute_rows)}
        column_of = {pair: index for index, pair in enumerate(unique_pairs)}
        for name, item, bounds in located:
            column = column_of[bounds]
            for group_name, group in groups.items():
                applies_to = _field(group, "applies_to")
                if applies_to is not None and name not in applies_to:
                    continue
                present = [(label, row_of[label]) for label in (_field(group, "labels", []) or []) if label in row_of]
                if not present:
                    continue
                labels, rows = zip(*present)
                values = logits[list(rows), column]
                if _field(group, "multi_label", False):
                    probabilities = torch.sigmoid(values)
                    threshold = float(_field(group, "threshold", 0.5))
                    item[group_name] = [
                        {"label": label, "confidence": float(probabilities[index])}
                        for index, label in enumerate(labels)
                        if float(probabilities[index]) >= threshold
                    ]
                else:
                    probabilities = torch.softmax(values, dim=-1)
                    best = int(probabilities.argmax())
                    item[group_name] = {"label": labels[best], "confidence": float(probabilities[best])}
        return decoded


def _iter_windows(processor, text: str, chunk_size: int | None, chunk_overlap: int | None):
    """Yield `(window_text, start_char)`. A short string is yielded unchanged."""
    if chunk_size is None or chunk_size <= 0:
        yield text, 0
        return
    overlap = 0 if chunk_overlap is None or chunk_overlap < 0 else chunk_overlap
    if overlap >= chunk_size:
        overlap = chunk_size - 1
    chunk_words = getattr(processor, "chunk_words", None)
    chunks = chunk_words(text, chunk_size=chunk_size, chunk_overlap=overlap) if callable(chunk_words) else None
    if not chunks or len(chunks) <= 1:
        yield text, 0
        return
    for chunk in chunks:
        yield chunk.text, int(chunk.start_char)


def _read(value, name):
    if isinstance(value, Mapping) and name in value:
        return value[name]
    return getattr(value, name, None)


def _cached_states(outputs):
    """Return word and query states when the forward output already has them."""
    states = _states_from(outputs)
    if states is not None:
        return states
    return _states_from(_read(outputs, "boundary"))


def _states_from(value):
    if value is None or not is_torch_available():
        return None
    text_states = _read(value, "text_states")
    text_mask = _read(value, "text_mask")
    if text_mask is None:
        text_mask = _read(value, "text_word_mask")
    query_states = _read(value, "query_states")
    query_mask = _read(value, "query_mask")
    if query_mask is None:
        query_mask = _read(value, "query_marker_mask")
    tensors = (text_states, text_mask, query_states, query_mask)
    if any(item is None or not torch.is_tensor(item) for item in tensors):
        return None
    return tensors


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


def _schema_meta(metadata) -> dict[str, Any]:
    if isinstance(metadata, (list, tuple)):
        metadata = metadata[0] if metadata else {}
    if not isinstance(metadata, Mapping):
        return {}
    if "schema_meta" in metadata:
        meta = dict(metadata["schema_meta"])
        meta.setdefault("words", list(metadata.get("words") or []))
        return meta
    return dict(metadata)


def _classification_tasks(metadata) -> set[str]:
    meta = _schema_meta(metadata)
    tasks = list(meta.get("classification_tasks") or [])
    if not tasks:
        tasks = [
            item.get("task")
            for item in (meta.get("classifications") or [])
            if isinstance(item, Mapping) and item.get("task")
        ]
    return {task for task in tasks if task}


def _field(group, name, default=None):
    if isinstance(group, Mapping):
        return group.get(name, default)
    return getattr(group, name, default)


def _entity_query_index(meta) -> dict[str, int]:
    """Map entity field names to their index on the query axis."""
    index = {}
    cursor = 0
    for group in meta.get("groups") or []:
        if group.get("task_type") == "classifications":
            continue
        for field in group.get("fields") or []:
            if group.get("task_type") == "entities":
                index[field] = cursor
            cursor += 1
    return index


def _entity_nodes(decoded):
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


def _is_span(value) -> bool:
    return (
        isinstance(value, dict)
        and "text" in value
        and isinstance(value.get("start"), int)
        and isinstance(value.get("end"), int)
    )


def _is_joint_graph(result) -> bool:
    return (
        isinstance(result, Mapping)
        and isinstance(result.get("entities"), list)
        and isinstance(result.get("relations"), list)
    )


def _confidence(value) -> float:
    if isinstance(value, Mapping):
        score = value.get("confidence")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
        nested = [_confidence(item) for item in value.values()]
        return max(nested) if nested else 0.0
    if isinstance(value, list):
        nested = [_confidence(item) for item in value]
        return max(nested) if nested else 0.0
    return 0.0


def _is_classification(value) -> bool:
    return isinstance(value, Mapping) and "label" in value and "confidence" in value and "start" not in value


def _shift_spans(value, start_char: int, document: str):
    """Move window-local character offsets onto the original document."""
    if isinstance(value, list):
        return [_shift_spans(item, start_char, document) for item in value]
    if not isinstance(value, dict):
        return value
    shifted = {key: _shift_spans(item, start_char, document) for key, item in value.items()}
    if not _is_span(shifted):
        return shifted
    start = int(shifted["start"]) + start_char
    end = int(shifted["end"]) + start_char
    shifted["start"] = start
    shifted["end"] = end
    if document and 0 <= start <= end <= len(document):
        shifted["text"] = document[start:end]
    return shifted


def _merge_windows(decoded: list[dict], classification_tasks: set[str]) -> dict:
    """Prefer the higher-confidence label and dedupe spans by text and offsets."""
    merged: dict[str, Any] = {}
    keys: list[str] = []
    seen = set()
    for result in decoded:
        for key in result:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    for key in keys:
        values = [result[key] for result in decoded if key in result]
        merged[key] = _merge_values(key, values, classification_tasks)
    return merged


def _merge_values(key: str, values: list, classification_tasks: set[str]):
    present = [value for value in values if value not in (None, "", [], {})]
    if not present:
        return values[0] if values else None
    if key in classification_tasks or all(_is_classification(value) for value in present):
        return _prefer_classification(present)
    if key == "entities" and all(isinstance(value, Mapping) for value in present):
        return _merge_entity_dicts(present)
    if key == "relation_extraction" and all(isinstance(value, Mapping) for value in present):
        return _merge_relation_dicts(present)
    if all(_is_span(value) for value in present):
        return max(_dedupe_items(present), key=_confidence)
    if all(isinstance(value, list) for value in present):
        return _dedupe_items([item for value in present for item in value])
    if all(isinstance(value, Mapping) for value in present):
        return _merge_windows(present, classification_tasks)
    return _prefer_classification(present)


def _prefer_classification(values: list):
    """Keep the label from the window with the higher confidence."""
    if all(_is_classification(value) for value in values):
        return max(values, key=_confidence)
    if all(isinstance(value, list) for value in values):
        best = {}
        order = []
        for value in values:
            for item in value:
                if _is_classification(item):
                    label = item["label"]
                    score = _confidence(item)
                else:
                    label = item
                    score = 0.0
                if label not in best:
                    order.append(label)
                    best[label] = (score, item)
                elif score > best[label][0]:
                    best[label] = (score, item)
        return [best[label][1] for label in order]
    return max(values, key=_confidence)


def _merge_entity_dicts(values: list) -> dict:
    labels: list[str] = []
    seen = set()
    for value in values:
        for label in value:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    merged = {}
    for label in labels:
        items = []
        scalar = True
        for value in values:
            if label not in value:
                continue
            entry = value[label]
            if isinstance(entry, list):
                scalar = False
                items.extend(entry)
            elif entry not in (None, ""):
                items.append(entry)
        deduped = _dedupe_items(items)
        if scalar:
            merged[label] = max(deduped, key=_confidence) if deduped else None
        else:
            merged[label] = deduped
    return merged


def _merge_relation_dicts(values: list) -> dict:
    """Dedupe relations that were decoded inside a window. Do not pair across windows."""
    labels: list[str] = []
    seen = set()
    for value in values:
        for label in value:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    merged = {}
    for label in labels:
        items = []
        for value in values:
            if label not in value:
                continue
            entry = value[label]
            items.extend(entry if isinstance(entry, list) else [entry])
        merged[label] = _dedupe_items(items)
    return merged


def _dedupe_items(items: list) -> list:
    """Collapse repeats that share text and character offsets, keeping the higher score."""
    best = {}
    order = []
    for item in items:
        if _is_span(item):
            key = (item.get("text"), int(item["start"]), int(item["end"]))
        else:
            key = _identity(item)
        score = _confidence(item)
        if key not in best:
            order.append(key)
            best[key] = (score, item)
        elif score > best[key][0]:
            best[key] = (score, item)
    return [best[key][1] for key in order]


def _identity(value):
    if isinstance(value, Mapping):
        return tuple(
            (key, _identity(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key != "confidence"
        )
    if isinstance(value, list):
        return tuple(_identity(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_identity(item) for item in value)
    return value


def _merge_joint(decoded: list[dict], classification_tasks: set[str]) -> dict:
    """Merge joint spans. Relations stay inside the window that decoded both ends."""
    if not decoded or not all(_is_joint_graph(item) for item in decoded):
        return _merge_windows(decoded, classification_tasks)
    entity_by_key = {}
    relation_rows = {}
    for result in decoded:
        local = {}
        for entity in result.get("entities") or []:
            if not isinstance(entity, Mapping) or "start" not in entity or "end" not in entity:
                continue
            key = (
                str(entity.get("type", entity.get("label", ""))),
                str(entity.get("text", "")),
                int(entity["start"]),
                int(entity["end"]),
            )
            entity_id = entity.get("id") or id(entity)
            local[str(entity_id)] = key
            score = _confidence(entity)
            previous = entity_by_key.get(key)
            if previous is None or score > previous[0]:
                entity_by_key[key] = (score, dict(entity))
        for relation in result.get("relations") or []:
            if not isinstance(relation, Mapping):
                continue
            head = str(relation.get("head"))
            tail = str(relation.get("tail"))
            if head not in local or tail not in local:
                continue
            rel_key = (str(relation.get("type", relation.get("label", ""))), local[head], local[tail])
            score = _confidence(relation)
            previous = relation_rows.get(rel_key)
            if previous is None or score > previous[0]:
                relation_rows[rel_key] = (score, dict(relation), local[head], local[tail])
    ordered = sorted(entity_by_key, key=lambda key: (key[2], key[3], key[0], key[1]))
    ids = {key: f"e{index + 1}" for index, key in enumerate(ordered)}
    entities = []
    for key in ordered:
        item = entity_by_key[key][1]
        item["id"] = ids[key]
        entities.append(item)
    relations = []
    for rel_key, (_, relation, head_key, tail_key) in relation_rows.items():
        relation["type"] = rel_key[0]
        relation["head"] = ids[head_key]
        relation["tail"] = ids[tail_key]
        relations.append(relation)
    relations.sort(
        key=lambda value: (str(value.get("type", "")), str(value.get("head", "")), str(value.get("tail", "")))
    )
    extra = _merge_windows(
        [{key: value for key, value in result.items() if key not in ("entities", "relations")} for result in decoded],
        classification_tasks,
    )
    extra["entities"] = entities
    extra["relations"] = relations
    return extra


def _strip_result(result, include_confidence: bool, include_spans: bool):
    if _is_joint_graph(result):
        stripped = {
            key: _strip_extraction(value, include_confidence, include_spans)
            for key, value in result.items()
            if key not in ("entities", "relations")
        }
        stripped["entities"] = [
            _strip_joint_entity(entity, include_confidence, include_spans) for entity in result["entities"]
        ]
        stripped["relations"] = [
            _strip_joint_relation(relation, include_confidence) for relation in result["relations"]
        ]
        return stripped
    return _strip_extraction(result, include_confidence, include_spans)


def _strip_joint_entity(entity, include_confidence: bool, include_spans: bool) -> dict:
    item = {"id": entity.get("id"), "type": entity.get("type", entity.get("label")), "text": entity.get("text", "")}
    if include_spans:
        item["start"] = entity.get("start")
        item["end"] = entity.get("end")
        if entity.get("sentence_id") is not None:
            item["sentence_id"] = entity["sentence_id"]
    if include_confidence and "confidence" in entity:
        item["confidence"] = entity["confidence"]
    if entity.get("rescued"):
        item["rescued"] = True
    for key, value in entity.items():
        if key not in {"id", "type", "label", "text", "start", "end", "confidence", "sentence_id", "rescued"}:
            item[key] = value
    return item


def _strip_joint_relation(relation, include_confidence: bool) -> dict:
    item = {
        "type": relation.get("type", relation.get("label")),
        "head": relation.get("head"),
        "tail": relation.get("tail"),
    }
    if include_confidence and "confidence" in relation:
        item["confidence"] = relation["confidence"]
    if relation.get("derived"):
        item["derived"] = True
    return item


def _strip_extraction(value, include_confidence: bool, include_spans: bool):
    if isinstance(value, list):
        return [_strip_extraction(item, include_confidence, include_spans) for item in value]
    if not isinstance(value, dict):
        return value
    if _is_span(value):
        extras = {key: item for key, item in value.items() if key not in _SPAN_KEYS}
        if not include_confidence and not include_spans and not extras:
            return value.get("text", "")
        stripped = {"text": value.get("text", "")}
        if include_confidence and "confidence" in value:
            stripped["confidence"] = value["confidence"]
        if include_spans:
            stripped["start"] = value["start"]
            stripped["end"] = value["end"]
        stripped.update(extras)
        return stripped
    if _is_classification(value):
        if include_confidence:
            return {"label": value["label"], "confidence": value["confidence"]}
        return value["label"]
    if "text" in value and "confidence" in value and "start" not in value and "end" not in value:
        if include_confidence:
            return {"text": value["text"], "confidence": value["confidence"]}
        return value["text"]
    return {key: _strip_extraction(item, include_confidence, include_spans) for key, item in value.items()}
