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
from types import SimpleNamespace
from typing import Any

from ..utils import is_torch_available
from .base import ChunkPipeline


if is_torch_available():
    import torch


class SchemaExtractionPipeline(ChunkPipeline):
    """Extract entities, labels, relations, and records for one schema."""

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
        """Split schema and windowing from the decode flags."""
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

    def preprocess(self, inputs, schema=None, max_len=None, chunk_size=384, chunk_overlap=64):
        """Encode one word window at a time and mark the last window."""
        if schema is None:
            raise ValueError("schema-extraction requires a schema dict.")
        architecture = getattr(self.model.config, "architecture", "span")
        document = inputs if isinstance(inputs, str) else str(inputs)
        chunks = _windows(self.processor, document, chunk_size, chunk_overlap)
        last = len(chunks) - 1
        for index, chunk in enumerate(chunks):
            encoding = self.processor(
                chunk.text,
                schema=schema,
                return_tensors="pt",
                max_len=max_len,
                architecture=architecture,
            )
            metadata = encoding.pop("metadata")
            yield {
                "model_inputs": encoding,
                "metadata": metadata,
                "chunk": chunk,
                "document_text": document,
                "is_last": index == last,
            }

    def _forward(self, model_inputs):
        """Run one window and forward is_last so chunks regroup."""
        metadata = model_inputs.pop("metadata")
        chunk = model_inputs.pop("chunk")
        document_text = model_inputs.pop("document_text", "")
        is_last = model_inputs.pop("is_last")
        outputs = self.model(**model_inputs["model_inputs"])
        return {
            "outputs": outputs,
            "metadata": metadata,
            "chunk": chunk,
            "document_text": document_text,
            "is_last": is_last,
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
        """Decode each window, rescore attributes, and merge the chunks."""
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
        first = windows[0]
        return self.processor.merge_chunk_results(
            first.get("document_text", "") if isinstance(first, Mapping) else "",
            [window.get("chunk") if isinstance(window, Mapping) else window for window in windows],
            decoded,
            include_confidence=include_confidence,
            include_spans=include_spans,
            scalar_entity_labels=_scalar_entity_labels(first.get("metadata") if isinstance(first, Mapping) else None),
            overlap_policy=self._overlap_policy(),
        )

    def _classification_temperature(self) -> float:
        """Read the boundary temperature, otherwise the span temperature."""
        config = self.model.config
        if getattr(config, "architecture", "span") == "boundary":
            boundary = getattr(config, "boundary_config", None)
            if boundary is not None:
                return float(getattr(boundary, "classification_temperature", 1.0) or 1.0)
        return float(getattr(config, "classification_temperature", 1.0) or 1.0)

    def _overlap_policy(self):
        """Return the boundary overlap policy. Span models leave it unset."""
        config = self.model.config
        if getattr(config, "architecture", "span") != "boundary":
            return None
        boundary = getattr(config, "boundary_config", None)
        if boundary is None:
            return None
        return getattr(boundary, "overlap_policy", None)

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
        temperature = self._classification_temperature()
        overlap_policy = self._overlap_policy()
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
                overlap_policy=overlap_policy,
            )
        else:
            decoded = self.processor.post_process_extraction(
                outputs,
                metadata,
                threshold=threshold,
                include_confidence=use_confidence,
                include_spans=use_spans,
                overlap_policy=overlap_policy,
                temperature=temperature,
            )
        decoded = decoded[0]
        if rescore:
            decoded = self._attach_attributes(decoded, outputs, metadata)
        return decoded

    def _can_rescore_attributes(self, outputs, metadata) -> bool:
        """Return whether this window can rescore attributes without another encoder pass."""
        if not is_torch_available() or getattr(self.model, "boundary_head", None) is None:
            return False
        groups = _schema_meta(metadata).get("entity_attribute_groups") or {}
        return bool(groups) and _cached_states(outputs) is not None

    def _attach_attributes(self, decoded, outputs, metadata) -> dict[str, Any]:
        """Write attribute labels onto kept entity spans using score_spans."""
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
        boundary = getattr(self.model.config, "boundary_config", None)
        temperature = float(getattr(boundary, "pair_temperature", 1.0) or 1.0)
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


def _windows(processor, text: str, chunk_size: int | None, chunk_overlap: int | None):
    """One chunk per word window. A short text stays a single window."""
    whole = SimpleNamespace(text=text, start_char=0, end_char=len(text))
    if chunk_size is None or chunk_size <= 0:
        return [whole]
    overlap = 0 if chunk_overlap is None or chunk_overlap < 0 else chunk_overlap
    if overlap >= chunk_size:
        overlap = chunk_size - 1
    chunks = processor.chunk_words(text, chunk_size=chunk_size, chunk_overlap=overlap)
    if len(chunks) <= 1:
        return [whole]
    return list(chunks)


def _read(value, name):
    """Read a mapping key or an attribute."""
    if isinstance(value, Mapping) and name in value:
        return value[name]
    return getattr(value, name, None)


def _cached_states(outputs):
    """Return word and query states cached on the forward output."""
    states = _states_from(outputs)
    if states is not None:
        return states
    return _states_from(_read(outputs, "boundary"))


def _states_from(value):
    """Read the four state tensors from one output object."""
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
    """Add a batch axis and keep the first row."""
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
    """Return the schema metadata dict for one window."""
    if isinstance(metadata, (list, tuple)):
        metadata = metadata[0] if metadata else {}
    if not isinstance(metadata, Mapping):
        return {}
    if "schema_meta" in metadata:
        meta = dict(metadata["schema_meta"])
        meta.setdefault("words", list(metadata.get("words") or []))
        return meta
    return dict(metadata)


def _scalar_entity_labels(metadata) -> set[str]:
    """Names of entity types declared with a non-list dtype."""
    labels = set()
    for name, spec in (_schema_meta(metadata).get("entity_metadata") or {}).items():
        if isinstance(spec, Mapping) and spec.get("dtype", "list") != "list":
            labels.add(str(name))
    return labels


def _field(group, name, default=None):
    """Read a field from a mapping or an object."""
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
    """Yield entity name and span-dict pairs."""
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
