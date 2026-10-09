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

from .base import ChunkPipeline


def _setting(config: Any, name: str, default: Any) -> Any:
    """Read one config field from a mapping or an object."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return config.__dict__.get(name, default)


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
        """Split schema and windowing from the decode flags.

        Args:
            schema (`dict`, *optional*):
                Schema passed through to preprocessing.
            threshold (`float`, *optional*):
                Score cutoff forwarded to postprocessing.
            include_confidence (`bool`, *optional*):
                Whether decoded values keep scores.
            include_spans (`bool`, *optional*):
                Whether decoded values keep character offsets.
            constraints (`list`, *optional*):
                Cross-task classification constraints.
            joint (`bool`, *optional*):
                Decode entities and relations together.
            max_len (`int`, *optional*):
                Maximum document words per window.
            chunk_size (`int`, *optional*):
                Word window length.
            chunk_overlap (`int`, *optional*):
                Overlap between word windows.
            **kwargs:
                Extra decode flags forwarded to postprocessing.
        """
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
        """Encode one word window at a time and mark the last window.

        Args:
            inputs (`str`):
                Document text.
            schema (`dict`):
                Extraction schema.
            max_len (`int`, *optional*):
                Maximum document words encoded in each window.
            chunk_size (`int`, *optional*, defaults to 384):
                Word window length.
            chunk_overlap (`int`, *optional*, defaults to 64):
                Overlap between word windows.
        """
        if schema is None:
            raise ValueError("schema-extraction requires a schema dict.")
        architecture = self.model.config.architecture
        document = inputs if isinstance(inputs, str) else str(inputs)
        chunks = self.processor.windows(document, chunk_size, chunk_overlap)
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
        """Run one window and forward is_last so chunks regroup.

        Args:
            model_inputs (`dict`):
                One preprocess yield, including `model_inputs` and `is_last`.
        """
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

    def _classification_temperature(self) -> float:
        """Read the boundary temperature, otherwise the span temperature."""
        config = self.model.config
        if config.architecture == "boundary":
            boundary = config.boundary_config
            return float(_setting(boundary, "classification_temperature", 1.0) or 1.0)
        return float(config.classification_temperature or 1.0)

    def _overlap_policy(self):
        """Return the boundary overlap policy. Span models leave it unset."""
        config = self.model.config
        if config.architecture != "boundary":
            return None
        return _setting(config.boundary_config, "overlap_policy", None)

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
        """Decode each window, rescore attributes, and merge the chunks.

        Args:
            model_outputs (`dict` or `list`):
                Forward outputs for one document's windows.
            threshold (`float`, *optional*, defaults to 0.5):
                Score cutoff.
            include_confidence (`bool`, *optional*, defaults to `False`):
                Keep scores on the merged result.
            include_spans (`bool`, *optional*, defaults to `False`):
                Keep character offsets on the merged result.
            constraints (`list`, *optional*):
                When set, decode labels with the constrained classifier.
            joint (`bool`, *optional*, defaults to `False`):
                Decode entities and relations with the joint optimizer.
            **kwargs:
                `decoder` and `optimizer` flags for those wrappers.
        """
        windows = model_outputs if isinstance(model_outputs, list) else [model_outputs]
        multiple = len(windows) > 1
        temperature = self._classification_temperature()
        overlap_policy = self._overlap_policy()
        decoded = []
        for window in windows:
            outputs = window["outputs"] if isinstance(window, Mapping) else window
            metadata = window.get("metadata") if isinstance(window, Mapping) else None
            rescore = self.processor.can_rescore_attributes(outputs, metadata, self.model)
            use_spans = include_spans or multiple or rescore
            use_confidence = include_confidence or multiple or rescore
            if constraints is not None:
                piece = self.processor.post_process_constrained_classification(
                    outputs,
                    metadata,
                    threshold=threshold,
                    include_confidence=use_confidence,
                    temperature=temperature,
                    decoder=kwargs.get("decoder", "auto"),
                )
            elif joint:
                piece = self.processor.post_process_joint_extraction(
                    outputs,
                    metadata,
                    threshold=threshold,
                    include_confidence=use_confidence,
                    include_spans=use_spans,
                    optimizer=kwargs.get("optimizer", "beam"),
                    overlap_policy=overlap_policy,
                )
            else:
                piece = self.processor.post_process_extraction(
                    outputs,
                    metadata,
                    threshold=threshold,
                    include_confidence=use_confidence,
                    include_spans=use_spans,
                    overlap_policy=overlap_policy,
                    temperature=temperature,
                )
            piece = piece[0]
            if rescore:
                piece = self.processor.rescore_attributes(piece, self.model, outputs, metadata)
            decoded.append(piece)
        first = windows[0]
        return self.processor.merge_chunk_results(
            first.get("document_text", "") if isinstance(first, Mapping) else "",
            [window.get("chunk") if isinstance(window, Mapping) else window for window in windows],
            decoded,
            include_confidence=include_confidence,
            include_spans=include_spans,
            scalar_entity_labels=self.processor.scalar_entity_labels(
                first.get("metadata") if isinstance(first, Mapping) else None
            ),
            overlap_policy=overlap_policy,
        )
