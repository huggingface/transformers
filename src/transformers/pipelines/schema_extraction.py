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

from typing import Any

from .base import Pipeline


class SchemaExtractionPipeline(Pipeline):
    """Extract entities, labels, relations, and records from text and a schema.

    Uses [`AutoModelForSchemaExtraction`]. Every label in the schema is scored in one forward pass.
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
        **kwargs,
    ):
        preprocess_params = {}
        if schema is not None:
            preprocess_params["schema"] = schema
        if max_len is not None:
            preprocess_params["max_len"] = max_len
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

    def preprocess(self, inputs, schema=None, max_len=None):
        architecture = getattr(self.model.config, "architecture", "span")
        encoding = self.processor(
            inputs,
            schema=schema,
            return_tensors="pt",
            max_len=max_len,
            architecture=architecture,
        )
        metadata = encoding.pop("metadata")
        return {"model_inputs": encoding, "metadata": metadata}

    def _forward(self, model_inputs):
        metadata = model_inputs.pop("metadata")
        outputs = self.model(**model_inputs["model_inputs"])
        return {"outputs": outputs, "metadata": metadata}

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
        outputs = model_outputs["outputs"]
        metadata = model_outputs["metadata"]
        temperature = float(getattr(self.model.config, "classification_temperature", 1.0))
        if constraints is not None:
            decoded = self.processor.post_process_constrained_classification(
                outputs,
                metadata,
                threshold=threshold,
                include_confidence=include_confidence,
                temperature=temperature,
                decoder=kwargs.get("decoder", "auto"),
            )
        elif joint:
            decoded = self.processor.post_process_joint_extraction(
                outputs,
                metadata,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
                optimizer=kwargs.get("optimizer", "beam"),
            )
        else:
            decoded = self.processor.post_process_extraction(
                outputs,
                metadata,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
                temperature=temperature,
            )
        return decoded[0]
