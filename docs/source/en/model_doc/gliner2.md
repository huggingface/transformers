<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published on 2025-09-01 and contributed to Hugging Face Transformers on 2026-10-08.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# GLiNER2

[GLiNER2](https://huggingface.co/fastino) is a schema-conditioned encoder model. One forward pass scores every label in a schema: entities, classification, relations, JSON structures, attributes, and records. Span checkpoints use a marker span head. Boundary checkpoints (GLiNER2.5) use a shared candidate pool, a record decoder, and a relation scorer.

Use [`pipeline`] with the `schema-extraction` task, or [`AutoModelForSchemaExtraction`] with [`Gliner2Processor`].

```python
from transformers import pipeline

extractor = pipeline("schema-extraction", model="fastino/GLiNER2.5-Decide")
extractor(
    "Ada Lovelace wrote notes about the analytical engine.",
    schema={"entities": {"person": {}, "work": {}}},
)
```

## Gliner2Config

[[autodoc]] Gliner2Config

## Gliner2BoundaryConfig

[[autodoc]] Gliner2BoundaryConfig

## Gliner2Processor

[[autodoc]] Gliner2Processor
    - __call__
    - post_process_extraction
    - post_process_constrained_classification
    - post_process_joint_extraction

## Gliner2Model

[[autodoc]] Gliner2Model
    - forward

## Gliner2ForSchemaExtraction

[[autodoc]] Gliner2ForSchemaExtraction
    - forward

## Gliner2ModelOutput

[[autodoc]] Gliner2ModelOutput

## Gliner2SchemaExtractionOutput

[[autodoc]] Gliner2SchemaExtractionOutput
