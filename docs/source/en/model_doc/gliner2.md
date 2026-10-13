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
*This model was contributed to Hugging Face Transformers on 2026-10-08.*

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

extractor = pipeline("schema-extraction", model="fastino/gliner2.5-base-v1")
extractor(
    "Ada Lovelace wrote notes about the analytical engine in London.",
    schema={
        "entities": {"person": {}, "work": {}, "location": {}},
        "classifications": [{"task": "topic", "labels": ["person", "place", "other"]}],
        "relations": [{"wrote": {"head": {}, "tail": {}}}],
        "json_structures": [{"product": {"name": {"dtype": "str"}, "price": {"dtype": "str"}}}],
    },
)
```

## Fine-tuning

Each training row is `text`, `schema`, and `labels`. A string label matches every occurrence of that string, and `{"text", "start", "end"}` selects one occurrence.

```json
{"text": "Ada Lovelace wrote notes.", "schema": {"entities": ["person"]}, "labels": {"entities": {"person": ["Ada Lovelace"]}}}
```

The processor builds training targets when `labels` is passed, so [`Trainer`] only needs a collator. GLiNER2 trains the encoder and the task heads with separate learning rates.

```python
from datasets import load_dataset
from torch.optim import AdamW
from transformers import Gliner2ForSchemaExtraction, Gliner2Processor, Trainer, TrainingArguments

model = Gliner2ForSchemaExtraction.from_pretrained("fastino/gliner2.5-base-v1")
processor = Gliner2Processor.from_pretrained("fastino/gliner2.5-base-v1")


def collate(rows):
    batch = processor(
        [row["text"] for row in rows],
        [row["schema"] for row in rows],
        labels=[row["labels"] for row in rows],
        architecture=model.config.architecture,
    )
    batch.pop("metadata")
    return dict(batch)


encoder = [param for name, param in model.named_parameters() if "encoder" in name]
heads = [param for name, param in model.named_parameters() if "encoder" not in name]
optimizer = AdamW([{"params": encoder, "lr": 1e-5}, {"params": heads, "lr": 5e-4}], weight_decay=0.01)
trainer = Trainer(
    model=model,
    args=TrainingArguments("gliner2-finetuned", remove_unused_columns=False),
    train_dataset=load_dataset("json", data_files="train.jsonl")["train"],
    data_collator=collate,
    optimizers=(optimizer, None),
)
trainer.train()
```

Boundary checkpoints can leave the record and relation heads unused on span-only batches. Set `ddp_find_unused_parameters=True` when training them with DDP.

## Gliner2Config

[[autodoc]] Gliner2Config

## Gliner2BoundaryConfig

[[autodoc]] Gliner2BoundaryConfig

## Gliner2Schema

[[autodoc]] Gliner2Schema

## Gliner2Labels

[[autodoc]] Gliner2Labels

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
