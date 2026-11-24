<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for the doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-24.*

# GS1

GS1 is a family of pretrained models from [BIGAI](https://gs1.mybigai.ac.cn) for structured decision tasks. Given a state
and a question, GS1 scores each candidate answer. It supports `choice`, `score`, and `noul` (yes/no) questions.

GS1 uses a Qwen3.5 text model and a Qwen3.5 vision encoder as its backbone. It includes GS1-specific bidirectional
decision-context processing, attention-residual routing, and a cross-attention decision head. The head reads the
question and option spans from the rendered prompt and returns one logit per option.

| Model | Checkpoint |
|---|---|
| GS1-0.8B | [bigai/GS1-0.8B](https://huggingface.co/bigai/GS1-0.8B) |
| GS1-2B | [bigai/GS1-2B](https://huggingface.co/bigai/GS1-2B) |
| GS1-4B | [bigai/GS1-4B](https://huggingface.co/bigai/GS1-4B) |
| GS1-9B | [bigai/GS1-9B](https://huggingface.co/bigai/GS1-9B) |

See the [open-source implementation](https://github.com/bigai-nlco/GS1)
for the model and inference reference.

## Usage

Use the [decision pipeline](../main_classes/pipelines#transformers.DecisionPipeline) to score options for a request.

```python
from transformers import pipeline

decide = pipeline("decision", model="bigai/GS1-0.8B")
answers = decide(
    {
        "state": "My package arrived damaged. Please send a replacement.",
        "questions": {
            "resolution": {
                "type": "choice",
                "instructions": "What does the customer want?",
                "criteria": {"refund": "A refund", "replacement": "A replacement"},
            },
        },
    }
)
print(answers["resolution"]["choice"])
```

## GS1Config

[[autodoc]] GS1Config

## GS1ForDecision

[[autodoc]] GS1ForDecision
    - forward
