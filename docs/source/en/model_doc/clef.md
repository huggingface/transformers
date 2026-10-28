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
*This model was contributed to Hugging Face Transformers on 2026-10-28.*

# Clef

Clef is a decision head that scores all the options of all the questions of a request jointly. It was introduced by [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) and is also used by [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash). The template marks the span of each instruction and of each of its options. The options of all the questions are packed in one sequence: pooled from the hidden states and from the output embeddings of their tokens, they attend to the prompt and are summarized into the field of their question. The fields then attend to each other and to the prompt, and each option is scored against the field of its question. The original `joint_head.safetensors` loads as is.

Decision heads are applied by [`DecisionPipeline`] to the hidden states of the original checkpoint. The chat template,
the settings and the head live in their own repo next to it.

```python
from transformers import pipeline

decide = pipeline("decision", model="Cloudflare/clef-flash", decision_config="ArthurZ/clef-flash")
decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is a refund requested?"},
            "topic": {"type": "choice", "instructions": "What is it about?", "criteria": {"billing": None, "shipping": None}},
        },
    }
)
```

## ClefConfig

[[autodoc]] ClefConfig

## ClefDecisionHead

[[autodoc]] ClefDecisionHead
    - forward

## ClefForDecision

[[autodoc]] ClefForDecision
    - from_pretrained
    - forward
