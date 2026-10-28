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
*This model was contributed to Hugging Face Transformers on 2026-10-16.*

# von

von is an option marker decision head introduced by [von](https://huggingface.co/wfzyx/von): an MLP scores each option from the hidden state of the `[MASK]` token in front of it. Options only attend to the prompt and to themselves, at positions that restart after the prompt, with masks built by the [masking API](../attention_interface#build-an-attention-mask) for `sdpa`, `eager` and `flex_attention`. The scores of a zero-shot `noul` question are debiased by the same question asked without the state.

Decision heads are applied by [`DecisionPipeline`] to the hidden states of the original checkpoint. The chat template,
the settings and the head live in their own repo next to it.

```python
from transformers import pipeline

decide = pipeline("decision", model="wfzyx/von", decision_config="ArthurZ/von")
decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    }
)
```

## VonConfig

[[autodoc]] VonConfig

## VonForDecision

[[autodoc]] VonForDecision
    - forward
