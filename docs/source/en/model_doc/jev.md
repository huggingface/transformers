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

# JEV

JEV is a decision head that scores single-token labels with a linear layer at the last token of the prompt, soft-capped like the logits of a language model head. It was introduced by [JEV](https://huggingface.co/autotrust/JEV-9B) and is also used by [GEV](https://huggingface.co/autotrust/GEV-26B-Decide).

Decision heads are applied by [`DecisionPipeline`] to the hidden states of the original checkpoint. The chat template,
the settings and the head live in their own repo next to it.

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide", decision_config="ArthurZ/GEV-26B-Decide")
decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    }
)
decide("Where is my parcel?")  # a bare state is asked the questions of the last request
```

Questions can also be set when creating the pipeline, or per call, and become the ones asked of the next bare states:

```python
decide = pipeline(
    "decision",
    model="autotrust/GEV-26B-Decide",
    decision_config="ArthurZ/GEV-26B-Decide",
    questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
)
decide("I was charged twice for my order and want a refund.")
decide("Buy cheap watches now!!!", questions={"spam": {"type": "noul", "instructions": "Is this spam?"}})
```

Images come as `assets`, like in the System One API: each placeholder in the state, the instructions or the criteria is
replaced with its image. An asset is `{"type": "image", "format": "png", "data_base64": ...}`, a PIL image, a path or a
URL, and the model needs a processor that handles images.

```python
decide(
    {
        "state": "The customer sent photo-1 and asks if it is damaged.",
        "assets": {"photo-1": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"},
        "questions": {"damaged": {"type": "noul", "instructions": "Is the product in the photo damaged?"}},
    }
)
```

The same head reads the 255 answer codes of [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
and [d3](https://huggingface.co/vllm-sr/d3), whose Qwen3.5 backbone attends to the whole prompt
(`use_bidirectional_attention`). `convert_decider_to_hf.py` writes their head, which loads the backbone from the
original repo:

```python
pplx = pipeline("decision", model="ArthurZ/pplx-decider-v1.1-27b")
d3 = pipeline("decision", model="ArthurZ/d3")
request = {
    "state": "I was charged twice for my order and I want my money back today.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today", "Now"]},
    },
}
pplx(request), d3(request)
```

## JevConfig

[[autodoc]] JevConfig

## JevDecisionHead

[[autodoc]] JevDecisionHead
    - forward
