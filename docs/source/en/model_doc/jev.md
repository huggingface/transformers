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

# JEV

JEV is a decision head that scores single-token labels with a linear layer at the last token of the prompt, soft-capped like the logits of a language model head. It was introduced by [JEV](https://huggingface.co/autotrust/JEV-9B) and is also used by [GEV](https://huggingface.co/autotrust/GEV-26B-Decide).

[`JevForDecision`] is the model of the checkpoint and its head, like a `ForCausalLM` is the model and its `lm_head`. It
reads the files that the checkpoint ships its head with, next to its weights (`readout.safetensors` and
`decision_config.json` for vLLM decision models, `head.safetensors` and `judge_config.json` for JEV).

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide")
decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    }
)
decide("Where is my parcel?", questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}})
```

See the [decision pipeline](../main_classes/pipelines#transformers.DecisionPipeline) for image assets.

Without the pipeline, decision models are used like `generate`: the decision config renders a prompt per question (or
one for all of them) with the chat template, `decide` scores the options of each question, and `decode_decision` turns
the scores into answers:

```python
import torch
from transformers import AutoTokenizer, JevForDecision

repo = "autotrust/JEV-9B"
model = JevForDecision.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")
tokenizer, config = AutoTokenizer.from_pretrained(repo), model.decision_config  # read from the checkpoint's files
if config.adapter:  # JEV keeps its LoRA in a subfolder
    model.load_adapter(repo, adapter_kwargs={"subfolder": config.adapter})

request = {
    "state": "I was charged twice for my order and I want my money back today.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today", "Now"]},
    },
}
answers = {}
for kwargs in config.chat_template_kwargs(request, tokenizer):
    questions = kwargs.pop("decision_questions")
    inputs = tokenizer.apply_chat_template(**kwargs, tokenize=True, return_dict=True, return_tensors="pt")
    answers |= tokenizer.decode_decision(model.decide(**inputs, decision_questions=questions), questions)
```

The same head reads the 255 answer codes of [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
and [d3](https://huggingface.co/vllm-sr/d3), whose Qwen3.5 backbone attends to the whole prompt
(`is_causal=False` in the `text_config` of their `JevForDecision`). Their head (`readout.safetensors`) and settings
(`decision_config.json`) are read as they are, next to their weights:

```python
pipeline("decision", model="vllm-sr/d3")(request)  # or perplexity-ai/pplx-decider-v1.1-27b
```

## JevConfig

[[autodoc]] JevConfig

## JevForDecision

[[autodoc]] JevForDecision
    - forward
