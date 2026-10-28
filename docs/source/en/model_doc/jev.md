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

[`JevForDecision`] is the model of the checkpoint and its head, like a `ForCausalLM` is the model and its `lm_head`. Its
files sit next to the ones of the checkpoint, in its repo: its config (`decision_head.json`), which names its weights,
and the decision config with its chat template (`decision.json`, `decision_template.jinja`).

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide")
decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    }
)
decide("Where is my parcel?")  # a bare state is asked the questions of the last request
```

See the [decision pipeline](../main_classes/pipelines#transformers.DecisionPipeline) for questions set up front,
bare states and image assets.

Without the pipeline, the template renders a prompt per question (or one for all of them), the model scores the options
of each question, and the decision config turns the scores into the probabilities of the answer:

```python
import torch
from transformers import AutoTokenizer, JevForDecision
from transformers.decision import DecisionConfig, DecisionTemplate

repo = "autotrust/JEV-9B"
model = JevForDecision.from_pretrained(repo, config_file_name="decision_head.json", dtype=torch.bfloat16, device_map="auto")
config = DecisionConfig.from_pretrained(repo)
if config.adapter:  # JEV keeps its LoRA in a subfolder
    model.load_adapter(repo, adapter_kwargs={"subfolder": config.adapter})
template = DecisionTemplate(AutoTokenizer.from_pretrained(repo), config)

request = {
    "state": "I was charged twice for my order and I want my money back today.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today", "Now"]},
    },
}
answers = {}
for prompt in template(request):
    inputs = prompt.model_inputs().to(model.device)
    with torch.no_grad():
        scores = model(**inputs)  # one score per option, for all the questions of the prompt
    for question, question_scores in zip(prompt.questions, scores.split(inputs["num_options"][0].tolist())):
        question_scores = config.get_logits_processor(question.type)(inputs["input_ids"], question_scores.float())
        answers[question.name] = question.answer(question_scores.softmax(-1).tolist())
```

The same head reads the 255 answer codes of [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
and [d3](https://huggingface.co/vllm-sr/d3), whose Qwen3.5 backbone attends to the whole prompt
(`is_causal=False` in the `text_config` of their `JevForDecision`). Their head (`readout.safetensors`) and settings
(`decision_config.json`) are read as they are, next to their weights:

```python
pplx = pipeline("decision", model="perplexity-ai/pplx-decider-v1.1-27b")
d3 = pipeline("decision", model="vllm-sr/d3")
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

## JevForDecision

[[autodoc]] JevForDecision
    - forward
