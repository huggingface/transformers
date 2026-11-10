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
*This model was contributed to Hugging Face Transformers on 2026-11-09.*

# JEV

> [!WARNING]
> Decision models, the `decision` pipeline and `DecisionConfig` are experimental: their API may change at any time, without deprecation.

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
```

See the [decision pipeline](../main_classes/pipelines#transformers.DecisionPipeline) for image assets.

Without the pipeline, decision models are used like `generate`: the decision config renders a prompt per question (or
one for all of them) with the chat template, `decide` scores the options of each question, and `Question.answer` turns
the scores into answers:

> [!IMPORTANT]
> `from_pretrained` does not load the LoRA of checkpoints that keep it in a subfolder (`decision_config.adapter`, e.g.
> `adapter/` for JEV). The pipeline loads it for you; without the pipeline, call `load_adapter` yourself as below, or
> the model answers with the base weights.

```python
import torch
from transformers import AutoTokenizer, JevForDecision

repo = "autotrust/JEV-9B"
model = JevForDecision.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")
tokenizer, config = AutoTokenizer.from_pretrained(repo), model.decision_config  # read from the checkpoint's files
if config.adapter:  # JEV keeps its LoRA in a subfolder: `from_pretrained` doesn't load it, the pipeline does
    model.load_adapter(repo, adapter_kwargs={"subfolder": config.adapter})

request = {
    "state": "I was charged twice for my order and I want my money back today.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today", "Now"]},
    },
}
answers = {}
for inputs, questions in config.prepare_inputs(request, tokenizer):
    scores = model.decide(**inputs, decision_questions=questions)
    answers |= {q.name: q.answer(s.softmax(-1).tolist()) for q, s in zip(questions, scores)}
```

The same head reads the 255 answer codes of [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
and [d3](https://huggingface.co/vllm-sr/d3), whose Qwen3.5 backbone attends to the whole prompt
(`is_causal=False` in the `text_config` of their `JevForDecision`). Their head (`readout.safetensors`) and settings
(`decision_config.json`) are read as they are, next to their weights:

```python
pipeline("decision", model="vllm-sr/d3")(request)  # or perplexity-ai/pplx-decider-v1.1-27b
```

## Training

`JevForDecision` returns a standard output with `loss` and `logits`. Its native loss is selected through
Transformers' `LOSS_MAPPING`, so it can train with the standard `Trainer` or TRL without a custom loss implementation.
Training uses uncalibrated logits; temperature calibration remains part of `decide`.

The default objective follows the [AutoTrust JEV-9B training description](https://huggingface.co/autotrust/JEV-9B#training-details):
KL(target || model) over the active options, plus `config.rps_weight` times the ranked probability score for
ordered `score` questions. RPS is the mean squared difference between target and predicted cumulative probabilities
over the first `K - 1` boundaries of the `K` ordered options. `rps_weight` defaults to 0.5. For one-hot labels, KL equals cross-entropy.
This is the open AutoTrust student's documented objective, not the unpublished loss of TypeSafe's closed Jev teacher.
The model card specifies the objective but does not publish its training-function code.
Other checkpoints using the same head are not assumed to have used this training recipe.
The loss computes in float32 and uses a max-excluded `log1p` normalization to preserve small losses and gradients
for confident predictions, following [SoftmaxFix](https://jonbarron.info/softmaxfix/). The paper's separate
soft-target cancellation fix requires teacher logits and is not implemented by the probability-target API.

Pass integer `labels` of shape `(1, num_questions)` containing zero-based option indices, or packed floating-point
labels of shape `(1, total_num_options)` containing one normalized target distribution per question. Integer `-100`
labels are ignored. Optional `loss_weights` weight each question; their weighted sum is divided by the valid-question
count rather than the sum of weights. Score supervision follows the question's
option order, and Noul labels follow `question.options`, including its configured true/false order.

Use `config.prepare_inputs` to obtain the same prompts and marks as inference. Each forward takes one packed prompt;
add a batch dimension to its marks, and supply option counts and type IDs for its questions:

```python
from transformers.decision import Question

# One prepared prompt and its questions; each target index follows that question's option order.
inputs, questions = config.prepare_inputs(request, tokenizer)[0]
inputs["marks"] = inputs["marks"][None]
inputs["num_options"] = torch.tensor([[len(q.options) for q in questions]])
inputs["question_type_ids"] = torch.tensor([[Question.questions_types.index(q.type) for q in questions]])
inputs["labels"] = torch.tensor([[q.options.index("true") if q.type == "noul" else 0 for q in questions]])
outputs = model(**{key: value.to(model.device) for key, value in inputs.items()})
outputs.loss.backward()
```

`outputs.logits` contains the active option scores concatenated in question order, including when labels are omitted.
With `return_dict=False`, the output is a tuple beginning with loss when labels are supplied, or logits otherwise.
When accumulating prompts with different question counts, explicitly pass their total valid-question count as
`num_items_in_batch`. Automatic Trainer item counting is disabled because packed soft targets count options rather
than questions; the default loss uses the local valid-question mean.

## JevConfig

[[autodoc]] JevConfig

## JevForDecision

[[autodoc]] JevForDecision
    - forward
