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
-->

# Sparse cosine decision head

`SparseCosineDecisionHead` is a trainable readout for dynamic question and option sets.
It mean-pools their hidden-state spans, applies separate Linear/LayerNorm projections,
and scores normalized question/option vectors with a learned positive scale. The scale
starts at 12 and is capped at 100. At hidden size and width 512 the head has 527,361 parameters.

Pooling gathers only tokens belonging to the supplied spans, without constructing a
full-sequence cumulative-sum buffer. The encoder and its dense hidden-state gradients
still scale with input length. Smaller parameter counts and bounded logits do not
guarantee better generalization or calibration; evaluate the complete model.

This is a reusable head module, not a complete decision pipeline model. Supply the
encoder's final normalized hidden states and CPU-native `(batch_index, start, end)`
span triples. End positions are exclusive. Each option points to its owning question.
With causal encoders, place the state before the scored spans so they can attend to it.

```python
import torch
from transformers.decision import SparseCosineDecisionHead

head = SparseCosineDecisionHead(hidden_size=512)
hidden_states = torch.randn(2, 64, 512)  # replace with encoder(...).last_hidden_state
question_spans = [(0, 20, 25), (1, 30, 35)]
option_spans = [(0, 25, 30), (0, 30, 36), (1, 35, 39), (1, 39, 44), (1, 44, 50)]
owners = [0, 0, 1, 1, 1]
logits = head(hidden_states, question_spans, option_spans, owners)
first_question = logits[:2].softmax(-1)
second_question = logits[2:].softmax(-1)
```

Use two explicit option spans for boolean questions, and ordered candidate spans for
score questions. The same scoring path handles each question type. Exclude `logit_scale`
from AdamW weight decay; retain the usual LayerNorm/bias exclusions in your optimizer.

The head saves `joint_head_config.json` and `joint_head.safetensors`, with the versioned
`cosine_span_v1` format and the same parameter names as its prefix-sum reference.
These files contain only the head; save the encoder, tokenizer and any inference
calibration settings separately. Loading a local directory or Hub repository returns
an fp32 CPU head, which can then be moved or cast with `.to(...)`.

```python
head.save_pretrained("cosine-head")
restored = SparseCosineDecisionHead.from_pretrained("cosine-head", local_files_only=True)
```

[[autodoc]] transformers.decision.SparseCosineDecisionHead
    - forward
    - save_pretrained
    - from_pretrained
