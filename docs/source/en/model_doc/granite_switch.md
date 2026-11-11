<!--Copyright 2026 IBM and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GraniteSwitch

## Overview

GraniteSwitch is a Granite model that carries several LoRA adapters inside a single checkpoint and
picks which one to apply **per token**, from control tokens in the prompt. One request can switch
adapters any number of times, so a multi-turn conversation can use a different adapter per turn
without reloading weights or paying for a second model.

The adapters are frozen at inference time and stacked into the model's own projections, so there is
no adapter loading step and no separate adapter graph: a token routed to adapter *i* takes that
adapter's low-rank update inside the same matmul that every other token uses.

### Routing

Each adapter owns a control token. Routing happens in two stages:

1. **`MultiSwitch`** reads the raw `input_ids` and emits a per-token `adapter_indices` tensor, where
   `0` means "base model, no adapter". It does this with two small attention heads rather than a
   learned classifier: a *counting* head turns "how many control tokens have been seen so far" into
   an address, and a *coded-memory* head reads back which adapter was written at that address, using
   a constructed Kerdock / Delsarte-Goethals codebook. Nothing here is trained — the codebook is
   generated from the config.
2. **Token exchange** rewrites each control token to a substitute id before the embedding lookup, so
   the decoder never embeds a control token and the adapter marker stays invisible to the model.

Routing is latest-wins, and a checkpoint composed with a base-reset token can also route back to the
base model mid-sequence.

Because `MultiSwitch` is implemented as attention, it needs two cache slots of its own. These are
the first two layers of the cache, and `num_hidden_layers` in the config counts them, so a
GraniteSwitch checkpoint reports two more layers than it has decoder layers.

### Shadow Residual

When `dual_stream=True` the model runs two streams through the same layer: a frozen base stream and
a LoRA'd adapter stream. Every parameter is shared — the base stream is the same modules called with
`adapter_indices=None` — and K/V always come from the base stream. The two are connected by a single
low-rank `cross_stream` injection per layer, which is initialized to exactly zero, so an untrained
Shadow Residual model is numerically identical to the base model.

A checkpoint holds either Shadow Residual adapters or standard LoRA adapters, never a mix.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "<granite-switch-checkpoint>"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()

messages = [{"role": "user", "content": "Summarize the attached document."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)

output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
```

The chat template places the control tokens, so selecting an adapter is a property of the prompt
rather than of the model object.

## Notes

- Setting `num_adapters=0` builds a plain Granite model: `MultiSwitch` is not constructed, every
  projection is an ordinary `nn.Linear`, and no control tokens exist.
- The model does not support `torch.compile` in fullgraph mode. The switched LoRA layers call
  `.unique()` on the per-token adapter indices to find which adapters are present in a batch, which
  is data-dependent, so `_can_compile_fullgraph` is `False`.
- Control tokens are freely generatable; nothing suppresses them during decoding.
- Granite's scalar multipliers all apply: `attention_multiplier`, `embedding_multiplier`,
  `residual_multiplier` and `logits_scaling`.
- Use left padding for batched generation. See [Padding side](../llm_tutorial#padding-side).

## GraniteSwitchConfig

[[autodoc]] GraniteSwitchConfig

## GraniteSwitchModel

[[autodoc]] GraniteSwitchModel
    - forward

## GraniteSwitchForCausalLM

[[autodoc]] GraniteSwitchForCausalLM
    - forward
