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
*This model was contributed to Hugging Face Transformers on 2026-11-16.*

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# GraniteSwitch

## Overview

GraniteSwitch is a [GraniteMoeHybrid](./granitemoehybrid) model that ships several LoRA adapters inside
one checkpoint and chooses which adapter to apply for each token, based on control tokens in the
prompt. A single request can change adapters as often as it needs to. A multi-turn conversation can
use a different adapter on every turn without reloading weights or running a second model.

The adapters are frozen and stored inside the model's own projections. There is no adapter loading
step and no separate adapter graph: a token routed to adapter *i* gets that adapter's low-rank update
inside the same matmul every other token uses.

You can find the GraniteSwitch checkpoints under the [IBM Granite](https://huggingface.co/ibm-granite)
organization, for example [ibm-granite/granite-switch-4.1-3b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview).
The tools that compose a checkpoint from a base model and a set of adapters live in the
[granite-switch](https://github.com/generative-computing/granite-switch) repository.

### How routing works

Each adapter owns a control token. Before the decoder runs, the model does two things with the raw
`input_ids`:

1. **It computes an adapter index for every token**, where `0` means the base model with no adapter.
   The most recent control token wins. Two small attention heads do the work: one counts how many
   control tokens came before each position, and the other looks up which adapter that control
   token selected, using a fixed Kerdock / Delsarte-Goethals codebook. These heads are generated
   from the config and have no trained weights.
2. **It replaces each control token with a substitute id** before the embedding lookup, so the
   decoder never sees a control token.

A checkpoint composed with a base-reset token can also switch back to the base model in the middle
of a sequence.

Routing depends only on the token ids, not on hidden states, so it keeps no cache and no state
between calls. On each forward pass it runs over the full sequence and keeps the part that lines up
with the tokens being decoded. During generation,
[`~GraniteSwitchForCausalLM.prepare_inputs_for_generation`] passes that full sequence as
`switch_input_ids`, because a single decode token cannot tell which control tokens came before it.

### Shadow Residual

With `dual_stream=True` each layer runs two streams: a frozen base stream and an adapter stream
with LoRA applied. Both streams share every parameter, and keys and values always come from the base
stream. A low-rank `cross_stream` projection per layer connects them. It starts at exactly zero, so
an untrained Shadow Residual model produces the same outputs as the base model.

A checkpoint contains either Shadow Residual adapters or standard LoRA adapters, never both.

### Activation probes

An adapter slot can hold an *activation probe* instead of a LoRA adapter. You mark it with
`"probe"` in `adapter_kinds`. A probe is a small linear head on one hidden state, for detection tasks
such as a safety check. It is selected with its control token like any adapter, and LoRA and probe
slots can live in the same checkpoint.

A probe request ends with the probe's control token. At that position the probe reads a base-model
hidden state (the final normalized one by default, or the output of a layer listed in
`probe_read_layers`) and scores each label. The model then rewrites the logits at that position so
only the probe's label tokens (`probe_label_token_ids`) can be generated, each with its score. The
answer comes back through normal generation: the first generated token is the label.

## Usage

The chat template places the control tokens. Pass `adapter_name` to pick an adapter, or leave it out
to use the base model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ibm-granite/granite-switch-4.1-3b-preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "When was the Eiffel Tower completed?"}]
documents = [{"doc_id": "1", "text": "The Eiffel Tower was completed in 1889 for the World's Fair."}]

# Route through the "answerability" adapter
inputs = tokenizer.apply_chat_template(
    messages,
    documents=documents,
    adapter_name="answerability",
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))

# Same prompt, base model
inputs = tokenizer.apply_chat_template(
    messages, documents=documents, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

`model.config.adapter_names` lists the adapters a checkpoint contains.

## Notes

- `num_adapters=0` builds a plain Granite model: no router, no control tokens, and every projection
  is a regular `nn.Linear`.
- In `config.json`, a checkpoint with adapters counts two extra layer slots for the router in
  `num_hidden_layers` and `layer_types`, so a 40-layer checkpoint lists 42. The config removes them
  on load (`config.num_hidden_layers` is 40) and writes them back on save, so the same file still
  loads in vLLM.
- `torch.compile` with `fullgraph=True` is not supported. The LoRA layers call `.unique()` on the
  adapter indices to find which adapters a batch uses, which is data-dependent.
- Control tokens are ordinary tokens and the model can generate them.
- If you pass `inputs_embeds` instead of `input_ids`, routing has no ids to read and every token
  uses the base model. Pass `switch_input_ids` with the embeddings to route anyway. Control tokens
  still need to go through `input_ids`, because an embedding computed from a control token cannot be
  swapped for its substitute.
- A probe's control token must be the last token of its sequence (padding aside), since the verdict
  replaces that position's logits. The model raises an error otherwise. [`GraniteSwitchModel`]
  returns the raw scores as `probe_logits` together with `probe_indices`.
- The Granite multipliers `attention_multiplier`, `embedding_multiplier`, `residual_multiplier` and
  `logits_scaling` all apply.
- Use left padding for batched generation. See [Padding side](../llm_tutorial#padding-side).

## GraniteSwitchConfig

[[autodoc]] GraniteSwitchConfig

## GraniteSwitchModel

[[autodoc]] GraniteSwitchModel
    - forward

## GraniteSwitchForCausalLM

[[autodoc]] GraniteSwitchForCausalLM
    - forward
