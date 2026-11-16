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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# GraniteSwitch

## Overview

GraniteSwitch is a [GraniteMoeHybrid](./granitemoehybrid) model with several LoRA adapters built into one
checkpoint. Control tokens in the prompt select the adapter for each token, so one request, or each turn of a
conversation, can use a different adapter without loading weights or running a second model.

Checkpoints are available under the [IBM Granite](https://huggingface.co/ibm-granite) organization, for example
[ibm-granite/granite-switch-4.1-3b-preview](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview).
They are built with the [granite-switch](https://github.com/generative-computing/granite-switch) tools.

**Routing.** Every adapter has a control token, and the most recent control token decides the adapter for the
tokens after it. Two small attention heads, built from the config with no trained weights, compute this index from
`input_ids`. Each control token is then replaced with a substitute id before the embedding lookup, so the decoder
never sees it. A checkpoint can also include a token that switches back to the base model.

**Shadow Residual.** With `dual_stream=True`, each layer runs a frozen base stream and an adapter stream that share
all weights, connected by a low-rank `cross_stream` projection that is zero at initialization.

**Probes.** An adapter slot marked `"probe"` in `adapter_kinds` holds a linear classifier on a hidden state instead
of a LoRA adapter. When a request ends with a probe's control token, the next generated token is the probe's label,
chosen from `probe_label_token_ids`.

## Usage

The chat template places the control tokens. Pass `adapter_name` to select an adapter, or leave it out to use the
base model. `model.config.adapter_names` lists the available adapters.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ibm-granite/granite-switch-4.1-3b-preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "When was the Eiffel Tower completed?"}]
documents = [{"doc_id": "1", "text": "The Eiffel Tower was completed in 1889 for the World's Fair."}]

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
```

## Notes

- Use left padding for batched generation. See [Padding side](../llm_tutorial#padding-side).
- With `inputs_embeds`, pass the token ids as `switch_input_ids` to keep routing. Without them every token uses the
  base model.
- A probe's control token must be the last non-padding token of its sequence.
- With adapters, `num_hidden_layers` in `config.json` includes two layer slots used by the router (42 for a 40-layer
  model). They are removed on load and added back on save.
- `torch.compile` with `fullgraph=True` is not supported.

## GraniteSwitchConfig

[[autodoc]] GraniteSwitchConfig

## GraniteSwitchModel

[[autodoc]] GraniteSwitchModel
    - forward

## GraniteSwitchForCausalLM

[[autodoc]] GraniteSwitchForCausalLM
    - forward
