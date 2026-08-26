<!--Copyright 2026 The Complexity-ML team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-26.*

# TR-HASH

TR-HASH is a decoder-only Mixture-of-Experts (MoE) transformer whose sparse expert selection is a deterministic
function of token IDs and the layer index. Instead of a learned router, every layer stores a route table as part of the
checkpoint. This makes expert selection reproducible and keeps routing metadata paired with the weights it controls.

The released 201.2M-parameter model uses grouped-query attention, query/key RMS normalization, four stored routed
experts with two active per token, and an always-active shared SwiGLU branch. Routed projections are stored as stacked
three-dimensional tensors rather than as separate `Linear` modules.

The example below demonstrates how to generate text with [`AutoModelForCausalLM`].

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


checkpoint = "AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)

messages = [{"role": "user", "content": "Explain deterministic token routing briefly."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=80)

print(tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

## Deterministic routing

For each layer, multiple token-ID hash channels assign a score to every expert. The top two experts are compiled into
the persisted `route_table`. Runtime dispatch therefore performs a table lookup rather than a learned routing
projection. The route table is the single canonical routing artifact; the original release's redundant compact
pair/code buffers are ignored by the native checkpoint conversion.

For every token, the model adds the always-on shared SwiGLU output to the weighted outputs of the two selected routed
experts. The public checkpoint stores routed gate, up, and down projections as tensors shaped respectively
`[num_experts, hidden_size, expert_width]`, `[num_experts, hidden_size, expert_width]`, and
`[num_experts, expert_width, hidden_size]`. Native loading transposes and combines these into the standard fused
`gate_up_proj` and `down_proj` expert layout used by Transformers MoE implementations.

## TRHashConfig

[[autodoc]] TRHashConfig

## TRHashModel

[[autodoc]] TRHashModel
    - forward

## TRHashForCausalLM

[[autodoc]] TRHashForCausalLM
    - forward
