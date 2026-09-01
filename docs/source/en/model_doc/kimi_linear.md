<!--Copyright 2026 the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-10-30 and contributed to Hugging Face Transformers on 2026-08-24.*

## Overview

Kimi Linear is a hybrid linear attention architecture from Moonshot AI, introduced in
[Kimi Linear: An Expressive, Efficient Attention Architecture](https://huggingface.co/papers/2510.26692).

At its core is **Kimi Delta Attention (KDA)**, a refinement of [Gated DeltaNet](https://huggingface.co/papers/2412.06464)
that gives each key channel its own forget gate, so the recurrent state decays per channel instead of per head. KDA is
used in most layers; every fourth layer keeps a full-attention block that reuses DeepSeek-V3's Multi-head Latent
Attention (MLA), and the feed-forward blocks are DeepSeek-V3-style MoE with a shared expert.

The abstract from the paper is the following:

*We introduce Kimi Linear, a hybrid linear attention architecture that, for the first time, outperforms full attention
under fair comparisons across various scenarios -- including short-context, long-context, and reinforcement learning
(RL) scaling regimes. At its core lies Kimi Delta Attention (KDA), an expressive linear attention module that extends
Gated DeltaNet with a finer-grained gating mechanism.*

Two things are worth knowing when reading the modeling code:

- **The model is NoPE.** Every released checkpoint sets `mla_use_nope=True`, so no rotary embedding is applied
  anywhere: the KDA layers encode position through their recurrence, and the full-attention layers are left without
  positional encoding. The `qk_rope_head_dim` slice still exists in the projections, it is simply never rotated.
- **The layer pattern comes from the checkpoint.** `linear_attn_config` lists `kda_layers` / `full_attn_layers` with
  1-based indices; the config converts them into the standard `layer_types` list.

This model was contributed by [Moonshot AI](https://huggingface.co/moonshotai).
The original code can be found [here](https://github.com/MoonshotAI/Kimi-Linear).

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

# the checkpoint ships a custom tiktoken-based tokenizer, hence `trust_remote_code`
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

messages = [{"role": "user", "content": "Tell me about the french revolution."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=128)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]

print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

The KDA layers run on a pure PyTorch implementation by default. Installing
[`fla-core`](https://github.com/fla-org/flash-linear-attention) (`pip install -U fla-core`) makes them dispatch to the
original Triton kernels instead, which is considerably faster for long sequences.

## KimiLinearConfig

[[autodoc]] KimiLinearConfig

## KimiLinearModel

[[autodoc]] KimiLinearModel
    - forward

## KimiLinearForCausalLM

[[autodoc]] KimiLinearForCausalLM
    - forward
