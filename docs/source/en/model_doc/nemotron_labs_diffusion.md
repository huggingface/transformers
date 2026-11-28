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
*This model was contributed to Hugging Face Transformers on 2026-11-27.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Nemotron-Labs Diffusion

[Nemotron-Labs Diffusion](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion) is a family of 3B, 8B, and 14B diffusion language models developed by [NVIDIA](https://huggingface.co/nvidia). Its transformer backbone builds on [Ministral 3](./ministral3) (YaRN rotary embeddings and post-RoPE Llama-4 query attention scaling) and supports masked block-diffusion decoding, linear self-speculation, and autoregressive generation with an untied output head.

Released checkpoints store backbone weights under `encoder.*` and the output head under `diffusion_head.*`, which Transformers renames to `model.*` and `lm_head.*` when loading and back when saving [`NemotronLabsDiffusionForCausalLM`].

- [`NemotronLabsDiffusionModel`] supports causal, bidirectional, block (`block_sequence_ids`), and two-stream `block_diff` attention modes via `attention_mode`, and reads committed KV prefixes without mutating them when `use_cache=False`.
- [`NemotronLabsDiffusionForCausalLM`] provides `diffusion_generate` (block-wise masked diffusion decoding with prefix KV caching), `linear_spec_generate` (bidirectional draft plus causal AR verification), and standard autoregressive `generate` (pass `use_cache=True` because released checkpoints default to `use_cache=false`).

The example below demonstrates block-diffusion and autoregressive generation with [`AutoModelForCausalLM`].

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-Labs-Diffusion-3B")
model = AutoModelForCausalLM.from_pretrained("nvidia/Nemotron-Labs-Diffusion-3B", device_map="auto")

messages = [{"role": "user", "content": "Explain the Pythagorean theorem in one sentence."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
inputs = inputs.to(model.device)

# Masked block-diffusion generation
diffusion_ids, _ = model.diffusion_generate(inputs["input_ids"], max_new_tokens=64, block_length=32, threshold=0.9)
print(tokenizer.decode(diffusion_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))

# Autoregressive generation
ar_ids = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.decode(ar_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## NemotronLabsDiffusionConfig

[[autodoc]] NemotronLabsDiffusionConfig

## NemotronLabsDiffusionModel

[[autodoc]] NemotronLabsDiffusionModel
    - forward

## NemotronLabsDiffusionForCausalLM

[[autodoc]] NemotronLabsDiffusionForCausalLM
    - forward
    - diffusion_generate
    - linear_spec_generate
