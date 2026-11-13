<!--Copyright 2026 Ricursive AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-12.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Ripple

[Ripple-65B](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) is Ricursive Intelligence's dense 65B-parameter
language model, pretrained on 12 trillion tokens with a context length of up to 1M tokens. It starts from the
[OLMo 3](./olmo3) recipe (per-head QK normalization, the four Gemma-style norms around attention and MLP, SwiGLU with
activation clipping) and adds:

- **Hybrid attention with global NoPE.** Four out of five layers use sliding-window attention over the current token
  and the 1023 before it, with rotary position embeddings. Every fifth layer, listed in `layer_types`, uses full
  attention with no positional encoding at all.
- **Sigmoid-gated attention.** A fifth projection, `g_proj`, gates the attention output head-wise with a sigmoid before
  the output projection.
- **Value residuals.** Every layer projects its values from an interpolation between the scaled input embeddings and
  the layer input, with a learned per-layer blend (`value_res_lambda`).
- **Scaled embeddings.** The token embeddings are multiplied by `embed_scale` (`sqrt(hidden_size)`) before the first
  layer, and the scaled embeddings are what the value residuals read.

## Usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="ricursiveAI/Ripple-65B-Preview", dtype="auto", device_map="auto")
messages = [{"role": "user", "content": "Write a haiku about the sea."}]
print(pipe(messages, max_new_tokens=50)[0]["generated_text"][-1]["content"])
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ricursiveAI/Ripple-65B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Write a haiku about the sea."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- **Thinking.** The instruction-tuned checkpoint reasons inside a `<think>...</think>` block before answering. These
  tags are plain text for the tokenizer, so `skip_special_tokens` keeps them.

## RippleConfig

[[autodoc]] RippleConfig

## RippleModel

[[autodoc]] RippleModel
    - forward

## RippleForCausalLM

[[autodoc]] RippleForCausalLM
    - forward
