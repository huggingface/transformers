<!--Copyright 2026 Recursive and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-21.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# Flint

Flint is Recursive's mixture-of-experts language model. Compared to a standard SwiGLU mixture-of-experts transformer
such as [Qwen3-MoE](./qwen3_moe), it has:

- **Hybrid attention.** Most layers use sliding-window attention over the current token and the 512 tokens before
  it, with rotary position embeddings. The remaining layers, listed in `layer_types`, use full attention without
  positional encoding.
- **Attention sinks.** Every attention head has a learned sink logit that enters the softmax normalization without
  attending to any value, as in [GPT-OSS](./gpt_oss). Queries and keys are RMS-normalized per head.
- **Sigmoid routing and a shared expert.** Each sparse layer routes a token to 6 of 256 experts by sigmoid score,
  renormalizes the selected scores, and adds an ungated shared expert. The first two layers are dense.

## Usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="RecursiveSI/flint", dtype="auto", device_map="auto")
messages = [{"role": "user", "content": "Write a haiku about autumn leaves."}]
print(pipe(messages, max_new_tokens=1024)[0]["generated_text"][-1]["content"])
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "RecursiveSI/flint"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Write a haiku about autumn leaves."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- **Attention backends.** SDPA cannot apply attention sinks, so Flint defaults to the `eager` implementation, which
  computes attention scores in float32. For long inputs, use a sink-aware FlashAttention kernel:
  `kernels-community/vllm-flash-attn3` on Hopper or `flash_attention_4` on Blackwell. Requesting
  `flash_attention_2` selects one of these instead.
- **Checkpoint format.** Checkpoints store one tensor per expert and the sinks as `self_attn.softmax_offset`. They are
  fused and renamed on load, and saved back in the same format.

## FlintConfig

[[autodoc]] FlintConfig

## FlintModel

[[autodoc]] FlintModel
    - forward

## FlintForCausalLM

[[autodoc]] FlintForCausalLM
    - forward
