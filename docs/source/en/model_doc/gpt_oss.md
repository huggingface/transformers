<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-08-05 and added to Hugging Face Transformers on 2025-08-05.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# GptOss

[GptOss](https://openai.com/index/introducing-gpt-oss/) is a sparse mixture-of-experts (MoE) language model from OpenAI that routes each token to 4 of 128 experts. It uses attention sinks — learnable auxiliary tokens appended to each attention head — and YaRN rotary embeddings for sequences up to 131k tokens.

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="openai/gpt-oss-20b",
    dtype=torch.bfloat16,
)
pipe("Plants create energy through a process known as")
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    dtype=torch.bfloat16,
    device_map="auto",
)
input_ids = tokenizer("Plants create energy through a process known as", return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- SDPA is not supported because attention sinks require direct access to the full attention logits before softmax. Use Flash Attention or Flex Attention instead.
- When using Flex Attention, attention sinks require special handling. The `score_mod` function operates on individual score elements rather than the full attention matrix, so sink renormalization is applied after computation using the log-sum-exp (LSE) values returned by Flex Attention.

## GptOssConfig

[[autodoc]] GptOssConfig

## GptOssModel

[[autodoc]] GptOssModel
    - forward

## GptOssForCausalLM

[[autodoc]] GptOssForCausalLM
    - forward

## GptOssForSequenceClassification

[[autodoc]] GptOssForSequenceClassification
    - forward

## GptOssForTokenClassification

[[autodoc]] GptOssForTokenClassification
    - forward
