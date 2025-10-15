<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-04-22 and added to Hugging Face Transformers on 2024-04-24.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Phi-3

[Phi-3](https://huggingface.co/papers/2404.14219) introduces phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, achieving performance comparable to Mixtral 8x7B and GPT-3.5 on benchmarks like MMLU and MT-bench. The model's dataset includes heavily filtered web data and synthetic data, ensuring robustness, safety, and chat format alignment. Additionally, phi-3-small (7B parameters) and phi-3-medium (14B parameters) models, trained on 4.8T tokens, demonstrate higher capabilities with improved MMLU and MT-bench scores.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="microsoft/Phi-3-mini-4k-instruct", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- This model is very similar to Llama. The main difference is [`Phi3SuScaledRotaryEmbedding`] and [`Phi3YarnScaledRotaryEmbedding`], which extend the context of rotary embeddings.
- Query, key, and values are fused. The MLP's up and gate projection layers are also fused.
- The tokenizer is identical to [`LlamaTokenizer`], except for additional tokens.

## Phi3Config

[[autodoc]] Phi3Config

## Phi3Model

[[autodoc]] Phi3Model
    - forward

## Phi3ForCausalLM

[[autodoc]] Phi3ForCausalLM
    - forward
    - generate

## Phi3ForSequenceClassification

[[autodoc]] Phi3ForSequenceClassification
    - forward

## Phi3ForTokenClassification

[[autodoc]] Phi3ForTokenClassification
    - forward

