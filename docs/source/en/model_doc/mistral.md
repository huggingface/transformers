<!--Copyright 2023 Mistral AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-10-10 and added to Hugging Face Transformers on 2023-09-27 and contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Mistral

[Mistral](https://huggingface.co/papers/2310.06825) is a 7-billion-parameter language model designed for high efficiency and strong performance. It consistently outperforms larger models, beating Llama 2 13B on all benchmarks and Llama 1 34B in reasoning, math, and coding tasks. The model’s architecture combines grouped-query attention (GQA) for faster inference with sliding window attention (SWA) to process sequences of any length at lower cost. A fine-tuned variant, Mistral 7B-Instruct, further surpasses Llama 2 13B-Chat on human and automated evaluations, with all models released under Apache 2.0.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mistralai/Mistral-7B-v0.3", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## MistralConfig

[[autodoc]] MistralConfig

## MistralModel

[[autodoc]] MistralModel
    - forward

## MistralForCausalLM

[[autodoc]] MistralForCausalLM
    - forward

## MistralForSequenceClassification

[[autodoc]] MistralForSequenceClassification
    - forward

## MistralForTokenClassification

[[autodoc]] MistralForTokenClassification
    - forward

## MistralForQuestionAnswering

[[autodoc]] MistralForQuestionAnswering
    - forward

## MistralCommonTokenizer

[[autodoc]] MistralCommonTokenizer

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mistralai/Mistral-7B-v0.1", dtype="auto")
pipeline("The future of artificial intelligence is")
```
