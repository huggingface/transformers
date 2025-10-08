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
*This model was released on 2023-12-11 and added to Hugging Face Transformers on 2023-12-11 and contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).*

# Mixtral

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[Mixtral](https://huggingface.co/papers/2401.04088) is a Sparse Mixture of Experts (SMoE) model based on the Mistral 7B architecture, where each layer contains 8 experts but only 2 are activated per token via a router network. This design allows tokens to access a total of 47B parameters while only using 13B active parameters during inference, improving efficiency. The model was trained with a 32k token context window and outperforms or matches larger models like Llama 2 70B and GPT-3.5, with especially strong results in mathematics, coding, and multilingual tasks. A fine-tuned instruction-following version, Mixtral 8x7B-Instruct, also surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B-chat, with both versions released under Apache 2.0.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mistralai/Mixtral-8x7B-v0.1", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## MixtralConfig

[[autodoc]] MixtralConfig

## MixtralModel

[[autodoc]] MixtralModel
    - forward

## MixtralForCausalLM

[[autodoc]] MixtralForCausalLM
    - forward

## MixtralForSequenceClassification

[[autodoc]] MixtralForSequenceClassification
    - forward

## MixtralForTokenClassification

[[autodoc]] MixtralForTokenClassification
    - forward

## MixtralForQuestionAnswering

[[autodoc]] MixtralForQuestionAnswering
    - forward