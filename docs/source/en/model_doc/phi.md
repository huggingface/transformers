<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-06-20 and added to Hugging Face Transformers on 2023-11-10 and contributed by [susnato](https://huggingface.co/susnato).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Phi

[Phi](https://huggingface.co/papers/2306.11644) is a 1.3 billion parameter Transformer-based language model for code, trained for 4 days on 8 A100 GPUs using 6 billion tokens of high-quality web data and 1 billion tokens of GPT-3.5–generated synthetic textbooks and exercises. Despite its relatively small size, it achieves strong performance, with 50.6% pass@1 on HumanEval and 55.5% on MBPP. Compared to phi-1-base (pre-finetuning) and phi-1-small (350M parameters), phi-1 shows notable emergent abilities after finetuning on coding exercises. This demonstrates that targeted data and finetuning can yield competitive coding performance even in smaller models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="microsoft/phi-1.5", dtype="auto",)
pipeline("def fibonacci(n):")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1.5")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1.5", dtype="auto",)

inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- For Transformers < 4.37.0.dev, set `trust_remote_code=True` in [`~AutoModel.from_pretrained`].
- Otherwise, update Transformers to the latest stable version.

## PhiConfig

[[autodoc]] PhiConfig

## PhiModel

[[autodoc]] PhiModel
    - forward

## PhiForCausalLM

[[autodoc]] PhiForCausalLM
    - forward
    - generate

## PhiForSequenceClassification

[[autodoc]] PhiForSequenceClassification
    - forward

## PhiForTokenClassification

[[autodoc]] PhiForTokenClassification
    - forward

