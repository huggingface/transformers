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

*This model was released on 2025-08-05 and added to Hugging Face Transformers on 2025-08-05 and contributed by [<INSERT YOUR HF USERNAME HERE>](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# GptOss

[GptOss](https://huggingface.co/papers/2508.10925) are open-weight reasoning models built on a mixture-of-experts transformer for improved accuracy and lower inference cost. Training combines large-scale distillation with reinforcement learning, and the models are optimized for agentic tasks like web research, Python execution, and integration with external developer tools. They use a rendered chat format to improve instruction following and role clarity, and demonstrate strong performance across benchmarks in math, coding, and safety. All components—model weights, inference systems, tool environments, and tokenizers are released under Apache 2.0 for broad accessibility.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai/gpt-oss-20b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

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

