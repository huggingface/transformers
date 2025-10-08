<!--Copyright 2024 The GLM & ZhipuAI team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-06-18 and added to Hugging Face Transformers on 2024-10-18 and contributed by [THUDM](https://huggingface.co/THUDM).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# GLM

[ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools](https://huggingface.co/papers/2406.12793v1) presents the GLM-4 series, including GLM-4, GLM-4-Air, and GLM-4-9B. These models are pre-trained on ten trillions of tokens in Chinese and English, with additional data from 24 languages. They undergo a multi-stage post-training process involving supervised fine-tuning and human feedback for high-quality alignment. Evaluations demonstrate that GLM-4 models rival or outperform GPT-4 in various metrics, match GPT-4 Turbo in instruction following, and excel in long context tasks and Chinese alignment. The GLM-4 All Tools model autonomously selects and uses tools like web browsers, Python interpreters, and text-to-image models to complete complex tasks, performing on par with or better than GPT-4 All Tools. The series includes open-sourced models like ChatGLM-6B, GLM-4-9B, GLM-4V-9B, WebGLM, and CodeGeeX, which have garnered over 10 million downloads on Hugging Face in 2023.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="zai-org/glm-4-9b-hf", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zai-org/glm-4-9b-hf")
model = AutoModelForCausalLM.from_pretrained("zai-org/glm-4-9b-hf", dtype="auto",)
inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## GlmConfig

[[autodoc]] GlmConfig

## GlmModel

[[autodoc]] GlmModel
    - forward

## GlmForCausalLM

[[autodoc]] GlmForCausalLM
    - forward

## GlmForSequenceClassification

[[autodoc]] GlmForSequenceClassification
    - forward

## GlmForTokenClassification

[[autodoc]] GlmForTokenClassification
    - forward

