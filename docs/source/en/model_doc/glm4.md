<!--Copyright 2025 The GLM & ZhipuAI team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-06-18 and added to Hugging Face Transformers on 2025-04-09.*

# Glm4

[Glm4](https://huggingface.co/papers/2406.12793) is a family of large language models, with the latest GLM-4 series (GLM-4, GLM-4-Air, GLM-4-9B) trained on over ten trillion tokens primarily in Chinese and English, plus data from 24 other languages. The models use a multi-stage alignment process combining supervised fine-tuning and human feedback to optimize performance for Chinese and English. GLM-4 rivals or surpasses GPT-4 across benchmarks like MMLU, GSM8K, and HumanEval, achieves near-GPT-4-Turbo results in instruction following and long-context tasks, and outperforms GPT-4 in Chinese alignment. The GLM-4 All Tools model autonomously selects tools such as web browsing, Python, and text-to-image generation, matching or exceeding GPT-4 All Tools in complex task handling.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="zai-org/GLM-4.5-Air", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5-Air")
model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.5-Air", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Glm4Config

[[autodoc]] Glm4Config

## Glm4Model

[[autodoc]] Glm4Model
    - forward

## Glm4ForCausalLM

[[autodoc]] Glm4ForCausalLM
    - forward

## Glm4ForSequenceClassification

[[autodoc]] Glm4ForSequenceClassification
    - forward

## Glm4ForTokenClassification

[[autodoc]] Glm4ForTokenClassification
    - forward
