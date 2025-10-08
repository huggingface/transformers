<!--Copyright 2024 The Qwen Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-04-29 and added to Hugging Face Transformers on 2025-03-31.*

# Qwen3

## Overview

[Qwen3](https://huggingface.co/papers/2505.09388) is the latest iteration of the Qwen large language model family, featuring both dense and Mixture-of-Expert (MoE) architectures with 0.6 to 235 billion parameters. It introduces a unified framework combining thinking mode for complex reasoning and non-thinking mode for fast, context-driven responses, with a dynamic thinking budget to balance computational cost and performance. Qwen3 leverages knowledge from flagship models to efficiently build smaller models without sacrificing capability and achieves state-of-the-art performance across tasks like code generation, mathematical reasoning, and agent tasks. Compared to Qwen2.5, it expands multilingual support from 29 to 119 languages, and all models are publicly available under Apache 2.0.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen3-8B", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Qwen3Config

[[autodoc]] Qwen3Config

## Qwen3Model

[[autodoc]] Qwen3Model
    - forward

## Qwen3ForCausalLM

[[autodoc]] Qwen3ForCausalLM
    - forward

## Qwen3ForSequenceClassification

[[autodoc]] Qwen3ForSequenceClassification
    - forward

## Qwen3ForTokenClassification

[[autodoc]] Qwen3ForTokenClassification
    - forward

## Qwen3ForQuestionAnswering

[[autodoc]] Qwen3ForQuestionAnswering
    - forward
