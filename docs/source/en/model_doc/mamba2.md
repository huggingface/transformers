<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-05-31 and added to Hugging Face Transformers on 2024-08-06 and contributed by [Molbap](https://huggingface.co/Molbap).*

# Mamba 2

[Mamba2](https://huggingface.co/papers/2405.21060) introduce the State Space Duality (SSD) framework, which unifies aspects of attention mechanisms and SSMs. Using this framework, they design Mamba-2, an improved version of the Mamba architecture, with a refined selective SSM core layer. Mamba-2 achieves 2–8× faster performance than its predecessor while maintaining competitiveness with Transformers on language modeling tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="mistralai/Mamba-Codestral-7B-v0.1", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Mamba2Config

[[autodoc]] Mamba2Config

## Mamba2Model

[[autodoc]] Mamba2Model
    - forward

## Mamba2LMHeadModel

[[autodoc]] Mamba2ForCausalLM
    - forward

