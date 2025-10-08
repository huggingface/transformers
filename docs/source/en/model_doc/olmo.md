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
*This model was released on 2024-02-01 and added to Hugging Face Transformers on 2024-04-17 and contributed by [shanearora](https://huggingface.co/shanearora).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OLMo

[OLMo: Accelerating the Science of Language Models](https://huggingface.co/papers/2402.00838) is a series of Open Language Models designed to facilitate the scientific study of language models. Trained on the Dolma dataset, OLMo releases not only the model weights and inference code but also the entire framework, including training data and training/evaluation code. This comprehensive release aims to empower the open research community and encourage further innovation by providing detailed access to powerful, open models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="allenai/OLMo-7B-hf", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-hf", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## OlmoConfig

[[autodoc]] OlmoConfig

## OlmoModel

[[autodoc]] OlmoModel
    - forward

## OlmoForCausalLM

[[autodoc]] OlmoForCausalLM
    - forward

