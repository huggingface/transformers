<!--Copyright 2024 Kyutai and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-01-13 and added to Hugging Face Transformers on 2025-01-13 and contributed by [lmz](https://huggingface.co/lmz).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Helium

[Helium](https://kyutai.org/2025/01/13/helium.html) is a lightweight 2B-parameter multilingual language model designed for edge and mobile deployment, emphasizing low latency and privacy. Architecturally, it follows the LLaMA-1 transformer design with enhancements like RMSNorm, rotary embeddings, and gated linear units, trained on 2.5T tokens with token-level distillation from a 7B model. Its training dataset combines Wikipedia, Stack Exchange, scientific papers, and filtered Common Crawl, with a strong quality filtering pipeline and curriculum learning. In benchmarks, Helium-1 shows competitive performance in English and strong results across six languages (English, French, German, Italian, Portuguese, Spanish), with plans for more languages, open-source training code, and a larger full release soon.

hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="kyutai/helium-1-preview-2b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("kyutai/helium-1-preview-2b")
model = AutoModelForCausalLM.from_pretrained("kyutai/helium-1-preview-2b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## HeliumConfig

[[autodoc]] HeliumConfig

## HeliumModel

[[autodoc]] HeliumModel
    - forward

## HeliumForCausalLM

[[autodoc]] HeliumForCausalLM
    - forward

## HeliumForSequenceClassification

[[autodoc]] HeliumForSequenceClassification
    - forward

## HeliumForTokenClassification

[[autodoc]] HeliumForTokenClassification
    - forward
