<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2024-04-18 and contributed by [eitanturok](https://huggingface.co/eitanturok) and [abhi-db](https://huggingface.co/abhi-db).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DBRX

[DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) is an open, general-purpose large language model introduced by Databricks that achieves state-of-the-art performance among open LLMs, surpassing GPT-3.5 and competing with Gemini 1.0 Pro. It uses a fine-grained mixture-of-experts (MoE) architecture, making inference up to 2x faster than LLaMA2-70B and about 4x more compute-efficient than Databricks’ previous MPT models. DBRX excels at programming tasks, outperforming specialized models like CodeLLaMA-70B, and is strong in language understanding and math benchmarks. Both base and instruction-tuned versions are openly released on Hugging Face, with availability via APIs and integration into Databricks products

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("text-generation", model="databricks/dbrx-instruct", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("databricks/dbrx-instruct", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## DbrxConfig

[[autodoc]] DbrxConfig

## DbrxModel

[[autodoc]] DbrxModel
    - forward

## DbrxForCausalLM

[[autodoc]] DbrxForCausalLM
    - forward

