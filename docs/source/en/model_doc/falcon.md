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
*This model was released on 2023-06-01 and added to Hugging Face Transformers on 2023-07-11.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Falcon

[Falcon](https://huggingface.co/papers/2306.01116) challenges the assumption that curated datasets are essential for training high-performing language models. The authors demonstrate that carefully filtered and deduplicated web data alone can produce models that surpass state-of-the-art systems trained on curated datasets like The Pile. They extract five trillion tokens from CommonCrawl, showing that high-quality web data is still abundant at scale. To support the community, they release the RefinedWeb dataset (600B tokens) along with models of 1.3B and 7.5B parameters trained on it.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("text-generation", model="tiiuae/falcon-rw-1b", dtype="auto")
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## FalconConfig

[[autodoc]] FalconConfig
    - all

## FalconModel

[[autodoc]] FalconModel
    - forward

## FalconForCausalLM

[[autodoc]] FalconForCausalLM
    - forward

## FalconForSequenceClassification

[[autodoc]] FalconForSequenceClassification
    - forward

## FalconForTokenClassification

[[autodoc]] FalconForTokenClassification
    - forward

## FalconForQuestionAnswering

[[autodoc]] FalconForQuestionAnswering
    - forward

## FalconMambaCache

[[autodoc]] FalconMambaCache

